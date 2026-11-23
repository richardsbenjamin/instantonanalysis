import logging
import argparse
import os
from pathlib import Path
import time

from hydra import initialize, compose
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
import numpy as np
import pandas as pd
import dask.array as da
import h5py
import torch as th
import xarray as xr
from tqdm import tqdm
import cftime
from torchinfo import summary
from dask.diagnostics import ProgressBar

from training.dlwp.utils import to_chunked_dataset, encode_variables_as_int, configure_logging, get_best_checkpoint_path
from scripts.forecast import _convert_time_step
from scripts.coupled_forecast import get_coupled_time_dim

logger = logging.getLogger(__name__)
logging.getLogger('cfgrib').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


class LatentCapture:
    def __init__(self, unfold, save_levels=None):
        self.unfold = unfold
        self.save_levels = save_levels
        self.latents_per_step = []

    def start_capture(self):
        self.latents_per_step = []

    def hook(self, module, input, output):
        step_levels = [
            self.unfold(level).detach().cpu().numpy()
            for i, level in enumerate(output)
            if i in self.save_levels
        ]
        self.latents_per_step.append(step_levels)

    @property
    def n_levels(self):
        return len(self.latents_per_step[0]) if self.latents_per_step else 0

    @property
    def n_steps(self):
        return len(self.latents_per_step)

def to_cftime(dt):
    return cftime.DatetimeGregorian(dt.astype(object).year, dt.astype(object).month, dt.astype(object).day, dt.astype(object).hour, dt.astype(object).minute)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def _get_datetime_range(start, end_date, time_step):
    if len(start) > 1:
        raise ValueError('Datetime forecasts require a single initialisation date.')
    start = start[0]
    def fix_time_step(time_step):
        num, unit = time_step[:-1], time_step[-1]
        if unit == 'H':
            unit = 'h'
        return num, unit
    step_num, step_unit = fix_time_step(time_step)
    dt = np.timedelta64(step_num, step_unit)

    start = start.astype('datetime64[m]')
    end_date = np.datetime64(end_date).astype('datetime64[m]')
    dt = dt.astype('timedelta64[m]')
    
    forecast_dates = np.arange(start, end_date  + np.timedelta64(1, 'm'), dt)
    return forecast_dates[1:], forecast_dates[0]


def coupled_inference_hdf5(args: argparse.Namespace):
    logger.info(f" Starting coupled inference with args: {args}")
    atmos_zarr_path = os.path.join(args.data_directory, args.atmos_dataset_name + '.zarr')
    _init_ds = xr.open_zarr(atmos_zarr_path)
    forecast_dates = np.array(
        _init_ds.attrs["forecast_init_times"],
        dtype="datetime64[m]" if args.datetime else "datetime64[ns]")
    os.makedirs(args.output_directory, exist_ok=True)

    if args.gpu == -1:
        device = th.device('cpu')
    else:
        device = th.device(f'cuda:{args.gpu}' if th.cuda.is_available() else 'cpu')
    with initialize(config_path=os.path.join(args.atmos_hydra_path, '.hydra'), version_base=None):
        atmos_cfg = compose('config.yaml')
    with initialize(config_path=os.path.join(args.ocean_hydra_path, '.hydra'), version_base=None):
        ocean_cfg = compose('config.yaml')

    batch_size = 1
    atmos_cfg.num_workers = 0
    atmos_cfg.batch_size = batch_size
    atmos_cfg.data.prebuilt_dataset = True
    ocean_cfg.num_workers = 0
    ocean_cfg.batch_size = batch_size
    ocean_cfg.data.prebuilt_dataset = True


    atmos_cfg.data.data_time_step = "6H"
    ocean_cfg.data.data_time_step = "48H"


    if not hasattr(atmos_cfg.model,'enable_healpixpad'):
        OmegaConf.set_struct(atmos_cfg,True)
        with open_dict(atmos_cfg):
            atmos_cfg.model.enable_healpixpad = False
    else:
        atmos_cfg.model.enable_healpixpad = False
        
    if not hasattr(ocean_cfg.model,'enable_healpixpad'):
        OmegaConf.set_struct(ocean_cfg,True)
        with open_dict(ocean_cfg):
            ocean_cfg.model.enable_healpixpad = False
    else:
        ocean_cfg.model.enable_healpixpad = False
 
    extended_range = args.datetime and args.end_date is not None

    if extended_range:
        atmos_output_lead_times, first_step_datetime_atmos = _get_datetime_range(
            start = forecast_dates,
            end_date = args.end_date,
            time_step = atmos_cfg.data.time_step
        )
        ocean_output_lead_times, first_step_datetime_ocean = _get_datetime_range(
            start = forecast_dates,
            end_date = args.end_date,
            time_step = ocean_cfg.data.time_step
        )
    else:
        atmos_output_lead_times = np.arange(
            _convert_time_step(atmos_cfg.data.gap),
            _convert_time_step(args.lead_time) + pd.Timedelta(seconds=1),
            _convert_time_step(atmos_cfg.data.time_step)
        )
        ocean_output_lead_times = np.arange(
            _convert_time_step(ocean_cfg.data.gap),
            _convert_time_step(args.lead_time) + pd.Timedelta(seconds=1),
            _convert_time_step(ocean_cfg.data.time_step)
        )
    atmos_coupled_time_dim, ocean_coupled_time_dim = get_coupled_time_dim(atmos_cfg, ocean_cfg)
    forecast_integrations = len(ocean_output_lead_times) // ocean_coupled_time_dim
    try:
        nc = len(atmos_cfg.data.module.couplings)
        for i in range(nc):
            atmos_cfg.data.module.couplings[i]['params']['output_time_dim'] = atmos_coupled_time_dim 
    except AttributeError:
        print(f'model {args.atmos_model_path} is not interpreted as a coupled model, cannot perform coupled forecast. Aborting.')
    try:
        nc = len(ocean_cfg.data.module.couplings)
        for i in range(nc):
            ocean_cfg.data.module.couplings[i]['params']['output_time_dim'] = ocean_coupled_time_dim 
    except AttributeError:
        print(f'model {args.atmos_model_path} is not interpreted as a coupled model, cannot perform coupled forecast. Aborting.')
    
    optional_kwargs_atmos = {k: v for k, v in {
        'dst_directory': args.data_directory,
        'prefix': args.data_prefix,
        'suffix': args.data_suffix,
        'dataset_name': args.atmos_dataset_name,
    }.items() if v is not None}
    optional_kwargs_ocean = {k: v for k, v in {
        'dst_directory': args.data_directory,
        'prefix': args.data_prefix,
        'suffix': args.data_suffix,
        'dataset_name': args.ocean_dataset_name,
    }.items() if v is not None}
    
    atmos_data_module = instantiate(
        atmos_cfg.data.module,
        output_time_dim=atmos_coupled_time_dim,
        forecast_init_times=forecast_dates,
        shuffle=False,
        **optional_kwargs_atmos
    )
    atmos_loader, _ = atmos_data_module.test_dataloader()
    ocean_data_module = instantiate(
        ocean_cfg.data.module,
        output_time_dim=ocean_coupled_time_dim,
        forecast_init_times=forecast_dates,
        shuffle=False,
        **optional_kwargs_ocean
    )
    ocean_loader, _ = ocean_data_module.test_dataloader()
    if args.datetime:
        atmos_data_module.set_datetime()
        ocean_data_module.set_datetime()
    
    if extended_range:
        if atmos_output_lead_times[-1] != ocean_output_lead_times[-1]:
            raise ValueError('Last output time of atmos and ocean models does not align. Make sure lead time is compatible with component model intersections.')
        if forecast_integrations*ocean_coupled_time_dim*1/len(ocean_output_lead_times) != forecast_integrations*atmos_coupled_time_dim*1/len(atmos_output_lead_times):
            raise ValueError(f'Requested leadtime and coupled integration are not the same. Make sure lead time is compatible with component model intersections.')
    else:
        if forecast_integrations*ocean_coupled_time_dim*pd.Timedelta(ocean_data_module.time_step) != \
            forecast_integrations*atmos_coupled_time_dim*pd.Timedelta(atmos_data_module.time_step):
            raise ValueError('Lead times of atmos and ocean models does not align.')
        if forecast_integrations*ocean_coupled_time_dim*pd.Timedelta(ocean_data_module.time_step) != _convert_time_step(args.lead_time):
            raise ValueError(f'Requested leadtime ({_convert_time_step(args.lead_time)}) and coupled integration ({forecast_integrations*ocean_coupled_time_dim*pd.Timedelta(ocean_data_module.time_step)}) are not the same. Make sure lead time is compatible with component model intersections.')


    atmos_input_channels = len(atmos_cfg.data.input_variables)
    atmos_output_channels = len(atmos_cfg.data.output_variables) if atmos_cfg.data.output_variables is not None else atmos_input_channels
    atmos_constants_arr = atmos_data_module.constants
    atmos_n_constants = 0 if atmos_constants_arr is None else len(atmos_constants_arr.keys())
    ocean_input_channels = len(ocean_cfg.data.input_variables)
    ocean_output_channels = len(ocean_cfg.data.output_variables) if ocean_cfg.data.output_variables is not None else ocean_input_channels
    ocean_constants_arr = ocean_data_module.constants
    ocean_n_constants = 0 if ocean_constants_arr is None else len(ocean_constants_arr.keys())

    atmos_decoder_input_channels = int(atmos_cfg.data.get('add_insolation', 0))
    atmos_cfg.model['input_channels'] = atmos_input_channels
    atmos_cfg.model['output_channels'] = atmos_output_channels
    atmos_cfg.model['n_constants'] = atmos_n_constants
    atmos_cfg.model['decoder_input_channels'] = atmos_decoder_input_channels
    ocean_decoder_input_channels = int(ocean_cfg.data.get('add_insolation', 0))
    ocean_cfg.model['input_channels'] = ocean_input_channels
    ocean_cfg.model['output_channels'] = ocean_output_channels
    ocean_cfg.model['n_constants'] = ocean_n_constants
    ocean_cfg.model['decoder_input_channels'] = ocean_decoder_input_channels
    
    atmos_model = instantiate(atmos_cfg.model, output_time_dim=atmos_coupled_time_dim)
    atmos_model_name = Path(args.atmos_model_path).name
    atmos_checkpoint_basepath = os.path.join(args.atmos_model_path, "tensorboard", "checkpoints")
    if args.atmos_model_checkpoint is None:
        atmos_checkpoint_path = get_best_checkpoint_path(path=atmos_checkpoint_basepath)
    else:
        atmos_checkpoint_path = os.path.join(atmos_checkpoint_basepath, args.atmos_model_checkpoint)
    logger.info("load model checkpoint %s", atmos_checkpoint_path)
    ocean_model = instantiate(ocean_cfg.model, output_time_dim=ocean_coupled_time_dim)
    ocean_model_name = Path(args.ocean_model_path).name
    ocean_checkpoint_basepath = os.path.join(args.ocean_model_path, "tensorboard", "checkpoints")
    if args.ocean_model_checkpoint is None:
        ocean_checkpoint_path = get_best_checkpoint_path(path=ocean_checkpoint_basepath)
    else:
        ocean_checkpoint_path = os.path.join(ocean_checkpoint_basepath, args.ocean_model_checkpoint)
    logger.info("load model checkpoint %s", ocean_checkpoint_path)

    atmos_checkpoint = th.load(atmos_checkpoint_path, map_location=device)
    atmos_model_state_dict = atmos_checkpoint["model_state_dict"]
    atmos_model.load_state_dict(atmos_model_state_dict)
    atmos_model = atmos_model.to(device)
    atmos_model.eval()
    print(f'Atmos Model Summary:')
    summary(atmos_model)
    ocean_checkpoint = th.load(ocean_checkpoint_path, map_location=device)
    ocean_model_state_dict = ocean_checkpoint["model_state_dict"]
    ocean_model.load_state_dict(ocean_model_state_dict)
    ocean_model = ocean_model.to(device)
    ocean_model.eval()
    print(f'Ocean Model Summary:')
    summary(ocean_model)

    if args.save_latent:
        atmos_latent_capture = LatentCapture(atmos_model.unfold, args.save_levels)
        atmos_hook_handle = atmos_model.encoder.register_forward_hook(atmos_latent_capture.hook)
        logger.info("Registered encoder hooks for latent extraction")

    atmos_coupler = atmos_data_module.test_dataset.couplings[0]
    atmos_coupler.setup_coupling(ocean_data_module)
    atmos_loader_iter = iter(atmos_loader)
    ocean_coupler = ocean_data_module.test_dataset.couplings[0]
    ocean_coupler.setup_coupling(atmos_data_module)
    ocean_loader_iter = iter(ocean_loader)
    
    pbar = tqdm(forecast_dates, position=0)
    atmos_constants = None 
    ocean_constants = None

    for model, data_module in [[atmos_model,atmos_data_module], [ocean_model, ocean_data_module]]: 
        if getattr(model,'debugging_model', False):
            model.set_output(forecast_dates, 
                             forecast_integrations,
                             data_module,)
   
    atmos_prediction_shape = (len(forecast_dates),
                              len(atmos_output_lead_times) + 1,
                              len(atmos_data_module.output_variables)) + \
                              atmos_data_module.test_dataset.spatial_dims
    ocean_prediction_shape = (len(forecast_dates),
                              len(ocean_output_lead_times) + 1,
                              len(ocean_data_module.output_variables)) + \
                              ocean_data_module.test_dataset.spatial_dims
    

    required_space_atmos = np.prod(atmos_prediction_shape) * 4 * 1.12
    required_space_ocean = np.prod(ocean_prediction_shape) * 4 * 1.12
    
    statvfs = os.statvfs(args.cache_dir)

    available_space = statvfs.f_frsize * statvfs.f_bavail

    if required_space_atmos + required_space_ocean > available_space:
        if available_space < 1024 ** 4:
            required_space_gb = (required_space_atmos + required_space_ocean) / (1024 ** 3)
            available_space_gb = available_space / (1024 ** 3)
            logger.info(f"Not enough space on disk. Required: {required_space_gb:.2f} GB, available: {available_space_gb:.2f} GB")
        else:
            required_space_tb = (required_space_atmos + required_space_ocean) / (1024 ** 4)
            available_space_tb = available_space / (1024 ** 4)
            logger.info(f"Not enough space on disk. Required: {required_space_tb:.2f} TB, available: {available_space_tb:.2f} TB")
        exit(1)

    if os.path.exists(f'{args.cache_dir}/atmos_prediction.hdf5'):
        logger.info(f'Clearing cache {args.cache_dir}/atmos_prediction.hdf5')
        os.remove(f'{args.cache_dir}/atmos_prediction.hdf5')
    if os.path.exists(f'{args.cache_dir}/ocean_prediction.hdf5'):
        os.remove(f'{args.cache_dir}/ocean_prediction.hdf5')
        logger.info(f'Clearing cache {args.cache_dir}/ocean_prediction.hdf5')

    if args.save_latent:
        for f in ['atmos_latent.hdf5']:
            fpath = f'{args.cache_dir}/{f}'
            if os.path.exists(fpath):
                logger.info(f'Clearing cache {fpath}')
                os.remove(fpath)
    
    latent_hdf5_files = {}
    atmos_latent_datasets = None
    atmos_latent_shapes = None
    n_atmos_levels = None

    with h5py.File(f'{args.cache_dir}/atmos_prediction.hdf5', 'w') as f_atmos, h5py.File(f'{args.cache_dir}/ocean_prediction.hdf5', 'w') as f_ocean:
        atmos_prediction = f_atmos.create_dataset('atmos_prediction', atmos_prediction_shape, dtype='float32')
        ocean_prediction = f_ocean.create_dataset('ocean_prediction', ocean_prediction_shape , dtype='float32')
 
        for i,init in enumerate(pbar):
            logger.info(f'Forecast init date: {init}')
            pbar.postfix = 'init ' + np.datetime_as_string(
                forecast_dates[i].astype('datetime64[h]'), unit='h'
            ).replace('T', ' ') + 'Z'
            pbar.update()
            new_init = True
            atmos_coupler.reset_coupler()
            ocean_coupler.reset_coupler()

            nested_pbar = tqdm(range(forecast_integrations), position=1, leave=False)
            for j in nested_pbar:
                logger.info(f'Forecast integration step: {j}')
                if j==0:
                    atmos_input = [k.to(device) for k in next(atmos_loader_iter)]
                    if atmos_constants is None:
                        atmos_constants = atmos_input[2]
                    if args.save_latent:
                        atmos_latent_capture.start_capture()
                    with th.no_grad():
                        logger.info(f"Init Atmos Input: {[k.shape for k in atmos_input]}")
                        atmos_output = atmos_model(atmos_input)

                        logger.info(f"Init Atmos Output: {atmos_output.shape}")
                    
                    if args.save_latent and atmos_latent_datasets is None:
                        logger.info(f"Saving latent embeddings to HDF5 files with shape: {len(forecast_dates)}x{len(atmos_output_lead_times)}x...")
                        logger.info(f"Captured {atmos_latent_capture.n_steps} internal integration steps with {atmos_latent_capture.n_levels} encoder levels")
                        n_atmos_levels = atmos_latent_capture.n_levels
                        atmos_latent_shapes = []
                        atmos_latent_datasets = []
                        latent_hdf5_files['atmos'] = h5py.File(f'{args.cache_dir}/atmos_latent.hdf5', 'w')
                        for lvl in range(n_atmos_levels):
                            lat = atmos_latent_capture.latents_per_step[0][lvl]
                            shape = (
                                len(forecast_dates),
                                len(atmos_output_lead_times),
                                lat.shape[2],
                                lat.shape[1],
                                lat.shape[3],
                                lat.shape[4],
                            )
                            atmos_latent_shapes.append(shape)
                            ds = latent_hdf5_files['atmos'].create_dataset(
                                f'encoder_l{lvl}', shape, dtype='float32'
                            )
                            atmos_latent_datasets.append(ds)
                            logger.info(f"Atmos encoder_l{lvl} shape: {shape}")

                    ocean_coupler.set_coupled_fields(atmos_output.cpu())
                    ocean_input = [k.to(device) for k in next(ocean_loader_iter)]
                    if ocean_constants is None:
                        ocean_constants = ocean_input[2]
                    with th.no_grad():
                        logger.info(f"Init Ocean Input: {[k.shape for k in ocean_input]}")
                        ocean_output = ocean_model(ocean_input)
                        logger.info(f"Init Ocean Output: {ocean_output.shape}")

                else:
                    atmos_coupler.set_coupled_fields(ocean_output)
                    atmos_input = [k.to(device) for k in atmos_data_module.test_dataset.next_integration(
                                                            atmos_output, 
                                                            constants = atmos_constants,
                                                        )]
                    logger.info(f"Step Atmos Input: {[k.shape for k in atmos_input]}")
                    if args.save_latent:
                        atmos_latent_capture.start_capture()
                    with th.no_grad():
                        atmos_output = atmos_model(atmos_input)
                        logger.info(f"Step Atmos Output: {atmos_output.shape}")

        
                    ocean_coupler.set_coupled_fields(atmos_output.cpu())
                    ocean_input = [k.to(device) for k in ocean_data_module.test_dataset.next_integration(
                                                            ocean_output, 
                                                            constants = ocean_constants,
                                                        )]
                    logger.info(f"Step Ocean Input: {[k.shape for k in ocean_input]}")
                    with th.no_grad():
                        ocean_output = ocean_model(ocean_input)
                        logger.info(f"Step Ocean Output: {ocean_output.shape}")


                if new_init:
                    ocean_prediction[i*batch_size:(i+1)*batch_size,0,:,:,:,:] = ocean_input[0].permute(0, 2, 3, 1, 4, 5)[:, -1].cpu().numpy()
                    atmos_prediction[i*batch_size:(i+1)*batch_size,0,:,:,:,:] = atmos_input[0].permute(0, 2, 3, 1, 4, 5)[:, -1].cpu().numpy()
                    new_init = False

                ocean_prediction[i*batch_size:(i+1)*batch_size, slice(1+j*ocean_coupled_time_dim,(j+1)*ocean_coupled_time_dim+1), :, :, :, :] = ocean_output.permute(0, 2, 3, 1, 4, 5).cpu().numpy()
                atmos_prediction[i*batch_size:(i+1)*batch_size, slice(1+j*atmos_coupled_time_dim,(j+1)*atmos_coupled_time_dim+1), :, :, :, :] = atmos_output.permute(0, 2, 3, 1, 4, 5).cpu().numpy()

                if args.save_latent:
                    all_latents = atmos_latent_capture.latents_per_step
                    n_captured = len(all_latents)
                    integration_steps = getattr(atmos_model, 'integration_steps', n_captured)

                    if n_captured > integration_steps:
                        presteps = getattr(atmos_model, 'presteps', 0)
                        delta_t = atmos_model.delta_t
                        itd = atmos_model.input_time_dim
                        reset_cycle = atmos_model.reset_cycle
                        real_indices = []
                        call_idx = 0
                        for step in range(integration_steps):
                            if (step * (delta_t * itd)) % reset_cycle == 0:
                                call_idx += presteps
                            real_indices.append(call_idx)
                            call_idx += 1
                        real_latents = [all_latents[idx] for idx in real_indices]
                    else:
                        real_latents = all_latents

                    n_real_steps = len(real_latents)
                    itd_repeat = atmos_coupled_time_dim // n_real_steps
                    for lvl in range(n_atmos_levels):
                        stacked = np.stack([
                            real_latents[s][lvl].transpose(0, 2, 1, 3, 4)
                            for s in range(n_real_steps)
                        ], axis=1)
                        stacked = np.repeat(stacked, itd_repeat, axis=1)
                        atmos_latent_datasets[lvl][
                            i*batch_size:(i+1)*batch_size,
                            slice(j*atmos_coupled_time_dim, (j+1)*atmos_coupled_time_dim),
                            :, :, :, :
                        ] = stacked


    if args.save_latent:
        for fh in latent_hdf5_files.values():
            fh.close()

    f_atmos = h5py.File(f'{args.cache_dir}/atmos_prediction.hdf5', 'r')
    f_ocean = h5py.File(f'{args.cache_dir}/ocean_prediction.hdf5', 'r')
    atmos_prediction = da.from_array(f_atmos['atmos_prediction'], 
                                        chunks=(args.time_chunk, 
                                                args.step_chunk, 
                                                atmos_prediction_shape[2],
                                                atmos_prediction_shape[3],
                                                atmos_prediction_shape[4],
                                                atmos_prediction_shape[5],))
    ocean_prediction = da.from_array(f_ocean['ocean_prediction'], 
                                        chunks=(args.time_chunk, 
                                                args.step_chunk, 
                                                ocean_prediction_shape[2],
                                                ocean_prediction_shape[3],
                                                ocean_prediction_shape[4],
                                                ocean_prediction_shape[5],))
           
    ocean_meta_ds = ocean_data_module.test_dataset.ds
    atmos_meta_ds = atmos_data_module.test_dataset.ds

    if extended_range:
        ocean_steps = [to_cftime(lt) for lt in [first_step_datetime_ocean] + list(ocean_output_lead_times)]
        atmos_steps = [to_cftime(lt) for lt in [first_step_datetime_atmos] + list(atmos_output_lead_times)]
    else:
        ocean_steps = [pd.Timedelta(hours=0)] + list(ocean_output_lead_times)
        atmos_steps = [pd.Timedelta(hours=0)] + list(atmos_output_lead_times)
    if args.datetime:
        forecast_dates = [to_cftime(f.astype('datetime64[m]')) for f in forecast_dates]
    ocean_prediction_da = xr.DataArray(
        ocean_prediction,
        dims=['time', 'step', 'channel_out', 'face', 'height', 'width'],
        coords={
            'time': forecast_dates,
            'step': ocean_steps,
            'channel_out': ocean_cfg.data.output_variables or ocean_cfg.data.input_variables,
            'face': ocean_meta_ds.face,
            'height': ocean_meta_ds.height,
            'width': ocean_meta_ds.width
        }
    )
    atmos_prediction_da = xr.DataArray(
        atmos_prediction,
        dims=['time', 'step', 'channel_out', 'face', 'height', 'width'],
        coords={
            'time': forecast_dates,
            'step': atmos_steps,
            'channel_out': atmos_cfg.data.output_variables or atmos_cfg.data.input_variables,
            'face': atmos_meta_ds.face,
            'height': atmos_meta_ds.height,
            'width': atmos_meta_ds.width
        }
    )

    ocean_prediction_da[:] *= ocean_data_module.test_dataset.target_scaling['std']
    ocean_prediction_da[:] += ocean_data_module.test_dataset.target_scaling['mean']
    ocean_prediction_ds = ocean_prediction_da.to_dataset(dim='channel_out')
    for variable in ocean_prediction_ds.data_vars:
        if ocean_cfg.data.scaling[variable].get('log_epsilon', None) is not None:
            ocean_prediction_ds[variable] = np.exp(
                ocean_prediction_ds[variable] + np.log(ocean_cfg.data.scaling[variable]['log_epsilon'])
            ) - ocean_cfg.data.scaling[variable]['log_epsilon']
    atmos_prediction_da[:] *= atmos_data_module.test_dataset.target_scaling['std']
    atmos_prediction_da[:] += atmos_data_module.test_dataset.target_scaling['mean']
    atmos_prediction_ds = atmos_prediction_da.to_dataset(dim='channel_out')
    for variable in atmos_prediction_ds.data_vars:
        if atmos_cfg.data.scaling[variable].get('log_epsilon', None) is not None:
            atmos_prediction_ds[variable] = np.exp(
                atmos_prediction_ds[variable] + np.log(atmos_cfg.data.scaling[variable]['log_epsilon'])
            ) - atmos_cfg.data.scaling[variable]['log_epsilon']

    write_time = time.time()
    ocean_prediction_ds = to_chunked_dataset(ocean_prediction_ds, {'time': args.time_chunk,'step': args.step_chunk})
    atmos_prediction_ds = to_chunked_dataset(atmos_prediction_ds, {'time': args.time_chunk,'step': args.step_chunk})
    if args.encode_int:
        ocean_prediction_ds = encode_variables_as_int(ocean_prediction_ds, compress=1)
        atmos_prediction_ds = encode_variables_as_int(atmos_prediction_ds, compress=1)

    if getattr(args,'ocean_output_filename',None) is not None:
        ocean_output_file = os.path.join(args.output_directory, f"{args.ocean_output_filename}.nc")
    if getattr(args,'atmos_output_filename',None) is not None:
        atmos_output_file = os.path.join(args.output_directory, f"{args.atmos_output_filename}.nc")
    else:
        ocean_output_file = os.path.join(args.output_directory, f"forecast_{ocean_model_name}.nc")
        atmos_output_file = os.path.join(args.output_directory, f"forecast_{atmos_model_name}.nc")
    logger.info(f"writing forecasts to {atmos_output_file} and {ocean_output_file}")
    if args.zarr:
        ocean_write_job = ocean_prediction_ds.to_zarr(ocean_output_file, compute=False, mode="w")
        atmos_write_job = atmos_prediction_ds.to_zarr(atmos_output_file, compute=False, mode="w")
    else:
        ocean_write_job = ocean_prediction_ds.to_netcdf(ocean_output_file, compute=False)
        atmos_write_job = atmos_prediction_ds.to_netcdf(atmos_output_file, compute=False)
    with ProgressBar():
        ocean_write_job.compute()
        atmos_write_job.compute()
    logger.debug("wrote file in %0.1f s", time.time() - write_time)

    if args.save_latent:
        logger.info("Writing latent embeddings to zarr...")
        latent_write_time = time.time()

        f_atmos_latent = h5py.File(f'{args.cache_dir}/atmos_latent.hdf5', 'r')

        if extended_range:
            atmos_latent_steps = [to_cftime(lt) for lt in list(atmos_output_lead_times)]
        else:
            atmos_latent_steps = list(atmos_output_lead_times)

        atmos_latent_vars = {}
        for lvl in range(n_atmos_levels):
            shape = atmos_latent_shapes[lvl]
            latent_step_chunk = min(args.step_chunk, shape[1])
            arr = da.from_array(
                f_atmos_latent[f'encoder_l{lvl}'],
                chunks=(args.time_chunk, latent_step_chunk, shape[2], shape[3], shape[4], shape[5])
            )
            atmos_latent_vars[f'encoder_l{lvl}'] = xr.DataArray(
                arr,
                dims=['time', 'step', f'channel_l{lvl}', 'face', f'height_l{lvl}', f'width_l{lvl}'],
                coords={
                    'time': forecast_dates,
                    'step': atmos_latent_steps,
                    f'channel_l{lvl}': np.arange(shape[2]),
                    'face': np.arange(shape[3]),
                    f'height_l{lvl}': np.arange(shape[4]),
                    f'width_l{lvl}': np.arange(shape[5]),
                }
            )
        atmos_latent_xds = xr.Dataset(atmos_latent_vars)
        atmos_latent_xds = to_chunked_dataset(atmos_latent_xds, {'time': args.time_chunk, 'step': latent_step_chunk})


        atmos_latent_file = os.path.join(args.output_directory, args.atmos_latent_out_path)
        logger.info(f"Writing latent embeddings to {atmos_latent_file}")

        atmos_latent_write = atmos_latent_xds.to_zarr(atmos_latent_file, compute=False, mode="w")
        with ProgressBar():
            atmos_latent_write.compute()
        logger.debug("wrote latent files in %0.1f s", time.time() - latent_write_time)

        f_atmos_latent.close()
        os.remove(f'{args.cache_dir}/atmos_latent.hdf5')

        atmos_hook_handle.remove()
        logger.info("Latent embedding extraction complete.")

    f_atmos.close()
    f_ocean.close()
    os.remove(f'{args.cache_dir}/atmos_prediction.hdf5')
    os.remove(f'{args.cache_dir}/ocean_prediction.hdf5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce forecasts from a DLWP model.')
    parser.add_argument('--atmos-model-path', type=str, required=True,
                        help="Path to model training outputs directory for the coupled atmos model")
    parser.add_argument('--ocean-model-path', type=str, required=True,
                        help="Path to model training outputs directory for the coupled ocean model")
    parser.add_argument('--atmos-model-checkpoint', type=str, default=None,
                        help="Atmos Model checkpoint file name (include ending). Set 'last.ckpt' to use last checkpoint. If "
                             "None, the best will be chosen (according to validation error).")
    parser.add_argument('--ocean-model-checkpoint', type=str, default=None,
                        help="Ocean Model checkpoint file name (include ending). Set 'last.ckpt' to use last checkpoint. If "
                             "None, the best will be chosen (according to validation error).")
    parser.add_argument('--non-strict', action='store_true',
                        help="Disable strict mode for model checkpoint loading")
    parser.add_argument('-l', '--lead-time', type=int, default=168,
                        help="Maximum forecast lead time to predict, in integer hours")
    parser.add_argument('-s', '--forecast-init-start', type=str, default='2017-01-02',
                        help="")
    parser.add_argument('-e', '--forecast-init-end', type=str, default='2018-12-30',
                        help="")
    parser.add_argument('-f', '--freq', type=str, default='biweekly',
                        help="Frequency of forecast initialisation. There is a special case, 'biweekly', which will "
                             "follow the ECMWF standard of two forecasts per week, with a 3- followed by 4-day gap. "
                             "Otherwise, interpretable by pandas.")
    parser.add_argument('-b', '--batch-size', type=int, default=None,
                        help="The batch size that is used to generate the forecast.")
    parser.add_argument('--time-chunk', type=int, default=1,
                        help="The size of chunks in time dimension used to postprocess and predictions")
    parser.add_argument('--step-chunk', type=int, default=1,
                        help="The size of chunks in step dimension used to postprocess and predictions")
    parser.add_argument('--cache-dir', type=str, default=".",
                        help="This is where the intermediate hdf5 files will be stored during integration")
    parser.add_argument('-o', '--output-directory', type=str, default='forecasts/',
                        help="Directory in which to save output forecast")
    parser.add_argument('--atmos-output-filename', type=str, default=None,
                        help="Filename used to save atmos output forecast")
    parser.add_argument('--ocean-output-filename', type=str, default=None,
                        help="Filename used to save ocean output forecast")
    parser.add_argument('--encode-int', action='store_true',
                        help="Encode data variables as int16 type (may not be compatible with tempest-remap)")
    parser.add_argument('-d', '--data-directory', type=str, default=None,
                        help="Path to test data, if different from files used for model training")
    parser.add_argument('--atmos-dataset-name', type=str, default=None,
                        help="Name of init dataset, if different from files used for model training")  
    parser.add_argument('--ocean-dataset-name', type=str, default=None,
                        help="Name of init dataset, if different from files used for model training")
    parser.add_argument('--data-prefix', type=str, default=None,
                        help="Prefix for test data files")
    parser.add_argument('--data-suffix', type=str, default=None,
                        help="Suffix for test data files")
    parser.add_argument('--datetime', type=str2bool, default=False,
                        help="Use numpy.datetime64[m] for time encoding, necessary for longer "
                             "simulations (>200 years). type=str2bool, not bool: with the latter "
                             "argparse calls bool('false'), which is True.")
    parser.add_argument('--end-date', type=str, default=None,
                        help="Last time in forecast. Only used if forecast is using datetime64")
    parser.add_argument('--gpu', type=int, default=0,
                        help="Index of GPU device on which to run model. If -1 forecast will be done on CPU")
    parser.add_argument('--zarr', type=str2bool, default=False, const=True, nargs="?",
                        help="Whether the output directory is for writing a zarr file.")
    parser.add_argument('--save-latent', type=str2bool, default=True, const=True, nargs="?",
                        help="Whether to extract and save encoder bottleneck latent embeddings as zarr.")
    parser.add_argument('--save-levels', type=int, nargs='+', default=[0,1,2], help="Levels to save latents for (default: 0 1 2)")
    parser.add_argument('--atmos-latent-out-path', type=str, default=None,
                        help="Path to output latent hdf5 file for atmosphere.")
    parser.add_argument('--ocean-latent-out-path', type=str, default=None,
                        help="Path to output latent hdf5 file for ocean.")

    configure_logging(2)
    run_args = parser.parse_args()

    run_args.atmos_hydra_path = os.path.relpath(os.path.abspath(run_args.atmos_model_path), os.path.dirname(os.path.abspath(__file__)))
    run_args.ocean_hydra_path = os.path.relpath(os.path.abspath(run_args.ocean_model_path), os.path.dirname(os.path.abspath(__file__)))
    logger.debug("model paths: %s", (run_args.atmos_model_path, run_args.ocean_model_path))
    logger.debug("python working dir: %s", os.getcwd())
    logger.debug("hydra paths: %s", (run_args.atmos_hydra_path, run_args.ocean_hydra_path))
    coupled_inference_hdf5(run_args)
