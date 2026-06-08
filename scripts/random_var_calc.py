from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import dask
import pandas as pd
import numpy as np
import xarray as xr
from hydra.utils import instantiate

from instantonanalysis.instanton.analysis import (
    calculate_chi_mask,
    calculate_degrees_of_freedom,
)
from instantonanalysis.instanton.utils import (
    build_event_cube,
    load_config,
    read_dataset,
)
from instantonanalysis.instanton.utils.parsers import get_calc_args

if TYPE_CHECKING:
    from instantonanalysis.instanton.xconfig import XConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_random_event_dates(
        base_array: xr.DataArray,
        all_dates: np.ndarray,
        nsamples: int,
        xconfig: XConfig,
        seed: int = 42,
    ) -> xr.DataArray:
    time_dim = xconfig.time_dim
    rolling_dim = xconfig.rolling_period
    quantile_dim = xconfig.quantile

    all_dates_idx = pd.DatetimeIndex(all_dates).sort_values()
    quantiles = base_array[xconfig.quantile].values
    rolling_periods = base_array[xconfig.rolling_period].values
    sampled_array = np.empty((nsamples, len(quantiles), len(rolling_periods)), dtype='datetime64[ns]')

    for q_idx, q in enumerate(quantiles):
        for r_idx, r in enumerate(rolling_periods):
            
            subset = base_array.sel(quantile=q, rolling_period=r)
            valid_steps = pd.DatetimeIndex(subset.dropna(dim=time_dim)[time_dim].values)
            
            positions = all_dates_idx.get_indexer(valid_steps)
            positions = positions[positions >= 0] 
            
            exclude_positions = set()
            for pos in positions:
                for offset in range(int(r)):
                    exclude_positions.add(pos + offset)
            
            max_idx = len(all_dates_idx)
            candidate_positions = [
                i for i in range(max_idx) 
                if i not in exclude_positions
            ]
            
            if len(candidate_positions) < nsamples:
                raise ValueError(f"Not enough valid dates for q={q}, r={r}. Found {len(candidate_positions)}, need {nsamples}.")
            
            rng = np.random.default_rng(seed=seed)
            sampled_positions = rng.choice(candidate_positions, size=nsamples, replace=False)
            sampled_array[:, q_idx, r_idx] = all_dates_idx[sampled_positions].values

    return xr.DataArray(
        sampled_array,
        dims=[time_dim, quantile_dim, rolling_dim],
        coords={
            time_dim: np.arange(nsamples),
            quantile_dim: quantiles,
            rolling_dim: rolling_periods
        },
        name='random_dates'
    )


if __name__ == "__main__":
    logger.info("Loading config")
    args = get_calc_args()
    cfg = load_config(args.config_name, args.overrides)
    
    data_root_path = Path(cfg.paths.data_root)
    output_dir = Path(cfg.paths.results_root + cfg.locations.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    vars_list = [var.name for var in cfg.variables.values()]
    xconfig = instantiate(cfg.xconfig)

    results_comp_anom = []
    results_norm_hat = []
    results_chi_masks = []

    logger.info("Loading datasets")
    climate_mean = read_dataset(data_root_path / cfg.paths.climate_mean)[vars_list]
    climate_var = read_dataset(data_root_path / cfg.paths.climate_variance)[vars_list]
    data_in = read_dataset(data_root_path / cfg.paths.data_file)[vars_list]

    series_obs = xr.open_dataset(output_dir / cfg.paths.series_obs)
    closest_n = xr.open_dataset(output_dir / cfg.paths.closest_neighbours)

    all_dates = series_obs.step
    random_dates = get_random_event_dates(
        closest_n,
        all_dates,
        cfg.analysis.nb_closest,
        xconfig,
        cfg.seed,
    )

    for var_cfg in cfg.variables.values():
        logger.info("Building event cube for " + var_cfg.name)

        dataset = data_in[var_cfg.name].squeeze()

        dataset = dataset.chunk({
            xconfig.spatial_dims[0]: 1,
        })

        event_cube = build_event_cube(
            dataset, random_dates, xconfig,
        )
        logger.info("Writing event cube")

        event_cube = event_cube.chunk({
            xconfig.rolling_period: 1,
            xconfig.quantile: 1,
            xconfig.lag: -1,
            xconfig.event: -1,
            xconfig.spatial_dims[0]: 1,
            xconfig.spatial_dims[1]: 'auto',
        })

        event_cube_path = output_dir / f"{var_cfg.name}_event_cube.zarr"
        with dask.config.set(scheduler='synchronous'):
            event_cube.to_zarr(event_cube_path, mode="w")

        del event_cube
        event_cube = xr.open_zarr(event_cube_path)[var_cfg.name]

        logger.info("Computing statistics")
        comp_anom = (event_cube.mean(xconfig.lag).mean(xconfig.event) - climate_mean[var_cfg.name]).rename(var_cfg.name)
        norm_var_hat = (event_cube.mean(xconfig.lag).var(xconfig.event) / climate_var[var_cfg.name]).rename(var_cfg.name)

        df = calculate_degrees_of_freedom(
            event_cube,
            count_dim=xconfig.event,
            max_dims=xconfig.spatial_dims,
            pre_mean_dim=xconfig.lag
        )
        chi_mask = calculate_chi_mask(norm_var_hat, df)

        comp_anom, norm_var_hat, chi_mask = dask.compute(comp_anom, norm_var_hat, chi_mask)

        results_chi_masks.append(chi_mask)
        results_comp_anom.append(comp_anom)
        results_norm_hat.append(norm_var_hat)
        
    climate_mean.close()
    climate_var.close()
    dataset.close()
        
    logger.info("Merging and saving all variables")
    outputs = {
        cfg.paths.composite_anomalies: results_comp_anom,
        cfg.paths.normalised_var_hat: results_norm_hat,
        cfg.paths.chi_masks: results_chi_masks,
    }
    for path, data_list in outputs.items():
        combined_ds = xr.merge(data_list).reset_coords(drop=True)
        try:
            combined_ds.to_netcdf(output_dir / path)
        except PermissionError:
            logger.warning(f"Permission denied when saving {path}")

