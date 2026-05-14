from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import dask
import numpy as np
import xarray as xr
from hydra.utils import instantiate

from instantonanalysis.instanton.analysis import (
    calculate_chi_mask,
    calculate_degrees_of_freedom,
    calculate_observable,
)
from instantonanalysis.instanton.nbclosest import NClosestCalc, get_nclosest_config
from instantonanalysis.instanton.utils import (
    build_event_cube,
    convert_timedelta2datetime,
    get_distance_function,
    load_config,
    read_dataset,
)
from instantonanalysis.instanton.utils.parsers import get_calc_args

if TYPE_CHECKING:
    from instantonanalysis.instanton.nbclosest import NClosestConfig
    from instantonanalysis.instanton.xconfig import XConfig
    

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_chunks_spec(xconfig: XConfig) -> dict[str, int]:
    chunks_spec = {xconfig.time_dim: -1} 
    for dim in xconfig.spatial_dims:
        chunks_spec[dim] = 'auto'
    return chunks_spec


if __name__ == "__main__":
    logger.info("Loading config")
    args = get_calc_args()
    cfg = load_config(args.config_name, args.overrides)
    
    data_root_path = Path(cfg.paths.data_root)
    na_box = instantiate(cfg.na_box)
    spatial_box = instantiate(cfg.box)
    output_dir = Path(cfg.paths.results_root + cfg.locations.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    vars_list = [var.name for var in cfg.variables.values()]
    xconfig = instantiate(cfg.xconfig)

    results_obs = []
    results_ac = []
    results_qa = []
    results_nb = []
    results_nb_dates = []
    results_comp_anom = []
    results_norm_tilde = []
    results_weight_tilde = []
    results_norm_hat = []
    results_chi_masks = []

    logger.info("Loading datasets")
    datasets = {}
    climate_mean_in = read_dataset(data_root_path / f"climate_mean_{cfg.paths.data_file.replace('.nc', '.zarr')}")[vars_list]
    climate_var_in = read_dataset(data_root_path / f"climate_variance_{cfg.paths.data_file.replace('.nc', '.zarr')}")[vars_list]
    dataset_in = convert_timedelta2datetime(
        read_dataset(data_root_path / cfg.paths.data_file),
        xconfig,
    )[vars_list]

    for var_cfg in cfg.variables.values():
        datasets[var_cfg.name] = {
            "dataset": dataset_in[var_cfg.name],
            "climate_mean": climate_mean_in[var_cfg.name],
            "climate_var": climate_var_in[var_cfg.name],
        }

    logger.info("Calculating observable")
    var_cfg = cfg.variables["t2m"]

    series_obs = calculate_observable(
        datasets[var_cfg.name]["dataset"],
        calc_months=cfg.analysis.calc_months_init,
        spatial_box=spatial_box,
        xconfig=xconfig,
    ).squeeze()

    logger.info("Calculating nclosest")
    nclosest_config = get_nclosest_config(
        analysis_cfg=cfg.analysis,
        time_dim=xconfig.time_dim,
    )
    nclosest_calc = NClosestCalc(
        series_obs=series_obs,
        config=nclosest_config,
        xconfig=xconfig,
        dist_func=get_distance_function(cfg.analysis.dist_func),
    )
    nclosest_calc.calculate()
    results_obs.append(series_obs.rename(var_cfg.name))
    results_ac.append(nclosest_calc.results_ac.rename(var_cfg.name))
    results_qa.append(nclosest_calc.results_qa.rename(var_cfg.name))
    results_nb.append(nclosest_calc.results_nb.rename(var_cfg.name))
    results_nb_dates.append(nclosest_calc.results_nb_dates.rename(var_cfg.name))

    for var_cfg in cfg.variables.values():
        logger.info("Building event cube")

        dataset = datasets[var_cfg.name]["dataset"].squeeze()

        # 6D cube: rolling_period, quantile, lag, event, spatial...
        event_cube = build_event_cube(
            dataset, var_cfg, nclosest_calc.results_nb_dates, xconfig,
        )

        # Write event cube to zarr FIRST, before computing any derived stats.
        # This prevents Dask from holding both the raw event_cube computation
        # graph and the derived statistics graphs in memory simultaneously.
        logger.info("Writing event cube")

        event_cube = event_cube.chunk({
            xconfig.rolling_period: 1,
            xconfig.quantile: 1,
            xconfig.lag: -1,
            xconfig.event: -1,
            xconfig.spatial_dims[0]: 'auto',
            xconfig.spatial_dims[1]: 'auto',
        })

        event_cube_path = output_dir / f"{var_cfg.name}_event_cube.zarr"
        event_cube.to_zarr(event_cube_path, mode="w")

        # Re-open from zarr so all downstream work reads from disk,
        # not from the in-memory task graph.
        del event_cube
        event_cube = xr.open_zarr(event_cube_path)[var_cfg.name]

        climate_mean = datasets[var_cfg.name]["climate_mean"]
        climate_var = datasets[var_cfg.name]["climate_var"]

        logger.info("Computing statistics")
        comp_anom = (event_cube.mean(xconfig.lag).mean(xconfig.event) - climate_mean).rename(var_cfg.name)
        norm_var_tilde = (event_cube.var(xconfig.event) / climate_var).rename(var_cfg.name)
        weight_var_tilde = na_box.spatial_mean(norm_var_tilde * 100).rename(var_cfg.name)
        norm_var_hat = (event_cube.mean(xconfig.lag).var(xconfig.event) / climate_var).rename(var_cfg.name)

        df = calculate_degrees_of_freedom(
            event_cube,
            count_dim=xconfig.event,
            max_dims=xconfig.spatial_dims,
            pre_mean_dim=xconfig.lag
        )
        chi_mask = calculate_chi_mask(norm_var_hat, df)
        # norm_var_hat = norm_var_hat.where(chi_mask)

        comp_anom, norm_var_tilde, weight_var_tilde, norm_var_hat = dask.compute(
            comp_anom,
            norm_var_tilde,
            weight_var_tilde,
            norm_var_hat
        )

        results_chi_masks.append(chi_mask)
        results_comp_anom.append(comp_anom)
        results_norm_tilde.append(norm_var_tilde)
        results_weight_tilde.append(weight_var_tilde)
        results_norm_hat.append(norm_var_hat)
        
        climate_mean.close()
        climate_var.close()
        dataset.close()
        
    logger.info("Merging and saving all variables")
    outputs = {
        cfg.paths.series_obs: results_obs,
        cfg.paths.auto_correlation: results_ac,
        cfg.paths.quantile_threshold: results_qa,
        cfg.paths.closest_neighbours: results_nb,
        cfg.paths.composite_anomalies: results_comp_anom,
        cfg.paths.normalised_var_tilde: results_norm_tilde,
        cfg.paths.weighted_var_tilde: results_weight_tilde,
        cfg.paths.normalised_var_hat: results_norm_hat,
        cfg.paths.chi_masks: results_chi_masks,
    }
    for path, data_list in outputs.items():
        combined_ds = xr.merge(data_list).reset_coords(drop=True)
        try:
            combined_ds.to_netcdf(output_dir / path)
        except PermissionError:
            logger.warning(f"Permission denied when saving {path}")

