from __future__ import annotations

import logging
from pathlib import Path

import dask
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
    write_event_cube,
)
from instantonanalysis.instanton.utils.parsers import get_calc_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Loading config")
    args = get_calc_args()
    cfg = load_config(args.config_name, args.overrides)

    data_root_path = Path(cfg.paths.data_root)
    domain_box = instantiate(cfg.domain_box)
    output_dir = Path(cfg.paths.results_root) / cfg.locations.output_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    vars_list = [var.name for var in cfg.variables.values()]
    xconfig = instantiate(cfg.xconfig)

    results_comp_anom = []
    results_norm_tilde = []
    results_weight_tilde = []
    results_norm_hat = []
    results_chi_masks = []

    logger.info("Loading datasets")
    datasets = {}
    climate_mean_in = read_dataset(data_root_path / cfg.paths.climate_mean)[vars_list]
    climate_var_in = read_dataset(data_root_path / cfg.paths.climate_variance)[vars_list]
    dataset_in = read_dataset(data_root_path / cfg.paths.data_file)[vars_list]

    for var_cfg in cfg.variables.values():
        datasets[var_cfg.name] = {
            "dataset": dataset_in[var_cfg.name],
            "climate_mean": climate_mean_in[var_cfg.name],
            "climate_var": climate_var_in[var_cfg.name],
        }

    logger.info("Loading neighbour dates")
    neighbour_dates = read_dataset(
        output_dir / cfg.paths.neighbour_dates
    )[cfg.variables["t2m"].name]

    for var_cfg in cfg.variables.values():
        logger.info("Building event cube")

        dataset = datasets[var_cfg.name]["dataset"].squeeze()

        dataset = dataset.chunk({
            xconfig.time_dim: 1000,
            **{dim: -1 for dim in xconfig.spatial_dims},
        })

        event_cube = build_event_cube(
            dataset, neighbour_dates, xconfig,
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
        write_event_cube(event_cube, event_cube_path, xconfig)

        del event_cube
        event_cube = xr.open_zarr(event_cube_path)[var_cfg.name]

        climate_mean = datasets[var_cfg.name]["climate_mean"]
        climate_var = datasets[var_cfg.name]["climate_var"]

        logger.info("Computing statistics")
        comp_anom = (event_cube.mean(xconfig.lag).mean(xconfig.event) - climate_mean).rename(var_cfg.name)
        norm_var_tilde = (event_cube.var(xconfig.event) / climate_var).rename(var_cfg.name)
        weight_var_tilde = domain_box.spatial_mean(norm_var_tilde * 100).rename(var_cfg.name)
        norm_var_hat = (event_cube.mean(xconfig.lag).var(xconfig.event) / climate_var).rename(var_cfg.name)

        df = calculate_degrees_of_freedom(
            event_cube,
            count_dim=xconfig.event,
            max_dims=xconfig.spatial_dims,
            pre_mean_dim=xconfig.lag
        )
        chi_mask = calculate_chi_mask(norm_var_hat, df)

        comp_anom, norm_var_tilde, weight_var_tilde, norm_var_hat, chi_mask = dask.compute(
            comp_anom,
            norm_var_tilde,
            weight_var_tilde,
            norm_var_hat,
            chi_mask,
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

    neighbour_dates.close()
