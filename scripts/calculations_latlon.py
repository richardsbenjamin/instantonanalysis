import logging
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr
from hydra.utils import instantiate

from instantonanalysis.instanton.analysis import (
    calculate_chi_mask,
    calculate_observable,
    calculate_weighted_spatial_mean,
)
from instantonanalysis.instanton.nbclosest import NClosestCalc, get_nclosest_config
from instantonanalysis.instanton.utils import (
    build_event_cube,
    get_distance_function,
    get_df_array,
    load_config,
    read_dataset,
)
from instantonanalysis.utils.parsers import get_calc_args

if TYPE_CHECKING:
    from instantonanalysis.instanton.nbclosest import NClosestConfig
    

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNKS = {
    'lat': 65, 
    'lon': 180
}

if __name__ == "__main__":
    logger.info("Loading config")
    args = get_calc_args()
    cfg = load_config(args.config_name, args.overrides)
    
    data_root_path = Path(cfg.paths.data_root)
    na_box = instantiate(cfg.na_box)
    spatial_box = instantiate(cfg.box)
    output_dir = Path(cfg.paths.results_root + cfg.locations.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    xconfig = instantiate(cfg.xconfig)

    results_obs = []
    results_ac = []
    results_qa = []
    results_nb = []
    results_comp_anom = []
    results_norm_tilde = []
    results_weight_tilde = []
    results_norm_hat = []

    logger.info("Loading datasets")
    datasets = {}
    for var_cfg in cfg.variables.values():
        datasets[var_cfg.name] = {
            "dataset": read_dataset(data_root_path / var_cfg.data_path)[var_cfg.name],
            "climate_mean": read_dataset(data_root_path / var_cfg.climate_mean_path)[var_cfg.name],
            "climate_var": read_dataset(data_root_path / var_cfg.climate_var_path)[var_cfg.name],
        }

    logger.info("Calculating observable")
    var_cfg = cfg.variables["t2m"]

    series_obs = calculate_observable(
        datasets[var_cfg.name]["dataset"],
        var=var_cfg.name,
        calc_months=cfg.analysis.calc_months_init,
        spatial_box=spatial_box,
        xconfig=xconfig,
    )

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

    for var_cfg in cfg.variables.values():
        logger.info("Building event cube")
        dataset = datasets[var_cfg.name]["dataset"].chunk(CHUNKS)
        # 6D cube: rolling_period, quantile, lag, event, lat, lon
        event_cube = build_event_cube(
            dataset, var_cfg, nclosest_calc.results_nb, xconfig,
        )
        event_cube = event_cube.chunk(CHUNKS)

        logger.info("Calculating composite anomalies")
        climate_mean = datasets[var_cfg.name]["climate_mean"]
        climate_var = datasets[var_cfg.name]["climate_var"]
        results_comp_anom.append(
            (event_cube.mean(xconfig.lag).mean(xconfig.event) - climate_mean)
            .compute()
            .rename(var_cfg.name)
        )
        logger.info("Calculating normalised var tilde")
        results_norm_tilde.append(
            (event_cube.var(xconfig.event) / climate_var)
            .compute()
            .rename(var_cfg.name)
        )
        logger.info("Calculating weighted var tilde")
        results_weight_tilde.append(
            calculate_weighted_spatial_mean(results_norm_tilde[-1] * 100, na_box)
            .compute()
            .rename(var_cfg.name)
        )
        logger.info("Calculating normalised var hat")
        normalised_var_hat = (
            event_cube.mean(xconfig.lag).var(xconfig.event) / climate_var
        )
        df = get_df_array(event_cube, xconfig.event, [xconfig.lag, xconfig.lat_dim, xconfig.lon_dim])
        chi_mask = calculate_chi_mask(normalised_var_hat, df)
        results_norm_hat.append(
            normalised_var_hat.where(chi_mask)
            .compute()
            .rename(var_cfg.name)
        )
        
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
    }
    for path, data_list in outputs.items():
        combined_ds = xr.merge(data_list).reset_coords(drop=True)
        try:
            combined_ds.to_netcdf(output_dir / path)
        except PermissionError:
            logger.warning(f"Permission denied when saving {path}")

