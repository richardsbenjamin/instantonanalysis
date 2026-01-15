import logging
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

from instantonanalysis.instanton.analysis import (
    calculate_chi_mask,
    calculate_observable,
    calculate_weighted_spatial_mean,
)
from instantonanalysis.instanton.lonlat import (
    LatitudeSystem,
    LonLatBox,
    LongitudeSystem,
    get_lon_lat_box,
)
from instantonanalysis.instanton.nbclosest import NClosestCalc, get_nclosest_config
from instantonanalysis.instanton.utils import (
    build_event_cube,
    get_distance_function,
    get_df_array,
    load_config,
    read_dataset,
)

if TYPE_CHECKING:
    from instantonanalysis.instanton.nbclosest import NClosestConfig
    

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNKS = {
    'lat': 65, 
    'lon': 180
}
na_box = LonLatBox(
    lon_min=-80,
    lon_max=50,
    lat_min=22.5,
    lat_max=70,
    lon_system=LongitudeSystem.EAST_WEST,
    lat_system=LatitudeSystem.SOUTH_NORTH,
)

if __name__ == "__main__":
    logger.info("Loading config")
    cfg = load_config()
    data_root_path = Path(cfg.paths.data_root)
    j_list = range(*cfg.analysis.j_list)
    lon_lat_box = get_lon_lat_box(cfg.locations)
    output_dir = Path(cfg.paths.results_root + cfg.locations.output_folder)
    xconfig = cfg.xarray

    results_obs = []
    results_ac = []
    results_qa = []
    results_nb = []
    results_comp_anom = []
    results_norm_tilde = []
    results_weight_tilde = []
    results_norm_hat = []

    for var_cfg in cfg.variables.values():
        logger.info("Loading datasets")
        climate_mean = read_dataset(data_root_path / var_cfg.climate_mean_path)[var_cfg.name]
        climate_var = read_dataset(data_root_path / var_cfg.climate_var_path)[var_cfg.name]
        dataset = read_dataset(data_root_path / var_cfg.data_path)[var_cfg.name]
    
        logger.info("Calculating observable")
        series_obs = calculate_observable(
            dataset,
            var=var_cfg.name,
            calc_months=cfg.analysis.calc_months_init,
            lon_lat_box=lon_lat_box,
            xconfig=xconfig,
        )

        logger.info("Calculating nclosest")
        nclosest_config = get_nclosest_config(
            analysis_cfg=cfg.analysis,
            time_dim=cfg.xarray.time_dim,
        )
        nclosest_calc = NClosestCalc(
            series_obs=series_obs,
            config=nclosest_config,
            xconfig=cfg.xarray,
            output_dir=output_dir,
            dist_func=get_distance_function(cfg.analysis.dist_func),
        )
        nclosest_calc.calculate()

        results_obs.append(series_obs.rename(var_cfg.name))
        results_ac.append(nclosest_calc.results_ac.rename(var_cfg.name))
        results_qa.append(nclosest_calc.results_qa.rename(var_cfg.name))
        results_nb.append(nclosest_calc.results_nb.rename(var_cfg.name))

        logger.info("Building event cube")
        dataset = dataset.chunk(CHUNKS)
        # 6D cube: rolling_period, quantile, lag, event, lat, lon
        event_cube = build_event_cube(
            dataset, var_cfg, j_list, nclosest_calc.results_nb, xconfig,
        )
        event_cube = event_cube.chunk(CHUNKS)

        logger.info("Calculating climate data")
        logger.info("Calculating composite anomalies")
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
        combined_ds.to_netcdf(output_dir / path)

