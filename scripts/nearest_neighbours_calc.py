from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr
from hydra.utils import instantiate

from instantonanalysis.instanton.nbclosest import NClosestCalc, get_nclosest_config
from instantonanalysis.instanton.utils import (
    get_distance_function,
    load_config,
    read_dataset,
)
from instantonanalysis.instanton.utils.parsers import get_calc_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Loading config")
    args = get_calc_args()
    cfg = load_config(args.config_name, args.overrides)

    xconfig = instantiate(cfg.xconfig)
    output_dir = Path(cfg.paths.results_root) / cfg.locations.output_folder

    var_cfg = cfg.variables["t2m"]

    logger.info("Loading observable")
    series_obs = read_dataset(output_dir / cfg.paths.series_obs)[var_cfg.name]

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

    logger.info("Saving nearest neighbours")
    outputs = {
        cfg.paths.auto_correlation: nclosest_calc.results_ac,
        cfg.paths.quantile_threshold: nclosest_calc.results_qa,
        cfg.paths.closest_neighbours: nclosest_calc.results_nb,
        cfg.paths.neighbour_dates: nclosest_calc.results_nb_dates,
    }
    for path, result in outputs.items():
        combined_ds = xr.merge([result.rename(var_cfg.name)]).reset_coords(drop=True)
        try:
            combined_ds.to_netcdf(output_dir / path)
        except PermissionError:
            logger.warning(f"Permission denied when saving {path}")

    series_obs.close()
