from __future__ import annotations

import logging
from pathlib import Path

import dask

from instantonanalysis.instanton.utils import (
    filter_by_months,
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

    data_root_path = Path(cfg.paths.data_root)
    time_dim = cfg.xconfig.time_dim
    vars_list = [var.name for _, var in cfg.variables.items()]

    logger.info(f"Loading {cfg.paths.data_file}")
    dataset_in = read_dataset(data_root_path / cfg.paths.data_file)
    data = dataset_in[vars_list]

    dropped_coords = {k: v for k, v in dataset_in.coords.items() if k not in data.coords}
    if dropped_coords:
        data = data.assign_coords(dropped_coords)

    calc_months = cfg.analysis.calc_months
    if calc_months:
        logger.info(f"Filtering to months {list(calc_months)}")
        period_data = filter_by_months(data, time_dim, calc_months)
    else:
        period_data = data

    logger.info(f"Computing climatology over {period_data.sizes[time_dim]} steps")
    with dask.config.set(scheduler="synchronous"):
        mean_ds, var_ds = dask.compute(
            period_data.mean(dim=time_dim),
            period_data.var(dim=time_dim),
        )

    logger.info(f"Writing {cfg.paths.climate_mean} and {cfg.paths.climate_variance}")
    mean_ds.to_netcdf(data_root_path / cfg.paths.climate_mean)
    var_ds.to_netcdf(data_root_path / cfg.paths.climate_variance)
