from __future__ import annotations

import logging
from pathlib import Path

import dask
import pandas as pd

from instantonanalysis.instanton.utils import (
    filter_by_months,
    load_config,
    read_dataset,
    transform_data,
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
    preprocess = cfg.preprocess

    logger.info(f"Loading {cfg.paths.data_file}")
    dataset_in = read_dataset(data_root_path / cfg.paths.data_file)
    data = dataset_in[vars_list]

    # Selecting variables drops coords none of them use -- on the daily zarrs
    # that silently loses the size-1 `time`, which the existing climatology
    # files carry and `step_to_datetime` needs. Put them back.
    dropped_coords = {k: v for k, v in dataset_in.coords.items() if k not in data.coords}
    if dropped_coords:
        data = data.assign_coords(dropped_coords)

    # Raw inputs carry `step` as hours since `time`; the derived daily zarrs
    # already store it as datetimes.
    if preprocess.step_to_datetime:
        base_time = pd.Timestamp(str(data["time"].values.flat[0]))
        datetimes = base_time + pd.to_timedelta(data[time_dim].values, unit="h")
        data = data.assign_coords({time_dim: datetimes})

    # Likewise the derived zarrs already have offset/scale baked in, so
    # re-applying the transform there would double-count it.
    if preprocess.transform:
        for var_cfg in cfg.variables.values():
            data[var_cfg.name] = transform_data(data[var_cfg.name].squeeze(), var_cfg)

    if preprocess.resample:
        data = data.resample({time_dim: "24h"}).mean()

    if cfg.paths.daily_data:
        logger.info(f"Writing {cfg.paths.daily_data}")
        data.to_zarr(data_root_path / cfg.paths.daily_data, mode="w")

    calc_months = cfg.analysis.calc_months
    if calc_months:
        logger.info(f"Filtering to months {list(calc_months)}")
        period_data = filter_by_months(data, time_dim, calc_months)
    else:
        period_data = data

    # Both statistics stream over the same chunks, so compute them together
    # and let the synchronous scheduler keep peak memory to one chunk.
    logger.info(f"Computing climatology over {period_data.sizes[time_dim]} steps")
    with dask.config.set(scheduler="synchronous"):
        mean_ds, var_ds = dask.compute(
            period_data.mean(dim=time_dim),
            period_data.var(dim=time_dim),
        )

    logger.info(f"Writing {cfg.paths.climate_mean} and {cfg.paths.climate_variance}")
    mean_ds.to_netcdf(data_root_path / cfg.paths.climate_mean)
    var_ds.to_netcdf(data_root_path / cfg.paths.climate_variance)
