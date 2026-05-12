from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr
from pandas import Timedelta
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore

import instantonanalysis.hydra_logic
from instantonanalysis.instanton._typing import TYPE_CHECKING
from instantonanalysis.instanton.metrics import DISTANCE_FUNCTIONS
from instantonanalysis.instanton.schemas.box.lonlat import LonLatBox
from instantonanalysis.instanton.schemas.box.healpix import HealPixBox

if TYPE_CHECKING:
    from datetime import datetime

    from omegaconf import DictConfig

    from instantonanalysis.instanton._typing import DistanceFunction
    from instantonanalysis.instanton.variable import VariableConfig
    from instantonanalysis.instanton.xconfig import XConfig


def _get_r_lags(r: int) -> np.ndarray:
    return np.arange(-r // 2 + 1, r // 2 + 1)

def build_event_cube(
        data: xr.Dataset, 
        var_cfg: VariableConfig, 
        event_dates: xr.Dataset, 
        xconfig: XConfig,
    ) -> xr.Dataset:
    time_dim = xconfig.time_dim
    lag_dim = xconfig.lag
    event_dim = xconfig.event
    rolling_dim = xconfig.rolling_period
    quantile_dim = xconfig.quantile

    rolling_values = event_dates[rolling_dim]

    # Transform data before any other operations 
    data = transform_data(data, var_cfg)

    # Get max lags across all r
    max_r = int(rolling_values.max())
    global_lags = np.arange(-max_r // 2 + 1, max_r // 2 + 1)
    lag_da = xr.DataArray(global_lags, dims=[lag_dim], coords={lag_dim: global_lags})

    # dropna is not lazy but event_dates should already have been computed
    event_dates = event_dates.dropna(dim=time_dim).rename({time_dim: event_dim})

    # Compute target_times eagerly — it's small (rolling_period × quantile × event × lag)
    target_times = xr.apply_ufunc(
        lambda dates, lag: np.datetime64(int(dates), 'ns') + np.timedelta64(int(lag), 'D'),
        event_dates,
        lag_da,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[event_dates.dtype],
    )
    if hasattr(target_times, 'compute'):
        target_times = target_times.compute()

    # Instead of data.sel(time=target_times, method='nearest') which requires
    # the full time axis in memory, find the unique indices we need and load
    # only those timesteps.
    time_coords = data[time_dim].values
    target_flat = target_times.values.ravel()
    nearest_indices = np.searchsorted(time_coords, target_flat, side='left')
    # Clamp and pick the closer of the two neighbors
    nearest_indices = np.clip(nearest_indices, 0, len(time_coords) - 1)
    left = np.clip(nearest_indices - 1, 0, len(time_coords) - 1)
    d_left = np.abs(time_coords[left] - target_flat)
    d_right = np.abs(time_coords[nearest_indices] - target_flat)
    nearest_indices = np.where(d_left < d_right, left, nearest_indices)

    # Get only the unique indices we need, load that small subset
    unique_indices, inverse = np.unique(nearest_indices, return_inverse=True)
    data_subset = data.isel({time_dim: unique_indices}).compute()

    # Map the flat inverse back to the shape of target_times to build the cube
    idx_into_subset = inverse.reshape(target_times.shape)

    cube = data_subset.isel({time_dim: xr.DataArray(idx_into_subset, dims=target_times.dims)})
    # Drop the original time coordinate to avoid confusion
    if time_dim in cube.coords:
        cube = cube.drop_vars(time_dim)

    # Mask to filter our maximum lags (not all rolling periods go to max lag)
    lag_mask = xr.DataArray(
        np.stack([np.isin(global_lags, _get_r_lags(r)) for r in rolling_values]),
        dims=[rolling_dim, lag_dim],
        coords={rolling_dim: rolling_values, lag_dim: global_lags}
    )
    cube = cube.where(lag_mask)
    return cube.transpose(
        rolling_dim, quantile_dim, lag_dim, 
        event_dim, *xconfig.spatial_dims
    )

def concat(
        data: list[xr.DataArray], 
        dim: str, 
        assign_coords: dict = {}, 
        **expand_kwargs: dict
    ) -> xr.DataArray:
    res = xr.concat(data, dim=dim, join="outer").assign_coords(assign_coords)
    for k, v in expand_kwargs.items():
        res = res.expand_dims({k: v})
    return res

def convert_timedelta2datetime(dataset: xr.Dataset, xconfig: XConfig) -> xr.Dataset:
    init_time = dataset["time"].values.astype("datetime64[ns]")
    steps = dataset[xconfig.time_dim].values 

    valid_times = init_time + steps 

    valid_time_da = xr.DataArray(valid_times, coords={xconfig.time_dim: dataset[xconfig.time_dim]}, dims=[xconfig.time_dim])
    dataset = dataset.assign_coords({xconfig.time_dim: valid_time_da})

    return dataset

def create_folder(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def filter_by_lon_lat(
    series: xr.DataArray, 
    lon_dim: str, 
    lat_dim: str, 
    lon_lat_box: LonLatBox
) -> xr.DataArray:
    return series.sel({
        lat_dim: slice(*lon_lat_box.lat_min_max),
        lon_dim: slice(*lon_lat_box.lon_min_max),
    })
    
def filter_by_months(
    series: xr.DataArray, 
    time_dim: str, 
    filter_months: tuple[int, int],
) -> xr.DataArray:
    return series.sel({
        time_dim: 
            (series[f"{time_dim}.month"] >= filter_months[0]) & 
            (series[f"{time_dim}.month"] <= filter_months[1])
    })

def generate_panels(rows: int, cols: int) -> list[list[str]]:
    start_code = ord('a')
    panels = []
    
    for r in range(rows):
        row_start = r * cols
        row = [f"({chr(start_code + row_start + c)})" for c in range(cols)]
        panels.append(row)
        
    return panels

def get_distance_function(dist_func: str) -> DistanceFunction:
    return DISTANCE_FUNCTIONS[dist_func]

def get_df_array(event_cube: xr.DataArray, count_dim: str, max_dims: list[str]) -> xr.DataArray:
    # degrees of freedom
    df = event_cube.count(dim=count_dim)
    return df.max(dim=max_dims) - 1

def load_config(
        config_name: str = "config", 
        overrides: Optional[List[str]] = [],
        schema_node: Optional[Node] = None,
    ) -> DictConfig:
    if overrides:
        overrides = sys.argv[1:]
    if schema_node is not None:
        cs = ConfigStore.instance()
        cs.store(name=config_name, node=schema_node)
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name=config_name, overrides=overrides)
        return cfg

def read_dataset(path: str, cftime: bool = True) -> xr.Dataset:
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=cftime)
    dataset = xr.open_dataset(path, decode_times=time_coder)
    return dataset

def select_data(data: xr.Dataset, d, j_begin: int, j_end: int, time_dim: str) -> None:
    return data.sel({
        time_dim: slice(str(d+Timedelta(days=j_begin)), str(d+Timedelta(days=j_end)))
    })

def transform_data(
        data: xr.Dataset, 
        var_cfg: VariableConfig, 
    ) -> None:
    if var_cfg.squeeze:
        data = data.squeeze(*var_cfg.squeeze)
    if var_cfg.transpose:
        data = data.transpose(*var_cfg.transpose)
    return data * var_cfg.scale_factor + var_cfg.offset