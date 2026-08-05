from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from pandas import Timedelta

from instantonanalysis.instanton._typing import TYPE_CHECKING
from instantonanalysis.instanton.metrics import DISTANCE_FUNCTIONS
from instantonanalysis.instanton.schemas.box.lonlat import LonLatBox

if TYPE_CHECKING:
    from instantonanalysis.instanton._typing import DistanceFunction
    from instantonanalysis.instanton.variable import VariableConfig
    from instantonanalysis.instanton.xconfig import XConfig


def _get_r_lags(r: int) -> np.ndarray:
    return np.arange(-r // 2 + 1, r // 2 + 1)

def _nearest_indices(time_coords: np.ndarray, target_flat: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(time_coords, target_flat, side='left')
    idx = np.clip(idx, 0, len(time_coords) - 1)
    left = np.clip(idx - 1, 0, len(time_coords) - 1)
    d_left = np.abs(time_coords[left] - target_flat)
    d_right = np.abs(time_coords[idx] - target_flat)
    return np.where(d_left < d_right, left, idx)

def build_event_cube(
        data: xr.Dataset, 
        event_dates: xr.Dataset, 
        xconfig: XConfig,
    ) -> xr.Dataset:
    time_dim = xconfig.time_dim
    lag_dim = xconfig.lag
    event_dim = xconfig.event
    rolling_dim = xconfig.rolling_period
    quantile_dim = xconfig.quantile

    rolling_values = event_dates[rolling_dim]

    # Get max lags across all r
    max_r = int(rolling_values.max())
    global_lags = np.arange(-max_r // 2 + 1, max_r // 2 + 1)
    lag_da = xr.DataArray(global_lags, dims=[lag_dim], coords={lag_dim: global_lags})

    event_dates = event_dates.dropna(dim=time_dim).rename({time_dim: event_dim})

    target_times = xr.apply_ufunc(
        lambda dates, lag: np.datetime64(dates, 'ns') + np.timedelta64(int(lag), 'D'),
        event_dates,
        lag_da,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.dtype('datetime64[ns]')],
    )
    if hasattr(target_times, 'compute'):
        target_times = target_times.compute()

    # Find the nearest index in data's time axis for every target time
    time_coords = np.array([np.datetime64(t, 'ns') for t in data[time_dim].values])
    nearest_indices = _nearest_indices(time_coords, target_times.values.ravel())
    nearest_indices = nearest_indices.reshape(target_times.shape)

    idx_da = xr.DataArray(
        nearest_indices,
        dims=target_times.dims,
        coords=target_times.coords,
    )
    cube = data.isel({time_dim: idx_da})

    if time_dim in cube.coords:
        cube = cube.drop_vars(time_dim)

    # Mask to filter lags (not all rolling periods use the full lag range)
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
    """Select the (start, end) month window, wrapping over the year boundary.

    A start month after the end month (e.g. DJF = (12, 2)) selects the months
    that wrap through December, rather than the empty contiguous range.
    """
    months = series[f"{time_dim}.month"]
    start, end = filter_months
    if start <= end:
        mask = (months >= start) & (months <= end)
    else:
        mask = (months >= start) | (months <= end)
    return series.sel({time_dim: mask})

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
    """Put raw data into the units the pipeline stores, e.g. t2m K -> degC.

    Only `offset` is applied. `var_cfg.scale` is a *plotting* factor that
    `plot.py` divides by (z500 geopotential -> metres); applying it here would
    inflate the stored data by g.
    """
    if var_cfg.squeeze:
        data = data.squeeze(*var_cfg.squeeze)
    if var_cfg.transpose:
        data = data.transpose(*var_cfg.transpose)
    return data + var_cfg.offset
