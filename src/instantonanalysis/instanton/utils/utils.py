from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from pandas import isnull

from instantonanalysis.instanton._typing import TYPE_CHECKING
from instantonanalysis.instanton.metrics import DISTANCE_FUNCTIONS

if TYPE_CHECKING:
    from instantonanalysis.instanton._typing import DistanceFunction
    from instantonanalysis.instanton.schemas.xconfig import XConfig


PAD_DATE_YEAR = 1


def _get_r_lags(r: int) -> np.ndarray:
    return np.arange(-r // 2 + 1, r // 2 + 1)


def pad_date_like(date) -> object:
    if isinstance(date, np.datetime64):
        return np.datetime64("NaT")
    return type(date)(PAD_DATE_YEAR, 1, 1)


def is_pad_date(values: np.ndarray) -> np.ndarray:
    if _is_cftime(values):
        years = np.array([
            getattr(v, "year", PAD_DATE_YEAR) for v in values.ravel()
        ]).reshape(values.shape)
        return years == PAD_DATE_YEAR
    return np.asarray(isnull(values))


def _is_cftime(values: np.ndarray) -> bool:
    return values.dtype == object


def _as_datetime64(values: np.ndarray) -> np.ndarray:
    return np.array([np.datetime64(t, 'ns') for t in values])


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

    max_r = int(rolling_values.max())
    global_lags = np.arange(-max_r // 2 + 1, max_r // 2 + 1)
    lag_da = xr.DataArray(global_lags, dims=[lag_dim], coords={lag_dim: global_lags})

    event_dates = event_dates.rename({time_dim: event_dim})

    pad_mask = is_pad_date(event_dates.values)
    if pad_mask.any():
        first_event = event_dates.values[~pad_mask][0]
        event_dates = event_dates.copy(
            data=np.where(pad_mask, first_event, event_dates.values)
        )
    event_mask = xr.DataArray(
        ~pad_mask, dims=event_dates.dims, coords=event_dates.coords
    )

    cftime_dates = _is_cftime(event_dates.values)
    if cftime_dates:
        shift_by_lag = lambda dates, lag: dates + timedelta(days=int(lag))
    else:
        shift_by_lag = lambda dates, lag: np.datetime64(dates, 'ns') + np.timedelta64(int(lag), 'D')

    target_times = xr.apply_ufunc(
        shift_by_lag,
        event_dates,
        lag_da,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.dtype('O') if cftime_dates else np.dtype('datetime64[ns]')],
    )
    if hasattr(target_times, 'compute'):
        target_times = target_times.compute()

    time_coords = data[time_dim].values
    if not _is_cftime(time_coords):
        time_coords = _as_datetime64(time_coords)
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

    lag_mask = xr.DataArray(
        np.stack([np.isin(global_lags, _get_r_lags(r)) for r in rolling_values]),
        dims=[rolling_dim, lag_dim],
        coords={rolling_dim: rolling_values, lag_dim: global_lags}
    )
    cube = cube.where(lag_mask & event_mask)
    return cube.transpose(
        rolling_dim, quantile_dim, lag_dim, 
        event_dim, *xconfig.spatial_dims
    )


def write_event_cube(
        event_cube: xr.DataArray,
        path: str | Path,
        xconfig: XConfig,
    ) -> None:
    rolling_dim, quantile_dim = xconfig.rolling_period, xconfig.quantile
    event_cube.to_zarr(path, mode="w", compute=False)

    for i in range(event_cube.sizes[rolling_dim]):
        for j in range(event_cube.sizes[quantile_dim]):
            region = {rolling_dim: slice(i, i + 1), quantile_dim: slice(j, j + 1)}
            slab = event_cube.isel(region).drop_vars(event_cube.coords)
            with dask.config.set(scheduler="synchronous"):
                slab.to_zarr(path, region=region)

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


def filter_by_months(
    series: xr.DataArray,
    time_dim: str,
    filter_months: tuple[int, int],
) -> xr.DataArray:
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
