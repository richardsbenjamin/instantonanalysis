from __future__ import annotations

import numpy as np

from instantonanalysis.instanton._typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr

def euclidean(series: xr.DataArray, v: float, dim: str) -> xr.DataArray:
    return np.sqrt(((series - v) ** 2).sum(dim=dim))

def mean_squared_error(series: xr.DataArray, v: float, dim: str) -> xr.DataArray:
    return ((series - v) ** 2).mean(dim=dim)

def squared_error(series: xr.DataArray, v: float) -> xr.DataArray:
    return (series - v) ** 2


DISTANCE_FUNCTIONS = {
    "euclidean": euclidean,
    "mean_squared_error": mean_squared_error,
    "squared_error": squared_error,
}
