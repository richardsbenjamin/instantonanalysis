from __future__ import annotations

from typing import (
    Callable,
    Union,
    TYPE_CHECKING,
)

import xarray as xr

xrArray = Union[xr.Dataset, xr.DataArray]
DistanceFunction = Callable[[xr.DataArray, float, str], xr.DataArray]
