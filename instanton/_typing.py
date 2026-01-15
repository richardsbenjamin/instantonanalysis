from __future__ import annotations

from typing import (
    Callable,
    TYPE_CHECKING,
)

import xarray as xr

DistanceFunction = Callable[[xr.DataArray, float, str], xr.DataArray]
