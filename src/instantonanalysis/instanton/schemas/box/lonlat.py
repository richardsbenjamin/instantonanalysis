from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import xarray as xr

from instantonanalysis.instanton.schemas.box.ibox import IBox

if TYPE_CHECKING:
    from typing import Optional

    from instantonanalysis.instanton._typing import xrArray
    from instantonanalysis.instanton.schemas.xconfig import XConfig


class LongitudeSystem(str, Enum):
    """Enum for longitude coordinate systems."""
    EAST_WEST = "EAST_WEST"      # -180° to 180°
    CONTINUOUS = "CONTINUOUS"    # 0° to 360°


class LatitudeSystem(str,Enum):
    """Enum for latitude coordinate systems."""
    NORTH_SOUTH = "NORTH_SOUTH"  # 90°N to -90°S
    SOUTH_NORTH = "SOUTH_NORTH"  # -90°N to 90°S


@dataclass
class LonLatBox(IBox):
    """Dataclass for storing longitude and latitude coordinates with explicit longitude system."""
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    lon_system: LongitudeSystem = LongitudeSystem.EAST_WEST
    lat_system: LatitudeSystem = LatitudeSystem.NORTH_SOUTH
    _target_: str = "instantonanalysis.instanton.schemas.box.lonlat.LonLatBox"

    _possible_names = {
        "Longitude": ["lon", "longitude", "lons"],
        "Latitude": ["lat", "latitude", "lats"]
    }
    
    def __post_init__(self):
        """Basic coordinate validation."""
        # Longitude validation
        if self.lon_system == LongitudeSystem.EAST_WEST:
            if not (-180 <= self.lon_min <= 180 and -180 <= self.lon_max <= 180):
                raise ValueError("Longitudes must be between -180 and 180 for EAST_WEST system")
        elif self.lon_system == LongitudeSystem.CONTINUOUS:
            if not (0 <= self.lon_min <= 360 and 0 <= self.lon_max <= 360):
                raise ValueError("Longitudes must be between 0 and 360 for CONTINUOUS system")
        else:
            raise ValueError(f"Invalid longitude system {self.lon_system}")

        if self.lon_min > self.lon_max:
            raise ValueError(f"lon_min ({self.lon_min}) cannot be greater than lon_max ({self.lon_max})")
        
        # Latitude validation  
        if not (-90 <= self.lat_min <= 90 and -90 <= self.lat_max <= 90):
            raise ValueError("Latitudes must be between -90 and 90")
        
        if self.lat_system == LatitudeSystem.NORTH_SOUTH:
            if self.lat_min < self.lat_max:
                raise ValueError("In NORTH_SOUTH system, lat_min (south) cannot be less than lat_max (north)")
        elif self.lat_system == LatitudeSystem.SOUTH_NORTH:
            if self.lat_min > self.lat_max:
                raise ValueError("In SOUTH_NORTH system, lat_min (north) cannot be greater than lat_max (south)")
        else:
            raise ValueError(f"Invalid latitude system {self.lat_system}")

    def _check_bounds(self, ds: xrArray, dims: Tuple[str, str]):
        lon_name, lat_name = dims
        self._check_lon_bounds(ds[lon_name])
        self._check_lat_bounds(ds[lat_name])

    def _check_lon_bounds(self, lon_coord: xr.DataArray):
        lons = lon_coord.values
        if not (lons.min() <= self.lon_min <= lons.max() or 
                lons.min() <= self.lon_max <= lons.max()):
            raise ValueError(
                f"Longitude box [{self.lon_min}, {self.lon_max}] is outside "
                f"dataset bounds [{lons.min():.2f}, {lons.max():.2f}]"
            )

    def _check_lat_bounds(self, lat_coord: xr.DataArray):
        lats = lat_coord.values
        if not (lats.min() <= min(self.lat_min, self.lat_max) <= lats.max() or 
                lats.min() <= max(self.lat_min, self.lat_max) <= lats.max()):
            raise ValueError(
                f"Latitude box [{self.lat_min}, {self.lat_max}] is outside "
                f"dataset bounds [{lats.min():.2f}, {lats.max():.2f}]"
            )

    def add_cyclic_point(self, ds: xrArray, xconfig: XConfig | None = None) -> xrArray:
        lon_name, _ = self.get_names(ds, xconfig)
        
        axis = ds.get_axis_num(lon_name)
        
        cyclic_data, cyclic_lon = add_cyclic_point(
            ds.values, 
            coord=ds[lon_name].values, 
            axis=axis
        )
        
        new_coords = dict(ds.coords)
        new_coords[lon_name] = cyclic_lon
        
        return xr.DataArray(
            data=cyclic_data,
            dims=ds.dims,
            coords=new_coords,
            name=ds.name,
            attrs=ds.attrs
        )

    @staticmethod
    def convert_lon_to_continuous(lons: List, lon_system: LongitudeSystem) -> List:
        if lon_system == LongitudeSystem.EAST_WEST:
            return [(lon + 360) % 360 for lon in lons]
        return lons
        
    @staticmethod
    def convert_lat_to_north_south(lats: List, lat_system: LatitudeSystem) -> List:
        if lat_system == LatitudeSystem.SOUTH_NORTH:
            return [-lat for lat in lats]
        return lats

    def enforce_coords(self, ds: xrArray, xconfig: Optional[XConfig] = None) -> xrArray:
        res = ds.copy()
        lon_name, lat_name = self.get_names(res, xconfig)

        if self.lon_system == LongitudeSystem.EAST_WEST:
            res = res.assign_coords({lon_name: (res[lon_name] + 180) % 360 - 180})
        else:
            res = res.assign_coords({lon_name: res[lon_name] % 360})
        
        res = res.sortby([lon_name, lat_name])

        is_ds_ascending = res[lat_name].values[0] < res[lat_name].values[-1]
        target_ascending = (self.lat_system == LatitudeSystem.SOUTH_NORTH)
        
        if is_ds_ascending != target_ascending:
            res = res.isel({lat_name: slice(None, None, -1)})
            
        return res

    def select(self, series: xrArray, dims: Tuple[str, str]) -> xrArray:
        lon_dim, lat_dim = dims
        return series.sel({
            lat_dim: slice(*self.lat_min_max),
            lon_dim: slice(*self.lon_min_max),
        })

    def spatial_mean(
            self,
            da: xrArray, 
            xconfig: Optional[XConfig] = None
        ) -> np.ndarray:
        if xconfig:
            lon_name, lat_name = xconfig.lon_dim, xconfig.lat_dim
        else:
            lon_name, lat_name = self.get_names(da, xconfig=None)
        da = self.extract(da, xconfig=xconfig)
        weights = np.cos(np.deg2rad(da[lat_name]))
        weights.name = "weights"
        return da.weighted(weights).mean((lon_name, lat_name))

    @property
    def attributes(self) -> tuple:
        return (
            self.lon_min,
            self.lon_max,
            self.lat_min,
            self.lat_max,
            self.lon_system,
            self.lat_system
        )

    @property
    def lon_min_max(self) -> tuple:
        return self.lon_min, self.lon_max
    
    @property
    def lat_min_max(self) -> tuple:
        return self.lat_min, self.lat_max