from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from instantonanalysis.instanton.schema import LocationConfig


LAT_NAMES = {'lat', 'latitude'}
LON_NAMES = {'lon', 'longitude'}


def get_lon_lat_box(loc_cfg: LocationConfig) -> LonLatBox:
    return LonLatBox(
        loc_cfg.box.lon_min,
        loc_cfg.box.lon_max,
        loc_cfg.box.lat_max,
        loc_cfg.box.lat_min,
        LongitudeSystem[loc_cfg.box.lon_sys],
        LatitudeSystem[loc_cfg.box.lat_sys]
    )


class LongitudeSystem(Enum):
    """Enum for longitude coordinate systems."""
    EAST_WEST = "east_west"      # -180° to 180°
    CONTINUOUS = "continuous"    # 0° to 360°


class LatitudeSystem(Enum):
    """Enum for latitude coordinate systems."""
    NORTH_SOUTH = "north_south"  # 90°N to -90°S
    SOUTH_NORTH = "south_north"  # -90°N to 90°S


@dataclass
class LonLatBox:
    """Dataclass for storing longitude and latitude coordinates with explicit longitude system."""
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    lon_system: LongitudeSystem = LongitudeSystem.EAST_WEST
    lat_system: LatitudeSystem = LatitudeSystem.NORTH_SOUTH
    xconfig: Optional[XConfig] = None
    
    def __post_init__(self):
        """Basic coordinate validation."""
        # Longitude validation
        if self.lon_system == LongitudeSystem.EAST_WEST:
            if not (-180 <= self.lon_min <= 180 and -180 <= self.lon_max <= 180):
                raise ValueError("Longitudes must be between -180 and 180 for EAST_WEST system")
        else:
            if not (0 <= self.lon_min <= 360 and 0 <= self.lon_max <= 360):
                raise ValueError("Longitudes must be between 0 and 360 for CONTINUOUS system")

        if self.lon_min > self.lon_max:
            raise ValueError("lon_min cannot be greater than lon_max")
        
        # Latitude validation  
        if not (-90 <= self.lat_min <= 90 and -90 <= self.lat_max <= 90):
            raise ValueError("Latitudes must be between -90 and 90")
        
        if self.lat_system == LatitudeSystem.NORTH_SOUTH:
            if self.lat_min < self.lat_max:
                raise ValueError("In NORTH_SOUTH system, lat_min (south) cannot be less than lat_max (north)")
        else:
            if self.lat_min > self.lat_max:
                raise ValueError("In SOUTH_NORTH system, lat_min (north) cannot be greater than lat_max (south)")

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

    def _find_coord_names(self, ds: xr.Dataset | xr.DataArray):
        found_coords = set(ds.coords) | set(ds.dims)
        
        lon_name = next((c for c in LON_NAMES if c in found_coords), None)
        lat_name = next((c for c in LAT_NAMES if c in found_coords), None)
        
        if not lon_name:
            raise ValueError(
                "Longitude coordinate not found. Expected one of {LON_NAMES}. Found: {list(ds.coords)}"
            )
        if not lat_name:
            raise ValueError(
                "Latitude coordinate not found. Expected one of {LAT_NAMES}. Found: {list(ds.coords)}"
            )
        return lon_name, lat_name

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
        
    def extract(self, ds: xrArray, xconfig: XConfig | None = None) -> xrArray:
        res = self.enforce_coords(ds, xconfig)
        
        lon_name, lat_name = self.get_names(res, xconfig)

        self._check_lon_bounds(res[lon_name])
        self._check_lat_bounds(res[lat_name])

        try:
            subset = res.sel({
                lat_name: slice(self.lat_min, self.lat_max),
                lon_name: slice(self.lon_min, self.lon_max)
            })
            
            if subset.sizes[lat_name] == 0 or subset.sizes[lon_name] == 0:
                raise ValueError(f"Extraction resulted in an empty dataset for box: {self}")
                
            return subset
            
        except Exception as e:
            raise RuntimeError(f"Error during spatial extraction: {str(e)}")

    def get_names(self, ds: xrArray, xconfig: Optional[XConfig] = None) -> Tuple[str, str]:
        if xconfig:
            return xconfig.lon_dim, xconfig.lat_dim
        if self.xconfig:
            return self.xconfig.lon_dim, self.xconfig.lat_dim
        return self._find_coord_names(ds)

    @property
    def lon_min_max(self) -> tuple:
        return self.lon_min, self.lon_max
    
    @property
    def lat_min_max(self) -> tuple:
        return self.lat_min, self.lat_max