from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from instantonanalysis.instanton._typing import TYPE_CHECKING


@dataclass
class XConfig:
    time_dim: str = "time"
    rolling_period: Optional[str] = "rolling_period"
    quantile: Optional[str] = "quantile"
    lag: Optional[str] = "lag"
    event: Optional[str] = "event"

    @property
    def spatial_dims(self) -> Tuple[str, ...]:
        """Abstract property to return relevant spatial dimensions."""
        raise NotImplementedError

@dataclass
class XConfigHealPix(XConfig):
    face_dim: str = "face"
    height_dim: str = "height"
    width_dim: str = "width"
    
    @property
    def spatial_dims(self) -> Tuple[str, str, str]:
        return (self.face_dim, self.height_dim, self.width_dim)

@dataclass
class XConfigLonLat(XConfig):
    lon_dim: str = "lon"
    lat_dim: str = "lat"
    
    @property
    def spatial_dims(self) -> Tuple[str, str]:
        return (self.lon_dim, self.lat_dim)


