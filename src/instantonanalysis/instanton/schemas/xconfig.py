from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class XConfig:
    time_dim: str = "time"
    channel_dim: Optional[str] = None
    rolling_period: Optional[str] = "rolling_period"
    quantile: Optional[str] = "quantile"
    lag: Optional[str] = "lag"
    event: Optional[str] = "event"
    chunk_size: Optional[str] = "1GiB"

    @property
    def spatial_dims(self) -> Tuple[str, ...]:
        raise NotImplementedError

@dataclass
class XConfigHealPix(XConfig):
    face_dim: str = "face"
    height_dim: str = "height"
    width_dim: str = "width"
    
    @property
    def spatial_dims(self) -> Tuple[str, str, str]:
        return (self.face_dim, self.height_dim, self.width_dim)
