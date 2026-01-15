from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from omegaconf import MISSING

if TYPE_CHECKING:
    from typing import Any, List, Optional


@dataclass
class AnalysisConfig:
    autocorr_days: int = 31
    calc_months_init: tuple[int, int] = (5, 9)
    calc_months: tuple[int, int] = (6, 8)
    dist_func: str = "squared_error"
    j_list: tuple[int, int] = (-2, 2)
    nb_closest: int = 50
    rolling_periods: tuple[int, int] = (1, 5)
    quantiles: tuple[float, float] = (0.75, 0.95, 0.99, 0.999)
    
@dataclass
class BoxConfig:
    lon_min: float
    lon_max: float
    lat_max: float
    lat_min: float
    lon_sys: str = "CONTINUOUS"
    lat_sys: str = "NORTH_SOUTH"

@dataclass
class LocationConfig:
    name: str
    box: BoxConfig

@dataclass
class NClosestConfig:
    ac_days: int
    calc_months: tuple[int, int]    
    nb_closest: int
    rolling_periods: tuple[int, int]
    quantiles: tuple[float, float]
    time_dim: str

@dataclass
class PathConfig:
    data_root: str
    input_data: str
    results_root: str
    auto_correlation: Optional[str] = None
    closest_neighbours: Optional[str] = None
    climate_mean: Optional[str] = None
    climate_var: Optional[str] = None
    event_cube: Optional[str] = None
    event_mean: Optional[str] = None
    event_var: Optional[str] = None
    event_normalised_var: Optional[str] = None
    event_weighted_var: Optional[str] = None
    quantile_threshold: Optional[str] = None

@dataclass
class VariableConfig:
    name: str
    alias: str
    unit: str
    offset: float
    scale_factor: float
    contour_levels: tuple[int, int]
    transpose: Optional[tuple[str]] = None 
    squeeze: Optional[tuple[str]] = None
    
@dataclass
class XConfig:
    time_dim: str = "time"
    lon_dim: str = "lon"
    lat_dim: str = "lat"
    rolling_period: Optional[str] = "rolling_period"
    quantile: Optional[str] = "quantile"
    lag: Optional[str] = "lag"
    event: Optional[str] = "event"

@dataclass
class Config:
    location: LocationConfig = field(default_factory=LocationConfig)
    variable: VariableConfig = field(default_factory=VariableConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    xconfig: XConfig = field(default_factory=XConfig)
