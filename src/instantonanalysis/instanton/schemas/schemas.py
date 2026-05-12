from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from instantonanalysis.instanton.schemas.box import LonLatBox


@dataclass
class AnalysisConfig:
    autocorr_days: int = 31
    calc_months_init: tuple[int, int] = (5, 9)
    calc_months: tuple[int, int] = (6, 8)
    dist_func: str = "squared_error"
    nb_closest: int = 50
    rolling_periods: tuple[int, int] = (1, 5)
    quantiles: tuple[float, float] = (0.75, 0.95, 0.99, 0.999)
    
@dataclass
class LocationConfig:
    name: str
    box: LonLatBox #Any # schemas.box.IBox

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
    data_file: str
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
    