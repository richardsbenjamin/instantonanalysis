from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING

from instantonanalysis.instanton._typing import TYPE_CHECKING
from instantonanalysis.instanton.schemas.xconfig import XConfig
from instantonanalysis.instanton.schemas.schemas import (
    AnalysisConfig,
    LocationConfig,
    NClosestConfig,
    PathConfig,
    VariableConfig,
)


@dataclass
class Config:
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    location: LocationConfig = MISSING
    paths: PathConfig = field(
        default_factory=lambda: PathConfig(
            data_root="./data", input_data="", results_root="./outputs"
        )
    )
    variable: VariableConfig = MISSING
    xconfig: XConfig = MISSING



__all__ = [
    "AnalysisConfig",
    "LocationConfig",
    "NClosestConfig",
    "PathConfig",
    "VariableConfig",
    "XConfig",
    "Config",
]
