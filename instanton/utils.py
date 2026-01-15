from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore

from instantonanalysis.instanton._typing import TYPE_CHECKING
from instantonanalysis.instanton.metrics import DISTANCE_FUNCTIONS

if TYPE_CHECKING:
    from datetime import datetime

    from omegaconf import DictConfig

    from instantonanalysis.instanton._typing import DistanceFunction
    from instantonanalysis.instanton.lonlat import LonLatBox
    from instantonanalysis.instanton.variable import VariableConfig
    from instantonanalysis.instanton.xconfig import XConfig


def build_event_cube(
        data: xr.Dataset, 
        var_cfg: VariableConfig, 
        j_list: list[int], 
        neighbors_da: xr.Dataset, 
        xconfig: XConfig,
    ) -> xr.Dataset:
    time_dim = xconfig.time_dim
    r_results = []

    for r_val in neighbors_da[xconfig.rolling_period].values:
        q_results = []
        r_slice = neighbors_da.sel({xconfig.rolling_period: r_val})

        for q_val in r_slice[xconfig.quantile].values:
            dates = (r_slice
                .sel({xconfig.quantile: q_val})
                .dropna(time_dim)
                .sortby(time_dim, ascending=False)
                [time_dim].values
            )
            event_windows = []
            for d in dates:
                window = select_data(data, d, j_list[0], j_list[-1])
                window = transform_data(window, var_cfg)
                
                if window.sizes[time_dim] == len(j_list):
                    window = window.rename({time_dim: xconfig.lag}).assign_coords({xconfig.lag: j_list})
                    event_windows.append(window)
            
            q_results.append(
                concat(
                    event_windows,
                    xconfig.event,
                    assign_coords={xconfig.event: np.arange(len(event_windows))},
                    **{xconfig.quantile: [q_val]})
            )
        r_results.append(
            concat(q_results, xconfig.quantile, **{xconfig.rolling_period: [r_val]}),
        )
    return concat(r_results, xconfig.rolling_period).transpose(
        xconfig.rolling_period, xconfig.quantile, xconfig.lag,
        xconfig.event, xconfig.lat_dim, xconfig.lon_dim,
    )

def concat(
        data: list[xr.DataArray], 
        dim: str, 
        assign_coords: dict = {}, 
        **expand_kwargs: dict
    ) -> xr.DataArray:
    res = xr.concat(data, dim=dim, join="outer").assign_coords(assign_coords)
    for k, v in expand_kwargs.items():
        res = res.expand_dims({k: v})
    return res

def create_folder(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def filter_by_lon_lat(
    series: xr.DataArray, 
    lon_dim: str, 
    lat_dim: str, 
    lon_lat_box: LonLatBox
) -> xr.DataArray:
    return series.sel({
        lat_dim: slice(*lon_lat_box.lat_min_max),
        lon_dim: slice(*lon_lat_box.lon_min_max),
    })
    
def filter_by_months(
    series: xr.DataArray, 
    time_dim: str, 
    filter_months: tuple[int, int],
) -> xr.DataArray:
    return series.sel({
        time_dim: 
            (series[f"{time_dim}.month"] >= filter_months[0]) & 
            (series[f"{time_dim}.month"] <= filter_months[1])
    })

def generate_panels(rows: int, cols: int) -> list[list[str]]:
    start_code = ord('a')
    panels = []
    
    for r in range(rows):
        row_start = r * cols
        row = [f"({chr(start_code + row_start + c)})" for c in range(cols)]
        panels.append(row)
        
    return panels

def get_distance_function(dist_func: str) -> DistanceFunction:
    return DISTANCE_FUNCTIONS[dist_func]

def get_df_array(event_cube: xr.DataArray, count_dim: str, max_dims: list[str]) -> xr.DataArray:
    # degrees of freedom
    df = event_cube.count(dim=count_dim)
    return df.max(dim=max_dims).compute() - 1

def load_config(
        config_name: str = "config", 
        overrides: Optional[List[str]] = None,
        schema_node: Optional[Node] = None,
    ) -> DictConfig:
    if overrides is None:
        overrides = sys.argv[1:]
    if schema_node is not None:
        cs = ConfigStore.instance()
        cs.store(name=config_name, node=schema_node)
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name=config_name, overrides=overrides)
        return cfg

def read_dataset(path: str, cftime: bool = True) -> xr.Dataset:
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=cftime)
    dataset = xr.open_dataset(path, decode_times=time_coder)
    return dataset

def select_data(data: xr.Dataset, d: datetime, j_begin: int, j_end: int) -> None:
    return data.sel(
        time=slice(str(d+timedelta(days=j_begin)), str(d+timedelta(days=j_end)))
    )

def transform_data(
        data: xr.Dataset, 
        var_cfg: VariableConfig, 
    ) -> None:
    if var_cfg.squeeze:
        data = data.squeeze(*var_cfg.squeeze)
    if var_cfg.transpose:
        data = data.transpose(*var_cfg.transpose)
    return data * var_cfg.scale_factor + var_cfg.offset