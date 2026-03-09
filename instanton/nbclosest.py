from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import dask
import numpy as np
import xarray as xr

from instantonanalysis.instanton.analysis import (
    calculate_autocorrelation,
    calculate_closest_neighbors,
    calculate_quantiles,
    calculate_rolling,
)
from instantonanalysis.instanton.utils import (
    create_folder,
    filter_by_months,
)
from instantonanalysis.instanton.schemas import NClosestConfig

if TYPE_CHECKING:
    from instantonanalysis.instanton._typing import DistanceFunction
    from instantonanalysis.instanton.schemas import AnalysisConfig


def get_nclosest_config(analysis_cfg: AnalysisConfig, time_dim: str) -> NClosestConfig:
    return NClosestConfig(
        ac_days=analysis_cfg.autocorr_days,
        calc_months=analysis_cfg.calc_months,
        nb_closest=analysis_cfg.nb_closest,
        rolling_periods=analysis_cfg.rolling_periods,
        quantiles=analysis_cfg.quantiles,
        time_dim=time_dim,
    )

class NClosestCalc:

    def __init__(
            self,
            series_obs: xr.DataArray,
            config: NClosestConfig,
            xconfig: XConfig,
            dist_func: DistanceFunction,
        ) -> None:
        self.series_obs = series_obs
        self.config = config
        self.xconfig = xconfig
        self.n_q = len(config.quantiles)
        self.n_r = len(config.rolling_periods)
        self.time_dim = config.time_dim
        self.dist_func = dist_func

    def _calculate_autocorrelation(self, series_obs_rolling: xr.DataArray) -> xr.DataArray:
        ac_values = calculate_autocorrelation(
            series_obs_rolling, self.time_dim, self.config.ac_days
        )
        return xr.DataArray(
            ac_values, 
            dims=[self.xconfig.lag], 
            coords={self.xconfig.lag: np.arange(self.config.ac_days)},
            name="autocorrelation"
        )

    def _calculate_closest_neighbors(
            self,
            series_obs_rolling: xr.DataArray,
            q_array: xr.DataArray,
            spacing: int,
        ) -> Tuple[xr.DataArray, xr.DataArray]:
        q_list, q_dates_list = [], []
        for q in q_array[self.xconfig.quantile].values:
            result, dates = calculate_closest_neighbors(
                series_obs_rolling,
                float(q_array.sel({self.xconfig.quantile: q})),
                spacing,
                self.time_dim,
                self.config.nb_closest,
                self.dist_func,
            )
            q_list.append(result.expand_dims({self.xconfig.quantile: [q]}))
            q_dates_list.append(dates.expand_dims({self.xconfig.quantile: [q]}))
            
        return (
            xr.concat(q_list, dim=self.xconfig.quantile, join="outer"),
            xr.concat(q_dates_list, dim=self.xconfig.quantile, join="outer"),
        )

    def _calculate_quantiles(self, series_obs_rolling: xr.DataArray) -> xr.DataArray:
        return calculate_quantiles(
            series_obs_rolling, self.config.quantiles, self.time_dim,
        ).rename({"quantile": self.xconfig.quantile})

    def calculate(self, *, save: bool = True) -> None:
        ac_list, qa_list, nb_list, nb_dates_list = [], [], [], []

        for i, r in enumerate(self.config.rolling_periods):
            spacing = 15 if r <= 20 else 30
            
            series_obs_rolling = calculate_rolling(self.series_obs, self.time_dim, r)
            series_obs_rolling_filtered = filter_by_months(
                series_obs_rolling, self.time_dim, self.config.calc_months
            )
            ac_da = self._calculate_autocorrelation(series_obs_rolling_filtered).expand_dims({self.xconfig.rolling_period: [r]})
            qa_da = self._calculate_quantiles(series_obs_rolling_filtered).expand_dims({self.xconfig.rolling_period: [r]})
            nb_da, nb_dates_da = self._calculate_closest_neighbors(
                series_obs_rolling_filtered, qa_da, spacing
            )

            ac_list.append(ac_da)
            qa_list.append(qa_da)
            nb_list.append(nb_da.expand_dims({self.xconfig.rolling_period: [r]}))
            nb_dates_list.append(nb_dates_da.expand_dims({self.xconfig.rolling_period: [r]}))

        self.results_ac, self.results_qa, self.results_nb, self.results_nb_dates = dask.compute(
            xr.concat(ac_list, dim=self.xconfig.rolling_period, join="outer"),
            xr.concat(qa_list, dim=self.xconfig.rolling_period, join="outer"),
            xr.concat(nb_list, dim=self.xconfig.rolling_period, join="outer"),
            xr.concat(nb_dates_list, dim=self.xconfig.rolling_period, join="outer"),
        )



"""
def calculate_quantiles(
        series_obs: xr.DataArray,
        quantiles: list[float],
        time_dim: str
    ) -> xr.DataArray:
    return series_obs.quantile(q=quantiles, dim=time_dim)

def calculate_rolling(series: xr.DataArray, time_dim: str, rol_days: int) -> xr.DataArray:
    return series.rolling(**{time_dim: rol_days, "center": True}).mean()

def calculate_autocorrelation(
        series_obs_rolling: xr.DataArray,
        time_dim: str,
        ac_days: int,
    ) -> np.ndarray:
    doy_str = f'{time_dim}.dayofyear'
    
    # Calculate groupby mean once, let xarray handle broadcasting during subtraction
    rolling_deseasonalised = series_obs_rolling - series_obs_rolling.groupby(doy_str).mean()
    
    auto_corr_tab = np.zeros(ac_days)
    for i in range(ac_days):
        rolled = rolling_deseasonalised.roll(**{time_dim: i})
        auto_corr_tab[i] = xr.corr(rolling_deseasonalised, rolled)
    return auto_corr_tab

def calculate_closest_neighbors_from_distances(
        series_obs: xr.DataArray,
        distances: np.ndarray,
        spacing: int, 
        time_dim: str,
        nb_closest: int, 
    ) -> Tuple[xr.DataArray, xr.DataArray]:
    
    times = series_obs[time_dim]
    years = times.dt.year.values
    doys = times.dt.dayofyear.values
    
    sort_idx = np.argsort(distances)
    keep_original_indices = []
    
    if len(years) > 0:
        min_year = years.min()
        selected_doys_per_year = [[] for _ in range(years.max() - min_year + 1)]
    else:
        selected_doys_per_year = []

    for idx in sort_idx:
        if len(keep_original_indices) == nb_closest:
            break  
            
        y_idx = years[idx] - min_year
        d = doys[idx]
        
        is_blocked = False
        for selected_d in selected_doys_per_year[y_idx]:
            if abs(selected_d - d) < spacing:
                is_blocked = True
                break
                
        if not is_blocked:
            keep_original_indices.append(idx)
            selected_doys_per_year[y_idx].append(d)

    if not keep_original_indices:
        empty_da = xr.DataArray([], coords={time_dim: np.array([], dtype='datetime64[ns]')}, dims=(time_dim,))
        return empty_da, empty_da.copy()
        
    result = series_obs.isel({time_dim: keep_original_indices})
    
    n = len(result)
    indices_da = xr.DataArray(
        result[time_dim].values,
        dims=[time_dim],
        coords={time_dim: np.arange(n)},
    )
    
    return result, indices_da

class NClosestCalc2:

    def __init__(
            self,
            series_obs: xr.DataArray,
            config: 'NClosestConfig',
            xconfig: 'XConfig',
            dist_func: Callable,
        ) -> None:
        self.series_obs = series_obs
        self.config = config
        self.xconfig = xconfig
        self.n_q = len(config.quantiles)
        self.n_r = len(config.rolling_periods)
        self.time_dim = config.time_dim
        self.dist_func = dist_func

    def _calculate_autocorrelation(self, series_obs_rolling: xr.DataArray) -> xr.DataArray:
        ac_values = calculate_autocorrelation(
            series_obs_rolling, self.time_dim, self.config.ac_days
        )
        return xr.DataArray(
            ac_values, 
            dims=[self.xconfig.lag], 
            coords={self.xconfig.lag: np.arange(self.config.ac_days)},
            name="autocorrelation"
        )

    def _calculate_quantiles(self, series_obs_rolling: xr.DataArray) -> xr.DataArray:
        return calculate_quantiles(
            series_obs_rolling, self.config.quantiles, self.time_dim,
        ).rename({"quantile": self.xconfig.quantile})

    def _calculate_closest_neighbors(
            self,
            series_obs_rolling: xr.DataArray,
            q_array: xr.DataArray,
            spacing: int,
        ) -> Tuple[xr.DataArray, xr.DataArray]:
        q_list, q_dates_list = [], []
        for q in q_array[self.xconfig.quantile].values:
            result, dates = calculate_closest_neighbors_optimised(
                series_obs_rolling,
                float(q_array.sel({self.xconfig.quantile: q})),
                spacing,
                self.time_dim,
                self.config.nb_closest,
                self.dist_func,
            )
            q_list.append(result.expand_dims({self.xconfig.quantile: [q]}))
            q_dates_list.append(dates.expand_dims({self.xconfig.quantile: [q]}))
            
        return (
            xr.concat(q_list, dim=self.xconfig.quantile, join="outer"),
            xr.concat(q_dates_list, dim=self.xconfig.quantile, join="outer"),
        )

    def calculate(self, *, save: bool = True) -> None:
        rolling_series_dict, qa_da_dict, distance_graph = {}, {}, {}
        ac_list, qa_list, nb_list, nb_dates_list = [], [], [], []
        
        # Build the Dask graph for distances
        for r in self.config.rolling_periods:
            series_obs_rolling = calculate_rolling(self.series_obs, self.time_dim, r)
            series_obs_rolling_filtered = filter_by_months(
                series_obs_rolling, self.time_dim, self.config.calc_months
            )
            rolling_series_dict[r] = series_obs_rolling_filtered
            
            qa_da = self._calculate_quantiles(series_obs_rolling_filtered)
            qa_da_dict[r] = qa_da
            
            for q in qa_da[self.xconfig.quantile].values:
                obs_q_val = float(qa_da.sel({self.xconfig.quantile: q}))
                distance_graph[(r, q)] = self.dist_func(series_obs_rolling_filtered, obs_q_val)

        computed_distances = dask.compute(distance_graph)[0]

        for r in self.config.rolling_periods:
            spacing = 15 if r <= 20 else 30
            series_filtered = rolling_series_dict[r]
            qa_da = qa_da_dict[r]
            
            ac_da = self._calculate_autocorrelation(series_filtered).expand_dims({self.xconfig.rolling_period: [r]})
            qa_da_expanded = qa_da.expand_dims({self.xconfig.rolling_period: [r]})
            
            q_list, q_dates_list = [], []
            for q in qa_da[self.xconfig.quantile].values:
                distances_arr = computed_distances[(r, q)]
                
                result, dates = calculate_closest_neighbors_from_distances(
                    series_filtered,
                    distances_arr,
                    spacing,
                    self.time_dim,
                    self.config.nb_closest,
                )
                q_list.append(result.expand_dims({self.xconfig.quantile: [q]}))
                q_dates_list.append(dates.expand_dims({self.xconfig.quantile: [q]}))
                
            nb_da = xr.concat(q_list, dim=self.xconfig.quantile, join="outer")
            nb_dates_da = xr.concat(q_dates_list, dim=self.xconfig.quantile, join="outer")

            ac_list.append(ac_da)
            qa_list.append(qa_da_expanded)
            nb_list.append(nb_da.expand_dims({self.xconfig.rolling_period: [r]}))
            nb_dates_list.append(nb_dates_da.expand_dims({self.xconfig.rolling_period: [r]}))

        self.results_ac, self.results_qa, self.results_nb, self.results_nb_dates = dask.compute(
            xr.concat(ac_list, dim=self.xconfig.rolling_period, join="outer"),
            xr.concat(qa_list, dim=self.xconfig.rolling_period, join="outer"),
            xr.concat(nb_list, dim=self.xconfig.rolling_period, join="outer"),
            xr.concat(nb_dates_list, dim=self.xconfig.rolling_period, join="outer"),
        )
"""