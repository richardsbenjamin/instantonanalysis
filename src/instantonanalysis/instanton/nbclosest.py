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
from instantonanalysis.instanton.utils import filter_by_months

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
                float(q_array.sel({self.xconfig.quantile: q}).squeeze()),
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

