from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr
import numpy as np
from scipy.stats import chi2

from instantonanalysis.instanton.utils import (
    create_folder,
    filter_by_lon_lat,
    filter_by_months,   
)

if TYPE_CHECKING:
    from typing import Optional

    from instantonanalysis.instanton._typing import DistanceFunction, XArray
    from instantonanalysis.instanton.schemas.box.ibox import IBox
    from instantonanalysis.instanton.schemas import VariableConfig
    from instantonanalysis.instanton.schemas.xconfig import XConfig


def calculate_autocorrelation(
        series_obs_rolling: xr.DataArray,
        time_dim: str,
        ac_days: int,
    ) -> np.ndarray:
    doy_str = f'{time_dim}.dayofyear'
    rolling_deseasonalised = series_obs_rolling.groupby(doy_str) - series_obs_rolling.groupby(doy_str).mean()
    auto_corr_tab = np.zeros(ac_days)
    for i in range(ac_days):
        rolled = rolling_deseasonalised.roll(**{time_dim: i})
        auto_corr_tab[i] = xr.corr(rolling_deseasonalised, rolled)
    return auto_corr_tab

def calculate_chi_mask(
        var_ratio: xr.DataArray,
        df: xr.DataArray,
        confidence_level: float = 0.05,
    ) -> xr.DataArray:
    test_statistic = df * var_ratio
    critical_values = xr.apply_ufunc(
        chi2.ppf,
        confidence_level,
        df,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )
    return test_statistic < critical_values

def calculate_closest_neighbors(
        series_obs: xr.DataArray,
        obs_q: float,
        spacing: int, 
        time_dim: str,
        nb_closest: list[int],
        dist_func: DistanceFunction,
    ) -> xr.DataArray:
    yearly_results = []
    
    for _, group in series_obs.groupby(f'{time_dim}.year'):
        sorted_group = group.sortby(dist_func(group, obs_q), ascending=False)
        
        doy = getattr(sorted_group, time_dim).dt.dayofyear.values
        
        keep_indices = []
        candidates = np.arange(len(doy))
        
        while candidates.size > 0:
            best_idx = candidates[-1]
            keep_indices.append(best_idx)
            
            current_day = doy[best_idx]
            candidates = candidates[np.abs(doy[candidates] - current_day) >= spacing]

        year_filtered = sorted_group.isel({time_dim: keep_indices[::-1]})
        yearly_results.append(year_filtered)

    if not yearly_results:
        return xr.DataArray([], coords={time_dim: np.array([], dtype='datetime64[ns]')}, dims=(time_dim,))
        
    yearly_results = xr.concat(yearly_results, dim=time_dim)
    result = yearly_results.sortby(dist_func(yearly_results, obs_q))[:nb_closest]

    n = len(result)
    return result, xr.DataArray(
        result[time_dim].values,
        dims=[time_dim],
        coords={time_dim: np.arange(n)},
    )

def calculate_degrees_of_freedom(
        event_cube: xr.DataArray,
        count_dim: str,
        max_dims: list[str],
        pre_mean_dim: Optional[str] = None, 
    ) -> xr.DataArray:
    cube = event_cube.mean(dim=pre_mean_dim) if pre_mean_dim is not None else event_cube
    df = cube.count(dim=count_dim)
    return df.max(dim=max_dims) - nb_dim(cube, dims)

def calculate_mean(
        dataset: xr.Dataset, 
        dim: str = "time"
    ) -> np.ndarray:
    return dataset.mean(dim).values

def calculate_mean_anomaly(
        mean: np.ndarray,   
        climate_mean: np.ndarray, 
        climate_var: np.ndarray
    ) -> np.ndarray:
    return (mean - climate_mean) / np.sqrt(climate_var)

def calculate_observable(
        dataset: XArray,
        calc_months: tuple[int, int],
        spatial_box: IBox,
        xconfig: XConfig,
    ) -> xr.DataArray: 
    res = spatial_box.select(dataset, dims=xconfig.spatial_dims)
    res = filter_by_months(res, xconfig.time_dim, calc_months)
    return (
        res.mean(dim="points")
        .astype(np.float64)
    )

def calculate_quantiles(
        series_obs: xr.DataArray,
        quantiles: list[float],
        time_dim: str
    ) -> xr.DataArray:
    return series_obs.quantile(q=quantiles, dim=time_dim)

def calculate_rolling(series: xr.DataArray, time_dim: str, rol_days: int) -> xr.DataArray:
    return series.rolling(**{time_dim: rol_days, "center": True}).mean()

def calculate_var(
        dataset: xr.Dataset, 
        dim: str = "time", 
        scale: float = 100.0
    ) -> np.ndarray:
    return dataset.var(dim).values

