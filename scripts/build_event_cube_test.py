import logging
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr
from hydra.utils import instantiate

from instantonanalysis.instanton.analysis import (
    calculate_chi_mask,
    calculate_observable,
    calculate_weighted_spatial_mean,
)
import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
from instantonanalysis.instanton.nbclosest import NClosestCalc, get_nclosest_config
from instantonanalysis.instanton.utils import (
    build_event_cube,
    get_distance_function,
    get_df_array,
    load_config,
    read_dataset,
)
from instantonanalysis.utils.parsers import get_calc_args

from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from instantonanalysis.instanton.schemas import Config
from hydra.utils import instantiate
from omegaconf import DictConfig
import instantonanalysis.hydra_logic

if TYPE_CHECKING:
    from instantonanalysis.instanton.nbclosest import NClosestConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# from typing import List, Optional
# def load_config(
#         config_name: str = "config", 
#         config_path: str = "../config",
#         overrides: Optional[List[str]] = None,
#         schema_node: Optional = None,
#     ):
#     if overrides is None:
#         overrides = sys.argv[1:]
#     if schema_node is not None:
#         cs = ConfigStore.instance()
#         cs.store(name=config_name, node=schema_node)
#     with initialize(version_base=None, config_path=config_path):
#         cfg = compose(config_name=config_name, overrides=overrides)
#         return cfg


import xarray as xr
import numpy as np
import pandas as pd

def build_event_cube2(
    data: xr.Dataset, 
    var_cfg, 
    neighbors_da: xr.Dataset, 
    xconfig,
) -> xr.Dataset:
    time_dim = xconfig.time_dim
    lag_dim = xconfig.lag
    event_dim = xconfig.event
    rolling_dim = xconfig.rolling_period
    quantile_dim = xconfig.quantile

    # 1. Pre-calculate Global Lags
    max_r = int(neighbors_da[rolling_dim].max())
    global_lags = np.arange(-max_r // 2 + 1, max_r // 2 + 1)
    
    # 2. Vectorized Indexing: Identify event dates
    # We stack to create a flat dimension across R, Q, and Time
    stacked_events = neighbors_da.stack(event_id=(rolling_dim, quantile_dim, time_dim))
    
    # Lazy masking: keep only the coordinates where events actually exist
    valid_events = stacked_events.where(stacked_events.notnull(), drop=True)
    event_dates = valid_events[time_dim]

    # 3. Handle cftime Arithmetic
    # Create the lag offsets as an array of timedeltas compatible with cftime
    # We use a list comprehension or map to ensure compatibility across types
    lag_tds = np.array([pd.Timedelta(days=int(l)) for l in global_lags])

    
    lag_offsets = xr.DataArray(
        lag_tds, 
        coords={lag_dim: global_lags}, 
        dims=[lag_dim]
    )
    
    # Vectorized addition: [event_id] + [lag] -> [event_id, lag]
    # This now works with cftime objects because of the pd.Timedelta conversion
    target_times = event_dates + lag_offsets

    # 4. Optimized Selection (One-shot)
    # We use method='nearest' if your calendar has small alignment drifts, 
    # otherwise method=None is stricter and faster.
    cube = data.sel({time_dim: target_times})
    
    # 5. Apply Transformations (Broadcasting)
    cube = cube * var_cfg.scale_factor + var_cfg.offset
    if var_cfg.squeeze:
        cube = cube.squeeze(var_cfg.squeeze)

    # 6. Apply Lag Masking (Maintain Dask Laziness)
    r_for_event = valid_events[rolling_dim]
    half_r = r_for_event // 2
    
    # Define bounds for each event's specific R-window
    lag_indices = xr.DataArray(global_lags, coords={lag_dim: global_lags}, dims=[lag_dim])
    lag_mask = (lag_indices >= -half_r + 1) & (lag_indices <= half_r)
    
    # Apply mask to the cube
    cube = cube.where(lag_mask)

    # 7. Final Reshape and Cleanup
    # Unstack 'event_id' to recover the (rolling_period, quantile) hierarchy
    cube = cube.unstack("event_id")
    
    # Rename the original time dimension to the abstract 'event' dimension
    # and reorder to your preferred orientation
    return cube.rename({time_dim: event_dim}).transpose(
        rolling_dim, quantile_dim, lag_dim,
        event_dim, *xconfig.spatial_dims,
    )


def get_chunks_spec(xconfig: XConfig) -> dict[str, int]:
    chunks_spec = {xconfig.time_dim: -1} 
    for dim in xconfig.spatial_dims:
        chunks_spec[dim] = 'auto'
    return chunks_spec


if __name__ == "__main__":

    logger.info("Loading config")
    config_name = "calc_config_healpix"
    overrides = []
    cfg = load_config(config_name, overrides)

    data_root_path = Path(cfg.paths.data_root)
    na_box = instantiate(cfg.na_box)
    spatial_box = instantiate(cfg.box)
    output_dir = Path(cfg.paths.results_root + cfg.locations.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    vars_list = [var.name for var in cfg.variables.values()]
    xconfig = instantiate(cfg.xconfig)

    results_obs = []
    results_ac = []
    results_qa = []
    results_nb = []
    results_comp_anom = []
    results_norm_tilde = []
    results_weight_tilde = []
    results_norm_hat = []

    logger.info("Loading datasets")
    datasets = {}
    climate_mean_in = read_dataset(data_root_path / f"climate_mean_{cfg.paths.data_file}")[vars_list]
    climate_var_in = read_dataset(data_root_path / f"climate_variance_{cfg.paths.data_file}")[vars_list]
    dataset_in = read_dataset(data_root_path / cfg.paths.data_file)[vars_list]
    for var_cfg in cfg.variables.values():
        datasets[var_cfg.name] = {
            "dataset": dataset_in[var_cfg.name].squeeze(),
            "climate_mean": climate_mean_in[var_cfg.name].squeeze(),
            "climate_var": climate_var_in[var_cfg.name].squeeze(),
        }

    logger.info("Calculating observable")
    var_cfg = cfg.variables["t2m"]

    series_obs = calculate_observable(
        datasets[var_cfg.name]["dataset"],
        var=var_cfg.name,
        calc_months=cfg.analysis.calc_months_init,
        spatial_box=spatial_box,
        xconfig=xconfig,
    )

    logger.info("Calculating nclosest")
    nclosest_config = get_nclosest_config(
        analysis_cfg=cfg.analysis,
        time_dim=xconfig.time_dim,
    )
    nclosest_calc = NClosestCalc(
        series_obs=series_obs,
        config=nclosest_config,
        xconfig=xconfig,
        dist_func=get_distance_function(cfg.analysis.dist_func),
    )
    nclosest_calc.calculate()

    results_obs.append(series_obs.rename(var_cfg.name))
    results_ac.append(nclosest_calc.results_ac.rename(var_cfg.name))
    results_qa.append(nclosest_calc.results_qa.rename(var_cfg.name))
    results_nb.append(nclosest_calc.results_nb.rename(var_cfg.name))

    for var_cfg in cfg.variables.values():
        logger.info("Building event cube")
        dataset = datasets[var_cfg.name]["dataset"] #.chunk(CHUNKS)
        with dask.config.set({"array.chunk-size": "1GiB"}):
            data_chunked = data.chunk(chunks_spec)

        # 6D cube: rolling_period, quantile, lag, event, lat, lon
        event_cube = build_event_cube(
            dataset, var_cfg, nclosest_calc.results_nb, xconfig,
        )

        event_cube2 = build_event_cube2(
            dataset, var_cfg, nclosest_calc.results_nb, xconfig,
        )