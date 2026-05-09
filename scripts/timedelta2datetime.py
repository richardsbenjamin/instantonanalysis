from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr
from hydra.utils import instantiate
from instantonanalysis.instanton.utils import (
    load_config,
    read_dataset,
)
from instantonanalysis.utils.parsers import get_calc_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_chunks_spec(xconfig: XConfig) -> dict[str, int]:
    chunks_spec = {xconfig.time_dim: -1} 
    for dim in xconfig.spatial_dims:
        chunks_spec[dim] = 'auto'
    return chunks_spec


if __name__ == "__main__":
    logger.info("Loading config")
    args = get_calc_args()
    cfg = load_config(args.config_name, args.overrides)
    
    data_root_path = Path(cfg.paths.data_root)
    xconfig = instantiate(cfg.xconfig)

    logger.info("Loading datasets")
    dataset_in = read_dataset(data_root_path / cfg.paths.data_file)
    
    try:
        _ = dataset_in[f"{xconfig.time_dim}.month"]
        print("Computed month!")
    except AttributeError:
        raise ValueError(f"Failed to compute month from {xconfig.time_dim}")
        

