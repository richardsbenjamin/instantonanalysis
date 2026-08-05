from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr
from hydra.utils import instantiate

from instantonanalysis.instanton.analysis import calculate_observable
from instantonanalysis.instanton.utils import load_config, read_dataset
from instantonanalysis.instanton.utils.parsers import get_calc_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Loading config")
    args = get_calc_args()
    cfg = load_config(args.config_name, args.overrides)

    data_root_path = Path(cfg.paths.data_root)
    spatial_box = instantiate(cfg.box)
    xconfig = instantiate(cfg.xconfig)
    output_dir = Path(cfg.paths.results_root + cfg.locations.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    var_cfg = cfg.variables["t2m"]

    logger.info("Loading dataset")
    dataset_in = read_dataset(data_root_path / cfg.paths.data_file)[var_cfg.name]

    logger.info("Calculating observable")
    series_obs = calculate_observable(
        dataset_in,
        calc_months=cfg.analysis.calc_months_init,
        spatial_box=spatial_box,
        xconfig=xconfig,
    ).squeeze()

    logger.info("Saving observable")
    combined_ds = xr.merge([series_obs.rename(var_cfg.name)]).reset_coords(drop=True)
    try:
        combined_ds.to_netcdf(output_dir / cfg.paths.series_obs)
    except PermissionError:
        logger.warning(f"Permission denied when saving {cfg.paths.series_obs}")

    dataset_in.close()
