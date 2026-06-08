from __future__ import annotations

import itertools
import logging
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

from hydra.utils import instantiate
import numpy as np
import xarray as xr

from instantonanalysis.instanton.schemas.xconfig import XConfigHealPix
from instantonanalysis.instanton.utils import load_config
from instantonanalysis.instanton.utils.parsers import get_calc_args
from instantonanalysis.latent.pca import plot_pca, run_incremental_pca, run_pca

if TYPE_CHECKING:
    from typing import Iterator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def open_da(path: str, var: str, xcfg: XConfigHealPix) -> xr.DataArray:
    da = xr.open_zarr(path)[var].isel(step=0)
    sample_dims = (xcfg.time_dim, *xcfg.spatial_dims)
    return da.transpose(*sample_dims, xcfg.channel_dim)


def iter_batches(
    da: xr.DataArray, xcfg: XConfigHealPix, batch_size: int
) -> Iterator[np.ndarray]:
    n_time = da.sizes[xcfg.time_dim]
    n_channels = da.sizes[xcfg.channel_dim]
    for start in range(0, n_time, batch_size):
        chunk = da.isel({xcfg.time_dim: slice(start, start + batch_size)})
        yield chunk.values.reshape(-1, n_channels)


def load_vectors(path: str, var: str, xcfg: XConfigHealPix) -> np.ndarray:
    da = open_da(path, var, xcfg)
    n_channels = da.sizes[xcfg.channel_dim]
    logger.info(f"  {path!r}: shape = {da.shape}")
    return da.values.reshape(-1, n_channels)


if __name__ == "__main__":
    logger.info("Loading config")
    args = get_calc_args()
    cfg = load_config(args.config_name, args.overrides)

    var = cfg.analysis.var
    batch_size = cfg.analysis.batch_size
    xcfg = instantiate(cfg.xconfig)

    output_path = Path(cfg.paths.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if batch_size is None:
        logger.info("Loading heatwave dataset")
        hw = load_vectors(cfg.paths.heatwave, var, xcfg)
        logger.info("Loading non-heatwave dataset")
        no_hw = load_vectors(cfg.paths.no_heatwave, var, xcfg)

        x = np.concatenate([hw, no_hw], axis=0)
        mask = np.concatenate([np.ones(len(hw)), np.zeros(len(no_hw))])
        logger.info(f"Combined shape: {x.shape}  ({len(hw)} heatwave, {len(no_hw)} non-heatwave)")

        logger.info("Running PCA")
        pca_res = run_pca(x, batch_size=None)
    else:
        logger.info("Opening datasets lazily")
        hw_da = open_da(cfg.paths.heatwave, var, xcfg)
        no_hw_da = open_da(cfg.paths.no_heatwave, var, xcfg)

        spatial_size = prod(hw_da.sizes[d] for d in xcfg.spatial_dims)
        n_hw_time = hw_da.sizes[xcfg.time_dim]
        n_no_hw_time = no_hw_da.sizes[xcfg.time_dim]
        mask = np.concatenate([
            np.ones(n_hw_time * spatial_size),
            np.zeros(n_no_hw_time * spatial_size),
        ])
        logger.info(
            f"Sizes: {n_hw_time} heatwave steps, {n_no_hw_time} non-heatwave steps, "
            f"spatial_size={spatial_size}"
        )

        make_batches = lambda: itertools.chain(
            iter_batches(hw_da, xcfg, batch_size),
            iter_batches(no_hw_da, xcfg, batch_size),
        )

        logger.info(f"Running incremental PCA (batch_size={batch_size} time steps)")
        pca_res = run_incremental_pca(make_batches)

    ev = pca_res["pca_model"].explained_variance_ratio_
    logger.info(f"PC1: {ev[0]*100:.2f}%  PC2: {ev[1]*100:.2f}%")
    logger.info(f"Saving plot to {str(output_path)!r}")
    plot_pca(pca_res, mask, str(output_path))
