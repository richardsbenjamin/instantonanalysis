from __future__ import annotations

import csv
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
from instantonanalysis.latent.centroid import box_extract
from instantonanalysis.latent.metrics import classifier_auc, pc_separation
from instantonanalysis.latent.pca import plot_pca, run_incremental_pca, run_pca

if TYPE_CHECKING:
    from typing import Iterator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def open_da(path: str, var: str, xcfg: XConfigHealPix, box=None) -> xr.DataArray:
    da = xr.open_zarr(path)[var].isel(step=0)
    if xcfg.face_dim not in da.dims:
        # Pre-filtered region_latents_*.zarr: already reduced to a points_l{N} dim.
        pts = [d for d in da.dims if d not in (xcfg.time_dim, xcfg.channel_dim)]
        return da.transpose(xcfg.time_dim, *pts, xcfg.channel_dim)
    da = da.transpose(xcfg.time_dim, *xcfg.spatial_dims, xcfg.channel_dim)
    if box is not None:
        # Spatially filter to the event region: face/height/width -> points.
        da = box_extract(da, box, xcfg).transpose(
            xcfg.time_dim, "points", xcfg.channel_dim)
    return da


def _spatial_size(da: xr.DataArray, xcfg: XConfigHealPix) -> int:
    """Per-event sample count (number of grid points), robust to the box-filtered
    ``points`` layout and the full ``face/height/width`` layout."""
    return int(prod(s for d, s in da.sizes.items()
                    if d not in (xcfg.time_dim, xcfg.channel_dim)))


def iter_batches(
        da: xr.DataArray, xcfg: XConfigHealPix, batch_size: int
    ) -> Iterator[np.ndarray]:
    n_time = da.sizes[xcfg.time_dim]
    n_channels = da.sizes[xcfg.channel_dim]
    for start in range(0, n_time, batch_size):
        chunk = da.isel({xcfg.time_dim: slice(start, start + batch_size)})
        yield chunk.values.reshape(-1, n_channels)


def load_vectors(path: str, var: str, xcfg: XConfigHealPix, box=None) -> np.ndarray:
    da = open_da(path, var, xcfg, box=box)
    n_channels = da.sizes[xcfg.channel_dim]
    logger.info(f"  {path!r}: shape = {da.shape}")
    return da.values.reshape(-1, n_channels)


def load_auc_subsample(
        hw_path: str,
        no_hw_path: str,
        var: str,
        xcfg: XConfigHealPix,
        box=None,
        max_rows: int = 200_000,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Materialise a small balanced subsample of raw latent vectors for the
    classifier AUC, reading only a few events per class from the zarr.

    Used on the incremental (large-level) path so the AUC is computed on the
    full-dimensional latent vectors without loading the whole array.
    """
    rng = np.random.default_rng(seed)
    per_class = max_rows // 2
    parts, labels = [], []
    for path, label in [(hw_path, 1), (no_hw_path, 0)]:
        da = open_da(path, var, xcfg, box=box)
        n_channels = da.sizes[xcfg.channel_dim]
        n_time = da.sizes[xcfg.time_dim]
        spatial = _spatial_size(da, xcfg)
        n_keep = max(1, min(n_time, int(np.ceil(per_class / spatial))))
        keep = np.sort(rng.choice(n_time, size=n_keep, replace=False))
        sub = da.isel({xcfg.time_dim: keep}).values.reshape(-1, n_channels)
        if len(sub) > per_class:
            sub = sub[rng.choice(len(sub), size=per_class, replace=False)]
        parts.append(sub)
        labels.append(np.full(len(sub), label))
    return np.concatenate(parts, axis=0), np.concatenate(labels)


def append_metrics(metrics_path: Path, row: dict) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_path.exists()
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    logger.info("Loading config")
    args = get_calc_args()
    cfg = load_config(args.config_name, args.overrides)

    var = cfg.analysis.var
    batch_size = cfg.analysis.batch_size
    xcfg = instantiate(cfg.xconfig)
    box = instantiate(cfg.box) if "box" in cfg else None

    output_path = Path(cfg.paths.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if box is not None:
        logger.info(f"Spatial filter: {len(box.f_list)} points over faces {sorted(set(box.f_list))}")

    if batch_size is None:
        logger.info("Loading heatwave dataset")
        hw = load_vectors(cfg.paths.heatwave, var, xcfg, box=box)
        logger.info("Loading non-heatwave dataset")
        no_hw = load_vectors(cfg.paths.no_heatwave, var, xcfg, box=box)

        x = np.concatenate([hw, no_hw], axis=0)
        mask = np.concatenate([np.ones(len(hw)), np.zeros(len(no_hw))])
        logger.info(f"Combined shape: {x.shape}  ({len(hw)} heatwave, {len(no_hw)} non-heatwave)")

        logger.info("Running PCA")
        pca_res = run_pca(x, batch_size=None)
        x_auc, mask_auc = x, mask
    else:
        logger.info("Opening datasets lazily")
        hw_da = open_da(cfg.paths.heatwave, var, xcfg, box=box)
        no_hw_da = open_da(cfg.paths.no_heatwave, var, xcfg, box=box)

        spatial_size = _spatial_size(hw_da, xcfg)
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

        logger.info("Loading subsample of raw vectors for AUC")
        x_auc, mask_auc = load_auc_subsample(
            cfg.paths.heatwave, cfg.paths.no_heatwave, var, xcfg, box=box,
        )

    ev = pca_res["pca_model"].explained_variance_ratio_
    logger.info(f"PC1: {ev[0]*100:.2f}%  PC2: {ev[1]*100:.2f}%")
    logger.info(f"Saving plot to {str(output_path)!r}")
    plot_pca(pca_res, mask, str(output_path))

    logger.info("Computing differentiation metrics")
    auc = classifier_auc(x_auc, mask_auc)
    pc_sep = pc_separation(pca_res["pca_transform"], mask)
    logger.info(f"Classifier AUC: {auc:.4f}   PC separation: {pc_sep:.4f}")

    level = cfg.analysis.get("level", -1)
    metrics_path = Path(cfg.paths.get("metrics", "./outputs/pca/level_metrics.csv"))
    append_metrics(metrics_path, {
        "level": level,
        "n_samples": int(len(mask)),
        "ev_pc1": round(float(ev[0]), 6),
        "ev_pc2": round(float(ev[1]), 6),
        "auc": round(auc, 6),
        "pc_separation": round(pc_sep, 6),
    })
    logger.info(f"Appended metrics row to {str(metrics_path)!r}")
