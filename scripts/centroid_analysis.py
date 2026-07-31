from __future__ import annotations

import csv
import logging
from math import prod
from pathlib import Path

from hydra.utils import instantiate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, silhouette_score

from instantonanalysis.instanton.schemas.xconfig import XConfigHealPix
from instantonanalysis.instanton.utils import load_config
from instantonanalysis.instanton.utils.parsers import get_calc_args
from instantonanalysis.latent.centroid import (
    box_extract,
    centroid_distances,
    cosine_distances,
    event_centroids,
    null_significance,
    open_trajectory_da,
)
from instantonanalysis.latent.metrics import pc_separation
from instantonanalysis.latent.pca import run_pca

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def append_metrics(metrics_path: Path, row: dict) -> None:
    """Append one summary row to the metrics CSV (header written on first use)."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_path.exists()
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_clim_stats(
        clim_path: str, level: int, template: xr.DataArray, xcfg: XConfigHealPix,
    ) -> tuple[np.ndarray, np.ndarray, int]:
    """Flattened climatological mean/std feature vectors from the sampled JJA
    climatology (``clim_stats.nc``, produced by ``materialize_region_latents.py``).

    Flattening must match :func:`event_centroids` (points slowest, channel
    fastest), and the climatology's point ordering must match the event zarr's —
    both come from the same box extraction, which is asserted via the pointwise
    height coordinate.
    """
    ds = xr.open_dataset(clim_path)
    pts, ch = f"points_l{level}", xcfg.channel_dim
    mean_da = ds[f"clim_mean_l{level}"].transpose(pts, ch)
    var_da = ds[f"clim_var_l{level}"].transpose(pts, ch)
    h = f"height_l{level}"
    if h in ds.coords and h in template.coords:
        assert np.array_equal(ds[h].values, template[h].values), \
            "clim_stats point ordering differs from the event zarr's"
    n = int(ds.attrs.get("n_samples", -1))
    return mean_da.values.ravel(), np.sqrt(var_da.values).ravel(), n


def dataset_a_event_scores(
        hw_path: str, no_hw_path: str, var: str, xcfg: XConfigHealPix, box=None,
    ) -> dict[str, np.ndarray]:
    """Per-event outlier score from the Dataset-A per-pixel PCA (step=0).

    Reproduces the existing PCA (per-pixel latent vectors at the first lead
    time), then aggregates the PC1/PC2 coordinates per event (mean over that
    event's pixels). The outlier score is the distance of each event's mean PC
    position from the non-heatwave mean PC position. The same ``box`` used for
    Dataset B is applied so the cross-check compares like with like.
    """
    def load_pixels(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        da = xr.open_zarr(path)[var].isel(step=0)
        if xcfg.face_dim not in da.dims:
            # Pre-filtered region_latents_*.zarr: already reduced to points_l{N}.
            pts = [d for d in da.dims if d not in (xcfg.time_dim, xcfg.channel_dim)]
            da = da.transpose(xcfg.time_dim, *pts, xcfg.channel_dim)
        else:
            da = da.transpose(xcfg.time_dim, *xcfg.spatial_dims, xcfg.channel_dim)
            if box is not None:
                da = box_extract(da, box, xcfg).transpose(
                    xcfg.time_dim, "points", xcfg.channel_dim)
        n_time = da.sizes[xcfg.time_dim]
        n_ch = da.sizes[xcfg.channel_dim]
        n_spatial = prod(s for d, s in da.sizes.items() if d not in (xcfg.time_dim, xcfg.channel_dim))
        x = da.values.reshape(-1, n_ch)
        event_idx = np.repeat(np.arange(n_time), n_spatial)
        return x, event_idx, da[xcfg.time_dim].values

    hw_x, hw_ev, hw_times = load_pixels(hw_path)
    no_x, no_ev, _ = load_pixels(no_hw_path)

    x = np.concatenate([hw_x, no_x], axis=0)
    transform = run_pca(x, batch_size=None)["pca_transform"]
    hw_t = transform[: len(hw_x)]
    no_t = transform[len(hw_x):]

    def event_means(t: np.ndarray, ev: np.ndarray) -> np.ndarray:
        return np.stack([t[ev == e].mean(axis=0) for e in np.unique(ev)])

    hw_pc = event_means(hw_t, hw_ev)
    no_pc = event_means(no_t, no_ev)
    ref = no_pc.mean(axis=0)
    return {
        "hw_score": np.linalg.norm(hw_pc - ref, axis=1),
        "hw_times": hw_times,
    }


def plot_centroid_pca(transform, mask, ev, save_path) -> None:
    plt.figure(figsize=(8, 6))
    for label, color, name in [(1, "tab:red", "Heatwave"), (0, "tab:blue", "Non-heatwave")]:
        sel = mask == label
        plt.scatter(transform[sel, 0], transform[sel, 1], c=color, label=name,
                    alpha=0.7, s=40, edgecolors="k", linewidths=0.3)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Event centroids in PCA space")
    plt.legend()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_anomaly_timeseries(hw_times, hw_dist, no_times, no_dist, save_path) -> None:
    plt.figure(figsize=(11, 5))
    plt.scatter(pd.to_datetime(no_times), no_dist, c="tab:blue", s=30,
                label="Non-heatwave", alpha=0.7)
    plt.scatter(pd.to_datetime(hw_times), hw_dist, c="tab:red", s=45,
                label="Heatwave", alpha=0.8, edgecolors="k", linewidths=0.3)
    plt.axhline(no_dist.mean(), color="tab:blue", ls="--", lw=1,
                label="Non-heatwave mean")
    plt.xlabel("Event date")
    plt.ylabel("Standardised latent anomaly (L2 to clim mean)")
    plt.title("Latent anomaly through events")
    plt.legend()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_crosscheck(a_score, b_dist, r_p, r_s, save_path) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(a_score, b_dist, c="tab:red", s=45, alpha=0.8,
                edgecolors="k", linewidths=0.3)
    plt.xlabel("Dataset A: per-event PCA outlier score")
    plt.ylabel("Dataset B: standardised centroid distance")
    plt.title(f"Cross-check (Pearson r={r_p:.2f}, Spearman r={r_s:.2f})")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    logger.info("Loading config")
    args = get_calc_args()
    cfg = load_config(args.config_name, args.overrides)

    var = cfg.analysis.var
    rolling_period = int(cfg.analysis.rolling_period)
    max_hours = rolling_period * 24
    level = cfg.analysis.get("level", -1)
    xcfg = instantiate(cfg.xconfig)
    box = instantiate(cfg.box) if "box" in cfg else None

    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if box is not None:
        logger.info(f"Spatial filter: {len(box.f_list)} points over faces {sorted(set(box.f_list))}")

    logger.info(f"Opening trajectories (window = {max_hours}h)")
    hw_da = open_trajectory_da(cfg.paths.heatwave, var, xcfg, max_hours, box=box)
    no_hw_da = open_trajectory_da(cfg.paths.no_heatwave, var, xcfg, max_hours, box=box)
    logger.info(
        f"Heatwave: {hw_da.sizes[xcfg.time_dim]} events x "
        f"{hw_da.sizes['step']} steps; non-heatwave: "
        f"{no_hw_da.sizes[xcfg.time_dim]} events x {no_hw_da.sizes['step']} steps"
    )

    logger.info("Loading sampled JJA climatological mean/std from clim_stats.nc")
    clim_mean, clim_std, clim_n = load_clim_stats(
        cfg.paths.clim_stats, level, hw_da, xcfg)
    logger.info(f"Climatology: n_samples={clim_n} (pooled over trajectory steps)")

    logger.info("Computing event centroids")
    hw_centroids, hw_times = event_centroids(hw_da, xcfg)
    no_centroids, no_times = event_centroids(no_hw_da, xcfg)

    # latent anomaly
    hw_raw = centroid_distances(hw_centroids, clim_mean)
    hw_std = centroid_distances(hw_centroids, clim_mean, clim_std)
    hw_cos = cosine_distances(hw_centroids, clim_mean)

    # Non-heatwave null distances against the same external climatology (the
    # events are not part of the climatology sample, so no leave-one-out is
    # needed — the old LOO-vs-own-mean baseline is retired).
    no_raw = centroid_distances(no_centroids, clim_mean)
    no_std = centroid_distances(no_centroids, clim_mean, clim_std)
    no_cos = cosine_distances(no_centroids, clim_mean)

    z_score, percentile = null_significance(hw_std, no_std)
    logger.info(
        f"Heatwave latent anomaly (std): mean={hw_std.mean():.3f} "
        f"vs non-heatwave ref mean={no_std.mean():.3f}; "
        f"mean z-score={np.nanmean(z_score):.3f}"
    )

    # --- Centroid clustering ---
    logger.info("PCA of event centroids")
    x_cent = np.concatenate([hw_centroids, no_centroids], axis=0)
    mask = np.concatenate([np.ones(len(hw_centroids)), np.zeros(len(no_centroids))])
    cent_transform = run_pca(x_cent, batch_size=None)["pca_transform"]
    pc_sep = pc_separation(cent_transform, mask)
    sil = float(silhouette_score(cent_transform, mask))
    # Separability of the scalar latent anomaly itself.
    all_std = np.concatenate([hw_std, no_std])
    auc_anomaly = float(roc_auc_score(mask, all_std))
    logger.info(
        f"Centroid clustering: pc_separation={pc_sep:.3f}, silhouette={sil:.3f}, "
        f"anomaly AUC={auc_anomaly:.3f}"
    )

    plot_centroid_pca(cent_transform, mask, None, str(out_dir / f"centroid_pca_l{level}.png"))
    plot_anomaly_timeseries(
        hw_times, hw_std, no_times, no_std,
        str(out_dir / f"latent_anomaly_timeseries_l{level}.png"),
    )

    # --- Cross-check vs Dataset A ---
    logger.info("Computing Dataset-A per-event outlier scores (step=0 PCA)")
    a = dataset_a_event_scores(cfg.paths.heatwave, cfg.paths.no_heatwave, var, xcfg, box=box)
    # Both score arrays come from the same zarr's `time` coordinate in the same
    # order, so they are already aligned event-for-event.
    assert np.array_equal(a["hw_times"], hw_times), "Dataset A/B event order mismatch"
    a_aligned = a["hw_score"]
    r_p = float(pearsonr(a_aligned, hw_std)[0])
    r_s = float(spearmanr(a_aligned, hw_std)[0])
    logger.info(f"Cross-check Pearson r={r_p:.3f}, Spearman r={r_s:.3f}")
    plot_crosscheck(a_aligned, hw_std, r_p, r_s, str(out_dir / f"crosscheck_l{level}.png"))

    # --- Persist per-event results ---
    rows = []
    for t, dr, ds, dc, z, p, a_s in zip(
        hw_times, hw_raw, hw_std, hw_cos, z_score, percentile, a_aligned
    ):
        rows.append({"event_date": str(pd.Timestamp(t)), "class": "heatwave",
                     "dist_raw": dr, "dist_std": ds, "dist_cos": dc,
                     "z_score": z, "percentile": p, "a_outlier_score": a_s})
    for t, dr, ds, dc in zip(no_times, no_raw, no_std, no_cos):
        rows.append({"event_date": str(pd.Timestamp(t)), "class": "non_heatwave",
                     "dist_raw": dr, "dist_std": ds, "dist_cos": dc,
                     "z_score": np.nan, "percentile": np.nan, "a_outlier_score": np.nan})
    per_event_path = out_dir / f"per_event_l{level}.csv"
    pd.DataFrame(rows).to_csv(per_event_path, index=False)
    logger.info(f"Wrote per-event results to {str(per_event_path)!r}")

    metrics_path = Path(cfg.paths.get("metrics", str(out_dir / "centroid_metrics.csv")))
    append_metrics(metrics_path, {
        "level": level,
        "clim_n_samples": clim_n,
        "rolling_period": rolling_period,
        "window_hours": max_hours,
        "n_steps": int(hw_da.sizes["step"]),
        "hw_anomaly_mean": round(float(hw_std.mean()), 6),
        "ref_anomaly_mean": round(float(no_std.mean()), 6),
        "mean_z": round(float(np.nanmean(z_score)), 6),
        "pc_separation": round(pc_sep, 6),
        "silhouette": round(sil, 6),
        "anomaly_auc": round(auc_anomaly, 6),
        "crosscheck_pearson": round(r_p, 6),
        "crosscheck_spearman": round(r_s, 6),
    })
    logger.info(f"Appended metrics row to {str(metrics_path)!r}")
