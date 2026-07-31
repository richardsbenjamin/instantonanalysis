from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from instantonanalysis.instanton.schemas.xconfig import XConfigHealPix


def open_trajectory_da(
        path: str,
        var: str,
        xcfg: XConfigHealPix,
        max_hours: int | None = None,
        box=None,
    ) -> xr.DataArray:
    """Open a latent zarr keeping the full ``step`` (trajectory) dimension.

    Unlike the PCA stage (which does ``.isel(step=0)``), this keeps every
    forecast lead time so each event becomes a sequence of latent embeddings.
    If ``max_hours`` is given the trajectory is sliced to steps at or below that
    lead time, so the window matches ``rolling_period * 24`` hours.

    If ``box`` (a ``HealPixBox`` built at this level's nside) is given, the
    latents are spatially filtered to that region first — the three spatial dims
    collapse into a single ``points`` dim — so the analysis only sees pixels over
    the event region rather than the whole sphere.

    A pre-filtered ``region_latents_*.zarr`` (no face dim, a single per-level
    ``points_l{N}`` dim from ``materialize_region_latents.py``) is detected and
    returned as-is spatially — ``box`` is not applied again.
    """
    da = xr.open_zarr(path)[var]
    if max_hours is not None:
        steps = da["step"]
        da = da.sel(step=steps[steps <= np.timedelta64(int(max_hours), "h")])
    if xcfg.face_dim not in da.dims:
        pts = [d for d in da.dims if d not in (xcfg.time_dim, "step", xcfg.channel_dim)]
        return da.transpose(xcfg.time_dim, "step", *pts, xcfg.channel_dim)
    da = da.transpose(xcfg.time_dim, "step", *xcfg.spatial_dims, xcfg.channel_dim)
    if box is not None:
        da = box_extract(da, box, xcfg)
        da = da.transpose(xcfg.time_dim, "step", "points", xcfg.channel_dim)
    return da


def box_extract(da: xr.DataArray, box, xcfg: XConfigHealPix) -> xr.DataArray:
    """Memory-safe spatial filtering of a (possibly dask-backed) latent array.

    Pointwise ``.sel`` over (face, height, width) on a large dask array blows up
    memory (it materialises the whole array). The box only touches a couple of
    HEALPix faces, so we first slice to those faces (by label) and load that much
    smaller subset into memory, then do the exact point selection in-memory.
    """
    faces = sorted(set(box.f_list))
    da = da.sel({xcfg.face_dim: faces})
    # Synchronous scheduler loads one chunk at a time; the l0 latent zarr has
    # ~0.57 GB chunks (all steps/channels/faces per time), so the default
    # threaded scheduler reads ~8 at once and OOMs. One-at-a-time stays small.
    da = da.compute(scheduler="synchronous")
    return box.extract(da, xconfig=xcfg)


def _feature_dims(da: xr.DataArray, xcfg: XConfigHealPix) -> list[str]:
    """Dims that make up a single sample's feature vector (everything that is
    neither the event axis nor the trajectory axis). Works for both the boxed
    ``points`` layout and the full ``face/height/width`` layout."""
    return [d for d in da.dims if d not in (xcfg.time_dim, "step")]


def _n_features(da: xr.DataArray, xcfg: XConfigHealPix) -> int:
    return int(prod(da.sizes[d] for d in _feature_dims(da, xcfg)))


def event_centroids(
        da: xr.DataArray, xcfg: XConfigHealPix
    ) -> tuple[np.ndarray, np.ndarray]:
    """Mean latent vector over each event's trajectory.

    Returns ``(centroids[n_events, n_features], event_times[n_events])``. The
    feature axis is the flattened ``(*spatial, channel)`` block, ordered
    consistently with :func:`climatological_stats`.
    """
    mean = da.mean(dim="step")
    n_events = da.sizes[xcfg.time_dim]
    centroids = mean.values.reshape(n_events, _n_features(da, xcfg))
    return centroids, da[xcfg.time_dim].values


def climatological_stats(
        da: xr.DataArray, xcfg: XConfigHealPix, time_batch: int = 4
    ) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and std over all ``(event, step)`` embeddings.

    Streams over the time dimension (accumulating sum / sum-of-squares) so the
    full latent matrix is never materialised — safe for the large l0 level as
    well as l1/l2. Returns ``(clim_mean[n_features], clim_std[n_features])``.
    """
    n_features = _n_features(da, xcfg)
    n_time = da.sizes[xcfg.time_dim]
    s = np.zeros(n_features, dtype=np.float64)
    ss = np.zeros(n_features, dtype=np.float64)
    count = 0
    for start in range(0, n_time, time_batch):
        chunk = da.isel({xcfg.time_dim: slice(start, start + time_batch)})
        flat = chunk.values.reshape(-1, n_features).astype(np.float64)
        s += flat.sum(axis=0)
        ss += (flat ** 2).sum(axis=0)
        count += flat.shape[0]
    mean = s / count
    var = np.maximum(ss / count - mean ** 2, 0.0)
    return mean, np.sqrt(var)


def centroid_distances(
        centroids: np.ndarray,
        clim_mean: np.ndarray,
        clim_std: np.ndarray | None = None,
        eps: float = 1e-8,
    ) -> np.ndarray:
    """L2 distance from each centroid to the climatological mean.

    If ``clim_std`` is given the difference is standardised per dimension first,
    giving a climatology-normalised "latent anomaly" — the analogue of a
    normalised temperature anomaly. Returns ``dist[n_events]``.
    """
    diff = centroids - clim_mean
    if clim_std is not None:
        diff = diff / (clim_std + eps)
    return np.linalg.norm(diff, axis=1)


def loo_reference_distances(
        ref_centroids: np.ndarray,
        clim_std: np.ndarray | None = None,
        eps: float = 1e-8,
    ) -> np.ndarray:
    """Leave-one-out centroid distances for the reference (non-heatwave) class.

    When every event shares the same trajectory length the climatological mean
    of all embeddings equals the mean of the event centroids, so a leave-one-out
    climatological mean is just the mean of the *other* centroids. Removing each
    event from its own reference avoids the self-inclusion shrinkage that would
    otherwise bias the null distribution low. (The std scale is left at the full
    climatological value; one event's effect on it is negligible.)
    """
    n = len(ref_centroids)
    total = ref_centroids.sum(axis=0)
    loo_means = (total - ref_centroids) / (n - 1)
    diff = ref_centroids - loo_means
    if clim_std is not None:
        diff = diff / (clim_std + eps)
    return np.linalg.norm(diff, axis=1)


def cosine_distances(centroids: np.ndarray, clim_mean: np.ndarray) -> np.ndarray:
    """Cosine distance (1 - cosine similarity) to the climatological mean.

    A direction-sensitive companion to the L2 anomaly; in high dimensions the
    orientation of the centroid can carry more signal than its magnitude.
    """
    ref = clim_mean / (np.linalg.norm(clim_mean) + 1e-12)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    cos = (centroids / norms) @ ref
    return 1.0 - cos


def null_significance(
        event_dist: np.ndarray, ref_dist: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """Significance of each event distance against the reference distribution.

    Returns ``(z_score, percentile)`` where the z-score uses the reference
    mean/std and the percentile is the fraction of reference distances below
    each event distance. Mirrors the chi-square significance idea used in the
    physical-space pipeline.
    """
    mu = ref_dist.mean()
    sigma = ref_dist.std(ddof=1)
    z = (event_dist - mu) / sigma if sigma > 0 else np.full_like(event_dist, np.nan)
    pct = (ref_dist[None, :] < event_dist[:, None]).mean(axis=1)
    return z, pct
