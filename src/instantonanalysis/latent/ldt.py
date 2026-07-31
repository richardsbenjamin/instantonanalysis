"""Basis-independent normalised variance in latent space.

The physical-space result this mirrors is ``norm_var_hat`` collapsing toward zero over
Western Europe: extreme events follow similar trajectories, so the event ensemble is
narrow where the event lives. Asking the same question of encoder latents cannot be done
per channel — a latent channel is an arbitrary coordinate, and under a reparametrisation
``z -> A z`` every per-channel variance changes while the system is unchanged.

The quantities that survive a change of latent basis are the eigenvalues of the
generalised problem ``Sigma_event v = lambda Sigma_clim v``: those eigenvalues *are* the
normalised variances, one per fluctuation mode. Everything here is built around solving
that problem stably (whitening + truncation of the climatological covariance) and around
calibrating it — a raw lambda from 50 events is badly biased low, so a null drawn from
the climatology at the same sample size is not optional.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from sklearn.covariance import LedoitWolf

from instantonanalysis.latent.centroid import box_extract

if TYPE_CHECKING:
    from instantonanalysis.instanton.schemas.box import HealPixBox
    from instantonanalysis.instanton.schemas.xconfig import XConfigHealPix


def _points_dim(da: xr.DataArray, xcfg: XConfigHealPix) -> list[str]:
    return [d for d in da.dims if d not in (xcfg.time_dim, "step", xcfg.channel_dim)]


def open_event_cube(
        path: str,
        var: str,
        xcfg: XConfigHealPix,
        window_hours: int | None = None,
        box: HealPixBox | None = None,
    ) -> xr.DataArray:
    da = xr.open_zarr(path, decode_timedelta=True)[var]
    if window_hours is not None:
        steps = da["step"]
        da = da.sel(step=steps[steps <= np.timedelta64(int(window_hours), "h")])
    if box is not None:
        da = da.transpose(xcfg.time_dim, "step", *xcfg.spatial_dims, xcfg.channel_dim)
        da = box_extract(da, box, xcfg)
    pts = _points_dim(da, xcfg)
    return da.transpose(xcfg.time_dim, "step", *pts, xcfg.channel_dim)


def region_mean(da: xr.DataArray, xcfg: XConfigHealPix) -> xr.DataArray:
    """Mean over the in-box ``points`` dim, keeping event/lag/channel."""
    return da.mean(dim=_points_dim(da, xcfg))


# --------------------------------------------------------------------------- #
# Sample construction
# --------------------------------------------------------------------------- #
def trajectory_vectors(
        path: str,
        var: str,
        xcfg: XConfigHealPix,
        keep_points: bool = False,
        time_chunk: int = 50,
        window_hours: int | None = None,
        sample: str = "trajectory_mean",
    ) -> tuple[np.ndarray, np.ndarray]:
    if sample not in ("trajectory_mean", "pooled"):
        raise ValueError(f"unknown sample construction {sample!r}")

    da = xr.open_zarr(path, decode_timedelta=True)[var]
    if window_hours is not None:
        steps = da["step"]
        da = da.sel(step=steps[steps <= np.timedelta64(int(window_hours), "h")])
    pts = _points_dim(da, xcfg)
    da = da.transpose(xcfg.time_dim, "step", *pts, xcfg.channel_dim)

    n_time = da.sizes[xcfg.time_dim]
    parts = []
    for start in range(0, n_time, time_chunk):
        block = da.isel({xcfg.time_dim: slice(start, start + time_chunk)})
        block = block.values.astype(np.float64)          # (t, step, *points, channel)
        if sample == "trajectory_mean":
            block = block.mean(axis=1)                   # (t, *points, channel)
        else:
            block = block.reshape(-1, *block.shape[2:])  # (t*step, *points, channel)
        parts.append(block)
    x = np.concatenate(parts, axis=0)

    if keep_points:
        x = x.reshape(x.shape[0], -1, x.shape[-1])
    else:
        x = x.mean(axis=tuple(range(1, x.ndim - 1)))
    return x, da[xcfg.channel_dim].values


def region_point_coords(path: str, level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(face, height, width)`` of each in-box point of a region-filtered zarr."""
    ds = xr.open_zarr(path, decode_timedelta=True)
    return (ds[f"face_l{level}"].values,
            ds[f"height_l{level}"].values,
            ds[f"width_l{level}"].values)


# --------------------------------------------------------------------------- #
# The generalised eigenproblem
# --------------------------------------------------------------------------- #
def sample_covariance(x: np.ndarray, shrinkage: str | None = None) -> np.ndarray:
    """Covariance of ``x[n_samples, n_dim]``.

    ``shrinkage="ledoit_wolf"`` swaps the plain sample covariance for the Ledoit-Wolf
    estimator, rescaled from its 1/n normalisation to the 1/(n-1) one used everywhere
    else so the two are directly comparable. Note that LW shrinks toward ``mu I``, which
    is itself basis-dependent — it is a robustness variant, not the headline.
    """
    if shrinkage in (None, "none", "None"):
        return np.cov(x, rowvar=False)
    if shrinkage == "ledoit_wolf":
        n = x.shape[0]
        return LedoitWolf(assume_centered=False).fit(x).covariance_ * n / (n - 1)
    raise ValueError(f"unknown shrinkage {shrinkage!r}")


def whitened_generalised_eigh(
        sigma_event: np.ndarray, sigma_clim: np.ndarray, k: int
    ) -> dict:
    lam, u = np.linalg.eigh(sigma_clim)
    lam, u = lam[::-1], u[:, ::-1]                       # descending
    k = int(min(k, np.sum(lam > lam[0] * 1e-12)))
    lam_k, u_k = lam[:k], u[:, :k]

    omega = (u_k / np.sqrt(lam_k)).T                     # (k, d)
    m = omega @ sigma_event @ omega.T
    m = 0.5 * (m + m.T)
    mu, u_m = np.linalg.eigh(m)                          # ascending
    return {
        "eigenvalues": mu,                               # ascending normalised variances
        "eigenvectors": omega.T @ u_m,                   # (d, k), Sigma_clim-orthonormal
        "whitener": omega,
        "k": k,
        "clim_explained": float(lam_k.sum() / lam.sum()),
        "clim_spectrum": lam,
    }


def normalised_variance_spectrum(
        x_event: np.ndarray,
        x_clim: np.ndarray,
        k: int,
        shrinkage: str | None = None,
    ) -> dict:
    """Normalised-variance spectrum of an event set against the climatology.

    ``lambda_i < 1`` means the events fluctuate *less* than climatology along mode ``i``
    — the latent-space statement of the physical ``norm_var_hat < 1``.
    """
    sigma_event = sample_covariance(x_event, shrinkage)
    sigma_clim = sample_covariance(x_clim, None)
    out = whitened_generalised_eigh(sigma_event, sigma_clim, k)
    out["sigma_event"] = sigma_event
    out["sigma_clim"] = sigma_clim
    out["n_event"] = int(x_event.shape[0])
    out["n_clim"] = int(x_clim.shape[0])
    return out


def null_spectrum(
        x_clim: np.ndarray,
        k: int,
        n_draw: int = 50,
        n_rep: int = 200,
        seed: int = 42,
    ) -> np.ndarray:
    """Null distribution of the spectrum at the event sample size.

    Draws ``n_draw`` climatology samples **without replacement** and solves them against
    the *full* climatological covariance, exactly as the events are treated. This is the
    bias correction the raw lambda needs: with 50 samples in 90 dimensions the smallest
    lambda is far below 1 even when the samples are pure climatology, so a raw lambda
    reported on its own would badly overstate the result.

    Returns ``null[n_rep, k]``, each row ascending.
    """
    rng = np.random.default_rng(seed)
    sigma_clim = sample_covariance(x_clim, None)
    n = x_clim.shape[0]
    rows = []
    for _ in range(n_rep):
        idx = rng.choice(n, size=n_draw, replace=False)
        sigma_draw = sample_covariance(x_clim[idx], None)
        rows.append(whitened_generalised_eigh(sigma_draw, sigma_clim, k)["eigenvalues"])
    return np.stack(rows)


def bootstrap_spectrum(
        x_event: np.ndarray,
        x_clim: np.ndarray,
        k: int,
        n_boot: int = 1000,
        seed: int = 42,
    ) -> np.ndarray:
    """Event-resampling uncertainty on the spectrum.

    Resamples the events **with replacement** against a fixed climatological covariance,
    giving percentile confidence intervals per mode. Returns ``boot[n_boot, k]``.
    """
    rng = np.random.default_rng(seed)
    sigma_clim = sample_covariance(x_clim, None)
    n = x_event.shape[0]
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sigma_draw = sample_covariance(x_event[idx], None)
        rows.append(whitened_generalised_eigh(sigma_draw, sigma_clim, k)["eigenvalues"])
    return np.stack(rows)


def invariance_diagnostic(
        x_event: np.ndarray,
        x_clim: np.ndarray,
        k: int,
        seed: int = 42,
        n_map: int = 5,
    ) -> dict[str, float]:
    """How basis-independent the truncated spectrum actually is.

    Applies random *orthogonal* ``Q`` and random *general invertible*
    ``A = Q diag(exp(N(0,1)))`` reparametrisations to both sample sets and reports the max
    relative shift in lambda, medianed over ``n_map`` draws (one draw is far too noisy to
    compare across k). Without truncation the eigenvalues are exactly invariant under any
    invertible map; with truncation only the orthogonal case stays exact, because the
    top-k ``Sigma_clim`` eigenbasis is itself basis-dependent. Reporting the general-map
    error as a number is the honest version of hedging about it in prose.
    """
    rng = np.random.default_rng(seed)
    d = x_event.shape[1]
    base = normalised_variance_spectrum(x_event, x_clim, k)["eigenvalues"]

    errors: dict[str, list[float]] = {"orthogonal": [], "general": []}
    for _ in range(n_map):
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        maps = {"orthogonal": q, "general": q @ np.diag(np.exp(rng.standard_normal(d)))}
        for name, a in maps.items():
            lam = normalised_variance_spectrum(x_event @ a.T, x_clim @ a.T, k)["eigenvalues"]
            errors[name].append(float(np.max(np.abs(lam - base) / np.abs(base))))
    return {name: float(np.median(v)) for name, v in errors.items()}


def per_point_spectrum(
        x_event: np.ndarray,
        x_clim: np.ndarray,
        k: int,
        n_null: int = 100,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
    """Component A run independently at every in-box grid point.

    ``x_event[n_event, n_point, n_channel]`` and ``x_clim[n_clim, n_point, n_channel]``:
    each point gets its own channel-space generalised eigenproblem, so ``lambda_min`` and
    the geometric-mean lambda become basis-independent *maps* — the purely latent-side
    analogue of the physical ``var_hat`` figure. Null-calibrated per point.
    """
    n_point = x_event.shape[1]
    lam_min = np.empty(n_point)
    lam_geo = np.empty(n_point)
    null_p05 = np.empty(n_point)
    null_p50 = np.empty(n_point)
    n_lt1 = np.empty(n_point, dtype=int)
    for p in range(n_point):
        res = normalised_variance_spectrum(x_event[:, p, :], x_clim[:, p, :], k)
        lam = res["eigenvalues"]
        lam_min[p] = lam[0]
        lam_geo[p] = float(np.exp(np.mean(np.log(np.maximum(lam, 1e-300)))))
        n_lt1[p] = int((lam < 1).sum())
        null = null_spectrum(x_clim[:, p, :], k, n_draw=x_event.shape[0],
                             n_rep=n_null, seed=seed + p)
        null_p05[p] = np.percentile(null[:, 0], 5)
        null_p50[p] = np.percentile(null[:, 0], 50)
    return {
        "lambda_min": lam_min,
        "lambda_geomean": lam_geo,
        "n_lambda_lt1": n_lt1,
        "null_p05": null_p05,
        "null_p50": null_p50,
        "lambda_min_ratio": lam_min / null_p50,
    }
