from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr


if TYPE_CHECKING:
    from instantonanalysis.instanton.schemas.xconfig import XConfigHealPix


def _points_dim(da: xr.DataArray, xcfg: XConfigHealPix) -> list[str]:
    return [d for d in da.dims if d not in (xcfg.time_dim, "step", xcfg.channel_dim)]


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
        block = block.values.astype(np.float64)
        if sample == "trajectory_mean":
            block = block.mean(axis=1)
        else:
            block = block.reshape(-1, *block.shape[2:])
        parts.append(block)
    x = np.concatenate(parts, axis=0)

    if keep_points:
        x = x.reshape(x.shape[0], -1, x.shape[-1])
    else:
        x = x.mean(axis=tuple(range(1, x.ndim - 1)))
    return x, da[xcfg.channel_dim].values


def sample_covariance(x: np.ndarray) -> np.ndarray:
    return np.cov(x, rowvar=False)


def whitened_generalised_eigh(
        sigma_event: np.ndarray, sigma_clim: np.ndarray, k: int
    ) -> dict:
    lam, u = np.linalg.eigh(sigma_clim)
    lam, u = lam[::-1], u[:, ::-1]
    k = int(min(k, np.sum(lam > lam[0] * 1e-12)))
    lam_k, u_k = lam[:k], u[:, :k]

    W = (u_k / np.sqrt(lam_k)).T
    m = W @ sigma_event @ W.T
    m = 0.5 * (m + m.T)
    mu, u_m = np.linalg.eigh(m)
    return {
        "eigenvalues": mu,
        "eigenvectors": W.T @ u_m,
        "whitener": W,
        "k": k,
        "clim_explained": float(lam_k.sum() / lam.sum()),
        "clim_spectrum": lam,
    }


def normalised_variance_spectrum(
        x_event: np.ndarray,
        x_clim: np.ndarray,
        k: int,
    ) -> dict:
    sigma_event = sample_covariance(x_event)
    sigma_clim = sample_covariance(x_clim)
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
    rng = np.random.default_rng(seed)
    sigma_clim = sample_covariance(x_clim)
    n = x_clim.shape[0]

    rows = []
    for _ in range(n_rep):
        draw = x_clim[rng.integers(0, n, size=n_draw)]
        rows.append(whitened_generalised_eigh(
            sample_covariance(draw), sigma_clim, k)["eigenvalues"])
    return np.stack(rows)
