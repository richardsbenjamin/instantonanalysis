from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from instantonanalysis.instanton.schemas.box import HealPixBox

ALPHAS = np.logspace(-2, 5, 15)


def physical_region_field(
        physical_path: str,
        box64: HealPixBox,
        var_names: list[str],
        steps: np.ndarray,
    ) -> xr.Dataset:
    ds = xr.open_zarr(physical_path, decode_timedelta=True)[list(var_names)]
    ds = ds.sel(step=steps)

    if "face" not in ds.dims:
        pts = [d for d in ds.dims if d not in ("time", "step")]
        if len(pts) != 1:
            raise ValueError(
                f"{physical_path}: expected one spatial dim besides time/step, "
                f"got {pts}")
        if ds.sizes[pts[0]] != len(box64.f_list):
            raise ValueError(
                f"{physical_path}: has {ds.sizes[pts[0]]} points but box64 has "
                f"{len(box64.f_list)}; it was filtered to a different region.")
        ds = ds.rename({pts[0]: "points"}) if pts[0] != "points" else ds
        return ds.transpose("time", "step", "points").compute(scheduler="synchronous")

    faces = sorted(set(box64.f_list))
    ds = ds.sel(face=faces).compute(scheduler="synchronous")
    out = box64.extract(ds)
    return out.transpose("time", "step", "points")


def _ridge_grouped(
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        alphas: np.ndarray = ALPHAS,
        n_splits: int = 5,
    ) -> dict:
    y2 = y if y.ndim == 2 else y[:, None]
    cv = GroupKFold(n_splits=n_splits)
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])

    search = GridSearchCV(pipe, {"ridge__alpha": alphas}, cv=cv, scoring="r2")
    search.fit(x, y2, groups=groups)
    alpha = float(search.best_params_["ridge__alpha"])

    best = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    pred = cross_val_predict(best, x, y2, cv=cv, groups=groups)
    pred = pred if pred.ndim == 2 else pred[:, None]
    ss_res = ((y2 - pred) ** 2).sum(axis=0)
    ss_tot = ((y2 - y2.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot

    best.fit(x, y2)
    scaler, ridge = best["scale"], best["ridge"]
    coef = np.atleast_2d(ridge.coef_) / scaler.scale_
    intercept = np.atleast_1d(ridge.intercept_) - coef @ scaler.mean_
    return {"coef": coef, "intercept": intercept, "r2": r2, "alpha": alpha,
            "oof_prediction": pred}


def mode_weights(
        w: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        sigma_clim: np.ndarray,
    ) -> dict:
    a = eigenvectors.T @ (sigma_clim @ w)
    captured = float(np.sum(a ** 2))
    total = float(w @ sigma_clim @ w)
    norms = np.linalg.norm(eigenvectors, axis=0) * np.linalg.norm(w)
    return {
        "a": a,
        "weights": a ** 2 / captured,
        "ratio_from_modes": float(np.sum(eigenvalues * a ** 2) / captured),
        "truncation_residual": float(1.0 - captured / total),
        "cosine": (eigenvectors.T @ w) / norms,
        "w_projected": eigenvectors @ a,
    }


def pushforward_variance_ratio(
        w_map: np.ndarray, x_event_traj: np.ndarray, x_clim_traj: np.ndarray
    ) -> dict[str, np.ndarray]:
    y_event = x_event_traj @ w_map.T
    y_clim = x_clim_traj @ w_map.T
    var_event = y_event.var(axis=0, ddof=1)
    var_clim = y_clim.var(axis=0, ddof=1)
    return {"var_event": var_event, "var_clim": var_clim,
            "ratio": var_event / var_clim}
