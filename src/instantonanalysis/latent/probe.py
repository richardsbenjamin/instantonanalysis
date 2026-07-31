"""Physical labelling of the latent space: probe directions and the pushforward map.

The generalised eigenvalues in :mod:`instantonanalysis.latent.ldt` are basis-independent
but anonymous — they say *how much* the event ensemble is constrained, not *in what*. The
labelling is recovered here by an explicit linear probe fitted from latents to physical
fields, rather than by hoping that some latent channel means "temperature" (it does not:
a channel is an arbitrary coordinate).

Two maps, same protocol:

* **probe** (Component B) — region-mean latent vector -> region-mean physical scalar
  (``t2m0``, with ``z500`` as a contrast). Gives one physically-labelled direction ``w``
  whose normalised variance is directly comparable to the physical-space number.
* **pushforward** (Component C1) — full latent feature vector -> ``t2m0`` at every in-box
  physical pixel. Pushing the event and climatology ensembles through it gives a variance
  ratio map on the physical grid, in temperature units.

Both are ridge regressions cross-validated with ``GroupKFold`` grouped **by event**, so
steps from one trajectory can never leak between folds, and both report out-of-fold R² so
the reader knows how much of the physical field is actually recoverable.
"""
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


# --------------------------------------------------------------------------- #
# Physical fields
# --------------------------------------------------------------------------- #
def physical_region_field(
        physical_path: str,
        box64: HealPixBox,
        var_names: list[str],
        steps: np.ndarray,
    ) -> xr.Dataset:
    """In-box physical fields as ``(time, step, points)`` per variable.

    The forecast output is nside-64 over the full sphere; only a couple of HEALPix faces
    touch the box, so those are sliced and loaded first (one chunk at a time) before the
    pointwise selection. ``steps`` aligns the lead times to the latents' — the physical
    output carries an extra init step (0h) that the latents do not.

    The resulting ``points`` order is ``box64``'s own, which is 1:1 with the ``points_l0``
    order of the region-filtered latent zarrs (both come from the same box construction).
    """
    ds = xr.open_zarr(physical_path, decode_timedelta=True)[list(var_names)]
    ds = ds.sel(step=steps)
    faces = sorted(set(box64.f_list))
    ds = ds.sel(face=faces).compute(scheduler="synchronous")
    out = box64.extract(ds)
    return out.transpose("time", "step", "points")


def physical_region_series(
        physical_path: str,
        box64: HealPixBox,
        var_names: list[str],
        steps: np.ndarray,
    ) -> xr.DataArray:
    """Region-mean physical series, ``(event, step, var)``."""
    ds = physical_region_field(physical_path, box64, var_names, steps).mean("points")
    return ds.to_array(dim="var").transpose("time", "step", "var")


# --------------------------------------------------------------------------- #
# Grouped-CV ridge
# --------------------------------------------------------------------------- #
def _ridge_grouped(
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        alphas: np.ndarray = ALPHAS,
        n_splits: int = 5,
    ) -> dict:
    """Ridge with the alpha chosen by event-grouped CV; out-of-fold R² per output.

    Features are standardised inside the pipeline (so the scaler is refit per fold), then
    the fitted map is converted back to raw latent coordinates: the returned ``coef`` /
    ``intercept`` satisfy ``y ~ coef @ x + intercept`` with ``x`` in the original latent
    units. That matters downstream — the map has to be applied to trajectory means, and an
    affine map in the original coordinates commutes with averaging.
    """
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
    coef = np.atleast_2d(ridge.coef_) / scaler.scale_          # (n_out, n_feature)
    intercept = np.atleast_1d(ridge.intercept_) - coef @ scaler.mean_
    return {"coef": coef, "intercept": intercept, "r2": r2, "alpha": alpha,
            "oof_prediction": pred}


def fit_probe(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Probe direction ``w`` mapping a latent vector to one physical scalar."""
    res = _ridge_grouped(x, y, groups)
    return {"w": res["coef"][0], "intercept": float(res["intercept"][0]),
            "r2": float(res["r2"][0]), "alpha": res["alpha"]}


def fit_pushforward(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Linear map ``W`` from the full latent feature vector to every physical pixel.

    ``oof_prediction`` (held-out predictions for every sample) is returned so the map can
    also be validated at the trajectory-mean level, which is the level it is actually used
    at — the step-level R² understates it, because averaging 16 steps cancels the fast
    residual the ridge cannot see.
    """
    res = _ridge_grouped(x, y, groups)
    return {"W": res["coef"], "intercept": res["intercept"],
            "r2_per_pixel": res["r2"], "alpha": res["alpha"],
            "oof_prediction": res["oof_prediction"]}


def r2_per_column(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Column-wise coefficient of determination."""
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# Applying the maps to the ensembles
# --------------------------------------------------------------------------- #
def probe_normalised_variance(
        w: np.ndarray, x_event_traj: np.ndarray, x_clim_traj: np.ndarray
    ) -> float:
    """``var_event(w.x) / var_clim(w.x)`` over trajectory-mean vectors.

    The hat construction along one physically-labelled direction — the single number in
    this stage that is directly comparable to the physical-space ``norm_var_hat``.
    """
    return float(np.var(x_event_traj @ w, ddof=1) / np.var(x_clim_traj @ w, ddof=1))


def mode_weights(
        w: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        sigma_clim: np.ndarray,
    ) -> dict:
    """Decompose the probe's normalised variance over the fluctuation modes.

    The generalised eigenvectors are ``Sigma_clim``-orthonormal, not Euclidean-orthogonal,
    so a plain cosine between ``w`` and ``v_i`` does not tell you how much of the probe
    lives in mode ``i``. The right coordinates are ``a_i = v_i^T Sigma_clim w``, for which

        ``w^T Sigma_event w / w^T Sigma_clim w = sum_i lambda_i a_i^2 / sum_i a_i^2``

    holds exactly inside the retained subspace. The probe's normalised variance is
    therefore a weighted average of the lambdas with weights ``a_i^2 / sum a_i^2``, and
    that weight distribution ordered by lambda is the rigorous version of "does t2m live
    in the constrained subspace": weight piled on the low-lambda modes *is* the latent
    statement of "variance is lowest where temperatures are highest".

    ``truncation_residual`` is the fraction of the probe's climatological variance falling
    outside the retained k-subspace; plain cosines are reported alongside.
    """
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
    """Per-pixel pushed-forward variance ratio (the hat construction, in t2m units).

    ``diag(W Sigma W^T)`` is exactly the sample variance of ``W x`` across samples, so the
    samples are pushed through ``W`` directly and their variance taken — numerically
    identical at O(n d) instead of forming a 1710x1710 covariance. The additive intercept
    is irrelevant to a variance and is omitted.
    """
    y_event = x_event_traj @ w_map.T
    y_clim = x_clim_traj @ w_map.T
    var_event = y_event.var(axis=0, ddof=1)
    var_clim = y_clim.var(axis=0, ddof=1)
    return {"var_event": var_event, "var_clim": var_clim,
            "ratio": var_event / var_clim}


def measured_variance_ratio(
        y_event: np.ndarray, y_reference: np.ndarray
    ) -> np.ndarray:
    """Directly measured per-pixel variance ratio of two physical ensembles.

    The honesty check on the pushforward: the same ratio the map *predicts* can be
    measured straight off the physical forecast fields.
    """
    return y_event.var(axis=0, ddof=1) / y_reference.var(axis=0, ddof=1)
