from __future__ import annotations

import logging
from pathlib import Path

from hydra.utils import instantiate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import zarr

from instantonanalysis.instanton.utils import load_config
from instantonanalysis.instanton.utils.parsers import get_calc_args
from instantonanalysis.latent.ldt import (
    normalised_variance_spectrum,
    null_spectrum,
    trajectory_vectors,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAMP = ["#9ec5f4", "#5598e7", "#256abf", "#104281"]
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def ramp_colours(n: int) -> list[str]:
    if n <= len(RAMP):
        idx = np.linspace(0, len(RAMP) - 1, n).round().astype(int)
        return [RAMP[i] for i in idx]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ramp", RAMP)
    return [matplotlib.colors.to_hex(cmap(t)) for t in np.linspace(0, 1, n)]


def closest_indices(store_path: str, neighbours_path: str, threshold_path: str,
                    quantile: float, rolling_period: int, n_closest: int,
                    obs_var: str = "t2m0") -> np.ndarray:
    stamped = list(zarr.open_group(str(store_path), mode="r").attrs["dates"])

    nn = xr.open_dataset(neighbours_path, use_cftime=True)[obs_var]
    sel = nn.sel(quantile=quantile, rolling_period=rolling_period).dropna(dim="step")
    value_of = {d.isoformat(): float(v)
                for d, v in zip(sel.step.values, sel.values)}

    thr = float(xr.open_dataset(threshold_path)[obs_var]
                .sel(quantile=quantile, rolling_period=rolling_period))

    missing = [d for d in stamped if d not in value_of]
    if missing:
        raise KeyError(f"{len(missing)} stamped dates absent from "
                       f"{Path(neighbours_path).name} (first {missing[0]}); the "
                       f"store and the neighbours file have drifted apart.")

    d2 = np.array([(value_of[d] - thr) ** 2 for d in stamped])
    keep = np.sort(np.argsort(d2, kind="stable")[:n_closest])
    logger.info(f"    threshold={thr:.3f}; kept {len(keep)} of {len(stamped)} "
                f"closest, |obs-threshold| <= {np.sqrt(d2[keep]).max():.3f} "
                f"(dropped up to {np.sqrt(d2).max():.3f})")
    return keep


def load_vectors(path: str, var: str, xcfg, window_hours) -> np.ndarray:
    x, _ = trajectory_vectors(path, var, xcfg=xcfg, window_hours=window_hours,
                              sample="trajectory_mean")
    return x


if __name__ == "__main__":
    args = get_calc_args()
    cfg = load_config(args.config_name, args.overrides)

    var = cfg.analysis.var
    level = int(cfg.analysis.get("level", 2))
    window_hours = cfg.analysis.get("window_hours", None)
    k = int(cfg.analysis.k)
    rolling_period = int(cfg.analysis.get("rolling_period", 1))
    n_closest = cfg.analysis.get("n_closest", None)
    n_closest = None if n_closest is None else int(n_closest)
    n_null = int(cfg.analysis.n_null)
    seed = int(cfg.analysis.seed)
    quantiles = [float(q) for q in
                 cfg.analysis.get("sweep_quantiles", [0.75, 0.95, 0.99, 0.999])]
    xcfg = instantiate(cfg.xconfig)
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Climatology: {cfg.paths.clim_latents}")
    x_cl = load_vectors(cfg.paths.clim_latents, var, xcfg, window_hours)
    logger.info(f"  climatology {x_cl.shape}")

    spectra: dict[float, np.ndarray] = {}
    n_event: dict[float, int] = {}
    for q in quantiles:
        qcfg = load_config(args.config_name,
                           list(args.overrides or []) + [f"analysis.event_quantile={q}"])
        x = load_vectors(qcfg.paths.heatwave, var, xcfg, window_hours)
        if n_closest is not None and n_closest < len(x):
            keep = closest_indices(qcfg.paths.heatwave, cfg.paths.neighbours,
                                   cfg.paths.quantile_threshold, q,
                                   rolling_period, n_closest)
            x = x[keep]
        spectra[q] = normalised_variance_spectrum(x, x_cl, k)["eigenvalues"]
        n_event[q] = len(x)
        logger.info(f"  q={q:<6g} n={len(x):4d}  lambda_min={spectra[q][0]:.4f}  "
                    f"lambda_max={spectra[q][-1]:.4f}")

    nulls = {}
    for n in sorted(set(n_event.values())):
        nulls[n] = null_spectrum(x_cl, k, n_draw=n, n_rep=n_null, seed=seed)
        logger.info(f"  null n={n}: lambda_min 5/50/95 = "
                    + "/".join(f"{v:.3f}" for v in
                               np.percentile(nulls[n][:, 0], [5, 50, 95])))

    lam_no = n_no = None
    no_path = cfg.paths.get("no_heatwave", None)
    if no_path and Path(no_path).exists():
        x_no = load_vectors(no_path, var, xcfg, window_hours)
        lam_no, n_no = normalised_variance_spectrum(x_no, x_cl, k)["eigenvalues"], len(x_no)
        if n_no not in nulls:
            nulls[n_no] = null_spectrum(x_cl, k, n_draw=n_no, n_rep=n_null,
                                        seed=seed)
        p5, p50, p95 = np.percentile(nulls[n_no][:, 0], [5, 50, 95])
        inside = p5 <= lam_no[0] <= p95
        logger.info(
            f"  non-heatwave control n={n_no}: lambda_min={lam_no[0]:.4f} vs its own "
            f"null (n={n_no}) 5/50/95 = {p5:.3f}/{p50:.3f}/{p95:.3f} -> "
            f"{'on the floor (unconstrained, as expected)' if inside else 'OUTSIDE its null'}"
            " [not plotted: different n from the event sets]")

    colours = ramp_colours(len(quantiles))
    modes = np.arange(1, k + 1)
    plt.rc("font", family="serif", size=20 * 8 / 13.5)
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=160)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    event_ns = sorted(set(n_event.values()))
    for n in event_ns:
        lo, hi = np.percentile(nulls[n][:, :k], [5, 95], axis=0)
        ax.fill_between(modes, lo, hi, color=MUTED, alpha=0.16, lw=0, zorder=1)
        ax.plot(modes, np.percentile(nulls[n][:, :k], 50, axis=0), color=MUTED,
                lw=1.4, ls=(0, (5, 3)), zorder=2)

    for q, c in zip(quantiles, colours):
        lam = spectra[q]
        ax.plot(modes, lam, color=c, lw=2, marker="o", ms=4.5, mec=SURFACE,
                mew=1.0, zorder=5, label=f"q={q:g}")

    ax.set_xlabel("mode (ascending λ)")
    ax.set_ylabel("λ   normalised variance")
    ax.set_xlim(0.5, k + 0.5)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    png = out_dir / f"lambda_sweep_l{level}.png"
    fig.savefig(png, facecolor=fig.get_facecolor(), dpi=300, bbox_inches="tight")
    logger.info(f"Wrote {png}")

    rows = {"mode": modes}
    for q in quantiles:
        rows[f"lambda_q{q:g}"] = spectra[q]
    for n in sorted(nulls):
        rows[f"null_p05_n{n}"] = np.percentile(nulls[n][:, :k], 5, axis=0)
        rows[f"null_p50_n{n}"] = np.percentile(nulls[n][:, :k], 50, axis=0)
        rows[f"null_p95_n{n}"] = np.percentile(nulls[n][:, :k], 95, axis=0)
    if lam_no is not None:
        rows["lambda_nonheatwave"] = lam_no
    csv = out_dir / f"lambda_sweep_l{level}.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    logger.info(f"Wrote {csv}")
