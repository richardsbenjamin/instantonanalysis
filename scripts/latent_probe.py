from __future__ import annotations

import sys
from pathlib import Path

from hydra.utils import instantiate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from scipy.stats import pearsonr, spearmanr

from instantonanalysis.instanton.schemas.box.healpix import fyx_to_lonlat
from instantonanalysis.instanton.utils import load_config
from instantonanalysis.instanton.utils.parsers import get_calc_args
from instantonanalysis.latent.ldt import normalised_variance_spectrum, trajectory_vectors
from instantonanalysis.latent.probe import (
    _ridge_grouped, mode_weights, physical_region_field, pushforward_variance_ratio,
)

SERIES = {"t2m0": "#2a78d6", "z500": "#eb6834"}
SEQ = "magma"
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def qconfig(args, q):
    return load_config(args.config_name,
                       list(args.overrides or []) + [f"analysis.event_quantile={q}"])


def steps_of(path, window_hours):
    steps = xr.open_zarr(path, decode_timedelta=True)["step"]
    if window_hours is not None:
        steps = steps[steps <= np.timedelta64(int(window_hours), "h")]
    return steps.values


def latent_repeat(path, var, steps):
    a = xr.open_zarr(path, decode_timedelta=True)[var].isel(time=0).sel(step=steps).values
    a = a.reshape(len(steps), -1)
    rep = 1
    while rep < len(a) and np.array_equal(a[0], a[rep]):
        rep += 1
    if len(a) % rep or not all(np.array_equal(a[i], a[i - i % rep])
                               for i in range(len(a))):
        raise ValueError(f"{path}: the {len(a)} stored steps do not split into "
                         f"equal blocks of identical latents (found rep={rep})")
    return rep


def init_keys(path):
    return [t.isoformat() for t in xr.open_zarr(path, use_cftime=True).time.values]


def subset_r2(y, pred, traj, n_step):
    rows = (np.asarray(traj)[:, None] * n_step + np.arange(n_step)[None, :]).ravel()
    yt, pt = y[rows].astype(np.float64), pred[rows].astype(np.float64)
    return 1.0 - ((yt - pt) ** 2).sum(0) / ((yt - yt.mean(0)) ** 2).sum(0)


def traj_r2(y, pred, traj, n_step):
    yt = y.reshape(-1, n_step, y.shape[-1])[traj].mean(1).astype(np.float64)
    pt = pred.reshape(-1, n_step, pred.shape[-1])[traj].mean(1).astype(np.float64)
    return 1.0 - ((yt - pt) ** 2).sum(0) / ((yt - yt.mean(0)) ** 2).sum(0)


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
    probe_vars = [str(v) for v in cfg.analysis.probe_vars]
    push_var = str(cfg.analysis.pushforward_var)
    quantiles = [float(q) for q in
                 cfg.analysis.get("sweep_quantiles", [0.75, 0.95, 0.99, 0.999])]
    xcfg = instantiate(cfg.xconfig)
    box64 = instantiate(cfg.box64)
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).parent))

    qcfgs = {q: qconfig(args, q) for q in quantiles}
    keep = {}
    if n_closest is not None:
        from latent_lambda import closest_indices
        for q in quantiles:
            p = qcfgs[q].paths.heatwave
            if n_closest < xr.open_zarr(p, decode_timedelta=True).sizes["time"]:
                keep[q] = closest_indices(p, cfg.paths.neighbours,
                                          cfg.paths.quantile_threshold, q,
                                          rolling_period, n_closest)

    steps = steps_of(qcfgs[quantiles[0]].paths.heatwave, window_hours)
    n_raw = len(steps)
    rep = latent_repeat(qcfgs[quantiles[0]].paths.heatwave, var, steps)
    n_step = n_raw // rep
    print(f"latent step axis: {n_raw} stored steps = {n_step} distinct latents "
          f"(each written {rep}x); targets averaged within each block")
    kw = dict(xcfg=xcfg, window_hours=window_hours)

    sources = [(q, qcfgs[q].paths.heatwave, qcfgs[q].paths.physical_heatwave, keep.get(q))
               for q in quantiles]
    sources.append(("nonhw", cfg.paths.no_heatwave, cfg.paths.physical_no_heatwave, None))

    index, members = {}, {}
    xp_pool, xf_pool, yf_pool = [], [], []
    yp_pool = {pv: [] for pv in probe_vars}
    phys_vars = sorted(set(probe_vars) | {push_var})
    for tag, lpath, ppath, sel in sources:
        keys = init_keys(lpath)
        n = len(keys)
        xp, _ = trajectory_vectors(lpath, var, sample="pooled", **kw)
        xf, _ = trajectory_vectors(lpath, var, sample="pooled", keep_points=True, **kw)
        xp = xp.reshape(n, n_raw, -1)[:, ::rep]
        xf = xf.reshape(n, n_raw, -1).astype(np.float32)[:, ::rep]
        phys = physical_region_field(ppath, box64, phys_vars, steps)
        yp = {pv: phys[pv].mean("points").values.reshape(n, n_step, rep).mean(-1)
              for pv in probe_vars}
        yf = phys[push_var].values.astype(np.float32).reshape(
            n, n_step, rep, -1).mean(2)
        mem = []
        for i in (range(n) if sel is None else sel):
            if keys[i] not in index:
                index[keys[i]] = len(index)
                xp_pool.append(xp[i]); xf_pool.append(xf[i]); yf_pool.append(yf[i])
                for pv in probe_vars:
                    yp_pool[pv].append(yp[pv][i])
            mem.append(index[keys[i]])
        members[tag] = np.asarray(mem)
        del xp, xf, phys, yf

    n_traj = len(index)
    x_probe = np.stack(xp_pool).reshape(n_traj * n_step, -1)
    x_push = np.stack(xf_pool).reshape(n_traj * n_step, -1)
    y_push = np.stack(yf_pool).reshape(n_traj * n_step, -1)
    del xp_pool, xf_pool, yf_pool
    groups = np.repeat(np.arange(n_traj), n_step)
    print(f"pooled training set: {n_traj} unique trajectories "
          f"({sum(len(members[q]) for q in quantiles)} event slots + "
          f"{len(members['nonhw'])} non-heatwave), design {x_push.shape}")

    probe = {}
    for pv in probe_vars:
        y = np.stack(yp_pool[pv]).reshape(-1, 1)
        res = _ridge_grouped(x_probe, y, groups)
        probe[pv] = {"w": res["coef"][0], "alpha": res["alpha"],
                     "r2": float(res["r2"][0]), "oof": res["oof_prediction"], "y": y}
    push = _ridge_grouped(x_push, y_push, groups)
    W = push["coef"]

    x_cl, _ = trajectory_vectors(cfg.paths.clim_latents, var, sample="trajectory_mean", **kw)
    x_cl_full, _ = trajectory_vectors(cfg.paths.clim_latents, var, sample="trajectory_mean",
                                      keep_points=True, **kw)
    flat = lambda a: a.reshape(a.shape[0], -1)
    lon, lat = fyx_to_lonlat(64, np.asarray(box64.f_list), np.asarray(box64.h_list),
                             np.asarray(box64.w_list))
    phys_hat_all = xr.open_dataset(cfg.paths.physical_var_hat)

    results, comp, summary = {}, {}, []
    for q in quantiles:
        lpath = qcfgs[q].paths.heatwave
        x_hw, _ = trajectory_vectors(lpath, var, sample="trajectory_mean", **kw)
        x_hw_full, _ = trajectory_vectors(lpath, var, sample="trajectory_mean",
                                          keep_points=True, **kw)
        if q in keep:
            x_hw, x_hw_full = x_hw[keep[q]], x_hw_full[keep[q]]
        res = normalised_variance_spectrum(x_hw, x_cl, k)
        eval_traj = np.unique(np.concatenate([members[q], members["nonhw"]]))

        weights, r2 = {}, {}
        for pv in probe_vars:
            mw = mode_weights(probe[pv]["w"], res["eigenvalues"], res["eigenvectors"],
                              res["sigma_clim"])
            weights[pv] = mw["weights"]
            r2[pv] = float(subset_r2(probe[pv]["y"], probe[pv]["oof"], eval_traj, n_step)[0])
        results[q] = {"lambda": res["eigenvalues"], "n_event": len(x_hw),
                      "weights": weights, "r2": r2}

        pf = pushforward_variance_ratio(W, flat(x_hw_full), flat(x_cl_full))
        ph = phys_hat_all.sel(rolling_period=rolling_period, quantile=q)
        phys_hat = box64.extract(ph[push_var]).values.astype(np.float64)
        comp[q] = {"lon": lon, "lat": lat, "latent": pf["ratio"], "physical": phys_hat,
                   "r": float(pearsonr(phys_hat, pf["ratio"])[0]),
                   "rho": float(spearmanr(phys_hat, pf["ratio"])[0])}

        summary.append({
            "quantile": q, "n_event": len(x_hw),
            "lambda_min": float(res["eigenvalues"][0]),
            **{f"probe_r2_{pv}": r2[pv] for pv in probe_vars},
            **{f"weight_mode1_{pv}": float(weights[pv][0]) for pv in probe_vars},
            **{f"weight_low7_{pv}": float(weights[pv][:7].sum()) for pv in probe_vars},
            "pushforward_r2_traj_median": float(np.median(
                traj_r2(y_push, push["oof_prediction"], eval_traj, n_step))),
            "physical_pattern_pearson_r": comp[q]["r"],
            "physical_pattern_spearman_r": comp[q]["rho"],
        })

    pd.DataFrame(summary).to_csv(out_dir / f"probe_pooled_summary_l{level}.csv", index=False)
    all_traj = np.arange(n_traj)
    pd.DataFrame([{
        "n_train_trajectories": n_traj,
        **{f"alpha_probe_{pv}": probe[pv]["alpha"] for pv in probe_vars},
        **{f"r2_probe_{pv}": probe[pv]["r2"] for pv in probe_vars},
        "alpha_pushforward": push["alpha"],
        "r2_pushforward_traj_median": float(np.median(
            traj_r2(y_push, push["oof_prediction"], all_traj, n_step))),
    }]).to_csv(out_dir / f"probe_pooled_fit_l{level}.csv", index=False)

    ncol = 2
    nrow = int(np.ceil(len(quantiles) / ncol))
    plt.rc("font", family="serif", size=20 * 11 / 27)
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.4 * nrow), dpi=160,
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()

    modes = np.arange(1, k + 1)
    width = 0.38
    ymax = max(w.max() for r in results.values() for w in r["weights"].values())

    for ax, q in zip(axes, quantiles):
        r = results[q]
        ax.set_facecolor(SURFACE)
        for i, (pv, colour) in enumerate(SERIES.items()):
            if pv not in r["weights"]:
                continue
            ax.bar(modes + (i - 0.5) * width, r["weights"][pv], width * 0.94,
                   color=colour, lw=0, zorder=3,
                   label=f"{pv}  (R²={r['r2'][pv]:.2f})")
        ax.set_title(f"({chr(97 + list(quantiles).index(q))})")
        ax.set_ylim(0, ymax * 1.12)
        ax.legend(frameon=False, loc="upper right")

    for ax in axes[len(quantiles):]:
        ax.set_visible(False)
    for ax in axes[-ncol:]:
        ax.set_xlabel("mode  (ascending λ — mode 1 is the most constrained)")
    for ax in axes[::ncol]:
        ax.set_ylabel("share of probe direction\n(climatological variance)")

    fig.tight_layout()
    png = out_dir / f"probe_modes_panels_pooled_l{level}.png"
    fig.savefig(png, facecolor=fig.get_facecolor(), dpi=300, bbox_inches="tight")

    rows = {"mode": modes}
    for q in quantiles:
        rows[f"lambda_q{q:g}"] = results[q]["lambda"]
        for pv in probe_vars:
            rows[f"weight_{pv}_q{q:g}"] = results[q]["weights"][pv]
    pd.DataFrame(rows).to_csv(out_dir / f"probe_modes_panels_pooled_l{level}.csv", index=False)
    print(f"wrote {png}")

    proj = ccrs.PlateCarree()
    plt.rc("font", family="serif", size=20 * 12.5 / 27)
    small = 0.75 * plt.rcParams["font.size"]
    fig = plt.figure(figsize=(12.5, 3.4 * len(quantiles)), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(len(quantiles), 3, width_ratios=[1, 1, 0.85],
                          hspace=0.16, wspace=0.38)
    lon_min, lon_max = cfg.locations.box.lon_min, cfg.locations.box.lon_max
    lat_min, lat_max = cfg.locations.box.lat_min, cfg.locations.box.lat_max

    for i, q in enumerate(quantiles):
        c = comp[q]
        first_ax = None
        for j, (field, label) in enumerate([(c["latent"], "latent"),
                                            (c["physical"], "physical")]):
            ax = fig.add_subplot(gs[i, j], projection=proj)
            if first_ax is None:
                first_ax = ax
            sc = ax.scatter(c["lon"], c["lat"], c=field, cmap=SEQ, s=175,
                            marker="s", lw=0, transform=proj)
            ax.coastlines("50m", color="#1a1a19", linewidth=0.6)
            ax.set_extent([lon_min - 1, lon_max + 1, lat_max - 1, lat_min + 1], crs=proj)
            gl = ax.gridlines(draw_labels=True, linewidth=0)
            gl.top_labels = gl.right_labels = False
            gl.bottom_labels = (i == len(quantiles) - 1)
            gl.left_labels = (j == 0)
            if hasattr(gl, "xlabel_style"):
                gl.xlabel_style = {"size": plt.rcParams["font.size"], "color": "black"}
                gl.ylabel_style = {"size": plt.rcParams["font.size"], "color": "black"}
            cb = fig.colorbar(sc, ax=ax, fraction=0.040, pad=0.02)
            ax.text(0.02, 0.96, label, transform=ax.transAxes, color="black",
                    fontsize=small, va="top",
                    bbox=dict(fc=SURFACE, ec="none", alpha=0.8, pad=1.8))

        ax = fig.add_subplot(gs[i, 2])
        ax.set_facecolor(SURFACE)
        ax.scatter(c["physical"], c["latent"], s=14, color=SERIES["t2m0"],
                   lw=0, alpha=0.75, zorder=3)
        ax.text(0.04, 0.93, f"r = {c['r']:.2f}\nρ = {c['rho']:.2f}",
                transform=ax.transAxes, fontsize=small, va="top")
        if i == len(quantiles) - 1:
            ax.set_xlabel("physical  $\\hat{V}$")
        ax.set_ylabel("latent  $\\hat{V}$")

        first_ax.set_title(f"({chr(97 + i)})", loc="left")

    png2 = out_dir / f"pushforward_vs_physical_panels_pooled_l{level}.png"
    fig.savefig(png2, facecolor=fig.get_facecolor(), dpi=300, bbox_inches="tight")

    pd.DataFrame({"quantile": quantiles,
                  "pearson_r": [comp[q]["r"] for q in quantiles],
                  "spearman_r": [comp[q]["rho"] for q in quantiles]}).to_csv(
        out_dir / f"pushforward_vs_physical_panels_pooled_l{level}.csv", index=False)
    print(f"wrote {png2}")
