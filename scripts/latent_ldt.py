"""Basis-independent normalised variance of encoder latents during heat extremes.

The physical-space figure this stage answers to is ``var_hat_t2m0_r1.png``:
``norm_var_hat`` collapses toward zero over Western Europe, exactly where the temperature
anomaly peaks — the large-deviations statement that extreme events follow similar
trajectories. The question here is whether the same collapse appears in the latents.

Three components, run in order:

  A. the generalised eigenvalues of ``Sigma_event v = lambda Sigma_clim v`` — the
     normalised variances themselves, basis-independent, calibrated against a null drawn
     from the climatology at the event sample size;
  B. a ridge probe from latents to region-mean ``t2m0`` (and ``z500`` as a contrast),
     giving one physically-labelled direction and its decomposition over the modes;
  C. the pushforward of the latent ensembles to physical pixels (C1) and the per-point
     latent lambda_min map (C2) — the two spatial views.
"""
from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

from hydra.utils import instantiate
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr, spearmanr

from instantonanalysis.instanton.schemas.box.healpix import fyx_to_lonlat
from instantonanalysis.instanton.utils import load_config
from instantonanalysis.instanton.utils.parsers import get_calc_args
from instantonanalysis.latent.ldt import (
    bootstrap_spectrum,
    invariance_diagnostic,
    normalised_variance_spectrum,
    null_spectrum,
    per_point_spectrum,
    region_point_coords,
    trajectory_vectors,
)
from instantonanalysis.latent.plotting import (
    plot_k_scan,
    plot_lambda_spectrum,
    plot_physical_comparison,
    plot_probe_modes,
    plot_pushforward_validation,
    plot_region_map,
)
from instantonanalysis.latent.probe import (
    fit_probe,
    fit_pushforward,
    measured_variance_ratio,
    mode_weights,
    physical_region_field,
    probe_normalised_variance,
    pushforward_variance_ratio,
    r2_per_column,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def append_metrics(metrics_path: Path, row: dict) -> None:
    """Append one summary row to the metrics CSV.

    If the existing file's header does not match this row's fields it belongs to a
    superseded version of the stage; it is rotated aside rather than silently corrupted.
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = True
    if metrics_path.exists():
        with open(metrics_path, newline="") as f:
            existing = next(csv.reader(f), [])
        if existing == list(row):
            write_header = False
        else:
            superseded = metrics_path.with_suffix(".superseded.csv")
            logger.warning(f"Metrics header changed; moving old rows to {str(superseded)!r}")
            shutil.move(str(metrics_path), str(superseded))
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def pooled_groups(n_event: int, n_step: int, offset: int = 0) -> np.ndarray:
    """Event id per pooled ``(event, step)`` sample — the CV grouping key."""
    return np.repeat(np.arange(n_event) + offset, n_step)


if __name__ == "__main__":
    logger.info("Loading config")
    args = get_calc_args()
    cfg = load_config(args.config_name, args.overrides)

    var = cfg.analysis.var
    level = int(cfg.analysis.get("level", 2))
    window_hours = cfg.analysis.get("window_hours", None)
    sample = str(cfg.analysis.get("sample", "trajectory_mean"))
    k = int(cfg.analysis.k)
    k_scan = [int(x) for x in cfg.analysis.k_scan]
    n_null = int(cfg.analysis.n_null)
    n_boot = int(cfg.analysis.n_boot)
    seed = int(cfg.analysis.seed)
    shrinkage = cfg.analysis.get("shrinkage", None)
    probe_vars = [str(v) for v in cfg.analysis.probe_vars]
    push_var = str(cfg.analysis.pushforward_var)
    nside = int(cfg.nside)

    xcfg = instantiate(cfg.xconfig)
    box64 = instantiate(cfg.box64)
    lonlat_box = (cfg.locations.box.lon_min, cfg.locations.box.lon_max,
                  cfg.locations.box.lat_min, cfg.locations.box.lat_max)
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- #
    # Component A — basis-independent normalised variance
    # ----------------------------------------------------------------- #
    logger.info(f"Component A: loading {sample} vectors (window = {window_hours}h)")
    kw = dict(xcfg=xcfg, window_hours=window_hours, sample=sample)
    x_hw, channels = trajectory_vectors(cfg.paths.heatwave, var, **kw)
    x_no, _ = trajectory_vectors(cfg.paths.no_heatwave, var, **kw)
    x_cl, _ = trajectory_vectors(cfg.paths.clim_latents, var, **kw)
    logger.info(f"Samples: heatwave {x_hw.shape}, non-heatwave {x_no.shape}, "
                f"climatology {x_cl.shape}")

    logger.info(f"Truncation scan over k = {k_scan}")
    scan_rows = []
    for k_i in k_scan:
        res_i = normalised_variance_spectrum(x_hw, x_cl, k_i)
        lam_i = res_i["eigenvalues"]
        lam_no_i = normalised_variance_spectrum(x_no, x_cl, k_i)["eigenvalues"]
        null_i = null_spectrum(x_cl, k_i, n_draw=len(x_hw), n_rep=n_null, seed=seed)
        inv_i = invariance_diagnostic(x_hw, x_cl, k_i, seed=seed)
        pct = np.percentile(null_i[:, 0], [5, 50, 95])
        scan_rows.append({
            "k": k_i,
            "lambda_min": float(lam_i[0]),
            "lambda_min_nonhw": float(lam_no_i[0]),
            "null_p05": float(pct[0]), "null_p50": float(pct[1]), "null_p95": float(pct[2]),
            "n_lambda_lt1": int((lam_i < 1).sum()),
            "clim_explained": float(res_i["clim_explained"]),
            "invariance_orthogonal": inv_i["orthogonal"],
            "invariance_general": inv_i["general"],
        })
        logger.info(f"  k={k_i:3d}  lambda_min={lam_i[0]:.4f}  null 5/50/95 = "
                    f"{pct[0]:.3f}/{pct[1]:.3f}/{pct[2]:.3f}  nonhw={lam_no_i[0]:.3f}  "
                    f"clim_expl={scan_rows[-1]['clim_explained']:.3f}  "
                    f"inv(orth/gen)={inv_i['orthogonal']:.1e}/{inv_i['general']:.3f}")
    pd.DataFrame(scan_rows).to_csv(out_dir / f"k_scan_l{level}.csv", index=False)
    plot_k_scan(scan_rows, str(out_dir / f"k_scan_l{level}.png"), k_choice=k)

    logger.info(f"Solving at k = {k}")
    res = normalised_variance_spectrum(x_hw, x_cl, k)
    lam = res["eigenvalues"]
    lam_no = normalised_variance_spectrum(x_no, x_cl, k)["eigenvalues"]
    lam_lw = normalised_variance_spectrum(x_hw, x_cl, k, shrinkage=shrinkage)["eigenvalues"]
    null = null_spectrum(x_cl, k, n_draw=len(x_hw), n_rep=n_null, seed=seed)
    boot = bootstrap_spectrum(x_hw, x_cl, k, n_boot=n_boot, seed=seed)
    inv = invariance_diagnostic(x_hw, x_cl, k, seed=seed)

    assert inv["orthogonal"] < 1e-10, (
        f"orthogonal reparametrisation must leave lambda unchanged, got {inv['orthogonal']:.2e}")
    logger.info(f"Invariance: orthogonal {inv['orthogonal']:.2e} (exact), "
                f"general invertible {inv['general']:.3f}")

    null_pct = np.percentile(null, [5, 50, 95], axis=0)
    boot_pct = np.percentile(boot, [2.5, 97.5], axis=0)
    lam_geomean = float(np.exp(np.mean(np.log(lam))))
    logger.info(f"lambda_min = {lam[0]:.4f} vs null 5/50/95 = "
                f"{null_pct[0, 0]:.3f}/{null_pct[1, 0]:.3f}/{null_pct[2, 0]:.3f}; "
                f"bootstrap 95% CI [{boot_pct[0, 0]:.3f}, {boot_pct[1, 0]:.3f}]; "
                f"{int((lam < 1).sum())}/{len(lam)} modes with lambda<1; "
                f"geomean {lam_geomean:.3f}")
    logger.info(f"Non-heatwave control lambda_min = {lam_no[0]:.4f} "
                f"(null median {null_pct[1, 0]:.3f})")

    pd.DataFrame({
        "mode": np.arange(1, len(lam) + 1),
        "lambda": lam,
        "boot_lo": boot_pct[0], "boot_hi": boot_pct[1],
        "null_p05": null_pct[0], "null_p50": null_pct[1], "null_p95": null_pct[2],
        "lambda_ledoitwolf": lam_lw,
        "lambda_nonheatwave": lam_no,
    }).to_csv(out_dir / f"lambda_spectrum_l{level}.csv", index=False)
    pd.DataFrame(res["eigenvectors"], index=channels,
                 columns=[f"mode{i + 1}" for i in range(len(lam))]).to_csv(
        out_dir / f"lambda_loadings_l{level}.csv", index_label="channel")
    plot_lambda_spectrum(lam, null, boot, str(out_dir / f"lambda_spectrum_l{level}.png"),
                         eigenvalues_nonhw=lam_no, eigenvalues_shrunk=lam_lw,
                         clim_spectrum=res["clim_spectrum"], k=k)

    # ----------------------------------------------------------------- #
    # Component B — t2m probe direction
    # ----------------------------------------------------------------- #
    logger.info("Component B: fitting the physical probe")
    steps = xr.open_zarr(cfg.paths.heatwave, decode_timedelta=True)["step"]
    if window_hours is not None:
        steps = steps[steps <= np.timedelta64(int(window_hours), "h")]
    steps = steps.values
    n_step = len(steps)

    pooled_kw = dict(xcfg=xcfg, window_hours=window_hours, sample="pooled")
    x_hw_pool, _ = trajectory_vectors(cfg.paths.heatwave, var, **pooled_kw)
    x_no_pool, _ = trajectory_vectors(cfg.paths.no_heatwave, var, **pooled_kw)
    x_probe = np.concatenate([x_hw_pool, x_no_pool], axis=0)
    groups = np.concatenate([pooled_groups(len(x_hw), n_step),
                             pooled_groups(len(x_no), n_step, offset=len(x_hw))])

    phys_hw = physical_region_field(cfg.paths.physical_heatwave, box64,
                                    sorted(set(probe_vars) | {push_var}), steps)
    phys_no = physical_region_field(cfg.paths.physical_no_heatwave, box64,
                                    sorted(set(probe_vars) | {push_var}), steps)

    probe_rows, weight_rows = [], {}
    cosine_rows, probes = {}, {}
    for pv in probe_vars:
        y = np.concatenate([phys_hw[pv].mean("points").values.reshape(-1),
                            phys_no[pv].mean("points").values.reshape(-1)])
        fit = fit_probe(x_probe, y, groups)
        nv = probe_normalised_variance(fit["w"], x_hw, x_cl)
        mw = mode_weights(fit["w"], lam, res["eigenvectors"], res["sigma_clim"])
        nv_proj = probe_normalised_variance(mw["w_projected"], x_hw, x_cl)
        assert abs(nv_proj - mw["ratio_from_modes"]) < 1e-8 * max(1.0, nv_proj), (
            f"{pv}: mode decomposition disagrees with the direct ratio "
            f"({mw['ratio_from_modes']:.6g} vs {nv_proj:.6g})")
        low_half = float(mw["weights"][: len(lam) // 2].sum())
        logger.info(
            f"  {pv}: out-of-fold R2={fit['r2']:.3f} (alpha={fit['alpha']:.3g}); "
            f"norm_var_hat_probe={nv:.3f} (projected {nv_proj:.3f}); "
            f"truncation residual={mw['truncation_residual']:.3f}; "
            f"weight on the {len(lam) // 2} lowest-lambda modes={low_half:.3f}")
        probe_rows.append({
            "var": pv, "r2_out_of_fold": fit["r2"], "alpha": fit["alpha"],
            "norm_var_hat_probe": nv, "norm_var_hat_probe_projected": nv_proj,
            "ratio_from_modes": mw["ratio_from_modes"],
            "truncation_residual": mw["truncation_residual"],
            "weight_low_half": low_half,
            "max_abs_cosine": float(np.max(np.abs(mw["cosine"]))),
            "argmax_abs_cosine_mode": int(np.argmax(np.abs(mw["cosine"])) + 1),
        })
        weight_rows[pv] = mw["weights"]
        cosine_rows[pv] = mw["cosine"]
        probes[pv] = fit

    pd.DataFrame(probe_rows).to_csv(out_dir / f"probe_l{level}.csv", index=False)
    pd.DataFrame({"mode": np.arange(1, len(lam) + 1), "lambda": lam,
                  **{f"weight_{v}": w for v, w in weight_rows.items()},
                  **{f"cosine_{v}": c for v, c in cosine_rows.items()}}).to_csv(
        out_dir / f"probe_mode_weights_l{level}.csv", index=False)
    plot_probe_modes(lam, weight_rows, str(out_dir / f"probe_modes_l{level}.png"),
                     cosines=cosine_rows)

    # ----------------------------------------------------------------- #
    # Component C1 — pushforward to physical pixels
    # ----------------------------------------------------------------- #
    logger.info("Component C1: fitting the latent -> physical pushforward")
    full_kw = dict(xcfg=xcfg, window_hours=window_hours, keep_points=True)
    xp_hw, _ = trajectory_vectors(cfg.paths.heatwave, var, sample="pooled", **full_kw)
    xp_no, _ = trajectory_vectors(cfg.paths.no_heatwave, var, sample="pooled", **full_kw)
    flat = lambda a: a.reshape(a.shape[0], -1)
    x_push = np.concatenate([flat(xp_hw), flat(xp_no)], axis=0)
    y_push = np.concatenate([phys_hw[push_var].values.reshape(-1, phys_hw.sizes["points"]),
                             phys_no[push_var].values.reshape(-1, phys_no.sizes["points"])])
    logger.info(f"  design {x_push.shape} -> {y_push.shape} "
                f"({len(np.unique(groups))} groups)")

    push = fit_pushforward(x_push, y_push, groups)
    logger.info(f"  out-of-fold R2 per pixel: median={np.median(push['r2_per_pixel']):.3f}, "
                f"min={push['r2_per_pixel'].min():.3f}, max={push['r2_per_pixel'].max():.3f} "
                f"(alpha={push['alpha']:.3g})")

    # W is linear, so W . mean_step(x) == mean_step(W . x): the map may be fitted on
    # step-level samples and applied to trajectory means without inconsistency.
    step_wise = (flat(xp_hw) @ push["W"].T).reshape(len(x_hw), n_step, -1).mean(axis=1)
    x_hw_traj, _ = trajectory_vectors(cfg.paths.heatwave, var,
                                      sample="trajectory_mean", **full_kw)
    x_no_traj, _ = trajectory_vectors(cfg.paths.no_heatwave, var,
                                      sample="trajectory_mean", **full_kw)
    x_cl_traj, _ = trajectory_vectors(cfg.paths.clim_latents, var,
                                      sample="trajectory_mean", **full_kw)
    traj_wise = flat(x_hw_traj) @ push["W"].T
    commutation = float(np.max(np.abs(step_wise - traj_wise)))
    assert commutation < 1e-6 * float(np.std(traj_wise)), (
        f"pushforward does not commute with the trajectory mean ({commutation:.3g})")
    logger.info(f"  commutation check |W.mean(x) - mean(W.x)|max = {commutation:.2e}")

    # Skill at the level the map is actually used: the trajectory mean. Averaging 16
    # steps cancels the fast residual the ridge cannot see, so the step-level R2
    # understates it. Held-out predictions are used, so this stays out-of-sample.
    n_all = len(x_hw) + len(x_no)
    oof_traj = push["oof_prediction"].reshape(n_all, n_step, -1).mean(axis=1)
    y_traj = y_push.reshape(n_all, n_step, -1).mean(axis=1)
    r2_traj = r2_per_column(y_traj, oof_traj)
    logger.info(f"  out-of-fold R2 of the trajectory mean: "
                f"median={np.median(r2_traj):.3f}, min={r2_traj.min():.3f}")

    pf = pushforward_variance_ratio(push["W"], flat(x_hw_traj), flat(x_cl_traj))
    pf_nonhw = pushforward_variance_ratio(push["W"], flat(x_no_traj), flat(x_cl_traj))
    predicted_ratio = pf["var_event"] / pf_nonhw["var_event"]
    measured_ratio = measured_variance_ratio(
        phys_hw[push_var].mean("step").values, phys_no[push_var].mean("step").values)
    # Same ratio from held-out predictions only — the fit never saw the event it predicts.
    oof_ratio = measured_variance_ratio(oof_traj[:len(x_hw)], oof_traj[len(x_hw):])
    r_valid = float(pearsonr(measured_ratio, predicted_ratio)[0])
    r_valid_oof = float(pearsonr(measured_ratio, oof_ratio)[0])
    logger.info(f"  pushforward hat ratio (heatwave/climatology): "
                f"min={pf['ratio'].min():.3f}, median={np.median(pf['ratio']):.3f}, "
                f"max={pf['ratio'].max():.3f}")
    logger.info(f"  validation: predicted vs measured hw/nonhw ratio Pearson r={r_valid:.3f} "
                f"(out-of-fold {r_valid_oof:.3f})")

    lon64, lat64 = fyx_to_lonlat(64, np.asarray(box64.f_list), np.asarray(box64.h_list),
                                 np.asarray(box64.w_list))
    assert lon64.min() > -20 and lon64.max() < 30 and lat64.min() > 30 and lat64.max() < 65, \
        "pushforward pixels fall outside the Western-Europe region"

    # The physical-space norm_var_hat this stage answers to, on the same 207 pixels.
    # Absolute levels are not comparable (different data, different climatological
    # normaliser) — the spatial pattern is what is being tested.
    phys_hat = xr.open_dataset(cfg.paths.physical_var_hat).sel(
        rolling_period=int(cfg.analysis.physical_rolling_period),
        quantile=float(cfg.analysis.physical_quantile))
    phys_hat = box64.extract(phys_hat[push_var]).values.astype(np.float64)
    r_phys = float(pearsonr(phys_hat, pf["ratio"])[0])
    rho_phys = float(spearmanr(phys_hat, pf["ratio"])[0])
    logger.info(f"  vs physical norm_var_hat over the same {len(phys_hat)} pixels: "
                f"Pearson r={r_phys:.3f}, Spearman={rho_phys:.3f} "
                f"(physical min {phys_hat.min():.3f} at "
                f"{lon64[np.argmin(phys_hat)]:.1f}E/{lat64[np.argmin(phys_hat)]:.1f}N, "
                f"latent min {pf['ratio'].min():.3f} at "
                f"{lon64[np.argmin(pf['ratio'])]:.1f}E/{lat64[np.argmin(pf['ratio'])]:.1f}N)")

    xr.Dataset(
        {
            "lon": ("points", lon64), "lat": ("points", lat64),
            "r2": ("points", push["r2_per_pixel"]),
            "r2_trajectory": ("points", r2_traj),
            "var_event": ("points", pf["var_event"]),
            "var_clim": ("points", pf["var_clim"]),
            "ratio": ("points", pf["ratio"]),
            "var_nonheatwave": ("points", pf_nonhw["var_event"]),
            "ratio_predicted_hw_nonhw": ("points", predicted_ratio),
            "ratio_predicted_oof_hw_nonhw": ("points", oof_ratio),
            "ratio_measured_hw_nonhw": ("points", measured_ratio),
            "physical_var_hat": ("points", phys_hat),
        },
        attrs={"pushforward_var": push_var, "alpha": push["alpha"],
               "validation_pearson_r": r_valid,
               "validation_pearson_r_oof": r_valid_oof,
               "physical_pearson_r": r_phys, "physical_spearman_r": rho_phys,
               "level": level},
    ).to_netcdf(out_dir / f"pushforward_l{level}.nc")

    plot_physical_comparison(lon64, lat64, pf["ratio"], phys_hat, r_phys, rho_phys,
                             str(out_dir / f"pushforward_vs_physical_l{level}.png"),
                             box=lonlat_box)

    plot_region_map(
        lon64, lat64, pf["ratio"] * 100.0,
        str(out_dir / f"pushforward_var_hat_{push_var}_l{level}.png"),
        label=r"$\hat{V}$ [%]",
        title=f"Pushed-forward normalised variance of {push_var} (l{level} latents)\n"
              f"heatwave trajectory means vs JJA climatology",
        cmap="viridis", extend="max", marker_size=45,
        vmax=float(np.percentile(pf["ratio"] * 100.0, 98)), box=lonlat_box)
    plot_region_map(
        lon64, lat64, push["r2_per_pixel"],
        str(out_dir / f"pushforward_r2_{push_var}_l{level}.png"),
        label=r"out-of-fold $R^2$",
        title=f"Pushforward skill: l{level} latents -> {push_var} per pixel",
        cmap="magma", extend="neither", marker_size=45, box=lonlat_box)
    plot_pushforward_validation(predicted_ratio, measured_ratio, r_valid,
                                str(out_dir / f"pushforward_validation_l{level}.png"),
                                r2_per_pixel=push["r2_per_pixel"],
                                predicted_oof=oof_ratio, r_oof=r_valid_oof,
                                r2_trajectory=r2_traj)

    # ----------------------------------------------------------------- #
    # Component C2 — per-point latent lambda_min map
    # ----------------------------------------------------------------- #
    logger.info("Component C2: per-point latent spectrum")
    # x_hw_traj / x_cl_traj are already (n, n_points, n_channel) from C1.
    pp = per_point_spectrum(x_hw_traj, x_cl_traj, k, n_null=max(n_null // 2, 50), seed=seed)
    f_l, h_l, w_l = region_point_coords(cfg.paths.heatwave, level)
    lon_l, lat_l = fyx_to_lonlat(nside, f_l, h_l, w_l)
    assert lon_l.min() > -20 and lon_l.max() < 30 and lat_l.min() > 30 and lat_l.max() < 65, \
        "latent grid points fall outside the Western-Europe region"
    logger.info(f"  lambda_min over {len(lon_l)} points: min={pp['lambda_min'].min():.3f}, "
                f"median={np.median(pp['lambda_min']):.3f}; "
                f"points below their own null 5th pct: "
                f"{int((pp['lambda_min'] < pp['null_p05']).sum())}/{len(lon_l)}")

    xr.Dataset(
        {"lon": ("points", lon_l), "lat": ("points", lat_l),
         **{name: ("points", val) for name, val in pp.items()}},
        attrs={"k": k, "level": level, "n_event": int(len(x_hw))},
    ).to_netcdf(out_dir / f"lambda_per_point_l{level}.nc")

    plot_region_map(
        lon_l, lat_l, pp["lambda_min"],
        str(out_dir / f"lambda_min_map_l{level}.png"),
        label=r"$\lambda_{min}$", marker_size=900,
        title=f"Per-point latent normalised variance (l{level}, k={k})\n"
              f"smallest generalised eigenvalue, heatwave vs JJA climatology",
        cmap="viridis", extend="neither", box=lonlat_box)
    plot_region_map(
        lon_l, lat_l, pp["lambda_min_ratio"],
        str(out_dir / f"lambda_min_vs_null_map_l{level}.png"),
        label=r"$\lambda_{min}$ / null median", marker_size=900,
        title=f"Per-point lambda_min relative to its own null (l{level}, k={k})\n"
              f"<1 = more constrained than a same-size climatology draw",
        cmap="viridis", extend="neither", box=lonlat_box)

    # ----------------------------------------------------------------- #
    logger.info("Writing metrics row")
    probe_df = pd.DataFrame(probe_rows).set_index("var")
    append_metrics(Path(cfg.paths.get("metrics", str(out_dir / "ldt_metrics.csv"))), {
        "level": level,
        "sample": sample,
        "k": k,
        "lambda_min": round(float(lam[0]), 6),
        "lambda_min_null_p05": round(float(null_pct[0, 0]), 6),
        "lambda_min_null_p50": round(float(null_pct[1, 0]), 6),
        "lambda_min_boot_lo": round(float(boot_pct[0, 0]), 6),
        "lambda_min_boot_hi": round(float(boot_pct[1, 0]), 6),
        "n_lambda_lt1": int((lam < 1).sum()),
        "lambda_geomean": round(lam_geomean, 6),
        "lambda_min_nonhw_control": round(float(lam_no[0]), 6),
        "probe_r2_t2m0": round(float(probe_df.loc[push_var, "r2_out_of_fold"]), 6),
        "probe_norm_var_t2m0": round(float(probe_df.loc[push_var, "norm_var_hat_probe"]), 6),
        "pushforward_r2_median": round(float(np.median(push["r2_per_pixel"])), 6),
        "pushforward_r2_traj_median": round(float(np.median(r2_traj)), 6),
        "pushforward_ratio_min": round(float(pf["ratio"].min()), 6),
        "pushforward_ratio_median": round(float(np.median(pf["ratio"])), 6),
        "pushforward_validation_r": round(r_valid, 6),
        "pushforward_validation_r_oof": round(r_valid_oof, 6),
        "physical_pattern_pearson_r": round(r_phys, 6),
        "physical_pattern_spearman_r": round(rho_phys, 6),
        "lambda_min_per_point_median": round(float(np.median(pp["lambda_min"])), 6),
        "invariance_err_orthogonal": f"{inv['orthogonal']:.3e}",
        "invariance_err_general": round(inv["general"], 6),
    })
    logger.info(f"Done. Outputs in {str(out_dir)!r}")
