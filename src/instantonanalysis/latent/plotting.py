"""Figures for the latent LDT stage."""
from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np


def plot_lambda_spectrum(
        eigenvalues: np.ndarray,
        null: np.ndarray,
        boot: np.ndarray,
        save_path: str,
        eigenvalues_nonhw: np.ndarray | None = None,
        eigenvalues_shrunk: np.ndarray | None = None,
        clim_spectrum: np.ndarray | None = None,
        k: int | None = None,
    ) -> None:
    """Normalised-variance spectrum against its null, with the climatological spectrum.

    Left: lambda per mode with the bootstrap CI as error bars and the null's 5-95
    percentile band; the non-heatwave control should sit *on* the null band. Right: the
    climatological eigenspectrum with the truncation k marked, which is what the choice of
    k is really about.
    """
    modes = np.arange(1, len(eigenvalues) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]

    ax.fill_between(modes, np.percentile(null, 5, axis=0), np.percentile(null, 95, axis=0),
                    color="0.75", alpha=0.6, label="null 5-95% (50 clim draws)")
    ax.plot(modes, np.percentile(null, 50, axis=0), color="0.35", lw=1.2, ls="-",
            label="null median")

    lo = eigenvalues - np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0) - eigenvalues
    ax.errorbar(modes, eigenvalues, yerr=np.vstack([np.abs(lo), np.abs(hi)]),
                fmt="o-", ms=4, lw=1.4, color="tab:red", capsize=2,
                label="heatwave (95% event bootstrap)")

    if eigenvalues_nonhw is not None:
        ax.plot(modes, eigenvalues_nonhw, "s--", ms=3.5, color="tab:blue",
                label="non-heatwave control")
    if eigenvalues_shrunk is not None:
        ax.plot(modes, eigenvalues_shrunk, "^:", ms=3.5, color="tab:orange",
                label="heatwave (Ledoit-Wolf)")

    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("mode (ascending lambda)")
    ax.set_ylabel(r"$\lambda$ = normalised variance")
    ax.set_title("Basis-independent normalised variance\n(<1 = constrained vs climatology)")
    ax.legend(fontsize=8)

    ax = axes[1]
    if clim_spectrum is not None:
        cum = np.cumsum(clim_spectrum) / clim_spectrum.sum()
        ax.semilogy(np.arange(1, len(clim_spectrum) + 1), clim_spectrum, "o-", ms=3,
                    color="tab:green", label=r"$\Sigma_{clim}$ eigenvalue")
        ax2 = ax.twinx()
        ax2.plot(np.arange(1, len(cum) + 1), cum, "-", color="0.4", lw=1)
        ax2.set_ylabel("cumulative explained fraction", color="0.4")
        ax2.set_ylim(0, 1.02)
        if k is not None:
            ax.axvline(k, color="tab:red", ls="--", lw=1,
                       label=f"k = {k} ({cum[k - 1]:.1%} of clim variance)")
        ax.set_xlabel("climatological PC")
        ax.set_ylabel("eigenvalue")
        ax.set_title("Climatological covariance spectrum")
        ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_probe_modes(
        eigenvalues: np.ndarray,
        weights: dict[str, np.ndarray],
        save_path: str,
        cosines: dict[str, np.ndarray] | None = None,
    ) -> None:
    """Where each physical probe's variance sits in the lambda spectrum.

    Bars are the mode weights ``a_i^2 / sum a_i^2`` (the exact decomposition of the
    probe's normalised variance), modes ordered by increasing lambda. Weight concentrated
    on the left = the probe lives in the constrained subspace. Plain cosine similarities
    are drawn underneath for reference.
    """
    modes = np.arange(1, len(eigenvalues) + 1)
    n_row = 2 if cosines else 1
    fig, axes = plt.subplots(n_row, 1, figsize=(10, 4.2 * n_row), squeeze=False)

    ax = axes[0, 0]
    width = 0.8 / len(weights)
    for i, (name, wt) in enumerate(weights.items()):
        ax.bar(modes + (i - (len(weights) - 1) / 2) * width, wt, width=width, label=name)
    ax2 = ax.twinx()
    ax2.plot(modes, eigenvalues, "k.--", lw=1, ms=5)
    ax2.set_yscale("log")
    ax2.axhline(1.0, color="k", ls=":", lw=0.8)
    ax2.set_ylabel(r"$\lambda$")
    ax.set_xlabel("mode (ascending lambda)")
    ax.set_ylabel(r"probe weight $a_i^2/\sum a_j^2$")
    ax.set_title("Probe decomposition over fluctuation modes")
    ax.legend(fontsize=9)

    if cosines:
        ax = axes[1, 0]
        for i, (name, cs) in enumerate(cosines.items()):
            ax.bar(modes + (i - (len(cosines) - 1) / 2) * width, cs, width=width, label=name)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel("mode (ascending lambda)")
        ax.set_ylabel(r"cosine$(w, v_i)$")
        ax.set_title("Plain (Euclidean) cosine similarity — shown for reference only")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_region_map(
        lon: np.ndarray,
        lat: np.ndarray,
        values: np.ndarray,
        save_path: str,
        label: str,
        title: str,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        extend: str = "max",
        marker_size: float = 60.0,
        box: tuple[float, float, float, float] | None = None,
    ) -> None:
    """Scatter one per-point field on the region.

    A scatter rather than a filled contour on purpose: the latent grids are coarse (19
    points at nside 16 are ~440 km cells) and interpolating between them would draw
    structure the data does not have.
    """
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    sc = ax.scatter(lon, lat, c=values, s=marker_size, cmap=cmap, vmin=vmin, vmax=vmax,
                    marker="s", edgecolors="none", transform=ccrs.PlateCarree())
    ax.coastlines("50m", color="0", linewidth=0.5)
    ax.gridlines(draw_labels=True, color="0.7", alpha=0.4, linewidth=0.3)
    if box is not None:
        lon_min, lon_max, lat_min, lat_max = box
        ax.plot([lon_min, lon_min, lon_max, lon_max, lon_min],
                [lat_min, lat_max, lat_max, lat_min, lat_min],
                color="lime", lw=1.5, transform=ccrs.PlateCarree())
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=label, extend=extend, shrink=0.8)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pushforward_validation(
        predicted: np.ndarray,
        measured: np.ndarray,
        r: float,
        save_path: str,
        r2_per_pixel: np.ndarray | None = None,
        predicted_oof: np.ndarray | None = None,
        r_oof: float | None = None,
        r2_trajectory: np.ndarray | None = None,
    ) -> None:
    """Predicted vs directly measured heatwave/non-heatwave variance ratio per pixel.

    The honesty check on the pushforward: the map is fitted to reproduce physical fields,
    so the variance ratio it predicts can be compared against the one measured straight
    off those fields. Points on the 1:1 line mean the pushed-forward variance is
    trustworthy; systematic compression toward 1 means the linear map is smoothing away
    ensemble spread. The out-of-fold series repeats the comparison using only held-out
    predictions, which is the version that is not partly self-fulfilling.

    The skill histogram carries both R²: at the ``(event, step)`` level the map is fitted
    at, and at the trajectory-mean level it is actually *used* at — the latter is much
    higher because averaging the trajectory cancels the fast residual.
    """
    n_panel = 2 if r2_per_pixel is not None else 1
    fig, axes = plt.subplots(1, n_panel, figsize=(11 if n_panel == 2 else 6, 5),
                             squeeze=False)
    ax = axes[0, 0]
    ax.scatter(measured, predicted, s=18, alpha=0.75, c="tab:red", edgecolors="k",
               linewidths=0.2, label=f"full-fit map (r = {r:.3f})")
    if predicted_oof is not None:
        ax.scatter(measured, predicted_oof, s=18, alpha=0.6, c="tab:blue",
                   edgecolors="k", linewidths=0.2,
                   label=f"out-of-fold (r = {r_oof:.3f})" if r_oof is not None
                   else "out-of-fold")
    hi = max(measured.max(), predicted.max())
    lo = min(measured.min(), predicted.min())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xlabel("measured var ratio (heatwave / non-heatwave)")
    ax.set_ylabel("pushforward-predicted var ratio")
    ax.set_title(f"Pushforward validation ({len(measured)} pixels)")
    ax.legend(fontsize=9)

    if r2_per_pixel is not None:
        ax = axes[0, 1]
        ax.hist(r2_per_pixel, bins=25, color="tab:blue", alpha=0.8,
                label=f"per (event, step): median {np.median(r2_per_pixel):.2f}")
        if r2_trajectory is not None:
            ax.hist(r2_trajectory, bins=25, color="tab:red", alpha=0.7,
                    label=f"per trajectory mean: median {np.median(r2_trajectory):.2f}")
        ax.set_xlabel("out-of-fold $R^2$ per pixel")
        ax.set_ylabel("pixels")
        ax.set_title("Pushforward skill")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_physical_comparison(
        lon: np.ndarray,
        lat: np.ndarray,
        latent_ratio: np.ndarray,
        physical_var_hat: np.ndarray,
        r_pearson: float,
        r_spearman: float,
        save_path: str,
        box: tuple[float, float, float, float] | None = None,
    ) -> None:
    """The stage's headline question, side by side.

    Left: the physical-space ``norm_var_hat`` (the figure that motivated the stage),
    restricted to the same pixels. Middle: the latent ensembles pushed forward to those
    pixels. Right: the two against each other. The absolute levels are *not* comparable —
    different data and different climatological normalisers — so the panels are on their
    own scales and the comparison is of spatial pattern.
    """
    fig = plt.figure(figsize=(16, 4.8))
    panels = [
        (physical_var_hat * 100.0, r"physical $\hat{V}$ [%]",
         "Physical space (reanalysis pipeline)"),
        (latent_ratio * 100.0, r"pushed-forward $\hat{V}$ [%]",
         "Latent pushforward (l2 encoder)"),
    ]
    for i, (values, label, title) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection=ccrs.PlateCarree())
        sc = ax.scatter(lon, lat, c=values, s=45, cmap="viridis", marker="s",
                        edgecolors="none", vmax=np.percentile(values, 98),
                        transform=ccrs.PlateCarree())
        ax.coastlines("50m", color="0", linewidth=0.5)
        ax.gridlines(draw_labels=False, color="0.7", alpha=0.4, linewidth=0.3)
        if box is not None:
            lon_min, lon_max, lat_min, lat_max = box
            ax.plot([lon_min, lon_min, lon_max, lon_max, lon_min],
                    [lat_min, lat_max, lat_max, lat_min, lat_min],
                    color="lime", lw=1.5, transform=ccrs.PlateCarree())
        ax.set_title(title, fontsize=10)
        fig.colorbar(sc, ax=ax, label=label, extend="max", shrink=0.75)

    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(physical_var_hat, latent_ratio, s=16, alpha=0.75, c="tab:red",
               edgecolors="k", linewidths=0.2)
    ax.set_xlabel(r"physical $\hat{V}$")
    ax.set_ylabel(r"latent pushforward $\hat{V}$")
    ax.set_title(f"Pearson r = {r_pearson:.2f}, Spearman = {r_spearman:.2f}\n"
                 f"({len(lon)} shared pixels)", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_k_scan(
        rows: list[dict], save_path: str, k_choice: int | None = None
    ) -> None:
    """lambda_min vs its null across the truncation scan, with the invariance error.

    The k choice is a tradeoff — more modes resolve more structure but the truncated
    subspace becomes more basis-dependent — so both sides of it are plotted together
    rather than the choice being made by taste.
    """
    ks = [r["k"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.fill_between(ks, [r["null_p05"] for r in rows], [r["null_p95"] for r in rows],
                    color="0.75", alpha=0.6, label="null 5-95%")
    ax.plot(ks, [r["null_p50"] for r in rows], color="0.35", lw=1.2, label="null median")
    ax.plot(ks, [r["lambda_min"] for r in rows], "o-", color="tab:red", label="heatwave")
    ax.plot(ks, [r["lambda_min_nonhw"] for r in rows], "s--", color="tab:blue",
            label="non-heatwave control")
    if k_choice is not None:
        ax.axvline(k_choice, color="k", ls=":", lw=1)
    ax.set_xlabel("k (climatological-PC truncation)")
    ax.set_ylabel(r"$\lambda_{min}$")
    ax.set_yscale("log")
    ax.set_title("Signal vs null across the truncation scan")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogy(ks, [max(r["invariance_general"], 1e-16) for r in rows], "o-",
                color="tab:purple", label="general invertible map")
    ax.semilogy(ks, [max(r["invariance_orthogonal"], 1e-16) for r in rows], "s-",
                color="tab:green", label="orthogonal map")
    if k_choice is not None:
        ax.axvline(k_choice, color="k", ls=":", lw=1)
    ax.set_xlabel("k (climatological-PC truncation)")
    ax.set_ylabel(r"max relative $\lambda$ shift")
    ax.set_title("Basis-invariance of the truncated spectrum")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
