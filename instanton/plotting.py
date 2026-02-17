from __future__ import annotations

from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from typing import Iterable, Tuple, List

    from cartopy import crs as ccrs
    import xarray as xr

DEFAULT_ADD_AXES = [0.935, 0.05, 0.02, 0.9]
DEFAULT_COLOURS = ['gold','darkorange', 'red', 'black']
KELVIN_OFFSET = 273.15
DEFAULT_BIN_EDGES = np.arange(152, 248, 5) - 0.5
DEFAULT_DEC_OFFSETS = [0, 0.01, 0.02, 0.03]
DEFAULT_PLOT_X = np.arange(152, 243, 5)
DEFAULT_SUBPLOTS_ADJUST = {
    "top": 0.92,
    "bottom": 0.05,
    "left": 0.05,
    "right": 0.95,
    "hspace": 0.2,
    "wspace": 0.1
}
HIST_KWARGS = {
    "bins": 100,
    "histtype": "step",
    "density": True,
    "color": "black",
}
LEGEND_KWARGS = {
    "loc": "lower right",
    "prop": {"size": 13},
}


def plot_autocorrelation(
        auto_corr_series: np.ndarray, 
        rolling_periods_tab: Iterable,
        figsize: Tuple[int, int] = (20, 10),
        xlim: Tuple[int, int] = (0, 30),
        ylim: Tuple[int, int] = (0, 1),
        output_filename: str = "autocorrelation.png",
    ) -> None:
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    
    for r, ac_series in zip(rolling_periods_tab, auto_corr_series):
        plt.plot(np.arange(0,31,1), ac_series, label=f"rolling period = {r}")
        
    plt.legend()
    plt.xlabel("Days")
    plt.ylabel("Autocorrelation")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    
    plt.savefig(output_filename)
    plt.close()

def plot_density_panel(
    ax: matplotlib.axes.Axes,
    data: xr.DataArray,
    neighbor_ranges: xr.DataArray,
    levels: List[float],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    offsets: List[float] = DEFAULT_DEC_OFFSETS,
    colours: List[str] = DEFAULT_COLOURS,
    show_legend: bool = False,
    jitter: float = 0.05,
    linewidth: float = 2,
    xlim: Tuple[float, float] = (10, 30),
    ylim: Tuple[float, float] = (0, 0.2),
    hist_kwargs: Optional[Dict[str, Any]] = HIST_KWARGS,
    legend_kwargs: Optional[Dict[str, Any]] = LEGEND_KWARGS,
) -> None:
    ax.hist(data, **hist_kwargs)
    for (lvl, col, dec) in zip(levels, colours, offsets):
        q_val = float(data.quantile(q=lvl))
        ax.plot([q_val, q_val], [ylim[0], ylim[1]], color=col, label=rf"$\alpha = {lvl}$")
        
        q_sel = neighbor_ranges.sel(quantiles=lvl)
        n_min = float(q_sel.min())
        n_max = float(q_sel.max())
        ax.plot([n_min, n_max], [jitter + dec, jitter + dec], color=col, linewidth=linewidth)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if show_legend:
        ax.legend(**legend_kwargs)

def plot_quantile_panels(
    quantile_dim: str,
    contourf_data: xr.DataArray,
    contour_data: xr.DataArray,
    box: LonLatBox,
    title_func: Callable,
    levels_cf: np.ndarray,
    levels_c: np.ndarray,
    label: str,
    xticks: np.ndarray,
    yticks: np.ndarray,
    xticklabels_func: Callable,
    yticklabels_func: Callable,
    cmap: str = "RdBu_r",
    extend: str = "both",
    alpha: float = 1,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    transform: ccrs.Projection = ccrs.PlateCarree(),
    ncols: int = 2,
    figsize: tuple = (27, 10),
    subplots_adjust: dict = DEFAULT_SUBPLOTS_ADJUST,
    add_axes: list[float] = DEFAULT_ADD_AXES,
) -> plt.Figure:
    quantiles = getattr(contour_data, quantile_dim).values
    nrows = (len(quantiles) - 1) // ncols + 1
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols)

    if not all(contourf_data['lon'].values == contour_data['lon'].values):
        raise ValueError("Longitude values do not match")
    if not all(contourf_data['lat'].values == contour_data['lat'].values):
        raise ValueError("Latitude values do not match")

    lon, lat = contour_data['lon'], contour_data['lat']

    for i, q in enumerate(quantiles):
        ax = fig.add_subplot(gs[i // ncols, i % ncols], projection=projection)
        
        cp = ax.contourf(
            lon, lat, contourf_data.sel({quantile_dim: q}),
            levels=levels_cf, extend=extend, cmap=cmap, transform=transform, alpha=alpha
        )
        cl = ax.contour(
            lon, lat, contour_data.sel({quantile_dim: q}),
            levels=levels_c, extend="both", colors='black', transform=transform
        )
        ax.clabel(cl, cl.levels, inline=True, fontsize=15)
        
        ax.coastlines('50m', color='0', linewidth=.4)
        ax.gridlines(draw_labels=False, xlocs=xticks, ylocs=yticks, color='.7', alpha=0.4, linewidth=.3)
        ax.set_xticks(xticks, crs=transform)
        ax.set_yticks(yticks, crs=transform)
        ax.set_xticklabels(xticklabels_func(i))
        ax.set_yticklabels(yticklabels_func(i))
        ax.set_title(title_func(i, q))
        
        ax.plot([box.lon_min, box.lon_min, box.lon_max, box.lon_max, box.lon_min], 
                [box.lat_min, box.lat_max, box.lat_max, box.lat_min, box.lat_min], 
                color='lime', linewidth=2, transform=transform)

    if add_axes:
        cbar_ax = fig.add_axes(add_axes)
    fig.colorbar(cp, label=label, spacing='proportional', cax=cbar_ax)
    fig.subplots_adjust(**subplots_adjust)
    return fig