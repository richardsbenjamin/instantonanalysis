from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from hydra.utils import instantiate

from instantonanalysis.hydra_logic.resolvers import resolve_latlon_from_cfg

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Iterable, Optional, Tuple, List

    from cartopy import crs as ccrs

    from instantonanalysis.instanton.schemas.box import IBox

logger = logging.getLogger(__name__)

DEFAULT_ADD_AXES = [0.91, 0.05, 0.02, 0.9]
DEFAULT_COLOURS = ['gold','darkorange', 'red', 'black']
KELVIN_OFFSET = 273.15
DEFAULT_BIN_EDGES = np.arange(152, 248, 5) - 0.5
DEFAULT_DEC_OFFSETS = [0, 0.01, 0.02, 0.03]
DEFAULT_PLOT_X = np.arange(152, 243, 5)
DEFAULT_SUBPLOTS_ADJUST = {
    "top": 0.92,
    "bottom": 0.05,
    "left": 0.05,
    "right": 0.9,
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


def get_location(locations: Dict, output_folder: str):
    """The location config whose ``output_folder`` matches ``output_folder``."""
    return next(
        location_obj for location_obj in locations.values()
        if location_obj.output_folder == output_folder
    )

def latlon_box(cfg_box) -> IBox:
    """Build a LonLatBox from a location config's ``box`` block."""
    return instantiate(resolve_latlon_from_cfg(cfg_box))

def make_ticklabel_funcs(domain_cfg) -> Tuple[Callable, Callable]:
    """Panel-position-aware tick labellers for one location's domain.

    Labels are drawn on the bottom row and the left column only.
    """
    xtick_labels = list(domain_cfg.xtick_labels)
    ytick_labels = list(domain_cfg.ytick_labels)
    return (
        lambda i: xtick_labels if i // 2 == 1 else "",
        lambda i: ytick_labels if i % 2 == 0 else "",
    )

def load_location_data(
        data_root: str,
        locations: List[str],
        filenames: List[str],
    ) -> Dict[str, Dict[str, xr.Dataset]]:
    """Open each location's outputs, skipping locations that have not been run."""
    data_in = {}
    for location in locations:
        folder = Path(f"{data_root}/{location}")
        missing = [f for f in filenames if not (folder / f"{f}.nc").exists()]
        if missing:
            logger.warning(
                f"Skipping {location}: no {', '.join(missing)} in {folder} "
                "(has the analysis been run for this location?)"
            )
            continue
        data_in[location] = {
            filename: xr.open_dataset(folder / f"{filename}.nc")
            for filename in filenames
        }
    return data_in

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
    quantile_dim: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    offsets: List[float] = DEFAULT_DEC_OFFSETS,
    colours: List[str] = DEFAULT_COLOURS,
    show_legend: bool = False,
    jitter: float = 0.05,
    linewidth: float = 2,
    xlim: Tuple[float, float] = (10, 30),
    hist_kwargs: Optional[Dict[str, Any]] = HIST_KWARGS,
    legend_kwargs: Optional[Dict[str, Any]] = LEGEND_KWARGS,
) -> None:
    ax.hist(data, **hist_kwargs)
    for (lvl, col, dec) in zip(levels, colours, offsets):
        q_val = float(data.quantile(q=lvl))
        ax.axvline(x=q_val, color=col, label=rf"$\alpha = {lvl}$")
        
        q_sel = neighbor_ranges.sel(**{quantile_dim: lvl})
        n_min = float(q_sel.min())
        n_max = float(q_sel.max())
        ax.plot([n_min, n_max], [jitter + dec, jitter + dec], color=col, linewidth=linewidth)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    
    if show_legend:
        ax.legend(**legend_kwargs)

def plot_quantile_panels(
        quantile_dim: str,
        contourf_data: xr.DataArray,
        contour_data: xr.DataArray,
        box: IBox,
        title_func: Callable,
        label: str,
        xticks: np.ndarray,
        yticks: np.ndarray,
        xticklabels_func: Callable,
        yticklabels_func: Callable,
        levels_cf: Optional[np.ndarray] = None,
        levels_c: Optional[np.ndarray] = None,
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
    quantiles = contour_data[quantile_dim].values
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