from __future__ import annotations

from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from typing import Iterable, Tuple, List

    from cartopy import crs as ccrs
    import xarray as xr

COLOURS = ['midnightblue','blue','magenta', 'green', 'orange', 'red']
KELVIN_OFFSET = 273.15
DEFAULT_BIN_EDGES = np.arange(152, 248, 5) - 0.5
DEFAULT_DEC_OFFSETS = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
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


def plot_histograms(
    closest_neighbors_list: Iterable[float], 
    quantile_tab: Iterable[float], 
    rolling_periods_tab: Iterable[int], 
    series_obs: np.ndarray, 
    quantiles: List[float],
    time_dim: str,
    figsize: Tuple[int, int] = (20, 10),
    bins: int = 100,
    xlim: Tuple[float, float] = (10, 35),
    ylim: Tuple[float, float] = (0, 0.25),
    v_line_max: float = 0.15,
    h_line_base: float = 0.05,
    month_range: Tuple[int, int] = (6, 8),
    cols: int = 3,
    output_filename: str = "histograms.png",
) -> None:
    num_plots = len(closest_neighbors_list)
    rows = (num_plots - 1) // cols + 1
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(rows, cols)
    
    for i in range(num_plots):
        month_mask = (series_obs[f'{time_dim}.month'] >= month_range[0]) & (series_obs[f'{time_dim}.month'] <= month_range[1])
        series_obs_rolling = (
            series_obs.rolling({time_dim: rolling_periods_tab[i]}, center=True)
            .mean()
            .sel({time_dim: month_mask})
        )

        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.hist(series_obs_rolling - KELVIN_OFFSET, bins=bins, histtype='step', density=True)
        
        for idj, j in enumerate(quantiles):
            q_val = quantile_tab[i, idj] - KELVIN_OFFSET
            color = COLOURS[idj]
            
            ax.plot([q_val, q_val], [0, v_line_max], color=color, label=f"q = {j}")
            
            neighbor_min = closest_neighbors_list[i][idj].min() - KELVIN_OFFSET
            neighbor_max = closest_neighbors_list[i][idj].max() - KELVIN_OFFSET
            h_level = h_line_base + DEFAULT_DEC_OFFSETS[idj]
            
            ax.plot([neighbor_min, neighbor_max], [h_level, h_level], color=color, linewidth=2)
            
        ax.set_title(f"r = {rolling_periods_tab[i]} days")
        ax.set_xlabel("Temperature [°C]")
        ax.set_ylabel("Density")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def plot_histograms_dates(
    closest_neighbors_list: List[xr.DataArray], 
    rolling_periods_tab: List[int], 
    quantiles: List[float], 
    time_dim: str,
    output_filename: str,
    figsize: Tuple[int, int] = (20, 10),
    cols: int = 3,
    xlim: Tuple[int, int] = (152, 242),
    ylim: Tuple[int, int] = (0, 15),
    colors: List[str] = COLOURS,
) -> None:
    num_plots = len(closest_neighbors_list)
    rows = (num_plots - 1) // cols + 1
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(rows, cols)
    
    for i in range(num_plots):
        ax = fig.add_subplot(gs[i // cols, i % cols])
                
        for idj, j in enumerate(quantiles):
            if idj % 2 == 0:
                day_of_year = closest_neighbors_list[i][idj][time_dim].dt.dayofyear
                counts, _ = np.histogram(day_of_year, bins=DEFAULT_BIN_EDGES)
                
                ax.plot(
                    DEFAULT_PLOT_X, 
                    counts, 
                    color=colors[idj], 
                    label=f"q = {j}"
                )
        
        ax.set_title(f"r = {rolling_periods_tab[i]} days")
        ax.set_xlabel("Calendar days")
        ax.set_ylabel("Density")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def plot_density_panel(
    ax: matplotlib.axes.Axes,
    data: xr.DataArray,
    neighbor_ranges: xr.DataArray,
    levels: List[float],
    colors: List[str],
    offsets: List[float],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    show_legend: bool = False,
    jitter: float = 0.05,
    linewidth: float = 2,
    xlim: Tuple[float, float] = (10, 30),
    ylim: Tuple[float, float] = (0, 0.2),
    hist_kwargs: Optional[Dict[str, Any]] = HIST_KWARGS,
    legend_kwargs: Optional[Dict[str, Any]] = LEGEND_KWARGS,
) -> None:
    ax.hist(data, **hist_kwargs)
    for (lvl, col, dec) in zip(levels, colors, offsets):
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
        data: xr.DataArray, 
        box: LonLatBox,
        title_base: str,
        levels: np.ndarray,
        label: str,
        projection: ccrs.Projection = ccrs.PlateCarree(),
        transform: ccrs.Projection = ccrs.PlateCarree(),
        ncols: int = 2,
        figsize: tuple = (20, 10),
        subplots_adjust: dict = DEFAULT_SUBPLOTS_ADJUST
    ) -> plt.Figure:
    quantiles = data.quantile.values
    nrows = (len(quantiles) - 1) // ncols + 1
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols)
    plt.subplots_adjust(**subplots_adjust)

    for i, q in enumerate(quantiles):
        ax = fig.add_subplot(gs[i // ncols, i % ncols], projection=projection)
        
        p = data.sel(quantile=q).plot.contourf(
            ax=ax, transform=transform, levels=levels, 
            cmap='coolwarm', add_labels=False, add_colorbar=False, extend='both'
        )
        plt.colorbar(p, ax=ax, label=label)
        ax.coastlines()
        ax.set_title(f"{title_base} q={q}")
        
        ax.plot([box.lon_min, box.lon_min, box.lon_max, box.lon_max, box.lon_min], 
                [box.lat_min, box.lat_max, box.lat_max, box.lat_min, box.lat_min], 
                color='lime', linewidth=2, transform=transform)

    return fig

