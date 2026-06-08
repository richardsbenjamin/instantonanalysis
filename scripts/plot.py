from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from hydra.utils import instantiate
from tqdm import tqdm

from instantonanalysis.instanton.schemas.box import (
    HealPixBox,
    LonLatBox, 
    LongitudeSystem, 
    LatitudeSystem,
)
from instantonanalysis.instanton.plotting import (
    plot_density_panel,
    plot_quantile_panels,
)
from instantonanalysis.instanton.utils import (
    filter_by_months,
    generate_panels,
    healpix_to_latlon,
    load_config,
    read_dataset,
)

if TYPE_CHECKING:
    from typing import Optional

    from instantonanalysis.instanton.schema import VariableConfig
    from instantonanalysis.instanton.schema.box import IBox
    from instantonanalysis.instanton.nbclosest import NClosestConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

YTICKS = [25, 45, 65]
XTICKS = [0, 30, 60, -30, -60]
PANEL_LABELS = ['(a)','(b)','(c)','(d)']
XTICK_LABELS = ['0°', '30°E', '60°E', '30°W', '60°W']
YTICK_LABELS = ['25°N', '45°N', '65°N']


def load_data(data_root: str, locations: list[str], filenames: list[str]) -> dict:
    return {
        location: {
            filename: xr.open_dataset(f"{data_root}/{location}/{filename}.nc")
            for filename in filenames
        }
        for location in locations
    }

def plot_densities(
        var_cfg: VariableConfig, 
        data_in: dict, 
        locations: dict[str, str], 
        rolling_periods_tab: list[int], 
        select_months: list[int], 
        time_dim: str,
        quantile_dim: str,
        results_path: str,
    ) -> None:
    var = var_cfg.name

    rows, cols = len(locations), len(rolling_periods_tab)
    panels = generate_panels(rows, cols)

    plt.rc('font', family='serif', size=20)
    fig, axes = plt.subplots(
        nrows=rows, 
        ncols=cols, 
        figsize=(20, 10), 
        sharex=True, 
        sharey=True, 
        constrained_layout=True
    )
    axes = axes.flatten()
    for i, location in enumerate(locations):
        for j, r in enumerate(rolling_periods_tab):
            rolling = data_in[location]["series_obs"][var].rolling(**{time_dim: r, "center": True}).mean()
            data_slice = filter_by_months(rolling, time_dim, select_months)
            
            neighbors_celsius = data_in[location]["closest_neighbours"][var].sel(rolling_period=r) 
            
            is_last_row = (i == rows - 1)
            is_first_col = (j == 0)
            is_legend_cell = (i == 3 and j == 2)
            
            plot_density_panel(
                ax=axes[i * cols + j],
                data=data_slice,
                neighbor_ranges=neighbors_celsius,
                levels=quantiles,
                quantile_dim=quantile_dim,
                title=f"{panels[i][j]} {location.replace('/', '').title()} r={r}",
                xlabel=f"{var_cfg.long_name} [{var_cfg.unit}]" if is_last_row else "",
                ylabel="Density" if is_first_col else "",
                show_legend=is_legend_cell,
            )
    plt.savefig(f"{results_path}/{location}/{var}_density_plot.png", dpi=300, bbox_inches='tight')

def plot_quantile_data(
        data_in: dict, 
        plot_data: str,
        locations: dict[str, str], 
        rolling_periods_tab: list[int], 
        select_months: list[int], 
        time_dim: str,
        quantile_dim: str,
        extract_box: IBox,
        results_path: str,
        cb_label: str,
        levels_cf: Optional[list[float]] = None,
        levels_c: Optional[list[float]] = None,
        cmap: str = "RdBu_r",
        extend: str = "both",
    ) -> None:
    for location, data_dict in data_in.items():
        composites = extract_box.extract(data_dict[plot_data])
        
        for r in rolling_periods_tab:
            fig = plt.figure(figsize=(27,10))
            gs = fig.add_gridspec((len(composites[quantile_dim])-1)//2+1,2)
            lon_lat_box = instantiate(next(
                location_obj.box for location_obj in locations.values() if location_obj.output_folder == location
            ))
            contour_in = composites.sel({"rolling_period": r})
            fig = plot_quantile_panels(
                quantile_dim=quantile_dim, 
                contourf_data=contour_in["t2m0"], 
                contour_data=contour_in["z500"] / 9.80665, 
                box=lon_lat_box, 
                title_func=title_func, 
                label=cb_label, 
                levels_cf=levels_cf,
                levels_c=levels_c, 
                xticks=XTICKS, 
                yticks=YTICKS, 
                xticklabels_func=xticklabels_func, 
                yticklabels_func=yticklabels_func,
                cmap=cmap, 
                extend=extend,
            )
            plt.savefig(f"{results_path}/{location}/composite_anomalies_r{r}.png", dpi=300, bbox_inches='tight')


na_box = LonLatBox(
    lon_min=-80,
    lon_max=50,
    lat_min=22.5,
    lat_max=70,
    lon_system=LongitudeSystem.EAST_WEST,
    lat_system=LatitudeSystem.SOUTH_NORTH,
)


if __name__ == "__main__":
    logger.info("Loading config")
    cfg = load_config("plot_config")    
      
    filenames = cfg.paths.filenames
    locations = cfg.locations
    location_folders = [location.output_folder for location in locations.values()]
    quantiles = cfg.analysis.quantiles
    rolling_periods_tab = cfg.analysis.rolling_periods
    select_months = cfg.analysis.select_months
    var_names = [var.name for var in cfg.variables.values()]

    xconfig = instantiate(cfg.xconfig)
    time_dim = xconfig.time_dim
    quantile_dim = xconfig.quantile

    logger.info("Loading data")
    data_in = load_data(cfg.paths.results_root, location_folders, filenames)

    logger.info("Plotting density panels")
    var_cfg = cfg.variables["t2m"]
    plot_densities(
        var_cfg,
        data_in,
        location_folders,
        rolling_periods_tab,
        select_months,
        time_dim,
        quantile_dim,
        cfg.paths.results_root,
    )
    
    xticklabels_func = lambda i: XTICK_LABELS if i // 2 == 1  else ""
    yticklabels_func = lambda i: YTICK_LABELS if i % 2 == 0 else ""
    title_func = lambda i, q: fr"({chr(ord('a') + i)}) $\alpha = {q}$"

    data_in_ll = {}
    for location, data_dict in data_in.items():
        data_in_ll[location] = {}
        for data_type in cfg.paths.converted_filenames:
            data_hp = data_dict[data_type]
            if data_type == "normalised_var_hat":
                chi_mask = data_dict["chi_masks"]
                chi_maxes = data_hp.where(chi_mask).max()
                data_hp = data_hp.where(chi_mask, chi_maxes)

            data_in_ll[location][data_type] = healpix_to_latlon(
                data_hp,
                spatial_dims=xconfig.spatial_dims,
                var=var_names,
            )

    logger.info("Plotting composite anomalies")
    plot_quantile_data(
        data_in=data_in_ll,
        plot_data="composite_anomalies",
        locations=locations,
        rolling_periods_tab=rolling_periods_tab,
        select_months=select_months,
        time_dim=time_dim,
        quantile_dim=quantile_dim,
        extract_box=na_box,
        results_path=cfg.paths.results_root,
        cb_label="Anomaly of T2M [°C]",
        levels_cf=cfg.variables["t2m"].contour_levels.composite,
    )

    logger.info("Plotting normalised var hat")
    for var_cfg in cfg.variables.values():
        for location, data_dict in data_in_ll.items():
            composites = na_box.extract(data_dict["composite_anomalies"][var_cfg.name]) / var_cfg.scale
            var_hats = na_box.extract(data_dict["normalised_var_hat"][var_cfg.name]) * 100
            
            for r in rolling_periods_tab:
                fig = plt.figure(figsize=(27,10))
                gs = fig.add_gridspec((len(composites[quantile_dim])-1)//2+1,2)
                lon_lat_box = instantiate(next(
                    location_obj.box for location_obj in locations.values() if location_obj.output_folder == location
                ))
                fig = plot_quantile_panels(
                    quantile_dim=quantile_dim, 
                    contourf_data=var_hats.sel({"rolling_period": r}), 
                    contour_data=composites.sel({"rolling_period": r}), 
                    box=lon_lat_box, 
                    title_func=title_func, 
                    label=r"$\hat{V}$ [%]",
                    xticks=XTICKS, 
                    yticks=YTICKS, 
                    xticklabels_func=xticklabels_func, 
                    yticklabels_func=yticklabels_func,
                    cmap="viridis", 
                    extend="max",
                    alpha=0.8,
                    levels_cf=var_cfg.contour_levels.var_hat,
                )
                plt.savefig(f"{cfg.paths.results_root}/{location}/var_hat_{var_cfg.name}_r{r}.png", dpi=300, bbox_inches='tight')

