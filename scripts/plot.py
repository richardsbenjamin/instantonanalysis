from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from hydra.utils import instantiate

from instantonanalysis.instanton.plotting import (
    latlon_box,
    load_outputs,
    make_ticklabel_funcs,
    plot_density_panel,
    plot_quantile_panels,
)
from instantonanalysis.instanton.utils import (
    filter_by_months,
    generate_panels,
    healpix_to_latlon,
    load_config,
)
from instantonanalysis.instanton.utils.parsers import get_calc_args

if TYPE_CHECKING:
    from typing import Callable, Optional

    from omegaconf import DictConfig
    from xarray import DataArray, Dataset

    from instantonanalysis.instanton.schemas import VariableConfig, XConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_densities(
        var_cfg: VariableConfig,
        data_in: dict[str, Dataset],
        location: DictConfig,
        rolling_periods: list[int],
        select_months: list[int],
        quantiles: list[float],
        xconfig: XConfig,
        output_path: Path,
    ) -> None:
    var = var_cfg.name
    cols = len(rolling_periods)
    panels = generate_panels(1, cols)[0]

    plt.rc('font', family='serif', size=20)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=cols,
        figsize=(10 * cols, 10),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[0]

    for j, r in enumerate(rolling_periods):
        rolling = data_in["series_obs"][var].rolling(
            **{xconfig.time_dim: r, "center": True}
        ).mean()
        data_slice = filter_by_months(
            rolling, xconfig.time_dim, select_months
        ).dropna(xconfig.time_dim)

        neighbours = data_in["closest_neighbours"][var].sel({xconfig.rolling_period: r})

        plot_density_panel(
            ax=axes[j],
            data=data_slice,
            neighbour_ranges=neighbours,
            levels=quantiles,
            quantile_dim=xconfig.quantile,
            title=f"{panels[j]} {location.name.title()} r={r}",
            xlabel=f"{var_cfg.long_name} [{var_cfg.unit}]",
            ylabel="Density" if j == 0 else "",
            xlim=tuple(location.plot.density_xlim),
            show_legend=(j == cols - 1),
        )

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_quantile_maps(
        contourf_data: DataArray,
        contour_data: DataArray,
        location: DictConfig,
        xconfig: XConfig,
        rolling_periods: list[int],
        title_func: Callable,
        label: str,
        output_dir: Path,
        filename_template: str,
        levels_cf: Optional[list[float]] = None,
        levels_c: Optional[list[float]] = None,
        cmap: str = "RdBu_r",
        extend: str = "both",
        alpha: float = 1.0,
    ) -> None:
    domain = location.domain
    lon_lat_box = latlon_box(location.box)
    xticklabels_func, yticklabels_func = make_ticklabel_funcs(domain)

    for r in rolling_periods:
        fig = plot_quantile_panels(
            quantile_dim=xconfig.quantile,
            contourf_data=contourf_data.sel({xconfig.rolling_period: r}),
            contour_data=contour_data.sel({xconfig.rolling_period: r}),
            box=lon_lat_box,
            title_func=title_func,
            label=label,
            levels_cf=levels_cf,
            levels_c=levels_c,
            xticks=list(domain.xticks),
            yticks=list(domain.yticks),
            xticklabels_func=xticklabels_func,
            yticklabels_func=yticklabels_func,
            cmap=cmap,
            extend=extend,
            alpha=alpha,
        )
        fig.savefig(
            output_dir / filename_template.format(r=r), dpi=300, bbox_inches='tight'
        )
        plt.close(fig)


if __name__ == "__main__":
    logger.info("Loading config")
    args = get_calc_args()
    cfg = load_config(args.config_name, args.overrides)

    location = cfg.locations
    output_dir = Path(cfg.paths.results_root) / location.output_folder

    quantiles = list(cfg.analysis.quantiles)
    rolling_periods = list(cfg.analysis.rolling_periods)
    select_months = list(cfg.analysis.calc_months)

    xconfig = instantiate(cfg.xconfig)
    t2m_cfg = cfg.variables["t2m"]
    z500_cfg = cfg.variables["z500"]
    var_names = [var_cfg.name for var_cfg in cfg.variables.values()]

    title_func = lambda i, q: fr"({chr(ord('a') + i)}) $\alpha = {q}$"

    logger.info(f"Loading {location.name} outputs from {output_dir}")
    data_in = load_outputs(
        output_dir,
        {
            key: cfg.paths[key]
            for key in (
                "series_obs",
                "closest_neighbours",
                "composite_anomalies",
                "normalised_var_hat",
                "chi_masks",
            )
        },
    )

    logger.info("Plotting density panels")
    plot_densities(
        var_cfg=t2m_cfg,
        data_in=data_in,
        location=location,
        rolling_periods=rolling_periods,
        select_months=select_months,
        quantiles=quantiles,
        xconfig=xconfig,
        output_path=output_dir / cfg.paths.density_plot.format(var=t2m_cfg.name),
    )

    logger.info("Reprojecting HEALPix fields to lon/lat")
    data_ll = {}
    for data_type in cfg.paths.remap_keys:
        data_hp = data_in[data_type]
        if data_type == "normalised_var_hat":
            chi_mask = data_in["chi_masks"]
            chi_maxes = data_hp.where(chi_mask).max()
            data_hp = data_hp.where(chi_mask, chi_maxes)

        data_ll[data_type] = healpix_to_latlon(
            data_hp,
            spatial_dims=xconfig.spatial_dims,
            var=var_names,
            nside=cfg.nside,
        )

    domain_box = latlon_box(location.domain.box)
    composites = domain_box.extract(data_ll["composite_anomalies"])
    var_hats = domain_box.extract(data_ll["normalised_var_hat"])

    logger.info("Plotting composite anomalies")
    plot_quantile_maps(
        contourf_data=composites[t2m_cfg.name],
        contour_data=composites[z500_cfg.name] / z500_cfg.scale,
        location=location,
        xconfig=xconfig,
        rolling_periods=rolling_periods,
        title_func=title_func,
        label=f"Anomaly of {t2m_cfg.alias.upper()} [{t2m_cfg.unit}]",
        output_dir=output_dir,
        filename_template=cfg.paths.composite_plot,
        levels_cf=list(t2m_cfg.contour_levels.composite),
    )

    logger.info("Plotting normalised var hat")
    for var_cfg in cfg.variables.values():
        plot_quantile_maps(
            contourf_data=var_hats[var_cfg.name] * 100,
            contour_data=composites[var_cfg.name] / var_cfg.scale,
            location=location,
            xconfig=xconfig,
            rolling_periods=rolling_periods,
            title_func=title_func,
            label=r"$\hat{V}$ [%]",
            output_dir=output_dir,
            filename_template=cfg.paths.var_hat_plot.format(
                var=var_cfg.name, r="{r}"
            ),
            levels_cf=list(var_cfg.contour_levels.var_hat),
            cmap="viridis",
            extend="max",
            alpha=0.8,
        )

    for dataset in data_in.values():
        dataset.close()
