import logging
from typing import TYPE_CHECKING

from cartopy import crs as ccrs

from instantonanalysis.instanton.analysis import (
    get_climate_anomaly,
    get_mean,
    get_var,
    select_climato,
)
from instantonanalysis.instanton.lonlat import get_lon_lat_box
from instantonanalysis.instanton.utils import (
    build_event_cube,
    load_config,
    read_dataset,
)

if TYPE_CHECKING:
    from typing import Callable
    import xarray as xr


def get_area_configs(lonlat_box: LonLatBox) -> dict:
    return {
        "north-atlantic": {
            "transform": lonlat_box.extract,
            "projection": ccrs.PlateCarree(),
        },
        "north-hemisphere": {
            "transform": lonlat_box.add_cyclic_point,
            "projection": ccrs.NorthPolarStereo(central_longitude=0),
        },  
    }

def get_climate_data(
        event_set: xr.Dataset,
        climate_mean: xr.Dataset,
        climate_var: xr.Dataset,
        dim: str = "time",
    ) -> dict:
    mean = get_mean(event_set, dim)
    climate_anomaly = get_climate_anomaly(mean, climate_mean, climate_var)
    var = get_var(event_set, dim)
    return {
        "mean": mean,
        "mean_anomaly": climate_anomaly,
        "var": var,
    }

def get_contour_labels(var_cfg: VariableConfig):
    return {
        'mean': f"[{var_cfg.unit}]",
        'mean_anomaly': "[STD]",
        'var': "%",
    }

def get_contour_levels(var_cfg: VariableConfig):
    return {
        'mean': var_cfg.levels,
        'mean_anomaly': np.arange(-1,1.1,0.1),
        'var': np.arange(30,130,10),
    }

def plot_variance_evolution_grid(
        var_cfg: VariableConfig, 
        normalised_var: xr.Dataset,
        box: LonLatBox, 
        region_name: str,
    ) -> plt.Figure:
    rolling_period = normalised_var.rolling_period.values
    quantiles = normalised_var.quantile.values
    lags = normalised_var.lag.values

    fig = plt.figure(figsize=(20, 10))
    plt.tight_layout(pad=3.0)
    gs = fig.add_gridspec((len(rolling_period) - 1) // 3 + 1, 3)
    
    for idr, r in enumerate(rolling_period):
        ax = fig.add_subplot(gs[idr // 3, idr % 3])
        for idl, q in enumerate(quantiles):
            mean_variance = []
            for j in lags:
                temp = normalised_var.sel(rolling_period=r, quantile=q, lag=j)
                val = get_weighted_spatial_mean(temp, box)
                mean_variance.append(val)
            
            ax.plot(j_list, mean_variance, color=colors[idl % len(colors)], label=f"q = {level}")
            ax.scatter(j_list, mean_variance, s=4, color=colors[idl % len(colors)])

        ax.set_title(f"r = {r} days")
        ax.set_xlabel("Days relative to maximum")
        ax.set_ylabel(f"{region_name} normalized variance")
        ax.set_xlim(j_list[0], j_list[-1])
        ax.set_ylim(0.3, 1.2)
        ax.legend(prop={'size': 8})

    return fig


def save_results(results: xr.Dataset, config: NClosestConfig, output_dir: str) -> None:
    plot_autocorrelation(
        auto_corr_series=results.autocorrelation,
        rolling_periods_tab=config.rolling_periods,
        output_filename=f"{output_dir}/autocorrelation.png",
    )
    plot_histograms(
        closest_neighbors_list=results.closest_neighbors,
        quantile_tab=results.quantile_threshold,
        rolling_periods_tab=config.rolling_periods,
        series_obs=results.series_obs,
        quantiles=config.quantiles,
        time_dim=config.time_dim,
        output_filename=f"{output_dir}/histograms.png",
    )
    plot_histograms_dates(
        closest_neighbors_list=results.closest_neighbors,
        rolling_periods_tab=config.rolling_periods,
        quantiles=config.quantiles,
        time_dim=config.time_dim,
        output_filename=f"{output_dir}/histograms_dates.png",
    )



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Loading config")
    cfg = load_config()
    output_dir = cfg.paths.results_root + cfg.locations.output_folder

    lon_lat_box = get_lon_lat_box(cfg.locations)
    var_cfg = VariableConfig(cfg.variables)
    xconfig = cfg.xconfig

    contour_levels = get_contour_levels(var_cfg)
    contour_labels = get_contour_labels(var_cfg)
    area_configs = get_area_configs(lon_lat_box)

    mean_dims = [xconfig.time_dim, "lag"]

    logger.info("Loading datasets")
    dataset = read_dataset(cfg.paths.input_data)
    climate_mean = read_dataset(cfg.paths.climate_mean)
    climate_var = read_dataset(cfg.paths.climate_var)

    logger.info("Plotting dataset")

    for area_name, area_cfg in area_configs.items():
        transform = area_cfg['transform']
        event_select = transform(event_cube)
        climate_mean = transform(climate_mean)
        climate_var = transform(climate_var)

        # plot quantile panels
        for dim in mean_dims:            
            spatial_data = get_climate_data(
                event_select, climate_mean, climate_var, dim,
            )
            for r in event_cube.rolling_period.values:
                for j in j_list:
                    for metric, metric_data in spatial_data.items():
                        fig = plot_quantile_panels(
                            metric_data.sel(rolling_period=r, lag=j),
                            lon_lat_box,
                            f"{v_cfg} r={r}, j={j}",
                            contour_levels[metric],
                            contour_labels[metric],
                            projection=area_cfg['projection'],                    
                        )
                        fig.savefig(f"{output_dir}/{area_name}_j{j}_{metric}.png")
                        plt.close()

        # plot variance evolution
        variance_evolution = plot_variance_evolution_grid(
            var_cfg,
            normalised_var,
            lon_lat_box,
            area_name,
        )
        variance_evolution.savefig(f"{output_dir}/{area_name}_variance_evolution.png")
        plt.close()
    
    logger.info("Saving results")
