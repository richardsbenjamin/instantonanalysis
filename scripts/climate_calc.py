from pathlib import Path

from instantonanalysis.instanton.utils import load_config, read_dataset
from instantonanalysis.utils.parsers import get_calc_args


if __name__ == "__main__":
    args = get_calc_args()
    cfg = load_config(args.config_name, args.overrides)

    data_root_path = Path(cfg.paths.data_root)
    time_dim = cfg.xconfig.time_dim
    vars_list = [var.name for _, var in cfg.variables.items()] 

    data = read_dataset(data_root_path / cfg.paths.data_file)[vars_list]

    for var_cfg in cfg.variables.values():
        data[var_cfg.name] = transform_data(data[var_cfg.name], var_cfg)

    data = data.resample({time_dim: "24h"}).mean()
    data.to_zarr(data_root_path / f"daily_{cfg.paths.data_file}", mode="w")

    mean_ds = data.mean(dim=time_dim)
    mean_ds.to_netcdf(data_root_path / f"climate_mean_{cfg.paths.data_file}")

    var_ds = data.var(dim=time_dim)
    var_ds.to_netcdf(data_root_path / f"climate_variance_{cfg.paths.data_file}")

