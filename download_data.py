import xarray as xr


def read_edh(edh_path: str) -> xr.Dataset:
    return xr.open_dataset(
        edh_path,
        storage_options={"client_kwargs":{"trust_env":True}},
        chunks={"time": 1},
        engine="zarr",
    )

singles_path = "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr"
era5_ds = read_edh(singles_path)

era5_ds = era5_ds.sel(valid_time=slice("1940-01-01", "1945-01-01"))
res = era5_ds["t2m"].resample({"valid_time": '1D'}).mean().rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
res_ds = res.to_dataset(name='t2m')

res_ds.to_netcdf("./data/t2m.nc")

mean_ds = res_ds.mean(dim="time")
mean_ds.to_netcdf("./data/t2m_mean.nc")

var_ds = res_ds.mean(dim="time")
var_ds.to_netcdf("./data/t2m_variance.nc")