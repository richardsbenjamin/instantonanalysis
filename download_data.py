import xarray as xr


def read_edh(edh_path: str) -> xr.Dataset:
    return xr.open_dataset(
        edh_path,
        storage_options={"client_kwargs":{"trust_env":True}},
        chunks={"time": 1},
        engine="zarr",
    )


singles_path = "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr"
levels_path = "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-pressure-levels-v0.zarr"

path = levels_path
var = "z"
bis = "z500"
level = 500
levels = True


era5_ds = read_edh(path)



era5_ds = era5_ds.sel(valid_time=slice("1940-01-01", "1945-01-01"))
res = era5_ds[var]
if levels:
    res = res.sel(isobaricInhPa=level)
res = res.resample({"valid_time": '1D'}).mean().rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
res_ds = res.to_dataset(name=bis)

res_ds.to_netcdf("./data/"+bis+".nc")

mean_ds = res_ds.mean(dim="time")
mean_ds.to_netcdf("./data/"+bis+"_mean.nc")

var_ds = res_ds.var(dim="time")
var_ds.to_netcdf("./data/"+bis+"_variance.nc")
