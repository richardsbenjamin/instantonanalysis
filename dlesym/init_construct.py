import argparse
import re

import xarray as xr
import numpy as np

from paths import DLESYM_DIR, REPO_DIR

TIME_UNIT = "m"
TIME_DTYPE = f"datetime64[{TIME_UNIT}]"
DELTA_DTYPE = f"timedelta64[{TIME_UNIT}]"

NS_CEILING = np.datetime64("2262-04-11T00:00", TIME_UNIT)

_UNIT_MINUTES = {
    "minute": 1, "minutes": 1, "m": 1,
    "hour": 60, "hours": 60, "h": 60,
    "day": 1440, "days": 1440, "d": 1440, "D": 1440,
    "second": 1 / 60, "seconds": 1 / 60, "s": 1 / 60,
}

_CF_UNITS = re.compile(r"^\s*(?P<unit>\w+)\s+since\s+(?P<reference>.+?)\s*$")


def to_datetime64(value) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value.astype(TIME_DTYPE)
    if hasattr(value, "isoformat"):
        return np.datetime64(value.isoformat(), TIME_UNIT)
    return np.datetime64(str(value), TIME_UNIT)


def to_datetime64_array(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind == "M":
        return arr.astype(TIME_DTYPE)
    flat = [to_datetime64(v) for v in arr.ravel()]
    return np.array(flat, dtype=TIME_DTYPE).reshape(arr.shape)


def to_timedelta64(values, unit: str = "h") -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind == "m":
        return arr.astype(DELTA_DTYPE)
    if unit not in _UNIT_MINUTES:
        raise ValueError(f"unrecognised step unit {unit!r}")
    minutes = np.asarray(arr, dtype="float64") * _UNIT_MINUTES[unit]
    return np.rint(minutes).astype("int64").astype(DELTA_DTYPE)


def step_to_time(base_times, steps, step_unit: str = "h"):
    base = to_datetime64_array(base_times)
    offsets = to_timedelta64(steps, step_unit)
    return (base[:, None] + offsets[None, :]).ravel()


def _decode_cf_axis(values, units: str) -> np.ndarray:
    match = _CF_UNITS.match(units)
    if match is None:
        raise ValueError(f"cannot parse CF time units {units!r}")
    reference = np.datetime64(
        match.group("reference").strip().replace(" ", "T"), TIME_UNIT)
    return reference + to_timedelta64(values, match.group("unit"))


def open_state(path: str) -> xr.Dataset:
    path = str(path)
    if path.endswith(".zarr") or path.startswith("gs://"):
        ds = xr.open_zarr(path)
        time = to_datetime64_array(ds["time"].values)
        step = to_timedelta64(ds["step"].values)
    else:
        ds = xr.open_dataset(path, decode_times=False, chunks={})
        time = _decode_cf_axis(ds["time"].values, ds["time"].attrs["units"])
        step = _decode_cf_axis(ds["step"].values, ds["step"].attrs["units"]) - time[0]

    if time.size != 1:
        raise ValueError(f"{path}: expected a single `time`, got {time.size}")
    if step.size > 1 and np.any(np.diff(step.astype("int64")) <= 0):
        raise ValueError(f"{path}: `step` must be strictly increasing")
    return ds.assign_coords(time=("time", time), step=("step", step))


def _locate_exact(available: np.ndarray, requested: np.ndarray,
                  label: str) -> np.ndarray:
    idx = np.clip(np.searchsorted(available, requested), 0, available.size - 1)
    bad = available[idx] != requested
    if bad.any():
        raise ValueError(
            f"{label}: {int(bad.sum())} of {requested.size} required times are "
            f"absent from the store (first missing {requested[bad][0]}). The "
            f"init window for this date was never extracted.")
    return idx


def _assert_contiguous(times: np.ndarray, start: int, stop: int,
                       spacing: np.timedelta64, label: str) -> None:
    segment = times[start:stop]
    if segment.size < 2:
        return
    gaps = np.diff(segment)
    if not np.all(gaps == spacing):
        first = int(np.argmax(gaps != spacing))
        raise ValueError(
            f"{label}: slice spans a gap between {segment[first]} and "
            f"{segment[first + 1]} (expected {spacing}). Slicing across it "
            f"would splice in another event's data.")

def ds1_to_ds2(
    ds1: xr.Dataset,
    constants: xr.DataArray,
    step_unit: str = "h",
) -> xr.Dataset:
    channel_in_vars = list(ds1.data_vars)
    missing = [v for v in channel_in_vars if v not in ds1]
    if missing:
        raise ValueError(f"Variables missing from ds1: {missing}")

    time_axis = step_to_time(
        ds1["time"].values,
        ds1["step"].values,
        step_unit=step_unit,
    ) 

    channel_arrays = []
    for var in channel_in_vars:
        da = ds1[var]

        da = da.transpose("time", "step", "face", "height", "width")

        n_time, n_step, n_face, n_h, n_w = da.shape
        flat = da.values.reshape(n_time * n_step, n_face, n_h, n_w)
        channel_arrays.append(flat)

    inputs_np = np.stack(channel_arrays, axis=1)

    inputs_da = xr.DataArray(
        inputs_np,
        dims=["time", "channel_in", "face", "height", "width"],
        coords={
            "time":       time_axis,
            "channel_in": channel_in_vars,
        },
        name="inputs",
    )

    ds2 = xr.Dataset(
        {
            "inputs":    inputs_da,
            "constants": constants,
        }
    )

    return ds2

def flatten_aux_time(
        aux_ds: xr.Dataset,
        var: str,
        step_unit: str = "h",
    ) -> xr.DataArray:
    da = aux_ds[var].transpose("time", "step", "face", "height", "width")

    flat_times = step_to_time(
        aux_ds["time"].values,
        aux_ds["step"].values,
        step_unit,
    )

    n_time, n_step, n_face, n_h, n_w = da.shape
    flat_data = da.values.reshape(n_time * n_step, n_face, n_h, n_w)

    return xr.DataArray(
        flat_data,
        dims=["time", "face", "height", "width"],
        coords={"time": flat_times},
        name=var,
    )

def add_constant_channel(
    ds2: xr.Dataset,
    aux_ds: xr.Dataset,
    var: str,
    step_unit: str = "h",
) -> xr.Dataset:
    if var in ds2["inputs"].coords["channel_in"].values:
        raise ValueError(f"'{var}' is already a channel in ds2.")

    ds2_times = ds2["inputs"].coords["time"]

    aux_flat = flatten_aux_time(aux_ds, var, step_unit=step_unit)

    assert aux_flat.time.min() <= ds2_times.min(), \
        "aux_ds starts after ds2 — ffill would produce leading NaNs"

    aux_aligned = aux_flat.reindex(time=ds2_times, method="ffill")

    aux_as_channel = (
        aux_aligned
        .expand_dims({"channel_in": [var]}, axis=1)
        .transpose("time", "channel_in", "face", "height", "width")
    )

    existing_channels = list(ds2["inputs"].coords["channel_in"].values)
    new_channel_in = existing_channels + [var]

    new_inputs = (
        xr.concat([ds2["inputs"], aux_as_channel], dim="channel_in")
        .assign_coords(channel_in=new_channel_in)
    )

    return (
        ds2.drop_vars("inputs")
        .assign_coords(channel_in=new_channel_in)
        .assign(inputs=new_inputs)
    )


def add_historic_avg_channel(
    ds2: xr.Dataset,
    aux_ds: xr.Dataset,
    var: str | list[str],
    window_h: int = 48,
    aux_step_h: int = 6,
    step_unit: str = "h",
) -> xr.Dataset:
    vars_ = [var] if isinstance(var, str) else var

    ds2_times = ds2["inputs"].coords["time"]
    n_steps = window_h // aux_step_h + 1
    new_channels = []

    for v in vars_:
        new_var = f"{v}-{window_h}"

        if new_var in ds2["inputs"].coords["channel_in"].values:
            raise ValueError(f"'{new_var}' is already a channel in ds2.")

        aux_flat = flatten_aux_time(aux_ds, v, step_unit=step_unit)

        required_start = (ds2_times.min().values.astype(TIME_DTYPE)
                          - np.timedelta64(window_h, "h").astype(DELTA_DTYPE))
        aux_start = aux_flat.time.min().values.astype(TIME_DTYPE)
        assert aux_start <= required_start, (
            f"aux_ds starts at {aux_start} but must reach back to {required_start} "
            f"({window_h}h before ds2) to fill the first window"
        )

        avg_da = (
            aux_flat
            .rolling(time=n_steps, min_periods=n_steps)
            .mean()
            .sel(time=ds2_times)
        )
        avg_da.name = new_var

        new_channels.append(
            avg_da
            .expand_dims({"channel_in": [new_var]}, axis=1)
            .transpose("time", "channel_in", "face", "height", "width")
        )

    new_channel_in = list(ds2["inputs"].coords["channel_in"].values) + [f"{v}-{window_h}" for v in vars_]
    new_inputs = xr.concat([ds2["inputs"]] + new_channels, dim="channel_in")

    return (
        ds2.drop_vars("inputs")
        .assign_coords(channel_in=new_channel_in)
        .assign(inputs=new_inputs)
    )

def make_time_inputs(
    date_np: np.datetime64,
    ocean_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.datetime64]:
    ocean_idx = np.searchsorted(ocean_times, date_np, side='right') - 1
    if ocean_idx < 0:
        raise ValueError(
            f"{date_np} precedes the first ocean step {ocean_times[0]}; no "
            f"48h history exists for it.")
    forecast_date = ocean_times[ocean_idx]

    atmos_time_in = np.array([forecast_date - np.timedelta64((8 - i) * 6, 'h') for i in range(9)])
    ocean_time_in = np.array([forecast_date - np.timedelta64(48, 'h'), forecast_date])
    return atmos_time_in, ocean_time_in, forecast_date


def build_model_inputs(
    atmos_ds: xr.Dataset,
    ocean_ds: xr.Dataset,
    atmos_time_in: np.ndarray,
    ocean_time_in: np.ndarray,
    atmos_constants: xr.DataArray,
    ocean_constants: xr.DataArray,
    hist_avg_vars: list[str] = ['ws10', 'z1000', 'olr'],
) -> tuple[xr.Dataset, xr.Dataset]:
    ocean_times = step_to_time(ocean_ds.time, ocean_ds.step)
    atmos_times = step_to_time(atmos_ds.time, atmos_ds.step)

    six_hours = np.timedelta64(6, "h").astype(DELTA_DTYPE)
    two_days = np.timedelta64(48, "h").astype(DELTA_DTYPE)

    atmos_idx = np.sort(_locate_exact(atmos_times, atmos_time_in,
                                      "atmos init window"))
    ocean_time_in_ = ocean_times[
        np.searchsorted(ocean_times, atmos_time_in, side='right') - 1
    ]
    ocean_idx_c = np.sort(np.unique(np.searchsorted(ocean_times, ocean_time_in_)))

    _assert_contiguous(atmos_times, atmos_idx[0], atmos_idx[-1] + 1,
                       six_hours, "atmos init window")
    _assert_contiguous(ocean_times, ocean_idx_c[0], ocean_idx_c[-1] + 1,
                       two_days, "ocean sst window")

    atmos_slice_in = atmos_ds.isel(step=slice(atmos_idx[0],   atmos_idx[-1]   + 1))
    ocean_slice_c  = ocean_ds.isel(step=slice(ocean_idx_c[0], ocean_idx_c[-1] + 1))

    atmos_ds_in   = ds1_to_ds2(atmos_slice_in, atmos_constants)
    atmos_ds_in = add_constant_channel(atmos_ds_in, ocean_slice_c, var="sst")

    ocean_idx_h = np.sort(
        np.searchsorted(ocean_times, ocean_time_in, side='right') - 1
    )
    actual_ocean_times_h = ocean_times[ocean_idx_h]

    atmos_lookback_begin = actual_ocean_times_h - two_days
    atmos_idx_begin      = min(_locate_exact(atmos_times, atmos_lookback_begin,
                                             "atmos rolling-mean window"))
    atmos_idx_end        = max(_locate_exact(atmos_times, actual_ocean_times_h,
                                             "atmos rolling-mean window"))

    _assert_contiguous(atmos_times, atmos_idx_begin, atmos_idx_end + 1,
                       six_hours, "atmos rolling-mean window")
    _assert_contiguous(ocean_times, ocean_idx_h[0], ocean_idx_h[-1] + 1,
                       two_days, "ocean history window")

    ocean_slice_h  = ocean_ds.isel(step=slice(ocean_idx_h[0], ocean_idx_h[-1] + 1))
    atmos_slice_h  = atmos_ds.isel(step=slice(atmos_idx_begin, atmos_idx_end  + 1))

    ocean_ds_in   = ds1_to_ds2(ocean_slice_h, ocean_constants)
    ocean_ds_in = add_historic_avg_channel(ocean_ds_in, atmos_slice_h, var=hist_avg_vars)

    return atmos_ds_in, ocean_ds_in


def build_all_model_inputs(
    dates,
    atmos_ds: xr.Dataset,
    ocean_ds: xr.Dataset,
    atmos_constants: xr.DataArray,
    ocean_constants: xr.DataArray,
    hist_avg_vars: list[str] = ['ws10', 'z1000', 'olr'],
) -> tuple[xr.Dataset, xr.Dataset]:
    atmos_list, ocean_list = [], []
    forecast_dates = []

    ocean_times = step_to_time(ocean_ds.time, ocean_ds.step)

    for date in np.asarray(dates):
        print("INITIALISING FOR DATE", date)
        date_np = to_datetime64(date)
        atmos_time_in, ocean_time_in, forecast_date = make_time_inputs(date_np, ocean_times)
        forecast_dates.append(forecast_date)

        atmos_in, ocean_in = build_model_inputs(
            atmos_ds=atmos_ds, ocean_ds=ocean_ds,
            atmos_time_in=atmos_time_in, ocean_time_in=ocean_time_in,
            atmos_constants=atmos_constants, ocean_constants=ocean_constants,
            hist_avg_vars=hist_avg_vars,
        )

        atmos_list.append(atmos_in)
        ocean_list.append(ocean_in)

    print("CONCATENATING MODEL INPUTS")
    atmos_inputs = xr.concat([ds["inputs"] for ds in atmos_list], dim="time")
    ocean_inputs = xr.concat([ds["inputs"] for ds in ocean_list], dim="time")

    _, atmos_uniq = np.unique(atmos_inputs.time.values, return_index=True)
    _, ocean_uniq = np.unique(ocean_inputs.time.values, return_index=True)
    atmos_inputs = atmos_inputs.isel(time=np.sort(atmos_uniq))
    ocean_inputs = ocean_inputs.isel(time=np.sort(ocean_uniq))

    forecast_dates_iso = [str(np.datetime_as_string(d, unit='s')) for d in forecast_dates]

    atmos_all = xr.Dataset({"inputs": atmos_inputs, "constants": atmos_list[0]["constants"]})
    ocean_all = xr.Dataset({"inputs": ocean_inputs, "constants": ocean_list[0]["constants"]})
    atmos_all.attrs["forecast_init_times"] = forecast_dates_iso
    ocean_all.attrs["forecast_init_times"] = forecast_dates_iso

    return atmos_all, ocean_all


def candidate_init_times(
    ocean_ds: xr.Dataset,
    heatwave_dates: np.ndarray,
    months: tuple[int, int] | None = None,
    exclude_window_days: int = 4,
) -> np.ndarray:
    ocean_times = step_to_time(ocean_ds.time, ocean_ds.step)
    times = ocean_times[(ocean_times >= ocean_times[0] + np.timedelta64(96, "h"))
                        & (ocean_times <= ocean_times[-1] - np.timedelta64(96, "h"))]
    if months is not None:
        m = times.astype("datetime64[M]").astype("int64") % 12 + 1
        times = times[(m >= months[0]) & (m <= months[1])]
    if len(heatwave_dates) and exclude_window_days:
        hw = to_datetime64_array(heatwave_dates)
        gap = np.abs(times[:, None] - hw[None, :])
        times = times[(gap > np.timedelta64(exclude_window_days, "D")).all(axis=1)]
    return times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DLESyM model initialisation inputs.")
    parser.add_argument("--constants_path", default=str(DLESYM_DIR / "example_data" / "constants.nc"))
    parser.add_argument("--atmos_path", default="gs://dlesym-storage/atmos_hpx64_100year.zarr")
    parser.add_argument("--ocean_path", default="gs://dlesym-storage/ocean_hpx64_100year.zarr")
    parser.add_argument("--neighbours_path",
                        default=str(REPO_DIR / "outputs" / "paris" / "paris_closest_neighbours.nc"),
                        help="Path to <city>_closest_neighbours.nc used for heatwave date selection.")
    parser.add_argument("--quantile", type=float, default=0.999,
                        help="Event quantile to select from the neighbours file.")
    parser.add_argument("--rolling-period", type=int, default=1,
                        help="Rolling period to select from the neighbours file. "
                             "rp=5 needs a 192h forecast, so only rp=1 is supported "
                             "by the 96h init window this script builds.")
    parser.add_argument("--heatwave", action=argparse.BooleanOptionalAction, default=True,
                        help="Use heatwave dates (--heatwave) or randomly sample non-heatwave dates (--no-heatwave).")
    parser.add_argument("--n_random", type=int, default=50,
                        help="Number of non-heatwave dates to randomly sample (only used with --no-heatwave).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for non-heatwave date sampling.")
    parser.add_argument("--months", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"),
                        help="Inclusive month range for non-heatwave sampling, e.g. --months 6 8 "
                             "for JJA (mirrors the physical pipeline's calc_months). Applied to "
                             "the snapped init times, so the guarantee survives snapping.")
    parser.add_argument("--exclude_window_days", type=int, default=4,
                        help="Exclude sampled dates within this many days of any heatwave date "
                             "(only used with --no-heatwave / --dates-file sampling).")
    parser.add_argument("--dates-file", type=str, default=None,
                        help="Optional path to a text file of ISO datetimes (one per line) to use "
                             "as init dates, bypassing both heatwave and random selection.")
    parser.add_argument("--allow-ns-overflow", action="store_true",
                        help="Permit init dates past the datetime64[ns] ceiling "
                             "(2262-04-11). Off by default because the forecast "
                             "env is nanosecond-only and would fail silently.")
    parser.add_argument("--atmos_output",
                        default=str(DLESYM_DIR / "example_data" / "atmos_reconstructed_init_nonheatwave.zarr"))
    parser.add_argument("--ocean_output",
                        default=str(DLESYM_DIR / "example_data" / "ocean_reconstructed_init_nonheatwave.zarr"))
    args = parser.parse_args()

    constants_ds = xr.open_dataset(args.constants_path)
    atmos_ds = open_state(args.atmos_path)
    ocean_ds = open_state(args.ocean_path)

    def load_heatwave_dates() -> np.ndarray:
        nn = xr.open_dataset(args.neighbours_path, use_cftime=True)
        sel = nn["t2m0"].sel(quantile=args.quantile,
                             rolling_period=args.rolling_period)
        return to_datetime64_array(sel.dropna(dim="step").step.values)

    if args.dates_file is not None:
        print(f"USING DATES FROM {args.dates_file}")
        with open(args.dates_file) as f:
            dates = np.sort(np.array(
                [np.datetime64(line.strip(), TIME_UNIT) for line in f if line.strip()]))
    elif args.heatwave:
        print(f"USING HEATWAVE DATES (q={args.quantile}, rp={args.rolling_period})")
        dates = load_heatwave_dates()
    else:
        print("USING NON-HEATWAVE DATES")
        months = tuple(args.months) if args.months is not None else None
        candidates = candidate_init_times(
            ocean_ds, load_heatwave_dates(), months=months,
            exclude_window_days=args.exclude_window_days)
        print(f"{len(candidates)} candidate init times "
              f"(months={months}, heatwave window ±{args.exclude_window_days}d excluded)")
        rng = np.random.default_rng(args.seed)
        dates = np.sort(rng.choice(candidates, size=args.n_random, replace=False))

    atmos_constants = constants_ds["atmos_constants"]
    ocean_constants = constants_ds["ocean_constants"].sel(channel_c=["lsm"])

    print("BUILDING MODEL INPUTS")
    atmos_all, ocean_all = build_all_model_inputs(
        dates=dates,
        atmos_ds=atmos_ds, ocean_ds=ocean_ds,
        atmos_constants=atmos_constants, ocean_constants=ocean_constants,
    )

    overflow = np.asarray([d for d in dates if to_datetime64(d) >= NS_CEILING])
    if overflow.size and not args.allow_ns_overflow:
        raise SystemExit(
            f"{overflow.size} of {len(dates)} requested dates fall on or after "
            f"{NS_CEILING} (first {overflow[0]}), past the datetime64[ns] "
            f"ceiling. The forecast env cannot read them: it runs xarray "
            f"2024.10 / pandas 1.5, which are nanosecond-only.\n"
            f"Resolve by upgrading that env's xarray to >= 2025.01, or rerun "
            f"with --allow-ns-overflow if you have verified the consumer "
            f"handles non-nanosecond times.")

    print("SAVING MODEL INPUTS")
    for ds_ in (atmos_all, ocean_all):
        for name in list(ds_.coords) + list(ds_.data_vars):
            if ds_[name].dtype.kind == "U":
                ds_[name] = ds_[name].astype(object)
    try:
        atmos_all.to_zarr(args.atmos_output, mode="w", zarr_format=2)
        ocean_all.to_zarr(args.ocean_output, mode="w", zarr_format=2)
    except TypeError:
        atmos_all.to_zarr(args.atmos_output, mode="w")
        ocean_all.to_zarr(args.ocean_output, mode="w")
