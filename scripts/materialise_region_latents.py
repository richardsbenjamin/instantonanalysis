from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from instantonanalysis.instanton.schemas.box import HealPixBox, LonLatBox
from instantonanalysis.instanton.schemas.xconfig import XConfigHealPix
from instantonanalysis.instanton.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_region(name: str) -> tuple[LonLatBox, str]:
    cfg = load_config(f"locations/{name}").locations
    box = LonLatBox(
        lon_min=cfg.box.lon_min, lon_max=cfg.box.lon_max,
        lat_min=cfg.box.lat_min, lat_max=cfg.box.lat_max,
        lon_system=cfg.box.lon_sys, lat_system=cfg.box.lat_sys,
    )
    label = (f"{cfg.name} ({box.lon_min}..{box.lon_max}E, "
             f"{box.lat_min}..{box.lat_max}N)")
    return box, label


def latent_levels(ds: xr.Dataset) -> list[int]:
    return sorted(int(v.removeprefix("encoder_l")) for v in ds.data_vars
                  if v.startswith("encoder_l"))


def filter_level(src: xr.Dataset, level: int, region: LonLatBox, time_chunk: int) -> xr.DataArray:
    da = src[f"encoder_l{level}"]
    xcfg = XConfigHealPix(
        time_dim="time",
        channel_dim=f"channel_l{level}",
        face_dim="face",
        height_dim=f"height_l{level}",
        width_dim=f"width_l{level}",
    )
    nside = da.sizes[xcfg.height_dim]
    box = HealPixBox.from_lonlat_box(nside, region)
    faces = sorted(set(box.f_list))

    parts = []
    n_time = da.sizes["time"]
    for start in range(0, n_time, time_chunk):
        chunk = da.isel(time=slice(start, start + time_chunk)).sel(face=faces)
        chunk = chunk.compute(scheduler="synchronous")
        sub = box.extract(chunk, xconfig=xcfg)
        parts.append(sub.transpose("time", "step", "points", xcfg.channel_dim))
    out = xr.concat(parts, dim="time")
    out = out.rename({"points": f"points_l{level}", "face": f"face_l{level}"})
    logger.info(f"  l{level}: nside={nside}, in-box points={out.sizes[f'points_l{level}']}")
    return out


def filter_all_levels(src_path: str, region_name: str, time_chunk: int) -> xr.Dataset:
    src = xr.open_zarr(src_path, decode_timedelta=True)
    levels = latent_levels(src)
    if not levels:
        raise ValueError(f"No encoder_l* variables in {src_path!r}")
    region, region_label = load_region(region_name)
    logger.info(f"Filtering {src_path!r} (levels {levels}) to {region_label}")
    ds = xr.Dataset({f"encoder_l{lvl}": filter_level(src, lvl, region, time_chunk)
                     for lvl in levels})
    ds.attrs.update(src.attrs)
    ds.attrs["region"] = region_label
    ds.attrs["source"] = str(src_path)
    return ds


def filter_physical(src_path: str, region_name: str, nside: int,
                    var_names: list[str] | None, time_chunk: int) -> xr.Dataset:
    src = xr.open_zarr(src_path, decode_timedelta=True)
    if var_names:
        missing = [v for v in var_names if v not in src.data_vars]
        if missing:
            raise ValueError(f"{src_path!r}: missing variables {missing}")
        src = src[list(var_names)]
    else:
        src = src[list(src.data_vars)]

    region, region_label = load_region(region_name)
    box = HealPixBox.from_lonlat_box(nside, region)
    faces = sorted(set(box.f_list))
    logger.info(f"Filtering {src_path!r} ({list(src.data_vars)}) to {region_label} "
                f"at nside={nside}")

    parts = []
    n_time = src.sizes["time"]
    for start in range(0, n_time, time_chunk):
        chunk = src.isel(time=slice(start, start + time_chunk)).sel(face=faces)
        chunk = chunk.compute(scheduler="synchronous")
        parts.append(box.extract(chunk).transpose("time", "step", "points"))
    out = xr.concat(parts, dim="time")
    out.attrs.update(src.attrs)
    out.attrs["region"] = region_label
    out.attrs["source"] = str(src_path)
    out.attrs["nside"] = nside
    logger.info(f"  in-box points={out.sizes['points']}, steps={out.sizes['step']}")
    return out


def verify_region_ds(ds: xr.Dataset, expect_times: int | None) -> None:
    for name, da in ds.data_vars.items():
        if expect_times is not None and da.sizes["time"] != expect_times:
            raise AssertionError(
                f"{name}: expected {expect_times} times, got {da.sizes['time']}")
        sample = da.isel(time=0).values
        if not np.isfinite(sample).all():
            raise AssertionError(f"{name}: non-finite values in first event")
    logger.info("Verification passed: "
                + ", ".join(f"{n}{tuple(da.sizes.values())}" for n, da in ds.data_vars.items()))


def delete_source(path: str) -> None:
    logger.info(f"Deleting full-field source {path!r}")
    shutil.rmtree(path)


def run_batch(args) -> None:
    ds = filter_all_levels(args.source, args.region, args.time_chunk)
    verify_region_ds(ds, None)
    ds = ds.chunk({"time": args.time_chunk})
    out = Path(args.out)
    if out.exists():
        logger.info(f"Appending {ds.sizes['time']} dates to {args.out!r}")
        ds.to_zarr(args.out, append_dim="time")
    else:
        logger.info(f"Creating {args.out!r} with {ds.sizes['time']} dates")
        ds.to_zarr(args.out, mode="w")
    n_total = xr.open_zarr(args.out, decode_timedelta=True).sizes["time"]
    logger.info(f"{args.out} now holds {n_total} dates")
    if args.delete_source:
        delete_source(args.source)


def run_physical(args) -> None:
    ds = filter_physical(args.source, args.region, args.nside,
                         args.vars, args.time_chunk)
    for name, da in ds.data_vars.items():
        if not np.isfinite(da.isel(time=0).values).all():
            raise AssertionError(f"{name}: non-finite values in the first date")
    ds = ds.chunk({"time": args.time_chunk})
    out = Path(args.out)
    if out.exists():
        logger.info(f"Appending {ds.sizes['time']} dates to {args.out!r}")
        ds.to_zarr(args.out, append_dim="time")
    else:
        logger.info(f"Creating {args.out!r} with {ds.sizes['time']} dates")
        ds.to_zarr(args.out, mode="w")
    n_total = xr.open_zarr(args.out, decode_timedelta=True).sizes["time"]
    logger.info(f"{args.out} now holds {n_total} dates")
    if args.delete_source:
        delete_source(args.source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    region_help = "location config name under config/locations/ to filter to"

    cb = sub.add_parser("batch",
                        help="region-filter a full-field latent batch and "
                             "create-or-append to a region_latents zarr")
    cb.add_argument("--source", required=True)
    cb.add_argument("--out", required=True)
    cb.add_argument("--region", default="western_europe", help=region_help)
    cb.add_argument("--delete-source", action="store_true")
    cb.add_argument("--time-chunk", type=int, default=10)
    cb.set_defaults(func=run_batch)

    ph = sub.add_parser("physical",
                        help="region-filter a full-field physical forecast and "
                             "create-or-append to a region_physical zarr")
    ph.add_argument("--source", required=True)
    ph.add_argument("--out", required=True)
    ph.add_argument("--region", default="western_europe", help=region_help)
    ph.add_argument("--nside", type=int, default=64,
                    help="nside of the physical forecast grid (the box must "
                         "match probe.physical_region_field's box64).")
    ph.add_argument("--vars", nargs="+", default=None,
                    help="Variables to keep; default all in the source.")
    ph.add_argument("--delete-source", action="store_true")
    ph.add_argument("--time-chunk", type=int, default=10)
    ph.set_defaults(func=run_physical)

    args = parser.parse_args()
    args.func(args)
