"""Materialize small region-filtered latent zarrs (GLOBAL_PLAN Layer 3).

Filters full-field DLESyM latent zarrs (all encoder levels) to one region box
once, so the analysis stages read ~150 MB zarrs instead of 27 GB full-field ones. The
three spatial dims collapse to a per-level ``points_l{N}`` dim (the levels have
different nsides, hence different in-box point counts).

The region is a location config under ``config/locations/`` (``--region``,
default ``western_europe``); ``se_australia``, ``north_china`` and ``se_brazil``
are the regions around Adelaide, Beijing and Rio.

Modes (run from the instantonanalysis .venv):

  event       one full-field event zarr -> region_latents_<class>.zarr
              python scripts/materialize_region_latents.py event \
                  --source .../atmos_heatwave_latents_all.zarr \
                  --out .../region_latents_heatwave.zarr [--delete-source]

  clim-batch  append one full-field climatology batch to clim_latents.zarr
              (same filtering; ``--delete-source`` retires the batch afterwards)

  clim-stats  stream clim_latents.zarr -> clim_stats.nc with per-level
              clim_mean / clim_var (pooled over time x step) and clim_var_hat
              (variance of the per-date trajectory mean — the null-calibrated
              normalizer for norm_var_hat).
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from instantonanalysis.instanton.schemas.box import HealPixBox, LonLatBox
from instantonanalysis.instanton.schemas.xconfig import XConfigHealPix
from instantonanalysis.instanton.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_region(name: str) -> tuple[LonLatBox, str]:
    """The region filter box and its label, from ``config/locations/<name>.yaml``."""
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
    """Region-filter one encoder level to ``(time, step, points_l{N}, channel_l{N})``.

    Streams over time chunks: the face-sliced l0 array is ~95 MB per date, so a
    full 50-date load would be ~5 GB; chunked loads stay under ~1 GB.
    """
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
        # One-chunk-at-a-time load (same rationale as latent/centroid.box_extract).
        chunk = chunk.compute(scheduler="synchronous")
        sub = box.extract(chunk, xconfig=xcfg)
        parts.append(sub.transpose("time", "step", "points", xcfg.channel_dim))
    out = xr.concat(parts, dim="time")
    # Per-level points dim; the pointwise face coord is renamed too since a
    # shared "face(points_lN)" coord name would collide across levels.
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


def run_event(args) -> None:
    ds = filter_all_levels(args.source, args.region, args.time_chunk)
    verify_region_ds(ds, args.expect_times)
    ds = ds.chunk({"time": args.time_chunk})
    logger.info(f"Writing {args.out!r}")
    ds.to_zarr(args.out, mode="w")
    # Re-verify from disk before any deletion.
    verify_region_ds(xr.open_zarr(args.out, decode_timedelta=True), args.expect_times)
    if args.delete_source:
        delete_source(args.source)


def run_clim_batch(args) -> None:
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
    logger.info(f"clim_latents now holds {n_total} dates")
    if args.delete_source:
        delete_source(args.source)


def run_clim_stats(args) -> None:
    clim = xr.open_zarr(args.clim, decode_timedelta=True)
    out_vars = {}
    n_samples = clim.sizes["time"]
    n_pooled = n_samples * clim.sizes["step"]
    for lvl in latent_levels(clim):
        da = clim[f"encoder_l{lvl}"]
        pts, ch = f"points_l{lvl}", f"channel_l{lvl}"
        shape = (da.sizes[pts], da.sizes[ch])
        s = np.zeros(shape); ss = np.zeros(shape)
        traj_means = []
        for start in range(0, n_samples, args.time_chunk):
            block = (da.isel(time=slice(start, start + args.time_chunk))
                     .transpose("time", "step", pts, ch)
                     .values.astype(np.float64))
            s += block.sum(axis=(0, 1))
            ss += (block ** 2).sum(axis=(0, 1))
            traj_means.append(block.mean(axis=1))
        mean = s / n_pooled
        var = np.maximum(ss / n_pooled - mean ** 2, 0.0)
        var_hat = np.concatenate(traj_means, axis=0).var(axis=0)
        coords = {c: da[c] for c in da.coords if set(da[c].dims) <= {pts, ch}}
        dims = (pts, ch)
        out_vars[f"clim_mean_l{lvl}"] = xr.DataArray(mean, dims=dims, coords=coords)
        out_vars[f"clim_var_l{lvl}"] = xr.DataArray(var, dims=dims, coords=coords)
        out_vars[f"clim_var_hat_l{lvl}"] = xr.DataArray(var_hat, dims=dims, coords=coords)
        logger.info(f"  l{lvl}: clim_var range [{var.min():.4g}, {var.max():.4g}], "
                    f"var_hat/var median {np.median(var_hat / np.maximum(var, 1e-12)):.3f}")
        if not (np.isfinite(var).all() and (var > 0).all()):
            raise AssertionError(f"l{lvl}: clim_var must be finite and > 0")
    months = pd.DatetimeIndex(clim.time.values).month
    stats = xr.Dataset(out_vars, attrs={
        "n_samples": n_samples,
        "n_pooled": n_pooled,
        "months": str(sorted(set(months.tolist()))),
        "seed": args.seed,
        "source": str(args.clim),
        "note": ("clim_var_l* pools all (date, step) embeddings (normalizer for "
                 "norm_var_tilde); clim_var_hat_l* is the variance of per-date "
                 "trajectory means (null-calibrated normalizer for norm_var_hat)."),
    })
    logger.info(f"Writing {args.out!r} (n_samples={n_samples}, months={stats.attrs['months']})")
    stats.to_netcdf(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    region_help = "location config name under config/locations/ to filter to"

    ev = sub.add_parser("event", help="full-field event zarr -> region_latents zarr")
    ev.add_argument("--source", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--region", default="western_europe", help=region_help)
    ev.add_argument("--expect-times", type=int, default=50)
    ev.add_argument("--delete-source", action="store_true")
    ev.add_argument("--time-chunk", type=int, default=10)
    ev.set_defaults(func=run_event)

    cb = sub.add_parser("clim-batch", help="append full-field clim batch to clim_latents zarr")
    cb.add_argument("--source", required=True)
    cb.add_argument("--out", required=True)
    cb.add_argument("--region", default="western_europe", help=region_help)
    cb.add_argument("--delete-source", action="store_true")
    cb.add_argument("--time-chunk", type=int, default=10)
    cb.set_defaults(func=run_clim_batch)

    cs = sub.add_parser("clim-stats", help="clim_latents zarr -> clim_stats.nc")
    cs.add_argument("--clim", required=True)
    cs.add_argument("--out", required=True)
    cs.add_argument("--seed", type=int, default=42)
    cs.add_argument("--time-chunk", type=int, default=25)
    cs.set_defaults(func=run_clim_stats)

    args = parser.parse_args()
    args.func(args)
