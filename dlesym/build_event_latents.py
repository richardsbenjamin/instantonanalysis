from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from paths import (
    DATA_DIR, DLESYM_DIR as MODULE_DIR, FORECAST_ENV, FORECAST_SCRIPT, INIT_SCRIPT,
    MATERIALISE, PY_ANALYSIS, PY_FORECAST, REPO_DIR as ANALYSIS_DIR,
)
from init_construct import NS_CEILING, to_datetime64, to_datetime64_array

PY_INIT = PY_ANALYSIS

CITY_REGION = {"paris": "western_europe"}


def log(msg: str) -> None:
    print(f"[latent-driver {pd.Timestamp.now():%H:%M:%S}] {msg}", flush=True)


def run(cmd: list[str], log_file: Path, env: dict | None = None) -> None:
    log(f"run: {Path(cmd[1]).name} ... -> {log_file.name}")
    import os
    full_env = {**os.environ, **(env or {})}
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT,
                       cwd=str(MODULE_DIR), env=full_env)


def tag_for(quantile: float) -> str:
    return f"q{quantile:g}"


def event_dates(neighbours_path: str, quantile: float,
                rolling_period: int) -> np.ndarray:
    nn = xr.open_dataset(neighbours_path, use_cftime=True)
    sel = nn["t2m0"].sel(quantile=quantile, rolling_period=rolling_period)
    dates = to_datetime64_array(sel.dropna(dim="step").step.values)
    return np.sort(np.unique(dates))


def iso(dates: np.ndarray) -> list[str]:
    return [str(np.datetime_as_string(d, unit="s")) for d in dates]


def start_init(batch_id: str, dates: np.ndarray, work: Path,
               tag: str, args) -> subprocess.Popen | None:
    atmos_out = MODULE_DIR / "example_data" / f"atmos_reconstructed_init_{tag}_{batch_id}.zarr"
    ocean_out = MODULE_DIR / "example_data" / f"ocean_reconstructed_init_{tag}_{batch_id}.zarr"
    if atmos_out.exists() and ocean_out.exists():
        log(f"{tag}/{batch_id}: init zarrs already exist, skipping construction")
        return None
    for p in (atmos_out, ocean_out):
        if p.exists():
            shutil.rmtree(p)

    dates_file = work / f"dates_{batch_id}.txt"
    dates_file.write_text("\n".join(iso(dates)) + "\n")
    cmd = [PY_INIT, "-u", INIT_SCRIPT,
           "--dates-file", str(dates_file),
           "--atmos_path", args.atmos_path, "--ocean_path", args.ocean_path,
           "--atmos_output", str(atmos_out), "--ocean_output", str(ocean_out),
           "--allow-ns-overflow"]
    log(f"{tag}/{batch_id}: starting init construction ({len(dates)} dates, background)")
    f = open(work / f"init_{batch_id}.log", "a")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(MODULE_DIR))


def forecast_batch(batch_id: str, work: Path, tag: str, args) -> Path:
    latent_out = MODULE_DIR / "outputs" / f"atmos_latents_{tag}_{batch_id}.zarr"
    if latent_out.exists():
        shutil.rmtree(latent_out)
    cmd = [PY_FORECAST, FORECAST_SCRIPT,
           "--atmos-model-path", str(MODULE_DIR / "models" / "dlwp"),
           "--ocean-model-path", str(MODULE_DIR / "models" / "dlom"),
           "--non-strict", "--lead-time", "96",
           "--forecast-init-start", "2017-01-19", "--forecast-init-end", "2017-01-19",
           "--freq", "D", "--time-chunk", str(args.time_chunk),
           "--step-chunk", str(args.step_chunk),
           "--cache-dir", "./output_data", "--output-directory", "./outputs",
           "--atmos-output-filename", f"atmos_hpx64_4days_{tag}_{batch_id}",
           "--ocean-output-filename", f"ocean_hpx64_4days_{tag}_{batch_id}",
           "--data-directory", str(MODULE_DIR / "example_data"),
           "--atmos-dataset-name", f"atmos_reconstructed_init_{tag}_{batch_id}",
           "--ocean-dataset-name", f"ocean_reconstructed_init_{tag}_{batch_id}",
           "--gpu", "-1", "--save-latent", "true", "--save-levels", "0", "1", "2",
           "--datetime", "true",
           "--atmos-latent-out-path", latent_out.name, "--zarr", "true"]
    run(cmd, work / f"forecast_{batch_id}.log", env=FORECAST_ENV)
    if not latent_out.exists():
        raise RuntimeError(f"{tag}/{batch_id}: forecast finished but {latent_out} is missing")
    return latent_out


def batch_already_appended(out_zarr: Path, first_date: np.datetime64) -> bool:
    if not out_zarr.exists():
        return False
    times = to_datetime64_array(
        xr.open_zarr(str(out_zarr), use_cftime=True).time.values)
    return bool(np.isin(to_datetime64(first_date), times))


def cleanup_batch(batch_id: str, tag: str) -> None:
    for p in [MODULE_DIR / "example_data" / f"atmos_reconstructed_init_{tag}_{batch_id}.zarr",
              MODULE_DIR / "example_data" / f"ocean_reconstructed_init_{tag}_{batch_id}.zarr",
              MODULE_DIR / "outputs" / f"atmos_hpx64_4days_{tag}_{batch_id}.nc",
              MODULE_DIR / "outputs" / f"ocean_hpx64_4days_{tag}_{batch_id}.nc"]:
        if p.exists():
            shutil.rmtree(p)


def stamp_provenance(out_zarr: Path, args, quantile: float,
                     dates: np.ndarray) -> None:
    group = zarr.open_group(str(out_zarr), mode="a")
    group.attrs.update({
        "neighbours_source": str(args.neighbours_path),
        "quantile": float(quantile),
        "rolling_period": int(args.rolling_period),
        "city": args.city,
        "region": args.region,
        "n_dates": int(len(dates)),
        "dates": iso(dates),
    })


def verify(out_zarr: Path, args, quantile: float) -> bool:
    if not out_zarr.exists():
        log(f"VERIFY {out_zarr.name}: does not exist")
        return False
    group = zarr.open_group(str(out_zarr), mode="r")
    stamped = list(group.attrs.get("dates", []))
    if not stamped:
        log(f"VERIFY {out_zarr.name}: no stamped dates (built before provenance "
            f"stamping, or the run never completed)")
        return False

    current = iso(event_dates(args.neighbours_path, quantile, args.rolling_period))
    only_stamped = sorted(set(stamped) - set(current))
    only_current = sorted(set(current) - set(stamped))
    stored = to_datetime64_array(
        xr.open_zarr(str(out_zarr), use_cftime=True).time.values)

    log(f"VERIFY {out_zarr.name}: {len(stamped)} stamped, {len(current)} in "
        f"{Path(args.neighbours_path).name}, {len(stored)} in the store")
    if len(stored) != len(stamped):
        log(f"  MISMATCH: store holds {len(stored)} times but {len(stamped)} were stamped")
    if only_stamped:
        log(f"  MISMATCH: {len(only_stamped)} stamped dates are no longer in the "
            f"neighbours file (first {only_stamped[0]})")
    if only_current:
        log(f"  MISMATCH: {len(only_current)} neighbours dates are missing from the "
            f"store (first {only_current[0]})")
    ok = not only_stamped and not only_current and len(stored) == len(stamped)
    log("  OK: stamped dates match the neighbours file exactly" if ok
        else "  DRIFTED: rebuild, or point --neighbours-path at the file that built it")
    return ok


def build_quantile(quantile: float, args) -> None:
    tag = tag_for(quantile)
    work = MODULE_DIR / "outputs" / "latent_state" / f"{args.city}_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    out_zarr = Path(args.out_dir) / f"region_latents_{args.city}_{tag}.zarr"
    out_physical = Path(args.out_dir) / f"region_physical_{args.city}_{tag}.zarr"

    dates = event_dates(args.neighbours_path, quantile, args.rolling_period)
    overflow = int((dates >= NS_CEILING).sum())
    log(f"{tag}: {len(dates)} dates from {Path(args.neighbours_path).name} "
        f"(rp={args.rolling_period}); {overflow} past the datetime64[ns] ceiling "
        f"-> --datetime true is mandatory")
    (work / "dates_all.txt").write_text("\n".join(iso(dates)) + "\n")

    batches = [dates[i:i + args.batch_size]
               for i in range(0, len(dates), args.batch_size)]
    pending = [(f"b{k:03d}", b) for k, b in enumerate(batches)
               if not (work / f"b{k:03d}.done").exists()
               and (args.only_batches is None or k in args.only_batches)]
    done = len(batches) - len(pending)
    log(f"{tag}: {len(batches)} batches of <= {args.batch_size}; {done} already complete")
    if args.only_batches is not None:
        log(f"{tag}: --only-batches {args.only_batches} -> "
            f"{[b for b, _ in pending]} ({len(pending)} to run)")
    if args.max_batches is not None:
        pending = pending[:args.max_batches]
        log(f"{tag}: --max-batches {args.max_batches} -> stopping after "
            f"{len(pending)} more batch(es)")

    init_proc: subprocess.Popen | None = None
    for i, (batch_id, batch_dates) in enumerate(pending):
        if i == 0:
            init_proc = start_init(batch_id, batch_dates, work, tag, args)
        if init_proc is not None and init_proc.wait() != 0:
            raise RuntimeError(f"{tag}/{batch_id}: init construction failed "
                               f"(see {work}/init_{batch_id}.log)")
        init_proc = (start_init(*pending[i + 1][:2], work, tag, args)
                     if i + 1 < len(pending) else None)

        log(f"{tag}/{batch_id}: forecasting ({len(batch_dates)} dates, all levels)")
        latent_zarr = forecast_batch(batch_id, work, tag, args)

        if batch_already_appended(out_zarr, batch_dates[0]):
            log(f"{tag}/{batch_id}: already appended, deleting duplicate source")
            shutil.rmtree(latent_zarr)
        else:
            log(f"{tag}/{batch_id}: region-filtering ({args.region}) -> {out_zarr.name}")
            run([PY_ANALYSIS, MATERIALISE, "batch",
                 "--source", str(latent_zarr), "--out", str(out_zarr),
                 "--region", args.region, "--delete-source"],
                work / f"materialise_{batch_id}.log")

        if args.keep_physical:
            physical_src = MODULE_DIR / "outputs" / f"atmos_hpx64_4days_{tag}_{batch_id}.nc"
            if not physical_src.exists():
                raise RuntimeError(f"{tag}/{batch_id}: expected physical output "
                                   f"{physical_src} is missing")
            if batch_already_appended(out_physical, batch_dates[0]):
                log(f"{tag}/{batch_id}: physical already appended, skipping")
            else:
                log(f"{tag}/{batch_id}: region-filtering physical -> {out_physical.name}")
                cmd = [PY_ANALYSIS, MATERIALISE, "physical",
                       "--source", str(physical_src), "--out", str(out_physical),
                       "--region", args.region, "--nside", "64", "--delete-source"]
                if args.physical_vars:
                    cmd += ["--vars", *args.physical_vars]
                run(cmd, work / f"physical_{batch_id}.log")

        cleanup_batch(batch_id, tag)
        (work / f"{batch_id}.done").touch()
        log(f"{tag}/{batch_id}: done ({done + i + 1}/{len(batches)})")

    remaining = [k for k in range(len(batches))
                 if not (work / f"b{k:03d}.done").exists()]
    if remaining:
        log(f"{tag}: {len(remaining)} batches still pending; not stamping provenance yet")
        return
    stamp_provenance(out_zarr, args, quantile, dates)
    log(f"{tag}: complete -> {out_zarr} ({len(dates)} dates, stamped)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", default="paris", choices=sorted(CITY_REGION))
    parser.add_argument("--quantiles", type=float, nargs="+",
                        default=[0.75, 0.95, 0.99, 0.999],
                        help="Quantiles to build, in order.")
    parser.add_argument("--rolling-period", type=int, default=1,
                        help="rp=5 needs a 192h forecast; only rp=1 is supported.")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Dates per batch, bounding the full-field disk "
                             "transient (~640 MB/date). NOTE: the .done markers "
                             "are indexed by batch number, so changing this "
                             "mid-run re-maps them onto different dates — keep it "
                             "fixed for a given output store. It does not affect "
                             "peak memory; see --step-chunk for that.")
    parser.add_argument("--step-chunk", type=int, default=4,
                        help="Dask step-chunk for the latent write. The chunk is "
                             "(time_chunk, step_chunk, C, F, H, W); at 16 (the "
                             "full step axis) l0 chunks are 566 MB and the "
                             "forecast gets OOM-killed on this box.")
    parser.add_argument("--time-chunk", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Stop after this many batches (for a smoke test).")
    parser.add_argument("--only-batches", type=int, nargs="+", default=None,
                        help="Run only these batch indices. Batches are in date "
                             "order, so a high index is the way to exercise the "
                             "post-2262 dates that datetime64[ns] cannot hold.")
    parser.add_argument("--region", default=None,
                        help="Region config to filter to. Defaults per --city.")
    parser.add_argument("--neighbours-path", default=None)
    parser.add_argument("--atmos-path", default=None)
    parser.add_argument("--ocean-path", default=None)
    parser.add_argument("--keep-physical", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Region-filter each batch's physical forecast to "
                             "region_physical_*.zarr (~0.12 MB/date) before the "
                             "raw 26 MB/date version is deleted. Needed by "
                             "latent_probe.py.")
    parser.add_argument("--physical-vars", nargs="+", default=None,
                        help="Physical variables to keep; default all. "
                             "latent_probe.py needs t2m0 and z500.")
    parser.add_argument("--out-dir", default=str(MODULE_DIR / "outputs"))
    parser.add_argument("--verify", action="store_true",
                        help="Diff each output's stamped dates against the "
                             "neighbours file and exit without building.")
    args = parser.parse_args()

    if args.rolling_period != 1:
        parser.error("only --rolling-period 1 is supported: rp=5 spans a 120h "
                     "window and needs a 192h forecast.")
    if args.region is None:
        args.region = CITY_REGION[args.city]
    if args.neighbours_path is None:
        args.neighbours_path = str(
            ANALYSIS_DIR / "outputs" / args.city / f"{args.city}_closest_neighbours.nc")
    if args.atmos_path is None:
        args.atmos_path = str(
            DATA_DIR / f"atmos_dlesym_2000year_{args.city}_init_windows.nc")
    if args.ocean_path is None:
        args.ocean_path = str(
            DATA_DIR / f"ocean_dlesym_2000year_{args.city}_init_windows.nc")

    for path in (args.neighbours_path, args.atmos_path, args.ocean_path):
        if not Path(path).exists():
            parser.error(f"missing input: {path}")

    if args.verify:
        ok = all(verify(Path(args.out_dir) /
                        f"region_latents_{args.city}_{tag_for(q)}.zarr", args, q)
                 for q in args.quantiles)
        raise SystemExit(0 if ok else 1)

    log(f"city={args.city} region={args.region} quantiles={args.quantiles} "
        f"rp={args.rolling_period} batch_size={args.batch_size}")
    for quantile in args.quantiles:
        build_quantile(quantile, args)
    log("All quantiles complete")


if __name__ == "__main__":
    main()
