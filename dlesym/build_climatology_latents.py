from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from paths import (
    DLESYM_DIR as MODULE_DIR, FORECAST_ENV, FORECAST_SCRIPT, INIT_SCRIPT,
    MATERIALISE, PY_ANALYSIS, PY_FORECAST, REPO_DIR,
)
from init_construct import candidate_init_times

PY_INIT = PY_ANALYSIS


def log(msg: str) -> None:
    print(f"[clim-driver {pd.Timestamp.now():%H:%M:%S}] {msg}", flush=True)


def run(cmd: list[str], log_file: Path, env: dict | None = None) -> None:
    log(f"run: {' '.join(cmd[:3])} ... -> {log_file.name}")
    import os
    full_env = {**os.environ, **(env or {})}
    with open(log_file, "a") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT,
                       cwd=str(MODULE_DIR), env=full_env)


def sample_clim_dates(args, out_file: Path) -> np.ndarray:
    if out_file.exists():
        dates = np.array([np.datetime64(l.strip(), "ns")
                          for l in out_file.read_text().splitlines() if l.strip()])
        log(f"Reusing existing {out_file} ({len(dates)} dates)")
        return dates
    nn = xr.open_dataset(args.neighbours_path)
    heatwave_dates = (nn["t2m0"].sel(quantile=0.999, rolling_period=1)
                      .dropna(dim="step").step.values)
    ocean_ds = xr.open_zarr(args.ocean_path)
    candidates = candidate_init_times(
        ocean_ds, heatwave_dates, months=tuple(args.months),
        exclude_window_days=args.exclude_window_days)
    log(f"{len(candidates)} candidate JJA init times "
        f"(heatwave ±{args.exclude_window_days}d excluded)")
    rng = np.random.default_rng(args.seed)
    dates = np.sort(rng.choice(candidates, size=args.n_clim, replace=False))
    out_file.write_text("\n".join(np.datetime_as_string(dates, unit="s")) + "\n")
    log(f"Sampled {len(dates)} climatology dates -> {out_file}")
    return dates


def start_init(batch_id: str, dates: np.ndarray, work: Path,
               args) -> subprocess.Popen | None:
    atmos_out = MODULE_DIR / "example_data" / f"atmos_reconstructed_init_clim_{batch_id}.zarr"
    ocean_out = MODULE_DIR / "example_data" / f"ocean_reconstructed_init_clim_{batch_id}.zarr"
    if atmos_out.exists() and ocean_out.exists():
        log(f"{batch_id}: init zarrs already exist, skipping construction")
        return None
    for p in (atmos_out, ocean_out):
        if p.exists():
            shutil.rmtree(p)
    dates_file = work / f"clim_dates_{batch_id}.txt"
    dates_file.write_text("\n".join(np.datetime_as_string(dates, unit="s")) + "\n")
    log_file = work / f"init_{batch_id}.log"
    cmd = [PY_INIT, "-u", INIT_SCRIPT,
           "--dates-file", str(dates_file),
           "--neighbours_path", args.neighbours_path,
           "--atmos_path", args.atmos_path, "--ocean_path", args.ocean_path,
           "--atmos_output", str(atmos_out), "--ocean_output", str(ocean_out)]
    log(f"{batch_id}: starting init construction ({len(dates)} dates, background)")
    f = open(log_file, "a")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(MODULE_DIR))


def forecast_batch(batch_id: str, work: Path) -> Path:
    latent_out = MODULE_DIR / "outputs" / f"atmos_clim_latents_{batch_id}.zarr"
    if latent_out.exists():
        shutil.rmtree(latent_out)
    cmd = [PY_FORECAST, FORECAST_SCRIPT,
           "--atmos-model-path", str(MODULE_DIR / "models" / "dlwp"),
           "--ocean-model-path", str(MODULE_DIR / "models" / "dlom"),
           "--non-strict", "--lead-time", "96",
           "--forecast-init-start", "2017-01-19", "--forecast-init-end", "2017-01-19",
           "--freq", "D", "--time-chunk", "1", "--step-chunk", "1000",
           "--cache-dir", "./output_data", "--output-directory", "./outputs",
           "--atmos-output-filename", f"atmos_hpx64_4days_clim_{batch_id}",
           "--ocean-output-filename", f"ocean_hpx64_4days_clim_{batch_id}",
           "--data-directory", str(MODULE_DIR / "example_data"),
           "--atmos-dataset-name", f"atmos_reconstructed_init_clim_{batch_id}",
           "--ocean-dataset-name", f"ocean_reconstructed_init_clim_{batch_id}",
           "--gpu", "-1", "--save-latent", "true", "--save-levels", "0", "1", "2",
           "--atmos-latent-out-path", latent_out.name, "--zarr", "true"]
    run(cmd, work / f"forecast_{batch_id}.log", env=FORECAST_ENV)
    if not latent_out.exists():
        raise RuntimeError(f"{batch_id}: forecast finished but {latent_out} is missing")
    return latent_out


def batch_already_appended(clim_zarr: Path, first_date: np.datetime64) -> bool:
    if not clim_zarr.exists():
        return False
    times = xr.open_zarr(str(clim_zarr)).time.values
    return np.datetime64(first_date, "ns") in set(np.asarray(times, "datetime64[ns]"))


def cleanup_batch(batch_id: str) -> None:
    for p in [MODULE_DIR / "example_data" / f"atmos_reconstructed_init_clim_{batch_id}.zarr",
              MODULE_DIR / "example_data" / f"ocean_reconstructed_init_clim_{batch_id}.zarr",
              MODULE_DIR / "outputs" / f"atmos_hpx64_4days_clim_{batch_id}.nc",
              MODULE_DIR / "outputs" / f"ocean_hpx64_4days_clim_{batch_id}.nc"]:
        if p.exists():
            shutil.rmtree(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-clim", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123,
                        help="Clim sampling seed (distinct from the non-heatwave "
                             "null's seed 42 so the two sets are independent draws).")
    parser.add_argument("--months", type=int, nargs=2, default=[6, 8])
    parser.add_argument("--exclude-window-days", type=int, default=4)
    parser.add_argument("--neighbours_path",
                        default=str(REPO_DIR / "outputs" / "paris" / "paris_closest_neighbours.nc"))
    parser.add_argument("--atmos_path", default="gs://dlesym-storage/atmos_hpx64_100year.zarr")
    parser.add_argument("--ocean_path", default="gs://dlesym-storage/ocean_hpx64_100year.zarr")
    parser.add_argument("--clim-out", default=str(MODULE_DIR / "outputs" / "clim_latents.zarr"))
    args = parser.parse_args()

    work = MODULE_DIR / "outputs" / "clim_state"
    work.mkdir(parents=True, exist_ok=True)

    dates = sample_clim_dates(args, work / "clim_dates_all.txt")
    batches = [dates[i:i + args.batch_size] for i in range(0, len(dates), args.batch_size)]
    log(f"{len(batches)} batches of <= {args.batch_size} dates")

    pending = [(f"b{k:02d}", b) for k, b in enumerate(batches)
               if not (work / f"b{k:02d}.done").exists()]
    done = len(batches) - len(pending)
    if done:
        log(f"Resuming: {done} batches already complete")

    init_proc: subprocess.Popen | None = None
    for i, (batch_id, batch_dates) in enumerate(pending):
        if i == 0:
            init_proc = start_init(batch_id, batch_dates, work, args)
        if init_proc is not None and init_proc.wait() != 0:
            raise RuntimeError(f"{batch_id}: init construction failed "
                               f"(see {work}/init_{batch_id}.log)")
        init_proc = (start_init(*pending[i + 1][:2], work, args)
                     if i + 1 < len(pending) else None)

        log(f"{batch_id}: forecasting ({len(batch_dates)} dates, all levels)")
        latent_zarr = forecast_batch(batch_id, work)

        if batch_already_appended(Path(args.clim_out), batch_dates[0]):
            log(f"{batch_id}: already in clim_latents, deleting duplicates source")
            shutil.rmtree(latent_zarr)
        else:
            log(f"{batch_id}: region-filtering + appending to {args.clim_out}")
            run([PY_ANALYSIS, MATERIALISE, "batch",
                 "--source", str(latent_zarr), "--out", args.clim_out,
                 "--delete-source"], work / f"materialise_{batch_id}.log")
        cleanup_batch(batch_id)
        (work / f"{batch_id}.done").touch()
        log(f"{batch_id}: done ({done + i + 1}/{len(batches)})")

    log(f"Climatology complete: {args.clim_out}")
