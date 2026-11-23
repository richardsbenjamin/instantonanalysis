# A deep learning climate emulator encodes the typicality of extreme heatwaves in its latent space

Analysis code for the paper. Everything is conditioned on the Paris box (1.25–3.75°E, 49–50°N) in
JJA and is computed on the HEALPix nside-64 grid of DLESyM.

## Setup

Two environments are needed:

- **Analysis** (everything except the forecast step): Python 3.12 with [uv](https://docs.astral.sh/uv/), `uv sync`.
- **DLESyM forecasts**: a clone of [DLESyM](https://github.com/AtmosSci-DLESM/DLESyM) at commit
  `ec2f1d5`, with its trained `models/` and `example_data/constants.nc`, and the conda environment
  in `dlesym/environment.yaml`.

Paths in the Hydra configs (`src/instantonanalysis/config/`) are absolute.

## Data

1. A 2000-year free-running DLESyM simulation initialised on 2017-01-01. Daily averages of `t2m0`
   and `z500` were calculated from it, giving `daily_atmos_dlesym_2000year.zarr`.
2. The 6-hourly atmosphere and ocean states of the same run around the Paris event dates,
   `atmos_dlesym_2000year_paris_init_windows.nc` and `ocean_dlesym_2000year_paris_init_windows.nc`.
   The event latents are initialised from these.
3. The public 100-year DLESyM run (`gs://dlesym-storage/{atmos,ocean}_hpx64_100year.zarr`). The
   climatological and non-event control latents are initialised from it.

## 1. Physical analysis (Figures 1–4, S1–S9)

```bash
python scripts/climate_calc.py            --config-name climate_config
python scripts/observable_calc.py         --config-name observable_config_paris
python scripts/nearest_neighbours_calc.py --config-name nearest_neighbours_config_paris
python scripts/instanton_calc.py          --config-name instanton_config_paris
python scripts/plot.py                    --config-name plot_config_paris
```

This writes the n = 900 event sets to `outputs/paris/`, which step 2 also reads. For n = 300 or
n = 50, re-run the last four commands with these overrides (shown for n = 300):

```bash
--overrides locations.output_folder=paris_n300                                          # observable, instanton, plot
--overrides locations.output_folder=paris_n300 "analysis.nb_closest=[300,300,300,300]"  # nearest_neighbours
```

Each run covers both windows (r = 1 and r = 5).

| Figure | File |
|---|---|
| 1 | `paris_density_plot_t2m0.png` |
| 2, S1, S4, S7 | `paris_composite_anomalies_r{1,5}.png` |
| 3, S2, S5, S8 | `paris_var_hat_t2m0_r{1,5}.png` |
| 4, S3, S6, S9 | `paris_var_hat_z500_r{1,5}.png` |

## 2. Latent extraction

Each date is initialised from the model state, run forward 96 h, and its atmospheric encoder
latents (all three levels) are captured. The latents and the physical forecast are then cut down to
the 20°×10° region around Paris. Run with the analysis environment's Python, from `dlesym/`:

```bash
python build_event_latents.py         # 900 event dates per quantile, from outputs/paris/
python build_climatology_latents.py   # 500 random JJA dates, at least 4 days from any event
bash build_control_latents.sh         # 50 random JJA non-event dates for the probe fits
```

The builders process dates in batches, delete intermediate files as they go, and resume where
they stopped. `python build_event_latents.py --verify` checks that the stored event dates still
match `outputs/paris/paris_closest_neighbours.nc`.

## 3. Latent analysis (Figures 5–7, S10–S13)

For each n in {300, 900}:

```bash
python scripts/latent_lambda.py --config-name ldt_config_l2 --overrides paths.output_dir=./outputs/latent_n300 analysis.n_closest=300
python scripts/latent_probe.py  --config-name ldt_config_l2 --overrides paths.output_dir=./outputs/latent_n300 analysis.n_closest=300 \
    paths.physical_var_hat=./outputs/paris_n300/paris_normalised_var_hat.nc
```

then `python scripts/latent_ratio.py` once both are done.

| Figure | File |
|---|---|
| 5, S10 | `latent_n{300,900}/lambda_sweep_l2.png` |
| 6, S11 | `latent_n{300,900}/probe_modes_panels_pooled_l2.png` |
| 7, S12 | `latent_n{300,900}/pushforward_vs_physical_panels_pooled_l2.png` |
| S13 | `latent/lambda_null_ratio_l2.png` |
