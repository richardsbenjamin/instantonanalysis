#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "${HERE}")"
DLESYM_DIR="${DLESYM_DIR:-/home/benjamin/DLESyM_test}"
DLESYM_PYTHON="${DLESYM_PYTHON:-/home/benjamin/miniconda3/envs/dlesym-0.1/bin/python}"
PY="${REPO_DIR}/.venv/bin/python"

"${PY}" "${HERE}/init_construct.py" \
    --no-heatwave --n_random 50 --seed 42 --months 6 8 \
    --atmos_output "${DLESYM_DIR}/example_data/atmos_reconstructed_init_nonheatwave.zarr" \
    --ocean_output "${DLESYM_DIR}/example_data/ocean_reconstructed_init_nonheatwave.zarr"

cd "${DLESYM_DIR}"
PYTHONPATH="${DLESYM_DIR}" HDF5_USE_FILE_LOCKING=False HYDRA_FULL_ERROR=1 \
"${DLESYM_PYTHON}" "${HERE}/forecast_latents.py" \
    --atmos-model-path "${DLESYM_DIR}/models/dlwp" \
    --ocean-model-path "${DLESYM_DIR}/models/dlom" \
    --non-strict \
    --lead-time 96 \
    --forecast-init-start 2017-01-19 \
    --forecast-init-end 2017-01-19 \
    --freq D \
    --time-chunk 1 \
    --step-chunk 1000 \
    --cache-dir ./output_data \
    --output-directory ./outputs \
    --atmos-output-filename atmos_hpx64_4days_nonheatwave \
    --ocean-output-filename ocean_hpx64_4days_nonheatwave \
    --data-directory "${DLESYM_DIR}/example_data" \
    --atmos-dataset-name atmos_reconstructed_init_nonheatwave \
    --ocean-dataset-name ocean_reconstructed_init_nonheatwave \
    --gpu -1 \
    --save-latent true \
    --save-levels 0 1 2 \
    --atmos-latent-out-path atmos_nonheatwave_latents_all.zarr \
    --zarr true

"${PY}" "${REPO_DIR}/scripts/materialise_region_latents.py" batch \
    --source "${DLESYM_DIR}/outputs/atmos_nonheatwave_latents_all.zarr" \
    --out "${DLESYM_DIR}/outputs/region_latents_nonheatwave.zarr"
