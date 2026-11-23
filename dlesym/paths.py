import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_DIR = HERE.parent
DLESYM_DIR = Path(os.environ.get("DLESYM_DIR", "/home/benjamin/DLESyM_test"))
DATA_DIR = Path(os.environ.get("DLESYM_DATA_DIR", "/home/benjamin/data/dlesym-2000yr-forecast"))
PY_ANALYSIS = sys.executable
PY_FORECAST = os.environ.get("DLESYM_PYTHON", "/home/benjamin/miniconda3/envs/dlesym-0.1/bin/python")
INIT_SCRIPT = str(HERE / "init_construct.py")
FORECAST_SCRIPT = str(HERE / "forecast_latents.py")
MATERIALISE = str(REPO_DIR / "scripts" / "materialise_region_latents.py")
FORECAST_ENV = {"PYTHONPATH": str(DLESYM_DIR), "HDF5_USE_FILE_LOCKING": "False", "HYDRA_FULL_ERROR": "1"}
