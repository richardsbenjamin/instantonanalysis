import logging
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from instantonanalysis.instanton.nbclosest import NClosestConfig
    

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    ...