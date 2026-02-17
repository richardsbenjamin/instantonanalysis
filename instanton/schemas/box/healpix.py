from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List, Union

import healpy as hp
import numpy as np
import xarray as xr
from omegaconf import OmegaConf

from instantonanalysis.instanton._typing import TYPE_CHECKING
from instantonanalysis.instanton.schemas.box import IBox
from instantonanalysis.instanton.schemas.box.lonlat import LonLatBox
from instantonanalysis.instanton.schemas.xconfig import XConfig

if TYPE_CHECKING:
    from instantonanalysis.instanton._typing import xrArray


def hpxidx2fyx(nside: int, hpxidx: int) -> Tuple[int, int, int]:
    f = hpxidx // (nside**2)
    local_idx = hpxidx % (nside**2)
    
    x = 0
    y = 0
    for i in range(int(np.log2(nside))):
        x |= ((local_idx >> (2 * i)) & 1) << i
        y |= ((local_idx >> (2 * i + 1)) & 1) << i
        
    return (f, y, x)


@dataclass
class HealPixBox(IBox):
    """Dataclass for storing HealPix coordinates (12 faces, HxW grids)."""
    f_list: List[int]
    h_list: List[int]
    w_list: List[int]

    _possible_names = {
        "Face": ["face", "f", "tiles"],
        "Height": ["height", "h", "y"],
        "Width": ["width", "w", "x"]
    }

    def __post_init__(self):
        if not all(0 <= f < 12 for f in self.f_list):
            raise ValueError("HealPix faces must be in range [0, 11]")

    @staticmethod
    def from_lonlat_box(nside: int, lb: LonLatBox) -> HealPixBox:
        lons = LonLatBox.convert_lon_to_continuous([lb.lon_min, lb.lon_max], lb.lon_system)
        lats = LonLatBox.convert_lat_to_north_south([lb.lat_min, lb.lat_max], lb.lat_system)

        vertices_lon = [
            lons[0], lons[1], lons[1], lons[0],
        ]
        vertices_lat = [
            lats[0], lats[0], lats[1], lats[1],
        ]
        
        phis = np.deg2rad(vertices_lon)
        thetas = np.deg2rad(90 - np.array(vertices_lat))
        
        vecs = hp.ang2vec(thetas, phis)
        
        pix_indices = hp.query_polygon(nside, vecs, inclusive=True, nest=True)
        
        f_list, y_list, x_list = [], [], []
        for pix in pix_indices:
            f, y, x = hpxidx2fyx(nside, pix)
            f_list.append(int(f))
            y_list.append(int(y))
            x_list.append(int(x))
            
        return HealPixBox(f_list=f_list, h_list=y_list, w_list=x_list)

    def _check_bounds(self, ds: xrArray, dims: Tuple[str, str, str]):
        """Check if requested indices exist in the dataset."""
        face_dim, h_dim, w_dim = dims
        
        available_faces = ds[face_dim].values
        if not all(f in available_faces for f in self.f_list):
            raise ValueError(f"Requested faces {self.f_list} not found in dataset {available_faces}")

        h_vals = ds[h_dim].values
        if not all(h in h_vals for h in self.h_list):
             raise ValueError(f"Some height values in {self.h_list} are not in dataset")

        w_vals = ds[w_dim].values
        if not all(w in w_vals for w in self.w_list):
             raise ValueError(f"Some width values in {self.w_list} are not in dataset")

    def enforce_coords(self, ds: xrArray, xconfig: Optional[XConfig] = None) -> xrArray:
        """HealPix enforcement involves sorting indices for efficient slicing."""
        res = ds.copy()
        dims = self.get_names(res, xconfig)
        
        # HealPix is typically already indexed by integers, but we ensure order
        return res.sortby(list(dims))

    def select(self, series: xr.DataArray, dims: Tuple[str, str, str]) -> xr.DataArray:
        """The 3D implementation of the selection logic."""
        face_dim, h_dim, w_dim = dims
        return series.sel({
            face_dim: self.f_list,
            h_dim: self.h_list,
            w_dim: self.w_list
        })

