import itertools
from tqdm import tqdm

import numpy as np
import healpy as hp
import xarray as xr
import reproject as rp
import astropy as ap

import matplotlib.pyplot as plt

def healpix_to_latlon(
        fcst: xr.Dataset,
        var: str | list[str],
        spatial_dims: list[str] = ["face", "height", "width"],
        nside: int = 64,
    ):
    mapper = HEALPixRemap(latitudes=181, longitudes=360, nside=nside)

    if var is None:
        raise ValueError(f"No variable specified. Available: {list(fcst.data_vars)}")
    vars_ = [var] if isinstance(var, str) else var

    missing = [v for v in vars_ if v not in fcst.data_vars]
    if missing:
        raise ValueError(f"Variables {missing} not found. Available: {list(fcst.data_vars)}")

    remapped = {}
    for v in vars_:
        da = fcst[v].squeeze()
        other_dims  = {dim: da[dim] for dim in da.dims if dim not in spatial_dims}
        other_shape = [len(c) for c in other_dims.values()]
        buf         = np.zeros(other_shape + [181, 360], dtype=np.float32)

        for idx_combo, val_combo in tqdm(
            zip(itertools.product(*[range(s) for s in other_shape]),
                itertools.product(*other_dims.values())),
            total=int(np.prod(other_shape)),
            desc=f"Remapping {v}",
            unit="slice",
        ):
            select_ = dict(zip(other_dims.keys(), val_combo))
            buf[idx_combo] = mapper.hpx2ll(da.sel(**select_).values)

        remapped[v] = xr.DataArray(
            buf,
            dims=[*other_dims.keys(), 'lat', 'lon'],
            coords={
                **{dim: da[dim] for dim in other_dims},
                'lat': np.arange(90, -90.1, -1),
                'lon': np.arange(0, 360, 1),
            },
        )

    return xr.Dataset(remapped)


class HEALPixRemap:
    def __init__(
            self,
            latitudes: int,
            longitudes: int,
            nside: int,
            order: str = "bilinear",
            resolution_factor: float = 1.0,
            verbose: bool = True
            ):
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.nside = nside
        self.order = order
        self.nested = True
        self.verbose = verbose

        resolution = 360./longitudes
        self.npix = hp.nside2npix(nside)

        wcs_input_dict = {
            'CTYPE1': 'RA---CAR',
            'CUNIT1': 'deg',
            'CDELT1': -resolution*resolution_factor,
            'CRPIX1': (longitudes)/2,
            'CRVAL1': 180.0,
            'NAXIS1': longitudes,
            'CTYPE2': 'DEC--CAR',
            'CUNIT2': 'deg',
            'CDELT2': -resolution,
            'CRPIX2': (latitudes+1)/2,
            'CRVAL2': 0.0,
            'NAXIS2': latitudes
        }
        self.wcs_ll2hpx = ap.wcs.WCS(wcs_input_dict)

        wcs_input_dict = {
            'CTYPE1': 'RA---CAR',
            'CUNIT1': 'deg',
            'CDELT1': resolution*resolution_factor,
            'CRPIX1': (longitudes)/2,
            'CRVAL1': 179.0,
            'NAXIS1': longitudes,
            'CTYPE2': 'DEC--CAR',
            'CUNIT2': 'deg',
            'CDELT2': resolution,
            'CRPIX2': (latitudes+1)/2,
            'CRVAL2': 0.0,
            'NAXIS2': latitudes
        }
        self.wcs_hpx2ll = ap.wcs.WCS(wcs_input_dict)


    def hpx2ll(self, data: np.array, visualise: bool = False, **kwargs) -> np.array:
        data = data[[8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]]

        hpx1d = self.hpx3d2hpx1d(hpx3d=data)

        ll2d, ll2d_mask = rp.reproject_from_healpix(
            input_data=(hpx1d, "icrs"),
            output_projection=self.wcs_hpx2ll,
            shape_out=(self.latitudes, self.longitudes),
            nested=self.nested
            )

        if visualise:
            plt.imshow(ll2d, **kwargs)
            plt.title("HPX mapped to LL")
            plt.tight_layout()
            plt.savefig("hpx2ll.pdf", format="pdf")

        assert ll2d_mask.all(), ("Found NaN in the projected data. This can occur when the resolution of the "
                                 "HEALPix data is smaller than that of the target latlon grid.")
        return ll2d

    def hpx3d2hpx1d(self, hpx3d: np.array, dtype: np.dtype = np.float32) -> np.array:
        hpx1d = np.zeros(self.npix, dtype=dtype)
        for f in range(12):
            for y in range(self.nside):
                for x in range(self.nside):
                    hpxidx = self.fyx2hpxidx(f=f, y=y, x=x)
                    hpx1d[hpxidx] = hpx3d[f, y, x]
        return hpx1d

    def fyx2hpxidx(self, f: int, x: int, y: int) -> int:
        bits_eve = format(y, "b").zfill(self.nside//2)
        bits_odd = format(x, "b").zfill(self.nside//2)

        bitstring = ""
        for bit_idx in range(len(bits_eve)):
            bitstring += bits_eve[bit_idx]
            bitstring += bits_odd[bit_idx]

        return int(bitstring, 2) + f*self.nside**2
