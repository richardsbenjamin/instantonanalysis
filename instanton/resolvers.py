from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from instantonanalysis.instanton.schemas.box.lonlat import LonLatBox
from instantonanalysis.instanton.schemas.box.healpix import HealPixBox


def resolve_arange(start, stop, step):
    return np.arange(start, stop, step).tolist()

def resolve_healpix_from_config(nside: int, box_conf: dict | OmegaConf) -> dict:
    """Resolver that accepts a config object/dict for the box parameters."""
    
    lon_sys = box_conf.get("lon_sys") or box_conf.get("lon_system")
    lat_sys = box_conf.get("lat_sys") or box_conf.get("lat_system")
    
    lb = LonLatBox(
        lon_min=box_conf["lon_min"],
        lon_max=box_conf["lon_max"],
        lat_min=box_conf["lat_min"],
        lat_max=box_conf["lat_max"],
        lon_system=lon_sys,
        lat_system=lat_sys,
    )
    
    box = HealPixBox.from_lonlat_box(nside, lb)
    return {
        "_target_": "instantonanalysis.instanton.schemas.box.healpix.HealPixBox",
        "f_list": box.f_list,
        "h_list": box.h_list,
        "w_list": box.w_list,
    }

def resolve_healpix_from_lonlat(
    nside: int,
    lon_min: float, lon_max: float, 
    lat_min: float, lat_max: float, 
    lon_sys: str, lat_sys: str
) -> dict:
    """ Resolver to create a HealPixBox config from LonLat coordinates.
    Returns a dictionary with _target_ for Hydra instantiation.
    """
    lb = LonLatBox(
        lon_min=lon_min, lon_max=lon_max,
        lat_min=lat_min, lat_max=lat_max,
        lon_system=lon_sys,
        lat_system=lat_sys
    )
    box = HealPixBox.from_lonlat_box(nside, lb)
    return {
        "_target_": "instantonanalysis.instanton.schemas.box.healpix.HealPixBox",
        "f_list": box.f_list,
        "h_list": box.h_list,
        "w_list": box.w_list,
    }

def resolve_latlon_from_cfg(cfg_box: dict | OmegaConf) -> dict:
    return LonLatBox(
        lon_min=cfg_box["lon_min"],
        lon_max=cfg_box["lon_max"],
        lat_min=cfg_box["lat_min"],
        lat_max=cfg_box["lat_max"],
        lon_system=cfg_box.get("lon_sys") or cfg_box.get("lon_system"),
        lat_system=cfg_box.get("lat_sys") or cfg_box.get("lat_system"),
    )

resolvers = {
    "healpix_from_lonlat": resolve_healpix_from_lonlat,
    "healpix_from_config": resolve_healpix_from_config,
    "arange": resolve_arange,
    "latlon_from_cfg": resolve_latlon_from_cfg,
}

def register_resolvers():
    for resolver_name, resolver_func in resolvers.items():
        if not OmegaConf.has_resolver(resolver_name):
            OmegaConf.register_new_resolver(resolver_name, resolver_func)


register_resolvers()
