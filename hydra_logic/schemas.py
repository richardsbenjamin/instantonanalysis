from __future__ import annotations

from hydra.core.config_store import ConfigStore

from instantonanalysis.instanton.schemas import Config
from instantonanalysis.instanton.schemas.box import LonLatBox, HealPixBox


def register_schemas():
    cs = ConfigStore.instance()

    cs.store(group="spatial", name="lonlat_schema", node=LonLatBox)
    cs.store(group="spatial", name="healpix_schema", node=HealPixBox)

    cs.store(name="base_config", node=Config)

