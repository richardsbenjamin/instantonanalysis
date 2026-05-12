from __future__ import annotations

import sys

from hydra import initialize, compose
from hydra.core.config_store import ConfigStore

import instantonanalysis.hydra_logic
from instantonanalysis.instanton._typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Optional

    from omegaconf import DictConfig, Node


def load_config(
        config_name: str = "config", 
        overrides: Optional[List[str]] = [],
        schema_node: Optional[Node] = None,
    ) -> DictConfig:
    if overrides:
        overrides = sys.argv[1:]
    if schema_node is not None:
        cs = ConfigStore.instance()
        cs.store(name=config_name, node=schema_node)
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name=config_name, overrides=overrides)
        return cfg