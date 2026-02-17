from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from instantonanalysis.instanton.schemas import Config
from hydra.utils import instantiate
from omegaconf import DictConfig
from instantonanalysis.instanton.schemas.box.healpix import HealPixBox 
import instantonanalysis.instanton.utils # Ensure resolvers registered


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)

    test_files = ["calc_config", "calc_config_healpix"]

    for test_file in test_files:
        with initialize(version_base=None, config_path="../config"):
            cfg = compose(config_name=test_file, overrides=None)

            box = instantiate(cfg.box)
            if isinstance(box, DictConfig) and "_target_" in box:
                # Fallback: if resolution returned a DictConfig (which can happen with resolvers returning dicts),
                # we need to instantiate it again or ensure it's treated as a target.
                box = instantiate(box)
                
            print(f"File: {test_file}, Box Type: {type(box)}")
            print(f"Box Content: {box}")

            if test_file == "calc_config_healpix" and not isinstance(box, HealPixBox):
                raise ValueError("HealPixBox expected for calc_config_healpix")
                
