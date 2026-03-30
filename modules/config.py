from pathlib import Path

import hydra
from omegaconf import OmegaConf


def load_config():
    config_path = Path(__file__).parent.parent / "configs"
    config_name = "config"
    with hydra.initialize_config_dir(config_dir=str(config_path), version_base="1.2"):
        cfg = hydra.compose(config_name=config_name)
    OmegaConf.set_struct(cfg, False)
    return cfg
