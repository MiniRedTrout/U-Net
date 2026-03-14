import hydra
from omegaconf import DictConfig, OmegaConf
import os
def load_config():
    config_path = "/content/drive/MyDrive/segmentation_project/config"
    config_name = "config"
    with hydra.initialize_config_dir(config_dir=config_path, version_base="1.2"):
        cfg = hydra.compose(config_name=config_name)
    OmegaConf.set_struct(cfg, False)
    return cfg