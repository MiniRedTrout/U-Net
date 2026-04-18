import os

import hydra
import kagglehub
from clearml import Task
from omegaconf import DictConfig

from modules.train import train

access_key = os.getenv('CLEARML_API_ACCESS_KEY')
secret_key = os.getenv('CLEARML_API_SECRET_KEY')

Task.set_credentials(
    api_host="https://api.clear.ml",
    web_host="https://app.clear.ml",
    files_host="https://files.clear.ml",
    key=access_key,
    secret=secret_key
)
path = kagglehub.dataset_download("zaryabahmadkhan/2d-slicing-of-imagetbad-dataset")
@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
  cfg.data.original_dir = path
  model, trainer = train(cfg)
if __name__ == '__main__':
    main()

