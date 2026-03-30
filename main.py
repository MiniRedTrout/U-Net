import os
import sys

from clearml import Task
from google.colab import userdata

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

if __name__ == '__main__':
    model, trainer = train()

