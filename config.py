import os
import warnings
from pathlib import Path

BASE_PATH = Path().resolve()
DATA_PATH = BASE_PATH / 'data'
RAW_DATA_PATH = BASE_PATH / 'raw_data'
MODELS_PATH = BASE_PATH / 'models'

DATA_PATH.mkdir(exist_ok=True)
RAW_DATA_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)

os.environ['PYTHONPATH'] = BASE_PATH.as_posix()
warnings.filterwarnings("ignore")
