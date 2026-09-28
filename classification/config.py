from pathlib import Path

# paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

SCRAPED_JSON_PATH = DATA_DIR / "ngc_coins.json"
IMG_CACHE_DIR = DATA_DIR / "images"
DATASET_INDEX_PATH = DATA_DIR / "dataset_index.pkl"
CHECKPOINT_ROOT = ROOT_DIR / "runs"

# preprocessing
CANVAS_SIZE = 512
HELDOUT_FRAC = 0.15
SEED = 42
MAX_WORKERS = 16