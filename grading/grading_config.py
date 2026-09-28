from pathlib import Path


# paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "pcgs_dataset"

DB_PATH = DATA_DIR / "photograde_metadata.db"
IMG_BASE_DIR = PROJECT_ROOT
GRADING_INDEX_PATH = PROJECT_ROOT / "grading_index.pkl"
PREPPED_IMG_DIR = DATA_DIR / "prepped_images"


# preprocessing
VAL_FRAC = 0.15
SEED = 42
SAMPLES_PER_RECORD = 4

IMAGE_SIZE = 384
N_COLOR_CLUSTERS = 5
RADIUS_FRAC = 0.48
BACKGROUND_DIR = None


# training
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
BACKBONE_LR = 1e-5
MAX_GRADE = 70.0
NUM_WORKERS = 4