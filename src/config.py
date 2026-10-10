from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "spam.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

KAGGLE_DATASET = "uciml/sms-spam-collection-dataset"

SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1