import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("AISCOPE_RAW", REPO / "dataset"))
DATA_DIR = Path(os.environ.get("AISCOPE_DATA", REPO / "data"))
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = REPO / "models"
