from pathlib import Path

SEED = 42

CLASSES = ["car", "bus", "truck"]
DETRAC_TO_ID = {"car": 0, "van": 0, "bus": 1, "others": 2}

N_SELECT, N_DEV, N_TEST = 12, 5, 7

RAW_ROOT = Path("data/raw/UA-DETRAC")
OUT_DIR = Path("data/processed/detrac_yolo")
REPORTS_DIR = Path("reports")