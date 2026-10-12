from pathlib import Path

SEED = 42

CLASSES = ["car", "bus", "truck"]
DETRAC_TO_ID = {"car": 0, "van": 0, "bus": 1, "others": 2}

N_SELECT, N_DEV, N_TEST = 12, 5, 7

RAW_ROOT = Path("data/raw/UA-DETRAC")
OUT_DIR = Path("data/processed/detrac_yolo")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")


def weights_path(name) -> str:
    """Pretrained weights live in models/ (git-ignored); Ultralytics downloads them there if missing.

    A name with a directory part (e.g. a custom checkpoint path) is left untouched.
    """
    p = Path(name)
    if p.parent != Path("."):
        return str(p)
    MODELS_DIR.mkdir(exist_ok=True)
    return str(MODELS_DIR / p.name)
