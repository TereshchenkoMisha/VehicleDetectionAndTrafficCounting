"""Paths and experiment constants. Every path is built from the repository root, not the current directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = ROOT / "data/raw/UA-DETRAC"
OUT_DIR = ROOT / "data/processed/detrac_yolo"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "configs"
TRACKS_DIR = ROOT / "outputs/tracks"

SEED = 42
N_SELECT, N_DEV, N_TEST = 12, 5, 7

CLASSES = ["car", "bus", "truck"]
DETRAC_TO_ID = {"car": 0, "van": 0, "bus": 1, "others": 2}
COCO_TO_LOCAL = {2: 0, 5: 1, 7: 2}          # COCO car, bus, truck -> local ids
COCO_CLASSES = list(COCO_TO_LOCAL)

IMGSZ = 960
CONF = 0.25                  # tracker's high-score threshold and the P/R operating point
NMS_IOU = 0.7
MAP_CONF = 0.001             # detector confidence used for mAP
FPS = 25                     # UA-DETRAC frame rate (video output only)


def weights_path(name) -> str:
    """Pretrained weights live in models/ (git-ignored); Ultralytics downloads them there if missing.

    A name with a directory part (e.g. a custom checkpoint path) is left untouched.
    """
    p = Path(name)
    if p.parent != Path("."):
        return str(p)
    MODELS_DIR.mkdir(exist_ok=True)
    return str(MODELS_DIR / p.name)
