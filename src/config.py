"""Paths and experiment constants shared by all modules.

Everything that more than one module needs to agree on lives here: where the data sits on disk,
which classes exist, how COCO class ids map to the project's ids, and the detector/tracker
thresholds. Changing a value here changes it for the whole pipeline (detector evaluation,
tracking, counting), which is the point: all stages must use the same protocol.

Paths are built from the repository root, so scripts work no matter which directory they are
started from.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT: Path = Path(__file__).resolve().parents[1]        # this file is <root>/src/config.py

RAW_ROOT: Path = ROOT / "data/raw/UA-DETRAC"            # downloaded dataset (frames, XML)
OUT_DIR: Path = ROOT / "data/processed/detrac_yolo"     # YOLO-format copy used by the detector evaluation
REPORTS_DIR: Path = ROOT / "reports"                    # committed results and reference inputs
MODELS_DIR: Path = ROOT / "models"                      # pretrained weights
CONFIG_DIR: Path = ROOT / "configs"                     # tracker yaml files
TRACKS_DIR: Path = ROOT / "outputs/tracks"              # tracker output, one folder per configuration

# ---------------------------------------------------------------- split
SEED: int = 42
N_SELECT: int = 12           # sequences taken from the whole dataset
N_DEV: int = 5               # of them go to dev, the rest to test
N_TEST: int = N_SELECT - N_DEV

# ---------------------------------------------------------------- classes
# Project class ids are 0..2 in the order of CLASSES. DETRAC has no truck class: "others" is
# mapped to truck (a mixed bag, so truck numbers are noisy), "van" is merged into car because COCO
# has no van.
CLASSES: list[str] = ["car", "bus", "truck"]
DETRAC_TO_ID: dict[str, int] = {"car": 0, "van": 0, "bus": 1, "others": 2}
# The pretrained detectors speak COCO ids (2 car, 5 bus, 7 truck); this maps them to project ids.
COCO_TO_LOCAL: dict[int, int] = {2: 0, 5: 1, 7: 2}
COCO_CLASSES: list[int] = list(COCO_TO_LOCAL)           # passed to the detector to ignore all other COCO classes

# ---------------------------------------------------------------- detector and tracker protocol
IMGSZ: int = 960
NMS_IOU: float = 0.7
FPS: int = 25                # UA-DETRAC frame rate, used for output videos only

# Two confidence thresholds, for two different jobs:
#   CONF      a "decision" score. Precision/recall are computed at it, and the trackers use it as their
#             high-score threshold: only detections above it may start a track or be matched first.
#   MAP_CONF  a "collection" score. mAP integrates precision over the whole score range, so the detector
#             must keep nearly all boxes; a higher value would cut the curve and lower mAP.
# For tracking the detector runs lower still, at the tracker's low threshold (see
# tracking/pipeline.py `detector_conf`), so the trackers also get the low-score boxes they need.
CONF: float = 0.25
MAP_CONF: float = 0.001


def weights_path(name: str) -> str:
    """Resolve a weights name to a path inside models/.

    A bare name such as "yolov8n.pt" becomes "<root>/models/yolov8n.pt" and the models/ folder is
    created, so Ultralytics downloads a missing file there instead of into the working directory.
    A name that already has a folder part (a custom checkpoint) is returned unchanged.
    """
    path = Path(name)
    if path.parent != Path("."):
        return str(path)
    MODELS_DIR.mkdir(exist_ok=True)
    return str(MODELS_DIR / path.name)
