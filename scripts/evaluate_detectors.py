"""Detector mAP and precision/recall on the prepared YOLO store.

    python scripts/evaluate_detectors.py --models yolov8n.pt yolov8s.pt --split dev --device 0

Each model is run on every prepared image of the split. Predictions are compared with the GT labels
(objects in ignored regions excluded) and one report per model is written.

Needs:
    data/processed/detrac_yolo/            images/<split>, labels/<split>, ignored_regions.json (prepare_dataset)
    models/<weights>.pt                    downloaded on first use
Produces:
    <out-dir>/detection_<model>_<split>.json   mAP, per-class TP/FP/FN, precision, recall, environment
Refuses to overwrite existing reports.
"""
import argparse
import json
from pathlib import Path

from src.config import IMGSZ, OUT_DIR, REPORTS_DIR, SEED
from src.eval.detector import evaluate_detector
from src.utils import positive_int, seed_value, write_json


def parse_args() -> argparse.Namespace:
    """Command-line options; see the module docstring for what the script reads and writes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt"])
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--data", default=str(OUT_DIR))
    ap.add_argument("--imgsz", type=positive_int, default=IMGSZ)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=str(REPORTS_DIR))
    ap.add_argument("--seed", type=seed_value, default=SEED)
    return ap.parse_args()


def main() -> None:
    """Evaluate each model in turn and write its report."""
    args = parse_args()

    # One report file per model. Checked up front: evaluation takes minutes, and a name clash or an
    # existing report should stop the run before the first model starts.
    paths = [Path(args.out_dir) / f"detection_{Path(weights).stem}_{args.split}.json" for weights in args.models]
    if len(set(paths)) != len(paths):
        raise SystemExit("Model names produce duplicate report filenames")
    if any(path.exists() for path in paths):
        raise SystemExit("Detection reports already exist, choose a new --out-dir")

    for weights, out in zip(args.models, paths, strict=True):
        report = evaluate_detector(weights, args.data, split=args.split, imgsz=args.imgsz,
                                   device=args.device, seed=args.seed)
        write_json(out, report)
        print(json.dumps(report, indent=2))
        print("->", out)


if __name__ == "__main__":
    main()
