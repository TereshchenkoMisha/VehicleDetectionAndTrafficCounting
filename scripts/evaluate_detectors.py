"""Detector mAP and precision/recall on the prepared YOLO store.

    python scripts/evaluate_detectors.py --models yolov8n.pt yolov8s.pt --split dev --device 0
"""
import argparse
import json
from pathlib import Path

from src.config import IMGSZ, OUT_DIR, REPORTS_DIR, SEED
from src.eval.detector import evaluate_detector
from src.utils import positive_int, seed_value, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt"])
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--data", default=str(OUT_DIR))
    ap.add_argument("--imgsz", type=positive_int, default=IMGSZ)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=str(REPORTS_DIR))
    ap.add_argument("--seed", type=seed_value, default=SEED)
    args = ap.parse_args()

    paths = [Path(args.out_dir) / f"detection_{Path(weight).stem}_{args.split}.json" for weight in args.models]
    if len(set(paths)) != len(paths):
        ap.error("Model names produce duplicate report filenames")
    if any(path.exists() for path in paths):
        ap.error("Detection reports already exist, choose a new --out-dir")

    for weights, out in zip(args.models, paths):
        report = evaluate_detector(weights, args.data, split=args.split, imgsz=args.imgsz,
                                   device=args.device, seed=args.seed)
        write_json(out, report)
        print(json.dumps(report, indent=2))
        print("->", out)


if __name__ == "__main__":
    main()
