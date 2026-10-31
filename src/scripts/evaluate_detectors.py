import argparse
import json
from pathlib import Path

from ..eval.detector_eval import evaluate_detector
from ..data.config import OUT_DIR, REPORTS_DIR, SEED
from ..data.provenance import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt"])
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--data", default=str(OUT_DIR))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=str(REPORTS_DIR))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if not 0 <= args.seed < 2**31:
        ap.error("--seed must be between 0 and 2**31 - 1")
    if args.imgsz < 1:
        ap.error("--imgsz must be positive")
    paths = [Path(args.out_dir) / f"detection_{Path(weight).stem}_{args.split}.json"
             for weight in args.models]
    if len(set(paths)) != len(paths):
        ap.error("Model names produce duplicate report filenames")
    if any(path.exists() for path in paths):
        ap.error("Detection reports already exist, choose a new --out-dir")

    for w, out in zip(args.models, paths):
        rep = evaluate_detector(w, args.data, split=args.split,
                                imgsz=args.imgsz, device=args.device, seed=args.seed)
        write_json(out, rep)
        print(json.dumps(rep, indent=2))
        print("->", out)


if __name__ == "__main__":
    main()
