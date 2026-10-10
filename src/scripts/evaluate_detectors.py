import argparse
import json
from pathlib import Path

from ..eval.detector_eval import evaluate_detector
from ..data.config import OUT_DIR, REPORTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt"])
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--data", default=str(OUT_DIR))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    for w in args.models:
        rep = evaluate_detector(w, args.data, split=args.split,
                                imgsz=args.imgsz, device=args.device)
        out = REPORTS_DIR / f"detection_{Path(w).stem}_{args.split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2))
        print("->", out)


if __name__ == "__main__":
    main()