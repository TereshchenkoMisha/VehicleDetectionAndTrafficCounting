"""Replay the detectors on the frames around one failure case, to see which boxes they produce there.

    python scripts/probe_detector.py --analysis reports/failure_analysis_dev.json --case-id <case id> --device 0

Every detector used by the saved runs is run again on the frames `crossing frame + offset`, with the
settings and the exact weights recorded in the run's timing.json. The result lists each box with its
score and class, and whether it lies in an ignored region. This is a diagnostic replay of isolated frames:
the raw detections of the original run were not saved, the tracker is not run, and on another GPU or library
version the scores can differ slightly. It shows what the detector can see, not what the tracker did with it.

Needs:
    <analysis>                             from analyze_failures
    outputs/tracks/<split>/<configuration>/timing.json    settings and weights of the runs
    models/<weights>.pt                    the weights recorded there (checked by SHA-256)
    data/raw/UA-DETRAC/                    frames
    reports/ignored_regions.json
Produces:
    <out-dir>/<case id>.json               boxes, scores and classes per detector and frame
"""
import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.config import CLASSES, COCO_CLASSES, COCO_TO_LOCAL, RAW_ROOT, REPORTS_DIR, ROOT, TRACKS_DIR
from src.data.regions import box_centres, inside_any
from src.data.sequences import list_frames, read_frame
from src.eval.validation import run_dirs
from src.failure.diagnosis import event_of, find_case
from src.utils import env_info, file_info, read_json, set_seed, write_json


def parse_args() -> argparse.Namespace:
    """Command-line options; see the module docstring for what the script reads and writes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True, help="failure_analysis_<split>.json")
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--offsets", nargs="+", type=int, default=[-2, -1, 0, 1, 2], help="frames around the crossing")
    ap.add_argument("--tracks-dir", default=str(TRACKS_DIR))
    ap.add_argument("--ignored", default=str(REPORTS_DIR / "ignored_regions.json"))
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/failure_probes"))
    ap.add_argument("--device", default="cpu", help="e.g. 0 or cpu")
    return ap.parse_args()


def probe_detector(timing: dict[str, Any], frames: dict[int, Path], regions: list[list[float]],
                   device: str) -> list[dict[str, Any]]:
    """Run the detector of one saved run on `frames` with that run's settings; one record per frame."""
    weights = timing["provenance"]["weights"]
    if file_info(weights["path"])["sha256"] != weights["sha256"]:
        raise SystemExit(f"The weights changed since the run: {weights['path']}")
    set_seed(timing["seed"])
    model = YOLO(weights["path"])

    records = []
    for frame, path in frames.items():
        result = model.predict(read_frame(path), imgsz=timing["imgsz"], conf=timing["detector_conf"],
                               iou=timing["iou"], classes=COCO_CLASSES, device=device, verbose=False,
                               agnostic_nms=timing["agnostic_nms"])[0]
        boxes = result.boxes.cpu().numpy()
        ignored = inside_any(box_centres(boxes.xyxy), regions)
        records.append({"frame": frame, "detections": [
            {"box": box.tolist(), "score": float(score), "class": CLASSES[COCO_TO_LOCAL[int(cls)]],
             "in_ignored_region": bool(skip)}
            for box, score, cls, skip in zip(boxes.xyxy, boxes.conf, boxes.cls, ignored, strict=True)]})
    return records


def main() -> None:
    """Replay every detector of the saved runs on the frames of the case and write the boxes."""
    args = parse_args()
    analysis = read_json(args.analysis)
    case = find_case(analysis["rows"], args.case_id)
    sequence, center = case["sequence"], event_of(case)["frame"]

    available = dict(list_frames(sequence, args.raw))
    wanted = sorted({center + offset for offset in args.offsets})
    if set(wanted) - set(available):
        raise SystemExit(f"Frames outside the sequence {sequence}: {sorted(set(wanted) - set(available))}")
    frames = {frame: available[frame] for frame in wanted}
    regions = read_json(args.ignored)[sequence]

    # One replay per detector, not per configuration: the two trackers of a detector share its weights
    timings: dict[str, dict[str, Any]] = {}
    for run in run_dirs(Path(args.tracks_dir) / analysis["split"]):
        timing = read_json(run / "timing.json")
        timings.setdefault(timing["detector"], timing)

    result: dict[str, Any] = {"case_id": args.case_id, "frames": wanted, "device": args.device,
                              "env": env_info(args.device), "detectors": {}}
    for detector, timing in timings.items():
        print(f"[probe] {detector} on {len(frames)} frames of {sequence}")
        result["detectors"][detector] = probe_detector(timing, frames, regions, args.device)
    out_path = Path(args.out_dir) / f"{args.case_id}.json"
    write_json(out_path, result)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
