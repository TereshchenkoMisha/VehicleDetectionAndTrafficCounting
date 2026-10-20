"""Detector + tracker loop with separate timing and ignored-region filtering (plan 2.6, 3.1, 3.4).

The detector (`model.predict`) and the tracker (`tracker.update`) are called separately, instead of
`model.track`, so that (a) their times can be measured independently and (b) detections whose centre
lies in an ignored region are dropped BEFORE they reach the tracker - the same rule the detector eval uses.
"""
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace

from ..data.config import RAW_ROOT
from ..data.detrac_parser import find_sequence_dir

COCO_CLASSES = [2, 5, 7]                      # car, bus, truck
COCO_ID_TO_LOCAL = {2: 0, 5: 1, 7: 2}
TRACKER_CLASSES = {"bytetrack": BYTETracker, "botsort": BOTSORT}
CONFIG_DIR = Path("configs")


def build_tracker(name: str, device=None):
    cfg = IterableSimpleNamespace(**YAML.load(CONFIG_DIR / f"{name}.yaml"))
    cfg.device = device
    return TRACKER_CLASSES[cfg.tracker_type](args=cfg)


def load_ignored(path) -> dict:
    return json.loads(Path(path).read_text())


def list_frames(seq_name: str, raw_root=RAW_ROOT):
    """Full-rate frames of a sequence as [(frame_id, path)], sorted by frame id (1-based, as in DETRAC)."""
    seq_dir = find_sequence_dir(raw_root, seq_name)
    if seq_dir is None:
        raise FileNotFoundError(f"frames of {seq_name} not found under {raw_root}")
    frames = [(int(re.sub(r"\D", "", p.stem)), p) for p in seq_dir.glob("*.jpg")]
    return sorted(frames)


def keep_mask(xywh: np.ndarray, regions) -> np.ndarray:
    """True for detections whose centre is NOT inside any ignored region."""
    keep = np.ones(len(xywh), dtype=bool)
    for x1, y1, x2, y2 in regions:
        keep &= ~((xywh[:, 0] >= x1) & (xywh[:, 0] <= x2) & (xywh[:, 1] >= y1) & (xywh[:, 1] <= y2))
    return keep


def _sync(device):
    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.synchronize()


def warmup(model, frame, imgsz, device, n=10):
    for _ in range(n):
        model.predict(frame, imgsz=imgsz, conf=0.25, classes=COCO_CLASSES, device=device, verbose=False)
    _sync(device)


def run_sequence(model, tracker, frames, regions, conf, imgsz, iou, device, on_frame=None, writer=None):
    """Run detector+tracker over a sequence.

    Returns (mot_rows, det_s, trk_s). MOT row: (frame, id, x, y, w, h, score, cls) with cls = local class id.
    """
    rows, det_s, trk_s = [], 0.0, 0.0
    for fid, path in frames:
        img = cv2.imread(str(path))                                  # image I/O is not timed

        t0 = time.perf_counter()
        res = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, classes=COCO_CLASSES,
                            device=device, verbose=False)[0]
        _sync(device)
        det = res.boxes.cpu().numpy()
        t1 = time.perf_counter()

        if len(det):
            det = det[keep_mask(det.xywh, regions)]
        tracks = tracker.update(det, img)                            # [x1,y1,x2,y2,id,score,cls,idx]
        t2 = time.perf_counter()

        det_s += t1 - t0
        trk_s += t2 - t1
        for x1, y1, x2, y2, tid, score, cls, _ in tracks:
            rows.append((fid, int(tid), x1, y1, x2 - x1, y2 - y1, score, COCO_ID_TO_LOCAL[int(cls)]))
        if writer is not None:
            writer.write(_draw(img, tracks, regions, fid))
        if on_frame:
            on_frame(det_s, trk_s)
    return rows, det_s, trk_s


def write_mot(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for fid, tid, x, y, w, h, s, c in rows:
            f.write(f"{fid},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.4f},{c},-1,-1\n")


def _color(tid):
    rng = np.random.RandomState(int(tid) * 7919 % 2**31)
    return tuple(int(v) for v in rng.randint(60, 255, 3))


def _draw(img, tracks, regions, fid):
    shade = img.copy()
    for x1, y1, x2, y2 in regions:
        cv2.rectangle(shade, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), -1)
    out = cv2.addWeighted(shade, 0.35, img, 0.65, 0)
    for x1, y1, x2, y2, tid, score, cls, _ in tracks:
        c = _color(tid)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.putText(out, f"{int(tid)}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    cv2.putText(out, f"frame {fid}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out
