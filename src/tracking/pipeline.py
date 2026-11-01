"""Detector + tracker loop with separate timing and ignored-region filtering.

The detector (`model.predict`) and the tracker (`tracker.update`) are called separately, instead of
`model.track`, so that (a) their times are measured independently and (b) detections whose centre lies in an
ignored region are dropped BEFORE they reach the tracker, the same rule the detector evaluation uses.

Detector confidence: both trackers need low-score detections (`track_low_thresh`) for their second
association step, so the detector always runs at that low threshold. The tracker itself applies
`track_high_thresh` (our `conf`) to decide which detections start and extend tracks.
"""
import time

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML

from ..config import COCO_CLASSES, COCO_TO_LOCAL, CONFIG_DIR, CONF, weights_path
from ..data.mot import write_mot
from ..data.regions import inside_any
from ..data.sequences import read_frame
from ..utils import file_info, set_seed
from .viz import draw_tracks, video_writer

TRACKER_CLASSES = {"bytetrack": BYTETracker, "botsort": BOTSORT}


def tracker_settings(name: str, conf: float = CONF) -> dict:
    """configs/<name>.yaml with `conf` as the high-score threshold for starting and extending tracks."""
    settings = YAML.load(CONFIG_DIR / f"{name}.yaml")
    settings["track_high_thresh"] = settings["new_track_thresh"] = conf
    return settings


def detector_conf(settings: dict) -> float:
    """Lowest score any tracker stage can use."""
    return min(settings["track_low_thresh"], settings["track_high_thresh"])


def build_tracker(settings: dict, device=None):
    cfg = IterableSimpleNamespace(**settings)
    cfg.device = device
    return TRACKER_CLASSES[cfg.tracker_type](args=cfg)


def keep_mask(xywh: np.ndarray, regions) -> np.ndarray:
    """True for detections whose centre (x, y of xywh) is NOT inside any ignored region."""
    return ~inside_any(xywh[:, :2], regions)


def sync(device):
    """Wait for queued GPU work, so that time.perf_counter() measures finished work."""
    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.synchronize(device=int(device) if str(device).isdigit() else device)


def warmup(model, frame, imgsz, device, conf, n=10):
    for _ in range(n):
        model.predict(frame, imgsz=imgsz, conf=conf, classes=COCO_CLASSES, device=device, verbose=False)
    sync(device)


def run_sequence(model, tracker, frames, regions, conf, imgsz, iou, device, on_frame=None, writer=None):
    """Run detector + tracker over [(frame_number, path), ...].

    Returns (mot_rows, detector_seconds, tracker_seconds). MOT row: (frame, id, x, y, w, h, score, cls)
    with cls = local class id. `on_frame(frame_index, det_s, trk_s)` is called after every frame.
    """
    rows, det_s, trk_s = [], 0.0, 0.0
    expected_shape = None
    for index, (fid, path) in enumerate(frames, 1):
        img = read_frame(path)
        if expected_shape is None:
            expected_shape = img.shape[:2]
        elif img.shape[:2] != expected_shape:
            raise ValueError(f"Frame size changed at {path}: expected {expected_shape}, got {img.shape[:2]}")

        t0 = time.perf_counter()
        res = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, classes=COCO_CLASSES,
                            device=device, verbose=False)[0]
        sync(device)
        det = res.boxes.cpu().numpy()
        t1 = time.perf_counter()

        if len(det):
            det = det[keep_mask(det.xywh, regions)]
        tracks = tracker.update(det, img)       # rows: x1, y1, x2, y2, track_id, score, cls, detection_index
        sync(device)
        t2 = time.perf_counter()

        det_s += t1 - t0
        trk_s += t2 - t1
        for x1, y1, x2, y2, tid, score, cls, _ in tracks:
            rows.append((fid, int(tid), x1, y1, x2 - x1, y2 - y1, score, COCO_TO_LOCAL[int(cls)]))
        if writer is not None:
            writer.write(draw_tracks(img, tracks, regions, fid))
        if on_frame:
            on_frame(index, det_s, trk_s)
    return rows, det_s, trk_s


def track_config(weights, tracker_name, seq_frames, ignored, out_dir, *, conf=CONF, imgsz, iou, device,
                 seed, on_frame=None, vis_dir=None):
    """Track every sequence with one detector + tracker configuration.

    Writes <out_dir>/<seq>.txt (MOT rows) and returns the timing dict (without provenance and environment).
    seq_frames: {seq: [(frame_number, path), ...]}; ignored: {seq: [(x1, y1, x2, y2), ...]}.
    on_frame(seq, frame_index, det_s, trk_s) is called after every frame.
    """
    settings = tracker_settings(tracker_name, conf)
    det_conf = detector_conf(settings)
    set_seed(seed)
    model = YOLO(weights_path(weights))

    first_seq = next(iter(seq_frames))
    warmup(model, read_frame(seq_frames[first_seq][0][1]), imgsz, device, det_conf)
    run_sequence(model, build_tracker(settings, device), seq_frames[first_seq][:10],
                 ignored[first_seq], det_conf, imgsz, iou, device)         # warms up the tracker too

    per_seq, total_frames, total_det, total_trk = {}, 0, 0.0, 0.0
    for seq, frames in seq_frames.items():
        set_seed(seed)
        tick = None if on_frame is None else (lambda i, d, t, seq=seq: on_frame(seq, i, d, t))
        if vis_dir is None:
            rows, det_s, trk_s = run_sequence(model, build_tracker(settings, device), frames, ignored[seq],
                                              det_conf, imgsz, iou, device, on_frame=tick)
        else:
            height, width = read_frame(frames[0][1]).shape[:2]
            with video_writer(vis_dir / f"{out_dir.name.removesuffix('.partial')}_{seq}.mp4", (width, height)) as writer:
                rows, det_s, trk_s = run_sequence(model, build_tracker(settings, device), frames, ignored[seq],
                                                  det_conf, imgsz, iou, device, on_frame=tick, writer=writer)
        write_mot(rows, out_dir / f"{seq}.txt")
        n = len(frames)
        per_seq[seq] = {"frames": n, "det_ms": 1000 * det_s / n, "trk_ms": 1000 * trk_s / n,
                        "tracks": len({r[1] for r in rows}), "rows": len(rows),
                        "mot_sha256": file_info(out_dir / f"{seq}.txt")["sha256"]}
        total_frames += n
        total_det += det_s
        total_trk += trk_s

    return {"detector": weights, "tracker": tracker_name, "conf": conf, "detector_conf": det_conf,
            "iou": iou, "imgsz": imgsz, "seed": seed, "frames": total_frames,
            "det_ms_per_frame": 1000 * total_det / total_frames,
            "trk_ms_per_frame": 1000 * total_trk / total_frames,
            "total_ms_per_frame": 1000 * (total_det + total_trk) / total_frames,
            "fps": total_frames / (total_det + total_trk),
            "per_seq": per_seq, "tracker_settings": settings,
            "note": "image reading excluded; detector = predict (pre+infer+NMS+GPU->CPU); "
                    "tracker = ignored-region mask + tracker.update"}
