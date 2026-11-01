"""The detect-and-track loop, with separate timing for the detector and the tracker.

`model.predict` and `tracker.update` are called one after the other (not through `model.track`).
This gives two things the built-in call cannot: the detector and tracker times are measured
apart, and detections whose centre is in an ignored region are removed before the tracker sees
them, which is the same rule the detector evaluation uses.

Detector confidence matters here. Both trackers use a second matching step for low-score
detections (between `track_low_thresh` and `track_high_thresh`). If the detector already
dropped everything below the high threshold, that step would have nothing to match and the
tracker would degrade. So the detector runs at the low threshold, and the tracker's own high
threshold (`conf`) decides which detections can start or extend tracks.
"""
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace

from ..config import COCO_CLASSES, COCO_TO_LOCAL, CONF, CONFIG_DIR, weights_path
from ..data.mot import MotOutRow, write_mot
from ..data.regions import Region, inside_any
from ..data.sequences import read_frame
from ..utils import file_info, set_seed
from .viz import draw_tracks, video_writer

TRACKER_CLASSES: dict[str, type] = {"bytetrack": BYTETracker, "botsort": BOTSORT}

# Called after each frame: (frame index within the sequence, detector seconds so far, tracker seconds so far)
FrameCallback = Callable[[int, float, float], None]


def tracker_settings(name: str, conf: float = CONF) -> dict[str, Any]:
    """Tracker settings from `configs/<name>.yaml`, with `conf` as the high-score threshold.

    `conf` overrides both `track_high_thresh` (detections above it can extend tracks) and
    `new_track_thresh` (detections above it can start tracks), so one number controls both.
    """
    settings = YAML.load(CONFIG_DIR / f"{name}.yaml")
    settings["track_high_thresh"] = settings["new_track_thresh"] = conf
    return settings


def detector_conf(settings: dict[str, Any]) -> float:
    """Confidence the detector must run at: the lowest score any tracker stage can use.

    ByteTrack and BoT-SORT match in two rounds.
    Round 1: tracks are matched to high-score detections. Only these can start new tracks.
    Round 2: the unmatched tracks are matched to low-score detections.
    A partly hidden or blurry car often gets a low score for a few frames. Round 2 keeps its track alive.
    If the detector drops every box below the high threshold, round 2 has nothing to match.
    So the detector runs at min(low, high), and the tracker applies the high threshold itself.
    """
    return min(settings["track_low_thresh"], settings["track_high_thresh"])


def build_tracker(settings: dict[str, Any], device: str | int | None = None) -> Any:
    """Create a fresh tracker with an empty track list.

    A tracker keeps state between frames, so each sequence needs its own instance.
    """
    cfg = IterableSimpleNamespace(**settings)       # Ultralytics expects attribute access, not a dict
    cfg.device = device
    return TRACKER_CLASSES[cfg.tracker_type](args=cfg)


def keep_mask(xywh: NDArray[np.floating], regions: list[Region]) -> NDArray[np.bool_]:
    """Boolean mask of detections to keep: True where the box centre is outside all ignored regions.

    `xywh` is the (N, 4) array Ultralytics provides: centre x, centre y, width, height. The first
    two columns are therefore already the centre the rule needs.
    """
    return ~inside_any(xywh[:, :2], regions)


def sync(device: str | int | None) -> None:
    """Block until the GPU has finished all queued work; does nothing on CPU.

    CUDA calls return immediately and run asynchronously. Without this, `time.perf_counter()`
    would measure how long it took to queue the work, not to do it.
    """
    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.synchronize(device=int(device) if str(device).isdigit() else device)


def warmup(model: YOLO, frame: NDArray[np.uint8], imgsz: int, device: str | int | None, conf: float,
           n: int = 10) -> None:
    """Run the detector `n` times on one frame before timing starts.

    The first calls are slow (CUDA context, kernel selection, memory allocation). Doing them here
    keeps that one-time cost out of the measured frames.
    """
    for _ in range(n):
        model.predict(frame, imgsz=imgsz, conf=conf, classes=COCO_CLASSES, device=device, verbose=False)
    sync(device)


def run_sequence(model: YOLO, tracker: Any, frames: list[tuple[int, Path]], regions: list[Region], conf: float,
                 imgsz: int, iou: float, device: str | int | None, on_frame: FrameCallback | None = None,
                 writer: Any = None) -> tuple[list[MotOutRow], float, float]:
    """Detect and track one sequence, frame by frame.

    `frames` is [(frame number, image path), ...] in order, `conf` the detector confidence, `iou`
    the NMS threshold. Reading images and drawing are outside the timers. If `writer` is given, each
    annotated frame is written to it. Returns (rows, detector seconds, tracker seconds); a row is
    (frame, id, x, y, w, h, score, project class id).
    """
    rows: list[MotOutRow] = []
    det_s = trk_s = 0.0
    expected_shape: tuple[int, int] | None = None

    for index, (fid, path) in enumerate(frames, 1):
        # Read. All frames of a sequence must share one size; a change means a broken sequence.
        img = read_frame(path)
        if expected_shape is None:
            expected_shape = img.shape[:2]
        elif img.shape[:2] != expected_shape:
            raise ValueError(f"Frame size changed at {path}: expected {expected_shape}, got {img.shape[:2]}")

        # Detect. Timed: preprocessing, inference, NMS and the copy of the result to the CPU.
        t0 = time.perf_counter()
        res = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, classes=COCO_CLASSES,
                            device=device, verbose=False)[0]
        sync(device)
        det = res.boxes.cpu().numpy()
        t1 = time.perf_counter()

        # Track. Timed: the ignored-region filter and the tracker update, which associates the
        # detections with existing tracks and returns the confirmed tracks of this frame.
        if len(det):
            det = det[keep_mask(det.xywh, regions)]
        tracks = tracker.update(det, img)       # rows: x1, y1, x2, y2, track id, score, class, detection index
        sync(device)
        t2 = time.perf_counter()

        det_s += t1 - t0
        trk_s += t2 - t1

        # Collect. Boxes go from x1 y1 x2 y2 to x y w h, COCO class ids to project class ids.
        for x1, y1, x2, y2, tid, score, cls, _ in tracks:
            rows.append((fid, int(tid), x1, y1, x2 - x1, y2 - y1, score, COCO_TO_LOCAL[int(cls)]))
        if writer is not None:
            writer.write(draw_tracks(img, tracks, regions, fid))
        if on_frame:
            on_frame(index, det_s, trk_s)
    return rows, det_s, trk_s


def track_config(weights: str, tracker_name: str, seq_frames: dict[str, list[tuple[int, Path]]],
                 ignored: dict[str, list[Region]], out_dir: Path, *, imgsz: int, iou: float,
                 device: str | int | None, seed: int, conf: float = CONF,
                 on_frame: Callable[[str, int, float, float], None] | None = None,
                 vis_dir: Path | None = None) -> dict[str, Any]:
    """Run one detector + tracker configuration over several sequences.

    Writes `<out_dir>/<sequence>.txt` per sequence, and an mp4 per sequence into `vis_dir` if given.
    `on_frame(sequence, frame index, detector s, tracker s)` is called after every frame.
    Returns the timing report (ms per frame, FPS, tracker settings, MOT file hashes); the caller
    adds provenance and environment.
    """
    settings = tracker_settings(tracker_name, conf)
    det_conf = detector_conf(settings)
    set_seed(seed)
    model = YOLO(weights_path(weights))

    # Warm-up: 10 detector calls on one frame, then a full detect+track pass over the first 10
    # frames of the first sequence. Nothing from this pass is kept; every sequence below gets a
    # fresh tracker.
    first_seq = next(iter(seq_frames))
    warmup(model, read_frame(seq_frames[first_seq][0][1]), imgsz, device, det_conf)
    run_sequence(model, build_tracker(settings, device), seq_frames[first_seq][:10],
                 ignored[first_seq], det_conf, imgsz, iou, device)

    # Timed run. A fresh tracker and a reset seed per sequence keep sequences independent.
    per_seq: dict[str, dict[str, Any]] = {}
    total_frames, total_det, total_trk = 0, 0.0, 0.0
    for seq, frames in seq_frames.items():
        set_seed(seed)
        tick = None if on_frame is None else (lambda i, d, t, seq=seq: on_frame(seq, i, d, t))
        tracker = build_tracker(settings, device)

        if vis_dir is None:
            rows, det_s, trk_s = run_sequence(model, tracker, frames, ignored[seq], det_conf, imgsz, iou, device,
                                              on_frame=tick)
        else:
            # The video name uses the final folder name, even while the run is in its .partial folder
            height, width = read_frame(frames[0][1]).shape[:2]
            video = vis_dir / f"{out_dir.name.removesuffix('.partial')}_{seq}.mp4"
            with video_writer(video, (width, height)) as writer:
                rows, det_s, trk_s = run_sequence(model, tracker, frames, ignored[seq], det_conf, imgsz, iou,
                                                  device, on_frame=tick, writer=writer)

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
