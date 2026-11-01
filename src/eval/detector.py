"""Detector evaluation on the prepared YOLO store: mAP, and precision/recall at a fixed score.

The detector runs on every prepared image of a split. Its boxes are compared with the GT labels in
two ways: torchmetrics computes mAP over all scores, and `frame_pr` counts true/false positives
and misses at one score (CONF) for precision and recall. In both, GT boxes and predictions with the
centre in an ignored region are dropped, the same rule tracking and counting use.
"""
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm import tqdm
from ultralytics import YOLO

from ..config import CLASSES, COCO_CLASSES, COCO_TO_LOCAL, CONF, IMGSZ, MAP_CONF, NMS_IOU, SEED, weights_path
from ..data.regions import Region, box_centres, inside_any
from ..data.sequences import JPG_SUFFIXES
from ..utils import env_info, file_info, read_json, set_deterministic, set_seed

PR_IOU: float = 0.5         # IoU a prediction needs with a GT box to count as a match in precision/recall

Counts = dict[int, dict[str, int]]      # class id -> {"tp", "fp", "fn"}
GroundTruth = dict[str, torch.Tensor]   # {"boxes": (N, 4) xyxy in pixels, "labels": (N,) class ids}


def _load_gt(label_path: Path, width: int, height: int) -> GroundTruth:
    """Read a YOLO label file as GT boxes in pixels.

    YOLO stores `class cx cy w h` normalised to the image size; this converts them to pixel
    x1, y1, x2, y2. A missing file is an error, not "no objects": an image without objects must have an
    empty label file, otherwise a lost file would silently count as a perfect frame.
    """
    if not label_path.is_file():
        raise FileNotFoundError(f"GT label file is missing: {label_path}. "
                                "An image without objects must have an existing empty label file.")

    boxes: list[list[float]] = []
    labels: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c, cx, cy, w, h = (float(v) for v in line.split())

        if not np.isfinite([c, cx, cy, w, h]).all():
            raise ValueError(f"Nonfinite label values: {label_path}")
        if not c.is_integer() or not 0 <= c < len(CLASSES):
            raise ValueError(f"Invalid label class: {label_path}")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid normalized box: {label_path}")

        # Normalised centre + size -> pixel corners
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
        labels.append(int(c))

    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64)}


def _drop_ignored_gt(gt: GroundTruth, regions: list[Region]) -> GroundTruth:
    """Remove GT boxes whose centre is in an ignored region."""
    if gt["boxes"].numel() == 0 or not regions:
        return gt
    keep = torch.as_tensor(~inside_any(box_centres(gt["boxes"].numpy()), regions))
    return {"boxes": gt["boxes"][keep], "labels": gt["labels"][keep]}


def frame_pr(pb: NDArray, pl: NDArray, ps: NDArray, gb: NDArray, gl: NDArray, conf_thr: float = CONF,
             iou_thr: float = PR_IOU) -> Counts:
    """True positives, false positives and misses per class for one frame.

    `pb`, `pl`, `ps`: predicted boxes (xyxy), classes, scores; `gb`, `gl`: GT boxes and classes. Only
    predictions with score >= `conf_thr` count. Greedy matching by falling score: each prediction
    claims the free same-class GT box with the highest IoU if that is >= `iou_thr`, otherwise it is a
    false positive. GT boxes nobody claimed are misses.
    """
    out: Counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(len(CLASSES))}
    for c in gl.astype(int):
        out[c]["fn"] += 1       # every GT box starts as a miss; a match below turns it into a true positive

    keep = ps >= conf_thr
    pb, pl, ps = pb[keep], pl[keep], ps[keep]

    # IoU of every prediction with every GT box: (n_pred, n_gt)
    if len(pb) and len(gb):
        iou = box_iou(torch.as_tensor(pb, dtype=torch.float32), torch.as_tensor(gb, dtype=torch.float32)).numpy()
    else:
        iou = np.zeros((len(pb), len(gb)))

    # Greedy matching. For each prediction, candidates are the free GT boxes of the same class; the
    # others get IoU 0 so they can never win. The best candidate matches if it clears the threshold.
    taken = np.zeros(len(gb), dtype=bool)
    for i in np.argsort(-ps):
        c = int(pl[i])
        candidates = np.where(~taken & (gl.astype(int) == c), iou[i], 0.0)
        j = int(candidates.argmax()) if candidates.size else -1
        if j >= 0 and candidates[j] >= iou_thr:
            taken[j] = True
            out[c]["fn"] -= 1
            out[c]["tp"] += 1
        else:
            out[c]["fp"] += 1
    return out


def _round(value: float) -> float | None:
    """Value rounded to 4 decimals for reports; None for NaN or infinity (JSON cannot hold them)."""
    value = float(value)
    return round(value, 4) if np.isfinite(value) else None


def _ap(value: float) -> float | None:
    """Rounded AP from torchmetrics, which returns -1 for 'no GT of this class'; that becomes None."""
    return None if float(value) < 0 else _round(value)


def _per_class_report(counts: dict[str, dict[str, int]], map_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-class table: TP, FP, FN, precision, recall and F1 at the CONF operating point, and AP50:95.

    Precision, recall and F1 are None when their denominator is zero (the class never occurs or is
    never predicted), not 0, so "undefined" is not confused with "bad".
    """
    report: dict[str, dict[str, Any]] = {}
    for name in CLASSES:
        tp, fp, fn = counts[name]["tp"], counts[name]["fp"], counts[name]["fn"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        report[name] = {"TP": int(tp), "FP": int(fp), "FN": int(fn),
                        "P@0.25": None if precision is None else _round(precision),
                        "R@0.25": None if recall is None else _round(recall),
                        "F1@0.25": None if f1 is None else _round(f1),
                        "AP50_95": None}

    # torchmetrics returns AP only for classes that occur, together with their class ids
    class_ids = map_result.get("classes")
    ap_values = map_result.get("map_per_class")
    if class_ids is not None and ap_values is not None:
        for c, v in zip(torch.as_tensor(class_ids).reshape(-1).tolist(),
                        torch.as_tensor(ap_values).reshape(-1).tolist(), strict=True):
            if 0 <= int(c) < len(CLASSES):
                report[CLASSES[int(c)]]["AP50_95"] = None if v < 0 else _round(v)
    return report


def evaluate_detector(weights: str, data_root: str | Path, split: str = "test", imgsz: int = IMGSZ,
                      device: str | int | None = None, seed: int = SEED) -> dict[str, Any]:
    """Evaluate one detector on `<data_root>/images/<split>` and return the full report.

    The report holds mAP@0.5, mAP@0.5:0.95 and mAP@0.75, the per-class table, the protocol
    parameters, the input file hashes and the environment. Runs in deterministic mode, so the numbers
    are repeatable on one machine. `ms_per_frame` includes image loading and metric updates; it
    is not a speed measurement (speed comes from `run_tracking`).
    """
    set_seed(seed)
    set_deterministic()

    # Inputs: images of the split, and ignored regions for every sequence that appears in it
    data_root = Path(data_root)
    src = data_root / "images" / split
    images = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in JPG_SUFFIXES)
    if not images:
        raise ValueError(f"No prepared images in {src}")
    ignored = read_json(data_root / "ignored_regions.json")
    sequences = {image.stem.split("_f")[0] for image in images}      # file names are <sequence>_f<frame>
    if sequences - set(ignored):
        raise ValueError(f"Missing ignored region entries: {sorted(sequences - set(ignored))}")

    resolved_weights = weights_path(weights)
    model = YOLO(resolved_weights)
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric.warn_on_many_detections = False
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASSES}
    expected_stems = {image.stem for image in images}
    seen: set[str] = set()
    start = time.perf_counter()

    # The detector runs at MAP_CONF so almost no box is filtered: mAP integrates over the whole
    # score range. Precision/recall apply their own threshold inside frame_pr.
    stream = model.predict(source=str(src), stream=True, imgsz=imgsz, conf=MAP_CONF, iou=NMS_IOU, device=device,
                           classes=COCO_CLASSES, verbose=False)
    for res in tqdm(stream, total=len(images), unit="frame", desc=f"eval {Path(str(weights)).stem} [{split}]"):
        # Each image must be predicted exactly once
        stem = Path(res.path).stem
        if stem in seen or stem not in expected_stems:
            raise ValueError(f"Unexpected or duplicate prediction frame: {res.path}")
        seen.add(stem)
        height, width = res.orig_shape
        regions = ignored.get(stem.split("_f")[0], [])

        gt = _drop_ignored_gt(_load_gt(data_root / "labels" / split / f"{stem}.txt", width, height), regions)

        # Predictions of this frame: COCO class ids become project ids; boxes in ignored regions go
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        labels = np.array([COCO_TO_LOCAL[int(c)] for c in res.boxes.cls.cpu().numpy()], dtype=int)
        if regions and len(boxes):
            keep = ~inside_any(box_centres(boxes), regions)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # Feed both metrics: torchmetrics accumulates for mAP, frame_pr adds to the TP/FP/FN totals
        metric.update([{"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                        "scores": torch.as_tensor(scores, dtype=torch.float32),
                        "labels": torch.as_tensor(labels, dtype=torch.int64)}], [gt])
        frame = frame_pr(boxes, labels, scores, gt["boxes"].numpy(), gt["labels"].numpy())
        for c, d in frame.items():
            for k in ("tp", "fp", "fn"):
                counts[CLASSES[c]][k] += d[k]

    if len(seen) != len(images):
        raise ValueError(f"Incomplete detector evaluation: {len(seen)}/{len(images)} frames")
    ms_per_frame = (time.perf_counter() - start) / len(seen) * 1000
    result = metric.compute()

    return {"sources": {"weights": file_info(resolved_weights), "data_root": str(data_root.resolve()),
                        "ignored": file_info(data_root / "ignored_regions.json")},
            "protocol": {"predict_conf": MAP_CONF, "nms_iou": NMS_IOU, "pr_conf": CONF, "pr_iou": PR_IOU,
                         "ignored_rule": "observation_centres"},
            "model": weights, "split": split, "imgsz": imgsz, "seed": seed, "deterministic": True,
            "env": env_info(model.device), "frames": len(seen), "ms_per_frame": round(ms_per_frame, 1),
            "timing_scope": "prediction, image loading and metric updates",
            "mAP@0.5": _ap(result["map_50"]), "mAP@0.5:0.95": _ap(result["map"]),
            "mAP@0.75": _ap(result["map_75"]), "per_class": _per_class_report(counts, result)}
