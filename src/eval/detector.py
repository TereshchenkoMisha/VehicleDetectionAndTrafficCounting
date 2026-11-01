"""Detector evaluation on the prepared YOLO store: mAP (torchmetrics) and precision/recall at a fixed score.

GT boxes and predictions whose centres lie in ignored regions are dropped, as in tracking and counting.
"""
import time
from pathlib import Path

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm import tqdm
from ultralytics import YOLO

from ..config import CLASSES, COCO_CLASSES, COCO_TO_LOCAL, CONF, IMGSZ, MAP_CONF, NMS_IOU, SEED, weights_path
from ..data.regions import box_centres, inside_any
from ..data.sequences import JPG_SUFFIXES
from ..utils import env_info, file_info, read_json, set_deterministic, set_seed

PR_IOU = 0.5


def _load_gt(label_path: Path, width, height):
    """YOLO label file -> {"boxes": (N, 4) xyxy pixels, "labels": (N,)}. The file must exist (empty = no objects)."""
    if not label_path.is_file():
        raise FileNotFoundError(f"GT label file is missing: {label_path}. "
                                "An image without objects must have an existing empty label file.")
    boxes, labels = [], []
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
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
        labels.append(int(c))
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64)}


def _drop_ignored_gt(gt, regions):
    if gt["boxes"].numel() == 0 or not regions:
        return gt
    keep = torch.as_tensor(~inside_any(box_centres(gt["boxes"].numpy()), regions))
    return {"boxes": gt["boxes"][keep], "labels": gt["labels"][keep]}


def frame_pr(pb, pl, ps, gb, gl, conf_thr=CONF, iou_thr=PR_IOU):
    """Per-class TP / FP / FN of one frame at score >= conf_thr.

    Predictions are taken by descending score; each is matched to the unmatched same-class GT box with the
    highest IoU, if that IoU >= iou_thr.
    """
    out = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(len(CLASSES))}
    for c in gl.astype(int):
        out[c]["fn"] += 1
    keep = ps >= conf_thr
    pb, pl, ps = pb[keep], pl[keep], ps[keep]
    if len(pb) and len(gb):
        iou = box_iou(torch.as_tensor(pb, dtype=torch.float32), torch.as_tensor(gb, dtype=torch.float32)).numpy()
    else:
        iou = np.zeros((len(pb), len(gb)))
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


def _round(value):
    value = float(value)
    return round(value, 4) if np.isfinite(value) else None


def _ap(value):
    """torchmetrics returns -1 for 'no ground truth'."""
    return None if float(value) < 0 else _round(value)


def _per_class_report(counts, map_result):
    """TP/FP/FN, precision, recall, F1 at the P/R operating point, plus AP50:95 from torchmetrics."""
    report = {}
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
    class_ids = map_result.get("classes")
    ap_values = map_result.get("map_per_class")
    if class_ids is not None and ap_values is not None:
        for c, v in zip(torch.as_tensor(class_ids).reshape(-1).tolist(),
                        torch.as_tensor(ap_values).reshape(-1).tolist()):
            if 0 <= int(c) < len(CLASSES):
                report[CLASSES[int(c)]]["AP50_95"] = None if v < 0 else _round(v)
    return report


def evaluate_detector(weights, data_root, split="test", imgsz=IMGSZ, device=None, seed=SEED) -> dict:
    set_seed(seed)
    set_deterministic()
    data_root = Path(data_root)
    src = data_root / "images" / split
    images = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in JPG_SUFFIXES)
    if not images:
        raise ValueError(f"No prepared images in {src}")
    ignored = read_json(data_root / "ignored_regions.json")
    sequences = {image.stem.split("_f")[0] for image in images}
    if sequences - set(ignored):
        raise ValueError(f"Missing ignored region entries: {sorted(sequences - set(ignored))}")

    resolved_weights = weights_path(weights)
    model = YOLO(resolved_weights)
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric.warn_on_many_detections = False
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASSES}
    expected_stems = {image.stem for image in images}
    seen = set()
    start = time.perf_counter()

    stream = model.predict(source=str(src), stream=True, imgsz=imgsz, conf=MAP_CONF, iou=NMS_IOU, device=device,
                           classes=COCO_CLASSES, verbose=False)
    for res in tqdm(stream, total=len(images), unit="frame", desc=f"eval {Path(str(weights)).stem} [{split}]"):
        stem = Path(res.path).stem
        if stem in seen or stem not in expected_stems:
            raise ValueError(f"Unexpected or duplicate prediction frame: {res.path}")
        seen.add(stem)
        height, width = res.orig_shape
        regions = ignored.get(stem.split("_f")[0], [])

        gt = _drop_ignored_gt(_load_gt(data_root / "labels" / split / f"{stem}.txt", width, height), regions)

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        labels = np.array([COCO_TO_LOCAL[int(c)] for c in res.boxes.cls.cpu().numpy()], dtype=int)
        if regions and len(boxes):
            keep = ~inside_any(box_centres(boxes), regions)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

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
