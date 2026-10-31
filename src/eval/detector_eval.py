import json
import time
from pathlib import Path

import numpy as np


from ..data.config import CLASSES, SEED, weights_path
from ..reproducibility import set_seed
from ..data.provenance import file_info

COCO_ID_TO_LOCAL = {2: 0, 5: 1, 7: 2}


def _load_gt(label_path: Path, W, H):
    import torch

    if not label_path.is_file():
        raise FileNotFoundError(
            f"GT label file is missing: {label_path}. "
            "An image without objects must have an existing empty label file."
        )

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

        boxes.append([
            (cx - w / 2) * W,
            (cy - h / 2) * H,
            (cx + w / 2) * W,
            (cy + h / 2) * H,
        ])
        labels.append(int(c))

    return {
        "boxes": torch.tensor(
            boxes, dtype=torch.float32
        ).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def _in_ignored_region(cx, cy, regions):
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in regions)

def _drop_ignored_gt(gt, regions):
    """Exclude GT boxes with centres inside ignored regions"""
    import torch

    boxes = gt["boxes"]

    if boxes.numel() == 0 or not regions:
        return gt

    centres = ((boxes[:, :2] + boxes[:, 2:]) / 2).cpu().numpy()

    keep = np.array([
        not _in_ignored_region(float(cx), float(cy), regions)
        for cx, cy in centres
    ], dtype=bool)

    mask = torch.as_tensor(keep, dtype=torch.bool)

    return {
        "boxes": boxes[mask],
        "labels": gt["labels"][mask],
    }


def _iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _frame_pr(pb, pl, ps, gb, gl, conf_thr=0.25, iou_thr=0.5):
    out = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(len(CLASSES))}
    for c in gl.astype(int):
        out[c]["fn"] += 1
    keep = ps >= conf_thr
    pb, pl, ps = pb[keep], pl[keep], ps[keep]
    matched = set()
    for i in np.argsort(-ps):
        best_iou, best_j = 0.0, -1
        for j, b in enumerate(gb):
            if j in matched or int(gl[j]) != int(pl[i]):
                continue
            iou = _iou_xyxy(pb[i], b)
            if iou > best_iou:
                best_iou, best_j = iou, j
        c = int(pl[i])
        if best_j >= 0 and best_iou >= iou_thr:
            matched.add(best_j)
            out[c]["fn"] -= 1
            out[c]["tp"] += 1
        else:
            out[c]["fp"] += 1
    return out


def _f(v):
    v = float(v)
    return round(v, 4) if np.isfinite(v) else None


def _ap(value):
    return None if float(value) < 0 else _f(value)


def evaluate_detector(weights, data_root, split="test", imgsz=960, device=None, seed=SEED) -> dict:
    import torch
    from tqdm import tqdm
    from torchmetrics.detection import MeanAveragePrecision
    import cv2
    import platform
    import ultralytics
    import torchmetrics
    from ultralytics import YOLO

    set_seed(seed)
    data_root = Path(data_root)
    src = data_root / "images" / split
    images = sorted(path for path in src.iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"})
    if len({image.stem for image in images}) != len(images):
        raise ValueError(f"Duplicate image stems in {src}")
    if not images:
        raise ValueError(f"No prepared images in {src}")
    ignored = json.loads((data_root / "ignored_regions.json").read_text(encoding="utf-8"))
    sequences = {image.stem.split("_f")[0] for image in images}
    if sequences - set(ignored):
        raise ValueError(f"Missing ignored region entries: {sorted(sequences - set(ignored))}")
    resolved_weights = weights_path(weights)
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric.warn_on_many_detections = False
    model = YOLO(resolved_weights)
    agg = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASSES}
    n_frames, t0 = 0, time.perf_counter()

    n_total = len(images)
    seen = set()
    expected_stems = {image.stem for image in images}
    stream = model.predict(source=str(src), stream=True, imgsz=imgsz, conf=0.001, iou=0.7, device=device,
                           classes=sorted(COCO_ID_TO_LOCAL), verbose=False)
    for res in tqdm(stream, total=n_total, unit="frame", desc=f"eval {Path(str(weights)).stem} [{split}]"):
        n_frames += 1
        stem = Path(res.path).stem
        if stem in seen or stem not in expected_stems:
            raise ValueError(f"Unexpected or duplicate prediction frame: {res.path}")
        seen.add(stem)
        H, W = res.orig_shape

        regions = ignored.get(stem.split("_f")[0], [])

        gt = _load_gt(data_root / "labels" / split / f"{stem}.txt", W, H)
        gt = _drop_ignored_gt(gt, regions)


        boxes = res.boxes.xyxy.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()
        if regions and len(boxes):
            keep = np.array([not _in_ignored_region((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, regions)
                             for b in boxes])
            boxes, clss, scores = boxes[keep], clss[keep], scores[keep]
        local = np.array([COCO_ID_TO_LOCAL[c] for c in clss], dtype=int)

        preds = [{"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "scores": torch.as_tensor(scores, dtype=torch.float32),
                  "labels": torch.as_tensor(local, dtype=torch.int64)}]
        metric.update(preds, [gt])
        frame = _frame_pr(boxes, local, scores, gt["boxes"].numpy(), gt["labels"].numpy())
        for c, d in frame.items():
            for k in ("tp", "fp", "fn"):
                agg[CLASSES[c]][k] += d[k]

    if n_frames != n_total:
        raise ValueError(f"Incomplete detector evaluation: {n_frames}/{n_total} frames")

    dt = (time.perf_counter() - t0) / max(n_frames, 1) * 1000
    rm = metric.compute()

    per_class = {}

    for c, name in enumerate(CLASSES):
        tp = agg[name]["tp"]
        fp = agg[name]["fp"]
        fn = agg[name]["fn"]

        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None

        # Calculate F1 with the precision and recall thresholds
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else None

        per_class[name] = {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "P@0.25": _f(p) if p is not None else None,
            "R@0.25": _f(r) if r is not None else None,
            "F1@0.25": _f(f1) if f1 is not None else None,
            "AP50_95": None,
        }

    # Match AP values using the class IDs returned by TorchMetrics
    class_ids = rm.get("classes")
    map_per_class = rm.get("map_per_class")

    if class_ids is not None and map_per_class is not None:
        class_ids = torch.as_tensor(class_ids).reshape(-1).tolist()
        map_per_class = torch.as_tensor(map_per_class).reshape(-1).tolist()

        for c, v in zip(class_ids, map_per_class):
            c = int(c)

            if 0 <= c < len(CLASSES):
                per_class[CLASSES[c]]["AP50_95"] = (
                    None if v < 0 else _f(v)
                )

    return {"sources": {"weights": file_info(resolved_weights), "data_root": str(data_root.resolve()),
                        "ignored": file_info(data_root / "ignored_regions.json")},
            "protocol": {"predict_conf": 0.001, "nms_iou": 0.7, "pr_conf": 0.25, "pr_iou": 0.5,
                         "ignored_rule": "observation_centres"},
            "model": weights, "split": split, "imgsz": imgsz,
            "seed": seed, "deterministic": True,
            "env": {"torch": torch.__version__, "numpy": np.__version__,
                    "ultralytics": ultralytics.__version__,
                    "torchmetrics": torchmetrics.__version__,
                    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "opencv": cv2.__version__, "python": platform.python_version(),
                    "device": str(model.device)},
            "frames": n_frames, "ms_per_frame": round(dt, 1),
            "timing_scope": "prediction, image loading and metric updates",
            "mAP@0.5": _ap(rm["map_50"]), "mAP@0.5:0.95": _ap(rm["map"]),
            "mAP@0.75": _ap(rm["map_75"]), "per_class": per_class}
