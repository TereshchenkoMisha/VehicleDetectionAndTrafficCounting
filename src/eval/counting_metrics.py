import numpy as np


def _count_arrays(pred, gt):
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)

    if pred.ndim != 1 or gt.ndim != 1:
        raise ValueError("pred and gt must be one-dimensional arrays")

    if pred.shape != gt.shape:
        raise ValueError(
            f"Different shapes: pred={pred.shape}, gt={gt.shape}"
        )

    if pred.size == 0:
        raise ValueError("pred and gt must not be empty")

    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("pred and gt must contain only finite values")

    if (pred < 0).any() or (gt < 0).any():
        raise ValueError("Vehicle counts must not be negative")

    return pred, gt


def mae(pred, gt):
    pred, gt = _count_arrays(pred, gt)
    return float(np.abs(pred - gt).mean())


def wape(pred, gt):
    pred, gt = _count_arrays(pred, gt)
    total_gt = float(gt.sum())

    if total_gt == 0:
        return None

    return float(np.abs(pred - gt).sum() / total_gt)


def sce(pred, gt):
    pred, gt = _count_arrays(pred, gt)
    return (pred - gt).tolist()


def mean_sce(pred, gt):
    pred, gt = _count_arrays(pred, gt)
    return float((pred - gt).mean())


def sce_over_rate(pred, gt):
    pred, gt = _count_arrays(pred, gt)
    return float((pred > gt).mean())


def sce_under_rate(pred, gt):
    pred, gt = _count_arrays(pred, gt)
    return float((pred < gt).mean())