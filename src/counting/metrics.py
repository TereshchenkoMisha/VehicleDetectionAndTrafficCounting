"""Counting error metrics on per-sequence totals."""
import numpy as np


def count_errors(pred, gt):
    """pred, gt: counts per sequence (same length, non-empty).

    mae: mean absolute error. wape: sum |error| / sum gt (None if gt is all zero).
    sce: signed count error per sequence (pred - gt); mean_sce is the bias; over / under are the shares of
    sequences with too many / too few counts.
    """
    pred, gt = np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)
    if pred.ndim != 1 or pred.shape != gt.shape or pred.size == 0:
        raise ValueError(f"pred and gt must be non-empty 1-D arrays of equal length: {pred.shape}, {gt.shape}")
    error = pred - gt
    total_gt = gt.sum()
    return {"mae": float(np.abs(error).mean()),
            "wape": float(np.abs(error).sum() / total_gt) if total_gt else None,
            "mean_sce": float(error.mean()),
            "over": float((error > 0).mean()),
            "under": float((error < 0).mean()),
            "sce": error.tolist()}
