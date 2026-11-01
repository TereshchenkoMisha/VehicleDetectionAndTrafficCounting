"""Counting error metrics.

The inputs are per-sequence totals (one number per sequence), so every metric here compares how many
vehicles were counted with how many there were, sequence by sequence.
"""
from collections.abc import Sequence
from typing import Any

import numpy as np


def count_errors(pred: Sequence[float], gt: Sequence[float]) -> dict[str, Any]:
    """Compare predicted and GT counts; both have one entry per sequence, in the same order.

    Returns:
      mae       mean absolute error: by how many vehicles a sequence count is off on average;
      wape      weighted absolute percentage error, sum |error| / sum GT: the same error relative to
                traffic volume, so 5 off on 20 vehicles is worse than 5 off on 150 (None if GT sums to 0);
      sce       signed error per sequence, pred - gt: negative means undercounting;
      mean_sce  mean of sce, the bias of the counter;
      over / under  share of sequences counted too high / too low.
    Errors of opposite sign cancel in mean_sce but not in mae, so read them together.
    """
    # Inputs as float arrays; a wrong shape is a caller error and fails loudly
    pred_arr, gt_arr = np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)
    if pred_arr.ndim != 1 or pred_arr.shape != gt_arr.shape or pred_arr.size == 0:
        raise ValueError("pred and gt must be non-empty 1-D arrays of equal length: "
                         f"{pred_arr.shape}, {gt_arr.shape}")

    error = pred_arr - gt_arr
    total_gt = gt_arr.sum()
    return {"mae": float(np.abs(error).mean()),
            "wape": float(np.abs(error).sum() / total_gt) if total_gt else None,
            "mean_sce": float(error.mean()),
            "over": float((error > 0).mean()),
            "under": float((error < 0).mean()),
            "sce": error.tolist()}
