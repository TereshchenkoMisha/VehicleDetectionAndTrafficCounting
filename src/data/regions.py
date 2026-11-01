"""The ignored-region rule, in one place.

UA-DETRAC marks areas of the frame where annotation is unreliable. Every stage (detector
evaluation, tracking, GT export, counting) applies the same rule: an object counts as inside
an ignored region when the centre of its box is inside the region, borders included. Using one
implementation keeps ground truth and predictions filtered identically.
"""
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

Region = tuple[float, float, float, float]      # x1, y1, x2, y2 in pixels


def inside_any(points: ArrayLike, regions: Sequence[Region]) -> NDArray[np.bool_]:
    """Tell which points lie inside at least one region.

    `points` is anything convertible to an (N, 2) array of (x, y). Returns a bool array of length N.
    A point on a region's border counts as inside. With no regions, nothing is inside.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    hit = np.zeros(len(points), dtype=bool)

    # One vectorised box test per region, OR-ed together. There are only a few regions per sequence,
    # so looping over regions and vectorising over points is the cheap way round.
    for x1, y1, x2, y2 in regions:
        hit |= (points[:, 0] >= x1) & (points[:, 0] <= x2) & (points[:, 1] >= y1) & (points[:, 1] <= y2)
    return hit


def box_centres(boxes_xyxy: ArrayLike) -> NDArray[np.float64]:
    """Centres of boxes given as x1, y1, x2, y2. Input (N, 4) or flat, output (N, 2) of (x, y)."""
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    return (boxes[:, :2] + boxes[:, 2:]) / 2
