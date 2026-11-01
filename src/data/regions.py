"""The one ignored-region rule used everywhere: a box is ignored when its centre is inside a region."""
import numpy as np


def inside_any(points, regions):
    """points: (N, 2) x, y. regions: list of (x1, y1, x2, y2). Returns a bool array (N,). Borders are inside."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    hit = np.zeros(len(points), dtype=bool)
    for x1, y1, x2, y2 in regions:
        hit |= (points[:, 0] >= x1) & (points[:, 0] <= x2) & (points[:, 1] >= y1) & (points[:, 1] <= y2)
    return hit


def box_centres(boxes_xyxy):
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    return (boxes[:, :2] + boxes[:, 2:]) / 2
