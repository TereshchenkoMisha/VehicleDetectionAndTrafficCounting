"""Line-crossing counting core shared by the ground-truth and tracker counters (README todo:
counting line + line counter).

A target is counted when the segment between its box centre at frame t and at the next
annotated/tracked frame strictly crosses the counting line segment. Each target is counted
at most once - at its first crossing. Direction: +1 = the centre moves from the P1 side to
the P2 side of the oriented line (A->B), -1 = the opposite (B->A).

Coordinate system: absolute pixels, top-left origin (same as everywhere in the project).
"""
from ..data.config import CLASSES


def _orient(a, b, c):
    """Cross product (b - a) x (c - a): >0 c left of a->b, <0 right, =0 collinear."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segment_crossing(p1, p2, a, b, eps=1e-6):
    """Intersection point of the movement p1->p2 with the line segment a-b, or None.

    A crossing requires a strict side change (a centre that starts or ends exactly on the
    line is not counted - it would make appearance/disappearance at the line ambiguous) and
    the intersection point must lie within the line segment, endpoints inclusive.
    """
    d1, d2 = _orient(a, b, p1), _orient(a, b, p2)
    if d1 == 0 or d2 == 0 or (d1 > 0) == (d2 > 0):
        return None
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    lx, ly = b[0] - a[0], b[1] - a[1]
    denom = dx * ly - dy * lx
    if abs(denom) < eps:
        return None
    t = ((a[0] - p1[0]) * ly - (a[1] - p1[1]) * lx) / denom
    if not 0.0 <= t <= 1.0:
        return None
    x, y = p1[0] + t * dx, p1[1] + t * dy
    if min(a[0], b[0]) - eps <= x <= max(a[0], b[0]) + eps and \
            min(a[1], b[1]) - eps <= y <= max(a[1], b[1]) + eps:
        return (x, y)
    return None


def crossing_direction(p_from, p_to, a, b):
    """+1 (labelled 'ab') if the centre moved from the _orient < 0 half-plane to the
    _orient > 0 half-plane of the line oriented P1->P2, -1 ('ba') for the opposite.
    p_to is unused: segment_crossing already guarantees a strict side change."""
    return 1 if _orient(a, b, p_from) < 0 else -1


def first_crossing(points, a, b):
    """First crossing of a target trajectory.

    points: [(frame_id, x, y), ...] sorted by frame_id (gaps between frames are fine - the
    straight segment between the last seen and next seen centre is tested).
    Returns (frame_id, direction, (x, y)) of the crossing or None.
    """
    for (f1, x1, y1), (f2, x2, y2) in zip(points, points[1:]):
        pt = segment_crossing((x1, y1), (x2, y2), a, b)
        if pt is not None:
            return f2, crossing_direction((x1, y1), (x2, y2), a, b), pt
    return None


def count_events(trajectories, p1, p2):
    """Count line crossings for an iterable of (target_key, cls_id, points).

    Returns a list of events {target, cls_id, frame, direction, point} sorted by frame.
    """
    events = []
    for key, cls_id, points in trajectories:
        if len(points) < 2:
            continue
        hit = first_crossing(points, tuple(p1), tuple(p2))
        if hit is None:
            continue
        fid, direction, pt = hit
        events.append({"target": key, "cls_id": cls_id, "frame": fid,
                       "direction": direction, "point": [pt[0], pt[1]]})
    events.sort(key=lambda e: e["frame"])
    return events


def in_regions(point, regions_xyxy, margin=0.0):
    """True if point is inside any [x1, y1, x2, y2] region (centre rule, as keep_mask)."""
    x, y = point
    for x1, y1, x2, y2 in regions_xyxy:
        if x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin:
            return True
    return False


def tally(events, classes=CLASSES):
    """Aggregate events into per-class totals with direction breakdown (+ grand total).

    'ab'/'ba' follow crossing_direction: 'ab' = +1 (moved to the _orient > 0 side of the
    line P1->P2), 'ba' = -1.
    Returns {class_name: {"total", "ab", "ba"}, ..., "all": {...}}.
    """
    out = {c: {"total": 0, "ab": 0, "ba": 0} for c in classes}
    for e in events:
        d = out[classes[e["cls_id"]]]
        d["total"] += 1
        d["ab" if e["direction"] > 0 else "ba"] += 1
    out["all"] = {k: sum(out[c][k] for c in classes) for k in ("total", "ab", "ba")}
    return out
