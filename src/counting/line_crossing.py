"""Count the first observed crossing of a finite oriented line per target."""
import math

from ..config import CLASSES

DEFAULT_MAX_FRAME_GAP = 1
PROTOCOL_VERSION = 2


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def validate_line(a, b, shape=None):
    if len(a) != 2 or len(b) != 2:
        raise ValueError("Line endpoints must contain two coordinates")
    if not all(math.isfinite(v) for v in (*a, *b)) or tuple(a) == tuple(b):
        raise ValueError("Line endpoints must be finite and distinct")
    if shape is not None:
        width, height = shape
        if not all(0 <= x < width and 0 <= y < height for x, y in (a, b)):
            raise ValueError(f"Line endpoints fall outside image size {shape}")


def segment_crossing(p1, p2, a, b, eps=1e-6):
    d1, d2 = _orient(a, b, p1), _orient(a, b, p2)
    if d1 == 0 or d2 == 0 or (d1 > 0) == (d2 > 0):
        return None
    t = d1 / (d1 - d2)
    point = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
    if (min(a[0], b[0]) - eps <= point[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= point[1] <= max(a[1], b[1]) + eps):
        return point
    return None


def crossing_direction(p_from, a, b):
    """+1 when the target starts on the negative side of a->b, else -1."""
    return 1 if _orient(a, b, p_from) < 0 else -1


def first_crossing(points, a, b, max_frame_gap=DEFAULT_MAX_FRAME_GAP):
    """First crossing of segment a-b by a trajectory [(frame, x, y), ...], or None.

    Returns (frame, direction, point). A gap of more than max_frame_gap frames resets the trajectory,
    so a crossing that happens inside the gap is not counted. The line must be validated by the caller.
    """
    if type(max_frame_gap) is not int or max_frame_gap < 1:
        raise ValueError("max_frame_gap must be a positive integer")
    anchor = contact = previous_frame = None
    on_line = False
    tolerance = 1e-6 * math.dist(a, b)
    for frame, x, y in points:
        if not all(math.isfinite(v) for v in (frame, x, y)):
            raise ValueError("Trajectory values must be finite")
        if previous_frame is not None:
            if frame <= previous_frame:
                raise ValueError("Trajectory frames must be strictly increasing")
            if frame - previous_frame > max_frame_gap:
                anchor = contact = None
                on_line = False
        previous_frame = frame
        point = (x, y)
        side = _orient(a, b, point)
        if abs(side) <= tolerance:
            on_line = True
            if (anchor is not None and contact is None
                    and min(a[0], b[0]) <= x <= max(a[0], b[0])
                    and min(a[1], b[1]) <= y <= max(a[1], b[1])):
                contact = point
            continue
        if anchor is not None and (side > 0) != (_orient(a, b, anchor) > 0):
            hit = contact if on_line else segment_crossing(anchor, point, a, b)
            if hit is not None:
                return frame, crossing_direction(anchor, a, b), hit
        anchor, contact, on_line = point, None, False
    return None


def count_events(trajectories, p1, p2, max_frame_gap=DEFAULT_MAX_FRAME_GAP):
    """Crossing events for trajectories [(key, cls_id, points), ...], sorted by frame."""
    validate_line(p1, p2)
    events = []
    for key, cls_id, points in trajectories:
        hit = first_crossing(points, p1, p2, max_frame_gap=max_frame_gap)
        if hit is not None:
            frame, direction, point = hit
            events.append({"target": key, "cls_id": cls_id, "frame": frame,
                           "direction": direction, "point": list(point)})
    return sorted(events, key=lambda event: event["frame"])


def counting_protocol(line, regions, max_frame_gap):
    """Everything that must be equal for GT and tracker counts to be comparable."""
    validate_line(line["p1"], line["p2"])
    return {"version": PROTOCOL_VERSION, "line": line,
            "max_frame_gap": max_frame_gap, "ignored_rule": "observation_centres",
            "ignored_regions": [list(region) for region in regions]}


def tally(events, classes=CLASSES):
    """Counts per class and direction (ab / ba), plus an 'all' total."""
    out = {name: {"total": 0, "ab": 0, "ba": 0} for name in classes}
    for event in events:
        counts = out[classes[event["cls_id"]]]
        counts["total"] += 1
        counts["ab" if event["direction"] > 0 else "ba"] += 1
    out["all"] = {key: sum(out[name][key] for name in classes) for key in ("total", "ab", "ba")}
    return out
