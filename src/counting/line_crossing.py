"""Detect when a trajectory crosses a finite line segment, and count the crossings.

Counting rules, which GT and tracker counts share so they stay comparable:
  - a target is counted once, at its first crossing of the segment;
  - the direction is given by the side the target started on;
  - when two consecutive observations of a target are more than `max_frame_gap` frames apart, the
    earlier points are forgotten, so a crossing inside the gap is lost. With the default of 1, a
    single missed frame is enough.
"""
import math
from collections.abc import Iterable, Sequence
from typing import Any, NotRequired, TypedDict

from ..config import CLASSES

Point = tuple[float, float]
Trajectory = tuple[Any, int, Sequence[tuple[int, float, float]]]       # (target key, class id, [(frame, x, y), ...])

DEFAULT_MAX_FRAME_GAP: int = 1

# Version of the counting rules in this module (first crossing only, direction by start side, frame gap
# reset). Stored in every count file. Counts made under different rules are not comparable, so
# `evaluate_counting` rejects GT counts whose version is not the current one: bump this number
# whenever the rules change, and old files are refused until they are regenerated.
PROTOCOL_VERSION: int = 2


class CrossingEvent(TypedDict):
    """One counted crossing."""
    target: Any
    cls_id: int
    frame: int              # frame where the trajectory is first seen on the far side of the line
    direction: int          # +1: the target started on the negative side of p1->p2, -1: on the positive side
    point: list[float]      # where the path meets the line
    ignored: NotRequired[bool]      # set by count_gt: the crossing point lies in an ignored region


def _orient(a: Point, b: Point, c: Point) -> float:
    """Which side of the directed line a->b the point c is on: the cross product (b - a) x (c - a).

    Image coordinates have y pointing down, so positive means c appears to the right of a->b on
    screen, negative to the left, zero on the line. The magnitude is |ab| times the distance of c
    from the line.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def validate_line(a: Sequence[float], b: Sequence[float], shape: tuple[int, int] | None = None) -> None:
    """Raise ValueError unless a-b is a usable counting line.

    Both ends need two finite coordinates and must differ. With `shape` = (width, height) both
    ends must also lie inside the image.
    """
    if len(a) != 2 or len(b) != 2:
        raise ValueError("Line endpoints must contain two coordinates")
    if not all(math.isfinite(v) for v in (*a, *b)) or tuple(a) == tuple(b):
        raise ValueError("Line endpoints must be finite and distinct")
    if shape is not None:
        width, height = shape
        if not all(0 <= x < width and 0 <= y < height for x, y in (a, b)):
            raise ValueError(f"Line endpoints fall outside image size {shape}")


def segment_crossing(p1: Point, p2: Point, a: Point, b: Point, eps: float = 1e-6) -> Point | None:
    """Point where the step p1->p2 crosses the segment a-b, or None.

    The step must cross the infinite line (p1, p2 on opposite sides) and the crossing point must lie
    on the segment itself, within `eps` pixels. A step that starts or ends exactly on the line
    returns None; `first_crossing` handles that case.
    """
    d1, d2 = _orient(a, b, p1), _orient(a, b, p2)
    if d1 == 0 or d2 == 0 or (d1 > 0) == (d2 > 0):
        return None

    # d1 and d2 are proportional to the distances of p1 and p2 from the line, with opposite
    # signs. The step therefore meets the line at the fraction t = d1 / (d1 - d2) of its length.
    t = d1 / (d1 - d2)
    point = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    # The point is on the infinite line; check it is within the segment's bounding box
    if (min(a[0], b[0]) - eps <= point[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= point[1] <= max(a[1], b[1]) + eps):
        return point
    return None


def crossing_direction(p_from: Point, a: Point, b: Point) -> int:
    """+1 when the target started on the negative side of the directed line a->b, else -1."""
    return 1 if _orient(a, b, p_from) < 0 else -1


def first_crossing(points: Iterable[tuple[int, float, float]], a: Point, b: Point,
                   max_frame_gap: int = DEFAULT_MAX_FRAME_GAP) -> tuple[int, int, Point] | None:
    """First crossing of segment a-b by one trajectory, or None.

    `points` is [(frame, x, y), ...] with strictly increasing frames; the line must be validated by
    the caller. Returns (frame, direction, point), `frame` being the first frame seen beyond the line.
    A crossing is a new off-line point on the other side than the anchor (the last off-line point).
    A point exactly on the line is not decided at once: the target may touch the line and turn
    back, so the decision waits for the next off-line point.
    """
    if type(max_frame_gap) is not int or max_frame_gap < 1:
        raise ValueError("max_frame_gap must be a positive integer")

    anchor: Point | None = None         # last point that was off the line: start of the next step
    contact: Point | None = None        # first on-line point since the anchor, inside the segment's extent
    on_line = False                     # some point since the anchor was exactly on the line
    previous_frame: int | None = None
    # |cross product| = |ab| * distance from the line, so this limits the distance to 1e-6 px
    tolerance = 1e-6 * math.dist(a, b)

    for frame, x, y in points:
        if not all(math.isfinite(v) for v in (frame, x, y)):
            raise ValueError("Trajectory values must be finite")

        # Frame gap: forget the anchor. The step across a gap is not trusted, since the target
        # could have gone anywhere while unseen.
        if previous_frame is not None:
            if frame <= previous_frame:
                raise ValueError("Trajectory frames must be strictly increasing")
            if frame - previous_frame > max_frame_gap:
                anchor = contact = None
                on_line = False
        previous_frame = frame

        point = (x, y)
        side = _orient(a, b, point)

        # Point on the line: remember where it touched (only if inside the segment's extent and
        # after an anchor exists), then wait for the next off-line point to decide
        if abs(side) <= tolerance:
            on_line = True
            if (anchor is not None and contact is None
                    and min(a[0], b[0]) <= x <= max(a[0], b[0])
                    and min(a[1], b[1]) <= y <= max(a[1], b[1])):
                contact = point
            continue

        # Off-line point on the other side than the anchor: a crossing. If the path touched the
        # line on the way, the touch point is the crossing; otherwise intersect the step with the segment.
        if anchor is not None and (side > 0) != (_orient(a, b, anchor) > 0):
            hit = contact if on_line else segment_crossing(anchor, point, a, b)
            if hit is not None:
                return frame, crossing_direction(anchor, a, b), hit
        anchor, contact, on_line = point, None, False
    return None


def count_events(trajectories: Iterable[Trajectory], p1: Point, p2: Point,
                 max_frame_gap: int = DEFAULT_MAX_FRAME_GAP) -> list[CrossingEvent]:
    """Find the first crossing of the line p1-p2 for each trajectory.

    Trajectories that never cross produce no event. The events are returned sorted by frame. The
    line is validated once here, not per trajectory.
    """
    validate_line(p1, p2)
    events: list[CrossingEvent] = []
    for key, cls_id, points in trajectories:
        hit = first_crossing(points, p1, p2, max_frame_gap=max_frame_gap)
        if hit is not None:
            frame, direction, point = hit
            events.append({"target": key, "cls_id": cls_id, "frame": frame,
                           "direction": direction, "point": list(point)})
    return sorted(events, key=lambda event: event["frame"])


def counting_protocol(line: dict[str, list[int]], regions: Sequence[Sequence[float]],
                      max_frame_gap: int) -> dict[str, Any]:
    """Describe the settings a count was made with: line, ignored regions, frame gap, rule version.

    Stored with every count, for GT and tracker alike. A GT count and a tracker count measure the same
    thing only if all of these are equal: another line or frame gap counts other events, other ignored
    regions remove other observations. `evaluate_counting` compares the two protocols and refuses to
    compute errors when they differ, because the difference would come from the settings, not from
    the tracker.
    """
    validate_line(line["p1"], line["p2"])
    return {"version": PROTOCOL_VERSION, "line": line,
            "max_frame_gap": max_frame_gap, "ignored_rule": "observation_centres",
            "ignored_regions": [list(region) for region in regions]}


def tally(events: Iterable[CrossingEvent], classes: list[str] = CLASSES) -> dict[str, dict[str, int]]:
    """Count events per class: total, and per direction (ab = direction +1, ba = -1).

    The result also has an "all" entry that sums the classes.
    """
    out = {name: {"total": 0, "ab": 0, "ba": 0} for name in classes}
    for event in events:
        counts = out[classes[event["cls_id"]]]
        counts["total"] += 1
        counts["ab" if event["direction"] > 0 else "ba"] += 1
    out["all"] = {key: sum(out[name][key] for name in classes) for key in ("total", "ab", "ba")}
    return out
