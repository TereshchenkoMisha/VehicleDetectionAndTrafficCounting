"""Propose a counting line for a sequence from the dominant GT traffic flow.

The proposal is a starting point: a person looks at the preview and corrects the line if needed.
Idea: find the main direction in which vehicles move, then put a line across that direction, near the
middle of the traffic, long enough to cover all lanes.
"""
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..data.detrac import SequenceAnn
from ..data.regions import Region, inside_any
from .line_crossing import count_events, validate_line
from .trajectories import build_trajectories, gt_rows

Points = NDArray[np.float64]            # (N, 2): x, y
Stream = dict[str, Any]                 # {"tid", "ang", "len", "pts"}: one moving target

MIN_PATH_LEN: float = 40.0          # pixels travelled; excludes parked vehicles and annotation jitter
NET_RATIO: float = 0.5              # net displacement / path length; excludes targets that wander or turn around


def moving_streams(trajs: dict[Any, Points]) -> list[Stream]:
    """Pick the targets that really move, with their direction of travel.

    A target qualifies when it has at least 5 points, travelled at least MIN_PATH_LEN pixels, and
    ended up far from where it started (net displacement >= NET_RATIO of the path length), which
    rejects targets that jitter in place or loop. Each result has the angle from start to end,
    reduced mod pi (the axis of travel, direction ignored), and the path length.
    """
    out: list[Stream] = []
    for tid, a in trajs.items():
        if len(a) < 5:
            continue
        steps = np.linalg.norm(np.diff(a, axis=0), axis=1)      # distance between consecutive points
        length = float(steps.sum())
        net = a[-1] - a[0]
        if length < MIN_PATH_LEN or np.linalg.norm(net) < NET_RATIO * length:
            continue
        out.append({"tid": tid, "ang": math.atan2(net[1], net[0]) % math.pi, "len": length, "pts": a})
    return out


def dominant_direction(stream: list[Stream]) -> tuple[list[Stream], float]:
    """Main axis of traffic: the streams that follow it, and its angle.

    Angles are mod pi, so 0 and pi are the same axis and opposite directions of one road belong
    together. That is why the histogram wraps around and the mean uses doubled angles.
    """
    if not stream:
        raise ValueError("No moving trajectories available for line placement")

    # Histogram of travel angles weighted by path length (long paths are more reliable), in 18
    # bins of 10 degrees. Each bin is added to its two neighbours (np.roll wraps around the
    # circle) so a peak split by a bin edge is not lost. The biggest smoothed bin is the main axis.
    bins = np.linspace(0, math.pi, 19)
    hist, _ = np.histogram([item["ang"] for item in stream], bins=bins, weights=[item["len"] for item in stream])
    scores = np.roll(hist, 1) + hist + np.roll(hist, -1)
    centre = (bins[int(scores.argmax())] + bins[int(scores.argmax()) + 1]) / 2

    # Keep the streams within pi/6 (30 degrees) of the peak. The difference is wrapped into
    # [-pi/2, pi/2) so that angles just below pi and just above 0 count as close.
    selected = [item for item in stream
                if abs((item["ang"] - centre + math.pi / 2) % math.pi - math.pi / 2) <= math.pi / 6]

    # Mean axis of the selected streams. A plain mean of angles fails on a circle: the mean of
    # 1 degree and 179 degrees is 90 degrees, but they are almost the same axis. Doubling the angle
    # maps a and a + pi to the same point of a full circle; averaging the unit vectors there and
    # halving the resulting angle gives the right mean axis.
    x = sum(math.cos(2 * item["ang"]) * item["len"] for item in selected)
    y = sum(math.sin(2 * item["ang"]) * item["len"] for item in selected)
    return selected, (math.atan2(y, x) / 2) % math.pi


def clip_segment(a: Points, b: Points, shape: tuple[int, int]) -> tuple[Points, Points] | None:
    """Cut the segment a-b to the image (width, height). Returns the visible part, or None if none.

    Liang-Barsky clipping. The segment is the set of points a + t * (b - a) for t in [0, 1]. For
    each axis the points inside the image satisfy 0 <= a + t * d <= limit, which is a range of t.
    Intersecting those ranges with [0, 1] gives the visible part; if the range is empty nothing is visible.
    """
    width, height = shape
    lower, upper = 0.0, 1.0
    delta = b - a
    for axis, limit in enumerate((width - 1, height - 1)):
        # Segment parallel to this axis' boundary: either entirely inside or entirely outside
        if abs(delta[axis]) < 1e-12:
            if not 0 <= a[axis] <= limit:
                return None
            continue
        # t where the segment enters and leaves the [0, limit] band on this axis
        bounds = sorted((-a[axis] / delta[axis], (limit - a[axis]) / delta[axis]))
        lower, upper = max(lower, bounds[0]), min(upper, bounds[1])
        if lower >= upper:
            return None
    return a + lower * delta, a + upper * delta


def candidate_lines(sel: list[Stream], theta: float, shape: tuple[int, int]) -> Iterator[tuple[Points, Points, float]]:
    """Candidate lines across the main stream, at three positions along the direction of travel.

    `theta` is the travel axis. Each candidate is (start, end, shift), where `shift` is the
    offset from the median position as a fraction of the stream's extent (-0.15, 0, +0.15). The
    line is perpendicular to the travel axis and clipped to the image.
    """
    pts = np.concatenate([s["pts"] for s in sel])
    width, height = shape
    v = np.array([math.cos(theta), math.sin(theta)])       # along the traffic
    u = np.array([-math.sin(theta), math.cos(theta)])      # across the traffic: the line runs along u

    # Coordinates of all points in the (u, v) frame. Percentile ranges ignore outlier points.
    proj_u, proj_v = pts @ u, pts @ v
    span_u = np.percentile(proj_u, 99) - np.percentile(proj_u, 1)       # width of the traffic across the road
    extent_v = np.percentile(proj_v, 95) - np.percentile(proj_v, 5)     # length of the traffic along the road

    for frac in (-0.15, 0.0, 0.15):
        # Centre of the line: the median across the road, the median along the road plus the shift
        c_u, c_v = float(np.median(proj_u)), float(np.median(proj_v)) + frac * extent_v
        centre = c_u * u + c_v * v
        # Half-length: the traffic width plus 8% and 15 px of margin, capped so a degenerate stream
        # cannot produce a line much longer than the image
        half = min(span_u / 2 * 1.08 + 15, 1.6 * max(width, height))
        clipped = clip_segment(centre - u * half, centre + u * half, shape)
        if clipped is not None:
            yield *clipped, frac


def ignored_overlap(a: Points, b: Points, regions: list[Region]) -> float:
    """Share of the segment a-b that lies in ignored regions, estimated at 100 evenly spaced points."""
    if not regions:
        return 0.0
    t = np.linspace(0, 1, 100)[:, None]
    return float(inside_any(a[None, :] + t * (b - a)[None, :], regions).mean())


def propose_line(ann: SequenceAnn, shape: tuple[int, int]) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Choose the best counting line for a sequence. `shape` is (width, height).

    Returns ({"p1": [x, y], "p2": [x, y]}, diagnostics). The candidates are ranked by how many of
    the main stream's vehicles cross them (coverage), then by how little they overlap ignored
    regions. The diagnostics are for printing: angle, stream sizes, events, coverage, overlap, shift.
    """
    # Find the main stream of moving targets and its direction
    trajectories = build_trajectories(gt_rows(ann))
    stream = moving_streams({key: np.asarray([(x, y) for _, x, y in pts]) for key, _, pts in trajectories})
    sel, theta = dominant_direction(stream)
    selected_ids = {s["tid"] for s in sel}
    regions = [r.as_xyxy() for r in ann.ignored_regions]

    # Score each candidate with the real counting code, on all trajectories
    best: tuple[Points, Points, float, list, float, float] | None = None
    best_key: tuple[float, float] | None = None
    for a, b, frac in candidate_lines(sel, theta, shape):
        events = count_events(trajectories, tuple(a), tuple(b))
        coverage = sum(1 for e in events if e["target"] in selected_ids) / max(1, len(sel))
        overlap = ignored_overlap(a, b, regions)
        key = (round(coverage, 3), -round(overlap, 3))      # higher is better in both parts; rounding avoids float ties
        if best_key is None or key > best_key:
            best, best_key = (a, b, frac, events, coverage, overlap), key
    if best is None:
        raise ValueError("No valid counting line")

    # Pixel coordinates as ints, and a final check that the line fits the image
    a, b, frac, events, coverage, overlap = best
    line = {"p1": list(map(int, a)), "p2": list(map(int, b))}
    validate_line(line["p1"], line["p2"], shape)
    info = {"theta_deg": math.degrees(theta), "stream": len(sel), "moving": len(stream),
            "events": len(events), "ab": sum(1 for e in events if e["direction"] > 0),
            "coverage": coverage, "ignored_overlap": overlap, "shift": frac}
    return line, info
