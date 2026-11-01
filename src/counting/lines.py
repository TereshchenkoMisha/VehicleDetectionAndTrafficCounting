"""Propose a counting line per sequence from the dominant GT traffic flow (the result is reviewed by hand)."""
import math

import numpy as np

from ..data.regions import inside_any
from .line_crossing import count_events, validate_line
from .trajectories import build_trajectories, gt_rows

MIN_PATH_LEN = 40.0          # pixels; excludes parked targets and positional jitter
NET_RATIO = 0.5              # required net displacement relative to path length


def moving_streams(trajs):
    """Moving targets with their travel angle (mod pi) and path length. trajs: {key: (N, 2) array}."""
    out = []
    for tid, a in trajs.items():
        if len(a) < 5:
            continue
        steps = np.linalg.norm(np.diff(a, axis=0), axis=1)
        length = float(steps.sum())
        net = a[-1] - a[0]
        if length < MIN_PATH_LEN or np.linalg.norm(net) < NET_RATIO * length:
            continue
        out.append({"tid": tid, "ang": math.atan2(net[1], net[0]) % math.pi,
                    "len": length, "pts": a})
    return out


def dominant_direction(stream):
    """Streams within 30 degrees of the main direction and the length-weighted mean angle of that group."""
    if not stream:
        raise ValueError("No moving trajectories available for line placement")
    bins = np.linspace(0, math.pi, 19)
    hist, _ = np.histogram([item["ang"] for item in stream], bins=bins,
                           weights=[item["len"] for item in stream])
    scores = np.roll(hist, 1) + hist + np.roll(hist, -1)
    centre = (bins[int(scores.argmax())] + bins[int(scores.argmax()) + 1]) / 2
    selected = [item for item in stream
                if abs((item["ang"] - centre + math.pi / 2) % math.pi - math.pi / 2) <= math.pi / 6]
    x = sum(math.cos(2 * item["ang"]) * item["len"] for item in selected)
    y = sum(math.sin(2 * item["ang"]) * item["len"] for item in selected)
    return selected, (math.atan2(y, x) / 2) % math.pi


def clip_segment(a, b, shape):
    """Clip segment a-b to the image (width, height); None if nothing is left."""
    width, height = shape
    lower, upper = 0.0, 1.0
    delta = b - a
    for axis, limit in enumerate((width - 1, height - 1)):
        if abs(delta[axis]) < 1e-12:
            if not 0 <= a[axis] <= limit:
                return None
            continue
        bounds = sorted((-a[axis] / delta[axis], (limit - a[axis]) / delta[axis]))
        lower, upper = max(lower, bounds[0]), min(upper, bounds[1])
        if lower >= upper:
            return None
    return a + lower * delta, a + upper * delta


def candidate_lines(sel, theta, shape):
    """Candidate segments across the stream at three positions along the motion axis."""
    pts = np.concatenate([s["pts"] for s in sel])
    width, height = shape
    u = np.array([-math.sin(theta), math.cos(theta)])      # lateral direction (along the line)
    v = np.array([math.cos(theta), math.sin(theta)])       # motion direction
    proj_u, proj_v = pts @ u, pts @ v
    span_u = np.percentile(proj_u, 99) - np.percentile(proj_u, 1)
    extent_v = np.percentile(proj_v, 95) - np.percentile(proj_v, 5)
    for frac in (-0.15, 0.0, 0.15):
        c_u, c_v = float(np.median(proj_u)), float(np.median(proj_v)) + frac * extent_v
        p0 = c_u * u + c_v * v
        half = min(span_u / 2 * 1.08 + 15, 1.6 * max(width, height))
        clipped = clip_segment(p0 - u * half, p0 + u * half, shape)
        if clipped is not None:
            yield *clipped, frac


def ignored_overlap(a, b, regions):
    """Share of the segment a-b that lies inside ignored regions."""
    if not regions:
        return 0.0
    t = np.linspace(0, 1, 100)[:, None]
    return float(inside_any(a[None, :] + t * (b - a)[None, :], regions).mean())


def propose_line(ann, shape):
    """Best candidate line for one SequenceAnn. shape = (width, height).

    Returns {"p1", "p2"} (ints) and a diagnostics dict. Candidates are ranked by the share of the main
    stream that crosses them, then by the least overlap with ignored regions.
    """
    trajectories = build_trajectories(gt_rows(ann))
    stream = moving_streams({key: np.asarray([(x, y) for _, x, y in pts]) for key, _, pts in trajectories})
    sel, theta = dominant_direction(stream)
    selected_ids = {s["tid"] for s in sel}
    regions = [r.as_xyxy() for r in ann.ignored_regions]

    best, best_key = None, None
    for a, b, frac in candidate_lines(sel, theta, shape):
        events = count_events(trajectories, a, b)
        coverage = sum(1 for e in events if e["target"] in selected_ids) / max(1, len(sel))
        overlap = ignored_overlap(a, b, regions)
        key = (round(coverage, 3), -round(overlap, 3))
        if best_key is None or key > best_key:
            best, best_key = (a, b, frac, events, coverage, overlap), key
    if best is None:
        raise ValueError("No valid counting line")

    a, b, frac, events, coverage, overlap = best
    line = {"p1": list(map(int, a)), "p2": list(map(int, b))}
    validate_line(line["p1"], line["p2"], shape)
    info = {"theta_deg": math.degrees(theta), "stream": len(sel), "moving": len(stream),
            "events": len(events), "ab": sum(1 for e in events if e["direction"] > 0),
            "coverage": coverage, "ignored_overlap": overlap, "shift": frac}
    return line, info
