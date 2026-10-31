"""Propose counting lines from GT motion and save previews for manual review"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import DETRAC_TO_ID, RAW_ROOT, REPORTS_DIR
from src.data.detrac_parser import load_all
from src.eval.line_crossing import count_events, in_regions, validate_line
from src.data.frames import list_frames as _list_frames, read_frame
from src.data.provenance import write_json
from src.scripts.count_gt_crossings import _trajectories

MIN_PATH_LEN = 40.0          # Exclude parked targets and positional jitter
NET_RATIO = 0.5              # Require sufficient net displacement relative to path length


def trajectories(ann):
    trajs = {}
    for fid in sorted(ann.frames):
        for t in ann.frames[fid].targets:
            if DETRAC_TO_ID.get(t.cls) is None:
                continue
            trajs.setdefault(t.track_id, []).append((t.box.x + t.box.w / 2, t.box.y + t.box.h / 2))
    return {k: np.asarray(v) for k, v in trajs.items()}


def moving_streams(trajs):
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
    """Generate candidate segments across the stream at several positions"""
    pts = np.concatenate([s["pts"] for s in sel])
    W, H = shape
    u = np.array([-math.sin(theta), math.cos(theta)])      # Define the lateral line direction
    v = np.array([math.cos(theta), math.sin(theta)])       # Define the motion direction
    proj_u, proj_v = pts @ u, pts @ v
    span_u = np.percentile(proj_u, 99) - np.percentile(proj_u, 1)
    extent_v = np.percentile(proj_v, 95) - np.percentile(proj_v, 5)
    for frac in (-0.15, 0.0, 0.15):                        # Test placements along the motion axis
        c_u, c_v = float(np.median(proj_u)), float(np.median(proj_v)) + frac * extent_v
        p0 = c_u * u + c_v * v
        half = min(span_u / 2 * 1.08 + 15, 1.6 * max(W, H))
        a, b = p0 - u * half, p0 + u * half
        clipped = clip_segment(a, b, shape)
        if clipped is not None:
            yield *clipped, frac


def ignored_overlap(a, b, regions):
    if not regions:
        return 0.0
    t = np.linspace(0, 1, 100)[:, None]
    pts = a[None, :] + t * (b - a)[None, :]
    return float(np.mean([in_regions(p, regions) for p in pts]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev"])
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--out", default=str(REPORTS_DIR / "checks" / "proposed_counting_lines.json"))
    ap.add_argument("--preview-dir", default=str(REPORTS_DIR / "checks" / "proposed_line_previews"))
    args = ap.parse_args()
    if Path(args.out).exists():
        ap.error("Output lines already exist, choose a new --out")
    from src.scripts.annotate_line import write_preview

    splits = json.loads((REPORTS_DIR / "splits.json").read_text(encoding="utf-8"))
    anns, _ = load_all(args.raw)
    lines = {}

    for split in args.splits:
        for seq in splits[split]:
            frames = _list_frames(seq, args.raw)
            image = read_frame(frames[0][1])
            shape = (image.shape[1], image.shape[0])
            ann = anns[seq]
            regions = [r.as_xyxy() for r in ann.ignored_regions]
            trajs = trajectories(ann)
            stream = moving_streams(trajs)
            sel, theta = dominant_direction(stream)
            sels = {s["tid"] for s in sel}

            best, best_key = None, None
            for a, b, frac in candidate_lines(sel, theta, shape):
                events = count_events(((tid, cid, points)
                                       for tid, (cid, points) in _trajectories(ann).items()), a, b)
                n_stream = sum(1 for e in events if e["target"] in sels)
                cov = n_stream / max(1, len(sel))
                ign = ignored_overlap(a, b, regions)
                key = (round(cov, 3), -round(ign, 3))      # Prefer coverage before ignored region overlap
                if best_key is None or key > best_key:
                    best, best_key = (a, b, frac, events, cov, ign), key

            if best is None:
                raise ValueError(f"No valid counting line for {seq}")
            a, b, frac, events, cov, ign = best
            ab = sum(1 for e in events if e["direction"] > 0)
            p1, p2 = list(map(int, a)), list(map(int, b))
            validate_line(p1, p2, shape)
            lines[seq] = {"p1": p1, "p2": p2}
            write_preview(frames, seq, lines[seq], Path(args.preview_dir) / f"{seq}.png")
            print(f"{split} {seq}: theta={math.degrees(theta):5.1f}deg stream={len(sel):3d}/{len(stream):3d} "
                  f"cross={len(events):3d} (ab={ab:3d} ba={len(events)-ab:3d}) "
                  f"cov={cov:4.0%} ign={ign:4.0%} shift={frac:+.2f} "
                  f"p1=({a[0]:.0f},{a[1]:.0f}) p2=({b[0]:.0f},{b[1]:.0f})")

    write_json(args.out, lines)


if __name__ == "__main__":
    main()
