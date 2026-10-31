"""Place initial counting lines automatically from the GT flow field (one-off utility).

For every sequence of the reference split the dominant traffic stream is found from the
GT trajectories (length-weighted, undirected so both directions of one road merge), and
the counting line is built perpendicular to it, spanning the stream's lateral extent.
Each placement is scored by GT crossings and overlap with ignored regions and printed
for manual review; previews go to reports/line_previews/.

The result is a starting point for src/scripts/annotate_line.py (interactive or headless),
which is the tool used to correct any line by hand.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import DETRAC_TO_ID, RAW_ROOT, REPORTS_DIR
from src.data.detrac_parser import load_all
from src.eval.line_crossing import count_events, in_regions
from src.scripts.annotate_line import _list_frames, load_lines, save_line, write_preview

MIN_PATH_LEN = 40.0          # px, ignore parked / jitter-only targets
NET_RATIO = 0.5              # net displacement must be >= this fraction of path length


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
    bins = np.linspace(0, math.pi, 19)                     # 10-degree bins
    hist, _ = np.histogram([s["ang"] for s in stream], bins=bins,
                           weights=[s["len"] for s in stream])
    k = int(hist.argmax())
    sel = [s for s in stream if bins[k] - 0.26 <= s["ang"] <= bins[k + 1] + 0.26]
    x = sum(math.cos(2 * s["ang"]) * s["len"] for s in sel)
    y = sum(math.sin(2 * s["ang"]) * s["len"] for s in sel)
    return sel, (math.atan2(y, x) / 2) % math.pi


def candidate_lines(sel, theta, shape):
    """Candidate placements: the stream centre plus shifts along the motion axis."""
    pts = np.concatenate([s["pts"] for s in sel])
    W, H = shape
    u = np.array([-math.sin(theta), math.cos(theta)])      # lateral (line direction)
    v = np.array([math.cos(theta), math.sin(theta)])       # motion direction
    proj_u, proj_v = pts @ u, pts @ v
    span_u = np.percentile(proj_u, 99) - np.percentile(proj_u, 1)
    extent_v = np.percentile(proj_v, 95) - np.percentile(proj_v, 5)
    for frac in (-0.15, 0.0, 0.15):                        # nearer / centre / farther
        c_u, c_v = float(np.median(proj_u)), float(np.median(proj_v)) + frac * extent_v
        p0 = c_u * u + c_v * v
        half = min(span_u / 2 * 1.08 + 15, 1.6 * max(W, H))
        a, b = p0 - u * half, p0 + u * half
        a = np.clip(a, [4, 4], [W - 4, H - 4])
        b = np.clip(b, [4, 4], [W - 4, H - 4])
        yield a, b, frac


def ignored_overlap(a, b, regions):
    if not regions:
        return 0.0
    t = np.linspace(0, 1, 100)[:, None]
    pts = a[None, :] + t * (b - a)[None, :]
    return float(np.mean([in_regions(p, regions) for p in pts]))


def main():
    splits = json.loads((REPORTS_DIR / "splits.json").read_text())
    anns, _ = load_all(RAW_ROOT)
    lines = load_lines(REPORTS_DIR / "counting_lines.json")

    for split in ("dev", "test"):
        for seq in splits[split]:
            ann = anns[seq]
            regions = [r.as_xyxy() for r in ann.ignored_regions]
            trajs = trajectories(ann)
            stream = moving_streams(trajs)
            sel, theta = dominant_direction(stream)
            sels = {s["tid"] for s in sel}

            best, best_key = None, None
            for a, b, frac in candidate_lines(sel, theta, (960, 540)):
                events = count_events(((tid, 0, [(i, p[0], p[1]) for i, p in enumerate(pts)])
                                       for tid, pts in trajs.items()), a, b)
                n_stream = sum(1 for e in events if e["target"] in sels)
                cov = n_stream / max(1, len(sel))
                ign = ignored_overlap(a, b, regions)
                key = (round(cov, 3), -round(ign, 3))      # coverage first, then less ignored
                if best_key is None or key > best_key:
                    best, best_key = (a, b, frac, events, cov, ign), key

            a, b, frac, events, cov, ign = best
            ab = sum(1 for e in events if e["direction"] > 0)
            save_line(lines, REPORTS_DIR / "counting_lines.json", seq, a, b)
            frames = _list_frames(seq)
            write_preview(frames, seq, lines[seq], REPORTS_DIR / "line_previews" / f"{seq}.png")
            print(f"{split} {seq}: theta={math.degrees(theta):5.1f}deg stream={len(sel):3d}/{len(stream):3d} "
                  f"cross={len(events):3d} (ab={ab:3d} ba={len(events)-ab:3d}) "
                  f"cov={cov:4.0%} ign={ign:4.0%} shift={frac:+.2f} "
                  f"p1=({a[0]:.0f},{a[1]:.0f}) p2=({b[0]:.0f},{b[1]:.0f})")


if __name__ == "__main__":
    main()
