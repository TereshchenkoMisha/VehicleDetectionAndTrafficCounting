"""Line-crossing counts from the saved MOT tracks (outputs/tracks/<split>/<cfg>/<seq>.txt).

Same crossing rule as the ground-truth counter (src/eval/line_crossing.py): the segment
between the box centres of a track id at consecutive tracked frames is tested against the
counting line; each track id is counted once, at its first crossing. Ignored regions are
NOT re-applied here - detections with centres there were already dropped before the
tracker (keep_mask in the tracking pipeline), so the comparison with the GT
exclude_ignored variant is on equal terms.

Output: reports/track_counts.json

Usage:
    python -m src.scripts.count_tracks_crossings                  # dev + test, all configs
    python -m src.scripts.count_tracks_crossings --split dev
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from ..data.config import CLASSES, REPORTS_DIR
from ..eval.line_crossing import count_events, tally

TRACKS_DIR = Path("outputs/tracks")


def track_trajectories(path):
    """{track_id: (majority_cls, [(frame, cx, cy), ...])} parsed from a MOT txt file."""
    trajs = {}
    classes = {}
    with open(path) as f:
        for line in f:
            fid, tid, x, y, w, h, _s, c, *_ = line.split(",")
            tid, c = int(tid), int(c)
            trajs.setdefault(tid, []).append((int(fid), float(x) + float(w) / 2, float(y) + float(h) / 2))
            classes.setdefault(tid, Counter())[c] += 1
    return {tid: (classes[tid].most_common(1)[0][0], sorted(pts))
            for tid, pts in trajs.items()}


def count_file(path, line):
    trajs = track_trajectories(path)
    events = count_events(((tid, cid, pts) for tid, (cid, pts) in trajs.items()),
                          line["p1"], line["p2"])
    return {"counts": tally(events), "tracks": len(trajs),
            "events": [[e["frame"], e["target"], e["cls_id"], e["direction"]] for e in events]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--tracks-dir", default=str(TRACKS_DIR))
    ap.add_argument("--lines", default=str(REPORTS_DIR / "counting_lines.json"))
    ap.add_argument("--out", default=str(REPORTS_DIR / "track_counts.json"))
    args = ap.parse_args()

    lines = json.loads(Path(args.lines).read_text())
    out_path = Path(args.out)
    result = json.loads(out_path.read_text()) if out_path.exists() else {}  # merge, keep other splits
    for split in args.splits:
        split_dir = Path(args.tracks_dir) / split
        if not split_dir.exists():
            print(f"[warn] {split_dir} not found - run run_tracking first, skipping")
            continue
        result[split] = {}
        for cfg_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            result[split][cfg_dir.name] = {}
            for seq_txt in sorted(cfg_dir.glob("MVI_*.txt")):
                seq = seq_txt.stem
                if seq not in lines:
                    print(f"[warn] no counting line for {seq}, skipped")
                    continue
                res = count_file(seq_txt, lines[seq])
                result[split][cfg_dir.name][seq] = res
                t = res["counts"]
                print(f"[tracks] {split}/{cfg_dir.name}/{seq}: " +
                      " ".join(f"{c}={t[c]['total']}" for c in CLASSES) +
                      f" (all={t['all']['total']}, tracks={res['tracks']})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
