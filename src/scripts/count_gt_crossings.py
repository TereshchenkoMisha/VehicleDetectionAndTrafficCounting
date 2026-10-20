"""Ground-truth line-crossing counts per class from the raw UA-DETRAC XML annotations.

For every annotated target the box-centre trajectory is tested against the counting line
(reports/counting_lines.json); each target is counted once, at its first crossing
(src/eval/line_crossing.py). Counts are produced in two variants:
  all             - every annotated target (car/van->car, bus, others->truck)
  exclude_ignored - targets whose centre at the crossing moment lies in an ignored region
                    are dropped; the same rule the tracking pipeline applies to detections
                    (keep_mask before tracker.update), so this is the comparable variant.

Output: reports/gt_counts.json (counts + per-target events) and reports/gt_counts.md.
With --vis-dir an mp4 per sequence is written: line + GT boxes + crossing markers.

Usage:
    python -m src.scripts.count_gt_crossings                     # dev + test
    python -m src.scripts.count_gt_crossings --splits dev
    python -m src.scripts.count_gt_crossings --vis-dir outputs/vis
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

from ..data.config import CLASSES, DETRAC_TO_ID, RAW_ROOT, REPORTS_DIR
from ..data.detrac_parser import find_sequence_dir, load_all
from ..eval.line_crossing import count_events, in_regions, tally

CLASS_COLORS = {0: (80, 200, 80), 1: (40, 140, 255), 2: (80, 80, 255)}   # BGR: car, bus, truck
LINE_COLOR = (0, 255, 255)


def _list_frames(seq_name, raw_root=RAW_ROOT):
    """[(frame_id, path)] sorted by frame id - same rule as pipeline.list_frames
    (duplicated here so this script does not pull in torch/ultralytics)."""
    import re
    seq_dir = find_sequence_dir(raw_root, seq_name)
    if seq_dir is None:
        raise FileNotFoundError(f"frames of {seq_name} not found under {raw_root}")
    frames = [(int(re.sub(r"\D", "", p.stem)), p) for p in seq_dir.glob("*.jpg")]
    return sorted(frames)


def _resolve_naming(seq_dir, frame_ids):
    """Offset between XML frame ids and IMG%05d.jpg names - same rule as convert_yolo."""
    best, best_hits = 0, -1
    for off in (0, 1):
        hits = sum((seq_dir / f"IMG{fid + off:05d}.jpg").exists() for fid in frame_ids[:50])
        if hits > best_hits:
            best, best_hits = off, hits
    return best


def _trajectories(ann):
    """{target_id: (cls_id, [(fid, cx, cy), ...])} from a parsed SequenceAnn."""
    trajs = {}
    for fid in sorted(ann.frames):
        for t in ann.frames[fid].targets:
            cid = DETRAC_TO_ID.get(t.cls)
            if cid is None:
                continue
            c = (t.box.x + t.box.w / 2, t.box.y + t.box.h / 2)
            trajs.setdefault(t.track_id, (cid, []))[1].append((fid, c[0], c[1]))
    return trajs


def count_sequence(ann, line):
    """Events + both count variants for one sequence. Returns (result_dict, events_full)."""
    regions = [r.as_xyxy() for r in ann.ignored_regions]
    trajs = _trajectories(ann)
    events = count_events(((tid, cid, pts) for tid, (cid, pts) in trajs.items()),
                          line["p1"], line["p2"])
    for e in events:
        e["ignored"] = in_regions(e["point"], regions)
    kept = [e for e in events if not e["ignored"]]
    res = {"all": tally(events), "exclude_ignored": tally(kept),
           "events": [[e["frame"], e["target"], e["cls_id"], e["direction"], int(e["ignored"])]
                      for e in events]}
    return res, events


def write_video(ann, line, seq, events, out_path, fps=25, raw_root=RAW_ROOT):
    """GT boxes + line + crossing flashes + running (exclude_ignored) counter."""
    frame_ids = sorted(ann.frames)
    seq_dir = find_sequence_dir(raw_root, seq)
    offset = _resolve_naming(seq_dir, frame_ids)
    jpg = {f: seq_dir / f"IMG{f + offset:05d}.jpg" for f in frame_ids}

    crosses = {}                                   # frame -> [(x, y, cls_id)]
    for e in events:
        if not e.get("ignored"):
            crosses.setdefault(e["frame"], []).append((e["point"][0], e["point"][1], e["cls_id"]))
    events_at = Counter(e["frame"] for e in events if not e.get("ignored"))

    first = cv2.imread(str(jpg[frame_ids[0]]))
    W, H = first.shape[1], first.shape[0]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    p1, p2 = tuple(int(v) for v in line["p1"]), tuple(int(v) for v in line["p2"])
    last, run_total = first, 0
    for fid in tqdm(frame_ids, desc=f"video {seq}", unit="frame", leave=False):
        img = cv2.imread(str(jpg[fid]))
        if img is None:
            img = last.copy()                       # rare missing file: repeat the previous frame
        else:
            last = img
        for t in ann.frames[fid].targets:
            cid = DETRAC_TO_ID.get(t.cls)
            if cid is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in t.box.as_xyxy())
            cv2.rectangle(img, (x1, y1), (x2, y2), CLASS_COLORS[cid], 2)
        cv2.line(img, p1, p2, LINE_COLOR, 2, cv2.LINE_AA)
        for (x, y, _) in crosses.get(fid, []):
            cv2.circle(img, (int(x), int(y)), 18, LINE_COLOR, 3, cv2.LINE_AA)
        for back in range(1, 13):                   # keep the flash visible for 0.5 s
            for (x, y, _) in crosses.get(fid - back, []):
                cv2.circle(img, (int(x), int(y)), 18, (120, 120, 120), 2, cv2.LINE_AA)
        run_total += events_at.get(fid, 0)          # crossings up to and including this frame
        cv2.rectangle(img, (0, 0), (330, 30), (0, 0, 0), -1)
        cv2.putText(img, f"{seq} f{fid}  crossed: {run_total}", (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)
    writer.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--lines", default=str(REPORTS_DIR / "counting_lines.json"))
    ap.add_argument("--out", default=str(REPORTS_DIR / "gt_counts.json"))
    ap.add_argument("--md", default=str(REPORTS_DIR / "gt_counts.md"))
    ap.add_argument("--vis-dir", default=None, help="if set, write a check mp4 per sequence there")
    ap.add_argument("--raw", default=str(RAW_ROOT))
    args = ap.parse_args()

    splits = json.loads((REPORTS_DIR / "splits.json").read_text())
    lines = json.loads(Path(args.lines).read_text())
    anns, _ = load_all(args.raw)

    result, missing = {}, []
    for split in args.splits:
        result[split] = {}
        for seq in splits[split]:
            if seq not in lines:
                missing.append(seq)
                continue
            ann = anns[seq]
            res, ev_full = count_sequence(ann, lines[seq])
            res["line"] = lines[seq]
            result[split][seq] = res
            print(f"[gt] {split}/{seq}: " +
                  " ".join(f"{c}={res['exclude_ignored'][c]['total']}" for c in CLASSES) +
                  f" (all={res['all']['all']['total']})")

            if args.vis_dir:
                Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
                write_video(ann, lines[seq], seq, ev_full,
                            Path(args.vis_dir) / f"gt_line_{seq}.mp4", raw_root=args.raw)

    if missing:
        print(f"[warn] no line in {args.lines} for: {', '.join(missing)} - skipped")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")

    def table(variant):
        head = "| sequence | " + " | ".join(CLASSES) + " | all |\n"
        head += "|" + "---|" * (len(CLASSES) + 2) + "\n"
        rows = ""
        sums = Counter()
        for split in args.splits:
            rows += f"| **{split}** | | | | |\n"
            for seq in splits[split]:
                if seq not in result[split]:
                    rows += f"| {seq} | - | - | - | - |\n"
                    continue
                t = result[split][seq][variant]
                for c in CLASSES + ["all"]:
                    sums[c] += t[c]["total"]
                rows += f"| {seq} | " + " | ".join(str(t[c]["total"]) for c in CLASSES) + \
                        f" | **{t['all']['total']}** |\n"
        rows += f"| **sum** | " + " | ".join(f"**{sums[c]}**" for c in CLASSES) + f" | **{sums['all']}** |\n"
        return head + rows

    md = ("# Ground-truth line-crossing counts\n\n"
          "Counting line: `reports/counting_lines.json`; rule: box centre crosses the line once per target.\n\n"
          "## Tracking-comparable protocol (targets in ignored regions excluded)\n\n" + table("exclude_ignored") +
          "\n## All annotated targets\n\n" + table("all") +
          "\nPer-target events (frame, target, class, direction, ignored) are in `gt_counts.json`.\n")
    Path(args.md).write_text(md)
    print(f"[done] {args.out} + {args.md}")


if __name__ == "__main__":
    main()
