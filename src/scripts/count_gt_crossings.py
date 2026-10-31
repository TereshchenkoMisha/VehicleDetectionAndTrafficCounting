import argparse
import json
from collections import Counter
from pathlib import Path


from ..data.config import CLASSES, DETRAC_TO_ID, RAW_ROOT, REPORTS_DIR
from ..data.detrac_parser import load_all
from ..data.frames import list_frames as _list_frames, resolve_naming, read_frame
from ..data.provenance import write_json
from ..eval.line_crossing import DEFAULT_MAX_FRAME_GAP, count_events, counting_protocol, in_regions, tally, validate_line

CLASS_COLORS = {0: (80, 200, 80), 1: (40, 140, 255), 2: (80, 80, 255)}   # Define BGR colours for car bus and truck
LINE_COLOR = (0, 255, 255)


def _trajectories(ann, regions=(), offset=0):
    # Build target trajectories from sequence annotations
    trajs = {}
    for fid in sorted(ann.frames):
        for t in ann.frames[fid].targets:
            cid = DETRAC_TO_ID.get(t.cls)
            if cid is None:
                continue
            c = (t.box.x + t.box.w / 2, t.box.y + t.box.h / 2)
            if not in_regions(c, regions):
                trajs.setdefault(t.track_id, (cid, []))[1].append((fid + offset, c[0], c[1]))
    return trajs


def count_sequence(ann, line, max_frame_gap=DEFAULT_MAX_FRAME_GAP, offset=0):
    regions = [region.as_xyxy() for region in ann.ignored_regions]

    def events_for(excluded):
        trajectories = _trajectories(ann, excluded, offset)
        return count_events(((tid, cid, points) for tid, (cid, points) in trajectories.items()),
                            line["p1"], line["p2"], max_frame_gap=max_frame_gap)

    events = events_for(())
    kept = events_for(regions)
    for event in events:
        event["ignored"] = in_regions(event["point"], regions)
    result = {"all": tally(events), "exclude_ignored": tally(kept),
              "event_fields": ["frame", "target", "cls_id", "direction", "event_point_ignored"],
              "events": [[e["frame"], e["target"], e["cls_id"], e["direction"], int(e["ignored"])]
                         for e in events],
              "exclude_ignored_events": kept,
              "protocol": counting_protocol(line, regions, max_frame_gap)}
    return result, kept


def write_video(ann, line, seq, events, out_path, fps=25, raw_root=RAW_ROOT):
    import cv2
    from tqdm import tqdm

    frames = _list_frames(seq, raw_root)
    offset = resolve_naming(frames[0][1].parent, sorted(ann.frames))
    first = read_frame(frames[0][1])
    height, width = first.shape[:2]
    validate_line(line["p1"], line["p2"], (width, height))
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"Video already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    crosses = {}
    for event in events:
        crosses.setdefault(event["frame"], []).append(event["point"])
    p1, p2 = (tuple(map(int, line[key])) for key in ("p1", "p2"))
    run_total = 0
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {out_path}")
        for fid, path in tqdm(frames, desc=f"video {seq}", unit="frame", leave=False):
            image = read_frame(path)
            if image.shape[:2] != (height, width):
                raise ValueError(f"Frame size changed at {path}")
            frame_ann = ann.frames.get(fid - offset)
            for target in frame_ann.targets if frame_ann is not None else []:
                cid = DETRAC_TO_ID.get(target.cls)
                if cid is None:
                    continue
                x1, y1, x2, y2 = map(int, target.box.as_xyxy())
                cv2.rectangle(image, (x1, y1), (x2, y2), CLASS_COLORS[cid], 2)
            cv2.line(image, p1, p2, LINE_COLOR, 2, cv2.LINE_AA)
            for back in range(13):
                for x, y in crosses.get(fid - back, []):
                    colour = LINE_COLOR if back == 0 else (120, 120, 120)
                    cv2.circle(image, (int(x), int(y)), 18, colour, 2, cv2.LINE_AA)
            run_total += len(crosses.get(fid, []))
            cv2.rectangle(image, (0, 0), (400, 30), (0, 0, 0), -1)
            cv2.putText(image, f"{seq} f{fid} crossed: {run_total}", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--lines", default=str(REPORTS_DIR / "counting_lines.json"))
    ap.add_argument("--out", default=str(REPORTS_DIR / "gt_counts.json"))
    ap.add_argument("--md", default=str(REPORTS_DIR / "gt_counts.md"))
    ap.add_argument("--vis-dir", default=None, help="if set, write a check mp4 per sequence there")
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--max-frame-gap", type=int, default=DEFAULT_MAX_FRAME_GAP)
    args = ap.parse_args()
    if args.max_frame_gap < 1:
        ap.error("--max-frame-gap must be positive")

    from PIL import Image

    splits = json.loads((REPORTS_DIR / "splits.json").read_text())
    lines = json.loads(Path(args.lines).read_text())
    anns, _ = load_all(args.raw)

    out_path = Path(args.out)

    result = (
        json.loads(out_path.read_text(encoding="utf-8"))
        if out_path.is_file()
        else {}
    )

    missing_inputs = [
        seq
        for split in args.splits
        for seq in splits[split]
        if seq not in lines or seq not in anns
    ]

    if missing_inputs:
        ap.error("Missing counting line or XML annotations for: " + ", ".join(sorted(set(missing_inputs))))

    for split in args.splits:
        result[split] = {}
        for seq in splits[split]:
            ann = anns[seq]
            frames = _list_frames(seq, args.raw)
            offset = resolve_naming(frames[0][1].parent, sorted(ann.frames))
            with Image.open(frames[0][1]) as image:
                validate_line(lines[seq]["p1"], lines[seq]["p2"], image.size)
            res, ev_full = count_sequence(ann, lines[seq], args.max_frame_gap, offset)
            res["sources"] = {"lines": str(Path(args.lines).resolve()),
                              "raw_root": str(Path(args.raw).resolve())}
            res["line"] = lines[seq]
            result[split][seq] = res
            print(f"[gt] {split}/{seq}: " +
                  " ".join(f"{c}={res['exclude_ignored'][c]['total']}" for c in CLASSES) +
                  f" (all={res['all']['all']['total']})")

            if args.vis_dir:
                Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
                write_video(ann, lines[seq], seq, ev_full,
                            Path(args.vis_dir) / f"gt_line_{seq}.mp4", raw_root=args.raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, result)

    summary_splits = [
        split
        for split in ("dev", "test")
        if split in result
    ]

    def table(variant):
        head = "| sequence | " + " | ".join(CLASSES) + " | all |\n"
        head += "|" + "---|" * (len(CLASSES) + 2) + "\n"
        rows = ""
        sums = Counter()
        for split in summary_splits:
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
          f"Counting lines in {args.lines}\n\n"
          f"Regenerated splits {', '.join(args.splits)} with maximum observation gap {args.max_frame_gap} frames\n\n"
          "## Tracking-comparable protocol (observations in ignored regions excluded)\n\n" + table("exclude_ignored") +
          "\n## All annotated targets\n\n" + table("all") +
          "\nRead raw events from events and comparable events from exclude_ignored_events in the JSON output\n")
    md_path = Path(args.md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    print(f"[done] {args.out} + {args.md}")


if __name__ == "__main__":
    main()
