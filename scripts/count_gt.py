"""Ground-truth line-crossing counts from the XML annotations.

    python scripts/count_gt.py --splits dev test --vis-dir outputs/vis

For each sequence the GT trajectories are checked against the sequence's counting line, and the crossings
are tallied per class and direction. Two variants are stored: all annotated targets, and only the
observations outside ignored regions. The second is what tracker counts are compared with.

Needs:
    data/raw/UA-DETRAC/                    frames and XML annotations
    reports/splits.json
    reports/counting_lines.json            one line per sequence
Produces:
    <out>                                  counts per class and direction, events, counting protocol
                                           (default reports/gt_counts.json)
    <md>                                   the same as tables (default reports/gt_counts.md)
    <vis-dir>/gt_line_<sequence>.mp4       check videos, with --vis-dir
Splits that are not run keep their entries in an existing <out>.
"""
import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from src.config import CLASSES, RAW_ROOT, REPORTS_DIR
from src.counting.line_crossing import DEFAULT_MAX_FRAME_GAP, validate_line
from src.counting.trajectories import count_gt
from src.data.detrac import load_all
from src.data.sequences import list_frames, resolve_naming
from src.tracking.viz import write_gt_video
from src.utils import positive_int, read_json, write_json


def markdown_table(result: dict[str, Any], splits: dict[str, list[str]], summary_splits: list[str],
                   variant: str) -> str:
    """Markdown table of totals per sequence and class for one variant, with a sum row at the end.

    `variant` is "exclude_ignored" or "all". Sequences missing from `result` show as dashes.
    """
    rows = "| sequence | " + " | ".join(CLASSES) + " | all |\n" + "|" + "---|" * (len(CLASSES) + 2) + "\n"
    sums: Counter[str] = Counter()
    for split in summary_splits:
        rows += f"| **{split}** | | | | |\n"
        for seq in splits[split]:
            if seq not in result[split]:
                rows += f"| {seq} | - | - | - | - |\n"
                continue
            t = result[split][seq][variant]
            for c in CLASSES + ["all"]:
                sums[c] += t[c]["total"]
            rows += (f"| {seq} | " + " | ".join(str(t[c]["total"]) for c in CLASSES)
                     + f" | **{t['all']['total']}** |\n")
    rows += "| **sum** | " + " | ".join(f"**{sums[c]}**" for c in CLASSES) + f" | **{sums['all']}** |\n"
    return rows


def parse_args() -> argparse.Namespace:
    """Command-line options; see the module docstring for what the script reads and writes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--lines", default=str(REPORTS_DIR / "counting_lines.json"))
    ap.add_argument("--out", default=str(REPORTS_DIR / "gt_counts.json"))
    ap.add_argument("--md", default=str(REPORTS_DIR / "gt_counts.md"))
    ap.add_argument("--vis-dir", default=None, help="write a check mp4 per sequence there")
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--max-frame-gap", type=positive_int, default=DEFAULT_MAX_FRAME_GAP)
    return ap.parse_args()


def main() -> None:
    """Count the GT crossings of every sequence in the chosen splits and write JSON and Markdown."""
    args = parse_args()
    splits = read_json(REPORTS_DIR / "splits.json")
    lines = read_json(args.lines)
    anns, _ = load_all(args.raw)
    out_path = Path(args.out)
    result: dict[str, Any] = read_json(out_path) if out_path.is_file() else {}      # keeps splits not run now

    missing = sorted({seq for split in args.splits for seq in splits[split] if seq not in lines or seq not in anns})
    if missing:
        raise SystemExit("Missing counting line or XML annotations for: " + ", ".join(missing))

    for split in args.splits:
        result[split] = {}
        for seq in splits[split]:
            ann = anns[seq]
            frames = list_frames(seq, args.raw)
            offset = resolve_naming(frames[0][1].parent, sorted(ann.frames))        # XML -> JPG frame numbers
            with Image.open(frames[0][1]) as image:
                validate_line(lines[seq]["p1"], lines[seq]["p2"], image.size)       # the line must fit the image

            res, kept_events = count_gt(ann, lines[seq], args.max_frame_gap, offset)
            res["sources"] = {"lines": str(Path(args.lines).resolve()), "raw_root": str(Path(args.raw).resolve())}
            res["line"] = lines[seq]
            result[split][seq] = res
            print(f"[gt] {split}/{seq}: " + " ".join(f"{c}={res['exclude_ignored'][c]['total']}" for c in CLASSES)
                  + f" (all={res['all']['all']['total']})")

            if args.vis_dir:
                write_gt_video(ann, lines[seq], seq, kept_events, Path(args.vis_dir) / f"gt_line_{seq}.mp4",
                               raw_root=args.raw)

    write_json(out_path, result)
    summary_splits = [s for s in ("dev", "test") if s in result]
    md = ("# Ground-truth line-crossing counts\n\n"
          f"Counting lines in {args.lines}\n\n"
          f"Regenerated splits {', '.join(args.splits)} with maximum observation gap {args.max_frame_gap} frames\n\n"
          "## Tracking-comparable protocol (observations in ignored regions excluded)\n\n"
          + markdown_table(result, splits, summary_splits, "exclude_ignored")
          + "\n## All annotated targets\n\n" + markdown_table(result, splits, summary_splits, "all")
          + "\nRaw events are in `events`, comparable events in `exclude_ignored_events` of the JSON output\n")
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text(md, encoding="utf-8")
    print(f"[done] {args.out} + {args.md}")


if __name__ == "__main__":
    main()
