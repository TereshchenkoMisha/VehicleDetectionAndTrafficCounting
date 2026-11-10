"""Failure candidates of every configuration: missed, extra and misclassified crossings against the GT.

    python scripts/analyze_failures.py --splits dev

The predicted crossings of each configuration are paired with the GT crossings of the same sequence by
box overlap and timing (src/failure/matching.py). A GT crossing without a partner is a missed candidate, a
predicted one without a partner an extra candidate, and a pair with different classes a class confusion.
These are candidates, not confirmed causes, and their counts are not how often each cause occurs: a cause
is only established by looking at the frames (scripts/render_failure_cases.py, then the review).
Only saved files are read, no model is run.

Needs:
    reports/gt_counts.json                 from count_gt
    reports/track_counts.json              from count_tracks
    outputs/tracks/<split>/<configuration>/    complete runs (run_tracking)
    reports/gt_mot/<sequence>.txt          GT in MOT format; created from the XML annotations if missing
    data/raw/UA-DETRAC/                    only for that conversion
    reports/splits.json, reports/ignored_regions.json
Produces:
    <out-prefix>_<split>.json              settings, sources, events, summary, suggested cases
    <out-prefix>_<split>.md                the summary as tables (default prefix reports/failure_analysis)
Stops if track counts were made from different tracks or with a different counting protocol than GT.
"""
import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config import RAW_ROOT, REPORTS_DIR, TRACKS_DIR
from src.data.export import convert_all
from src.data.mot import MotRow, load_mot
from src.data.regions import Region, box_centres, inside_any
from src.failure.diagnosis import FailureRow, classify_events, saved_events, suggest_cases, summarize
from src.failure.matching import MatchSettings, match_boxes
from src.utils import file_info, fraction, matches_file_hash, positive_int, read_json, write_json


def drop_ignored(frames: dict[int, list[MotRow]], regions: list[Region]) -> dict[int, list[MotRow]]:
    """MOT rows whose box centre is outside the ignored regions, the rule the counting uses."""
    kept = {}
    for frame, rows in frames.items():
        ignored = inside_any(box_centres([row[1] for row in rows]), regions)
        kept[frame] = [row for row, skip in zip(rows, ignored, strict=True) if not skip]
    return kept


def analyze_run(sequence: str, config: str, track_file: Path, gt: dict[str, Any], gt_frames: dict[int, list[MotRow]],
                saved: dict[str, Any], regions: list[Region], settings: MatchSettings) -> list[FailureRow]:
    """Failure rows of one configuration on one sequence.

    `gt` is the GT count of the sequence and `saved` the track count of this file. A count that was
    not made from this track file, or under other counting rules than the GT, would not be comparable.
    """
    if not matches_file_hash(track_file, saved["source"]["sha256"]):
        raise SystemExit(f"Track counts do not belong to {track_file}; run count_tracks again")
    if saved["protocol"] != gt["protocol"]:
        raise SystemExit(f"Counting protocol mismatch in {config}/{sequence}; regenerate both counts")

    matches = match_boxes(gt_frames, drop_ignored(load_mot(track_file), regions), settings.match_iou)
    return classify_events(sequence, config, gt["exclude_ignored_events"], saved_events(saved["events"]), matches,
                           settings, gt["protocol"]["max_frame_gap"])


def render_markdown(split: str, result: dict[str, Any]) -> str:
    """Markdown report: outcome counts per configuration, per sequence, and the cases suggested for review."""
    settings = result["settings"]
    md = [f"# Failure candidates - {split} split", "",
          f"Crossings paired by box IoU >= {settings['match_iou']}, same direction, at most "
          f"{settings['event_window']} frames apart. The counts are candidates found automatically, "
          "not confirmed causes and not how often each cause occurs.", ""]
    for title, keep in (("All sequences", lambda row: row["sequence"] == "ALL"),
                        ("Per sequence", lambda row: row["sequence"] != "ALL")):
        md += [f"## {title}", "",
               "| config | sequence | GT | pred | paired | missed | extra | class confusion | signed error |",
               "|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {r['config']} | {r['sequence']} | {r['gt_events']} | {r['pred_events']} | {r['paired']} | "
               f"{r['missed']} | {r['extra']} | {r['class_confusion']} | {r['signed_error']:+d} |"
               for r in result["summary"] if keep(r)]
        md.append("")
    md += ["## Suggested cases for review", ""] + [f"- `{case_id}`" for case_id in result["suggested_cases"]]
    return "\n".join(md) + "\n"


def parse_args() -> argparse.Namespace:
    """Command-line options; see the module docstring for what the script reads and writes."""
    defaults = MatchSettings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev"])
    ap.add_argument("--gt", default=str(REPORTS_DIR / "gt_counts.json"))
    ap.add_argument("--tracks", default=str(REPORTS_DIR / "track_counts.json"))
    ap.add_argument("--tracks-dir", default=str(TRACKS_DIR))
    ap.add_argument("--gt-mot-dir", default=str(REPORTS_DIR / "gt_mot"))
    ap.add_argument("--out-prefix", default=str(REPORTS_DIR / "failure_analysis"))
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--ignored", default=str(REPORTS_DIR / "ignored_regions.json"))
    ap.add_argument("--match-iou", type=fraction, default=defaults.match_iou, help="smallest box overlap to pair boxes")
    ap.add_argument("--event-window", type=positive_int, default=defaults.event_window,
                    help="largest frame distance between paired crossings")
    ap.add_argument("--min-support", type=positive_int, default=defaults.min_support,
                    help="frames with a shared box needed to pair two crossings")
    ap.add_argument("--context", type=positive_int, default=defaults.context,
                    help="frames around a crossing inspected for flags")
    return ap.parse_args()


def main() -> None:
    """For each split: pair the events of every configuration and sequence, then write JSON and Markdown."""
    args = parse_args()
    settings = MatchSettings(args.match_iou, args.event_window, args.min_support, args.context)
    if settings.context < settings.event_window:
        raise SystemExit("--context must be at least --event-window")
    gt_all, pred_all, ignored = read_json(args.gt), read_json(args.tracks), read_json(args.ignored)
    all_splits = read_json(REPORTS_DIR / "splits.json")

    for split in args.splits:
        if split == "test":
            print("TEST split: diagnostics only. Do not change the protocol, thresholds or configurations after this.")
        seqs: list[str] = list(all_splits[split])
        configs = sorted(pred_all[split])
        if not configs:
            raise SystemExit(f"No predicted configurations found for split '{split}'.")
        gt_dir = Path(args.gt_mot_dir)
        if any(not (gt_dir / f"{seq}.txt").is_file() for seq in seqs):
            convert_all(seqs, gt_dir, raw_root=args.raw, ignored_path=args.ignored)

        # One pairing per sequence and configuration; the GT boxes of a sequence are read once
        rows: list[FailureRow] = []
        gt_frames = {seq: load_mot(gt_dir / f"{seq}.txt") for seq in seqs}
        for seq, config in tqdm(list(product(seqs, configs)), desc=split, unit="run"):
            rows += analyze_run(seq, config, Path(args.tracks_dir) / split / config / f"{seq}.txt",
                                gt_all[split][seq], gt_frames[seq], pred_all[split][config][seq], ignored[seq],
                                settings)

        result = {"split": split, "settings": asdict(settings),
                  "sources": {"gt": file_info(args.gt), "tracks": file_info(args.tracks)},
                  "summary": summarize(rows), "suggested_cases": suggest_cases(rows), "rows": rows}
        out_json, out_md = Path(f"{args.out_prefix}_{split}.json"), Path(f"{args.out_prefix}_{split}.md")
        write_json(out_json, result)
        out_md.write_text(render_markdown(split, result), encoding="utf-8")

        print(f"[done] {split}: {len(rows)} events")
        for r in (r for r in result["summary"] if r["sequence"] == "ALL"):
            print(f"  {r['config']}: GT={r['gt_events']} pred={r['pred_events']} missed={r['missed']} "
                  f"extra={r['extra']} class_confusion={r['class_confusion']}")
        print(f"  -> {out_md}\n  -> {out_json}")


if __name__ == "__main__":
    main()
