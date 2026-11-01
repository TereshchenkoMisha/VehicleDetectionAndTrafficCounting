"""Line-crossing counts from the saved tracks of every configuration.

    python scripts/count_tracks.py --splits dev test

Every tracking run is first checked for completeness. Then each track file is counted against its
sequence's counting line, with tracker observations in ignored regions dropped (the same rule the GT
"exclude_ignored" variant uses), so the counts can be compared with the GT counts.

Needs:
    outputs/tracks/<split>/<configuration>/    complete runs (run_tracking)
    data/raw/UA-DETRAC/                        frame lists for the run check
    reports/splits.json, reports/counting_lines.json, reports/ignored_regions.json
Produces:
    <out>                                      counts per class and direction, events, protocol, file hashes
                                               (default reports/track_counts.json)
Splits that are not run keep their entries in an existing <out>. Every run is checked before counting starts.
"""
import argparse
from pathlib import Path
from typing import Any

from src.config import CLASSES, RAW_ROOT, REPORTS_DIR, TRACKS_DIR
from src.counting.line_crossing import DEFAULT_MAX_FRAME_GAP
from src.counting.trajectories import count_tracks
from src.eval.validation import run_dirs, validate_tracking_run
from src.utils import file_info, positive_int, read_json, write_json


def parse_args() -> argparse.Namespace:
    """Command-line options; see the module docstring for what the script reads and writes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--tracks-dir", default=str(TRACKS_DIR))
    ap.add_argument("--lines", default=str(REPORTS_DIR / "counting_lines.json"))
    ap.add_argument("--out", default=str(REPORTS_DIR / "track_counts.json"))
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--ignored", default=str(REPORTS_DIR / "ignored_regions.json"))
    ap.add_argument("--max-frame-gap", type=positive_int, default=DEFAULT_MAX_FRAME_GAP)
    return ap.parse_args()


def main() -> None:
    """Check the runs of each split, count every track file and write the result."""
    args = parse_args()
    ignored, lines = read_json(args.ignored), read_json(args.lines)
    all_splits = read_json(REPORTS_DIR / "splits.json")
    out_path = Path(args.out)
    result: dict[str, Any] = read_json(out_path) if out_path.exists() else {}       # keeps splits not run now

    for split in args.splits:
        # Every sequence of the split needs a counting line and ignored regions
        seqs: list[str] = list(all_splits[split])
        split_dir = Path(args.tracks_dir) / split
        if not split_dir.is_dir():
            raise SystemExit(f"Tracking directory does not exist: {split_dir}")
        if set(seqs) - set(ignored):
            raise SystemExit(f"Missing ignored region entries for {sorted(set(seqs) - set(ignored))}")
        if set(seqs) - set(lines):
            raise SystemExit("Missing counting lines for: " + ", ".join(sorted(set(seqs) - set(lines))))
        cfg_dirs = run_dirs(split_dir)
        if not cfg_dirs:
            raise SystemExit(f"No tracking configurations found in {split_dir}")

        # Check every run before counting any. Besides completeness, the ignored regions recorded in
        # the run must equal the ones used here: tracks made with other regions are not comparable with GT.
        for cfg_dir in cfg_dirs:
            try:
                timing = validate_tracking_run(cfg_dir, seqs, raw_root=Path(args.raw), split=split)
                recorded = timing.get("provenance", {}).get("ignored_regions")
                if recorded is not None and any(recorded.get(seq) != ignored[seq] for seq in seqs):
                    raise ValueError(f"Ignored regions differ from the tracking run: {cfg_dir}")
            except (ValueError, OSError) as exc:
                raise SystemExit(str(exc)) from exc

        result[split] = {}
        for cfg_dir in cfg_dirs:
            result[split][cfg_dir.name] = {}
            for seq in seqs:
                res = count_tracks(cfg_dir / f"{seq}.txt", lines[seq], regions=ignored[seq],
                                   max_frame_gap=args.max_frame_gap)
                res["run_metadata"] = file_info(cfg_dir / "timing.json")        # ties the count to its run
                result[split][cfg_dir.name][seq] = res
                counts = res["counts"]
                print(f"[tracks] {split}/{cfg_dir.name}/{seq}: "
                      + " ".join(f"{c}={counts[c]['total']}" for c in CLASSES)
                      + f" (all={counts['all']['total']}, tracks={res['tracks']})")

    write_json(out_path, result)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
