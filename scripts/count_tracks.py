"""Line-crossing counts from the saved tracks of every configuration. Writes reports/track_counts.json

    python scripts/count_tracks.py --splits dev test
"""
import argparse
from pathlib import Path

from src.config import CLASSES, RAW_ROOT, REPORTS_DIR, TRACKS_DIR
from src.counting.line_crossing import DEFAULT_MAX_FRAME_GAP
from src.counting.trajectories import count_tracks
from src.eval.validation import run_dirs, validate_tracking_run
from src.utils import file_info, positive_int, read_json, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--tracks-dir", default=str(TRACKS_DIR))
    ap.add_argument("--lines", default=str(REPORTS_DIR / "counting_lines.json"))
    ap.add_argument("--out", default=str(REPORTS_DIR / "track_counts.json"))
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--ignored", default=str(REPORTS_DIR / "ignored_regions.json"))
    ap.add_argument("--max-frame-gap", type=positive_int, default=DEFAULT_MAX_FRAME_GAP)
    args = ap.parse_args()

    ignored, lines = read_json(args.ignored), read_json(args.lines)
    all_splits = read_json(REPORTS_DIR / "splits.json")
    out_path = Path(args.out)
    result = read_json(out_path) if out_path.exists() else {}     # keeps results of splits not run now

    for split in args.splits:
        seqs = list(all_splits[split])
        split_dir = Path(args.tracks_dir) / split
        if not split_dir.is_dir():
            ap.error(f"Tracking directory does not exist: {split_dir}")
        if set(seqs) - set(ignored):
            ap.error(f"Missing ignored region entries for {sorted(set(seqs) - set(ignored))}")
        if set(seqs) - set(lines):
            ap.error("Missing counting lines for: " + ", ".join(sorted(set(seqs) - set(lines))))
        cfg_dirs = run_dirs(split_dir)
        if not cfg_dirs:
            ap.error(f"No tracking configurations found in {split_dir}")

        for cfg_dir in cfg_dirs:        # check every run before counting anything
            try:
                timing = validate_tracking_run(cfg_dir, seqs, raw_root=Path(args.raw), split=split)
                recorded = timing.get("provenance", {}).get("ignored_regions")
                if recorded is not None and any(recorded.get(seq) != ignored[seq] for seq in seqs):
                    raise ValueError(f"Ignored regions differ from the tracking run: {cfg_dir}")
            except (ValueError, OSError) as exc:
                ap.error(str(exc))

        result[split] = {}
        for cfg_dir in cfg_dirs:
            result[split][cfg_dir.name] = {}
            for seq in seqs:
                res = count_tracks(cfg_dir / f"{seq}.txt", lines[seq], regions=ignored[seq],
                                   max_frame_gap=args.max_frame_gap)
                res["run_metadata"] = file_info(cfg_dir / "timing.json")
                result[split][cfg_dir.name][seq] = res
                counts = res["counts"]
                print(f"[tracks] {split}/{cfg_dir.name}/{seq}: "
                      + " ".join(f"{c}={counts[c]['total']}" for c in CLASSES)
                      + f" (all={counts['all']['total']}, tracks={res['tracks']})")

    write_json(out_path, result)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
