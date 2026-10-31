import argparse
import json
from collections import Counter
from pathlib import Path


from ..data.config import CLASSES, RAW_ROOT, REPORTS_DIR
from ..data.mot import load_mot
from ..data.provenance import file_info, write_json
from ..eval.line_crossing import DEFAULT_MAX_FRAME_GAP, count_events, counting_protocol, in_regions, tally
from ..eval.run_validation import validate_tracking_run

TRACKS_DIR = Path("outputs/tracks")


def track_trajectories(path, regions=()):
    trajs, classes = {}, {}
    for fid, targets in sorted(load_mot(path).items()):
        for tid, box, cid, _ in targets:
            centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            if in_regions(centre, regions):
                continue
            trajs.setdefault(tid, []).append((fid, *centre))
            classes.setdefault(tid, Counter())[cid] += 1
    return {tid: (classes[tid].most_common(1)[0][0], points)
            for tid, points in trajs.items()}


def count_file(path, line, regions=(), max_frame_gap=DEFAULT_MAX_FRAME_GAP):
    trajs = track_trajectories(path, regions)
    events = count_events(((tid, cid, points) for tid, (cid, points) in trajs.items()),
                          line["p1"], line["p2"], max_frame_gap=max_frame_gap)
    return {"counts": tally(events), "tracks": len(trajs),
            "events": [[e["frame"], e["target"], e["cls_id"], e["direction"]] for e in events],
            "protocol": counting_protocol(line, regions, max_frame_gap),
            "source": file_info(path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--tracks-dir", default=str(TRACKS_DIR))
    ap.add_argument("--lines", default=str(REPORTS_DIR / "counting_lines.json"))
    ap.add_argument("--out", default=str(REPORTS_DIR / "track_counts.json"))
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--ignored", default=str(REPORTS_DIR / "ignored_regions.json"))
    ap.add_argument("--max-frame-gap", type=int, default=DEFAULT_MAX_FRAME_GAP)
    args = ap.parse_args()
    if args.max_frame_gap < 1:
        ap.error("--max-frame-gap must be positive")
    ignored = json.loads(Path(args.ignored).read_text(encoding="utf-8"))

    lines = json.loads(Path(args.lines).read_text())

    splits = json.loads((REPORTS_DIR / "splits.json").read_text(encoding="utf-8"))

    out_path = Path(args.out)
    result = json.loads(out_path.read_text()) if out_path.exists() else {}  # Preserve results for other splits

    for split in args.splits:
        seqs = list(splits[split])
        split_dir = Path(args.tracks_dir) / split

        if not split_dir.is_dir():
            ap.error(
                f"Tracking directory does not exist: {split_dir}"
            )

        if set(seqs) - set(ignored):
            ap.error(f"Missing ignored region entries for {sorted(set(seqs) - set(ignored))}")

        missing_lines = [
            seq for seq in seqs if seq not in lines
        ]

        if missing_lines:
            ap.error(
                "Missing counting lines for: "
                + ", ".join(missing_lines)
            )

        cfg_dirs = sorted(
            p for p in split_dir.iterdir()
            if p.is_dir()
        )

        if not cfg_dirs:
            ap.error(
                f"No tracking configurations found in {split_dir}"
            )

        # Check all configurations before counting events
        for cfg_dir in cfg_dirs:
            try:
                timing = validate_tracking_run(
                    cfg_dir,
                    seqs,
                    raw_root=Path(args.raw),
                    split=split,
                )
                recorded_regions = timing.get("provenance", {}).get("ignored_regions")
                if recorded_regions is not None and any(recorded_regions.get(seq) != ignored[seq] for seq in seqs):
                    raise ValueError(f"Ignored regions differ from the tracking run: {cfg_dir}")
            except (ValueError, OSError) as exc:
                ap.error(str(exc))

        result[split] = {}

        for cfg_dir in cfg_dirs:
            result[split][cfg_dir.name] = {}

            for seq in seqs:
                res = count_file(
                    cfg_dir / f"{seq}.txt",
                    lines[seq],
                    regions=ignored[seq],
                    max_frame_gap=args.max_frame_gap,
                )

                res["run_metadata"] = file_info(cfg_dir / "timing.json")
                result[split][cfg_dir.name][seq] = res
                counts = res["counts"]

                print(
                    f"[tracks] {split}/{cfg_dir.name}/{seq}: "
                    + " ".join(
                        f"{c}={counts[c]['total']}"
                        for c in CLASSES
                    )
                    + (
                        f" (all={counts['all']['total']}, "
                        f"tracks={res['tracks']})"
                    )
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, result)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
