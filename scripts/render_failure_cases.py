"""Pictures of failure cases: the same vehicle and frames in every configuration, side by side.

    python scripts/render_failure_cases.py --analysis reports/failure_analysis_dev.json --video

By default the cases of the review file are drawn; --case-id draws any case of the analysis instead (for
example one of its suggested cases, before it has been reviewed). Tracks are drawn from the saved MOT
files, no model is run. A case that has no GT crossing (an extra one) needs `reference_gt_id` in the
review to show which GT vehicle it duplicates.

Needs:
    <analysis>                             from analyze_failures
    reports/failure_review.json            reviewed cases (--review)
    outputs/tracks/<split>/<configuration>/    MOT tracks (run_tracking)
    reports/gt_mot/<sequence>.txt          GT in MOT format (analyze_failures creates it)
    data/raw/UA-DETRAC/                    frames
    reports/counting_lines.json, reports/ignored_regions.json
Produces, per case, in <out-dir>/<split>/<case id>/:
    comparison.png                         close-ups before, at and after the crossing
    overview.png                           the whole frame at the crossing
    comparison.mp4                         the clip around the crossing, with --video
An existing case folder is only replaced with --replace.
"""
import argparse
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config import FPS, RAW_ROOT, REPORTS_DIR, ROOT, TRACKS_DIR
from src.data.mot import load_mot
from src.data.sequences import list_frames, read_frame
from src.failure.diagnosis import FailureRow, event_of, find_case, peers_of
from src.failure.viz import CaseView, crop_around, write_case
from src.utils import positive_int, read_json


def build_view(case: FailureRow, rows: list[FailureRow], review: dict[str, Any], args: argparse.Namespace,
               split: str, line: dict[str, list[int]], regions: list[list[float]]) -> CaseView:
    """Load everything that is drawn for one case: frames, GT, the tracks of every configuration, captions."""
    sequence, center = case["sequence"], event_of(case)["frame"]
    frames = dict(list_frames(sequence, args.raw))
    gt = load_mot(Path(args.gt_mot_dir) / f"{sequence}.txt")
    configs = sorted({row["config"] for row in rows})
    tracks = {config: load_mot(Path(args.tracks_dir) / split / config / f"{sequence}.txt") for config in configs}

    # The same GT vehicle in every configuration; the reviewer's statements say how each one counted it
    reference = case["gt_event"]["target"] if case["gt_event"] else review.get("reference_gt_id")
    peers = peers_of(rows, case, review.get("reference_gt_id"))
    stated = review.get("counterpart_assessment", {})
    highlighted: dict[str, set[int]] = {}
    captions: dict[str, str] = {}
    for config in configs:
        peer = peers.get(config)
        highlighted[config] = set(peer["track_ids"]) if peer else set()
        if peer and peer["pred_event"]:
            highlighted[config].add(peer["pred_event"]["target"])
        highlighted[config] |= set(stated.get(config, {}).get("counted_track_ids", []))
        captions[config] = stated[config]["description"] if config in stated else (
            peer["outcome"] if peer else "GT vehicle not identified")

    # Close-up window: the GT vehicle over the clip, or the case's own track when there is no GT vehicle
    clip = [frame for frame in frames if abs(frame - center) <= args.radius]
    boxes = [row[1] for frame in clip for row in gt.get(frame, []) if row[0] == reference]
    if not boxes:
        boxes = [row[1] for frame in clip for row in tracks[case["config"]].get(frame, [])
                 if row[0] == event_of(case)["target"]]
    return CaseView(frames, line, regions, gt, tracks, reference, highlighted, captions,
                    crop_around(boxes, read_frame(frames[center]).shape[:2]))


def parse_args() -> argparse.Namespace:
    """Command-line options; see the module docstring for what the script reads and writes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True, help="failure_analysis_<split>.json")
    ap.add_argument("--review", default=str(REPORTS_DIR / "failure_review.json"))
    ap.add_argument("--case-id", nargs="+", default=None, help="cases to draw (default: the cases of the review)")
    ap.add_argument("--tracks-dir", default=str(TRACKS_DIR))
    ap.add_argument("--gt-mot-dir", default=str(REPORTS_DIR / "gt_mot"))
    ap.add_argument("--lines", default=str(REPORTS_DIR / "counting_lines.json"))
    ap.add_argument("--ignored", default=str(REPORTS_DIR / "ignored_regions.json"))
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/failures"))
    ap.add_argument("--radius", type=positive_int, default=25, help="frames drawn before and after the crossing")
    ap.add_argument("--frame-offsets", nargs="+", type=int, default=[-8, 0, 8], help="frames of comparison.png")
    ap.add_argument("--fps", type=positive_int, default=FPS)
    ap.add_argument("--video", action="store_true", help="also write comparison.mp4")
    ap.add_argument("--replace", action="store_true", help="replace the pictures of cases that already have some")
    return ap.parse_args()


def main() -> None:
    """Draw every selected case into its own folder."""
    args = parse_args()
    analysis = read_json(args.analysis)
    split, rows = analysis["split"], analysis["rows"]
    review_path = Path(args.review)
    if not review_path.is_file() and not args.case_id:
        raise SystemExit("No manual review found. Use --case-id from the analysis to inspect candidates first.")
    reviews = {case["case_id"]: case for case in read_json(review_path)["cases"]} if review_path.is_file() else {}
    lines, ignored = read_json(args.lines), read_json(args.ignored)

    for case_id in tqdm(args.case_id or list(reviews), desc="cases", unit="case"):
        case = find_case(rows, case_id)
        out_dir = Path(args.out_dir) / split / case_id
        if out_dir.exists():
            if not args.replace:
                raise SystemExit(f"Pictures already exist: {out_dir} (use --replace)")
            shutil.rmtree(out_dir)

        view = build_view(case, rows, reviews.get(case_id, {}), args, split, lines[case["sequence"]],
                          ignored[case["sequence"]])
        write_case(view, event_of(case)["frame"], out_dir, args.radius, args.frame_offsets, args.fps, args.video)
        tqdm.write(f"[done] {case_id} -> {out_dir}")


if __name__ == "__main__":
    main()
