"""Report of the reviewed failure cases, after checking the review against the saved events.

    python scripts/report_failures.py --analysis reports/failure_analysis_dev.json --picture-dir outputs/failures/dev

Each reviewed case must name an existing case of the analysis, fill in the required text fields and
state how every configuration counted the vehicle; those statements are compared with the saved
crossing events, and the cases must cover at least three different vehicles. Nothing is written
when a check fails.

Needs:
    <analysis>                             from analyze_failures
    reports/failure_review.json            reviewed cases (--review)
    reports/track_counts.json              from count_tracks, the same file the analysis used
    <picture-dir>/<case id>/comparison.png from render_failure_cases (linked, not read)
Produces:
    <out>                                  Markdown report (default reports/failure_cases.md)
"""
import argparse
from pathlib import Path

from src.config import REPORTS_DIR, ROOT
from src.failure.review import check_review, markdown_report
from src.utils import read_json


def parse_args() -> argparse.Namespace:
    """Command-line options; see the module docstring for what the script reads and writes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True, help="failure_analysis_<split>.json")
    ap.add_argument("--review", default=str(REPORTS_DIR / "failure_review.json"))
    ap.add_argument("--tracks", default=str(REPORTS_DIR / "track_counts.json"))
    ap.add_argument("--picture-dir", default=str(ROOT / "outputs/failures/dev"))
    ap.add_argument("--out", default=str(REPORTS_DIR / "failure_cases.md"))
    return ap.parse_args()


def main() -> None:
    """Check the review, then write the Markdown report."""
    args = parse_args()
    analysis, review = read_json(args.analysis), read_json(args.review)
    predictions = read_json(args.tracks)[analysis["split"]]
    try:
        check_review(review["cases"], analysis["rows"], predictions)
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"Review rejected: {exc}") from exc

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown_report(review["cases"], analysis["rows"], Path(args.picture_dir), out_path),
                        encoding="utf-8")
    print(f"[done] {len(review['cases'])} reviewed cases -> {out_path}")


if __name__ == "__main__":
    main()
