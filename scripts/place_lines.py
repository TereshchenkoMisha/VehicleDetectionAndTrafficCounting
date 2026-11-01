"""Propose a counting line per sequence from the GT traffic flow. Check the previews and fix lines by hand.

    python scripts/place_lines.py --splits dev --out reports/proposed_counting_lines.json

For each sequence the script finds the main direction of the moving GT vehicles, tries three lines
across it and keeps the one that the largest share of those vehicles crosses. The result is a proposal:
look at the previews and correct bad lines with scripts/annotate_line.py before using them.

Needs:
    data/raw/UA-DETRAC/                    frames and XML annotations
    reports/splits.json
Produces:
    <out>                                  {sequence: {"p1": [x, y], "p2": [x, y]}}
    <preview-dir>/<sequence>.png           middle frame with the line
It uses GT trajectories: lines proposed for test sequences use test GT, so say so when reporting results.
Refuses to overwrite <out>.
"""
import argparse
from pathlib import Path

from src.config import RAW_ROOT, REPORTS_DIR
from src.counting.lines import propose_line
from src.data.detrac import load_all
from src.data.sequences import list_frames, read_frame
from src.tracking.viz import write_preview
from src.utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    """Command-line options; see the module docstring for what the script reads and writes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev"])
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--out", default=str(REPORTS_DIR / "proposed_counting_lines.json"))
    ap.add_argument("--preview-dir", default=str(REPORTS_DIR / "proposed_line_previews"))
    return ap.parse_args()


def main() -> None:
    """Propose a line for each sequence of the chosen splits, save previews and the line file."""
    args = parse_args()
    if Path(args.out).exists():
        raise SystemExit("Output lines already exist, choose a new --out")

    splits = read_json(REPORTS_DIR / "splits.json")
    anns, _ = load_all(args.raw)

    lines: dict[str, dict[str, list[int]]] = {}
    for split in args.splits:
        for seq in splits[split]:
            # The first frame gives the image size, which the line has to fit into
            frames = list_frames(seq, args.raw)
            image = read_frame(frames[0][1])
            line, info = propose_line(anns[seq], (image.shape[1], image.shape[0]))
            lines[seq] = line
            write_preview(frames, seq, line, Path(args.preview_dir) / f"{seq}.png")

            # The diagnostics show how trustworthy the proposal is: coverage is the share of the main
            # stream that crosses the line, ign the share of the line inside ignored regions
            print(f"{split} {seq}: theta={info['theta_deg']:5.1f}deg stream={info['stream']:3d}/{info['moving']:3d} "
                  f"cross={info['events']:3d} (ab={info['ab']:3d} ba={info['events'] - info['ab']:3d}) "
                  f"cov={info['coverage']:4.0%} ign={info['ignored_overlap']:4.0%} shift={info['shift']:+.2f} "
                  f"p1={line['p1']} p2={line['p2']}")
    write_json(args.out, lines)


if __name__ == "__main__":
    main()
