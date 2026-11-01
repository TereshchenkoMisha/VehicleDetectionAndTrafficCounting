"""Set the counting line of a sequence by hand, in a window or from the command line.

    python scripts/annotate_line.py --sequence MVI_20033
    python scripts/annotate_line.py --split dev
    python scripts/annotate_line.py --sequence MVI_20033 --p1 60,300 --p2 900,300

Window mode shows a frame of the sequence. You click two points to place the line, move through the
sequence to check the line against the traffic, and save. With --p1 and --p2 no window opens and
the same line is stored for every sequence named by --sequence.

Needs:
    data/raw/UA-DETRAC/                    frames
    reports/splits.json                    sequences of --split (default: dev and test)
    reports/counting_lines.json            existing lines are loaded as the starting point, if present
Produces:
    reports/counting_lines.json            updated after each saved line
    reports/line_previews/<sequence>.png   frame with the line
Window keys: click two points, A/D and W/X move through frames (arrow keys too), S saves, N skips,
R resets the points, Q or Esc quits.
"""
import argparse
from pathlib import Path
from typing import Any

import cv2

from src.config import RAW_ROOT, REPORTS_DIR
from src.counting.line_crossing import validate_line
from src.data.sequences import list_frames, read_frame
from src.tracking.viz import TEXT_COLOR, Image, Point, draw_line, write_preview
from src.utils import read_json, write_json

# Frames to move per key press. ARROWS holds the key codes cv2.waitKeyEx returns for the arrow
# keys on Windows (left, right, up, down).
KEYS: dict[str, int] = {"a": -25, "d": 25, "w": -250, "x": 250}
ARROWS: dict[int, int] = {2424832: -25, 2555904: 25, 2490368: -250, 2621440: 250}


def save_line(lines: dict[str, Any], path: str | Path, seq: str, p1: Point, p2: Point) -> None:
    """Validate the line, store it in `lines` (integer pixels) and write the whole file."""
    validate_line(p1, p2)
    lines[seq] = {"p1": list(map(int, p1)), "p2": list(map(int, p2))}
    write_json(path, lines)


def parse_xy(text: str) -> Point:
    """Parse 'x,y' (parentheses allowed) into a point."""
    x, y = text.replace("(", "").replace(")", "").split(",")
    return float(x), float(y)


def draw_hud(img: Image, seq: str, idx: int, n: int, pts: list[Point], msg: str = "") -> Image:
    """Draw the status bar (sequence, frame, number of points, key help) on top of the frame.

    `msg` is an optional feedback line at the bottom, such as a validation error.
    """
    hint = "[click] 2 points | R reset | A/D +-25  W/X +-250 | S save | N skip | Q quit"
    cv2.rectangle(img, (0, 0), (img.shape[1], 52), (0, 0, 0), -1)
    cv2.putText(img, f"{seq}  frame {idx + 1}/{n}  points: {len(pts)}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, hint, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    if msg:
        cv2.putText(img, msg, (8, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 80), 2, cv2.LINE_AA)
    return img


def interactive(seq: str, frames: list[tuple[int, Path]], start_line: dict[str, list[int]] | None,
                lines_path: str | Path, preview_dir: str | Path, lines: dict[str, Any]) -> bool:
    """Edit the line of one sequence in a window.

    Starts from `start_line` if given. Returns True if the line was saved, False if the sequence was
    skipped or the window was closed. Raises KeyboardInterrupt when the user quits, so the caller can
    stop the whole session.
    """
    pts: list[Point] = [tuple(start_line["p1"]), tuple(start_line["p2"])] if start_line else []
    idx = len(frames) // 2
    msg = "existing line loaded - adjust or press R" if start_line else ""
    win = f"counting line - {seq}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        # Left click adds a point; a click when two points exist starts a new line
        nonlocal pts, msg
        if event == cv2.EVENT_LBUTTONDOWN:
            pts = [(x, y)] if len(pts) >= 2 else pts + [(x, y)]
            msg = ""

    cv2.setMouseCallback(win, on_mouse)
    while True:
        # Draw the current frame with the line and the status bar
        img = read_frame(frames[idx][1])
        p1, p2 = (pts + [None, None])[:2]
        cv2.imshow(win, draw_hud(draw_line(img, p1, p2), seq, idx, len(frames), pts, msg))

        # Read a key (30 ms poll). A closed window ends this sequence.
        k = cv2.waitKeyEx(30)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            return False
        key = chr(k & 0xFF).lower() if 32 <= (k & 0xFF) < 127 else ""

        if k in ARROWS or key in KEYS:
            idx = max(0, min(len(frames) - 1, idx + ARROWS.get(k, KEYS.get(key, 0))))
        elif key == "r":
            pts, msg = [], ""
        elif key == "s":
            # Save only a valid line that fits the image; otherwise show why and stay in the window
            if len(pts) < 2:
                msg = "need 2 points before saving"
                continue
            try:
                validate_line(pts[0], pts[1], (img.shape[1], img.shape[0]))
                save_line(lines, lines_path, seq, pts[0], pts[1])
            except ValueError as exc:
                msg = str(exc)
                continue
            write_preview(frames, seq, lines[seq], Path(preview_dir) / f"{seq}.png", idx)
            cv2.destroyWindow(win)
            return True
        elif key == "n":
            cv2.destroyWindow(win)
            return False
        elif key == "q" or k == 27:         # 27 is Esc
            cv2.destroyWindow(win)
            raise KeyboardInterrupt


def parse_args() -> argparse.Namespace:
    """Command-line options; see the module docstring for what the script reads and writes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", nargs="+", default=None, help="sequence names (default: from --split or both)")
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--p1", default=None, help="line start 'x,y' (no window)")
    ap.add_argument("--p2", default=None, help="line end 'x,y' (no window)")
    ap.add_argument("--frame", type=int, default=None, help="preview frame index with --p1/--p2 (default: middle)")
    ap.add_argument("--lines", default=str(REPORTS_DIR / "counting_lines.json"))
    ap.add_argument("--preview-dir", default=str(REPORTS_DIR / "line_previews"))
    ap.add_argument("--raw", default=str(RAW_ROOT))
    return ap.parse_args()


def main() -> None:
    """Run the command-line mode if --p1/--p2 are given, otherwise the window mode."""
    args = parse_args()
    splits = read_json(REPORTS_DIR / "splits.json")
    seqs: list[str] = args.sequence or (splits[args.split] if args.split else splits["dev"] + splits["test"])
    lines: dict[str, Any] = read_json(args.lines) if Path(args.lines).exists() else {}
    preview_dir = Path(args.preview_dir)

    # Command-line mode: the same line for every named sequence, validated against the image size
    if args.p1 or args.p2:
        if not (args.p1 and args.p2 and args.sequence):
            raise SystemExit("--p1 and --p2 require --sequence")
        for seq in args.sequence:
            frames = list_frames(seq, args.raw)
            image = read_frame(frames[0][1])
            p1, p2 = parse_xy(args.p1), parse_xy(args.p2)
            validate_line(p1, p2, (image.shape[1], image.shape[0]))
            save_line(lines, args.lines, seq, p1, p2)
            write_preview(frames, seq, lines[seq], preview_dir / f"{seq}.png", args.frame)
            print(f"[saved] {seq}: p1={lines[seq]['p1']} p2={lines[seq]['p2']} -> {args.lines}")
        return

    # Window mode: sequences one after another. Quitting keeps what was saved so far, because each
    # save already wrote the file.
    print(f"[annotate] {len(seqs)} sequence(s): {', '.join(seqs)}")
    try:
        for seq in seqs:
            frames = list_frames(seq, args.raw)
            print(f"[open] {seq}: {len(frames)} frames (existing line: {'yes' if seq in lines else 'no'})")
            interactive(seq, frames, lines.get(seq), args.lines, preview_dir, lines)
            print(f"[state] {len(lines)} line(s) saved in {args.lines}")
    except KeyboardInterrupt:
        print("\n[cancelled] lines saved so far are kept on disk")
    print(f"[done] {args.lines}")


if __name__ == "__main__":
    main()
