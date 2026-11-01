"""Draw the counting line on sequence frames and save it to reports/counting_lines.json.

Interactive mode (default): a window opens on a frame of the sequence. Click two points to place the line,
scrub through the sequence to check it, then S saves and moves on.
Headless mode: pass --p1 x,y --p2 x,y together with --sequence.

    python scripts/annotate_line.py --sequence MVI_20033
    python scripts/annotate_line.py --split dev
    python scripts/annotate_line.py --sequence MVI_20033 --p1 60,300 --p2 900,300
"""
import argparse
from pathlib import Path

import cv2

from src.config import RAW_ROOT, REPORTS_DIR
from src.counting.line_crossing import validate_line
from src.data.sequences import list_frames, read_frame
from src.tracking.viz import TEXT_COLOR, draw_line, write_preview
from src.utils import read_json, write_json

KEYS = {"a": -25, "d": 25, "w": -250, "x": 250}                       # frame navigation steps
ARROWS = {2424832: -25, 2555904: 25, 2490368: -250, 2621440: 250}     # Windows arrow-key codes


def save_line(lines, path, seq, p1, p2):
    validate_line(p1, p2)
    lines[seq] = {"p1": list(map(int, p1)), "p2": list(map(int, p2))}
    write_json(path, lines)


def parse_xy(text):
    x, y = text.replace("(", "").replace(")", "").split(",")
    return float(x), float(y)


def draw_hud(img, seq, idx, n, pts, msg=""):
    hint = "[click] 2 points | R reset | A/D +-25  W/X +-250 | S save | N skip | Q quit"
    cv2.rectangle(img, (0, 0), (img.shape[1], 52), (0, 0, 0), -1)
    cv2.putText(img, f"{seq}  frame {idx + 1}/{n}  points: {len(pts)}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, hint, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    if msg:
        cv2.putText(img, msg, (8, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 80), 2, cv2.LINE_AA)
    return img


def interactive(seq, frames, start_line, lines_path, preview_dir, lines):
    """Edit one line in a window. Returns True if saved, False if skipped; raises KeyboardInterrupt on quit."""
    pts = [tuple(start_line["p1"]), tuple(start_line["p2"])] if start_line else []
    idx = len(frames) // 2
    msg = "existing line loaded - adjust or press R" if start_line else ""
    win = f"counting line - {seq}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        nonlocal pts, msg
        if event == cv2.EVENT_LBUTTONDOWN:
            pts = [(x, y)] if len(pts) >= 2 else pts + [(x, y)]
            msg = ""

    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = read_frame(frames[idx][1])
        p1, p2 = (pts + [None, None])[:2]
        cv2.imshow(win, draw_hud(draw_line(img, p1, p2), seq, idx, len(frames), pts, msg))

        k = cv2.waitKeyEx(30)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            return False
        key = chr(k & 0xFF).lower() if 32 <= (k & 0xFF) < 127 else ""
        if k in ARROWS or key in KEYS:
            idx = max(0, min(len(frames) - 1, idx + ARROWS.get(k, KEYS.get(key, 0))))
        elif key == "r":
            pts, msg = [], ""
        elif key == "s":
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
        elif key == "q" or k == 27:
            cv2.destroyWindow(win)
            raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", nargs="+", default=None, help="sequence names (default: from --split or both)")
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--p1", default=None, help="headless mode: 'x,y' of the line start")
    ap.add_argument("--p2", default=None, help="headless mode: 'x,y' of the line end")
    ap.add_argument("--frame", type=int, default=None, help="headless preview frame (default: middle)")
    ap.add_argument("--lines", default=str(REPORTS_DIR / "counting_lines.json"))
    ap.add_argument("--preview-dir", default=str(REPORTS_DIR / "line_previews"))
    ap.add_argument("--raw", default=str(RAW_ROOT))
    args = ap.parse_args()

    splits = read_json(REPORTS_DIR / "splits.json")
    seqs = args.sequence or (splits[args.split] if args.split else splits["dev"] + splits["test"])
    lines = read_json(args.lines) if Path(args.lines).exists() else {}
    preview_dir = Path(args.preview_dir)

    if args.p1 or args.p2:
        if not (args.p1 and args.p2 and args.sequence):
            ap.error("--p1 and --p2 require --sequence")
        for seq in args.sequence:
            frames = list_frames(seq, args.raw)
            image = read_frame(frames[0][1])
            p1, p2 = parse_xy(args.p1), parse_xy(args.p2)
            validate_line(p1, p2, (image.shape[1], image.shape[0]))
            save_line(lines, args.lines, seq, p1, p2)
            write_preview(frames, seq, lines[seq], preview_dir / f"{seq}.png", args.frame)
            print(f"[saved] {seq}: p1={lines[seq]['p1']} p2={lines[seq]['p2']} -> {args.lines}")
        return

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
