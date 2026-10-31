"""Draw the counting line on sequence frames and save it to reports/counting_lines.json.

Interactive mode (default): a window opens on a frame of the sequence - click two points
to place the line, scrub through the sequence to check it, then S saves and moves on.
Headless mode: pass --p1 x,y --p2 x,y together with --sequence to write a line without
a window (used for bulk placement).

Usage:
    python -m src.scripts.annotate_line --sequence MVI_20033
    python -m src.scripts.annotate_line --split dev            # sequences of one split
    python -m src.scripts.annotate_line                        # dev + test, one after another
    python -m src.scripts.annotate_line --sequence MVI_20033 --p1 60,300 --p2 900,300
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..data.config import RAW_ROOT, REPORTS_DIR
from ..data.frames import list_frames as _list_frames, read_frame
from ..data.provenance import write_json
from ..eval.line_crossing import validate_line

LINE_COLOR, TEXT_COLOR = (0, 255, 255), (255, 255, 255)
KEYS = {"a": -25, "d": 25, "w": -250, "x": 250}          # Define frame navigation steps
ARROWS = {2424832: -25, 2555904: 25, 2490368: -250, 2621440: 250}  # Map arrow keys to frame navigation steps


def load_splits(path=None):
    """Reference split (committed copy in reports/)."""
    path = Path(path or REPORTS_DIR / "splits.json")
    return json.loads(path.read_text())


def load_lines(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def save_line(lines, path, seq, p1, p2):
    validate_line(p1, p2)
    value = {"p1": list(map(int, p1)), "p2": list(map(int, p2))}
    validate_line(value["p1"], value["p2"])
    updated = {**lines, seq: value}
    write_json(path, updated)
    lines.update(updated)


def _parse_xy(s):
    x, y = s.replace("(", "").replace(")", "").split(",")
    return float(x), float(y)


def draw_line(img, p1, p2=None, cursor=None):
    out = img.copy()
    if p1 is not None:
        end = p2 if p2 is not None else cursor
        if end is not None:
            cv2.line(out, (int(p1[0]), int(p1[1])), (int(end[0]), int(end[1])), LINE_COLOR, 2, cv2.LINE_AA)
        for p in (p1, end):
            if p is not None:
                cv2.circle(out, (int(p[0]), int(p[1])), 5, LINE_COLOR, -1, cv2.LINE_AA)
                cv2.circle(out, (int(p[0]), int(p[1])), 6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _hud(img, seq, idx, n, pts, msg=""):
    bar = f"{seq}  frame {idx + 1}/{n}  points: {len(pts)}"
    hint = "[click] 2 points | R reset | A/D +-25  W/X +-250 | S save | N skip | Q quit"
    cv2.rectangle(img, (0, 0), (img.shape[1], 52), (0, 0, 0), -1)
    cv2.putText(img, bar, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, hint, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    if msg:
        cv2.putText(img, msg, (8, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 80), 2, cv2.LINE_AA)
    return img


def write_preview(frames, seq, line, path, frame_idx=None):
    idx = len(frames) // 2 if frame_idx is None else max(0, min(len(frames) - 1, frame_idx))
    fid, fpath = frames[idx]
    image = read_frame(fpath)
    validate_line(line["p1"], line["p2"], (image.shape[1], image.shape[0]))
    img = draw_line(image, tuple(line["p1"]), tuple(line["p2"]))
    cv2.putText(img, f"counting line - {seq} (frame {fid})", (8, img.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"Cannot write preview image: {path}")


def interactive(seq, frames, start_pts, lines_path, preview_dir, lines):
    pts = [tuple(start_pts["p1"]), tuple(start_pts["p2"])] if start_pts else []
    idx = len(frames) // 2
    msg = "existing line loaded - adjust or press R" if start_pts else ""
    win = f"counting line - {seq}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        nonlocal pts, msg
        if event == cv2.EVENT_LBUTTONDOWN:
            pts = [ (x, y) ] if len(pts) >= 2 else pts + [(x, y)]
            msg = ""

    cv2.setMouseCallback(win, on_mouse)
    while True:
        _, fpath = frames[idx]
        img = read_frame(fpath)
        p1, p2 = (pts + [None, None])[:2]

        preview = draw_line(img, p1, p2)
        preview = _hud(preview, seq, idx, len(frames), pts, msg)
        cv2.imshow(win, preview)

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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sequence", nargs="+", default=None, help="sequence names (default: from --split or both splits)")
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--p1", default=None, help="headless mode: 'x,y' of the line start")
    ap.add_argument("--p2", default=None, help="headless mode: 'x,y' of the line end")
    ap.add_argument("--frame", type=int, default=None, help="headless preview frame (default: middle)")
    ap.add_argument("--lines", default=str(REPORTS_DIR / "counting_lines.json"))
    ap.add_argument("--preview-dir", default=str(REPORTS_DIR / "line_previews"))
    ap.add_argument("--raw", default=str(RAW_ROOT))
    args = ap.parse_args()

    splits = load_splits()
    seqs = (args.sequence or
            (splits[args.split] if args.split else splits["dev"] + splits["test"]))
    lines = load_lines(args.lines)
    preview_dir = Path(args.preview_dir)

    if args.p1 or args.p2:
        if not (args.p1 and args.p2 and args.sequence):
            ap.error("--p1 and --p2 require --sequence (headless mode places one line per call)")
        for seq in args.sequence:
            frames = _list_frames(seq, args.raw)
            image = read_frame(frames[0][1])
            p1, p2 = _parse_xy(args.p1), _parse_xy(args.p2)
            validate_line(p1, p2, (image.shape[1], image.shape[0]))
            save_line(lines, args.lines, seq, p1, p2)
            write_preview(frames, seq, lines[seq], preview_dir / f"{seq}.png", args.frame)
            print(f"[saved] {seq}: p1={lines[seq]['p1']} p2={lines[seq]['p2']} -> {args.lines}")
        return

    print(f"[annotate] {len(seqs)} sequence(s): {', '.join(seqs)}")
    try:
        for seq in seqs:
            frames = _list_frames(seq, args.raw)
            if not frames:
                print(f"[skip] {seq}: no frames")
                continue
            print(f"[open] {seq}: {len(frames)} frames "
                  f"(existing line: {'yes' if seq in lines else 'no'})")
            interactive(seq, frames, lines.get(seq), args.lines, preview_dir, lines)
            print(f"[state] {len(lines)} line(s) saved in {args.lines}")
    except KeyboardInterrupt:
        print("\n[cancelled] lines saved so far are kept on disk")
    print(f"[done] {args.lines}")


if __name__ == "__main__":
    main()
