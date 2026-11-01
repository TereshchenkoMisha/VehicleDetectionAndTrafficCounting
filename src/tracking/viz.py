"""Drawing helpers: annotated tracking videos, GT check videos with the counting line, and line previews.

All images are BGR uint8 arrays (OpenCV's order), and all colours are BGR tuples.
"""
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ..config import DETRAC_TO_ID, FPS, RAW_ROOT
from ..counting.line_crossing import validate_line
from ..data.detrac import SequenceAnn
from ..data.regions import Region
from ..data.sequences import list_frames, read_frame, resolve_naming

Image = NDArray[np.uint8]
Color = tuple[int, int, int]
Point = tuple[float, float]

LINE_COLOR: Color = (0, 255, 255)
TEXT_COLOR: Color = (255, 255, 255)
CLASS_COLORS: dict[int, Color] = {0: (80, 200, 80), 1: (40, 140, 255), 2: (80, 80, 255)}    # car, bus, truck
# A fixed palette, indexed by track id, so a track keeps its colour in every run and every video.
PALETTE: list[Color] = [(int(b), int(g), int(r)) for b, g, r in np.random.RandomState(0).randint(60, 255, (64, 3))]
TRAIL_FRAMES: int = 12                  # how long a crossing marker stays visible in GT videos, in frames


@contextmanager
def video_writer(path: str | Path, size: tuple[int, int], fps: int = FPS) -> Iterator[Any]:
    """Open an mp4 writer for frames of `size` = (width, height) and always release it.

    Used as `with video_writer(...) as writer:`. Releasing in a `finally` finishes the file even
    if drawing fails halfway. Refuses to overwrite an existing video.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Video already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, tuple(size))
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {path}")
        yield writer
    finally:
        writer.release()


def draw_tracks(img: Image, tracks: Sequence[Sequence[float]], regions: list[Region], fid: int) -> Image:
    """Return a copy of `img` showing the tracker output.

    Ignored regions are darkened, each track gets a box and its id in a colour taken from its id,
    and the frame number is printed in the corner. A track row starts with x1, y1, x2, y2, track id.
    """
    # Darken the ignored regions: paint them black on a copy, then blend that copy over the
    # original at 35% so the area stays visible but clearly marked
    shade = img.copy()
    for x1, y1, x2, y2 in regions:
        cv2.rectangle(shade, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), -1)
    out = cv2.addWeighted(shade, 0.35, img, 0.65, 0)

    for x1, y1, x2, y2, tid, *_ in tracks:
        colour = PALETTE[int(tid) % len(PALETTE)]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
        cv2.putText(out, f"{int(tid)}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    cv2.putText(out, f"frame {fid}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)
    return out


def draw_line(img: Image, p1: Point | None, p2: Point | None = None, cursor: Point | None = None) -> Image:
    """Return a copy of `img` with the line p1-p2 and a dot on each end.

    While the second point is not placed yet (interactive editing), the line follows `cursor`.
    With no first point nothing is drawn.
    """
    out = img.copy()
    if p1 is None:
        return out

    end = p2 if p2 is not None else cursor
    if end is not None:
        cv2.line(out, (int(p1[0]), int(p1[1])), (int(end[0]), int(end[1])), LINE_COLOR, 2, cv2.LINE_AA)
    for p in (p1, end):
        if p is not None:
            cv2.circle(out, (int(p[0]), int(p[1])), 5, LINE_COLOR, -1, cv2.LINE_AA)
            cv2.circle(out, (int(p[0]), int(p[1])), 6, (0, 0, 0), 1, cv2.LINE_AA)       # dark outline: visible on any background
    return out


def write_preview(frames: list[tuple[int, Path]], seq: str, line: dict[str, list[int]], path: str | Path,
                  frame_idx: int | None = None) -> None:
    """Save one frame of the sequence with the counting line drawn on it, as an image.

    `frames` is [(frame number, path), ...]. By default the middle frame is used; `frame_idx`
    picks another one and is clamped to the valid range. The line is validated against the frame size.
    """
    idx = len(frames) // 2 if frame_idx is None else max(0, min(len(frames) - 1, frame_idx))
    fid, frame_path = frames[idx]
    image = read_frame(frame_path)
    validate_line(line["p1"], line["p2"], (image.shape[1], image.shape[0]))

    img = draw_line(image, tuple(line["p1"]), tuple(line["p2"]))
    cv2.putText(img, f"counting line - {seq} (frame {fid})", (8, img.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"Cannot write preview image: {path}")


def write_gt_video(ann: SequenceAnn, line: dict[str, list[int]], seq: str, events: list[dict[str, Any]],
                   out_path: str | Path, raw_root: str | Path = RAW_ROOT) -> None:
    """Write a check video of the ground-truth counting of one sequence.

    Each frame shows the GT boxes (coloured by class), the counting line, a circle where a
    crossing happened (yellow at the crossing frame, grey for the next TRAIL_FRAMES frames) and a
    running total of the `events`. It is meant for judging by eye whether the line catches the traffic.
    """
    frames = list_frames(seq, raw_root)
    offset = resolve_naming(frames[0][1].parent, sorted(ann.frames))
    height, width = read_frame(frames[0][1]).shape[:2]
    validate_line(line["p1"], line["p2"], (width, height))

    # Crossing points grouped by frame, for quick lookup while drawing
    crosses: dict[int, list[list[float]]] = {}
    for event in events:
        crosses.setdefault(event["frame"], []).append(event["point"])

    p1, p2 = (tuple(map(int, line[key])) for key in ("p1", "p2"))
    running_total = 0
    with video_writer(out_path, (width, height)) as writer:
        for fid, path in tqdm(frames, desc=f"video {seq}", unit="frame", leave=False):
            image = read_frame(path)
            if image.shape[:2] != (height, width):
                raise ValueError(f"Frame size changed at {path}")

            # GT boxes. A frame may have no annotation entry at all; it is still written to the video.
            frame_ann = ann.frames.get(fid - offset)
            for target in frame_ann.targets if frame_ann is not None else []:
                cid = DETRAC_TO_ID.get(target.cls)
                if cid is not None:
                    x1, y1, x2, y2 = map(int, target.box.as_xyxy())
                    cv2.rectangle(image, (x1, y1), (x2, y2), CLASS_COLORS[cid], 2)

            # The line, and markers for crossings from this frame and the previous TRAIL_FRAMES
            cv2.line(image, p1, p2, LINE_COLOR, 2, cv2.LINE_AA)
            for back in range(TRAIL_FRAMES + 1):
                for x, y in crosses.get(fid - back, []):
                    cv2.circle(image, (int(x), int(y)), 18, LINE_COLOR if back == 0 else (120, 120, 120),
                               2, cv2.LINE_AA)

            running_total += len(crosses.get(fid, []))
            cv2.rectangle(image, (0, 0), (400, 30), (0, 0, 0), -1)
            cv2.putText(image, f"{seq} f{fid} crossed: {running_total}", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
            writer.write(image)
