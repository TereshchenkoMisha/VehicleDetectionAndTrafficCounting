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

from ..config import CLASSES, DETRAC_TO_ID, FPS, RAW_ROOT
from ..counting.line_crossing import CrossingEvent, count_events, tally, validate_line
from ..counting.trajectories import build_trajectories, mot_rows
from ..data.detrac import SequenceAnn
from ..data.mot import MotRow, load_mot
from ..data.regions import Region
from ..data.sequences import list_frames, read_frame, resolve_naming

Image = NDArray[np.uint8]
Color = tuple[int, int, int]
Point = tuple[float, float]

LINE_COLOR: Color = (0, 255, 255)
TEXT_COLOR: Color = (255, 255, 255)
CLASS_COLORS: dict[int, Color] = {0: (80, 200, 80), 1: (40, 140, 255), 2: (80, 80, 255)}  # car, bus, truck
TRAIL_FRAMES: int = 12  # how long a crossing marker stays visible, in frames
PANEL_SIZE: tuple[int, int] = (460, 52)  # width, height in pixels of the info box in tracking videos


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


def draw_tracks(img: Image, rows: Sequence[MotRow], regions: list[Region]) -> Image:
    """Return a copy of `img` showing the tracker output of one frame.

    Ignored regions are darkened. Each track gets a box in the colour of its class and a "class #id" label.
    """
    # Darken the ignored regions: paint them black on a copy, then blend that copy over the
    # original at 35% so the area stays visible but clearly marked
    shade = img.copy()
    for x1, y1, x2, y2 in regions:
        cv2.rectangle(shade, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), -1)
    out = cv2.addWeighted(shade, 0.35, img, 0.65, 0)

    for tid, (x1, y1, x2, y2), cid, _ in rows:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), CLASS_COLORS[cid], 2)
        cv2.putText(out, f"{CLASSES[cid]} #{tid}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[cid], 1, cv2.LINE_AA)
    return out


def crossing_points(events: Sequence[CrossingEvent]) -> dict[int, list[list[float]]]:
    """Crossing points grouped by the frame of the crossing, for quick lookup while drawing."""
    points: dict[int, list[list[float]]] = {}
    for event in events:
        points.setdefault(event["frame"], []).append(event["point"])
    return points


def draw_crossing_marks(img: Image, crossings: dict[int, list[list[float]]], fid: int) -> None:
    """Circle the crossings of frame `fid` and the previous TRAIL_FRAMES frames on `img`, in place.

    The circle is yellow in the crossing frame and grey afterwards.
    """
    for back in range(TRAIL_FRAMES + 1):
        for x, y in crossings.get(fid - back, []):
            cv2.circle(img, (int(x), int(y)), 18, LINE_COLOR if back == 0 else (120, 120, 120), 2, cv2.LINE_AA)


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
    ok, encoded = cv2.imencode(Path(path).suffix, img)
    if not ok:
        raise RuntimeError(f"Cannot write preview image: {path}")
    Path(path).write_bytes(encoded.tobytes())


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

    crossings = crossing_points(events)
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
            draw_crossing_marks(image, crossings, fid)

            running_total += len(crossings.get(fid, []))
            cv2.rectangle(image, (0, 0), (400, 30), (0, 0, 0), -1)
            cv2.putText(image, f"{seq} f{fid} crossed: {running_total}", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
            writer.write(image)


def write_tracking_video(frames: list[tuple[int, Path]], mot_path: str | Path, regions: list[Region],
                         line: dict[str, list[int]] | None, label: str, out_path: str | Path) -> None:
    """Write the tracking video of one sequence from its saved MOT file.

    Each frame shows the tracked boxes, the counting line, the crossings, the running count per class
    and `label` (detector + tracker). Crossings are found in the same file by the same rules as
    `count_tracks`, so the counts at the end of the video are the evaluated counts. Without a `line`
    the video shows boxes and label only.
    """
    tracks = load_mot(mot_path)
    height, width = read_frame(frames[0][1]).shape[:2]

    events: list[CrossingEvent] = []
    if line is not None:
        validate_line(line["p1"], line["p2"], (width, height))
        events = count_events(build_trajectories(mot_rows(tracks), regions), line["p1"], line["p2"])
    crossings = crossing_points(events)

    with video_writer(out_path, (width, height)) as writer:
        for fid, path in tqdm(frames, desc=f"video {Path(out_path).stem}", unit="frame", leave=False):
            image = draw_tracks(read_frame(path), tracks.get(fid, []), regions)

            # Line, crossing marks and the count of all crossings up to this frame
            if line is None:
                status = "no counting line"
            else:
                image = draw_line(image, line["p1"], line["p2"])
                draw_crossing_marks(image, crossings, fid)
                counts = tally(event for event in events if event["frame"] <= fid)
                status = "  ".join(f"{name} {counts[name]['total']}" for name in (*CLASSES, "all"))

            cv2.rectangle(image, (0, 0), PANEL_SIZE, (0, 0, 0), -1)
            cv2.putText(image, f"{label}  frame {fid}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1,
                        cv2.LINE_AA)
            cv2.putText(image, status, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
            writer.write(image)


def write_label_previews(data: str | Path, out: str | Path, limit: int = 6, seed: int = 42) -> None:
    """Draw prepared YOLO labels on a seeded sample of dev/test images."""
    import random

    data, out = Path(data), Path(out)
    images = sorted(p for split in ("dev", "test") for p in (data / "images" / split).glob("*.jpg"))
    if not images:
        raise ValueError(f"No prepared images in {data}")
    out.mkdir(parents=True, exist_ok=True)
    for path in random.Random(seed).sample(images, min(limit, len(images))):
        image = read_frame(path)
        h, w = image.shape[:2]
        labels = data / "labels" / path.parent.name / f"{path.stem}.txt"
        for line in labels.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls, cx, cy, bw, bh = map(float, line.split())
            a = (int((cx - bw / 2) * w), int((cy - bh / 2) * h))
            b = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
            cv2.rectangle(image, a, b, (0, 255, 0), 2)
            cv2.putText(image, ("car", "bus", "truck")[int(cls)], a, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError(f"Cannot encode preview: {path}")
        (out / f"{path.stem}_gt.jpg").write_bytes(encoded.tobytes())
