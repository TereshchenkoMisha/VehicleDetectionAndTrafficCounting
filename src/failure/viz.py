"""Draw a failure case: the same vehicle and frames for every configuration, side by side.

Green is the GT box, orange the tracks that matter for the case, grey the other tracks and cyan the
counting line; ignored regions are greyed out. All images are BGR uint8 arrays.
"""
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..config import CLASSES, FPS
from ..data.mot import MotRow
from ..data.regions import Region
from ..data.sequences import read_frame
from ..tracking.viz import LINE_COLOR, TEXT_COLOR, Color, Image, video_writer

PANEL_SIZE: tuple[int, int] = (640, 360)        # width, height of the picture part of a panel
HEADER: int = 56                                # room above the picture for the configuration and its caption
CROP_MARGIN: int = 50                           # pixels kept around the vehicle in a close-up
VIDEO_COLUMNS: int = 2

GT_COLOR: Color = (0, 230, 0)
TRACK_COLOR: Color = (0, 150, 255)
OTHER_COLOR: Color = (160, 160, 160)
REGION_COLOR: Color = (60, 60, 60)
FONT: int = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class CaseView:
    """What is drawn for one case: the frames, the saved GT and tracks, and what to emphasise."""
    frames: dict[int, Path]                         # frame number -> image
    line: dict[str, list[int]]
    regions: list[Region]
    gt: dict[int, list[MotRow]]
    tracks: dict[str, dict[int, list[MotRow]]]      # configuration -> its MOT rows per frame
    reference_gt: int | None                        # GT vehicle drawn in green, None when unknown
    highlighted: dict[str, set[int]]                # configuration -> track ids drawn in orange
    captions: dict[str, str]                        # configuration -> what happened to the vehicle there
    crop: tuple[int, int, int, int]                 # x1, y1, x2, y2 of the close-up


def crop_around(boxes: Sequence[Sequence[float]], shape: tuple[int, int], margin: int = CROP_MARGIN
                ) -> tuple[int, int, int, int]:
    """Window x1, y1, x2, y2 around all `boxes` (x1, y1, x2, y2) plus a margin, clipped to the image.

    `shape` is (height, width). Without boxes the window is the whole image.
    """
    height, width = shape
    if not len(boxes):
        return 0, 0, width, height
    array = np.array(boxes, dtype=float)
    x1, y1 = np.maximum(np.floor(array[:, :2].min(axis=0) - margin), 0).astype(int)
    x2, y2 = np.minimum(np.ceil(array[:, 2:].max(axis=0) + margin), (width, height)).astype(int)
    return int(x1), int(y1), int(x2), int(y2)


def put_label(image: Image, text: str, origin: tuple[int, int], color: Color) -> None:
    """Draw text with a dark outline, so it stays readable on any background."""
    cv2.putText(image, text, origin, FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, 0.45, color, 1, cv2.LINE_AA)


def fit_panel(image: Image, title: str, caption: str) -> Image:
    """Scale `image` into the panel (keeping its proportions) and put two lines of text above it."""
    scale = min(PANEL_SIZE[0] / image.shape[1], PANEL_SIZE[1] / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))

    panel = np.zeros((HEADER + PANEL_SIZE[1], PANEL_SIZE[0], 3), np.uint8)
    top, left = HEADER + (PANEL_SIZE[1] - resized.shape[0]) // 2, (PANEL_SIZE[0] - resized.shape[1]) // 2
    panel[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    for text, y in ((title, 20), (caption, 43)):
        cv2.putText(panel, text, (8, y), FONT, 0.48, TEXT_COLOR, 1, cv2.LINE_AA)
    return panel


def draw_panel(view: CaseView, config: str, frame: int, full: bool = False) -> Image:
    """One panel: `frame` with the GT vehicle and the tracks of `config`; a close-up unless `full`."""
    image = read_frame(view.frames[frame])

    # Grey out the ignored regions and draw the counting line
    shade = image.copy()
    for x1, y1, x2, y2 in view.regions:
        cv2.rectangle(shade, (int(x1), int(y1)), (int(x2), int(y2)), REGION_COLOR, -1)
    image = cv2.addWeighted(image, 0.8, shade, 0.2, 0)
    cv2.line(image, tuple(view.line["p1"]), tuple(view.line["p2"]), LINE_COLOR, 2, cv2.LINE_AA)

    # Tracks of this configuration: the ones that matter are labelled with their id and class
    labelled = 0
    for track, box, cls, _ in view.tracks[config].get(frame, []):
        important = track in view.highlighted[config]
        a, b = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, a, b, TRACK_COLOR if important else OTHER_COLOR, 2 if important else 1)
        if important:
            put_label(image, f"P{track} {CLASSES[cls]}", (a[0], max(14, a[1] - 5 - 15 * (labelled % 3))), TRACK_COLOR)
            labelled += 1

    # The GT vehicle
    for target, box, cls, _ in view.gt.get(frame, []):
        if target == view.reference_gt:
            a, b = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(image, a, b, GT_COLOR, 2)
            put_label(image, f"GT{target} {CLASSES[cls]}", (a[0], min(image.shape[0] - 8, b[1] + 16)), GT_COLOR)

    if not full:
        x1, y1, x2, y2 = view.crop
        image = image[y1:y2, x1:x2]
    return fit_panel(image, f"{config} | frame {frame}", view.captions[config])


def tile(panels: Sequence[Image], columns: int) -> Image:
    """Arrange equal-size panels in rows of `columns`; the last row is filled up with black."""
    blank = np.zeros_like(panels[0])
    rows = [list(panels[i:i + columns]) for i in range(0, len(panels), columns)]
    return np.vstack([np.hstack(row + [blank] * (columns - len(row))) for row in rows])


def save_image(path: Path, image: Image) -> None:
    """Write a PNG. Encoding in memory first keeps non-ASCII paths working on Windows."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot write image: {path}")
    path.write_bytes(encoded.tobytes())


def write_case(view: CaseView, center: int, out_dir: Path, radius: int = 25, offsets: Sequence[int] = (-8, 0, 8),
               fps: int = FPS, video: bool = False) -> None:
    """Write the pictures of one case into `out_dir` (created if missing).

      comparison.png  close-ups at the frame offsets around the crossing, one column per configuration;
      overview.png    the whole frame at the crossing, one panel per configuration;
      comparison.mp4  with `video`: every frame within `radius` frames of the crossing, panels in two columns.
    """
    configs = list(view.tracks)
    clip = [frame for frame in sorted(view.frames) if abs(frame - center) <= radius]
    if not clip:
        raise ValueError(f"No source frames within {radius} frames of frame {center}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stills. Offsets reaching past the sequence are moved to its first or last frame.
    shots = sorted({min(max(center + offset, clip[0]), clip[-1]) for offset in offsets})
    rows = [tile([draw_panel(view, config, frame) for config in configs], len(configs)) for frame in shots]
    save_image(out_dir / "comparison.png", np.vstack(rows))
    save_image(out_dir / "overview.png", tile([draw_panel(view, config, center, full=True) for config in configs],
                                              len(configs)))

    # Clip: the same panels for every frame of the window
    if video:
        size = (VIDEO_COLUMNS * PANEL_SIZE[0], ceil(len(configs) / VIDEO_COLUMNS) * (HEADER + PANEL_SIZE[1]))
        with video_writer(out_dir / "comparison.mp4", size, fps) as writer:
            for frame in tqdm(clip, desc=f"video {out_dir.name}", unit="frame", leave=False):
                writer.write(tile([draw_panel(view, config, frame) for config in configs], VIDEO_COLUMNS))
