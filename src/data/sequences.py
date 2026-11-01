"""Find the JPG frames of a sequence and relate XML frame numbers to image file numbers.

Frames live in one folder per sequence (`MVI_20011/img00001.jpg`, ...). The XML annotations number
frames independently, and the two numberings can be shifted by one, so the shift is detected
from the data instead of assumed.
"""
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..config import RAW_ROOT

JPG_SUFFIXES: set[str] = {".jpg", ".jpeg"}


@lru_cache(maxsize=None)
def _find_sequence_dir(frames_root: Path, name: str) -> Path | None:
    """Uncached search; `find_sequence_dir` adds the cache."""
    for candidate in frames_root.rglob(name):
        # Several paths can match the name (a folder and a file, for instance): take the one with JPGs
        if candidate.is_dir() and any(p.is_file() and p.suffix.lower() in JPG_SUFFIXES
                                      for p in candidate.iterdir()):
            return candidate
    return None


def find_sequence_dir(frames_root: str | Path, name: str) -> Path | None:
    """Folder that holds the frames of sequence `name` somewhere under `frames_root`, or None.

    The dataset mirror nests folders (`DETRAC-Images/DETRAC-Images/...`), so the folder is found
    by search, not by a fixed path. The search walks the whole image tree, which is slow, so
    results are cached per (root, name).
    """
    return _find_sequence_dir(Path(frames_root), name)


def frame_paths(seq_dir: str | Path) -> dict[int, Path]:
    """Map frame number -> image path for one sequence folder, sorted by number.

    Files must be named `imgNNNNN.jpg` (case-insensitive). Any other name or a repeated number
    raises, because a silent mismatch here would shift every annotation by a frame.
    """
    frames: dict[int, Path] = {}
    for path in Path(seq_dir).iterdir():
        if not path.is_file() or path.suffix.lower() not in JPG_SUFFIXES:
            continue
        match = re.fullmatch(r"img(\d+)", path.stem, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Unexpected frame filename: {path}")
        fid = int(match.group(1))
        if fid in frames:
            raise ValueError(f"Duplicate frame ID {fid}: {frames[fid]} and {path}")
        frames[fid] = path
    if not frames:
        raise FileNotFoundError(f"No JPG frames found in {seq_dir}")
    return dict(sorted(frames.items()))


def resolve_naming(seq_dir: str | Path, frame_ids: Iterable[int]) -> int:
    """Offset such that XML frame number + offset = JPG file number.

    The XML and the JPG files can number the same frame differently.
    This depends on the dataset copy: XML starts at 0 or 1, files start at 1, therefore 
    the offset is needed (0 or 1) to match each annotation to its image. This done by testing
    both offsets against all annotated frames: exactly one must fit. If both or neither fit, 
    we raise an error. A wrong offset crashes nothing. It just moves every box to the wrong frame.
    """
    image_ids = set(frame_paths(seq_dir))
    frame_ids = list(frame_ids)
    if not frame_ids:
        raise ValueError(f"No XML frames for {seq_dir}")

    offsets = [offset for offset in (0, 1) if all(fid + offset in image_ids for fid in frame_ids)]
    if len(offsets) != 1:
        raise ValueError(f"Cannot determine XML/JPG offset for {seq_dir}: candidates={offsets}")
    return offsets[0]


def list_frames(seq_name: str, raw_root: str | Path = RAW_ROOT) -> list[tuple[int, Path]]:
    """All frames of a sequence as [(frame number, image path), ...] in playback order."""
    seq_dir = find_sequence_dir(raw_root, seq_name)
    if seq_dir is None:
        raise FileNotFoundError(f"Frames of {seq_name} not found under {raw_root}")
    return list(frame_paths(seq_dir).items())


def read_frame(path: str | Path) -> NDArray[np.uint8]:
    """Load an image as a BGR uint8 array of shape (H, W, 3).

    BGR is OpenCV's native order and what Ultralytics expects for numpy input, so no colour
    conversion is needed anywhere in the pipeline.
    """
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image
