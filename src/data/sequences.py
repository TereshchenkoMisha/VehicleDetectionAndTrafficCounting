"""Find a sequence's JPG frames and align XML frame numbers with file numbers."""
import re
from functools import lru_cache
from pathlib import Path

from ..config import RAW_ROOT

JPG_SUFFIXES = {".jpg", ".jpeg"}


@lru_cache(maxsize=None)
def _find_sequence_dir(frames_root, name):
    for cand in Path(frames_root).rglob(name):
        if cand.is_dir() and any(p.is_file() and p.suffix.lower() in JPG_SUFFIXES for p in cand.iterdir()):
            return cand
    return None


def find_sequence_dir(frames_root, name):
    """Directory holding the JPG frames of one sequence, or None. Cached: the search walks the whole tree."""
    return _find_sequence_dir(Path(frames_root), name)


def frame_paths(seq_dir):
    """{frame_number: path} sorted by number, from files named imgNNNNN.jpg."""
    frames = {}
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


def resolve_naming(seq_dir, frame_ids):
    """XML frame number + offset = JPG frame number. The offset is 0 or 1; raises if it is not unique."""
    image_ids = set(frame_paths(seq_dir))
    frame_ids = list(frame_ids)
    if not frame_ids:
        raise ValueError(f"No XML frames for {seq_dir}")
    offsets = [offset for offset in (0, 1) if all(fid + offset in image_ids for fid in frame_ids)]
    if len(offsets) != 1:
        raise ValueError(f"Cannot determine XML/JPG offset for {seq_dir}: candidates={offsets}")
    return offsets[0]


def list_frames(seq_name, raw_root=RAW_ROOT):
    """[(frame_number, path), ...] of one sequence in order."""
    seq_dir = find_sequence_dir(raw_root, seq_name)
    if seq_dir is None:
        raise FileNotFoundError(f"Frames of {seq_name} not found under {raw_root}")
    return list(frame_paths(seq_dir).items())


def read_frame(path):
    """Read one image as a BGR HWC uint8 array."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image
