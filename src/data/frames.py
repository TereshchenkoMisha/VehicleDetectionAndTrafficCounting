"""Index source images and align XML frame numbers"""
import re
from pathlib import Path

from .config import RAW_ROOT
from .detrac_parser import find_sequence_dir


def frame_paths(seq_dir):
    frames = {}
    for path in Path(seq_dir).iterdir():
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
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
    image_ids = set(frame_paths(seq_dir))
    frame_ids = list(frame_ids)
    if not frame_ids:
        raise ValueError(f"No XML frames for {seq_dir}")
    offsets = [offset for offset in (0, 1)
               if all(fid + offset in image_ids for fid in frame_ids)]
    if len(offsets) != 1:
        raise ValueError(f"Cannot determine XML/JPG offset for {seq_dir}: candidates={offsets}")
    return offsets[0]


def list_frames(seq_name, raw_root=RAW_ROOT):
    seq_dir = find_sequence_dir(raw_root, seq_name)
    if seq_dir is None:
        raise FileNotFoundError(f"Frames of {seq_name} not found under {raw_root}")
    return list(frame_paths(seq_dir).items())


def read_frame(path):
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image
