"""Read, validate and write the project MOT format: frame,id,x,y,w,h,score,cls,-1,-1 (frame is 1-based)."""
import math
from pathlib import Path

from ..config import CLASSES


def load_mot(path):
    """Return {frame: [(track_id, [x1, y1, x2, y2], cls, score), ...]}. Raises ValueError with file:line."""
    frames = {}
    seen = set()
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                fields = line.strip().split(",")
                fid, tid = int(fields[0]), int(fields[1])
                x, y, w, h, conf = map(float, fields[2:7])
                cls = int(fields[7])
                if fid < 0 or tid < 0 or not 0 <= cls < len(CLASSES):
                    raise ValueError("Invalid frame, track or class ID")
                if not all(math.isfinite(v) for v in (x, y, w, h, conf)):
                    raise ValueError("Nonfinite MOT value")
                if w <= 0 or h <= 0 or not 0 <= conf <= 1:
                    raise ValueError("Invalid box size or confidence")
                if (fid, tid) in seen:
                    raise ValueError("Duplicate track ID within a frame")
                seen.add((fid, tid))
                frames.setdefault(fid, []).append((tid, [x, y, x + w, y + h], cls, conf))
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return frames


def write_mot(rows, path):
    """rows: (frame, id, x, y, w, h, score, cls)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for fid, tid, x, y, w, h, s, c in rows:
            f.write(f"{fid},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.4f},{c},-1,-1\n")
