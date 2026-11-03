"""Read and write MOT files, the text format used for both tracker output and ground truth.

One row per object per frame: `frame,id,x,y,w,h,score,class,-1,-1`. Frame numbers start at 1,
(x, y) is the top-left corner, the class is the project class id, and the last two columns are
unused placeholders required by the MOTChallenge layout.
"""
import math
from collections.abc import Iterable
from pathlib import Path

from ..config import CLASSES

# One row as returned by load_mot: (track id, [x1, y1, x2, y2], class id, score)
MotRow = tuple[int, list[float], int, float]
# One row as produced by the tracker: (frame, id, x, y, w, h, score, class id)
MotOutRow = tuple[int, int, float, float, float, float, float, int]


def load_mot(path: str | Path) -> dict[int, list[MotRow]]:
    """Read a MOT file into {frame: [rows]} and validate it.

    Boxes are converted from x, y, w, h to x1, y1, x2, y2. Every row is checked: numbers must be
    finite, the box must have a positive size, the score must be in [0, 1], the class must exist,
    and a track id may appear only once per frame. The first bad row raises
    ValueError("<path>:<line>: <reason>"), so a corrupt file never reaches the metrics.
    """
    frames: dict[int, list[MotRow]] = {}
    seen: set[tuple[int, int]] = set()          # (frame, track id) pairs, to catch duplicates

    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                # Parse the columns that matter, the two trailing placeholders are ignored
                fields = line.strip().split(",")
                fid, tid = int(fields[0]), int(fields[1])
                x, y, w, h, conf = map(float, fields[2:7])
                cls = int(fields[7])

                # Reject rows that would corrupt the metrics: bad ids, NaN, empty boxes, duplicates
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
                # IndexError means too few columns; both are reported with the file position
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return frames


def write_mot(rows: Iterable[MotOutRow], path: str | Path) -> None:
    """Write tracker rows to a MOT file, creating the folder if needed.

    Coordinates get two decimals, the score four. Rows are written in the order given.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for fid, tid, x, y, w, h, s, c in rows:
            f.write(f"{fid},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.4f},{c},-1,-1\n")
