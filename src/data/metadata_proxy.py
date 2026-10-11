import cv2
import numpy as np
from pathlib import Path

from tqdm import tqdm

from .config import RAW_ROOT
from .detrac_parser import SequenceMeta, find_sequence_dir

NIGHT_THRESHOLD = 90.0


def mean_brightness(seq_dir: Path) -> float | None:
    files = sorted(seq_dir.glob("*.jpg"))
    if not files:
        return None
    picks = [files[len(files) // 3], files[len(files) // 2], files[2 * len(files) // 3]]
    vals = []
    for p in picks:
        im = cv2.imread(str(p))
        if im is not None:
            vals.append(float(im.mean()))
    return float(np.mean(vals)) if vals else None


def build_proxy_metas(anns: dict, frames_root=RAW_ROOT) -> dict:
    metas, report = {}, []
    for name in tqdm(sorted(anns), desc="brightness proxy", unit="seq"):
        d = find_sequence_dir(frames_root, name)
        b = mean_brightness(d) if d else None
        if b is None:
            weather = "unknown"
        else:
            weather = "night" if b < NIGHT_THRESHOLD else "day"
        metas[name] = SequenceMeta(name, weather=weather, scene="unknown")
        report.append((name, b, weather))

    nights = [r for r in report if r[2] == "night"]
    days = [r for r in report if r[2] == "day"]
    print(f"[meta-proxy] night: {len(nights)}, day: {len(days)}, "
          f"unknown: {len(report) - len(nights) - len(days)}")
    print("[meta-proxy] night sequences:")
    for name, b, w in sorted(report, key=lambda r: r[1] if r[1] is not None else 999):
        if w == "night":
            print(f"  {name}: brightness {b:.1f}")
    return metas