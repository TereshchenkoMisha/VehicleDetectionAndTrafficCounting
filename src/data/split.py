"""Day/night proxy from image brightness and the stratified dev/test split of 12 sequences."""
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..config import N_DEV, N_SELECT, RAW_ROOT, SEED
from ..utils import read_json, write_json
from .detrac import SequenceMeta
from .sequences import find_sequence_dir

NIGHT_THRESHOLD = 90.0       # mean pixel value (0-255) below which a sequence counts as night


def mean_brightness(seq_dir: Path) -> float | None:
    """Mean pixel value of three frames (at 1/3, 1/2 and 2/3 of the sequence)."""
    files = sorted(seq_dir.glob("*.jpg"))
    if not files:
        return None
    picks = [files[len(files) // 3], files[len(files) // 2], files[2 * len(files) // 3]]
    values = [float(im.mean()) for im in (cv2.imread(str(p)) for p in picks) if im is not None]
    return float(np.mean(values)) if values else None


def build_proxy_metas(anns: dict, frames_root=RAW_ROOT) -> dict:
    """SequenceMeta per sequence with weather = day / night / unknown (the mirror has no weather labels)."""
    metas, night = {}, []
    for name in tqdm(sorted(anns), desc="brightness proxy", unit="seq"):
        seq_dir = find_sequence_dir(frames_root, name)
        brightness = mean_brightness(seq_dir) if seq_dir else None
        if brightness is None:
            weather = "unknown"
        else:
            weather = "night" if brightness < NIGHT_THRESHOLD else "day"
        metas[name] = SequenceMeta(name, weather=weather, scene="unknown")
        if weather == "night":
            night.append((name, brightness))

    print(f"[meta-proxy] night: {len(night)}, day: {sum(m.weather == 'day' for m in metas.values())}, "
          f"unknown: {sum(m.weather == 'unknown' for m in metas.values())}")
    for name, brightness in sorted(night, key=lambda item: item[1]):
        print(f"  {name}: brightness {brightness:.1f}")
    return metas


def _quotas(groups: dict, k: int) -> dict:
    """Split k picks across strata in proportion to their size (largest-remainder method)."""
    total = sum(len(v) for v in groups.values())
    exact = {s: len(v) * k / total for s, v in groups.items()}
    quota = {s: int(x) for s, x in exact.items()}
    by_remainder = sorted(sorted(exact), key=lambda s: exact[s] - quota[s], reverse=True)
    for s in by_remainder[: k - sum(quota.values())]:
        quota[s] += 1
    return quota


def _stratified_pick(groups: dict, k: int, seed: int) -> list:
    rng = random.Random(seed)
    quota = _quotas(groups, k)
    picked = []
    for s in sorted(groups):
        picked += rng.sample(groups[s], quota[s])
    return picked


def select_and_split(metas: dict, n_select=N_SELECT, n_dev=N_DEV, seed=SEED) -> dict:
    """Pick n_select sequences stratified by weather, then split them into dev (n_dev) and test (the rest)."""
    groups = {}
    for name, meta in metas.items():
        groups.setdefault(meta.weather, []).append(name)
    for names in groups.values():
        names.sort()

    selected = _stratified_pick(groups, n_select, seed)
    selected_groups = {}
    for name in selected:
        selected_groups.setdefault(metas[name].weather, []).append(name)
    dev = _stratified_pick(selected_groups, n_dev, seed + 1)
    test = [name for name in selected if name not in dev]
    return {"dev": sorted(dev), "test": sorted(test),
            "weather": {name: metas[name].weather for name in selected}}


def save_or_load_split(out_dir, split=None, force=False) -> dict:
    """Return the existing splits.json unless force is set; otherwise save `split`."""
    path = Path(out_dir) / "splits.json"
    if path.exists() and not force:
        print(f"[split] {path} already exists, reusing it")
        return read_json(path)
    write_json(path, split)
    return split
