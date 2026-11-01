"""Choose which sequences are used and divide them into dev and test.

The dataset mirror has no weather labels, so a day/night label is estimated from image
brightness. Sequences are then drawn with the day/night proportions preserved (stratified), first
for the whole selection and then for dev inside it. The split is by sequence, never by frame, so
dev and test never share a scene.
"""
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from ..config import N_DEV, N_SELECT, RAW_ROOT, SEED
from ..utils import read_json, write_json
from .detrac import SequenceAnn, SequenceMeta
from .sequences import find_sequence_dir

NIGHT_THRESHOLD: float = 90.0       # mean pixel value (0-255) below which a sequence counts as night


def mean_brightness(seq_dir: Path) -> float | None:
    """Mean pixel value (0-255) of three frames of a sequence, or None if none can be read.

    Three frames, at 1/3, 1/2 and 2/3 of the sequence, are enough to tell day from night and keep
    the cost low; averaging them reduces the effect of one unusual frame.
    """
    files = sorted(seq_dir.glob("*.jpg"))
    if not files:
        return None
    picks = [files[len(files) // 3], files[len(files) // 2], files[2 * len(files) // 3]]
    values = [float(im.mean()) for im in (cv2.imread(str(p)) for p in picks) if im is not None]
    return float(np.mean(values)) if values else None


def build_proxy_metas(anns: dict[str, SequenceAnn],
                      frames_root: str | Path = RAW_ROOT) -> dict[str, SequenceMeta]:
    """Estimate the weather label of every sequence from brightness: "night", "day" or "unknown".

    Used when the dataset copy has no metadata files. "unknown" means the frames could not be
    read. Prints the count per label and the brightness of the night sequences, so the threshold
    can be checked by eye.
    """
    metas: dict[str, SequenceMeta] = {}
    night: list[tuple[str, float]] = []

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

    counts = {w: sum(m.weather == w for m in metas.values()) for w in ("night", "day", "unknown")}
    print(f"[meta-proxy] night: {counts['night']}, day: {counts['day']}, unknown: {counts['unknown']}")
    for name, brightness in sorted(night, key=lambda item: item[1]):
        print(f"  {name}: brightness {brightness:.1f}")
    return metas


def _quotas(groups: dict[str, list[str]], k: int) -> dict[str, int]:
    """How many of `k` picks come from each group, proportional to group size (largest-remainder).

    Example: groups of 30 and 10 with k=12 have exact shares 9.0 and 3.0. With 31 and 9 the shares
    are 9.3 and 2.7; the whole parts sum to 11, so the one remaining pick goes to the group with
    the larger fractional part (2.7), giving 9 and 3.
    """
    total = sum(len(v) for v in groups.values())

    # Whole part of each proportional share, then hand the leftover picks out by fractional part.
    # The inner sorted() makes ties resolve alphabetically, so the result does not depend on dict order.
    exact = {s: len(v) * k / total for s, v in groups.items()}
    quota = {s: int(x) for s, x in exact.items()}
    by_remainder = sorted(sorted(exact), key=lambda s: exact[s] - quota[s], reverse=True)
    for s in by_remainder[: k - sum(quota.values())]:
        quota[s] += 1
    return quota


def _stratified_pick(groups: dict[str, list[str]], k: int, seed: int) -> list[str]:
    """Draw `k` names at random, taking each group's quota from that group."""
    rng = random.Random(seed)
    quota = _quotas(groups, k)
    picked: list[str] = []
    for s in sorted(groups):
        picked += rng.sample(groups[s], quota[s])
    return picked


def select_and_split(metas: dict[str, SequenceMeta], n_select: int = N_SELECT, n_dev: int = N_DEV,
                     seed: int = SEED) -> dict[str, Any]:
    """Select `n_select` sequences and split them into dev (`n_dev`) and test (the rest).

    Both draws are stratified by weather label. Returns {"dev": [...], "test": [...], "weather":
    {sequence: label}} with sorted name lists. Deterministic for a given seed and input.
    """
    # Group names by weather; sorted lists make the draw independent of dict order
    groups: dict[str, list[str]] = {}
    for name, meta in metas.items():
        groups.setdefault(meta.weather, []).append(name)
    for names in groups.values():
        names.sort()

    # Draw the selection, then draw dev from it. The second draw uses seed + 1 so the two draws
    # are independent of each other.
    selected = _stratified_pick(groups, n_select, seed)
    selected_groups: dict[str, list[str]] = {}
    for name in selected:
        selected_groups.setdefault(metas[name].weather, []).append(name)
    dev = _stratified_pick(selected_groups, n_dev, seed + 1)
    test = [name for name in selected if name not in dev]

    return {"dev": sorted(dev), "test": sorted(test),
            "weather": {name: metas[name].weather for name in selected}}


def save_or_load_split(out_dir: str | Path, split: dict[str, Any] | None = None,
                       force: bool = False) -> dict[str, Any]:
    """Keep one split per output folder: load `splits.json` if it exists, otherwise save `split`.

    With `force` the existing file is overwritten. Reusing the saved split by default guards
    against redrawing the sequences after experiments have started.
    """
    path = Path(out_dir) / "splits.json"
    if path.exists() and not force:
        print(f"[split] {path} already exists, reusing it")
        return read_json(path)
    write_json(path, split)
    return split
