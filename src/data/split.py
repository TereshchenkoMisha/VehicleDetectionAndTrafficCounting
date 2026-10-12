"""Weather‑stratified selection of 12 sequences + dev/test 5/7 (§2.2, §2.5)."""
import json, random
from pathlib import Path
from .config import SEED, N_SELECT, N_DEV, OUT_DIR


def _stratified_pick(groups: dict, k: int, seed: int) -> list:
    rng = random.Random(seed)
    strata = sorted(groups)
    total = sum(len(v) for v in groups.values())
    quota = {s: len(groups[s]) * k / total for s in strata}
    base = {s: int(quota[s]) for s in strata}
    order = sorted(strata, key=lambda s: quota[s] - base[s], reverse=True)
    i, guard = 0, 0
    while sum(base.values()) < k and guard < 100 * k:
        s = order[i % len(order)]
        if base[s] < len(groups[s]):
            base[s] += 1
        i += 1; guard += 1
    picked = []
    for s in strata:
        picked += rng.sample(groups[s], min(base[s], len(groups[s])))
    return picked


def select_and_split(metas: dict, n_select=N_SELECT, n_dev=N_DEV, seed=SEED) -> dict:
    groups = {}
    for name, m in metas.items():
        groups.setdefault(m.weather, []).append(name)
    for s in groups:
        groups[s].sort()

    selected = _stratified_pick(groups, n_select, seed)
    sel_groups = {}
    for n in selected:
        sel_groups.setdefault(metas[n].weather, []).append(n)
    dev = _stratified_pick(sel_groups, n_dev, seed + 1)
    test = [n for n in selected if n not in dev]
    assert len(dev) == n_dev, f"dev = {len(dev)}, was waiting {n_dev}"
    return {"dev": sorted(dev), "test": sorted(test),
            "weather": {n: metas[n].weather for n in selected}}


def save_or_load_split(out_dir=OUT_DIR, split=None, force=False) -> dict:
    f = Path(out_dir) / "splits.json"
    if f.exists() and not force:
        print(f"[split] {f} It already exists — I’ll reuse it (plan §2.2: no revision).")
        return json.loads(f.read_text())
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(split, indent=2))
    return split
