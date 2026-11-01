"""Check the results of one run (scripts/run_all.py) and compare them with the committed reference results.

    python scripts/verify_run.py --tag run_20261101_120000

Two kinds of output. PASS/FAIL checks test things that must hold: runs are complete, metrics are in a
valid range, the detector reproduces the reference mAP, GT counts equal the reference counts. [info]
lines show old -> new values of metrics that are expected to change between runs.

Needs:
    reports/<tag>/                         detector reports, tracking_eval_*, counting_eval_*, gt_counts.json
    outputs/tracks_<tag>/<split>/          tracking runs
    data/raw/UA-DETRAC/                    frame lists for the run check
    reports/*.json                         reference results; a missing reference skips its comparison
Produces nothing. Exit code 1 if any check failed.
"""
import argparse
import math
import sys
from pathlib import Path
from typing import Any

from src.config import CLASSES, REPORTS_DIR, ROOT
from src.eval.validation import run_dirs, validate_tracking_run
from src.utils import read_json

DETECTORS: tuple[str, ...] = ("yolov8n", "yolov8s")
# The reference was made with another Ultralytics version and GPU: one box can flip, moving mAP by about 0.002
MAP_TOLERANCE: float = 5e-3
failures: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    """Print one PASS/FAIL line and remember failures for the exit code."""
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def info(text: str) -> None:
    """Print a line that is not a check."""
    print(f"[info] {text}")


def check_detectors(new_dir: Path, split: str) -> None:
    """Detector reports: mAP in range, and mAP equal to the reference within MAP_TOLERANCE.

    The detector, data and protocol do not change between runs, so mAP should reproduce. A change in
    TP/FP/FN counts is only reported: a handful of flipped boxes is normal across library versions.
    """
    for det in DETECTORS:
        new_path, old_path = new_dir / f"detection_{det}_{split}.json", REPORTS_DIR / f"detection_{det}_{split}.json"
        if not new_path.is_file():
            check(False, f"detector report {det}/{split} exists", str(new_path))
            continue
        new = read_json(new_path)
        check(new["frames"] > 0 and 0 < new["mAP@0.5"] <= 1, f"detector {det}/{split} mAP in range",
              f"mAP@0.5={new['mAP@0.5']}, frames={new['frames']}")
        if not old_path.is_file():
            continue

        old = read_json(old_path)
        for key in ("mAP@0.5", "mAP@0.5:0.95"):
            check(abs(new[key] - old[key]) <= MAP_TOLERANCE, f"detector {det}/{split} {key} reproduces reference",
                  f"new={new[key]} old={old[key]}")
        for cls in CLASSES:
            n, o = new["per_class"][cls], old["per_class"][cls]
            if (n["TP"], n["FP"], n["FN"]) != (o["TP"], o["FP"], o["FN"]):
                info(f"{det}/{split}/{cls} TP/FP/FN changed: new={(n['TP'], n['FP'], n['FN'])} "
                     f"old={(o['TP'], o['FP'], o['FN'])}")


def check_tracks(tracks_root: Path, split: str, sequences: list[str], raw_root: Path) -> None:
    """Tracking runs of a split: four finished configurations, each complete and consistent with the raw
    dataset, run with the low-score matching step active and not in deterministic (slow) mode."""
    cfg_dirs = run_dirs(tracks_root / split) if (tracks_root / split).is_dir() else []
    check(len(cfg_dirs) == 4, f"{split}: 4 tracking configurations finished", str([d.name for d in cfg_dirs]))

    for cfg_dir in cfg_dirs:
        try:
            timing = validate_tracking_run(cfg_dir, sequences, raw_root=raw_root, split=split)
        except (ValueError, OSError) as exc:
            check(False, f"{split}/{cfg_dir.name} complete and consistent", str(exc))
            continue
        check(True, f"{split}/{cfg_dir.name} complete and consistent",
              f"{timing['frames']} frames, {timing['fps']:.1f} FPS "
              f"(det {timing['det_ms_per_frame']:.1f} ms, trk {timing['trk_ms_per_frame']:.1f} ms)")
        check(timing["detector_conf"] < timing["conf"],
              f"{split}/{cfg_dir.name} detector ran below the tracker threshold (low-score step active)",
              f"detector_conf={timing['detector_conf']} conf={timing['conf']}")
        check(timing["deterministic"] is False, f"{split}/{cfg_dir.name} timed without deterministic mode")


def check_gt_counts(new_dir: Path, split: str) -> None:
    """GT counts per sequence must equal the reference: ground truth does not depend on any model."""
    new_path, old_path = new_dir / "gt_counts.json", REPORTS_DIR / "gt_counts.json"
    if not new_path.is_file():
        check(False, "gt_counts.json exists", str(new_path))
        return
    new = read_json(new_path)[split]
    old = read_json(old_path).get(split, {}) if old_path.is_file() else {}
    for seq, res in new.items():
        if seq in old:
            n, o = res["exclude_ignored"]["all"]["total"], old[seq]["exclude_ignored"]["all"]["total"]
            check(n == o, f"GT count {split}/{seq} equals reference", f"new={n} old={o}")
    info(f"GT total {split}: {sum(r['exclude_ignored']['all']['total'] for r in new.values())}")


def check_tracking_eval(new_dir: Path, split: str) -> None:
    """Tracking metrics: dataset-level rates are within 0-100. HOTA and IDF1 old -> new are printed."""
    path = new_dir / f"tracking_eval_{split}.json"
    if not path.is_file():
        check(False, f"tracking_eval_{split}.json exists", str(path))
        return
    old_path = REPORTS_DIR / f"tracking_eval_{split}.json"
    old: dict[str, Any] = read_json(old_path) if old_path.is_file() else {}

    for cfg, body in read_json(path).items():
        c = body["aggregate"]["combined"]
        check(all(0 <= c[k] <= 100 for k in ("HOTA", "IDF1", "DetA", "AssA")), f"tracking metrics {split}/{cfg} in range",
              f"HOTA={c['HOTA']:.2f} IDF1={c['IDF1']:.2f} MOTA={c['MOTA']:.2f} IDSW={body['aggregate']['sum']['IDSW']}")
        if cfg in old:
            # Older reports may lack the combined block; fall back to the macro average then
            o = old[cfg]["aggregate"].get("combined", old[cfg]["aggregate"]["mean"])
            info(f"{split}/{cfg}: HOTA {o['HOTA']:.2f} -> {c['HOTA']:.2f}, "
                 f"IDF1 {o['IDF1']:.2f} -> {c['IDF1']:.2f} (old -> new)")


def check_counting_eval(new_dir: Path, split: str) -> None:
    """Counting metrics: finite, with one prediction per GT value. MAE old -> new is printed."""
    path = new_dir / f"counting_eval_{split}.json"
    if not path.is_file():
        check(False, f"counting_eval_{split}.json exists", str(path))
        return
    old_path = REPORTS_DIR / f"counting_eval_{split}.json"
    old: dict[str, Any] = read_json(old_path) if old_path.is_file() else {}

    for cfg, body in read_json(path).items():
        if cfg.startswith("_"):      # metadata entries (_sources, _gt), not configurations
            continue
        a = body["all"]
        wape = "n/a" if a["wape"] is None else f"{100 * a['wape']:.2f}%"
        check(math.isfinite(a["mae"]) and len(a["pred"]) == len(a["gt"]) > 0, f"counting metrics {split}/{cfg}",
              f"MAE={a['mae']:.2f} WAPE={wape} meanSCE={a['mean_sce']:+.2f}")
        if cfg in old:
            info(f"{split}/{cfg}: MAE {old[cfg]['all']['mae']:.2f} -> {a['mae']:.2f} (old -> new)")


def main() -> None:
    """Run all checks for the chosen splits and exit with 1 if any failed."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="run tag used by run_all.py")
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--raw", default=str(ROOT / "data/raw/UA-DETRAC"))
    args = ap.parse_args()

    new_dir, tracks_root = REPORTS_DIR / args.tag, ROOT / "outputs" / f"tracks_{args.tag}"
    splits = read_json(REPORTS_DIR / "splits.json")
    for split in args.splits:
        print(f"\n===== {split} =====")
        check_detectors(new_dir, split)
        check_tracks(tracks_root, split, splits[split], Path(args.raw))
        check_gt_counts(new_dir, split)
        check_tracking_eval(new_dir, split)
        check_counting_eval(new_dir, split)

    print(f"\n{len(failures)} check(s) failed" + (":" if failures else "."))
    for name in failures:
        print("  -", name)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
