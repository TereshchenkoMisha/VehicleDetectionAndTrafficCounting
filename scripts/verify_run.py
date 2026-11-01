"""Check the results of one full run (scripts/run_all.py) and compare them with the committed reference results.

    python scripts/verify_run.py --tag run_20261101_120000

Exit code 0 when every hard check passes. "info" lines are comparisons with the old (pre-fix) results.
"""
import argparse
import math
import sys
from pathlib import Path

from src.config import CLASSES, REPORTS_DIR, ROOT
from src.eval.validation import run_dirs, validate_tracking_run
from src.utils import read_json

DETECTORS = ("yolov8n", "yolov8s")
# The reference was made with another Ultralytics version / GPU: one box can flip, moving mAP by about 0.002.
MAP_TOLERANCE = 5e-3
failures = []


def check(ok, name, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def info(text):
    print(f"[info] {text}")


def check_detectors(new_dir, split):
    for det in DETECTORS:
        new_path, old_path = new_dir / f"detection_{det}_{split}.json", REPORTS_DIR / f"detection_{det}_{split}.json"
        if not new_path.is_file():
            check(False, f"detector report {det}/{split} exists", str(new_path))
            continue
        new = read_json(new_path)
        check(new["frames"] > 0 and 0 < new["mAP@0.5"] <= 1, f"detector {det}/{split} mAP in range",
              f"mAP@0.5={new['mAP@0.5']}, frames={new['frames']}")
        if old_path.is_file():          # same weights, data and deterministic mode: must reproduce the reference
            old = read_json(old_path)
            for key in ("mAP@0.5", "mAP@0.5:0.95"):
                diff = abs(new[key] - old[key])
                check(diff <= MAP_TOLERANCE, f"detector {det}/{split} {key} reproduces reference",
                      f"new={new[key]} old={old[key]}")
            for cls in CLASSES:
                n, o = new["per_class"][cls], old["per_class"][cls]
                if (n["TP"], n["FP"], n["FN"]) != (o["TP"], o["FP"], o["FN"]):
                    info(f"{det}/{split}/{cls} TP/FP/FN changed: new={(n['TP'], n['FP'], n['FN'])} "
                         f"old={(o['TP'], o['FP'], o['FN'])}")


def check_tracks(tracks_root, split, sequences, raw_root):
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


def check_gt_counts(new_dir, split):
    new_path, old_path = new_dir / "gt_counts.json", REPORTS_DIR / "gt_counts.json"
    if not new_path.is_file():
        return check(False, "gt_counts.json exists", str(new_path))
    new = read_json(new_path)[split]
    old = read_json(old_path).get(split, {}) if old_path.is_file() else {}
    for seq, res in new.items():
        if seq in old:
            n, o = res["exclude_ignored"]["all"]["total"], old[seq]["exclude_ignored"]["all"]["total"]
            check(n == o, f"GT count {split}/{seq} equals reference", f"new={n} old={o}")
    info(f"GT total {split}: {sum(r['exclude_ignored']['all']['total'] for r in new.values())}")


def check_tracking_eval(new_dir, split):
    path = new_dir / f"tracking_eval_{split}.json"
    if not path.is_file():
        return check(False, f"tracking_eval_{split}.json exists", str(path))
    old_path = REPORTS_DIR / f"tracking_eval_{split}.json"
    old = read_json(old_path) if old_path.is_file() else {}
    for cfg, body in read_json(path).items():
        c = body["aggregate"]["combined"]
        ok = all(0 <= c[k] <= 100 for k in ("HOTA", "IDF1", "DetA", "AssA"))
        check(ok, f"tracking metrics {split}/{cfg} in range",
              f"HOTA={c['HOTA']:.2f} IDF1={c['IDF1']:.2f} MOTA={c['MOTA']:.2f} IDSW={body['aggregate']['sum']['IDSW']}")
        if cfg in old:
            o = old[cfg]["aggregate"].get("combined", old[cfg]["aggregate"]["mean"])
            info(f"{split}/{cfg}: HOTA {o['HOTA']:.2f} -> {c['HOTA']:.2f}, IDF1 {o['IDF1']:.2f} -> {c['IDF1']:.2f} (old -> new)")


def check_counting_eval(new_dir, split):
    path = new_dir / f"counting_eval_{split}.json"
    if not path.is_file():
        return check(False, f"counting_eval_{split}.json exists", str(path))
    old_path = REPORTS_DIR / f"counting_eval_{split}.json"
    old = read_json(old_path) if old_path.is_file() else {}
    for cfg, body in read_json(path).items():
        if cfg.startswith("_"):
            continue
        a = body["all"]
        ok = math.isfinite(a["mae"]) and len(a["pred"]) == len(a["gt"]) > 0
        wape = "n/a" if a["wape"] is None else f"{100 * a['wape']:.2f}%"
        check(ok, f"counting metrics {split}/{cfg}", f"MAE={a['mae']:.2f} WAPE={wape} meanSCE={a['mean_sce']:+.2f}")
        if cfg in old:
            info(f"{split}/{cfg}: MAE {old[cfg]['all']['mae']:.2f} -> {a['mae']:.2f} (old -> new)")


def main():
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
