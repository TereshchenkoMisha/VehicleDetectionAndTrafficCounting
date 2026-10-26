import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..data.config import CLASSES, REPORTS_DIR
from ..eval.tracking_metrics import evaluate_sequence

GT_MOT_DIR = REPORTS_DIR / "gt_mot"
TRACKS_DIR = Path("outputs/tracks")


def _aggregate(per_seq, splits_seqs):
    # Micro- and macro-averaging across split sequences
    sums = {"IDSW": 0, "MT": 0, "ML": 0, "Frag": 0,
            "CLR_TP": 0, "CLR_FN": 0, "CLR_FP": 0,
            "GT_dets": 0, "TRK_dets": 0}
    means = {"HOTA": [], "DetA": [], "AssA": [], "LocA": [],
             "MOTA": [], "MOTP": [], "IDF1": [], "IDP": [], "IDR": [],
             "HOTA_50": []}
    for seq, m in per_seq.items():
        for k in sums:
            sums[k] += m["all"][k]
        for k in means:
            means[k].append(m["all"][k])
    n = len(splits_seqs)
    return {
        "mean": {k: (float(np.mean(v)) if v else 0.0) for k, v in means.items()},
        "sum":  sums,
        "n_seq": n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--gt-mot-dir", default=str(GT_MOT_DIR))
    ap.add_argument("--tracks-dir", default=str(TRACKS_DIR))
    ap.add_argument("--out-prefix", default=str(REPORTS_DIR / "tracking_eval"))
    ap.add_argument("--classes-aware", action="store_true",
                    help="calculate per-class car/bus/truck additionally (slower)")
    args = ap.parse_args()

    splits = json.loads((REPORTS_DIR / "splits.json").read_text())
    gt_mot_dir = Path(args.gt_mot_dir)

    if not gt_mot_dir.exists():
        raise FileNotFoundError(
            f"GT MOT directory not found: {gt_mot_dir}. "
            "Run `python -m src.scripts.prepare_gt_mot` first."
        )

    for split in args.splits:
        seqs = splits[split]
        cfg_dirs = sorted(d.name for d in (Path(args.tracks_dir) / split).iterdir() if d.is_dir())
        if not cfg_dirs:
            print(f"[warn] no configs in {Path(args.tracks_dir) / split}, skipped")
            continue

        results = {}
        for cfg in cfg_dirs:
            results[cfg] = {"per_seq": {}}
            for seq in tqdm(seqs, desc=f"{split}/{cfg}", unit="seq"):
                gt_path = gt_mot_dir / f"{seq}.txt"
                trk_path = Path(args.tracks_dir) / split / cfg / f"{seq}.txt"
                if not gt_path.exists():
                    print(f"[warn] GT MOT missing: {gt_path}, skipped")
                    continue
                if not trk_path.exists():
                    print(f"[warn] track missing: {trk_path}, skipped")
                    continue
                cls_list = CLASSES if args.classes_aware else None
                results[cfg]["per_seq"][seq] = evaluate_sequence(gt_path, trk_path, classes=cls_list)
            results[cfg]["aggregate"] = _aggregate(results[cfg]["per_seq"], seqs)

        out_json = Path(f"{args.out_prefix}_{split}.json")
        out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

        out_md = Path(f"{args.out_prefix}_{split}.md")
        _write_markdown(out_md, split, results, seqs)
        print(f"[done] {split}: {out_md} + {out_json}")
        _print_summary(results, split)


def _write_markdown(out_md, split, results, seqs):
    cfgs = list(results.keys())
    lines = [
        f"# Tracking evaluation — {split} split",
        "",
        "Metrics: **HOTA** (geometry mean of DetA & AssA, averaged over IoU 0.05-0.95), "
        "**MOTA** (1 - (FN+FP+IDSW)/GT), **IDSW** (count of identity switches), "
        "**IDF1** (F1 of identity matching).",
        "",
        f"Number of configs: {len(cfgs)}, Number of sequences: {len(seqs)}.",
        "",
        "## Aggregate (overall, class-agnostic)",
        "",
        "| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cfg in cfgs:
        a = results[cfg]["aggregate"]["mean"]
        s = results[cfg]["aggregate"]["sum"]
        lines.append(
            f"| {cfg} | {a['HOTA']:.2f} | {a['DetA']:.2f} | {a['AssA']:.2f} | "
            f"{a['MOTA']:.2f} | {a['MOTP']:.2f} | {a['IDF1']:.2f} | {s['IDSW']} |"
        )

    lines += ["", "## Per-sequence (HOTA / MOTA / IDF1 / IDSW)", "",
              "| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |",
              "|---|---|---|---|---|---|"]
    for cfg in cfgs:
        for seq in seqs:
            if seq not in results[cfg]["per_seq"]:
                continue
            m = results[cfg]["per_seq"][seq]["all"]
            lines.append(
                f"| {cfg} | {seq} | {m['HOTA']:.2f} | {m['MOTA']:.2f} | "
                f"{m['IDF1']:.2f} | {m['IDSW']} |"
            )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(results, split):
    print(f"\n=== {split} - aggregate (overall) ===")
    print(f"{'Config':<35} {'HOTA%':>7} {'MOTA%':>7} {'IDF1%':>7} {'IDSW':>6}")
    for cfg, r in results.items():
        a = r["aggregate"]["mean"]
        s = r["aggregate"]["sum"]
        print(f"{cfg:<35} {a['HOTA']:>7.2f} {a['MOTA']:>7.2f} "
              f"{a['IDF1']:>7.2f} {s['IDSW']:>6}")


if __name__ == "__main__":
    main()