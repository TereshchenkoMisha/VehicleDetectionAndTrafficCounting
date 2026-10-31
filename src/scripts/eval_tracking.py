import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..data.config import CLASSES, RAW_ROOT, REPORTS_DIR
from ..data.frames import list_frames
from ..data.provenance import file_info, write_json
from ..eval.tracking_metrics import combine_sequence_metrics, evaluate_sequence
from ..eval.run_validation import validate_tracking_run

GT_MOT_DIR = REPORTS_DIR / "gt_mot"
TRACKS_DIR = Path("outputs/tracks")


def _aggregate(per_seq, splits_seqs):
    # Average metrics across sequences and sum additive counters
    expected = set(splits_seqs)
    actual = set(per_seq)

    if not expected or actual != expected:
        raise ValueError(
            "Cannot aggregate incomplete tracking results: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
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
    n = len(per_seq)
    return {
        "aggregation": "macro average with equal sequence weights",
        "mean": {k: float(np.mean(v)) for k, v in means.items()},
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
    ap.add_argument("--raw", default=str(RAW_ROOT))
    args = ap.parse_args()

    splits = json.loads((REPORTS_DIR / "splits.json").read_text())
    gt_mot_dir = Path(args.gt_mot_dir)

    if not gt_mot_dir.exists():
        raise FileNotFoundError(
            f"GT MOT directory not found: {gt_mot_dir}. "
            "Run `python -m src.scripts.prepare_gt_mot` first."
        )

    for split in args.splits:
        seqs = list(splits[split])

        if not seqs or len(seqs) != len(set(seqs)):
            ap.error(f"Split '{split}' must contain unique sequences and be non-empty.")

        split_dir = Path(args.tracks_dir) / split

        if not split_dir.is_dir():
            ap.error(f"Tracking directory does not exist: {split_dir}")

        cfg_dirs = sorted(
            directory.name
            for directory in split_dir.iterdir()
            if directory.is_dir()
        )

        if not cfg_dirs:
            ap.error(f"No tracking configurations found in {split_dir}")

        missing_gt = [
            str(gt_mot_dir / f"{seq}.txt")
            for seq in seqs
            if not (gt_mot_dir / f"{seq}.txt").is_file()
        ]

        if missing_gt:
            ap.error("Missing GT MOT files:\n" + "\n".join(missing_gt))

        for cfg in cfg_dirs:
            try:
                validate_tracking_run(
                    split_dir / cfg,
                    seqs,
                    raw_root=Path(args.raw),
                    split=split,
                )
            except (ValueError, OSError) as exc:
                ap.error(str(exc))

        frame_ids = {seq: [fid for fid, _ in list_frames(seq, args.raw)] for seq in seqs}
        results = {}
        for cfg in cfg_dirs:
            raw_results = {}
            results[cfg] = {"per_seq": {},
                            "sources": {"timing": file_info(split_dir / cfg / "timing.json"),
                                        "tracks": {}, "gt": {}}}
            for seq in tqdm(seqs, desc=f"{split}/{cfg}", unit="seq"):
                gt_path = gt_mot_dir / f"{seq}.txt"
                trk_path = Path(args.tracks_dir) / split / cfg / f"{seq}.txt"
                
                cls_list = CLASSES if args.classes_aware else None
                raw_results[seq] = {}
                results[cfg]["per_seq"][seq] = evaluate_sequence(
                    gt_path, trk_path, classes=cls_list, frame_ids=frame_ids[seq],
                    raw_results=raw_results[seq])
                results[cfg]["sources"]["tracks"][seq] = file_info(trk_path)
                results[cfg]["sources"]["gt"][seq] = file_info(gt_path)
            results[cfg]["aggregate"] = _aggregate(results[cfg]["per_seq"], seqs)
            results[cfg]["aggregate"]["combined"] = combine_sequence_metrics(raw_results)

        out_json = Path(f"{args.out_prefix}_{split}.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_json, results)

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
        "## Macro average across sequences",
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

    lines += ["", "## Combined dataset metrics", "",
              "Combine sequence statistics with TrackEval before calculating dataset metrics", "",
              "| Config | HOTA (%) | MOTA (%) | IDF1 (%) |",
              "|---|---|---|---|"]
    for cfg in cfgs:
        combined = results[cfg]["aggregate"]["combined"]
        lines.append(f"| {cfg} | {combined['HOTA']:.2f} | {combined['MOTA']:.2f} | {combined['IDF1']:.2f} |")

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
    print(f"\n=== {split} - macro average across sequences ===")
    print(f"{'Config':<35} {'HOTA%':>7} {'MOTA%':>7} {'IDF1%':>7} {'IDSW':>6}")
    for cfg, r in results.items():
        a = r["aggregate"]["mean"]
        s = r["aggregate"]["sum"]
        print(f"{cfg:<35} {a['HOTA']:>7.2f} {a['MOTA']:>7.2f} "
              f"{a['IDF1']:>7.2f} {s['IDSW']:>6}")


if __name__ == "__main__":
    main()
