"""HOTA / MOTA / IDF1 / IDSW of every tracking run against GT (TrackEval). Writes reports/tracking_eval_<split>.*

    python scripts/evaluate_tracking.py --splits dev test

GT MOT files (reports/gt_mot/<seq>.txt) are created on the first run and need the raw dataset.
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import RAW_ROOT, REPORTS_DIR, TRACKS_DIR
from src.data.export import convert_all
from src.data.sequences import list_frames
from src.eval.tracking import combine_sequence_metrics, evaluate_sequence
from src.eval.validation import run_dirs, validate_tracking_run
from src.utils import file_info, read_json, write_json

SUMS = ("IDSW", "MT", "ML", "Frag", "CLR_TP", "CLR_FN", "CLR_FP", "GT_dets", "TRK_dets")
MEANS = ("HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1", "IDP", "IDR", "HOTA_50")


def aggregate(per_seq, sequences):
    """Macro average (equal weight per sequence) of the rate metrics and sum of the counters."""
    if not sequences or set(per_seq) != set(sequences):
        raise ValueError(f"Cannot aggregate incomplete tracking results: "
                         f"missing={sorted(set(sequences) - set(per_seq))}, "
                         f"unexpected={sorted(set(per_seq) - set(sequences))}")
    return {"aggregation": "macro average with equal sequence weights",
            "mean": {k: float(np.mean([m["all"][k] for m in per_seq.values()])) for k in MEANS},
            "sum": {k: sum(m["all"][k] for m in per_seq.values()) for k in SUMS},
            "n_seq": len(per_seq)}


def write_markdown(path, split, results, seqs):
    cfgs = list(results)
    lines = [f"# Tracking evaluation - {split} split", "",
             "Metrics: **HOTA** (geometric mean of DetA & AssA, averaged over IoU 0.05-0.95), "
             "**MOTA** (1 - (FN+FP+IDSW)/GT), **IDSW** (identity switches), **IDF1** (F1 of identity matching).", "",
             f"Number of configs: {len(cfgs)}, number of sequences: {len(seqs)}.", "",
             "## Macro average across sequences", "",
             "| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |",
             "|---|---|---|---|---|---|---|---|"]
    for cfg in cfgs:
        a, s = results[cfg]["aggregate"]["mean"], results[cfg]["aggregate"]["sum"]
        lines.append(f"| {cfg} | {a['HOTA']:.2f} | {a['DetA']:.2f} | {a['AssA']:.2f} | "
                     f"{a['MOTA']:.2f} | {a['MOTP']:.2f} | {a['IDF1']:.2f} | {s['IDSW']} |")
    lines += ["", "## Combined dataset metrics (TrackEval combination of all sequences)", "",
              "| Config | HOTA (%) | MOTA (%) | IDF1 (%) |", "|---|---|---|---|"]
    for cfg in cfgs:
        c = results[cfg]["aggregate"]["combined"]
        lines.append(f"| {cfg} | {c['HOTA']:.2f} | {c['MOTA']:.2f} | {c['IDF1']:.2f} |")
    lines += ["", "## Per-sequence (HOTA / MOTA / IDF1 / IDSW)", "",
              "| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |", "|---|---|---|---|---|---|"]
    for cfg in cfgs:
        for seq in seqs:
            m = results[cfg]["per_seq"][seq]["all"]
            lines.append(f"| {cfg} | {seq} | {m['HOTA']:.2f} | {m['MOTA']:.2f} | {m['IDF1']:.2f} | {m['IDSW']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--gt-mot-dir", default=str(REPORTS_DIR / "gt_mot"))
    ap.add_argument("--tracks-dir", default=str(TRACKS_DIR))
    ap.add_argument("--out-prefix", default=str(REPORTS_DIR / "tracking_eval"))
    ap.add_argument("--raw", default=str(RAW_ROOT))
    args = ap.parse_args()

    all_splits = read_json(REPORTS_DIR / "splits.json")
    gt_dir = Path(args.gt_mot_dir)

    for split in args.splits:
        seqs = list(all_splits[split])
        split_dir = Path(args.tracks_dir) / split
        if not split_dir.is_dir():
            ap.error(f"Tracking directory does not exist: {split_dir}")
        cfg_dirs = run_dirs(split_dir)
        if not cfg_dirs:
            ap.error(f"No tracking configurations found in {split_dir}")
        if any(not (gt_dir / f"{seq}.txt").is_file() for seq in seqs):
            convert_all(seqs, gt_dir, raw_root=args.raw, ignored_path=REPORTS_DIR / "ignored_regions.json")
        for cfg_dir in cfg_dirs:
            try:
                validate_tracking_run(cfg_dir, seqs, raw_root=Path(args.raw), split=split)
            except (ValueError, OSError) as exc:
                ap.error(str(exc))

        frame_ids = {seq: [fid for fid, _ in list_frames(seq, args.raw)] for seq in seqs}
        results = {}
        for cfg_dir in cfg_dirs:
            raw_results = {}
            results[cfg_dir.name] = {"per_seq": {}, "sources": {"timing": file_info(cfg_dir / "timing.json"),
                                                                "tracks": {}, "gt": {}}}
            for seq in tqdm(seqs, desc=f"{split}/{cfg_dir.name}", unit="seq"):
                raw_results[seq] = {}
                results[cfg_dir.name]["per_seq"][seq] = evaluate_sequence(
                    gt_dir / f"{seq}.txt", cfg_dir / f"{seq}.txt", frame_ids=frame_ids[seq],
                    raw_results=raw_results[seq])
                results[cfg_dir.name]["sources"]["tracks"][seq] = file_info(cfg_dir / f"{seq}.txt")
                results[cfg_dir.name]["sources"]["gt"][seq] = file_info(gt_dir / f"{seq}.txt")
            results[cfg_dir.name]["aggregate"] = aggregate(results[cfg_dir.name]["per_seq"], seqs)
            results[cfg_dir.name]["aggregate"]["combined"] = combine_sequence_metrics(raw_results)

        out_json, out_md = Path(f"{args.out_prefix}_{split}.json"), Path(f"{args.out_prefix}_{split}.md")
        write_json(out_json, results)
        write_markdown(out_md, split, results, seqs)
        print(f"[done] {split}: {out_md} + {out_json}")
        print(f"{'Config':<35} {'HOTA%':>7} {'MOTA%':>7} {'IDF1%':>7} {'IDSW':>6}")
        for cfg, r in results.items():
            a, s = r["aggregate"]["mean"], r["aggregate"]["sum"]
            print(f"{cfg:<35} {a['HOTA']:>7.2f} {a['MOTA']:>7.2f} {a['IDF1']:>7.2f} {s['IDSW']:>6}")


if __name__ == "__main__":
    main()
