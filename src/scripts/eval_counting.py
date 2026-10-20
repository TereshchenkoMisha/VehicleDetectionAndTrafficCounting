"""Compare line-crossing counts of the detector x tracker configurations against ground truth.

Builds per-sequence count vectors aligned in the splits.json order and evaluates them with
src/eval/counting_metrics.py (MAE, WAPE, SCE aggregates) - overall and per class. Ground
truth comes from reports/gt_counts.json; by default the exclude_ignored variant is used
because the tracking pipeline drops detections with centres in ignored regions before the
tracker, so GT with those targets would not be comparable.

Output: reports/counting_eval_<split>.md and .json

Usage:
    python -m src.scripts.eval_counting                    # dev + test
    python -m src.scripts.eval_counting --splits dev --gt-variant all
"""
import argparse
import json
from pathlib import Path

from ..data.config import CLASSES, REPORTS_DIR
from ..eval.counting_metrics import mae, mean_sce, sce, sce_over_rate, sce_under_rate, wape

VARIANTS = ["all", "exclude_ignored"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--gt", default=str(REPORTS_DIR / "gt_counts.json"))
    ap.add_argument("--tracks", default=str(REPORTS_DIR / "track_counts.json"))
    ap.add_argument("--out-prefix", default=str(REPORTS_DIR / "counting_eval"))
    ap.add_argument("--gt-variant", choices=VARIANTS, default="exclude_ignored",
                    help="GT counts used in the tables (exclude_ignored matches the tracking protocol)")
    args = ap.parse_args()

    gt_all = json.loads(Path(args.gt).read_text())
    pred_all = json.loads(Path(args.tracks).read_text())
    splits = json.loads((REPORTS_DIR / "splits.json").read_text())

    for split in args.splits:
        if split not in gt_all or split not in pred_all:
            print(f"[warn] no GT or track counts for split '{split}', skipped")
            continue
        seqs = [s for s in splits[split] if s in gt_all[split]]
        cfgs = sorted(pred_all[split])
        if not cfgs:
            print(f"[warn] no track configs for split '{split}', skipped")
            continue

        def gt_vec(cls):
            return [gt_all[split][s][args.gt_variant][cls]["total"] for s in seqs]

        def pred_vec(cfg, cls):
            return [pred_all[split][cfg].get(s, {}).get("counts", {}).get(cls, {"total": 0})["total"]
                    for s in seqs]

        rows, dump = {}, {}
        for cfg in cfgs:
            rows[cfg], dump[cfg] = {}, {}
            for cls in CLASSES + ["all"]:
                gt, pr = gt_vec(cls), pred_vec(cfg, cls)
                rows[cfg][cls] = {"mae": mae(pr, gt), "wape": wape(pr, gt),
                                  "mean_sce": mean_sce(pr, gt),
                                  "over": sce_over_rate(pr, gt), "under": sce_under_rate(pr, gt)}
                dump[cfg][cls] = {"pred": pr, "gt": gt, "sce": sce(pr, gt), **rows[cfg][cls]}

        gt_totals = {c: sum(gt_vec(c)) for c in CLASSES + ["all"]}
        dump["_gt"] = {"variant": args.gt_variant, "counts": {s: {c: gt_all[split][s][args.gt_variant][c]["total"]
                                                                  for c in CLASSES + ["all"]} for s in seqs},
                       "totals": gt_totals}

        # markdown: one table per class + a compact summary table
        def fmt(v):
            return f"{v:.2f}" if v == v else "n/a"

        def metric_row(cls):
            head = (f"| {cls} | {gt_totals[cls]} | " +
                    " | ".join(f"{fmt(rows[c][cls]['mae'])} / {fmt(100 * rows[c][cls]['wape'])}%"
                               for c in cfgs) + " |")
            return head

        md = [f"# Counting evaluation - {split} split",
              "",
              f"Line-crossing counts vs GT (`{args.gt_variant}`): MAE / WAPE per class, "
              f"sequences in splits.json order ({len(seqs)}).",
              "",
              "| class | GT total | " + " | ".join(cfgs) + " |",
              "|" + "---|" * (len(cfgs) + 2)]
        for cls in CLASSES + ["all"]:
            md.append(metric_row(cls))
        md += ["", "## Signed per-sequence error (SCE, pred - gt)", ""]
        for cls in CLASSES + ["all"]:
            md.append(f"**{cls}**")
            md.append("")
            md.append("| config | " + " | ".join(seqs) + " |")
            md.append("|" + "---|" * (len(seqs) + 1))
            for c in cfgs:
                md.append(f"| {c} | " + " | ".join(f"{e:+.0f}" for e in dump[c][cls]["sce"]) + " |")
            md.append("")
        md += [f"mean SCE / over-rate / under-rate (all classes): " +
               ", ".join(f"{c}: {rows[c]['all']['mean_sce']:+.2f} / {rows[c]['all']['over']:.0%} / "
                         f"{rows[c]['all']['under']:.0%}" for c in cfgs), ""]

        out_md = Path(f"{args.out_prefix}_{split}.md")
        out_json = Path(f"{args.out_prefix}_{split}.json")
        out_md.write_text("\n".join(md) + "\n")
        out_json.write_text(json.dumps(dump, indent=2) + "\n")
        print(f"[done] {split}: GT totals {gt_totals}")
        for c in cfgs:
            r = rows[c]["all"]
            print(f"  {c}: MAE={r['mae']:.2f} WAPE={100 * r['wape']:.1f}% meanSCE={r['mean_sce']:+.2f}")
        print(f"  -> {out_md}")
        print(f"  -> {out_json}")


if __name__ == "__main__":
    main()
