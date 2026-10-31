import argparse
import json
from pathlib import Path

from ..data.config import CLASSES, REPORTS_DIR
from ..data.provenance import file_info
from ..eval.line_crossing import PROTOCOL_VERSION
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
        if split not in gt_all:
            ap.error(f"GT counts are missing split '{split}'.")

        if split not in pred_all:
            ap.error(f"Predicted counts are missing split '{split}'.")

        seqs = list(splits[split])

        if not seqs or len(seqs) != len(set(seqs)):
            ap.error(f"Split '{split}' must contain unique sequences and be non-empty.")

        cfgs = sorted(pred_all[split])

        if not cfgs:
            ap.error(f"No predicted configurations found for split '{split}'.")

        expected = set(seqs)

        sources = [("GT", gt_all[split])]
        sources.extend(
            (cfg, pred_all[split][cfg])
            for cfg in cfgs
        )

        for source_name, sequence_results in sources:
            actual = set(sequence_results)
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)

            if missing or unexpected:
                ap.error(f"Sequence mismatch in {split}/{source_name}: missing={missing}, unexpected={unexpected}")

        for cfg in cfgs:
            for seq in seqs:
                gt_protocol = gt_all[split][seq].get("protocol")
                pred_protocol = pred_all[split][cfg][seq].get("protocol")
                if not gt_protocol or gt_protocol.get("version") != PROTOCOL_VERSION:
                    ap.error(f"Regenerate GT counts with the current counting protocol: {seq}")
                if pred_protocol != gt_protocol:
                    ap.error(f"Counting protocol mismatch in {split}/{cfg}/{seq}; regenerate both counts")
                if args.gt_variant == "all" and gt_protocol["ignored_regions"]:
                    ap.error("The all GT variant includes ignored observations absent from tracking results")

        def gt_vec(cls):
            return [gt_all[split][s][args.gt_variant][cls]["total"] for s in seqs]

        def pred_vec(cfg, cls):
            return [
                pred_all[split][cfg][s]["counts"][cls]["total"]
                for s in seqs
            ]

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
        dump["_sources"] = {"gt": file_info(args.gt), "tracks": file_info(args.tracks),
                            "splits": file_info(REPORTS_DIR / "splits.json")}
        dump["_gt"] = {"variant": args.gt_variant, "counts": {s: {c: gt_all[split][s][args.gt_variant][c]["total"]
                                                                  for c in CLASSES + ["all"]} for s in seqs},
                       "totals": gt_totals}

        def fmt(value):
            return "n/a" if value is None else f"{value:.2f}"


        def fmt_percent(value):
            return "n/a" if value is None else f"{100 * value:.2f}%"


        def metric_row(cls):
            cells = [
                f"{fmt(rows[c][cls]['mae'])} / "
                f"{fmt_percent(rows[c][cls]['wape'])}"
                for c in cfgs
            ]

            return (
                f"| {cls} | {gt_totals[cls]} | "
                + " | ".join(cells)
                + " |"
            )

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

        json_text = json.dumps(
            dump,
            indent=2,
            allow_nan=False,
        ) + "\n"

        out_md.parent.mkdir(parents=True, exist_ok=True)

        out_md.write_text(
            "\n".join(md) + "\n",
            encoding="utf-8",
        )

        out_json.write_text(
            json_text,
            encoding="utf-8",
        )
        
        print(f"[done] {split}: GT totals {gt_totals}")
        for c in cfgs:
            r = rows[c]["all"]
            print(f"  {c}: MAE={r['mae']:.2f} WAPE={fmt_percent(r['wape'])} meanSCE={r['mean_sce']:+.2f}")
        print(f"  -> {out_md}")
        print(f"  -> {out_json}")


if __name__ == "__main__":
    main()
