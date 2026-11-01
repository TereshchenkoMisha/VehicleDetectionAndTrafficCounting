"""Counting accuracy (MAE, WAPE, signed error) of every configuration against GT counts.

    python scripts/evaluate_counting.py --splits dev test
"""
import argparse
from pathlib import Path

from src.config import CLASSES, REPORTS_DIR
from src.counting.line_crossing import PROTOCOL_VERSION
from src.counting.metrics import count_errors
from src.utils import file_info, read_json, write_json

VARIANTS = ["all", "exclude_ignored"]
CLASSES_AND_ALL = CLASSES + ["all"]


def check_inputs(ap, split, seqs, gt, preds, gt_variant):
    """Same sequences everywhere, and GT and tracker counts made with the identical counting protocol."""
    for name, sequence_results in [("GT", gt)] + list(preds.items()):
        missing, unexpected = sorted(set(seqs) - set(sequence_results)), sorted(set(sequence_results) - set(seqs))
        if missing or unexpected:
            ap.error(f"Sequence mismatch in {split}/{name}: missing={missing}, unexpected={unexpected}")
    for cfg, per_seq in preds.items():
        for seq in seqs:
            gt_protocol = gt[seq].get("protocol")
            if not gt_protocol or gt_protocol.get("version") != PROTOCOL_VERSION:
                ap.error(f"Regenerate GT counts with the current counting protocol: {seq}")
            if per_seq[seq].get("protocol") != gt_protocol:
                ap.error(f"Counting protocol mismatch in {split}/{cfg}/{seq}; regenerate both counts")
            if gt_variant == "all" and gt_protocol["ignored_regions"]:
                ap.error("The 'all' GT variant includes ignored observations absent from tracking results")


def render_markdown(split, gt_variant, seqs, cfgs, rows, gt_totals):
    fmt = lambda v: "n/a" if v is None else f"{v:.2f}"                    # noqa: E731
    pct = lambda v: "n/a" if v is None else f"{100 * v:.2f}%"             # noqa: E731
    md = [f"# Counting evaluation - {split} split", "",
          f"Line-crossing counts vs GT (`{gt_variant}`): MAE / WAPE per class, "
          f"sequences in splits.json order ({len(seqs)}).", "",
          "| class | GT total | " + " | ".join(cfgs) + " |", "|" + "---|" * (len(cfgs) + 2)]
    for cls in CLASSES_AND_ALL:
        cells = " | ".join(f"{fmt(rows[c][cls]['mae'])} / {pct(rows[c][cls]['wape'])}" for c in cfgs)
        md.append(f"| {cls} | {gt_totals[cls]} | {cells} |")
    md += ["", "## Signed per-sequence error (SCE, pred - gt)", ""]
    for cls in CLASSES_AND_ALL:
        md += [f"**{cls}**", "", "| config | " + " | ".join(seqs) + " |", "|" + "---|" * (len(seqs) + 1)]
        md += [f"| {c} | " + " | ".join(f"{e:+.0f}" for e in rows[c][cls]["sce"]) + " |" for c in cfgs]
        md.append("")
    md += ["mean SCE / over-rate / under-rate (all classes): " + ", ".join(
        f"{c}: {rows[c]['all']['mean_sce']:+.2f} / {rows[c]['all']['over']:.0%} / {rows[c]['all']['under']:.0%}"
        for c in cfgs), ""]
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--gt", default=str(REPORTS_DIR / "gt_counts.json"))
    ap.add_argument("--tracks", default=str(REPORTS_DIR / "track_counts.json"))
    ap.add_argument("--out-prefix", default=str(REPORTS_DIR / "counting_eval"))
    ap.add_argument("--gt-variant", choices=VARIANTS, default="exclude_ignored",
                    help="GT counts used in the tables (exclude_ignored matches the tracking protocol)")
    args = ap.parse_args()

    gt_all, pred_all = read_json(args.gt), read_json(args.tracks)
    splits = read_json(REPORTS_DIR / "splits.json")

    for split in args.splits:
        for name, source in (("GT counts", gt_all), ("Predicted counts", pred_all)):
            if split not in source:
                ap.error(f"{name} are missing split '{split}'.")
        seqs, cfgs = list(splits[split]), sorted(pred_all[split])
        if not cfgs:
            ap.error(f"No predicted configurations found for split '{split}'.")
        check_inputs(ap, split, seqs, gt_all[split], pred_all[split], args.gt_variant)

        gt_counts = {cls: [gt_all[split][s][args.gt_variant][cls]["total"] for s in seqs] for cls in CLASSES_AND_ALL}
        rows = {cfg: {cls: count_errors([pred_all[split][cfg][s]["counts"][cls]["total"] for s in seqs],
                                        gt_counts[cls])
                      for cls in CLASSES_AND_ALL} for cfg in cfgs}
        gt_totals = {cls: sum(v) for cls, v in gt_counts.items()}

        dump = {cfg: {cls: {"pred": [pred_all[split][cfg][s]["counts"][cls]["total"] for s in seqs],
                            "gt": gt_counts[cls], **rows[cfg][cls]} for cls in CLASSES_AND_ALL} for cfg in cfgs}
        dump["_sources"] = {"gt": file_info(args.gt), "tracks": file_info(args.tracks),
                            "splits": file_info(REPORTS_DIR / "splits.json")}
        dump["_gt"] = {"variant": args.gt_variant, "totals": gt_totals,
                       "counts": {s: {cls: gt_all[split][s][args.gt_variant][cls]["total"]
                                      for cls in CLASSES_AND_ALL} for s in seqs}}

        out_md, out_json = Path(f"{args.out_prefix}_{split}.md"), Path(f"{args.out_prefix}_{split}.json")
        write_json(out_json, dump)
        out_md.write_text(render_markdown(split, args.gt_variant, seqs, cfgs, rows, gt_totals), encoding="utf-8")

        print(f"[done] {split}: GT totals {gt_totals}")
        for cfg in cfgs:
            r = rows[cfg]["all"]
            wape = "n/a" if r["wape"] is None else f"{100 * r['wape']:.2f}%"
            print(f"  {cfg}: MAE={r['mae']:.2f} WAPE={wape} meanSCE={r['mean_sce']:+.2f}")
        print(f"  -> {out_md}\n  -> {out_json}")


if __name__ == "__main__":
    main()
