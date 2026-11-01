"""Counting accuracy (MAE, WAPE, signed error) of every configuration against the GT counts.

    python scripts/evaluate_counting.py --splits dev test

For each split, the predicted count of each configuration is compared with the GT count per sequence and
class. The metrics (see src/counting/metrics.py) measure how far the totals are off, and in which direction.

Needs:
    reports/gt_counts.json                 from count_gt
    reports/track_counts.json              from count_tracks
    reports/splits.json
Produces:
    <out-prefix>_<split>.json              predicted and GT counts per sequence, metrics per class
    <out-prefix>_<split>.md                the same as tables (default prefix reports/counting_eval)
Stops if GT and tracker counts were made with different counting protocols.
"""
import argparse
from pathlib import Path
from typing import Any

from src.config import CLASSES, REPORTS_DIR
from src.counting.line_crossing import PROTOCOL_VERSION
from src.counting.metrics import count_errors
from src.utils import file_info, read_json, write_json

VARIANTS: list[str] = ["all", "exclude_ignored"]
CLASSES_AND_ALL: list[str] = CLASSES + ["all"]


def check_inputs(split: str, seqs: list[str], gt: dict[str, Any], preds: dict[str, dict[str, Any]],
                 gt_variant: str) -> None:
    """Exit unless GT and every configuration cover the same sequences with the same counting protocol.

    Counts made with a different line, ignored regions or frame gap do not measure the same thing, so
    comparing them would give meaningless errors.
    """
    for name, sequence_results in [("GT", gt), *preds.items()]:
        missing, unexpected = sorted(set(seqs) - set(sequence_results)), sorted(set(sequence_results) - set(seqs))
        if missing or unexpected:
            raise SystemExit(f"Sequence mismatch in {split}/{name}: missing={missing}, unexpected={unexpected}")

    for cfg, per_seq in preds.items():
        for seq in seqs:
            gt_protocol = gt[seq].get("protocol")
            if not gt_protocol or gt_protocol.get("version") != PROTOCOL_VERSION:
                raise SystemExit(f"Regenerate GT counts with the current counting protocol: {seq}")
            if per_seq[seq].get("protocol") != gt_protocol:
                raise SystemExit(f"Counting protocol mismatch in {split}/{cfg}/{seq}; regenerate both counts")
            if gt_variant == "all" and gt_protocol["ignored_regions"]:
                raise SystemExit("The 'all' GT variant includes ignored observations absent from tracking results")


def fmt(value: float | None) -> str:
    """Number with two decimals, or n/a."""
    return "n/a" if value is None else f"{value:.2f}"


def fmt_percent(value: float | None) -> str:
    """Ratio as a percentage with two decimals, or n/a."""
    return "n/a" if value is None else f"{100 * value:.2f}%"


def render_markdown(split: str, gt_variant: str, seqs: list[str], cfgs: list[str],
                    rows: dict[str, dict[str, dict[str, Any]]], gt_totals: dict[str, int]) -> str:
    """Markdown report: MAE / WAPE per class and configuration, signed error per sequence, bias summary."""
    md = [f"# Counting evaluation - {split} split", "",
          f"Line-crossing counts vs GT (`{gt_variant}`): MAE / WAPE per class, "
          f"sequences in splits.json order ({len(seqs)}).", "",
          "| class | GT total | " + " | ".join(cfgs) + " |", "|" + "---|" * (len(cfgs) + 2)]
    for cls in CLASSES_AND_ALL:
        cells = " | ".join(f"{fmt(rows[c][cls]['mae'])} / {fmt_percent(rows[c][cls]['wape'])}" for c in cfgs)
        md.append(f"| {cls} | {gt_totals[cls]} | {cells} |")

    # One table per class: a row per configuration, a column per sequence
    md += ["", "## Signed per-sequence error (SCE, pred - gt)", ""]
    for cls in CLASSES_AND_ALL:
        md += [f"**{cls}**", "", "| config | " + " | ".join(seqs) + " |", "|" + "---|" * (len(seqs) + 1)]
        md += [f"| {c} | " + " | ".join(f"{e:+.0f}" for e in rows[c][cls]["sce"]) + " |" for c in cfgs]
        md.append("")

    md += ["mean SCE / over-rate / under-rate (all classes): " + ", ".join(
        f"{c}: {rows[c]['all']['mean_sce']:+.2f} / {rows[c]['all']['over']:.0%} / {rows[c]['all']['under']:.0%}"
        for c in cfgs), ""]
    return "\n".join(md) + "\n"


def parse_args() -> argparse.Namespace:
    """Command-line options; see the module docstring for what the script reads and writes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--gt", default=str(REPORTS_DIR / "gt_counts.json"))
    ap.add_argument("--tracks", default=str(REPORTS_DIR / "track_counts.json"))
    ap.add_argument("--out-prefix", default=str(REPORTS_DIR / "counting_eval"))
    ap.add_argument("--gt-variant", choices=VARIANTS, default="exclude_ignored",
                    help="GT counts used in the tables (exclude_ignored matches the tracking protocol)")
    return ap.parse_args()


def main() -> None:
    """Compute the counting metrics of every configuration for each split and write JSON and Markdown."""
    args = parse_args()
    gt_all, pred_all = read_json(args.gt), read_json(args.tracks)
    splits = read_json(REPORTS_DIR / "splits.json")

    for split in args.splits:
        for name, source in (("GT counts", gt_all), ("Predicted counts", pred_all)):
            if split not in source:
                raise SystemExit(f"{name} are missing split '{split}'.")
        seqs: list[str] = list(splits[split])
        cfgs = sorted(pred_all[split])
        if not cfgs:
            raise SystemExit(f"No predicted configurations found for split '{split}'.")
        check_inputs(split, seqs, gt_all[split], pred_all[split], args.gt_variant)

        # Metrics per configuration and class. Each count vector has one number per sequence.
        gt_counts = {cls: [gt_all[split][s][args.gt_variant][cls]["total"] for s in seqs] for cls in CLASSES_AND_ALL}
        pred_counts = {cfg: {cls: [pred_all[split][cfg][s]["counts"][cls]["total"] for s in seqs]
                             for cls in CLASSES_AND_ALL} for cfg in cfgs}
        rows = {cfg: {cls: count_errors(pred_counts[cfg][cls], gt_counts[cls]) for cls in CLASSES_AND_ALL}
                for cfg in cfgs}
        gt_totals = {cls: sum(v) for cls, v in gt_counts.items()}

        # JSON keeps the raw count vectors next to the metrics, plus hashes of the inputs
        dump: dict[str, Any] = {cfg: {cls: {"pred": pred_counts[cfg][cls], "gt": gt_counts[cls], **rows[cfg][cls]}
                                      for cls in CLASSES_AND_ALL} for cfg in cfgs}
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
            print(f"  {cfg}: MAE={r['mae']:.2f} WAPE={fmt_percent(r['wape'])} meanSCE={r['mean_sce']:+.2f}")
        print(f"  -> {out_md}\n  -> {out_json}")


if __name__ == "__main__":
    main()
