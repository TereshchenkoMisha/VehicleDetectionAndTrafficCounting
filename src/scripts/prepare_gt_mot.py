"""Подготовка GT MOT-файлов для TrackEval.

Читает reports/splits.json, конвертирует все 12 последовательностей dev+test
в reports/gt_mot/<seq>.txt (формат MOTChallenge).

Использование:
    python -m src.scripts.prepare_gt_mot              # dev + test
    python -m src.scripts.prepare_gt_mot --splits dev
"""
import argparse
import json
from pathlib import Path

from ..data.config import REPORTS_DIR
from ..data.convert_mot import convert_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--raw", default="data/raw/UA-DETRAC")
    ap.add_argument("--out", default=str(REPORTS_DIR / "gt_mot"))
    ap.add_argument("--ignored", default=str(REPORTS_DIR / "ignored_regions.json"))
    args = ap.parse_args()

    splits = json.loads((REPORTS_DIR / "splits.json").read_text())
    sequences = []
    for s in args.splits:
        sequences.extend(splits[s])
    print(f"[prepare] {len(sequences)} sequences -> {args.out}")
    convert_all(sequences, args.out, raw_root=args.raw, ignored_path=args.ignored)
    print(f"[done] GT MOT files in {args.out}")


if __name__ == "__main__":
    main()