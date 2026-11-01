"""Select 12 sequences, split them into dev/test and export YOLO labels for detector evaluation.

    python scripts/prepare_dataset.py --frame-step 5 --copy
"""
import argparse
from pathlib import Path

import pandas as pd

from src.config import N_DEV, N_SELECT, OUT_DIR, RAW_ROOT
from src.data.detrac import load_all
from src.data.export import convert
from src.data.sequences import find_sequence_dir
from src.data.split import build_proxy_metas, save_or_load_split, select_and_split
from src.utils import positive_int, read_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--n-select", type=positive_int, default=N_SELECT)
    ap.add_argument("--n-dev", type=positive_int, default=N_DEV)
    ap.add_argument("--frame-step", type=positive_int, default=1)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    ap.add_argument("--force", action="store_true", help="re-draw the split (never after the split is fixed)")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--split-file", default=None, help="use an existing fixed split JSON")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if (out_dir / "images").exists() or (out_dir / "labels").exists():
        ap.error(f"Prepared images or labels already exist in {out_dir}. Choose a new --out directory.")

    anns, metas = load_all(args.raw)
    both = sorted(set(anns) & set(metas)) if metas else sorted(anns)
    with_frames = [n for n in both if find_sequence_dir(args.raw, n) is not None]
    print(f"[data] annotations: {len(anns)}, metadata: {len(metas)}, with frames: {len(with_frames)}")

    if metas:
        metas = {n: metas[n] for n in with_frames}
    else:
        print("[data] DETRAC-MetaData not found - using brightness-based day/night proxy")
        metas = build_proxy_metas({n: anns[n] for n in with_frames}, args.raw)

    cached_split = out_dir / "splits.json"
    if args.split_file is not None:
        requested = read_json(args.split_file)
    elif cached_split.is_file() and not args.force:
        requested = read_json(cached_split)
    else:
        requested = select_and_split(metas, args.n_select, args.n_dev)

    split = save_or_load_split(out_dir, requested, force=args.force)
    if split != requested:
        ap.error("The existing output split differs from the requested split. Choose a new --out directory.")
    print("[split] dev:", split["dev"])
    print("[split] test:", split["test"])

    stats = convert({n: anns[n] for n in with_frames}, split, frame_step=args.frame_step, copy=args.copy,
                    out_dir=out_dir, frames_root=Path(args.raw))
    rows = [{"split": s, "seq": q, "weather": v["weather"], "frames": v["frames"], **v["boxes"]}
            for s, d in stats.items() for q, v in d.items()]
    print(pd.DataFrame(rows).to_string(index=False))
    if sum(r["truck"] for r in rows if r["split"] == "test") == 0:
        print("!! 0 truck boxes in test - per-class truck numbers are impossible. "
              "Re-selection is allowed ONLY now, before any experiment (--force)")


if __name__ == "__main__":
    main()
