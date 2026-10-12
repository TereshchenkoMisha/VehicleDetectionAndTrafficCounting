import argparse

import pandas as pd

from ..data.config import N_SELECT, N_DEV, OUT_DIR
from ..data.detrac_parser import load_all, find_sequence_dir
from ..data.split import select_and_split, save_or_load_split
from ..data.convert_yolo import convert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/UA-DETRAC")
    ap.add_argument("--n-select", type=int, default=N_SELECT)
    ap.add_argument("--n-dev", type=int, default=N_DEV)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    anns, metas = load_all(args.raw)
    both = sorted(set(anns) & set(metas)) if metas else sorted(anns)
    with_frames = [n for n in both if find_sequence_dir(args.raw, n) is not None]
    print(f"[data] annotations: {len(anns)}, metadata: {len(metas)}, with frames: {len(with_frames)}")

    if not metas:
        print("[data] DETRAC-MetaData not found - using brightness-based day/night proxy")
        from ..data.metadata_proxy import build_proxy_metas
        metas = build_proxy_metas({n: anns[n] for n in with_frames}, args.raw)
    else:
        metas = {n: metas[n] for n in with_frames}

    split = save_or_load_split(OUT_DIR,
                               select_and_split(metas, args.n_select, args.n_dev),
                               force=args.force)
    print("[split] dev:", split["dev"])
    print("[split] test:", split["test"])

    stats = convert({n: anns[n] for n in with_frames}, split,
                    frame_step=args.frame_step, copy=args.copy)
    rows = [{"split": s, "seq": q, "weather": v["weather"], "frames": v["frames"], **v["boxes"]}
            for s, d in stats.items() for q, v in d.items()]
    print(pd.DataFrame(rows).to_string(index=False))
    tr = sum(r["truck"] for r in rows if r["split"] == "test")
    if tr == 0:
        print("!! 0 truck boxes in test - per-class truck impossible. "
              "Re-selection allowed ONLY now, before experiments (--force)")


if __name__ == "__main__":
    main()