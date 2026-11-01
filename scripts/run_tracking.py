"""Run detector x tracker configurations on full-rate frames and write MOT tracks + timing.json.

    python scripts/run_tracking.py --split dev
    python scripts/run_tracking.py --split dev --seqs MVI_20033 --max-frames 100 --vis-dir outputs/vis

Speed runs use plain seeds. Add --deterministic for bit-repeatable GPU results (slower: do not time with it).
"""
import argparse
import platform
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from src.config import (CONF, CONFIG_DIR, IMGSZ, NMS_IOU, OUT_DIR, RAW_ROOT, REPORTS_DIR, SEED, TRACKS_DIR,
                         weights_path)
from src.data.sequences import list_frames
from src.tracking.pipeline import track_config
from src.utils import (env_info, file_info, fraction, positive_int, read_json, seed_value, set_deterministic,
                        set_seed, write_json)


def find_metadata(name, data_dir):
    """Metadata file from the processed-data directory, falling back to the committed copy in reports/."""
    for directory in (data_dir, REPORTS_DIR):
        if (directory / name).exists():
            return directory / name
    raise FileNotFoundError(f"{name} not found in {data_dir} or {REPORTS_DIR} - run prepare_dataset "
                            "or copy it from reports/ (see README, section 3)")


def device_name(device):
    if torch.cuda.is_available() and str(device) != "cpu":
        return torch.cuda.get_device_name(int(device) if str(device).isdigit() else device)
    return platform.processor() or "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt"])
    ap.add_argument("--trackers", nargs="+", choices=["bytetrack", "botsort"], default=["bytetrack", "botsort"],
                    help="names of configs/<name>.yaml")
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--seqs", nargs="+", default=None, help="subset of sequences (default: whole split)")
    ap.add_argument("--data", default=str(OUT_DIR), help="processed-data dir (fallback: reports/)")
    ap.add_argument("--conf", type=fraction, default=CONF,
                    help="tracker high-score threshold; the detector itself runs at track_low_thresh")
    ap.add_argument("--iou", type=fraction, default=NMS_IOU, help="detector NMS IoU")
    ap.add_argument("--imgsz", type=positive_int, default=IMGSZ)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    ap.add_argument("--max-frames", type=positive_int, default=None, help="debug: limit frames per sequence")
    ap.add_argument("--vis-dir", default=None, help="write an annotated mp4 per sequence there")
    ap.add_argument("--out", default=str(TRACKS_DIR))
    ap.add_argument("--seed", type=seed_value, default=SEED)
    ap.add_argument("--deterministic", action="store_true", help="bit-repeatable but slower; not for timing")
    args = ap.parse_args()

    set_seed(args.seed)
    if args.deterministic:
        set_deterministic()
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    splits_path, ignored_path = find_metadata("splits.json", data), find_metadata("ignored_regions.json", data)
    ignored = read_json(ignored_path)

    split_sequences = read_json(splits_path)[args.split]
    seqs = args.seqs if args.seqs is not None else list(split_sequences)
    if not seqs or len(seqs) != len(set(seqs)):
        ap.error("--seqs must be non-empty and contain no duplicates")
    if set(seqs) - set(split_sequences):
        ap.error(f"Sequences do not belong to split '{args.split}': {sorted(set(seqs) - set(split_sequences))}")
    if set(seqs) - set(ignored):
        ap.error(f"Missing ignored region entries for {sorted(set(seqs) - set(ignored))}")
    if args.split == "test":
        print("TEST split: configs, conf and counting protocol must be frozen on dev before this run")

    seq_frames = {seq: list_frames(seq, args.raw)[:args.max_frames] for seq in seqs}

    configs = [(m, t) for m in args.models for t in args.trackers]
    out_dirs = [Path(args.out) / args.split / f"{Path(w).stem}_{t}_conf{args.conf:g}" for w, t in configs]
    if len(set(out_dirs)) != len(out_dirs):
        ap.error("Some configurations have the same output directory. Check duplicated model/tracker arguments.")
    existing = [d for d in out_dirs if d.exists()]
    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        existing += [vis_dir / f"{d.name}_{seq}.mp4" for d in out_dirs for seq in seqs
                     if (vis_dir / f"{d.name}_{seq}.mp4").exists()]
    if existing:
        ap.error("Output already exists, choose a new --out / --vis-dir:\n" + "\n".join(f"  {p}" for p in existing))

    total = sum(len(f) for f in seq_frames.values()) * len(configs)
    print(f"[run] {len(configs)} configs x {len(seqs)} seqs, {total} frames, device={device}, "
          f"conf={args.conf}, imgsz={args.imgsz}")

    with tqdm(total=total, unit="frame", dynamic_ncols=True) as bar:
        def on_frame(seq, index, det_s, trk_s):
            bar.update(1)
            bar.set_postfix(seq=seq, det_ms=f"{1000 * det_s / index:.1f}", trk_ms=f"{1000 * trk_s / index:.1f}")

        for (weights, tracker), out_dir in zip(configs, out_dirs):
            # Work in a ".partial" directory so a crash never leaves a half-finished result that blocks re-runs.
            partial = out_dir.with_name(out_dir.name + ".partial")
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir(parents=True)
            bar.set_description(f"{Path(weights).stem}+{tracker}")

            timing = track_config(weights, tracker, seq_frames, ignored, partial, conf=args.conf,
                                  imgsz=args.imgsz, iou=args.iou, device=device, seed=args.seed,
                                  on_frame=on_frame, vis_dir=vis_dir)
            timing.update(
                split=args.split, deterministic=args.deterministic, max_frames=args.max_frames,
                provenance={"weights": file_info(weights_path(weights)),
                            "tracker_config": file_info(CONFIG_DIR / f"{tracker}.yaml"),
                            "splits": file_info(splits_path), "ignored": file_info(ignored_path),
                            "ignored_regions": {seq: ignored[seq] for seq in seqs},
                            "raw_root": str(Path(args.raw).resolve()), "data_root": str(data.resolve())},
                env=env_info(device_name(device)))
            write_json(partial / "timing.json", timing)
            partial.rename(out_dir)
            tqdm.write(f"[done] {out_dir.name}: det {timing['det_ms_per_frame']:.1f} ms, "
                       f"trk {timing['trk_ms_per_frame']:.1f} ms, {timing['fps']:.1f} FPS -> {out_dir}")


if __name__ == "__main__":
    main()
