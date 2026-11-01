"""Run detector x tracker configurations on full-rate frames: MOT tracks and timing per configuration.

    python scripts/run_tracking.py --split dev
    python scripts/run_tracking.py --split dev --seqs MVI_20033 --max-frames 100 --vis-dir outputs/vis

Every combination of --models and --trackers is one configuration. For each configuration the script
detects and tracks every frame of every sequence of the split, timing the detector and the tracker
separately, and writes the tracks plus a timing report.

Needs:
    data/raw/UA-DETRAC/                    frames
    reports/splits.json                    sequences of the split (data/processed/detrac_yolo/ is preferred)
    reports/ignored_regions.json           ignored regions (same lookup)
    configs/{bytetrack,botsort}.yaml       tracker settings
    models/<weights>.pt                    downloaded on first use
Produces, per configuration, in <out>/<split>/<detector>_<tracker>_conf<conf>/:
    <sequence>.txt                         MOT rows: frame,id,x,y,w,h,score,class,-1,-1
    timing.json                            ms per frame, FPS, settings, hashes of inputs, environment
    <vis-dir>/<configuration>_<sequence>.mp4   with --vis-dir
A configuration is written to a `.partial` folder and renamed when finished. Seeds are always set;
--deterministic also makes the GPU path repeatable but slower, so it is not meant for timing.
"""
import argparse
import platform
import shutil
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.config import (CONF, CONFIG_DIR, IMGSZ, NMS_IOU, OUT_DIR, RAW_ROOT, REPORTS_DIR, SEED, TRACKS_DIR,
                        weights_path)
from src.data.sequences import list_frames
from src.tracking.pipeline import track_config
from src.utils import (env_info, file_info, fraction, positive_int, read_json, seed_value, set_deterministic,
                       set_seed, write_json)


def find_metadata(name: str, data_dir: Path) -> Path:
    """Locate a metadata file: in the processed-data folder first, then the committed copy in reports/.

    The processed folder is only there after `prepare_dataset`; the committed copy lets tracking run
    on a fresh clone with the raw dataset alone.
    """
    for directory in (data_dir, REPORTS_DIR):
        if (directory / name).exists():
            return directory / name
    raise FileNotFoundError(f"{name} not found in {data_dir} or {REPORTS_DIR} - run prepare_dataset "
                            "or copy it from reports/ (see README, section 3)")


def device_name(device: str | int) -> str:
    """Human-readable name of the compute device, for the timing report."""
    if torch.cuda.is_available() and str(device) != "cpu":
        return torch.cuda.get_device_name(int(device) if str(device).isdigit() else device)
    return platform.processor() or "cpu"


def parse_args() -> argparse.Namespace:
    """Command-line options; see the module docstring for what the script reads and writes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt"])
    ap.add_argument("--trackers", nargs="+", choices=["bytetrack", "botsort"], default=["bytetrack", "botsort"],
                    help="names of configs/<name>.yaml")
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--seqs", nargs="+", default=None, help="subset of sequences (default: whole split)")
    ap.add_argument("--data", default=str(OUT_DIR), help="processed-data folder (fallback: reports/)")
    ap.add_argument("--conf", type=fraction, default=CONF,
                    help="tracker high-score threshold; the detector itself runs at track_low_thresh")
    ap.add_argument("--iou", type=fraction, default=NMS_IOU, help="detector NMS IoU")
    ap.add_argument("--imgsz", type=positive_int, default=IMGSZ)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    ap.add_argument("--max-frames", type=positive_int, default=None, help="debug: limit frames per sequence")
    ap.add_argument("--vis-dir", default=None, help="write an annotated mp4 per sequence there")
    ap.add_argument("--out", default=str(TRACKS_DIR))
    ap.add_argument("--seed", type=seed_value, default=SEED)
    ap.add_argument("--deterministic", action="store_true", help="repeatable GPU results, slower; not for timing")
    return ap.parse_args()


def main() -> None:
    """Validate the request, then run every configuration and save its tracks and timing."""
    args = parse_args()
    set_seed(args.seed)
    if args.deterministic:
        set_deterministic()
    device: str | int = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")

    # Which sequences to run: the whole split, or a subset that must belong to it. Each needs
    # ignored regions, otherwise the filter would silently do nothing for it.
    data = Path(args.data)
    splits_path, ignored_path = find_metadata("splits.json", data), find_metadata("ignored_regions.json", data)
    ignored = read_json(ignored_path)
    split_sequences = read_json(splits_path)[args.split]
    seqs: list[str] = args.seqs if args.seqs is not None else list(split_sequences)
    if not seqs or len(seqs) != len(set(seqs)):
        raise SystemExit("--seqs must be non-empty and contain no duplicates")
    if set(seqs) - set(split_sequences):
        raise SystemExit(f"Sequences do not belong to split '{args.split}': {sorted(set(seqs) - set(split_sequences))}")
    if set(seqs) - set(ignored):
        raise SystemExit(f"Missing ignored region entries for {sorted(set(seqs) - set(ignored))}")
    if args.split == "test":
        print("TEST split: configs, conf and counting protocol must be frozen on dev before this run")
    seq_frames = {seq: list_frames(seq, args.raw)[:args.max_frames] for seq in seqs}

    # Output folders. Everything is checked before the first frame is processed, so a clash does not
    # cost a long run: results are never overwritten.
    configs = [(m, t) for m in args.models for t in args.trackers]
    out_dirs = [Path(args.out) / args.split / f"{Path(w).stem}_{t}_conf{args.conf:g}" for w, t in configs]
    if len(set(out_dirs)) != len(out_dirs):
        raise SystemExit("Some configurations have the same output folder. Check duplicated model/tracker arguments.")
    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    existing = [d for d in out_dirs if d.exists()]
    if vis_dir:
        existing += [v for d in out_dirs for seq in seqs if (v := vis_dir / f"{d.name}_{seq}.mp4").exists()]
    if existing:
        raise SystemExit("Output already exists, choose a new --out / --vis-dir:\n" + "\n".join(f"  {p}" for p in existing))

    total = sum(len(f) for f in seq_frames.values()) * len(configs)
    print(f"[run] {len(configs)} configs x {len(seqs)} seqs, {total} frames, device={device}, "
          f"conf={args.conf}, imgsz={args.imgsz}")

    # One progress bar over all frames of all configurations. It shows the running mean detector and
    # tracker time of the current sequence.
    with tqdm(total=total, unit="frame", dynamic_ncols=True) as bar:
        def on_frame(seq: str, index: int, det_s: float, trk_s: float) -> None:
            bar.update(1)
            bar.set_postfix(seq=seq, det_ms=f"{1000 * det_s / index:.1f}", trk_ms=f"{1000 * trk_s / index:.1f}")

        for (weights, tracker), out_dir in zip(configs, out_dirs, strict=True):
            # Work in `<name>.partial` and rename at the end: a crash leaves no folder under the final
            # name, so the next run is not blocked by a half-finished result.
            partial = out_dir.with_name(out_dir.name + ".partial")
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir(parents=True)
            bar.set_description(f"{Path(weights).stem}+{tracker}")

            timing: dict[str, Any] = track_config(
                weights, tracker, seq_frames, ignored, partial, conf=args.conf, imgsz=args.imgsz, iou=args.iou,
                device=device, seed=args.seed, on_frame=on_frame, vis_dir=vis_dir)

            # Record which inputs and software produced these tracks, so the result can be traced and
            # a changed input detected later
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
