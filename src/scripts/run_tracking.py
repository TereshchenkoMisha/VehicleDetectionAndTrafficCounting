import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils import YAML

from ..data.config import OUT_DIR, RAW_ROOT, REPORTS_DIR, SEED, weights_path
from ..reproducibility import set_seed
from ..data.provenance import file_info, write_json
from ..tracking.pipeline import (build_tracker, list_frames, load_ignored, read_frame, run_sequence, warmup, write_mot)

TRACKS_DIR = Path("outputs/tracks")


def _device_name(device):
    if torch.cuda.is_available() and str(device) != "cpu":
        selected = int(device) if str(device).isdigit() else device
        return torch.cuda.get_device_name(selected)
    return platform.processor() or "cpu"


def _find(name, data_dir):
    # Locate metadata in the dataset directory before reports
    for d in (data_dir, REPORTS_DIR):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} not found in {data_dir} or {REPORTS_DIR} - run prepare_dataset "
                            f"or copy it from reports/ (see README, section 3)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt"])
    ap.add_argument("--trackers", nargs="+", choices=["bytetrack", "botsort"], default=["bytetrack", "botsort"],
                    help="names of configs/<name>.yaml")
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--seqs", nargs="+", default=None, help="subset of sequences (default: whole split)")
    ap.add_argument("--data", default=str(OUT_DIR), help="processed-data dir with splits.json and ignored_regions.json (fallback: reports/)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7, help="detector NMS IoU")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    ap.add_argument("--max-frames", type=int, default=None, help="debug: limit frames per sequence")
    ap.add_argument("--vis-dir", default=None, help="if set, write annotated mp4 per sequence there")
    ap.add_argument("--out", default=str(TRACKS_DIR))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if not 0 <= args.seed < 2**31:
        ap.error("--seed must be between 0 and 2**31 - 1")

    if args.max_frames is not None and args.max_frames < 1:
        ap.error("--max-frames must be a positive integer")

    if args.imgsz < 1:
        ap.error("--imgsz must be a positive integer")

    if not 0.0 <= args.conf <= 1.0:
        ap.error("--conf must be between 0 and 1")

    if not 0.0 <= args.iou <= 1.0:
        ap.error("--iou must be between 0 and 1")

    set_seed(args.seed)
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    splits = json.loads(_find("splits.json", data).read_text())
    ignored = load_ignored(_find("ignored_regions.json", data))

    split_sequences = list(splits[args.split])
    seqs = args.seqs if args.seqs is not None else split_sequences

    if set(seqs) - set(ignored):
        ap.error(f"Missing ignored region entries for {sorted(set(seqs) - set(ignored))}")

    if not seqs:
        ap.error(f"No sequences selected for split '{args.split}'")

    if len(seqs) != len(set(seqs)):
        ap.error("--seqs must not contain duplicate sequence names")

    unknown_sequences = sorted(set(seqs) - set(split_sequences))

    if unknown_sequences:
        ap.error(
            f"Sequences do not belong to split '{args.split}': "
            + ", ".join(unknown_sequences)
        )

    if args.split == "test":
        print("TEST split: configs, conf, and counting line must be frozen on dev before this run")

    seq_frames = {}
    seq_shapes = {}
    first = None

    for seq in seqs:
        frames = list_frames(seq, args.raw)

        if not frames:
            ap.error(f"No frames found for sequence '{seq}'")

        selected_frames = (
            frames
            if args.max_frames is None
            else frames[:args.max_frames]
        )

        sample = read_frame(selected_frames[0][1])

        seq_frames[seq] = selected_frames
        seq_shapes[seq] = sample.shape[:2]

        if first is None:
            first = sample

    configs = [(m, t) for m in args.models for t in args.trackers]

    # Define output directories before running models
    out_dirs = [
        Path(args.out)
        / args.split
        / f"{Path(weights).stem}_{tracker}_conf{args.conf:g}"
        for weights, tracker in configs
    ]

    # Require a distinct output directory for each configuration
    if len(set(out_dirs)) != len(out_dirs):
        ap.error(
            "Some configurations have the same output directory. "
            "Check model filenames and duplicated model/tracker arguments."
        )

    # Preserve existing tracking results
    existing_dirs = [directory for directory in out_dirs if directory.exists()]

    if existing_dirs:
        paths = "\n".join(f"  {directory}" for directory in existing_dirs)
        ap.error(
            "Output directories already exist:\n"
            f"{paths}\n"
            "Choose a new --out directory for this run."
        )

    # Preserve existing visualization files
    if args.vis_dir:
        existing_videos = [
            Path(args.vis_dir) / f"{directory.name}_{seq}.mp4"
            for directory in out_dirs
            for seq in seq_frames
            if (
                Path(args.vis_dir) / f"{directory.name}_{seq}.mp4"
            ).exists()
        ]

        if existing_videos:
            paths = "\n".join(f"  {video}" for video in existing_videos)
            ap.error(
                "Visualization files already exist:\n"
                f"{paths}\n"
                "Choose a new --vis-dir directory for this run."
            )

    # Reserve directories before starting computations
    # Reject concurrent reservations of the same directory
    for directory in out_dirs:
        directory.mkdir(parents=True, exist_ok=False)
    
    total = sum(len(f) for f in seq_frames.values()) * len(configs)
    print(f"[run] {len(configs)} configs x {len(seqs)} seqs, {total} frames, device={device}, "
          f"conf={args.conf}, imgsz={args.imgsz}")

    bar = tqdm(total=total, unit="frame", dynamic_ncols=True)
    for (weights, trk_name), out_dir in zip(configs, out_dirs):
        det_name = Path(weights).stem
        cfg_name = out_dir.name
        resolved_weights = weights_path(weights)
        set_seed(args.seed)
        model = YOLO(resolved_weights)
        provenance = {"weights": file_info(resolved_weights),
                      "tracker_config": file_info(Path("configs") / f"{trk_name}.yaml"),
                      "splits": file_info(_find("splits.json", data)),
                      "ignored": file_info(_find("ignored_regions.json", data)),
                      "ignored_regions": {seq: ignored[seq] for seq in seqs},
                      "raw_root": str(Path(args.raw).resolve()),
                      "data_root": str(data.resolve()), "max_frames": args.max_frames,
                      "tracker_settings": YAML.load(Path("configs") / f"{trk_name}.yaml")}
        warmup(model, first, args.imgsz, device)
        first_seq = next(iter(seq_frames))
        run_sequence(model, build_tracker(trk_name, device), seq_frames[first_seq][:10],
                     ignored.get(first_seq, []), args.conf, args.imgsz, args.iou, device)

        per_seq, tot_f, tot_det, tot_trk = {}, 0, 0.0, 0.0
        for seq, frames in seq_frames.items():
            bar.set_description(f"{det_name}+{trk_name} {seq}")
            
            writer = None
            n = {"i": 0}

            def tick(d, t, n=n):
                n["i"] += 1
                bar.update(1)
                bar.set_postfix(
                    det_ms=f"{1000 * d / n['i']:.1f}",
                    trk_ms=f"{1000 * t / n['i']:.1f}",
                )

            try:
                set_seed(args.seed)
                tracker = build_tracker(trk_name, device)

                if args.vis_dir:
                    vis_dir = Path(args.vis_dir)
                    vis_dir.mkdir(parents=True, exist_ok=True)

                    h, w = seq_shapes[seq]
                    video_path = vis_dir / f"{cfg_name}_{seq}.mp4"

                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        25,
                        (w, h),
                    )

                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Cannot open video writer: {video_path}"
                        )

                rows, det_s, trk_s = run_sequence(
                    model,
                    tracker,
                    frames,
                    ignored.get(seq, []),
                    args.conf,
                    args.imgsz,
                    args.iou,
                    device,
                    on_frame=tick,
                    writer=writer,
                )
            finally:
                if writer is not None:
                    writer.release()

            write_mot(rows, out_dir / f"{seq}.txt")
            nf = len(frames)
            per_seq[seq] = {"frames": nf, "det_ms": 1000 * det_s / nf, "trk_ms": 1000 * trk_s / nf,
                            "tracks": len({r[1] for r in rows}), "rows": len(rows),
                            "mot_sha256": file_info(out_dir / f"{seq}.txt")["sha256"]}
            tot_f, tot_det, tot_trk = tot_f + nf, tot_det + det_s, tot_trk + trk_s

        timing = {
            "detector": weights, "tracker": trk_name, "split": args.split,
            "seed": args.seed, "deterministic": True,
            "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
            "frames": tot_f,
            "det_ms_per_frame": 1000 * tot_det / tot_f,
            "trk_ms_per_frame": 1000 * tot_trk / tot_f,
            "total_ms_per_frame": 1000 * (tot_det + tot_trk) / tot_f,
            "fps": tot_f / (tot_det + tot_trk),
            "per_seq": per_seq,
            "provenance": provenance,
            "env": {"device": _device_name(device), "torch": torch.__version__,
                    "ultralytics": ultralytics.__version__, "opencv": cv2.__version__, "python": platform.python_version(), "cuda": torch.version.cuda,
                    "numpy": np.__version__, "cudnn": torch.backends.cudnn.version(),
                    "date": time.strftime("%Y-%m-%d %H:%M:%S")},
            "note": "image reading excluded; detector = predict (pre+infer+NMS+GPU->CPU); "
                    "tracker = ignored-region mask + tracker.update",
        }
        write_json(out_dir / "timing.json", timing)
        tqdm.write(f"[done] {cfg_name}: det {timing['det_ms_per_frame']:.1f} ms, "
                   f"trk {timing['trk_ms_per_frame']:.1f} ms, {timing['fps']:.1f} FPS -> {out_dir}")
    bar.close()


if __name__ == "__main__":
    main()
