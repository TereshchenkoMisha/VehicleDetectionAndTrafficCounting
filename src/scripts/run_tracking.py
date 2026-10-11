"""Run YOLOv8 {n,s} x {ByteTrack, BoT-SORT} on dev/test sequences and save MOT tracks + timings.

Output per configuration: outputs/tracks/<split>/<det>_<trk>_conf<conf>/<seq>.txt  and  timing.json
MOT row: frame,id,x,y,w,h,conf,cls,-1,-1   (cls: 0=car, 1=bus, 2=truck)
"""
import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import torch
import ultralytics
from tqdm import tqdm
from ultralytics import YOLO

from ..data.config import OUT_DIR, REPORTS_DIR, weights_path
from ..tracking.pipeline import (build_tracker, list_frames, load_ignored, run_sequence,
                                 warmup, write_mot)

TRACKS_DIR = Path("outputs/tracks")


def _device_name(device):
    if torch.cuda.is_available() and str(device) != "cpu":
        return torch.cuda.get_device_name(0)
    return platform.processor() or "cpu"


def _find(name, data_dir):
    """splits.json / ignored_regions.json: processed-data dir first, then the committed copy in reports/."""
    for d in (data_dir, REPORTS_DIR):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} not found in {data_dir} or {REPORTS_DIR} - run prepare_dataset "
                            f"or copy it from the data branch (see README)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt"])
    ap.add_argument("--trackers", nargs="+", default=["bytetrack", "botsort"],
                    help="names of configs/<name>.yaml")
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
    args = ap.parse_args()

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    splits = json.loads(_find("splits.json", data).read_text())
    ignored = load_ignored(_find("ignored_regions.json", data))
    seqs = args.seqs or splits[args.split]
    if args.split == "test":
        print("!! TEST split: configs, conf, and counting line must be frozen on dev before this run.")

    seq_frames = {}
    for s in seqs:
        fr = list_frames(s)
        seq_frames[s] = fr[:args.max_frames] if args.max_frames else fr
    configs = [(m, t) for m in args.models for t in args.trackers]
    total = sum(len(f) for f in seq_frames.values()) * len(configs)
    print(f"[run] {len(configs)} configs x {len(seqs)} seqs, {total} frames, device={device}, "
          f"conf={args.conf}, imgsz={args.imgsz}")

    first = cv2.imread(str(next(iter(seq_frames.values()))[0][1]))
    bar = tqdm(total=total, unit="frame", dynamic_ncols=True)
    for weights, trk_name in configs:
        det_name = Path(weights).stem
        cfg_name = f"{det_name}_{trk_name}_conf{args.conf:g}"
        out_dir = Path(args.out) / args.split / cfg_name
        model = YOLO(weights_path(weights))
        warmup(model, first, args.imgsz, device)
        first_seq = next(iter(seq_frames))                         # also warm up the tracker (GMC, lap, ...)
        run_sequence(model, build_tracker(trk_name, device), seq_frames[first_seq][:10],
                     ignored.get(first_seq, []), args.conf, args.imgsz, args.iou, device)

        per_seq, tot_f, tot_det, tot_trk = {}, 0, 0.0, 0.0
        for seq, frames in seq_frames.items():
            bar.set_description(f"{det_name}+{trk_name} {seq}")
            writer = None
            if args.vis_dir:
                Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
                h, w = first.shape[:2]
                writer = cv2.VideoWriter(str(Path(args.vis_dir) / f"{cfg_name}_{seq}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
            tracker = build_tracker(trk_name, device)              # fresh tracker + ids per sequence
            n = {"i": 0}

            def tick(d, t, n=n):
                n["i"] += 1
                bar.update(1)
                bar.set_postfix(det_ms=f"{1000 * d / n['i']:.1f}", trk_ms=f"{1000 * t / n['i']:.1f}")

            rows, det_s, trk_s = run_sequence(model, tracker, frames, ignored.get(seq, []),
                                              args.conf, args.imgsz, args.iou, device,
                                              on_frame=tick, writer=writer)
            if writer:
                writer.release()
            write_mot(rows, out_dir / f"{seq}.txt")
            nf = len(frames)
            per_seq[seq] = {"frames": nf, "det_ms": 1000 * det_s / nf, "trk_ms": 1000 * trk_s / nf,
                            "tracks": len({r[1] for r in rows}), "rows": len(rows)}
            tot_f, tot_det, tot_trk = tot_f + nf, tot_det + det_s, tot_trk + trk_s

        timing = {
            "detector": weights, "tracker": trk_name, "split": args.split,
            "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
            "frames": tot_f,
            "det_ms_per_frame": 1000 * tot_det / tot_f,
            "trk_ms_per_frame": 1000 * tot_trk / tot_f,
            "total_ms_per_frame": 1000 * (tot_det + tot_trk) / tot_f,
            "fps": tot_f / (tot_det + tot_trk),
            "per_seq": per_seq,
            "env": {"device": _device_name(device), "torch": torch.__version__,
                    "ultralytics": ultralytics.__version__, "python": platform.python_version(),
                    "date": time.strftime("%Y-%m-%d %H:%M:%S")},
            "note": "image reading excluded; detector = predict (pre+infer+NMS+GPU->CPU); "
                    "tracker = ignored-region mask + tracker.update",
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "timing.json").write_text(json.dumps(timing, indent=2))
        tqdm.write(f"[done] {cfg_name}: det {timing['det_ms_per_frame']:.1f} ms, "
                   f"trk {timing['trk_ms_per_frame']:.1f} ms, {timing['fps']:.1f} FPS -> {out_dir}")
    bar.close()


if __name__ == "__main__":
    main()
