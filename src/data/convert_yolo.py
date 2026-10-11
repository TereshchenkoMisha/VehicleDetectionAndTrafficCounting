import json, shutil
from pathlib import Path
from PIL import Image

from .config import CLASSES, DETRAC_TO_ID, RAW_ROOT, OUT_DIR
from .detrac_parser import find_sequence_dir


def _resolve_naming(seq_dir: Path, frame_ids: list) -> int:
    best, best_hits = 0, -1
    for off in (0, 1):
        hits = sum((seq_dir / f"IMG{fid + off:05d}.jpg").exists() for fid in frame_ids[:50])
        if hits > best_hits:
            best, best_hits = off, hits
    return best


def _yolo_line(cls_id, b, W, H):
    x1, y1, x2, y2 = b.as_xyxy()
    x1, x2 = max(0.0, min(x1, W)), max(0.0, min(x2, W))
    y1, y2 = max(0.0, min(y1, H)), max(0.0, min(y2, H))
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return None
    return f"{cls_id} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {w/W:.6f} {h/H:.6f}"


def convert(anns, split, frame_step=1, copy=False, out_dir=OUT_DIR, frames_root=RAW_ROOT):
    stats, ignored_out = {"dev": {}, "test": {}}, {}
    for split_name in ("dev", "test"):
        for seq_name in split[split_name]:
            ann = anns[seq_name]
            seq_dir = find_sequence_dir(frames_root, seq_name)
            if seq_dir is None:
                raise FileNotFoundError(f"frames {seq_name} not found under {frames_root}")
            img_out = out_dir / "images" / split_name
            lbl_out = out_dir / "labels" / split_name
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            offset = _resolve_naming(seq_dir, sorted(ann.frames))
            per_class = {c: 0 for c in CLASSES}
            n_frames = 0
            for fid in sorted(ann.frames):
                if fid % frame_step:
                    continue
                img_path = seq_dir / f"IMG{fid + offset:05d}.jpg"
                if not img_path.exists():
                    continue
                stem = f"{seq_name}_f{fid:06d}"
                W, H = Image.open(img_path).size
                lines = []
                for t in ann.frames[fid].targets:
                    cid = DETRAC_TO_ID.get(t.cls)
                    if cid is None:
                        continue
                    line = _yolo_line(cid, t.box, W, H)
                    if line:
                        lines.append(line)
                        per_class[CLASSES[cid]] += 1
                (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
                dst = img_out / f"{stem}.jpg"
                if copy:
                    shutil.copy2(img_path, dst)
                else:
                    try:
                        dst.symlink_to(img_path.resolve())
                    except OSError:
                        shutil.copy2(img_path, dst)
                n_frames += 1
            if n_frames == 0:
                raise RuntimeError(f"{seq_name}: 0 frames — check the raw dataset layout")

            stats[split_name][seq_name] = {"weather": split["weather"][seq_name],
                                           "frames": n_frames, "boxes": per_class}
            ignored_out[seq_name] = [list(r.as_xyxy()) for r in ann.ignored_regions]

    (out_dir / "detrac.yaml").write_text(
        f"path: {out_dir.resolve()}\n"
        "train: images/dev   # not used (fine-tuning is not planned)\n"
        "val: images/dev\n"
        "test: images/test\n"
        "names:\n  0: car\n  1: bus\n  2: truck\n")
    (out_dir / "ignored_regions.json").write_text(json.dumps(ignored_out))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats