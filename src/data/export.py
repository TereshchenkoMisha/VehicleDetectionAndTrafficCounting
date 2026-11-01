"""Export the selected sequences: YOLO labels + images for detector evaluation, and GT in MOT format."""
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ..config import CLASSES, DETRAC_TO_ID, OUT_DIR, RAW_ROOT, REPORTS_DIR
from ..utils import read_json, write_json
from .detrac import load_all
from .regions import inside_any
from .sequences import find_sequence_dir, frame_paths, resolve_naming


# ---------- YOLO labels (detector evaluation) ----------

def _yolo_line(cls_id, box, width, height):
    x1, y1, x2, y2 = box.as_xyxy()
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return None
    return f"{cls_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {w / width:.6f} {h / height:.6f}"


def _plan_frames(anns, split, frames_root, frame_step):
    """{(split_name, seq): [(xml_frame_number, image_path), ...]} keeping every frame_step-th frame."""
    plans = {}
    for split_name in ("dev", "test"):
        for seq_name in split[split_name]:
            seq_dir = find_sequence_dir(frames_root, seq_name)
            if seq_dir is None:
                raise FileNotFoundError(f"Frames of {seq_name} not found under {frames_root}")
            offset = resolve_naming(seq_dir, sorted(anns[seq_name].frames))
            selected = [(image_id - offset, path) for image_id, path in frame_paths(seq_dir).items()
                        if (image_id - offset) % frame_step == 0]
            if not selected:
                raise ValueError(f"No frames selected for {seq_name} with frame_step={frame_step}")
            plans[(split_name, seq_name)] = selected
    return plans


def _place_image(src, dst, copy):
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except FileExistsError:      # FileExistsError is an OSError, so it must be re-raised before the fallback
        raise
    except OSError:              # symlinks need Developer Mode on Windows
        shutil.copy2(src, dst)


def convert(anns, split, frame_step=1, copy=False, out_dir=OUT_DIR, frames_root=RAW_ROOT):
    """Write images/<split>, labels/<split> (YOLO), detrac.yaml, ignored_regions.json and stats.json."""
    out_dir, frames_root = Path(out_dir), Path(frames_root)
    if frame_step < 1:
        raise ValueError("frame_step must be a positive integer")
    if (out_dir / "images").exists() or (out_dir / "labels").exists():
        raise FileExistsError(f"Prepared images or labels already exist in {out_dir}. "
                              "Choose a new output directory.")

    plans = _plan_frames(anns, split, frames_root, frame_step)
    stats, ignored_out = {"dev": {}, "test": {}}, {}
    bar = tqdm(total=sum(len(frames) for frames in plans.values()), unit="frame", desc="convert")
    for (split_name, seq_name), frames in plans.items():
        bar.set_postfix(seq=seq_name)
        ann = anns[seq_name]
        img_out = out_dir / "images" / split_name
        lbl_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        per_class = {c: 0 for c in CLASSES}
        for fid, img_path in frames:
            stem = f"{seq_name}_f{fid:06d}"
            with Image.open(img_path) as image:
                width, height = image.size
            frame_ann = ann.frames.get(fid)
            lines = []
            for target in frame_ann.targets if frame_ann is not None else []:
                cid = DETRAC_TO_ID.get(target.cls)
                line = None if cid is None else _yolo_line(cid, target.box, width, height)
                if line:
                    lines.append(line)
                    per_class[CLASSES[cid]] += 1
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
            _place_image(img_path, img_out / f"{stem}.jpg", copy)
            bar.update(1)

        stats[split_name][seq_name] = {"weather": split["weather"][seq_name],
                                       "frames": len(frames), "boxes": per_class}
        ignored_out[seq_name] = [list(r.as_xyxy()) for r in ann.ignored_regions]
    bar.close()

    (out_dir / "detrac.yaml").write_text(
        f"path: {out_dir.resolve()}\n"
        "train: images/dev   # not used (fine-tuning is not planned)\n"
        "val: images/dev\n"
        "test: images/test\n"
        "names:\n  0: car\n  1: bus\n  2: truck\n")
    write_json(out_dir / "ignored_regions.json", ignored_out)
    write_json(out_dir / "stats.json", stats)
    return stats


# ---------- GT in MOT format (tracking evaluation) ----------

def write_gt_mot(ann, seq_dir, ignored_regions, out_path: Path):
    """Write GT boxes with original target IDs and local class IDs; boxes with centres in ignored regions are dropped."""
    frame_ids = sorted(ann.frames)
    offset = resolve_naming(seq_dir, frame_ids)
    rows = []
    for fid in frame_ids:
        targets = [t for t in ann.frames[fid].targets if t.cls in DETRAC_TO_ID]
        if not targets:
            continue
        ignored = inside_any([t.box.centre() for t in targets], ignored_regions)
        for t, skip in zip(targets, ignored):
            if skip:
                continue
            rows.append(f"{fid + offset},{t.track_id},"
                        f"{t.box.x:.2f},{t.box.y:.2f},{t.box.w:.2f},{t.box.h:.2f},"
                        f"1.0,{DETRAC_TO_ID[t.cls]},1.0,-1")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""))


def convert_all(sequences, out_dir, raw_root=RAW_ROOT, ignored_path=None):
    """Write GT MOT files for the sequences. Checks every input first, so a failure writes nothing."""
    ignored = read_json(ignored_path or REPORTS_DIR / "ignored_regions.json")
    anns, _ = load_all(raw_root)
    out_dir = Path(out_dir)
    plans = []
    for seq in sequences:
        if seq not in anns:
            raise FileNotFoundError(f"No XML annotations for {seq}")
        seq_dir = find_sequence_dir(raw_root, seq)
        if seq_dir is None:
            raise FileNotFoundError(f"No source frames for {seq}")
        if seq not in ignored:
            raise ValueError(f"Missing ignored region entry for {seq}")
        resolve_naming(seq_dir, sorted(anns[seq].frames))
        plans.append((seq, seq_dir, ignored[seq]))

    for seq, seq_dir, regions in plans:
        write_gt_mot(anns[seq], seq_dir, regions, out_dir / f"{seq}.txt")
        print(f"[gt-mot] {seq}: written")
    return out_dir
