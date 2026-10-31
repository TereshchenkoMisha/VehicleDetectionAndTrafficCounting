import re, json, shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from .config import CLASSES, DETRAC_TO_ID, RAW_ROOT, OUT_DIR
from .detrac_parser import find_sequence_dir


def _frame_paths(seq_dir: Path) -> dict[int, Path]:
    """Index frame_id -> real image path"""
    frames = {}

    for image_path in Path(seq_dir).iterdir():
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue

        match = re.fullmatch(
            r"img(\d+)",
            image_path.stem,
            flags=re.IGNORECASE,
        )

        if match is None:
            raise ValueError(f"Unexpected frame filename: {image_path}")

        frame_id = int(match.group(1))

        if frame_id in frames:
            raise ValueError(
                f"Duplicate frame ID {frame_id}: "
                f"{frames[frame_id]} and {image_path}"
            )

        frames[frame_id] = image_path

    if not frames:
        raise FileNotFoundError(f"No JPG frames found in {seq_dir}")

    return dict(sorted(frames.items()))


def _resolve_naming(seq_dir: Path, frame_ids: list) -> int:
    """Defines offset XML frame_id -> JPG frame_id"""
    image_ids = set(_frame_paths(seq_dir))
    frame_ids = list(frame_ids)

    if not frame_ids:
        raise ValueError(f"No XML frames for {seq_dir}")

    offsets = [
        offset
        for offset in (0, 1)
        if all(fid + offset in image_ids for fid in frame_ids)
    ]

    if len(offsets) != 1:
        raise ValueError(
            f"Cannot determine XML/JPG offset unambiguously "
            f"for {seq_dir}: candidates={offsets}"
        )

    return offsets[0]


def _yolo_line(cls_id, b, W, H):
    x1, y1, x2, y2 = b.as_xyxy()
    x1, x2 = max(0.0, min(x1, W)), max(0.0, min(x2, W))
    y1, y2 = max(0.0, min(y1, H)), max(0.0, min(y2, H))
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return None
    return f"{cls_id} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {w/W:.6f} {h/H:.6f}"


def convert(anns, split, frame_step=1, copy=False, out_dir=OUT_DIR, frames_root=RAW_ROOT):
    out_dir = Path(out_dir)
    frames_root = Path(frames_root)

    if frame_step < 1:
        raise ValueError("frame_step must be a positive integer")

    # Do not mix the new preparation with the previous one
    if (out_dir / "images").exists() or (out_dir / "labels").exists():
        raise FileExistsError(
            f"Prepared images or labels already exist in {out_dir}. "
            "Choose a new output directory."
        )

    plans = {}

    for split_name in ("dev", "test"):
        for seq_name in split[split_name]:
            ann = anns[seq_name]
            seq_dir = find_sequence_dir(frames_root, seq_name)

            if seq_dir is None:
                raise FileNotFoundError(
                    f"Frames of {seq_name} not found under {frames_root}"
                )

            image_paths = _frame_paths(seq_dir)
            offset = _resolve_naming(seq_dir, sorted(ann.frames))

            selected = [
                (image_id - offset, image_path)
                for image_id, image_path in image_paths.items()
                if (image_id - offset) % frame_step == 0
            ]

            if not selected:
                raise ValueError(
                    f"No frames selected for {seq_name} "
                    f"with frame_step={frame_step}"
                )

            plans[(split_name, seq_name)] = selected

    stats, ignored_out = {"dev": {}, "test": {}}, {}
    total = sum(len(frames) for frames in plans.values())
    bar = tqdm(total=total, unit="frame", desc="convert")
    for split_name in ("dev", "test"):
        for seq_name in split[split_name]:
            bar.set_postfix(seq=seq_name)
            ann = anns[seq_name]

            img_out = out_dir / "images" / split_name
            lbl_out = out_dir / "labels" / split_name
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            per_class = {c: 0 for c in CLASSES}
            n_frames = 0
            for fid, img_path in plans[(split_name, seq_name)]:
                stem = f"{seq_name}_f{fid:06d}"

                with Image.open(img_path) as image:
                    W, H = image.size

                lines = []
                frame_ann = ann.frames.get(fid)
                targets = frame_ann.targets if frame_ann is not None else []

                for t in targets:
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
                    except FileExistsError:
                        raise
                    except OSError:
                        shutil.copy2(img_path, dst)
                n_frames += 1
                bar.update(1)
            if n_frames == 0:
                raise RuntimeError(f"{seq_name}: 0 frames — check the raw dataset layout")

            stats[split_name][seq_name] = {"weather": split["weather"][seq_name],
                                           "frames": n_frames, "boxes": per_class}
            ignored_out[seq_name] = [list(r.as_xyxy()) for r in ann.ignored_regions]

    bar.close()
    (out_dir / "detrac.yaml").write_text(
        f"path: {out_dir.resolve()}\n"
        "train: images/dev   # not used (fine-tuning is not planned)\n"
        "val: images/dev\n"
        "test: images/test\n"
        "names:\n  0: car\n  1: bus\n  2: truck\n")
    (out_dir / "ignored_regions.json").write_text(json.dumps(ignored_out))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats