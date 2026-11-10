"""Convert the selected sequences into the formats the evaluations read.

Two exports live here:
  - YOLO labels plus images, for the detector evaluation. Only every n-th frame is kept, because
    neighbouring frames at 25 fps are near-duplicates. Boxes in ignored regions are kept here (the
    evaluation drops them later); boxes of unmapped classes and boxes thinner than 2 px after
    clipping are skipped without a message.
  - Ground truth in MOT format, for the tracking evaluation. It keeps every frame and drops boxes
    whose centre is in an ignored region.
"""
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from ..config import CLASSES, DETRAC_TO_ID, OUT_DIR, RAW_ROOT, REPORTS_DIR
from ..utils import file_info, publish_directory, read_json, write_json
from .detrac import Box, SequenceAnn, load_all
from .regions import Region, inside_any
from .sequences import JPG_SUFFIXES, find_sequence_dir, frame_paths, resolve_naming

# ---------------------------------------------------------------- YOLO labels


def _yolo_line(cls_id: int, box: Box, width: int, height: int) -> str | None:
    """One YOLO label line: `class cx cy w h`, with centre and size divided by the image size.

    The box is first clipped to the image, because annotations can reach past the border. A box
    thinner than 2 px after clipping is useless as a label and yields None.
    """
    x1, y1, x2, y2 = box.as_xyxy()
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return None
    return f"{cls_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {w / width:.6f} {h / height:.6f}"


def _plan_frames(anns: dict[str, SequenceAnn], split: dict[str, Any], frames_root: Path,
                 frame_step: int) -> dict[tuple[str, str], list[tuple[int, Path]]]:
    """Decide which frames to export: {(split name, sequence): [(XML frame number, image path), ...]}.

    A frame is kept when its XML number is a multiple of `frame_step`. Selecting by XML number
    (not file number) makes the exported frames line up with the annotations. Raises for a missing
    sequence or an empty selection, so nothing is written for a bad plan.
    """
    plans: dict[tuple[str, str], list[tuple[int, Path]]] = {}
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


def _place_image(src: Path, dst: Path, copy: bool) -> None:
    """Put the image at `dst`: a copy, or a symlink that falls back to a copy.

    Symlinks save disk space but need a privilege on Windows (Developer Mode). If creating one is
    refused, the image is copied instead.
    """
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except FileExistsError:     # a subclass of OSError, so it must be re-raised before the copy fallback
        raise
    except OSError:
        shutil.copy2(src, dst)


def _label_lines(ann: SequenceAnn, fid: int, img_path: Path) -> list[str]:
    """Derive labels from the current annotations and actual image dimensions."""
    with Image.open(img_path) as image:
        width, height = image.size
    frame = ann.frames.get(fid)
    lines = []
    for target in frame.targets if frame is not None else []:
        cid = DETRAC_TO_ID.get(target.cls)
        line = None if cid is None else _yolo_line(cid, target.box, width, height)
        if line:
            lines.append(line)
    return lines


def _prepared_is_current(out_dir: Path, plans: dict, anns: dict[str, SequenceAnn], split: dict) -> bool:
    """Reuse only a complete export with the same frames, labels and metadata; repair partial exports."""
    try:
        if read_json(out_dir / "splits.json") != split:
            return False
        stats, ignored = read_json(out_dir / "stats.json"), read_json(out_dir / "ignored_regions.json")
        for name in ("dev", "test"):
            expected = {f"{seq}_f{fid:06d}" for (s, seq), frames in plans.items() if s == name for fid, _ in frames}
            images = {p.name for p in (out_dir / "images" / name).glob("*")
                      if p.is_file() and p.suffix.lower() in JPG_SUFFIXES}
            if images != {f"{stem}.jpg" for stem in expected}:
                return False
            if {p.stem for p in (out_dir / "labels" / name).glob("*.txt")} != expected:
                return False
            if set(stats[name]) != set(split[name]):
                return False
        for (name, seq), frames in plans.items():
            if ignored[seq] != [list(r.as_xyxy()) for r in anns[seq].ignored_regions]:
                return False
            per_class = {c: 0 for c in CLASSES}
            for fid, source in frames:
                stem = f"{seq}_f{fid:06d}"
                image, label = out_dir / "images" / name / f"{stem}.jpg", out_dir / "labels" / name / f"{stem}.txt"
                if file_info(image)["sha256"] != file_info(source)["sha256"]:
                    return False
                lines = _label_lines(anns[seq], fid, source)
                if label.read_text(encoding="utf-8") != "\n".join(lines):
                    return False
                for line in lines:
                    per_class[CLASSES[int(line.split()[0])]] += 1
            if stats[name][seq] != {"weather": split["weather"][seq], "frames": len(frames), "boxes": per_class}:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _write_dataset_yaml(out_dir: Path, final_dir: Path) -> None:
    (out_dir / "detrac.yaml").write_text(
        f"path: {final_dir.resolve()}\n"
        "train: images/dev   # not used (fine-tuning is not planned)\n"
        "val: images/dev\n"
        "test: images/test\n"
        "names:\n  0: car\n  1: bus\n  2: truck\n", encoding="utf-8")


def convert(anns: dict[str, SequenceAnn], split: dict[str, Any], frame_step: int = 1, copy: bool = False,
            out_dir: str | Path = OUT_DIR, frames_root: str | Path = RAW_ROOT) -> dict[str, dict[str, Any]]:
    """Reuse an identical YOLO export, or build and publish a complete replacement.

    Validate actual images and labels before reuse. An interrupted or incompatible previous export
    is kept in `.previous/` only after its replacement finishes, so it never blocks another run.
    """
    out_dir, frames_root = Path(out_dir), Path(frames_root)
    if frame_step < 1:
        raise ValueError("frame_step must be a positive integer")
    plans = _plan_frames(anns, split, frames_root, frame_step)
    if _prepared_is_current(out_dir, plans, anns, split):
        _write_dataset_yaml(out_dir, out_dir)  # refresh a path after moving the repository
        print(f"[data] complete matching export reused: {out_dir}")
        return read_json(out_dir / "stats.json")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out_dir.name}_", dir=out_dir.parent) as temporary:
        staged = Path(temporary) / "dataset"
        staged.mkdir()
        stats = _export_new(anns, split, plans, copy, staged)
        _write_dataset_yaml(staged, out_dir)
        write_json(staged / "splits.json", split)
        publish_directory(staged, out_dir)
    return stats


def _export_new(anns: dict[str, SequenceAnn], split: dict, plans: dict, copy: bool,
                out_dir: Path) -> dict[str, dict[str, Any]]:
    """Write a new export in a staging directory, leaving the active dataset untouched."""

    stats: dict[str, dict[str, Any]] = {"dev": {}, "test": {}}
    ignored_out: dict[str, list[list[float]]] = {}
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
            lines = _label_lines(ann, fid, img_path)
            for line in lines:
                per_class[CLASSES[int(line.split()[0])]] += 1
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            _place_image(img_path, img_out / f"{stem}.jpg", copy)
            bar.update(1)

        stats[split_name][seq_name] = {"weather": split["weather"][seq_name],
                                       "frames": len(frames), "boxes": per_class}
        ignored_out[seq_name] = [list(r.as_xyxy()) for r in ann.ignored_regions]
    bar.close()

    write_json(out_dir / "ignored_regions.json", ignored_out)
    write_json(out_dir / "stats.json", stats)
    return stats


# ---------------------------------------------------------------- GT in MOT format

def write_gt_mot(ann: SequenceAnn, seq_dir: Path, ignored_regions: list[Region], out_path: Path) -> None:
    """Write the ground truth of one sequence as a MOT file.

    Rows keep the original target ids, so identity metrics can follow objects across frames.
    Frame numbers are converted to JPG numbers, which is what tracker output uses. Classes are
    mapped to project ids and targets whose box centre is in an ignored region are left out, so GT
    and tracker output are filtered by the same rule.
    """
    frame_ids = sorted(ann.frames)
    offset = resolve_naming(seq_dir, frame_ids)

    rows: list[str] = []
    for fid in frame_ids:
        # Targets of a class we evaluate, then one vectorised test for all their centres
        targets = [t for t in ann.frames[fid].targets if t.cls in DETRAC_TO_ID]
        if not targets:
            continue
        ignored = inside_any([t.box.centre() for t in targets], ignored_regions)
        for t, skip in zip(targets, ignored, strict=True):
            if skip:
                continue
            rows.append(f"{fid + offset},{t.track_id},"
                        f"{t.box.x:.2f},{t.box.y:.2f},{t.box.w:.2f},{t.box.h:.2f},"
                        f"1.0,{DETRAC_TO_ID[t.cls]},1.0,-1")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""))


def convert_all(sequences: list[str], out_dir: str | Path, raw_root: str | Path = RAW_ROOT,
                ignored_path: str | Path | None = None) -> Path:
    """Write GT MOT files for `sequences` into `out_dir`, one `<sequence>.txt` each.

    Every sequence is checked first (annotations, frames, ignored regions, frame numbering), so a
    problem in the last sequence does not leave a half-written folder.
    """
    ignored = read_json(ignored_path or REPORTS_DIR / "ignored_regions.json")
    anns, _ = load_all(raw_root)
    out_dir = Path(out_dir)

    # Nothing is written until every sequence has annotations, frames, ignored regions and a clear frame offset
    plans: list[tuple[str, Path, list[Region]]] = []
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
