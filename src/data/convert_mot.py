"""Конвертация UA-DETRAC XML в MOT-формат с сохранением track_id.

Frame_id в MOT-файле = IMG filename digit (тот же, что в track-файлах pipeline.list_frames),
поэтому используется тот же offset, что и в convert_yolo._resolve_naming.

GT-боксы, центр которых попадает в ignored-регион, дропаются — тот же протокол,
что у детектора (detector_eval._drop_ignored_gt) и трекера (pipeline.keep_mask).
"""
import json
from pathlib import Path

from .config import CLASSES, DETRAC_TO_ID, RAW_ROOT, REPORTS_DIR
from .detrac_parser import find_sequence_dir, load_all


def _resolve_naming(seq_dir, frame_ids):
    """Тот же offset 0/1, что в convert_yolo._resolve_naming — НЕ дублировать логику,
    импортировать из convert_yolo при возможности."""
    from .convert_yolo import _resolve_naming as _r
    return _r(seq_dir, frame_ids)


def _in_ignored(cx, cy, regions_xyxy):
    for x1, y1, x2, y2 in regions_xyxy:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


def write_gt_mot(ann, seq_dir, ignored_regions, out_path: Path):
    """Конвертирует одну SequenceAnn в MOT-файл.

    Формат строки: frame,track_id,x,y,w,h,conf,cls,vis,-1
      - frame: IMG filename digit = XML frame_num + offset (как в треках)
      - track_id: из XML target id (int)
      - x,y,w,h: top-left + size в пикселях (как в треках)
      - conf: 1.0 для GT
      - cls: локальный id (0=car, 1=bus, 2=truck; van→0, others→2)
      - vis: 1.0 (полная видимость — упрощение, можно использовать occlusion из XML)
    """
    frame_ids = sorted(ann.frames)
    offset = _resolve_naming(seq_dir, frame_ids)
    rows = []
    for fid in frame_ids:
        mot_frame = fid + offset
        for t in ann.frames[fid].targets:
            cid = DETRAC_TO_ID.get(t.cls)
            if cid is None:
                continue
            cx, cy = t.box.x + t.box.w / 2, t.box.y + t.box.h / 2
            if _in_ignored(cx, cy, ignored_regions):
                continue
            rows.append(
                f"{mot_frame},{t.track_id},"
                f"{t.box.x:.2f},{t.box.y:.2f},{t.box.w:.2f},{t.box.h:.2f},"
                f"1.0,{cid},1.0,-1"
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""))


def convert_all(sequences, out_dir, raw_root=RAW_ROOT, ignored_path=None):
    """Конвертирует список последовательностей в GT MOT-файлы.

    ignored_path: путь к reports/ignored_regions.json (если None — берётся из REPORTS_DIR).
    """
    ignored = json.loads(Path(ignored_path or REPORTS_DIR / "ignored_regions.json").read_text())
    anns, _ = load_all(raw_root)
    out_dir = Path(out_dir)
    for seq in sequences:
        if seq not in anns:
            print(f"[warn] no XML for {seq}, skipped")
            continue
        seq_dir = find_sequence_dir(raw_root, seq)
        if seq_dir is None:
            print(f"[warn] no frames for {seq}, skipped")
            continue
        regions = ignored.get(seq, [])
        write_gt_mot(anns[seq], seq_dir, regions, out_dir / f"{seq}.txt")
        print(f"[gt-mot] {seq}: written")
    return out_dir