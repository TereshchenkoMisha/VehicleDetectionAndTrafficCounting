"""Convert XML boxes to MOT rows using source image frame IDs"""
import json
from pathlib import Path

from .config import DETRAC_TO_ID, RAW_ROOT, REPORTS_DIR
from .detrac_parser import find_sequence_dir, load_all
from .frames import resolve_naming as _resolve_naming


def _in_ignored(cx, cy, regions_xyxy):
    for x1, y1, x2, y2 in regions_xyxy:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


def write_gt_mot(ann, seq_dir, ignored_regions, out_path: Path):
    """Write GT boxes with original target IDs and local class IDs"""
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
    """Convert selected sequences with the configured ignored regions"""
    ignored = json.loads(Path(ignored_path or REPORTS_DIR / "ignored_regions.json").read_text())
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
        _resolve_naming(seq_dir, sorted(anns[seq].frames))
        plans.append((seq, seq_dir, ignored[seq]))

    for seq, seq_dir, regions in plans:
        write_gt_mot(anns[seq], seq_dir, regions, out_dir / f"{seq}.txt")
        print(f"[gt-mot] {seq}: written")
    return out_dir
