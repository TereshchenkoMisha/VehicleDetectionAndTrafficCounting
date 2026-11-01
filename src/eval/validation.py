"""Check that a tracking run is complete and unchanged before it is counted or evaluated."""
from pathlib import Path

from ..config import RAW_ROOT
from ..data.mot import load_mot
from ..data.sequences import list_frames
from ..utils import file_info, read_json


def run_dirs(split_dir):
    """Finished configuration directories of a split (unfinished `*.partial` ones are skipped)."""
    return sorted(p for p in Path(split_dir).iterdir() if p.is_dir() and p.suffix != ".partial")


def validate_tracking_run(cfg_dir, sequences, raw_root=RAW_ROOT, split=None):
    """Raise ValueError unless cfg_dir holds a complete, untampered run for exactly `sequences`. Returns timing.json."""
    cfg_dir = Path(cfg_dir)
    sequences = list(sequences)
    expected = set(sequences)
    if not sequences or len(expected) != len(sequences):
        raise ValueError("Expected sequences must be nonempty and unique")

    timing_path = cfg_dir / "timing.json"
    timing = read_json(timing_path)
    if split is not None and timing["split"] != split:
        raise ValueError(f"Split mismatch in {timing_path}")
    if set(timing["per_seq"]) != expected:
        raise ValueError(f"Incomplete sequence metadata in {timing_path}")
    if {path.stem for path in cfg_dir.glob("MVI_*.txt")} != expected:
        raise ValueError(f"MOT files mismatch in {cfg_dir}")

    total_frames = 0
    for seq in sequences:
        frame_ids = {fid for fid, _ in list_frames(seq, raw_root)}
        info = timing["per_seq"][seq]
        if info["frames"] != len(frame_ids):
            raise ValueError(f"Incomplete run {cfg_dir.name}/{seq}: "
                             f"processed={info['frames']}, expected={len(frame_ids)}")
        mot_path = cfg_dir / f"{seq}.txt"
        mot = load_mot(mot_path)
        if set(mot) - frame_ids:
            raise ValueError(f"Unknown MOT frame IDs in {mot_path}")
        if info["rows"] != sum(len(targets) for targets in mot.values()):
            raise ValueError(f"MOT row count mismatch in {mot_path}")
        if "mot_sha256" in info and info["mot_sha256"] != file_info(mot_path)["sha256"]:
            raise ValueError(f"MOT fingerprint mismatch in {mot_path}")
        total_frames += len(frame_ids)

    if timing["frames"] != total_frames:
        raise ValueError(f"Total frame count mismatch in {timing_path}")
    return timing
