"""Check that a tracking run is complete and unchanged before its results are used.

A run folder is `<tracks>/<split>/<configuration>/` with one MOT file per sequence and a
`timing.json` that records what the run contains. Counting and tracking evaluation call
`validate_tracking_run` first, so a half-finished, partly copied or edited run fails loudly instead
of producing plausible but wrong numbers.
"""
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import RAW_ROOT
from ..data.mot import load_mot
from ..data.sequences import list_frames
from ..utils import file_info, read_json


def run_dirs(split_dir: str | Path) -> list[Path]:
    """Finished configuration folders of a split, sorted by name.

    A run being written lives in a `<name>.partial` folder and is renamed when done, so
    `.partial` folders are unfinished or abandoned runs and are skipped.
    """
    return sorted(p for p in Path(split_dir).iterdir() if p.is_dir() and p.suffix != ".partial")


def validate_tracking_run(cfg_dir: str | Path, sequences: Sequence[str], raw_root: str | Path = RAW_ROOT,
                          split: str | None = None) -> dict[str, Any]:
    """Raise ValueError unless `cfg_dir` is a complete, unmodified run of exactly `sequences`.

    Checks, in order: the sequence list and the MOT files match the expected set; the number of
    frames processed equals the number of frames in the raw dataset; every MOT file parses, only
    uses real frame numbers, has the row count recorded in timing.json and has the recorded
    SHA-256; the total frame count adds up. Returns the parsed timing.json.
    """
    cfg_dir = Path(cfg_dir)
    sequences = list(sequences)
    expected = set(sequences)
    if not sequences or len(expected) != len(sequences):
        raise ValueError("Expected sequences must be nonempty and unique")

    # What the run claims to contain, against what is expected
    timing_path = cfg_dir / "timing.json"
    timing = read_json(timing_path)
    if split is not None and timing["split"] != split:
        raise ValueError(f"Split mismatch in {timing_path}")
    if set(timing["per_seq"]) != expected:
        raise ValueError(f"Incomplete sequence metadata in {timing_path}")
    if {path.stem for path in cfg_dir.glob("MVI_*.txt")} != expected:
        raise ValueError(f"MOT files mismatch in {cfg_dir}")

    # Each sequence: every frame processed, and the MOT file is the one the run wrote
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
        # Older runs have no hash; when present it must match
        if "mot_sha256" in info and info["mot_sha256"] != file_info(mot_path)["sha256"]:
            raise ValueError(f"MOT fingerprint mismatch in {mot_path}")
        total_frames += len(frame_ids)

    if timing["frames"] != total_frames:
        raise ValueError(f"Total frame count mismatch in {timing_path}")
    return timing
