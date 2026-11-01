"""Build trajectories from GT or tracker output, and count line crossings on them.

GT and tracks are first reduced to the same kind of row, (frame, target key, class, centre x,
centre y). From there on one code path serves both, so GT counts and tracker counts differ only
in their input and never in the counting logic.
"""
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ..config import DETRAC_TO_ID
from ..data.detrac import SequenceAnn
from ..data.mot import MotRow, load_mot
from ..data.regions import Region, inside_any
from ..utils import file_info
from .line_crossing import (DEFAULT_MAX_FRAME_GAP, CrossingEvent, Trajectory, count_events, counting_protocol,
                            tally)

# One observation: (frame, target key, class id, centre x, centre y)
Row = tuple[int, Any, int, float, float]


def gt_rows(ann: SequenceAnn, offset: int = 0) -> Iterator[Row]:
    """Observations of the GT targets of one sequence, in frame order.

    The key is the DETRAC target id. Targets of a class we do not evaluate are skipped. `offset`
    converts XML frame numbers to JPG frame numbers (see `sequences.resolve_naming`), so GT frames
    match the frame numbers in tracker output.
    """
    for fid in sorted(ann.frames):
        for t in ann.frames[fid].targets:
            cid = DETRAC_TO_ID.get(t.cls)
            if cid is not None:
                cx, cy = t.box.centre()
                yield fid + offset, t.track_id, cid, cx, cy


def mot_rows(frames: dict[int, list[MotRow]]) -> Iterator[Row]:
    """Observations from `load_mot` output, in frame order. The key is the tracker's track id."""
    for fid, targets in sorted(frames.items()):
        for tid, box, cid, _ in targets:
            yield fid, tid, cid, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def build_trajectories(rows: Iterable[Row], regions: list[Region] | tuple[()] = ()) -> list[Trajectory]:
    """Group observations into one trajectory per target: (key, class id, [(frame, cx, cy), ...]).

    Observations whose centre lies in `regions` are removed first. That can split a target's path
    into pieces with a gap between them, which the counting rule then treats as a break.
    A tracker may assign different classes to the same track over time, so the trajectory's class
    is the most frequent one (majority vote). For GT the class never changes.
    """
    rows = list(rows)

    # Drop observations in ignored regions, one vectorised test for all rows
    if regions and rows:
        ignored = inside_any([(row[3], row[4]) for row in rows], regions)
        rows = [row for row, skip in zip(rows, ignored, strict=True) if not skip]

    # Group by target and count the class votes
    points: dict[Any, list[tuple[int, float, float]]] = {}
    votes: dict[Any, Counter[int]] = {}
    for fid, key, cid, cx, cy in rows:
        points.setdefault(key, []).append((fid, cx, cy))
        votes.setdefault(key, Counter())[cid] += 1
    return [(key, votes[key].most_common(1)[0][0], pts) for key, pts in points.items()]


def count_gt(ann: SequenceAnn, line: dict[str, list[int]], max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
             offset: int = 0) -> tuple[dict[str, Any], list[CrossingEvent]]:
    """Count the GT crossings of one sequence, twice.

    Why two variants: the tracking stage never sees detections inside ignored regions, so a tracker
    cannot count a vehicle that is only visible there. "exclude_ignored" removes the same observations
    from GT before counting and is the fair reference for tracker counts. "all" is the unfiltered GT,
    kept to show how much the ignored regions change the count. Only "exclude_ignored" is used for
    scoring by default. Returns (result with tallies, events and protocol, events of exclude_ignored).
    """
    regions = [region.as_xyxy() for region in ann.ignored_regions]

    def events_for(excluded: list[Region] | tuple[()]) -> list[CrossingEvent]:
        return count_events(build_trajectories(gt_rows(ann, offset), excluded),
                            line["p1"], line["p2"], max_frame_gap=max_frame_gap)

    events, kept = events_for(()), events_for(regions)

    # Flag the "all" events whose crossing point is inside an ignored region
    flags = inside_any([e["point"] for e in events], regions) if events else []
    for event, flag in zip(events, flags, strict=True):
        event["ignored"] = bool(flag)

    result = {"all": tally(events), "exclude_ignored": tally(kept),
              "event_fields": ["frame", "target", "cls_id", "direction", "event_point_ignored"],
              "events": [[e["frame"], e["target"], e["cls_id"], e["direction"], int(e["ignored"])]
                         for e in events],
              "exclude_ignored_events": kept,
              "protocol": counting_protocol(line, regions, max_frame_gap)}
    return result, kept


def count_tracks(path: str | Path, line: dict[str, list[int]], regions: list[Region] | tuple[()] = (),
                 max_frame_gap: int = DEFAULT_MAX_FRAME_GAP) -> dict[str, Any]:
    """Count the crossings in one MOT track file.

    Observations in `regions` are dropped before counting, matching the GT "exclude_ignored"
    variant. Returns the tallies, the number of tracks, the events, the protocol used, and the
    hash of the file (so the count can be traced to its input).
    """
    trajectories = build_trajectories(mot_rows(load_mot(path)), regions)
    events = count_events(trajectories, line["p1"], line["p2"], max_frame_gap=max_frame_gap)
    return {"counts": tally(events), "tracks": len(trajectories),
            "events": [[e["frame"], e["target"], e["cls_id"], e["direction"]] for e in events],
            "protocol": counting_protocol(line, regions, max_frame_gap),
            "source": file_info(path)}
