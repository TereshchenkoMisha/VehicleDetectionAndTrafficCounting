"""Build trajectories from GT annotations or tracker output, and count line crossings on them.

Both sources produce the same rows (frame, key, cls_id, cx, cy), so GT and tracker counts use identical code.
"""
from collections import Counter

from ..config import DETRAC_TO_ID
from ..data.mot import load_mot
from ..data.regions import inside_any
from ..utils import file_info
from .line_crossing import DEFAULT_MAX_FRAME_GAP, count_events, counting_protocol, tally


def gt_rows(ann, offset=0):
    """Box centres of the GT targets of a SequenceAnn. offset converts XML frame numbers to JPG numbers."""
    for fid in sorted(ann.frames):
        for t in ann.frames[fid].targets:
            cid = DETRAC_TO_ID.get(t.cls)
            if cid is not None:
                cx, cy = t.box.centre()
                yield fid + offset, t.track_id, cid, cx, cy


def mot_rows(frames):
    """Box centres from load_mot() output."""
    for fid, targets in sorted(frames.items()):
        for tid, box, cid, _ in targets:
            yield fid, tid, cid, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def build_trajectories(rows, regions=()):
    """[(key, cls_id, [(frame, cx, cy), ...]), ...]. Rows with centres in `regions` are dropped.

    A trajectory's class is the most frequent class over its rows.
    """
    rows = list(rows)
    if regions and rows:
        ignored = inside_any([(row[3], row[4]) for row in rows], regions)
        rows = [row for row, skip in zip(rows, ignored) if not skip]
    points, votes = {}, {}
    for fid, key, cid, cx, cy in rows:
        points.setdefault(key, []).append((fid, cx, cy))
        votes.setdefault(key, Counter())[cid] += 1
    return [(key, votes[key].most_common(1)[0][0], pts) for key, pts in points.items()]


def count_gt(ann, line, max_frame_gap=DEFAULT_MAX_FRAME_GAP, offset=0):
    """GT crossings of one sequence. Returns (result_dict, events_without_ignored_observations)."""
    regions = [region.as_xyxy() for region in ann.ignored_regions]

    def events_for(excluded):
        return count_events(build_trajectories(gt_rows(ann, offset), excluded),
                            line["p1"], line["p2"], max_frame_gap=max_frame_gap)

    events, kept = events_for(()), events_for(regions)
    if events:
        for event, flag in zip(events, inside_any([e["point"] for e in events], regions)):
            event["ignored"] = bool(flag)
    result = {"all": tally(events), "exclude_ignored": tally(kept),
              "event_fields": ["frame", "target", "cls_id", "direction", "event_point_ignored"],
              "events": [[e["frame"], e["target"], e["cls_id"], e["direction"], int(e["ignored"])]
                         for e in events],
              "exclude_ignored_events": kept,
              "protocol": counting_protocol(line, regions, max_frame_gap)}
    return result, kept


def count_tracks(path, line, regions=(), max_frame_gap=DEFAULT_MAX_FRAME_GAP):
    """Crossings of one MOT track file."""
    trajs = build_trajectories(mot_rows(load_mot(path)), regions)
    events = count_events(trajs, line["p1"], line["p2"], max_frame_gap=max_frame_gap)
    return {"counts": tally(events), "tracks": len(trajs),
            "events": [[e["frame"], e["target"], e["cls_id"], e["direction"]] for e in events],
            "protocol": counting_protocol(line, regions, max_frame_gap),
            "source": file_info(path)}
