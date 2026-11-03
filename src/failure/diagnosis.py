"""Turn paired crossing events into failure candidates, and summarise and select them.

Every GT crossing and every predicted crossing of a configuration gets an outcome:
  matched                    GT crossing paired with a predicted one of the same class;
  missed_candidate           GT crossing without a predicted one (undercount);
  extra_candidate            predicted crossing without a GT one (overcount);
  class_confusion_candidate  paired, but the classes differ.
These are candidates found by geometry and timing, not confirmed causes. The flags next to them only
describe the evidence near the line; a cause is confirmed by looking at the frames (see review.py).
"""
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any, TypedDict

from .matching import BoxMatch, Event, MatchSettings, match_events

EVENT_FIELDS: tuple[str, ...] = ("frame", "target", "cls_id", "direction")      # order of a saved predicted event
NEAR_LINE: int = 2          # frames around the crossing that count as "near the line" for the flags


class FailureRow(TypedDict):
    """One crossing event of one configuration with its outcome and the evidence around it."""
    case_id: str
    sequence: str
    config: str
    outcome: str
    gt_event: Event | None
    pred_event: Event | None
    ambiguous: bool             # the event pairing was not unique, so the outcome is uncertain
    evidence: dict[str, Any] | None
    flags: list[str]            # hints from the boxes near the line, not causes
    track_ids: list[int]        # tracker ids that followed the GT vehicle (or the extra track)


def saved_events(rows: Sequence[Sequence[int]]) -> list[dict[str, int]]:
    """Predicted events as saved by `count_tracks` ([frame, target, class, direction]) in dict form."""
    return [dict(zip(EVENT_FIELDS, row, strict=True)) for row in rows]


def event_of(case: FailureRow) -> Event:
    """The GT event of a case, or the predicted one when the case has no GT event."""
    event = case["gt_event"] or case["pred_event"]
    assert event is not None        # every row has at least one of the two
    return event


def has_gap_across(matches: list[BoxMatch], frame: int, max_frame_gap: int) -> bool:
    """True when one tracker id is seen before and after `frame` with more than `max_frame_gap` frames between."""
    for track in {m["track"] for m in matches}:
        seen = [m["frame"] for m in matches if m["track"] == track]
        before, after = [f for f in seen if f < frame], [f for f in seen if f >= frame]
        if before and after and min(after) - max(before) > max_frame_gap:
            return True
    return False


def event_flags(gt: Event, matches: dict[int, list[BoxMatch]], pred_events: Sequence[Event],
                settings: MatchSettings, max_frame_gap: int) -> tuple[list[str], list[int]]:
    """Evidence around one GT crossing: flags that hint at what went wrong, and the tracker ids on the vehicle.

    Only boxes within `context` frames of the crossing are used, and ids are taken from unambiguous
    pairs only. The flags are hints for the manual review, not a diagnosis.
    """
    local = [m for m in matches.get(gt["target"], []) if abs(m["frame"] - gt["frame"]) <= settings.context]
    clear = [m for m in local if not m["ambiguous"]]
    tracks = sorted({m["track"] for m in clear})

    flags = []
    if not any(abs(m["frame"] - gt["frame"]) <= NEAR_LINE for m in clear):
        flags.append("no_unambiguous_track_match_near_line")
    if len(tracks) > 1:
        flags.append("multiple_track_ids_in_context")
    if len(clear) < len(local):
        flags.append("ambiguous_box_overlap")
    if has_gap_across(clear, gt["frame"], max_frame_gap):
        flags.append("matched_observation_gap_across_line")
    if any(e["target"] in tracks and abs(e["frame"] - gt["frame"]) > settings.event_window for e in pred_events):
        flags.append("same_track_event_outside_time_window")
    return flags, tracks


def classify_events(sequence: str, config: str, gt_events: Sequence[Event], pred_events: Sequence[Event],
                    matches: dict[int, list[BoxMatch]], settings: MatchSettings,
                    max_frame_gap: int) -> list[FailureRow]:
    """Outcome of every GT event and every unpaired predicted event of one sequence and configuration."""
    pairs = match_events(gt_events, pred_events, matches, settings)
    rows: list[FailureRow] = []

    # GT events: paired with a prediction (same class or not) or missed
    for i, gt in enumerate(gt_events):
        pair = pairs.get(i)
        pred = None if pair is None else pred_events[pair["pred"]]
        if pred is None:
            outcome = "missed_candidate"
        else:
            outcome = "matched" if pred["cls_id"] == gt["cls_id"] else "class_confusion_candidate"
        flags, tracks = event_flags(gt, matches, pred_events, settings, max_frame_gap)
        rows.append({"case_id": f"{sequence}__{config}__gt{gt['target']}__f{gt['frame']}", "sequence": sequence,
                     "config": config, "outcome": outcome, "gt_event": gt, "pred_event": pred,
                     "ambiguous": pair is not None and pair["ambiguous"], "evidence": pair,
                     "flags": flags, "track_ids": tracks})

    # Predicted events that no GT event explains
    paired = {pair["pred"] for pair in pairs.values()}
    for j, pred in enumerate(pred_events):
        if j not in paired:
            rows.append({"case_id": f"{sequence}__{config}__pred{pred['target']}__f{pred['frame']}",
                         "sequence": sequence, "config": config, "outcome": "extra_candidate", "gt_event": None,
                         "pred_event": pred, "ambiguous": False, "evidence": None,
                         "flags": ["unmatched_predicted_event"], "track_ids": [pred["target"]]})
    return rows


def summarize(rows: Sequence[FailureRow]) -> list[dict[str, Any]]:
    """Outcome counts per configuration and sequence, plus an ALL row per configuration.

    GT events = paired + missed and predicted events = paired + extra, so every share has a visible
    denominator. A class confusion is a paired event and does not change the total count.
    """
    groups: dict[tuple[str, str], list[FailureRow]] = defaultdict(list)
    for row in rows:
        groups[row["config"], row["sequence"]].append(row)
        groups[row["config"], "ALL"].append(row)

    summary = []
    for (config, sequence), part in sorted(groups.items()):
        outcomes = Counter(row["outcome"] for row in part)
        gt = sum(row["gt_event"] is not None for row in part)
        pred = sum(row["pred_event"] is not None for row in part)
        summary.append({"config": config, "sequence": sequence, "gt_events": gt, "pred_events": pred,
                        "paired": gt - outcomes["missed_candidate"], "missed": outcomes["missed_candidate"],
                        "extra": outcomes["extra_candidate"], "class_confusion": outcomes["class_confusion_candidate"],
                        "ambiguous_pairs": sum(row["ambiguous"] for row in part), "signed_error": pred - gt})
    return summary


def vehicle_key(case: FailureRow) -> tuple[str, int, bool]:
    """Identity of the vehicle behind a case: sequence, id, and whether the id is a GT id or a tracker id."""
    return case["sequence"], event_of(case)["target"], case["gt_event"] is not None


def suggest_cases(rows: Sequence[FailureRow], limit: int = 3) -> list[str]:
    """Case ids worth a manual look: failures of a vehicle that some configuration counted correctly.

    Such a case compares configurations on the same vehicle, which is what points to the detector or
    the tracker. The picks are spread over outcome kinds and sequences, one per vehicle. The result is a
    suggestion for the reviewer, not a sample of how often each failure happens.
    """
    solved = {vehicle_key(r) for r in rows if r["outcome"] == "matched"}
    candidates = [r for r in rows if r["outcome"] != "matched" and not r["ambiguous"]]

    chosen: list[FailureRow] = []
    while candidates and len(chosen) < limit:
        kinds, sequences = {c["outcome"] for c in chosen}, {c["sequence"] for c in chosen}
        best = min(candidates, key=lambda r: (r["outcome"] in kinds, r["sequence"] in sequences,
                                              vehicle_key(r) not in solved, r["case_id"]))
        chosen.append(best)
        candidates = [r for r in candidates if vehicle_key(r) != vehicle_key(best)]
    return [case["case_id"] for case in chosen]


def find_case(rows: Sequence[FailureRow], case_id: str) -> FailureRow:
    """The row with this case id; ValueError when there is none."""
    for row in rows:
        if row["case_id"] == case_id:
            return row
    raise ValueError(f"Unknown case id: {case_id}")


def peers_of(rows: Sequence[FailureRow], case: FailureRow, reference_gt: int | None = None) -> dict[str, FailureRow]:
    """The same GT vehicle in every configuration: {configuration: row}.

    A case without a GT event (an extra crossing) has no vehicle to follow until a reviewer looks at the
    frames and gives its GT id as `reference_gt`; without it the case stands alone.
    """
    target = case["gt_event"]["target"] if case["gt_event"] else reference_gt
    if target is None:
        return {case["config"]: case}
    return {r["config"]: r for r in rows
            if r["sequence"] == case["sequence"] and r["gt_event"] and r["gt_event"]["target"] == target}
