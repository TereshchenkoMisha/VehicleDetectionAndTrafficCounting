"""Pair ground truth with tracker output: first the boxes inside each frame, then the crossing events.

Both pairings are diagnostic. They use only geometry and timing and ignore classes and track ids, so a
wrong class or a duplicated track stays visible as a difference instead of hiding inside the pairing.
They are not the identity assignment that TrackEval computes for the tracking metrics.
"""
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from ..data.mot import MotRow

# A counted crossing: a dict with at least frame, target, cls_id and direction
Event = Mapping[str, Any]

NO_MATCH: float = 1e4       # assignment cost of a forbidden pair; larger than any set of allowed pairs together


@dataclass(frozen=True)
class MatchSettings:
    """Thresholds of the diagnostic pairing, saved next to its results.

    match_iou     boxes overlapping less than this are never paired;
    event_window  largest frame distance between a GT and a predicted crossing that may be paired;
    min_support   frames in which the two tracks must share a box near the crossing to pair the events;
    context       frames around a crossing that are inspected for failure flags.
    """
    match_iou: float = 0.3
    event_window: int = 10
    min_support: int = 2
    context: int = 25


class BoxMatch(TypedDict):
    """One GT box paired with one tracker box in one frame."""
    frame: int
    track: int              # tracker id of the paired box
    iou: float
    ambiguous: bool         # another box overlaps enough to be a possible pair too


class EventPair(TypedDict):
    """A GT crossing paired with a predicted crossing, and the evidence for it."""
    pred: int               # index of the predicted event
    support_frames: int     # frames where the GT target and the predicted track share a box
    mean_iou: float         # mean overlap of those boxes
    frame_delta: int        # predicted frame - GT frame
    ambiguous: bool         # the event had more than one possible partner, or the partner more than one event


def iou_matrix(gt: NDArray[np.floating], pred: NDArray[np.floating]) -> NDArray[np.floating]:
    """IoU of every GT box with every predicted box; both inputs are (N, 4) arrays of x1, y1, x2, y2."""
    gt, pred = np.asarray(gt, dtype=float), np.asarray(pred, dtype=float)
    top_left = np.maximum(gt[:, None, :2], pred[None, :, :2])
    bottom_right = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    union = ((gt[:, 2:] - gt[:, :2]).prod(axis=1)[:, None] + (pred[:, 2:] - pred[:, :2]).prod(axis=1)[None, :]
             - intersection)
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_boxes(gt: dict[int, list[MotRow]], pred: dict[int, list[MotRow]],
                min_iou: float) -> dict[int, list[BoxMatch]]:
    """Pair GT boxes with tracker boxes frame by frame: {GT target id: [matches in frame order]}.

    `gt` and `pred` are `load_mot` outputs. In every frame the pairing is one-to-one and uses as many
    pairs with IoU >= `min_iou` as possible; the overlap only decides between pairings of equal size.
    """
    matches: dict[int, list[BoxMatch]] = defaultdict(list)
    for frame in sorted(gt.keys() & pred.keys()):
        gt_rows, pred_rows = gt[frame], pred[frame]
        if not gt_rows or not pred_rows:
            continue

        # Overlap of every GT box with every tracker box; only pairs above the threshold are allowed
        overlap = iou_matrix(np.array([row[1] for row in gt_rows]), np.array([row[1] for row in pred_rows]))
        allowed = overlap >= min_iou

        # A forbidden pair costs more than all allowed pairs together, so the number of pairs comes first
        cost = np.where(allowed, 1 - overlap, NO_MATCH)
        for i, j in zip(*linear_sum_assignment(cost), strict=True):
            if allowed[i, j]:
                # A box that overlaps another candidate too makes this pairing uncertain
                ambiguous = bool(allowed[i].sum() > 1 or allowed[:, j].sum() > 1)
                matches[gt_rows[i][0]].append({"frame": frame, "track": pred_rows[j][0],
                                               "iou": float(overlap[i, j]), "ambiguous": ambiguous})
    return dict(matches)


def match_events(gt_events: Sequence[Event], pred_events: Sequence[Event], matches: dict[int, list[BoxMatch]],
                 settings: MatchSettings) -> dict[int, EventPair]:
    """Pair GT crossings with predicted crossings: {index of the GT event: EventPair}.

    Two events can be paired when they have the same direction, are at most `event_window` frames apart
    and the GT target and the predicted track share at least `min_support` unambiguous boxes around the
    crossing. The classes are not compared, so a class error shows up as a pair with different classes.
    The pairing is one-to-one: the lowest total of time distance and (1 - mean IoU) wins.
    """
    if not gt_events or not pred_events:
        return {}

    cost = np.full((len(gt_events), len(pred_events)), NO_MATCH)
    evidence: dict[tuple[int, int], tuple[int, float, int]] = {}
    for i, gt in enumerate(gt_events):
        for j, pred in enumerate(pred_events):
            delta = pred["frame"] - gt["frame"]
            if abs(delta) > settings.event_window or gt["direction"] != pred["direction"]:
                continue

            # Unambiguous boxes of the GT target that belong to the predicted track, around both crossings
            low = min(gt["frame"], pred["frame"]) - settings.event_window
            high = max(gt["frame"], pred["frame"]) + settings.event_window
            shared = [m for m in matches.get(gt["target"], [])
                      if m["track"] == pred["target"] and low <= m["frame"] <= high and not m["ambiguous"]]
            if len(shared) < settings.min_support:
                continue
            mean_iou = float(np.mean([m["iou"] for m in shared]))
            cost[i, j] = abs(delta) / settings.event_window + 1 - mean_iou
            evidence[i, j] = (len(shared), mean_iou, delta)

    allowed = cost < NO_MATCH
    return {int(i): {"pred": int(j), "support_frames": evidence[i, j][0], "mean_iou": evidence[i, j][1],
                     "frame_delta": evidence[i, j][2],
                     "ambiguous": bool(allowed[i].sum() > 1 or allowed[:, j].sum() > 1)}
            for i, j in zip(*linear_sum_assignment(cost), strict=True) if allowed[i, j]}
