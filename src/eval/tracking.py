"""Tracking metrics through TrackEval: HOTA, CLEAR (MOTA, IDSW, ...) and Identity (IDF1, ...).

The metrics are class-agnostic: all objects are matched regardless of class, so a wrong class does
not count as a broken identity. Ground truth and tracker output are read from MOT files and
turned into the per-frame arrays TrackEval expects.

What the reported numbers mean (all in percent unless noted; a box matches a GT box at IoU >= threshold):
  HOTA   Higher Order Tracking Accuracy = sqrt(DetA * AssA): one score for both finding objects and
         keeping their identity. Averaged over IoU thresholds 0.05-0.95, so it also rewards tight boxes.
  DetA   detection accuracy: how many GT objects are found, without false boxes.
  AssA   association accuracy: how well detections are linked into the right tracks over time.
  MOTA   1 - (misses + false positives + ID switches) / GT objects. Dominated by detection errors;
         can be negative.
  IDF1   F1 of identity matching: the share of boxes that carry the correct track ID for the whole
         trajectory. Reads like "how long a vehicle keeps its own ID".
  IDSW   number of ID switches: how often a GT vehicle's matched track ID changes. A count, not a rate.
"""
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from trackeval.metrics import CLEAR, HOTA, Identity

from ..data.mot import MotRow, load_mot

# HOTA is computed at 19 IoU thresholds (0.05 .. 0.95); this is the index of the one at 0.5
_HOTA_50: int = int(np.flatnonzero(np.isclose(HOTA().array_labels, 0.5))[0])


def _iou_matrix(gt_boxes: NDArray, trk_boxes: NDArray) -> NDArray[np.float64]:
    """IoU of every GT box with every tracker box, both given as (N, 4) xyxy. Result: (n_gt, n_trk)."""
    if len(gt_boxes) == 0 or len(trk_boxes) == 0:
        return np.zeros((len(gt_boxes), len(trk_boxes)), dtype=float)

    # Broadcasting: g has shape (n_gt, 1, 4) and t has (1, n_trk, 4), so every operation below
    # yields an (n_gt, n_trk) array holding one value per GT/tracker pair
    g = np.asarray(gt_boxes, dtype=float)[:, None, :]
    t = np.asarray(trk_boxes, dtype=float)[None, :, :]

    # Intersection: width and height of the overlap rectangle, clipped at 0 for disjoint boxes
    inter = (np.clip(np.minimum(g[..., 2], t[..., 2]) - np.maximum(g[..., 0], t[..., 0]), 0, None)
             * np.clip(np.minimum(g[..., 3], t[..., 3]) - np.maximum(g[..., 1], t[..., 1]), 0, None))
    area_g = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    union = area_g + area_t - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)       # 0 where a box has no area


def _build_trackeval_data(gt_frames: dict[int, list[MotRow]], trk_frames: dict[int, list[MotRow]],
                          frame_ids: list[int] | None = None) -> dict[str, Any]:
    """Arrange `load_mot` outputs into the structure TrackEval's `eval_sequence` takes.

    For each frame: GT ids and boxes, tracker ids and boxes, and their IoU matrix. Ids are
    renumbered 0..n-1 per side, as TrackEval requires. `frame_ids` is the complete frame list of
    the sequence; passing it makes frames without any box count as frames (TrackEval then sees every
    GT box missed in a frame the tracker left empty). Without it, only frames present in the files exist.
    """
    observed = set(gt_frames) | set(trk_frames)
    all_frames = sorted(observed) if frame_ids is None else list(frame_ids)
    if len(all_frames) != len(set(all_frames)) or all_frames != sorted(all_frames):
        raise ValueError("Frame IDs must be unique and sorted")
    if observed - set(all_frames):
        raise ValueError("MOT rows contain frames outside the source sequence")

    def id_map(frames: dict[int, list[MotRow]]) -> dict[int, int]:
        ids = sorted({row[0] for f in all_frames for row in frames.get(f, [])})
        return {orig: new for new, orig in enumerate(ids)}

    gt_id_map, trk_id_map = id_map(gt_frames), id_map(trk_frames)

    gt_ids, gt_dets, trk_ids, trk_dets, sims = [], [], [], [], []
    for f in all_frames:
        g, t = gt_frames.get(f, []), trk_frames.get(f, [])
        gt_ids.append(np.array([gt_id_map[x[0]] for x in g], dtype=int))
        gt_dets.append(np.array([x[1] for x in g], dtype=float).reshape(-1, 4))     # reshape keeps (0, 4) for empty frames
        trk_ids.append(np.array([trk_id_map[x[0]] for x in t], dtype=int))
        trk_dets.append(np.array([x[1] for x in t], dtype=float).reshape(-1, 4))
        sims.append(_iou_matrix(gt_dets[-1], trk_dets[-1]))

    return {"gt_ids": gt_ids, "gt_dets": gt_dets, "tracker_ids": trk_ids, "tracker_dets": trk_dets,
            "similarity_scores": sims, "num_frames": len(all_frames), "num_timesteps": len(all_frames),
            "num_gt_ids": len(gt_id_map), "num_gt_dets": sum(len(a) for a in gt_ids),
            "num_tracker_ids": len(trk_id_map), "num_tracker_dets": sum(len(a) for a in trk_ids)}


def evaluate_sequence(gt_path: str | Path, trk_path: str | Path, frame_ids: list[int] | None = None,
                      raw_results: dict[str, Any] | None = None) -> dict[str, dict[str, float]]:
    """Compute the tracking metrics of one sequence from a GT and a tracker MOT file.

    Returns {"all": {...}} with rates in percent and counters as ints. HOTA, DetA, AssA and LocA
    are averaged over the IoU thresholds 0.05-0.95; HOTA_50 is HOTA at IoU 0.5 only.
    If `raw_results` is given, TrackEval's unprocessed output is stored in it so
    `combine_sequence_metrics` can later merge several sequences correctly.
    """
    data = _build_trackeval_data(load_mot(gt_path), load_mot(trk_path), frame_ids=frame_ids)
    hota = HOTA().eval_sequence(data)
    clear = CLEAR().eval_sequence(data)
    identity = Identity().eval_sequence(data)
    if raw_results is not None:
        raw_results["all"] = {"hota": hota, "clear": clear, "identity": identity}

    return {"all": {
        "HOTA": float(np.mean(hota["HOTA"])) * 100,
        "DetA": float(np.mean(hota["DetA"])) * 100,
        "AssA": float(np.mean(hota["AssA"])) * 100,
        "LocA": float(np.mean(hota["LocA"])) * 100,
        "HOTA_50": float(hota["HOTA"][_HOTA_50]) * 100,
        "MOTA": float(clear["MOTA"]) * 100,
        "MOTP": float(clear["MOTP"]) * 100,
        "IDSW": int(clear["IDSW"]), "MT": int(clear["MT"]), "ML": int(clear["ML"]), "Frag": int(clear["Frag"]),
        "CLR_TP": int(clear["CLR_TP"]), "CLR_FN": int(clear["CLR_FN"]), "CLR_FP": int(clear["CLR_FP"]),
        "IDF1": float(identity["IDF1"]) * 100, "IDP": float(identity["IDP"]) * 100,
        "IDR": float(identity["IDR"]) * 100,
        "GT_IDs": data["num_gt_ids"], "TRK_IDs": data["num_tracker_ids"],
        "GT_dets": data["num_gt_dets"], "TRK_dets": data["num_tracker_dets"], "frames": data["num_frames"]}}


def combine_sequence_metrics(raw_per_seq: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Dataset-level metrics from the raw TrackEval results of all sequences, in percent.

    TrackEval merges the underlying counters (matches, false positives, identity matches) of all
    sequences and computes the rates from the sums. Long sequences therefore weigh more than short ones,
    unlike a plain average of per-sequence rates, which gives every sequence the same weight.
    """
    combined = {key: metric.combine_sequences({seq: body["all"][key] for seq, body in raw_per_seq.items()})
                for key, metric in (("hota", HOTA()), ("clear", CLEAR()), ("identity", Identity()))}
    hota, clear, identity = combined["hota"], combined["clear"], combined["identity"]

    out = {key: float(np.mean(hota[key])) * 100 for key in ("HOTA", "DetA", "AssA", "LocA")}
    out["HOTA_50"] = float(hota["HOTA"][_HOTA_50]) * 100
    out.update({key: float(clear[key]) * 100 for key in ("MOTA", "MOTP")})
    out.update({key: float(identity[key]) * 100 for key in ("IDF1", "IDP", "IDR")})
    return out
