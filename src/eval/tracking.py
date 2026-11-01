"""Tracking metrics (HOTA, CLEAR, Identity) via TrackEval. Class-agnostic: a class error is not an ID error."""
import numpy as np
from trackeval.metrics import CLEAR, HOTA, Identity

from ..data.mot import load_mot


def _iou_matrix(gt_boxes, trk_boxes):
    """IoU of every GT box with every tracker box (xyxy), shape (n_gt, n_trk)."""
    if len(gt_boxes) == 0 or len(trk_boxes) == 0:
        return np.zeros((len(gt_boxes), len(trk_boxes)), dtype=float)
    g = np.asarray(gt_boxes, dtype=float)[:, None, :]
    t = np.asarray(trk_boxes, dtype=float)[None, :, :]
    inter = (np.clip(np.minimum(g[..., 2], t[..., 2]) - np.maximum(g[..., 0], t[..., 0]), 0, None)
             * np.clip(np.minimum(g[..., 3], t[..., 3]) - np.maximum(g[..., 1], t[..., 1]), 0, None))
    area_g = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    union = area_g + area_t - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _build_trackeval_data(gt_frames, trk_frames, frame_ids=None):
    """Arrange load_mot() outputs into the per-frame arrays TrackEval expects, with contiguous 0-based IDs."""
    observed = set(gt_frames) | set(trk_frames)
    all_frames = sorted(observed) if frame_ids is None else list(frame_ids)
    if len(all_frames) != len(set(all_frames)) or all_frames != sorted(all_frames):
        raise ValueError("Frame IDs must be unique and sorted")
    if observed - set(all_frames):
        raise ValueError("MOT rows contain frames outside the source sequence")

    def id_map(frames):
        return {orig: new for new, orig in enumerate(sorted({t[0] for f in all_frames for t in frames.get(f, [])}))}

    gt_id_map, trk_id_map = id_map(gt_frames), id_map(trk_frames)
    gt_ids, gt_dets, trk_ids, trk_dets, sims = [], [], [], [], []
    for f in all_frames:
        g, t = gt_frames.get(f, []), trk_frames.get(f, [])
        gt_ids.append(np.array([gt_id_map[x[0]] for x in g], dtype=int))
        gt_dets.append(np.array([x[1] for x in g], dtype=float).reshape(-1, 4))
        trk_ids.append(np.array([trk_id_map[x[0]] for x in t], dtype=int))
        trk_dets.append(np.array([x[1] for x in t], dtype=float).reshape(-1, 4))
        sims.append(_iou_matrix(gt_dets[-1], trk_dets[-1]))

    return {"gt_ids": gt_ids, "gt_dets": gt_dets, "tracker_ids": trk_ids, "tracker_dets": trk_dets,
            "similarity_scores": sims, "num_frames": len(all_frames), "num_timesteps": len(all_frames),
            "num_gt_ids": len(gt_id_map), "num_gt_dets": sum(len(a) for a in gt_ids),
            "num_tracker_ids": len(trk_id_map), "num_tracker_dets": sum(len(a) for a in trk_ids)}


_HOTA_50 = int(np.flatnonzero(np.isclose(HOTA().array_labels, 0.5))[0])     # index of IoU threshold 0.5


def evaluate_sequence(gt_path, trk_path, frame_ids=None, raw_results=None):
    """Metrics of one sequence as {"all": {...}} in percent. raw_results (a dict) receives TrackEval's raw output
    for combine_sequence_metrics()."""
    data = _build_trackeval_data(load_mot(gt_path), load_mot(trk_path), frame_ids=frame_ids)
    hota = HOTA().eval_sequence(data)
    clear = CLEAR().eval_sequence(data)
    identity = Identity().eval_sequence(data)
    if raw_results is not None:
        raw_results["all"] = {"hota": hota, "clear": clear, "identity": identity}

    return {"all": {
        "HOTA": float(np.mean(hota["HOTA"])) * 100,        # averaged over the IoU thresholds
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


def combine_sequence_metrics(raw_per_seq):
    """Dataset-level metrics from the raw results of all sequences (TrackEval's own combination), in percent."""
    combined = {key: metric.combine_sequences({seq: body["all"][key] for seq, body in raw_per_seq.items()})
                for key, metric in (("hota", HOTA()), ("clear", CLEAR()), ("identity", Identity()))}
    hota, clear, identity = combined["hota"], combined["clear"], combined["identity"]
    out = {key: float(np.mean(hota[key])) * 100 for key in ("HOTA", "DetA", "AssA", "LocA")}
    out["HOTA_50"] = float(hota["HOTA"][_HOTA_50]) * 100
    out.update({key: float(clear[key]) * 100 for key in ("MOTA", "MOTP")})
    out.update({key: float(identity[key]) * 100 for key in ("IDF1", "IDP", "IDR")})
    return out
