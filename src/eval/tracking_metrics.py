import numpy as np
from ..data.config import CLASSES
from ..data.mot import load_mot


def _iou_matrix(gt_boxes, trk_boxes):
    # Calculate the IoU matrix from two box arrays
    if len(gt_boxes) == 0 or len(trk_boxes) == 0:
        return np.zeros((len(gt_boxes), len(trk_boxes)), dtype=float)
    g = np.asarray(gt_boxes, dtype=float)
    t = np.asarray(trk_boxes, dtype=float)
    # Broadcast box coordinates across pairs
    g1, t1 = g[:, None, :], t[None, :, :]
    ix1 = np.maximum(g1[..., 0], t1[..., 0])
    iy1 = np.maximum(g1[..., 1], t1[..., 1])
    ix2 = np.minimum(g1[..., 2], t1[..., 2])
    iy2 = np.minimum(g1[..., 3], t1[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_g = (g1[..., 2] - g1[..., 0]) * (g1[..., 3] - g1[..., 1])
    area_t = (t1[..., 2] - t1[..., 0]) * (t1[..., 3] - t1[..., 1])
    union = area_g + area_t - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _build_trackeval_data(gt_frames, trk_frames, cls_filter=None, frame_ids=None):
    # Assemble data for HOTA CLEAR and Identity
    observed = set(gt_frames) | set(trk_frames)
    all_frames = sorted(observed) if frame_ids is None else list(frame_ids)
    if len(all_frames) != len(set(all_frames)) or all_frames != sorted(all_frames):
        raise ValueError("Frame IDs must be unique and sorted")
    if observed - set(all_frames):
        raise ValueError("MOT rows contain frames outside the source sequence")

    # Collect unique IDs for remapping
    gt_id_set, trk_id_set = set(), set()
    for f in all_frames:
        for t in gt_frames.get(f, []):
            if cls_filter is None or t[2] == cls_filter:
                gt_id_set.add(t[0])
        for t in trk_frames.get(f, []):
            if cls_filter is None or t[2] == cls_filter:
                trk_id_set.add(t[0])

    # Map original IDs to contiguous indices
    gt_id_map = {orig: new for new, orig in enumerate(sorted(gt_id_set))}
    trk_id_map = {orig: new for new, orig in enumerate(sorted(trk_id_set))}

    # Collect detections with remapped IDs
    gt_ids, gt_dets, trk_ids, trk_dets, sims = [], [], [], [], []
    for f in all_frames:
        g = [t for t in gt_frames.get(f, []) if cls_filter is None or t[2] == cls_filter]
        t_ = [t for t in trk_frames.get(f, []) if cls_filter is None or t[2] == cls_filter]
        gt_ids.append(np.array([gt_id_map[x[0]] for x in g], dtype=int))
        gt_dets.append(np.array([x[1] for x in g], dtype=float).reshape(-1, 4))
        trk_ids.append(np.array([trk_id_map[x[0]] for x in t_], dtype=int))
        trk_dets.append(np.array([x[1] for x in t_], dtype=float).reshape(-1, 4))
        sims.append(_iou_matrix(gt_dets[-1], trk_dets[-1]))

    return {
        "gt_ids": gt_ids,
        "gt_dets": gt_dets,
        "tracker_ids": trk_ids,
        "tracker_dets": trk_dets,
        "similarity_scores": sims,
        "num_frames": len(all_frames),
        "num_timesteps": len(all_frames),
        "num_gt_ids": len(gt_id_map),
        "num_gt_dets": sum(len(arr) for arr in gt_ids),
        "num_tracker_ids": len(trk_id_map),
        "num_tracker_dets": sum(len(arr) for arr in trk_ids),
    }


def evaluate_sequence(gt_path, trk_path, classes=None, frame_ids=None, raw_results=None):
    from trackeval.metrics import HOTA, CLEAR, Identity

    # Calculate tracking metrics for one sequence
    gt = load_mot(gt_path)
    trk = load_mot(trk_path)

    hota_m, clear_m, id_m = HOTA(), CLEAR(), Identity()

    # Find the HOTA entry for IoU 0.5
    hota_50_index = int(
        np.flatnonzero(np.isclose(hota_m.array_labels, 0.5))[0]
    )

    out = {}

    targets = [("all", None)]
    if classes is not None:
        for cls_name, cls_id in zip(CLASSES, (0, 1, 2)):
            targets.append((cls_name, cls_id))

    for name, cls_filter in targets:
        data = _build_trackeval_data(gt, trk, cls_filter=cls_filter, frame_ids=frame_ids)
        h_res = hota_m.eval_sequence(data)
        c_res = clear_m.eval_sequence(data)
        i_res = id_m.eval_sequence(data)

        if raw_results is not None:
            raw_results[name] = {"hota": h_res, "clear": c_res, "identity": i_res}

        out[name] = {
            # Average HOTA over the configured IoU thresholds
            "HOTA": float(np.mean(h_res["HOTA"])) * 100,
            "DetA": float(np.mean(h_res["DetA"])) * 100,
            "AssA": float(np.mean(h_res["AssA"])) * 100,
            "LocA": float(np.mean(h_res["LocA"])) * 100,
            "HOTA_50": float(h_res["HOTA"][hota_50_index]) * 100,    # Select HOTA at IoU 0.5
            # Collect CLEAR metrics
            "MOTA": float(c_res["MOTA"]) * 100,
            "MOTP": float(c_res["MOTP"]) * 100,
            "IDSW": int(c_res["IDSW"]),
            "MT": int(c_res["MT"]),
            "ML": int(c_res["ML"]),
            "Frag": int(c_res["Frag"]),
            "CLR_TP": int(c_res["CLR_TP"]),
            "CLR_FN": int(c_res["CLR_FN"]),
            "CLR_FP": int(c_res["CLR_FP"]),
            # Collect identity metrics
            "IDF1": float(i_res["IDF1"]) * 100,
            "IDP": float(i_res["IDP"]) * 100,
            "IDR": float(i_res["IDR"]) * 100,
            "GT_IDs": data["num_gt_ids"],
            "TRK_IDs": data["num_tracker_ids"],
            "GT_dets": data["num_gt_dets"],
            "TRK_dets": data["num_tracker_dets"],
            "frames": data["num_frames"],
        }
    return out

def combine_sequence_metrics(raw_per_seq):
    from trackeval.metrics import HOTA, CLEAR, Identity

    combined = {}
    for key, metric in (("hota", HOTA()), ("clear", CLEAR()), ("identity", Identity())):
        combined[key] = metric.combine_sequences({seq: body["all"][key]
                                                  for seq, body in raw_per_seq.items()})
    hota, clear, identity = combined["hota"], combined["clear"], combined["identity"]
    index = int(np.flatnonzero(np.isclose(HOTA().array_labels, 0.5))[0])
    out = {key: float(np.mean(hota[key])) * 100 for key in ("HOTA", "DetA", "AssA", "LocA")}
    out["HOTA_50"] = float(hota["HOTA"][index]) * 100
    out.update({key: float(clear[key]) * 100 for key in ("MOTA", "MOTP")})
    out.update({key: float(identity[key]) * 100 for key in ("IDF1", "IDP", "IDR")})
    return out
