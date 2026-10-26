import numpy as np
from pathlib import Path
from trackeval.metrics import HOTA, CLEAR, Identity


CLASSES = ["car", "bus", "truck"]


def load_mot(path):
    # Parses an MOT file -> {frame_id: [(track_id, xyxy, cls, conf), ...]}
    frames = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            fid = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            cls = int(parts[7]) if len(parts) > 7 else 0
            xyxy = [x, y, x + w, y + h]
            frames.setdefault(fid, []).append((tid, xyxy, cls, conf))
    return frames


def _iou_matrix(gt_boxes, trk_boxes):
    # IoU matrix [num_gt × num_trk] from two xyxy arrays
    if len(gt_boxes) == 0 or len(trk_boxes) == 0:
        return np.zeros((len(gt_boxes), len(trk_boxes)), dtype=float)
    g = np.asarray(gt_boxes, dtype=float)
    t = np.asarray(trk_boxes, dtype=float)
    # pairwise: [G, 1, 4] vs [1, T, 4]
    g1, t1 = g[:, None, :], t[None, :, :]
    ix1 = np.maximum(g1[..., 0], t1[..., 0])
    iy1 = np.maximum(g1[..., 1], t1[..., 1])
    ix2 = np.minimum(g1[..., 2], t1[..., 2])
    iy2 = np.minimum(g1[..., 3], t1[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_g = (g1[..., 2] - g1[..., 0]) * (g1[..., 3] - g1[..., 1])
    area_t = (t1[..., 2] - t1[..., 0]) * (t1[..., 3] - t1[..., 1])
    union = area_g + area_t - inter
    return np.where(union > 0, inter / union, 0.0)


def _build_trackeval_data(gt_frames, trk_frames, cls_filter=None):
    # Assembles the data structure expected by HOTA/CLEAR/Identity
    all_frames = sorted(set(gt_frames) | set(trk_frames))

    # First pass: collect unique IDs and build remap tables
    gt_id_set, trk_id_set = set(), set()
    for f in all_frames:
        for t in gt_frames.get(f, []):
            if cls_filter is None or t[2] == cls_filter:
                gt_id_set.add(t[0])
        for t in trk_frames.get(f, []):
            if cls_filter is None or t[2] == cls_filter:
                trk_id_set.add(t[0])

    # Remap: {original_id -> 0, 1, 2, ...}
    gt_id_map = {orig: new for new, orig in enumerate(sorted(gt_id_set))}
    trk_id_map = {orig: new for new, orig in enumerate(sorted(trk_id_set))}

    # Second pass: collect data with remapped IDs
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


def evaluate_sequence(gt_path, trk_path, classes=None):
    # Calculates HOTA, MOTA, IDSW, IDF1 etc. for one seqeunce
    gt = load_mot(gt_path)
    trk = load_mot(trk_path)

    hota_m, clear_m, id_m = HOTA(), CLEAR(), Identity()
    out = {}

    targets = [("all", None)]
    if classes is not None:
        for cls_name, cls_id in zip(CLASSES, (0, 1, 2)):
            targets.append((cls_name, cls_id))

    for name, cls_filter in targets:
        data = _build_trackeval_data(gt, trk, cls_filter=cls_filter)
        h_res = hota_m.eval_sequence(data)
        c_res = clear_m.eval_sequence(data)
        i_res = id_m.eval_sequence(data)

        out[name] = {
            # HOTA (averaged over IoU 0.05..0.95)
            "HOTA": float(np.mean(h_res["HOTA"])) * 100,
            "DetA": float(np.mean(h_res["DetA"])) * 100,
            "AssA": float(np.mean(h_res["AssA"])) * 100,
            "LocA": float(np.mean(h_res["LocA"])) * 100,
            "HOTA_50": float(h_res["HOTA(0)"]) * 100,    # HOTA@IoU=0.5
            # CLEAR (MOTA, MOTP, IDSW, MT/ML/PT, Frag)
            "MOTA": float(c_res["MOTA"]) * 100,
            "MOTP": float(c_res["MOTP"]) * 100,
            "IDSW": int(c_res["IDSW"]),
            "MT": int(c_res["MT"]),
            "ML": int(c_res["ML"]),
            "Frag": int(c_res["Frag"]),
            "CLR_TP": int(c_res["CLR_TP"]),
            "CLR_FN": int(c_res["CLR_FN"]),
            "CLR_FP": int(c_res["CLR_FP"]),
            # Identity (IDF1, IDP, IDR)
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