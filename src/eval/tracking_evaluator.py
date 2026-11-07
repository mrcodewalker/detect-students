"""Multi-object tracking evaluation engine using official TrackEval metrics."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

# TrackEval compatibility patch for NumPy 2.x
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

from trackeval.metrics import CLEAR, HOTA, Identity

from src.eval.mot_export import MOTRecord, read_mot_format


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU similarity matrix between two sets of (x1, y1, x2, y2) boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float64)

    a = np.ascontiguousarray(boxes_a, dtype=np.float64)
    b = np.ascontiguousarray(boxes_b, dtype=np.float64)

    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    intersection = w * h

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection

    return intersection / np.maximum(union, 1e-6)


def prepare_trackeval_data(
    gt_records: Sequence[MOTRecord],
    tracker_records: Sequence[MOTRecord],
) -> Dict[str, Any]:
    """Format ground-truth and tracker records into the input dictionary for TrackEval metrics."""
    # Find active frame range (1-based normalized to 0-based index)
    gt_frames = [r.frame for r in gt_records]
    trk_frames = [r.frame for r in tracker_records]

    if not gt_frames and not trk_frames:
        min_frame, max_frame = 1, 1
    elif not gt_frames:
        min_frame, max_frame = min(trk_frames), max(trk_frames)
    elif not trk_frames:
        min_frame, max_frame = min(gt_frames), max(gt_frames)
    else:
        min_frame = min(min(gt_frames), min(trk_frames))
        max_frame = max(max(gt_frames), max(trk_frames))

    num_timesteps = max(1, max_frame - min_frame + 1)

    # Group records by frame offset
    gt_by_frame: Dict[int, List[MOTRecord]] = defaultdict(list)
    for r in gt_records:
        gt_by_frame[r.frame - min_frame].append(r)

    trk_by_frame: Dict[int, List[MOTRecord]] = defaultdict(list)
    for r in tracker_records:
        trk_by_frame[r.frame - min_frame].append(r)

    # Map unique GT IDs to contiguous integers 0..num_gt_ids-1
    gt_unique_ids = sorted(list(set(r.track_id for r in gt_records)))
    gt_id_map = {uid: idx for idx, uid in enumerate(gt_unique_ids)}

    # Map unique Tracker IDs to contiguous integers 0..num_tracker_ids-1
    trk_unique_ids = sorted(list(set(r.track_id for r in tracker_records)))
    trk_id_map = {uid: idx for idx, uid in enumerate(trk_unique_ids)}

    gt_ids_list: List[np.ndarray] = []
    trk_ids_list: List[np.ndarray] = []
    similarity_scores_list: List[np.ndarray] = []

    total_gt_dets = 0
    total_trk_dets = 0

    for t in range(num_timesteps):
        gts = gt_by_frame.get(t, [])
        trks = trk_by_frame.get(t, [])

        total_gt_dets += len(gts)
        total_trk_dets += len(trks)

        g_ids = np.array([gt_id_map[r.track_id] for r in gts], dtype=int) if gts else np.empty(0, dtype=int)
        t_ids = np.array([trk_id_map[r.track_id] for r in trks], dtype=int) if trks else np.empty(0, dtype=int)

        gt_ids_list.append(g_ids)
        trk_ids_list.append(t_ids)

        if gts and trks:
            boxes_g = np.array([r.bbox for r in gts], dtype=np.float64)
            boxes_t = np.array([r.bbox for r in trks], dtype=np.float64)
            sim = compute_iou_matrix(boxes_g, boxes_t)
        else:
            sim = np.empty((len(gts), len(trks)), dtype=np.float64)

        similarity_scores_list.append(sim)

    return {
        "num_timesteps": num_timesteps,
        "num_gt_ids": len(gt_unique_ids),
        "num_tracker_ids": len(trk_unique_ids),
        "num_gt_dets": total_gt_dets,
        "num_tracker_dets": total_trk_dets,
        "gt_ids": gt_ids_list,
        "tracker_ids": trk_ids_list,
        "similarity_scores": similarity_scores_list,
    }


def evaluate_sequences(
    gt_records: Sequence[MOTRecord],
    tracker_records: Sequence[MOTRecord],
) -> Dict[str, float]:
    """Calculate HOTA, MOTA, IDF1, ID switches, Fragmentation, DetA, AssA using TrackEval.

    Args:
        gt_records: Ground truth records in MOTChallenge format.
        tracker_records: Tracker prediction records in MOTChallenge format.

    Returns:
        Dictionary of computed metric values.
    """
    data = prepare_trackeval_data(gt_records, tracker_records)

    hota_metric = HOTA({"PRINT_CONFIG": False})
    clear_metric = CLEAR({"PRINT_CONFIG": False})
    identity_metric = Identity({"PRINT_CONFIG": False})

    hota_res = hota_metric.eval_sequence(data)
    clear_res = clear_metric.eval_sequence(data)
    identity_res = identity_metric.eval_sequence(data)

    hota_score = float(np.mean(hota_res["HOTA"])) if len(hota_res.get("HOTA", [])) > 0 else 0.0
    deta_score = float(np.mean(hota_res["DetA"])) if len(hota_res.get("DetA", [])) > 0 else 0.0
    assa_score = float(np.mean(hota_res["AssA"])) if len(hota_res.get("AssA", [])) > 0 else 0.0

    mota_score = float(clear_res.get("MOTA", 0.0))
    idsw_count = int(clear_res.get("IDSW", 0))
    frag_count = int(clear_res.get("Frag", 0))
    idf1_score = float(identity_res.get("IDF1", 0.0))

    return {
        "HOTA": hota_score,
        "MOTA": mota_score,
        "IDF1": idf1_score,
        "IDSW": float(idsw_count),
        "Frag": float(frag_count),
        "DetA": deta_score,
        "AssA": assa_score,
    }

