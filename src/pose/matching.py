"""Matching utilities to associate 2D pose estimations with object detections."""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detection.detection import Detection
from src.pose.base import PoseDetection


def compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2).

    Args:
        box1: (x1, y1, x2, y2).
        box2: (x1, y1, x2, y2).

    Returns:
        IoU value in [0.0, 1.0].
    """
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def compute_iou_matrix(
    boxes1: Sequence[Tuple[float, float, float, float]],
    boxes2: Sequence[Tuple[float, float, float, float]],
) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of bounding boxes.

    Args:
        boxes1: Array or list of shape (N, 4).
        boxes2: Array or list of shape (M, 4).

    Returns:
        Array of shape (N, M) containing IoU values.
    """
    n, m = len(boxes1), len(boxes2)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    iou_mat = np.zeros((n, m), dtype=np.float32)
    for i, b1 in enumerate(boxes1):
        for j, b2 in enumerate(boxes2):
            iou_mat[i, j] = compute_iou(b1, b2)
    return iou_mat


def match_pose_to_detections(
    detections: Sequence[Union[Detection, Tuple[float, float, float, float]]],
    poses: Sequence[PoseDetection],
    iou_threshold: float = 0.3,
) -> List[Optional[PoseDetection]]:
    """Match each detection to a PoseDetection via bounding box IoU.

    Uses optimal Hungarian bipartite matching to associate each detection box
    with the most overlapping pose estimation.

    Args:
        detections: Sequence of Detection instances or bbox tuples (x1, y1, x2, y2).
        poses: Sequence of PoseDetection instances.
        iou_threshold: Minimum IoU required for a valid association.

    Returns:
        List of length equal to len(detections), where element i is either
        the matched PoseDetection or None if no match above iou_threshold was found.
    """
    n_det = len(detections)
    n_pose = len(poses)

    if n_det == 0:
        return []
    if n_pose == 0:
        return [None] * n_det

    det_boxes = [d.bbox if isinstance(d, Detection) else d for d in detections]
    pose_boxes = [p.bbox for p in poses]

    iou_mat = compute_iou_matrix(det_boxes, pose_boxes)
    # Hungarian algorithm minimizes cost, so cost = -IoU
    row_ind, col_ind = linear_sum_assignment(-iou_mat)

    matched: List[Optional[PoseDetection]] = [None] * n_det
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= iou_threshold:
            matched[r] = poses[c]

    return matched

