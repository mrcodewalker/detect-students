"""OC-SORT (Observation-Centric SORT) tracker adapter.

Adapted from:
    https://github.com/noahcao/OC_SORT
    Reference: Cao et al., "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking", CVPR 2023.
    License: MIT
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detection.detection import Detection
from src.tracking.base import Tracker, TrackOutput
from src.utils.config import load_config


def kf_bbox_to_z(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert [x1, y1, x2, y2] box to [center_x, center_y, scale/area, aspect_ratio]."""
    x1, y1, x2, y2 = bbox
    w = max(1e-4, x2 - x1)
    h = max(1e-4, y2 - y1)
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def kf_x_to_bbox(x: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert state vector [center_x, center_y, scale, aspect_ratio] to [x1, y1, x2, y2]."""
    x_flat = x.flatten()
    w = np.sqrt(max(1e-4, x_flat[2] * x_flat[3]))
    h = max(1e-4, x_flat[2] / w)
    x1 = float(x_flat[0] - w / 2.0)
    y1 = float(x_flat[1] - h / 2.0)
    x2 = float(x_flat[0] + w / 2.0)
    y2 = float(x_flat[1] + h / 2.0)
    return x1, y1, x2, y2


class KalmanBoxTrackerOC:
    """Kalman Filter with Observation-Centric Online Smoothing (OOS) for OC-SORT."""

    count: int = 0

    def __init__(self, bbox: Tuple[float, float, float, float], score: float, delta_t: int = 3) -> None:
        self.dim_x = 7
        self.dim_z = 4

        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        self.H = np.eye(self.dim_z, self.dim_x)

        self.R = np.eye(self.dim_z)
        self.R[2:, 2:] *= 10.0

        self.P = np.eye(self.dim_x)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0

        self.Q = np.eye(self.dim_x)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros((self.dim_x, 1))
        self.x[:4] = kf_bbox_to_z(bbox)

        self.time_since_update: int = 0
        KalmanBoxTrackerOC.count += 1
        self.id: int = KalmanBoxTrackerOC.count
        self.hits: int = 0
        self.hit_streak: int = 0
        self.age: int = 0
        self.score: float = float(score)
        self.delta_t = delta_t

        # Track observation history for OOS and OCM: (frame_idx, bbox)
        self.observations: Dict[int, Tuple[float, float, float, float]] = {0: bbox}
        self.last_observation: Tuple[float, float, float, float] = bbox
        self.last_frame: int = 0

    def predict(self) -> Tuple[float, float, float, float]:
        """Kalman state prediction step."""
        if (self.x[6] + self.x[2]) <= 0:
            self.x[6] = 0.0
        self.x = np.dot(self.F, self.x)
        self.P = np.linalg.multi_dot((self.F, self.P, self.F.T)) + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return kf_x_to_bbox(self.x)

    def update(self, bbox: Tuple[float, float, float, float], score: float, frame_idx: int) -> None:
        """Update Kalman filter state with new measurement and perform Online Smoothing."""
        self.hits += 1
        self.hit_streak += 1
        self.score = float(score)

        # Observation-Centric Online Smoothing (OOS) when recovered after occlusion
        if self.time_since_update > 1 and len(self.observations) > 0:
            # Re-smooth state across the missed gap
            prev_bbox = self.last_observation
            gap = self.time_since_update
            dx = (bbox[0] - prev_bbox[0]) / gap
            dy = (bbox[1] - prev_bbox[1]) / gap
            # Update estimated linear velocities
            self.x[4] = dx
            self.x[5] = dy

        self.time_since_update = 0
        z = kf_bbox_to_z(bbox)
        y = z - np.dot(self.H, self.x)
        S = np.linalg.multi_dot((self.H, self.P, self.H.T)) + self.R
        K = np.linalg.multi_dot((self.P, self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.linalg.multi_dot((K, self.H, self.P))

        self.last_observation = bbox
        self.last_frame = frame_idx
        self.observations[frame_idx] = bbox

    def get_state(self) -> Tuple[float, float, float, float]:
        return kf_x_to_bbox(self.x)

    def get_velocity(self) -> np.ndarray:
        """Return [vx, vy] velocity vector."""
        return self.x[4:6].flatten()


def compute_iou_cost(
    boxes_a: Sequence[Tuple[float, float, float, float]],
    boxes_b: Sequence[Tuple[float, float, float, float]],
) -> np.ndarray:
    """Compute IoU cost matrix (1 - IoU)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = np.array(boxes_a, dtype=np.float32)
    b = np.array(boxes_b, dtype=np.float32)

    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    inter = w * h

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return 1.0 - (inter / np.maximum(union, 1e-6))


class OCSORTTracker(Tracker):
    """Adapter wrapping the OC-SORT (Observation-Centric SORT) algorithm."""

    def __init__(
        self,
        det_thresh: float = 0.5,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        inertia: float = 0.2,
        use_byte: bool = True,
    ) -> None:
        self.det_thresh = float(det_thresh)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.delta_t = int(delta_t)
        self.inertia = float(inertia)
        self.use_byte = bool(use_byte)

        self.trackers: List[KalmanBoxTrackerOC] = []
        self.frame_count: int = 0

    @classmethod
    def from_config(cls, config: Union[str, Path, Dict[str, Any]]) -> "OCSORTTracker":
        raw_cfg = load_config(config) if isinstance(config, (str, Path)) else config
        tracker_cfg = raw_cfg.get("tracker", raw_cfg)
        return cls(
            det_thresh=tracker_cfg.get("det_thresh", 0.5),
            max_age=tracker_cfg.get("max_age", 30),
            min_hits=tracker_cfg.get("min_hits", 3),
            iou_threshold=tracker_cfg.get("iou_threshold", 0.3),
            delta_t=tracker_cfg.get("delta_t", 3),
            inertia=tracker_cfg.get("inertia", 0.2),
            use_byte=tracker_cfg.get("use_byte", True),
        )

    def update(
        self,
        detections: Sequence[Detection],
        frame: Optional[np.ndarray] = None,
        keypoints: Optional[Any] = None,
    ) -> List[TrackOutput]:
        self.frame_count += 1

        # Predict existing trackers
        pred_boxes: List[Tuple[float, float, float, float]] = []
        to_del: List[int] = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(i)
            else:
                pred_boxes.append(pos)
        for i in reversed(to_del):
            self.trackers.pop(i)

        # Split detections for two-stage association
        dets_first = [d for d in detections if d.score >= self.det_thresh] if self.use_byte else list(detections)
        dets_second = [d for d in detections if d.score < self.det_thresh] if self.use_byte else []

        boxes_first = [d.bbox for d in dets_first]
        scores_first = [float(d.score) for d in dets_first]

        matched_trks: List[int] = []
        matched_dets: List[int] = []

        # Stage 1: match high-confidence detections
        if len(self.trackers) > 0 and len(boxes_first) > 0:
            cost_mat = compute_iou_cost(pred_boxes, boxes_first)

            # Apply Observation-Centric Momentum (OCM)
            if self.inertia > 0:
                for t_idx, trk in enumerate(self.trackers):
                    v_trk = trk.get_velocity()
                    norm_v = np.linalg.norm(v_trk)
                    if norm_v > 1e-3:
                        dir_trk = v_trk / norm_v
                        for d_idx, box_d in enumerate(boxes_first):
                            # Displacement from last observation
                            cx_d = (box_d[0] + box_d[2]) / 2.0
                            cy_d = (box_d[1] + box_d[3]) / 2.0
                            cx_last = (trk.last_observation[0] + trk.last_observation[2]) / 2.0
                            cy_last = (trk.last_observation[1] + trk.last_observation[3]) / 2.0
                            v_det = np.array([cx_d - cx_last, cy_d - cy_last])
                            norm_det = np.linalg.norm(v_det)
                            if norm_det > 1e-3:
                                dir_det = v_det / norm_det
                                cos_sim = np.dot(dir_trk, dir_det)
                                # Penalize opposing velocity directions
                                cost_mat[t_idx, d_idx] += self.inertia * (1.0 - cos_sim)

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            max_cost = 1.0 - self.iou_threshold
            for r, c in zip(row_ind, col_ind):
                if cost_mat[r, c] <= max_cost:
                    self.trackers[r].update(boxes_first[c], scores_first[c], self.frame_count)
                    matched_trks.append(r)
                    matched_dets.append(c)

        unmatched_trks = [i for i in range(len(self.trackers)) if i not in matched_trks]
        unmatched_dets = [c for c in range(len(boxes_first)) if c not in matched_dets]

        # Stage 2: associate unmatched tracks with low-confidence detections
        if self.use_byte and len(unmatched_trks) > 0 and len(dets_second) > 0:
            boxes_second = [d.bbox for d in dets_second]
            scores_second = [float(d.score) for d in dets_second]
            sub_pred_boxes = [pred_boxes[i] for i in unmatched_trks]

            cost_mat_low = compute_iou_cost(sub_pred_boxes, boxes_second)
            row_ind_2, col_ind_2 = linear_sum_assignment(cost_mat_low)
            max_cost_low = 1.0 - (self.iou_threshold * 0.5)

            for r, c in zip(row_ind_2, col_ind_2):
                if cost_mat_low[r, c] <= max_cost_low:
                    real_trk_idx = unmatched_trks[r]
                    self.trackers[real_trk_idx].update(boxes_second[c], scores_second[c], self.frame_count)
                    matched_trks.append(real_trk_idx)

        # Initialize new trackers for unmatched detections
        for c in unmatched_dets:
            self.trackers.append(KalmanBoxTrackerOC(boxes_first[c], scores_first[c], self.delta_t))

        # Filter output tracks and remove stale ones
        outputs: List[TrackOutput] = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                outputs.append((trk.id, d, trk.score))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return outputs

    def reset(self) -> None:
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTrackerOC.count = 0

