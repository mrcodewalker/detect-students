"""SORT (Simple Online and Realtime Tracking) adapter.

Adapted from:
    https://github.com/abewley/sort
    Reference: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
    License: GPL-3.0
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detection.detection import Detection
from src.tracking.base import Tracker, TrackOutput
from src.utils.config import load_config


def convert_bbox_to_z(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert [x1, y1, x2, y2] box to [center_x, center_y, scale/area, aspect_ratio]."""
    x1, y1, x2, y2 = bbox
    w = max(1e-4, x2 - x1)
    h = max(1e-4, y2 - y1)
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray, score: Optional[float] = None) -> Tuple[float, float, float, float]:
    """Convert state vector [center_x, center_y, scale, aspect_ratio] to [x1, y1, x2, y2]."""
    x_flat = x.flatten()
    w = np.sqrt(max(1e-4, x_flat[2] * x_flat[3]))
    h = max(1e-4, x_flat[2] / w)
    x1 = float(x_flat[0] - w / 2.0)
    y1 = float(x_flat[1] - h / 2.0)
    x2 = float(x_flat[0] + w / 2.0)
    y2 = float(x_flat[1] + h / 2.0)
    return x1, y1, x2, y2


class KalmanBoxTracker:
    """Kalman Filter state tracker for bounding box estimation in SORT."""

    count: int = 0

    def __init__(self, bbox: Tuple[float, float, float, float], score: float = 1.0) -> None:
        """Initialize state from an initial bounding box observation."""
        self.dim_x = 7
        self.dim_z = 4

        # State transition matrix
        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        # Measurement projection matrix
        self.H = np.eye(self.dim_z, self.dim_x)

        # Covariance matrices
        self.R = np.eye(self.dim_z)
        self.R[2:, 2:] *= 10.0

        self.P = np.eye(self.dim_x)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0

        self.Q = np.eye(self.dim_x)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros((self.dim_x, 1))
        self.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update: int = 0
        KalmanBoxTracker.count += 1
        self.id: int = KalmanBoxTracker.count
        self.history: List[Tuple[float, float, float, float]] = []
        self.hits: int = 0
        self.hit_streak: int = 0
        self.age: int = 0
        self.score: float = float(score)

    def update(self, bbox: Tuple[float, float, float, float], score: float) -> None:
        """Update state estimate with new measurement."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.score = float(score)

        z = convert_bbox_to_z(bbox)
        y = z - np.dot(self.H, self.x)
        S = np.linalg.multi_dot((self.H, self.P, self.H.T)) + self.R
        K = np.linalg.multi_dot((self.P, self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.linalg.multi_dot((K, self.H, self.P))

    def predict(self) -> Tuple[float, float, float, float]:
        """Advance state vector by one time step."""
        if (self.x[6] + self.x[2]) <= 0:
            self.x[6] = 0.0
        self.x = np.dot(self.F, self.x)
        self.P = np.linalg.multi_dot((self.F, self.P, self.F.T)) + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        pred_box = convert_x_to_bbox(self.x)
        self.history.append(pred_box)
        return pred_box

    def get_state(self) -> Tuple[float, float, float, float]:
        """Return current bounding box estimate."""
        return convert_x_to_bbox(self.x)


def compute_iou_matrix(
    boxes_a: Sequence[Tuple[float, float, float, float]],
    boxes_b: Sequence[Tuple[float, float, float, float]],
) -> np.ndarray:
    """Compute IoU matrix between two sets of (x1, y1, x2, y2) bounding boxes."""
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
    intersection = w * h

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection

    return intersection / np.maximum(union, 1e-6)


class SORTTracker(Tracker):
    """Adapter wrapping the SORT tracking algorithm."""

    def __init__(
        self,
        max_age: int = 1,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        """Initialize SORT tracker.

        Args:
            max_age: Maximum frames to retain a track without updates.
            min_hits: Minimum detections before track is reported.
            iou_threshold: Minimum IoU overlap for bipartite matching.
        """
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)

        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count: int = 0

    @classmethod
    def from_config(cls, config: Union[str, Path, Dict[str, Any]]) -> "SORTTracker":
        """Factory method to instantiate SORT from YAML configuration."""
        raw_cfg = load_config(config) if isinstance(config, (str, Path)) else config
        tracker_cfg = raw_cfg.get("tracker", raw_cfg)
        return cls(
            max_age=tracker_cfg.get("max_age", 1),
            min_hits=tracker_cfg.get("min_hits", 3),
            iou_threshold=tracker_cfg.get("iou_threshold", 0.3),
        )

    def update(
        self,
        detections: Sequence[Detection],
        frame: Optional[np.ndarray] = None,
        keypoints: Optional[Any] = None,
    ) -> List[TrackOutput]:
        """Update tracks using detections from current frame."""
        self.frame_count += 1

        # Predict new locations of existing trackers
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

        det_boxes = [d.bbox for d in detections]
        det_scores = [float(d.score) for d in detections]

        # Bipartite matching via Hungarian algorithm
        if len(self.trackers) > 0 and len(det_boxes) > 0:
            iou_mat = compute_iou_matrix(pred_boxes, det_boxes)
            row_ind, col_ind = linear_sum_assignment(-iou_mat)

            matched_track_indices = set()
            matched_det_indices = set()
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_threshold:
                    self.trackers[r].update(det_boxes[c], det_scores[c])
                    matched_track_indices.add(r)
                    matched_det_indices.add(c)

            # Create new trackers for unmatched detections
            for c in range(len(det_boxes)):
                if c not in matched_det_indices:
                    self.trackers.append(KalmanBoxTracker(det_boxes[c], det_scores[c]))
        else:
            for d in detections:
                self.trackers.append(KalmanBoxTracker(d.bbox, d.score))

        # Collect active tracks to return
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
        """Reset all active tracks and internal counters."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
