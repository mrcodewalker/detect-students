"""ByteTrack multi-object tracker adapter.

Adapted from:
    https://github.com/ifzhang/ByteTrack
    Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
    License: MIT
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment

from src.detection.detection import Detection
from src.tracking.base import Tracker, TrackOutput
from src.utils.config import load_config


class TrackState(Enum):
    """Lifecycle states of a tracked object."""

    NEW = 1
    TRACKED = 2
    LOST = 3
    REMOVED = 4


def tlbr_to_xyah(tlbr: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert (x1, y1, x2, y2) to (center_x, center_y, aspect_ratio, height)."""
    x1, y1, x2, y2 = tlbr
    w = max(1e-4, x2 - x1)
    h = max(1e-4, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    a = w / h
    return np.array([cx, cy, a, h], dtype=np.float64)


def xyah_to_tlbr(xyah: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert (center_x, center_y, aspect_ratio, height) to (x1, y1, x2, y2)."""
    cx, cy, a, h = xyah[:4]
    w = a * h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return float(x1), float(y1), float(x2), float(y2)


class KalmanFilter:
    """Standard Kalman filter for 2D bounding box motion tracking."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize state from an unassociated measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        pred_mean = np.dot(self._motion_mat, mean)
        pred_cov = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return pred_mean, pred_cov

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project state distribution into measurement space."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        meas_cov = np.diag(np.square(std))
        proj_mean = np.dot(self._update_mat, mean)
        proj_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + meas_cov
        return proj_mean, proj_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman filter update step with new measurement."""
        proj_mean, proj_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - proj_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_cov = covariance - np.linalg.multi_dot((kalman_gain, proj_cov, kalman_gain.T))
        return new_mean, new_cov


class STrack:
    """Single object tracklet representation for ByteTrack."""

    _count: int = 0

    def __init__(self, bbox: Tuple[float, float, float, float], score: float) -> None:
        self._bbox = bbox
        self.score = float(score)
        self.state = TrackState.NEW
        self.is_activated = False

        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.kalman_filter: Optional[KalmanFilter] = None

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self._bbox

    @classmethod
    def reset_counter(cls) -> None:
        cls._count = 0

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        meas = tlbr_to_xyah(self._bbox)
        self.mean, self.covariance = self.kalman_filter.initiate(meas)
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "STrack", frame_id: int) -> None:
        assert self.kalman_filter is not None
        meas = tlbr_to_xyah(new_track.bbox)
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, meas)
        self._bbox = new_track.bbox
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score

    def predict(self) -> None:
        if self.mean is None or self.covariance is None or self.kalman_filter is None:
            return
        if self.state != TrackState.TRACKED:
            self.mean[7] = 0.0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self._bbox = xyah_to_tlbr(self.mean[:4])

    def update(self, new_track: "STrack", frame_id: int) -> None:
        assert self.kalman_filter is not None
        self.frame_id = frame_id
        self.tracklet_len += 1
        meas = tlbr_to_xyah(new_track.bbox)
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, meas)
        self._bbox = new_track.bbox
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new_track.score

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


def compute_iou_cost(
    tracks_a: Sequence[STrack],
    tracks_b: Sequence[STrack],
) -> np.ndarray:
    """Compute (1 - IoU) cost matrix between two STrack sequences."""
    if len(tracks_a) == 0 or len(tracks_b) == 0:
        return np.zeros((len(tracks_a), len(tracks_b)), dtype=np.float32)

    boxes_a = np.array([t.bbox for t in tracks_a], dtype=np.float32)
    boxes_b = np.array([t.bbox for t in tracks_b], dtype=np.float32)

    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    inter = w * h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    ious = inter / np.maximum(union, 1e-6)
    return 1.0 - ious


def match_tracks(
    cost_matrix: np.ndarray, thresh: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match tracks using Hungarian algorithm bounded by distance threshold."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: List[Tuple[int, int]] = []
    unmatched_a: List[int] = list(range(cost_matrix.shape[0]))
    unmatched_b: List[int] = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            if r in unmatched_a:
                unmatched_a.remove(r)
            if c in unmatched_b:
                unmatched_b.remove(c)

    return matches, unmatched_a, unmatched_b


class ByteTracker(Tracker):
    """Adapter wrapping the ByteTrack multi-object tracking algorithm."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        low_thresh: float = 0.1,
        low_match_thresh: float = 0.5,
        confirm_thresh: int = 3,
        max_lost_frames: int = 30,
    ) -> None:
        """Initialize ByteTracker.

        Args:
            track_thresh: High-confidence detection threshold for 1st stage.
            match_thresh: Maximum IoU distance threshold for 1st stage matching.
            low_thresh: Minimum detection score threshold for 2nd stage.
            low_match_thresh: Maximum IoU distance for 2nd stage matching.
            confirm_thresh: Consecutive hits required before activating new track.
            max_lost_frames: Maximum frames to retain lost tracks.
        """
        self.track_thresh = float(track_thresh)
        self.match_thresh = float(match_thresh)
        self.low_thresh = float(low_thresh)
        self.low_match_thresh = float(low_match_thresh)
        self.confirm_thresh = int(confirm_thresh)
        self.max_lost_frames = int(max_lost_frames)

        self.kalman_filter = KalmanFilter()
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id: int = 0

    @classmethod
    def from_config(cls, config: Union[str, Path, Dict[str, Any]]) -> "ByteTracker":
        """Factory method to instantiate ByteTracker from YAML configuration."""
        raw_cfg = load_config(config) if isinstance(config, (str, Path)) else config
        tracker_cfg = raw_cfg.get("tracker", raw_cfg)
        return cls(
            track_thresh=tracker_cfg.get("track_thresh", 0.5),
            match_thresh=tracker_cfg.get("match_thresh", 0.8),
            low_thresh=tracker_cfg.get("low_thresh", 0.1),
            low_match_thresh=tracker_cfg.get("low_match_thresh", 0.5),
            confirm_thresh=tracker_cfg.get("confirm_thresh", 3),
            max_lost_frames=tracker_cfg.get("max_lost_frames", 30),
        )

    def update(
        self,
        detections: Sequence[Detection],
        frame: Optional[np.ndarray] = None,
        keypoints: Optional[Any] = None,
    ) -> List[TrackOutput]:
        """Run ByteTrack two-stage association step."""
        self.frame_id += 1

        # Step 1: Split detections into high and low score sets
        dets_high: List[STrack] = []
        dets_low: List[STrack] = []
        for det in detections:
            bbox = det.bbox if hasattr(det, "bbox") else det[0]
            score = float(det.score) if hasattr(det, "score") else float(det[1])
            strack = STrack(bbox, score)
            if score >= self.track_thresh:
                dets_high.append(strack)
            elif score >= self.low_thresh:
                dets_low.append(strack)

        # Predict Kalman states
        for trk in self.tracked_stracks:
            trk.predict()
        for trk in self.lost_stracks:
            trk.predict()

        # Step 2: First association (high-score detections with active tracks)
        cost_mat = compute_iou_cost(self.tracked_stracks, dets_high)
        matches_1, u_tracks_1, u_dets_high = match_tracks(cost_mat, self.match_thresh)

        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []

        for r_idx, c_idx in matches_1:
            track = self.tracked_stracks[r_idx]
            det = dets_high[c_idx]
            track.update(det, self.frame_id)
            activated_stracks.append(track)

        # Step 3: Second association (low-score detections with remaining tracks)
        r_tracks_pool = [self.tracked_stracks[i] for i in u_tracks_1]
        cost_mat_low = compute_iou_cost(r_tracks_pool, dets_low)
        matches_2, u_tracks_2, _ = match_tracks(cost_mat_low, self.low_match_thresh)

        for r_idx, c_idx in matches_2:
            track = r_tracks_pool[r_idx]
            det = dets_low[c_idx]
            track.update(det, self.frame_id)
            activated_stracks.append(track)

        # Tracks unmatched after both stages are marked lost
        for idx in u_tracks_2:
            track = r_tracks_pool[idx]
            track.mark_lost()
            self.lost_stracks.append(track)

        # Step 4: Deal with unmatched high-score detections (match with lost tracks)
        remaining_dets_high = [dets_high[i] for i in u_dets_high]
        cost_mat_lost = compute_iou_cost(self.lost_stracks, remaining_dets_high)
        matches_3, u_lost, u_dets_unmatched = match_tracks(cost_mat_lost, self.match_thresh)

        for r_idx, c_idx in matches_3:
            track = self.lost_stracks[r_idx]
            det = remaining_dets_high[c_idx]
            track.re_activate(det, self.frame_id)
            refind_stracks.append(track)

        # Remaining unmatched lost tracks remain lost
        self.lost_stracks = [self.lost_stracks[i] for i in u_lost]

        # Initialize remaining unmatched high-score detections as new tracks
        for idx in u_dets_unmatched:
            det = remaining_dets_high[idx]
            det.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(det)

        # Update tracked tracks list
        self.tracked_stracks = activated_stracks + refind_stracks

        # Remove dead tracks that exceeded max_lost_frames
        survived_lost: List[STrack] = []
        for trk in self.lost_stracks:
            if self.frame_id - trk.frame_id > self.max_lost_frames:
                trk.mark_removed()
                self.removed_stracks.append(trk)
            else:
                survived_lost.append(trk)
        self.lost_stracks = survived_lost

        # Format output: only return activated tracked tracks
        outputs: List[TrackOutput] = []
        for trk in self.tracked_stracks:
            if trk.is_activated:
                outputs.append((trk.track_id, trk.bbox, trk.score))

        return outputs

    def reset(self) -> None:
        """Reset internal tracks and ID counters."""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        STrack.reset_counter()

