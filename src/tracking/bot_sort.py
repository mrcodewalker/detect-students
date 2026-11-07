"""BoT-SORT tracker adapter.

Adapted from:
    https://github.com/NirAharon/BoT-SORT
    Reference: Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian Tracker", arXiv 2022.
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


def tlbr_to_xywh(tlbr: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert (x1, y1, x2, y2) to (center_x, center_y, width, height)."""
    x1, y1, x2, y2 = tlbr
    w = max(1e-4, x2 - x1)
    h = max(1e-4, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return np.array([cx, cy, w, h], dtype=np.float64)


def xywh_to_tlbr(xywh: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert (center_x, center_y, width, height) to (x1, y1, x2, y2)."""
    cx, cy, w, h = xywh[:4]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return float(x1), float(y1), float(x2), float(y2)


class BoTSORTKalmanFilter:
    """Kalman filter for BoT-SORT estimating [x, y, w, h, vx, vy, vw, vh] directly."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize state from an unassociated [cx, cy, w, h] measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        pred_mean = np.dot(self._motion_mat, mean)
        pred_cov = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return pred_mean, pred_cov

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project state distribution into measurement space."""
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        meas_cov = np.diag(np.square(std))
        proj_mean = np.dot(self._update_mat, mean)
        proj_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + meas_cov
        return proj_mean, proj_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman filter correction step with new measurement."""
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


class BoTTrack:
    """Individual tracklet for BoT-SORT."""

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
        self.kalman_filter: Optional[BoTSORTKalmanFilter] = None

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

    def activate(self, kalman_filter: BoTSORTKalmanFilter, frame_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        meas = tlbr_to_xywh(self._bbox)
        self.mean, self.covariance = self.kalman_filter.initiate(meas)
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "BoTTrack", frame_id: int) -> None:
        assert self.kalman_filter is not None
        meas = tlbr_to_xywh(new_track.bbox)
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
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self._bbox = xywh_to_tlbr(self.mean[:4])

    def update(self, new_track: "BoTTrack", frame_id: int) -> None:
        assert self.kalman_filter is not None
        self.frame_id = frame_id
        self.tracklet_len += 1
        meas = tlbr_to_xywh(new_track.bbox)
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, meas)
        self._bbox = new_track.bbox
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new_track.score

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


def compute_iou_cost(tracks_a: Sequence[BoTTrack], tracks_b: Sequence[BoTTrack]) -> np.ndarray:
    """Calculate 1 - IoU distance matrix between BoTTrack lists."""
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


class BoTSORTTracker(Tracker):
    """Adapter wrapping the BoT-SORT multi-object tracking algorithm."""

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        cmc_method: str = "none",
    ) -> None:
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.track_buffer = int(track_buffer)
        self.match_thresh = float(match_thresh)
        self.proximity_thresh = float(proximity_thresh)
        self.cmc_method = cmc_method

        self.kalman_filter = BoTSORTKalmanFilter()
        self.tracked_stracks: List[BoTTrack] = []
        self.lost_stracks: List[BoTTrack] = []
        self.removed_stracks: List[BoTTrack] = []
        self.frame_id: int = 0

    @classmethod
    def from_config(cls, config: Union[str, Path, Dict[str, Any]]) -> "BoTSORTTracker":
        raw_cfg = load_config(config) if isinstance(config, (str, Path)) else config
        tracker_cfg = raw_cfg.get("tracker", raw_cfg)
        return cls(
            track_high_thresh=tracker_cfg.get("track_high_thresh", 0.5),
            track_low_thresh=tracker_cfg.get("track_low_thresh", 0.1),
            new_track_thresh=tracker_cfg.get("new_track_thresh", 0.6),
            track_buffer=tracker_cfg.get("track_buffer", 30),
            match_thresh=tracker_cfg.get("match_thresh", 0.8),
            proximity_thresh=tracker_cfg.get("proximity_thresh", 0.5),
            cmc_method=tracker_cfg.get("cmc_method", "none"),
        )

    def update(
        self,
        detections: Sequence[Detection],
        frame: Optional[np.ndarray] = None,
        keypoints: Optional[Any] = None,
    ) -> List[TrackOutput]:
        self.frame_id += 1

        dets_high: List[BoTTrack] = []
        dets_low: List[BoTTrack] = []
        for det in detections:
            bbox = det.bbox if hasattr(det, "bbox") else det[0]
            score = float(det.score) if hasattr(det, "score") else float(det[1])
            t = BoTTrack(bbox, score)
            if score >= self.track_high_thresh:
                dets_high.append(t)
            elif score >= self.track_low_thresh:
                dets_low.append(t)

        for trk in self.tracked_stracks:
            trk.predict()
        for trk in self.lost_stracks:
            trk.predict()

        # Stage 1: match high-confidence detections
        cost_mat = compute_iou_cost(self.tracked_stracks, dets_high)
        row_ind, col_ind = linear_sum_assignment(cost_mat) if cost_mat.size > 0 else ([], [])

        matches_1: List[Tuple[int, int]] = []
        u_tracks_1 = list(range(len(self.tracked_stracks)))
        u_dets_high = list(range(len(dets_high)))

        for r, c in zip(row_ind, col_ind):
            if cost_mat[r, c] <= self.match_thresh:
                matches_1.append((r, c))
                if r in u_tracks_1:
                    u_tracks_1.remove(r)
                if c in u_dets_high:
                    u_dets_high.remove(c)

        activated_tracks: List[BoTTrack] = []
        refind_tracks: List[BoTTrack] = []

        for r_idx, c_idx in matches_1:
            track = self.tracked_stracks[r_idx]
            det = dets_high[c_idx]
            track.update(det, self.frame_id)
            activated_tracks.append(track)

        # Stage 2: match low-confidence detections
        r_tracks_pool = [self.tracked_stracks[i] for i in u_tracks_1]
        cost_mat_low = compute_iou_cost(r_tracks_pool, dets_low)
        row_ind_2, col_ind_2 = linear_sum_assignment(cost_mat_low) if cost_mat_low.size > 0 else ([], [])

        u_tracks_2 = list(range(len(r_tracks_pool)))
        for r, c in zip(row_ind_2, col_ind_2):
            if cost_mat_low[r, c] <= self.proximity_thresh:
                track = r_tracks_pool[r]
                det = dets_low[c]
                track.update(det, self.frame_id)
                activated_tracks.append(track)
                if r in u_tracks_2:
                    u_tracks_2.remove(r)

        for idx in u_tracks_2:
            track = r_tracks_pool[idx]
            track.mark_lost()
            self.lost_stracks.append(track)

        # Stage 3: match unmatched high-confidence detections with lost tracks
        rem_dets_high = [dets_high[i] for i in u_dets_high]
        cost_mat_lost = compute_iou_cost(self.lost_stracks, rem_dets_high)
        row_ind_3, col_ind_3 = linear_sum_assignment(cost_mat_lost) if cost_mat_lost.size > 0 else ([], [])

        u_lost = list(range(len(self.lost_stracks)))
        u_dets_unmatched = list(range(len(rem_dets_high)))

        for r, c in zip(row_ind_3, col_ind_3):
            if cost_mat_lost[r, c] <= self.match_thresh:
                track = self.lost_stracks[r]
                det = rem_dets_high[c]
                track.re_activate(det, self.frame_id)
                refind_tracks.append(track)
                if r in u_lost:
                    u_lost.remove(r)
                if c in u_dets_unmatched:
                    u_dets_unmatched.remove(c)

        self.lost_stracks = [self.lost_stracks[i] for i in u_lost]

        # Initialize remaining high-score detections as new tracks if above new_track_thresh
        for idx in u_dets_unmatched:
            det = rem_dets_high[idx]
            if det.score >= self.new_track_thresh:
                det.activate(self.kalman_filter, self.frame_id)
                activated_tracks.append(det)

        self.tracked_stracks = activated_tracks + refind_tracks

        # Remove dead tracks
        survived_lost: List[BoTTrack] = []
        for trk in self.lost_stracks:
            if self.frame_id - trk.frame_id > self.track_buffer:
                trk.mark_removed()
                self.removed_stracks.append(trk)
            else:
                survived_lost.append(trk)
        self.lost_stracks = survived_lost

        outputs: List[TrackOutput] = []
        for trk in self.tracked_stracks:
            if trk.is_activated:
                outputs.append((trk.track_id, trk.bbox, trk.score))

        return outputs

    def reset(self) -> None:
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        BoTTrack.reset_counter()

