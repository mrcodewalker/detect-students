"""Spatio-temporal action buffer maintaining per-track sliding windows of crops and keypoints."""

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameObservation:
    """Observation data for a single track at a single video frame.

    Attributes:
        frame_idx: 0-based or 1-based video frame index.
        bbox: Bounding box tuple (x1, y1, x2, y2).
        crop: Optional image crop array (H, W, 3).
        keypoints: Optional (17, 3) array [x, y, conf].
        score: Detection/tracking confidence score.
    """

    frame_idx: int
    bbox: Tuple[float, float, float, float]
    crop: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None
    score: float = 1.0


@dataclass
class ActionWindow:
    """Sliding temporal window for a track ready for action classification.

    Attributes:
        track_id: Unique integer tracking ID.
        start_frame: Frame index of the first observation in the window.
        end_frame: Frame index of the last observation in the window.
        frame_indices: List of frame indices in temporal order.
        bboxes: Array of shape (N, 4) with bounding boxes.
        crops: Optional array of shape (N, H, W, 3) or None.
        keypoints: Optional array of shape (N, 17, 3) or None.
        scores: Array of shape (N,) with detection/tracking scores.
    """

    track_id: int
    start_frame: int
    end_frame: int
    frame_indices: List[int]
    bboxes: np.ndarray
    crops: Optional[np.ndarray]
    keypoints: Optional[np.ndarray]
    scores: np.ndarray


class SingleTrackBuffer:
    """Buffer maintaining recent observations for a single track ID."""

    def __init__(self, track_id: int, window_size: int = 16) -> None:
        self.track_id = int(track_id)
        self.window_size = int(window_size)
        self.buffer: Deque[FrameObservation] = deque(maxlen=self.window_size)
        self.last_seen_frame: int = -1
        self.last_emitted_frame: Optional[int] = None

    def add(self, observation: FrameObservation) -> None:
        """Append an observation and update last seen frame."""
        self.buffer.append(observation)
        self.last_seen_frame = observation.frame_idx

    def is_ready(self, current_frame_idx: int, stride: int) -> bool:
        """Check if buffer has a full window and satisfies the stride timing."""
        if len(self.buffer) < self.window_size:
            return False

        if self.last_emitted_frame is None:
            return True

        return (current_frame_idx - self.last_emitted_frame) >= stride

    def get_window(self, mark_emitted: bool = True) -> Optional[ActionWindow]:
        """Extract ActionWindow from current buffer."""
        if len(self.buffer) < self.window_size:
            return None

        observations = list(self.buffer)
        frame_indices = [obs.frame_idx for obs in observations]
        bboxes = np.array([obs.bbox for obs in observations], dtype=np.float32)
        scores = np.array([obs.score for obs in observations], dtype=np.float32)

        # Process crops
        has_all_crops = all(obs.crop is not None for obs in observations)
        if has_all_crops:
            crops_array = np.stack([obs.crop for obs in observations], axis=0)
        else:
            crops_array = None

        # Process keypoints: if available, pack into (N, 17, 3)
        has_kpts = any(obs.keypoints is not None for obs in observations)
        if has_kpts:
            kpt_list = []
            for obs in observations:
                if obs.keypoints is not None:
                    kpt_list.append(obs.keypoints)
                else:
                    kpt_list.append(np.zeros((17, 3), dtype=np.float32))
            keypoints_array = np.stack(kpt_list, axis=0)
        else:
            keypoints_array = None

        end_frame = frame_indices[-1]
        if mark_emitted:
            self.last_emitted_frame = end_frame

        return ActionWindow(
            track_id=self.track_id,
            start_frame=frame_indices[0],
            end_frame=end_frame,
            frame_indices=frame_indices,
            bboxes=bboxes,
            crops=crops_array,
            keypoints=keypoints_array,
            scores=scores,
        )


class TrackActionBuffer:
    """Manages spatio-temporal buffers across multiple active track IDs.

    Maintains the last N frames of box crop and keypoints for each track,
    expires tracks inactive for more than K frames, and yields ready windows
    every S frames.

    Args:
        window_size: Temporal length N of each action window.
        max_missing_frames: Expiry threshold K. Tracks inactive > K frames are deleted.
        stride: Temporal step S between successive emitted windows.
        crop_size: Optional (width, height) to resize person crops.
    """

    def __init__(
        self,
        window_size: int = 16,
        max_missing_frames: int = 30,
        stride: int = 4,
        crop_size: Optional[Tuple[int, int]] = (112, 112),
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if max_missing_frames < 0:
            raise ValueError(f"max_missing_frames must be non-negative, got {max_missing_frames}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.window_size = int(window_size)
        self.max_missing_frames = int(max_missing_frames)
        self.stride = int(stride)
        self.crop_size = crop_size

        self.buffers: Dict[int, SingleTrackBuffer] = {}

    def __len__(self) -> int:
        """Return number of currently tracked buffers."""
        return len(self.buffers)

    def __contains__(self, track_id: int) -> bool:
        """Check if track ID exists in buffer collection."""
        return int(track_id) in self.buffers

    def reset(self) -> None:
        """Reset all internal track buffers."""
        self.buffers.clear()

    def _extract_crop(
        self,
        frame: Optional[np.ndarray],
        bbox: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """Extract and optionally resize bounding box crop from a video frame."""
        if frame is None:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        ix1 = max(0, min(w - 1, int(x1)))
        iy1 = max(0, min(h - 1, int(y1)))
        ix2 = max(0, min(w, int(x2)))
        iy2 = max(0, min(h, int(y2)))

        if (ix2 - ix1) <= 0 or (iy2 - iy1) <= 0:
            if self.crop_size is not None:
                return np.zeros((self.crop_size[1], self.crop_size[0], 3), dtype=np.uint8)
            return np.zeros((1, 1, 3), dtype=np.uint8)

        crop = frame[iy1:iy2, ix1:ix2]
        if self.crop_size is not None:
            crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)
        return crop

    def update_track(
        self,
        track_id: int,
        frame_idx: int,
        bbox: Tuple[float, float, float, float],
        score: float = 1.0,
        frame: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
    ) -> None:
        """Add an observation for a given track ID, instantiating buffer if new.

        Args:
            track_id: Unique integer ID of the tracked person.
            frame_idx: Index of current video frame.
            bbox: Bounding box tuple (x1, y1, x2, y2).
            score: Confidence score.
            frame: Optional full BGR frame array for crop extraction.
            keypoints: Optional (17, 3) keypoints array.
        """
        tid = int(track_id)
        if tid not in self.buffers:
            self.buffers[tid] = SingleTrackBuffer(tid, window_size=self.window_size)

        crop = self._extract_crop(frame, bbox) if frame is not None else None

        obs = FrameObservation(
            frame_idx=frame_idx,
            bbox=bbox,
            crop=crop,
            keypoints=keypoints,
            score=score,
        )
        self.buffers[tid].add(obs)

    def remove_expired(self, current_frame_idx: int) -> List[int]:
        """Prune tracks that disappeared for more than max_missing_frames (K).

        Args:
            current_frame_idx: Current video frame index.

        Returns:
            List of pruned track IDs.
        """
        expired: List[int] = []
        for tid, buf in list(self.buffers.items()):
            missing_frames = current_frame_idx - buf.last_seen_frame
            if missing_frames > self.max_missing_frames:
                expired.append(tid)
                del self.buffers[tid]
        return expired

    def is_window_ready(self, track_id: int, current_frame_idx: int) -> bool:
        """Check whether a sliding window is ready for the specified track ID.

        Args:
            track_id: Target track ID.
            current_frame_idx: Current frame index.

        Returns:
            True if buffer has N frames and >= S frames elapsed since last window.
        """
        tid = int(track_id)
        if tid not in self.buffers:
            return False
        return self.buffers[tid].is_ready(current_frame_idx, self.stride)

    def get_window(
        self,
        track_id: int,
        current_frame_idx: Optional[int] = None,
        force: bool = False,
    ) -> Optional[ActionWindow]:
        """Retrieve ready window for track ID.

        Args:
            track_id: Target track ID.
            current_frame_idx: Current frame index for timing check.
            force: If True, bypasses the stride timing check (requires full buffer).

        Returns:
            ActionWindow if ready (or forced), else None.
        """
        tid = int(track_id)
        if tid not in self.buffers:
            return None

        buf = self.buffers[tid]
        if not force:
            f_idx = current_frame_idx if current_frame_idx is not None else buf.last_seen_frame
            if not buf.is_ready(f_idx, self.stride):
                return None

        return buf.get_window(mark_emitted=True)

    def get_all_ready_windows(self, current_frame_idx: int) -> List[ActionWindow]:
        """Extract all ready action windows across all active tracks at current frame.

        Args:
            current_frame_idx: Current video frame index.

        Returns:
            List of ActionWindow ready for inference.
        """
        windows: List[ActionWindow] = []
        for tid, buf in self.buffers.items():
            if buf.is_ready(current_frame_idx, self.stride):
                win = buf.get_window(mark_emitted=True)
                if win is not None:
                    windows.append(win)
        return windows

    def step(
        self,
        frame_idx: int,
        tracks: Sequence[Tuple[int, Tuple[float, float, float, float], float]],
        frame: Optional[np.ndarray] = None,
        keypoints_map: Optional[Dict[int, np.ndarray]] = None,
    ) -> List[ActionWindow]:
        """Convenience method combining update, expiry removal, and ready window retrieval.

        Args:
            frame_idx: Current frame index.
            tracks: Sequence of (track_id, bbox_xyxy, score).
            frame: Optional full frame image for crops.
            keypoints_map: Optional mapping track_id -> (17, 3) keypoints.

        Returns:
            List of ready ActionWindow objects for this time step.
        """
        # 1. Update tracks
        for tid, bbox, score in tracks:
            kpts = keypoints_map.get(tid) if keypoints_map else None
            self.update_track(
                track_id=tid,
                frame_idx=frame_idx,
                bbox=bbox,
                score=score,
                frame=frame,
                keypoints=kpts,
            )

        # 2. Prune expired tracks
        self.remove_expired(frame_idx)

        # 3. Retrieve all ready windows
        return self.get_all_ready_windows(frame_idx)
