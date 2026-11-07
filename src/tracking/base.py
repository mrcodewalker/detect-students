"""Abstract tracker interface and dummy testing implementation."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.detection.detection import Detection


TrackOutput = Tuple[int, Tuple[float, float, float, float], float]


class Tracker(ABC):
    """Abstract base class for all multi-object trackers."""

    @abstractmethod
    def update(
        self,
        detections: Sequence[Detection],
        frame: Optional[np.ndarray] = None,
        keypoints: Optional[Any] = None,
    ) -> List[TrackOutput]:
        """Update tracker state with new frame detections.

        Args:
            detections: Sequence of Detection instances in current frame.
            frame: Optional RGB/BGR image array for visual appearance / ReID extraction.
            keypoints: Optional 2D pose keypoints for pose-guided tracking.

        Returns:
            List of tuples: (track_id, bbox_xyxy, score).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal track states and ID counters to initial conditions."""
        pass


class DummyTracker(Tracker):
    """Trivial reference tracker assigning a unique incrementing ID to every detection."""

    def __init__(self, start_id: int = 1) -> None:
        """Initialize DummyTracker.

        Args:
            start_id: Initial ID to assign to the first detection.
        """
        self._start_id = start_id
        self._next_id = start_id

    def update(
        self,
        detections: Sequence[Detection],
        frame: Optional[np.ndarray] = None,
        keypoints: Optional[Any] = None,
    ) -> List[TrackOutput]:
        """Assign a brand new incrementing ID to each incoming detection.

        Args:
            detections: Current frame detections.
            frame: Optional image (unused by dummy tracker).
            keypoints: Optional keypoints (unused by dummy tracker).

        Returns:
            List of (track_id, bbox_xyxy, score).
        """
        outputs: List[TrackOutput] = []
        for det in detections:
            track_id = self._next_id
            self._next_id += 1
            bbox = det.bbox if hasattr(det, "bbox") else det[0]
            score = float(det.score) if hasattr(det, "score") else float(det[1])
            outputs.append((track_id, bbox, score))
        return outputs

    def reset(self) -> None:
        """Reset ID counter back to start_id."""
        self._next_id = self._start_id

