"""Abstract base interfaces and data structures for 2D pose estimation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


# Standard COCO 17 keypoint names and pairs for skeleton visualization
COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Skeleton limb connections as index pairs (p1, p2)
COCO_SKELETON_EDGES: List[Tuple[int, int]] = [
    # Head
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    # Upper body / Arms
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    # Torso
    (5, 11),
    (6, 12),
    (11, 12),
    # Lower body / Legs
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass
class PoseDetection:
    """Represents estimated 2D pose keypoints and bounding box for a single person.

    Attributes:
        keypoints: Array of shape (17, 3) where each row is (x, y, confidence).
        bbox: Bounding box tuple (x1, y1, x2, y2) in pixel coordinates.
        score: Overall person detection / pose confidence score.
        frame_index: Frame index in the video sequence (optional, defaults to -1).
    """

    keypoints: np.ndarray  # shape: (17, 3)
    bbox: Tuple[float, float, float, float]
    score: float
    frame_index: int = -1

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32)
        if self.keypoints.shape != (17, 3):
            raise ValueError(
                f"Expected keypoints array of shape (17, 3), got {self.keypoints.shape}"
            )

    @property
    def x1(self) -> float:
        return float(self.bbox[0])

    @property
    def y1(self) -> float:
        return float(self.bbox[1])

    @property
    def x2(self) -> float:
        return float(self.bbox[2])

    @property
    def y2(self) -> float:
        return float(self.bbox[3])

    @property
    def xy(self) -> np.ndarray:
        """Returns 2D keypoint coordinates of shape (17, 2)."""
        return self.keypoints[:, :2]

    @property
    def confidences(self) -> np.ndarray:
        """Returns keypoint confidence scores of shape (17,)."""
        return self.keypoints[:, 2]

    def to_dict(self) -> dict:
        """Convert pose detection to JSON-serializable dictionary."""
        return {
            "bbox": [float(c) for c in self.bbox],
            "score": float(self.score),
            "frame_index": int(self.frame_index),
            "keypoints": self.keypoints.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PoseDetection":
        """Construct PoseDetection from a dictionary representation."""
        bbox = tuple(data["bbox"])
        return cls(
            keypoints=np.array(data["keypoints"], dtype=np.float32),
            bbox=bbox,  # type: ignore[arg-type]
            score=float(data.get("score", 1.0)),
            frame_index=int(data.get("frame_index", -1)),
        )


class PoseEstimator(ABC):
    """Abstract interface for 2D human pose estimation models."""

    @abstractmethod
    def estimate(
        self,
        frame: np.ndarray,
        bboxes: Optional[Sequence[Tuple[float, float, float, float]]] = None,
        frame_index: int = -1,
    ) -> List[PoseDetection]:
        """Estimate 17 2D keypoints on whole frame or provided person crops.

        Args:
            frame: Input BGR image array of shape (H, W, 3).
            bboxes: Optional sequence of bounding boxes (x1, y1, x2, y2).
                If provided and mode is 'crop', inference is executed on each crop.
                If None or mode is 'whole_frame', inference is executed on full frame.
            frame_index: Optional frame index for metadata tracking.

        Returns:
            List of PoseDetection instances, each containing (17, 3) keypoints.
        """
        pass

