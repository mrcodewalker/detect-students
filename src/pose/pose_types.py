"""Data structures, COCO skeleton definitions, and matching utilities for human pose estimation."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np

from src.detection.detection import Detection


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

COCO_SKELETON_EDGES: List[Tuple[int, int]] = [
    (0, 1),  # nose -> left_eye
    (0, 2),  # nose -> right_eye
    (1, 3),  # left_eye -> left_ear
    (2, 4),  # right_eye -> right_ear
    (5, 6),  # left_shoulder -> right_shoulder
    (5, 7),  # left_shoulder -> left_elbow
    (7, 9),  # left_elbow -> left_wrist
    (6, 8),  # right_shoulder -> right_elbow
    (8, 10), # right_elbow -> right_wrist
    (5, 11), # left_shoulder -> left_hip
    (6, 12), # right_shoulder -> right_hip
    (11, 12),# left_hip -> right_hip
    (11, 13),# left_hip -> left_knee
    (13, 15),# left_knee -> left_ankle
    (12, 14),# right_hip -> right_knee
    (14, 16),# right_knee -> right_ankle
]


@dataclass
class PersonPose:
    """Represents estimated 2D pose keypoints for a single person.

    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        keypoints: Array of shape (17, 3) where each row is [x, y, confidence].
        score: Overall person detection/pose confidence score.
        person_idx: Optional sequence index.
    """

    bbox: Tuple[float, float, float, float]
    keypoints: np.ndarray
    score: float
    person_idx: int = 0

    def to_dict(self) -> dict:
        """Convert pose instance to a JSON-serializable dictionary."""
        return {
            "bbox": [float(v) for v in self.bbox],
            "keypoints": self.keypoints.tolist(),
            "score": float(self.score),
            "person_idx": int(self.person_idx),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersonPose":
        """Instantiate PersonPose from dictionary."""
        return cls(
            bbox=tuple(data["bbox"]),
            keypoints=np.array(data["keypoints"], dtype=np.float32),
            score=float(data["score"]),
            person_idx=int(data.get("person_idx", 0)),
        )


def box_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_keypoints_to_detections(
    detections: Sequence[Detection],
    poses: Sequence[PersonPose],
    iou_thresh: float = 0.3,
) -> List[Optional[np.ndarray]]:
    """Match pose keypoints to detector outputs by bounding box IoU.

    This ensures that each detection passed to a tracker receives its corresponding
    (17, 3) keypoint array.

    Args:
        detections: Sequence of Detection objects from person detector.
        poses: Sequence of PersonPose objects estimated for the frame.
        iou_thresh: Minimum IoU overlap to accept association.

    Returns:
        List of (17, 3) keypoint arrays (or None if unmatched), aligned 1:1 with detections.
    """
    if not detections:
        return []
    if not poses:
        return [None] * len(detections)

    matched_keypoints: List[Optional[np.ndarray]] = [None] * len(detections)
    used_poses = set()

    for d_idx, det in enumerate(detections):
        best_iou = 0.0
        best_pose_idx = -1
        for p_idx, pose in enumerate(poses):
            if p_idx in used_poses:
                continue
            iou = box_iou(det.bbox, pose.bbox)
            if iou > best_iou and iou >= iou_thresh:
                best_iou = iou
                best_pose_idx = p_idx

        if best_pose_idx >= 0:
            matched_keypoints[d_idx] = poses[best_pose_idx].keypoints
            used_poses.add(best_pose_idx)

    return matched_keypoints
