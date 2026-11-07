"""Pose estimation module for detecting 17 2D human keypoints."""

from src.pose.base import (
    COCO_KEYPOINTS,
    COCO_SKELETON_EDGES,
    PoseDetection,
    PoseEstimator,
)
from src.pose.cache import (
    get_default_cache_path,
    has_pose_cache,
    load_pose_cache,
    save_pose_cache,
)
from src.pose.matching import (
    compute_iou,
    compute_iou_matrix,
    match_pose_to_detections,
)
from src.pose.visualizer import draw_pose_on_frame
from src.pose.yolo_pose import YOLOPoseEstimator

# Aliases for backward compatibility
from src.pose.pose_types import PersonPose, box_iou, match_keypoints_to_detections
from src.pose.pose_cache import PoseCacheManager

__all__ = [
    "COCO_KEYPOINTS",
    "COCO_SKELETON_EDGES",
    "PoseDetection",
    "PoseEstimator",
    "YOLOPoseEstimator",
    "compute_iou",
    "compute_iou_matrix",
    "match_pose_to_detections",
    "get_default_cache_path",
    "has_pose_cache",
    "load_pose_cache",
    "save_pose_cache",
    "draw_pose_on_frame",
    "PersonPose",
    "box_iou",
    "match_keypoints_to_detections",
    "PoseCacheManager",
]

