"""Unit tests for 2D pose estimation, matching, caching, and visualization."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.detection.detection import Detection
from src.pose.base import COCO_KEYPOINTS, COCO_SKELETON_EDGES, PoseDetection
from src.pose.cache import get_default_cache_path, has_pose_cache, load_pose_cache, save_pose_cache
from src.pose.matching import compute_iou, compute_iou_matrix, match_pose_to_detections
from src.pose.visualizer import draw_pose_on_frame
from src.pose.yolo_pose import YOLOPoseEstimator


def test_pose_detection_init_and_properties() -> None:
    """Test PoseDetection initialization, shape validation, and properties."""
    kpts = np.random.rand(17, 3).astype(np.float32)
    bbox = (10.0, 20.0, 110.0, 220.0)
    pose = PoseDetection(keypoints=kpts, bbox=bbox, score=0.92, frame_index=5)

    assert pose.x1 == 10.0
    assert pose.y1 == 20.0
    assert pose.x2 == 110.0
    assert pose.y2 == 220.0
    assert pose.xy.shape == (17, 2)
    assert pose.confidences.shape == (17,)
    assert pose.frame_index == 5
    assert len(COCO_KEYPOINTS) == 17
    assert len(COCO_SKELETON_EDGES) > 0


def test_pose_detection_invalid_shape() -> None:
    """Ensure invalid keypoint shapes raise ValueError."""
    bad_kpts = np.zeros((15, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        PoseDetection(keypoints=bad_kpts, bbox=(0, 0, 10, 10), score=0.5)


def test_pose_detection_serialization_roundtrip() -> None:
    """Test to_dict and from_dict serialization."""
    kpts = np.ones((17, 3), dtype=np.float32) * 42.0
    bbox = (15.0, 25.0, 65.0, 95.0)
    original = PoseDetection(keypoints=kpts, bbox=bbox, score=0.88, frame_index=12)

    d = original.to_dict()
    restored = PoseDetection.from_dict(d)

    assert restored.bbox == original.bbox
    assert restored.score == pytest.approx(original.score)
    assert restored.frame_index == original.frame_index
    assert np.allclose(restored.keypoints, original.keypoints)


def test_iou_and_matching() -> None:
    """Test box IoU computation and Hungarian association."""
    b1 = (0.0, 0.0, 100.0, 100.0)
    b2 = (50.0, 0.0, 150.0, 100.0)
    # intersection: 50 * 100 = 5000, union: 10000 + 10000 - 5000 = 15000 -> 1/3
    iou = compute_iou(b1, b2)
    assert pytest.approx(iou, rel=1e-3) == 1.0 / 3.0

    # Test match_pose_to_detections
    det1 = Detection(bbox=(10.0, 10.0, 100.0, 100.0), score=0.9, frame_index=0)
    det2 = Detection(bbox=(200.0, 200.0, 300.0, 300.0), score=0.85, frame_index=0)

    pose1 = PoseDetection(
        keypoints=np.zeros((17, 3), dtype=np.float32),
        bbox=(12.0, 11.0, 102.0, 101.0),
        score=0.9,
    )
    pose2 = PoseDetection(
        keypoints=np.ones((17, 3), dtype=np.float32),
        bbox=(205.0, 198.0, 302.0, 305.0),
        score=0.8,
    )

    matches = match_pose_to_detections([det1, det2], [pose1, pose2], iou_threshold=0.5)
    assert len(matches) == 2
    assert matches[0] is pose1
    assert matches[1] is pose2

    # Test non-overlapping detection falls back to None
    det_far = Detection(bbox=(800.0, 800.0, 900.0, 900.0), score=0.7, frame_index=0)
    matches_far = match_pose_to_detections([det_far], [pose1, pose2], iou_threshold=0.3)
    assert len(matches_far) == 1
    assert matches_far[0] is None


def test_pose_cache_save_and_load(tmp_path: Path) -> None:
    """Test disk caching of frame pose detections."""
    cache_file = tmp_path / "test_video_pose.json"
    assert not has_pose_cache(cache_file)

    p1 = PoseDetection(
        keypoints=np.full((17, 3), 10.0, dtype=np.float32),
        bbox=(10.0, 10.0, 50.0, 50.0),
        score=0.9,
        frame_index=0,
    )
    p2 = PoseDetection(
        keypoints=np.full((17, 3), 20.0, dtype=np.float32),
        bbox=(60.0, 60.0, 100.0, 100.0),
        score=0.85,
        frame_index=1,
    )

    frames_data = {0: [p1], 1: [p2]}
    save_pose_cache(cache_file, frames_data, video_path="mock_video.mp4")

    assert has_pose_cache(cache_file)
    loaded = load_pose_cache(cache_file)
    assert loaded is not None
    assert len(loaded) == 2
    assert len(loaded[0]) == 1
    assert loaded[0][0].score == pytest.approx(0.9)
    assert np.allclose(loaded[1][0].keypoints, p2.keypoints)


def test_draw_pose_visualizer() -> None:
    """Test visualization canvas rendering with skeletons and keypoints."""
    canvas = np.zeros((240, 320, 3), dtype=np.uint8)
    kpts = np.full((17, 3), 50.0, dtype=np.float32)
    kpts[:, 2] = 0.9  # high confidence
    pose = PoseDetection(keypoints=kpts, bbox=(20.0, 20.0, 150.0, 200.0), score=0.95)

    annotated = draw_pose_on_frame(canvas, [pose], kpt_conf_thresh=0.3)
    assert annotated.shape == canvas.shape
    assert annotated.dtype == np.uint8
    # Canvas should now have non-zero pixels from skeleton and keypoints
    assert np.any(annotated > 0)


def test_yolo_pose_estimator_execution() -> None:
    """Test YOLOPoseEstimator inference on synthetic image in both modes."""
    estimator = YOLOPoseEstimator(
        weights="yolov8n-pose.pt",
        conf=0.25,
        device="cpu",
        mode="whole_frame",
    )

    # Synthetic image
    img = np.zeros((320, 320, 3), dtype=np.uint8)

    # Whole frame test
    poses = estimator.estimate(img, frame_index=0)
    assert isinstance(poses, list)

    # Crop mode test
    estimator.mode = "crop"
    bboxes = [(20.0, 20.0, 100.0, 150.0)]
    crop_poses = estimator.estimate(img, bboxes=bboxes, frame_index=1)
    assert len(crop_poses) == 1
    assert crop_poses[0].keypoints.shape == (17, 3)
