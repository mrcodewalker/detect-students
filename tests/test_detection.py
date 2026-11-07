"""Unit tests for Detection dataclass and YOLODetector."""

import sys
from pathlib import Path
import numpy as np
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.detection.detection import Detection
from src.detection.yolo_detector import YOLODetector


def test_detection_dataclass() -> None:
    """Verify Detection dataclass properties and fields."""
    det = Detection(bbox=(10.0, 20.0, 110.0, 220.0), score=0.88, frame_index=5, class_id=0)
    assert det.bbox == (10.0, 20.0, 110.0, 220.0)
    assert det.score == 0.88
    assert det.frame_index == 5
    assert det.class_id == 0
    assert det.x1 == 10.0
    assert det.y1 == 20.0
    assert det.x2 == 110.0
    assert det.y2 == 220.0
    assert det.width == 100.0
    assert det.height == 200.0


def test_yolo_detector_instantiation_from_config(tmp_path: Path) -> None:
    """Verify loading YOLODetector from a YAML configuration dictionary."""
    config = {
        "model": {
            "weights": "yolov8n.pt",
            "conf": 0.4,
            "iou": 0.5,
            "imgsz": 320,
            "device": "cpu",
            "half": False,
            "classes": [0],
        }
    }
    detector = YOLODetector.from_config(config)
    assert detector.conf == 0.4
    assert detector.iou == 0.5
    assert detector.imgsz == 320
    assert detector.device == "cpu"
    assert detector.classes == [0]


def test_yolo_detector_detect_dummy_frame() -> None:
    """Verify detector runs on a blank image and returns valid detection list."""
    detector = YOLODetector(weights="yolov8n.pt", conf=0.5, imgsz=320, device="cpu")
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame, frame_index=0)
    assert isinstance(detections, list)

