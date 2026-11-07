"""Student detection modules and bounding box estimators."""

from src.detection.detection import Detection
from src.detection.yolo_detector import YOLODetector

__all__ = [
    "Detection",
    "YOLODetector",
]
