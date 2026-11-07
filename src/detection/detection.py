"""Data structures representing object detections."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    """Bounding box detection representation.

    Attributes:
        bbox: Bounding box in format (x1, y1, x2, y2) with pixel coordinates.
        score: Detection confidence score between 0.0 and 1.0.
        frame_index: Index of the frame where this detection occurred.
        class_id: Numerical class ID (defaults to 0 for 'person').
    """

    bbox: Tuple[float, float, float, float]
    score: float
    frame_index: int
    class_id: int = 0

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

