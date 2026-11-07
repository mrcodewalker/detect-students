"""Ultralytics YOLO wrapper for student detection in classroom video."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.detection.detection import Detection
from src.utils.config import load_config


class YOLODetector:
    """YOLO-based person detector configured for classroom environments."""

    PERSON_CLASS_ID: int = 0

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cuda:0",
        half: bool = False,
        classes: Optional[List[int]] = None,
    ) -> None:
        """Initialize the YOLO detector.

        Args:
            weights: Path to model weights or model name.
            conf: Confidence threshold for filtering bounding boxes.
            iou: Intersection-Over-Union threshold for NMS.
            imgsz: Inference input image size.
            device: Compute device ('cuda:0', 'cpu').
            half: Whether to use FP16 half-precision inference.
            classes: List of class IDs to filter. Defaults to [0] (person).
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "The 'ultralytics' package is required. "
                "Ensure it is installed with: pip install ultralytics"
            ) from exc

        self.weights = str(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.half = bool(half)
        self.classes = classes if classes is not None else [self.PERSON_CLASS_ID]

        # Resolve device availability
        self.device = self._resolve_device(device)

        # Load Ultralytics model
        self.model = YOLO(self.weights)

    def _resolve_device(self, requested_device: str) -> str:
        """Verify device availability and fallback gracefully if needed."""
        try:
            import torch

            if "cuda" in requested_device or requested_device.isdigit():
                if not torch.cuda.is_available():
                    return "cpu"
                return requested_device
            return "cpu"
        except ImportError:
            return "cpu"

    @classmethod
    def from_config(cls, config: Union[str, Path, Dict[str, Any]]) -> "YOLODetector":
        """Factory method to instantiate detector from a YAML configuration.

        Args:
            config: Path to YAML file or pre-loaded dictionary.

        Returns:
            Configured YOLODetector instance.
        """
        if isinstance(config, (str, Path)):
            raw_cfg = load_config(config)
        else:
            raw_cfg = config

        model_cfg = raw_cfg.get("model", raw_cfg)

        return cls(
            weights=model_cfg.get("weights", "yolov8n.pt"),
            conf=model_cfg.get("conf", 0.35),
            iou=model_cfg.get("iou", 0.45),
            imgsz=model_cfg.get("imgsz", 640),
            device=model_cfg.get("device", "cuda:0"),
            half=model_cfg.get("half", False),
            classes=model_cfg.get("classes", [cls.PERSON_CLASS_ID]),
        )

    def detect(self, frame: np.ndarray, frame_index: int) -> List[Detection]:
        """Run object detection on a single video frame.

        Args:
            frame: Input image array (BGR format from OpenCV).
            frame_index: Integer index of the processed frame.

        Returns:
            List of Detection instances for detected persons.
        """
        predict_kwargs: Dict[str, Any] = {
            "source": frame,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "classes": self.classes,
            "verbose": False,
        }
        if self.half and self.device not in ("cpu", "CPU"):
            predict_kwargs["half"] = True

        results = self.model.predict(**predict_kwargs)

        detections: List[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        if first_res.boxes is None or len(first_res.boxes) == 0:
            return detections

        boxes_xyxy = first_res.boxes.xyxy.cpu().numpy()
        scores = first_res.boxes.conf.cpu().numpy()
        classes = first_res.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes_xyxy)):
            cls_id = int(classes[i])
            # Extra safeguard: ensure only person class is returned
            if cls_id not in self.classes:
                continue

            x1, y1, x2, y2 = boxes_xyxy[i]
            score = float(scores[i])
            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    score=score,
                    frame_index=frame_index,
                    class_id=cls_id,
                )
            )

        return detections
