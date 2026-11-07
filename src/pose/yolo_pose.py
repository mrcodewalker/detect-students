"""YOLOv8-Pose wrapper implementing 17 COCO keypoint estimation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.pose.base import PoseDetection, PoseEstimator
from src.pose.matching import match_pose_to_detections
from src.utils.config import load_config

logger = logging.getLogger(__name__)


class YOLOPoseEstimator(PoseEstimator):
    """Ultralytics YOLO-Pose wrapper for 2D human pose estimation.

    Outputs 17 COCO keypoints per person with confidence scores:
    (x, y, confidence) for each joint.
    """

    def __init__(
        self,
        weights: str = "yolov8n-pose.pt",
        conf: float = 0.35,
        iou: float = 0.45,
        kpt_conf: float = 0.30,
        imgsz: int = 640,
        device: str = "cpu",
        half: bool = False,
        mode: str = "whole_frame",
    ) -> None:
        """Initialize the YOLO-Pose model.

        Args:
            weights: Path to pose model weights (e.g. 'yolov8n-pose.pt').
            conf: Confidence threshold for person bounding box.
            iou: NMS IoU threshold.
            kpt_conf: Minimum confidence threshold for keypoint validation.
            imgsz: Input image resolution for inference.
            device: Target compute device ('cuda:0', 'cpu').
            half: Whether to use FP16 half-precision inference.
            mode: Inference mode: 'whole_frame' or 'crop'.
        """
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics and torch are required for YOLOPoseEstimator. "
                "Install them via: pip install ultralytics torch"
            ) from exc

        # Device fallback if CUDA requested but unavailable
        if "cuda" in device.lower() and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
            half = False

        self.weights = str(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.kpt_conf = float(kpt_conf)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self.half = bool(half) and ("cuda" in self.device)
        self.mode = str(mode).lower()

        logger.info(
            "Loading YOLO-Pose model from %s on device=%s, mode=%s",
            self.weights,
            self.device,
            self.mode,
        )
        self.model = YOLO(self.weights)

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path] = "configs/pose.yaml",
        **kwargs: Any,
    ) -> "YOLOPoseEstimator":
        """Instantiate YOLOPoseEstimator from a YAML configuration file."""
        cfg = load_config(config_path)
        model_cfg = cfg.get("model", {})
        merged = {**model_cfg, **kwargs}
        return cls(**merged)

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
            frame_index: Index of the current video frame.

        Returns:
            List of PoseDetection instances.
        """
        if self.mode == "crop" and bboxes is not None:
            return self._estimate_crops(frame, bboxes, frame_index)
        else:
            return self._estimate_whole_frame(frame, bboxes, frame_index)

    def _estimate_whole_frame(
        self,
        frame: np.ndarray,
        bboxes: Optional[Sequence[Tuple[float, float, float, float]]] = None,
        frame_index: int = -1,
    ) -> List[PoseDetection]:
        """Run pose estimation on the entire frame in a single forward pass."""
        predict_kwargs: Dict[str, Any] = {
            "source": frame,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self.half:
            predict_kwargs["half"] = True

        results = self.model.predict(**predict_kwargs)

        if not results:
            return []

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return []

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        kpts_data = res.keypoints.data.cpu().numpy()  # shape (N, 17, 3)

        poses: List[PoseDetection] = []
        for i in range(len(boxes_xyxy)):
            box = (
                float(boxes_xyxy[i][0]),
                float(boxes_xyxy[i][1]),
                float(boxes_xyxy[i][2]),
                float(boxes_xyxy[i][3]),
            )
            score = float(scores[i])
            kpts = kpts_data[i]  # (17, 3)

            poses.append(
                PoseDetection(
                    keypoints=kpts,
                    bbox=box,
                    score=score,
                    frame_index=frame_index,
                )
            )

        # If external bboxes were provided and we ran whole frame,
        # we can align them using Hungarian IoU matching if desired
        if bboxes is not None and len(bboxes) > 0:
            matched_poses = match_pose_to_detections(bboxes, poses, iou_threshold=0.2)
            # Filter out unmatched or replace with fallback
            aligned: List[PoseDetection] = []
            for b, p in zip(bboxes, matched_poses):
                if p is not None:
                    aligned.append(p)
                else:
                    # Fallback pose detection with zero-conf keypoints
                    empty_kpts = np.zeros((17, 3), dtype=np.float32)
                    aligned.append(
                        PoseDetection(
                            keypoints=empty_kpts,
                            bbox=b,
                            score=0.0,
                            frame_index=frame_index,
                        )
                    )
            return aligned

        return poses

    def _estimate_crops(
        self,
        frame: np.ndarray,
        bboxes: Sequence[Tuple[float, float, float, float]],
        frame_index: int = -1,
    ) -> List[PoseDetection]:
        """Crop each person bbox from frame and estimate pose within the crop."""
        h, w = frame.shape[:2]
        poses: List[PoseDetection] = []

        for box in bboxes:
            x1 = max(0, min(int(round(box[0])), w - 1))
            y1 = max(0, min(int(round(box[1])), h - 1))
            x2 = max(0, min(int(round(box[2])), w))
            y2 = max(0, min(int(round(box[3])), h))

            crop_w = x2 - x1
            crop_h = y2 - y1

            if crop_w < 8 or crop_h < 8:
                # Box too small for reliable pose estimation
                empty_kpts = np.zeros((17, 3), dtype=np.float32)
                poses.append(
                    PoseDetection(
                        keypoints=empty_kpts,
                        bbox=box,
                        score=0.0,
                        frame_index=frame_index,
                    )
                )
                continue

            crop = frame[y1:y2, x1:x2]

            crop_kwargs: Dict[str, Any] = {
                "source": crop,
                "conf": self.conf * 0.8,
                "iou": self.iou,
                "imgsz": max(128, min(self.imgsz, max(crop_w, crop_h))),
                "device": self.device,
                "verbose": False,
            }
            if self.half:
                crop_kwargs["half"] = True

            results = self.model.predict(**crop_kwargs)

            found_pose = False
            if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
                res = results[0]
                kpts_data = res.keypoints.data.cpu().numpy()  # (M, 17, 3)
                if len(kpts_data) > 0:
                    best_idx = 0
                    if res.boxes is not None and len(res.boxes.conf) > 0:
                        best_idx = int(np.argmax(res.boxes.conf.cpu().numpy()))

                    best_kpts = kpts_data[best_idx].copy()
                    # Offset crop coordinates back to frame coordinates
                    best_kpts[:, 0] += x1
                    best_kpts[:, 1] += y1

                    score = float(res.boxes.conf[best_idx]) if res.boxes is not None else 1.0

                    poses.append(
                        PoseDetection(
                            keypoints=best_kpts,
                            bbox=box,
                            score=score,
                            frame_index=frame_index,
                        )
                    )
                    found_pose = True

            if not found_pose:
                empty_kpts = np.zeros((17, 3), dtype=np.float32)
                poses.append(
                    PoseDetection(
                        keypoints=empty_kpts,
                        bbox=box,
                        score=0.0,
                        frame_index=frame_index,
                    )
                )

        return poses
