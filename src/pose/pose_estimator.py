"""Pretrained human pose estimation wrapper supporting both whole-frame and crop modes."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.pose.pose_types import PersonPose
from src.utils.config import load_config


class BasePoseEstimator(ABC):
    """Abstract base class for human pose estimators."""

    @abstractmethod
    def estimate(
        self,
        frame: np.ndarray,
        bboxes: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    ) -> List[PersonPose]:
        """Estimate 17 2D keypoints for persons in the frame.

        Args:
            frame: Full BGR video frame array (H, W, 3).
            bboxes: Optional list of person bounding boxes (x1, y1, x2, y2).
                    If None, runs whole-frame detection + pose estimation.
                    If provided, runs pose estimation on person crops.

        Returns:
            List of PersonPose instances.
        """
        pass


class YOLOPoseEstimator(BasePoseEstimator):
    """Ultralytics YOLO-Pose wrapper (YOLOv8-Pose / YOLO11-Pose).

    Reference: Ultralytics (AGPL-3.0 License).
    """

    def __init__(
        self,
        weights: str = "yolov8n-pose.pt",
        imgsz: int = 640,
        conf: float = 0.3,
        kpt_conf_thresh: float = 0.4,
        device: str = "cpu",
        half: bool = False,
        mode: str = "whole_frame",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Package 'ultralytics' is required for YOLOPoseEstimator.") from exc

        self.weights = str(weights)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.kpt_conf_thresh = float(kpt_conf_thresh)
        self.half = bool(half)
        self.mode = mode.lower()

        # Check and resolve device
        self.device = self._resolve_device(device)
        self.model = YOLO(self.weights)

    def _resolve_device(self, requested_device: str) -> str:
        try:
            import torch

            if ("cuda" in requested_device or requested_device.isdigit()) and torch.cuda.is_available():
                return requested_device
            return "cpu"
        except ImportError:
            return "cpu"

    @classmethod
    def from_config(cls, config: Union[str, Path, Dict[str, Any]]) -> "YOLOPoseEstimator":
        raw_cfg = load_config(config) if isinstance(config, (str, Path)) else config
        pose_cfg = raw_cfg.get("pose", raw_cfg)
        return cls(
            weights=pose_cfg.get("weights", "yolov8n-pose.pt"),
            imgsz=pose_cfg.get("imgsz", 640),
            conf=pose_cfg.get("conf", 0.3),
            kpt_conf_thresh=pose_cfg.get("kpt_conf_thresh", 0.4),
            device=pose_cfg.get("device", "cpu"),
            half=pose_cfg.get("half", False),
            mode=pose_cfg.get("mode", "whole_frame"),
        )

    def estimate(
        self,
        frame: np.ndarray,
        bboxes: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    ) -> List[PersonPose]:
        if bboxes is not None and len(bboxes) > 0 and self.mode == "crops":
            return self._estimate_crops(frame, bboxes)
        return self._estimate_whole_frame(frame)

    def _estimate_whole_frame(self, frame: np.ndarray) -> List[PersonPose]:
        predict_kwargs: Dict[str, Any] = {
            "source": frame,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "device": self.device,
            "classes": [0],
            "verbose": False,
        }
        if self.half and self.device not in ("cpu", "CPU"):
            predict_kwargs["half"] = True

        results = self.model.predict(**predict_kwargs)
        if not results:
            return []

        res = results[0]
        if res.boxes is None or res.keypoints is None or len(res.boxes) == 0:
            return []

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        kpts_data = res.keypoints.data.cpu().numpy()  # shape (N, 17, 3)

        poses: List[PersonPose] = []
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            score = float(scores[i])
            kpts = kpts_data[i]  # shape (17, 3)
            poses.append(
                PersonPose(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    keypoints=kpts,
                    score=score,
                    person_idx=i,
                )
            )
        return poses

    def _estimate_crops(
        self,
        frame: np.ndarray,
        bboxes: Sequence[Tuple[float, float, float, float]],
    ) -> List[PersonPose]:
        h_frame, w_frame = frame.shape[:2]
        poses: List[PersonPose] = []

        for p_idx, (x1, y1, x2, y2) in enumerate(bboxes):
            ix1 = max(0, min(w_frame - 1, int(x1)))
            iy1 = max(0, min(h_frame - 1, int(y1)))
            ix2 = max(0, min(w_frame, int(x2)))
            iy2 = max(0, min(h_frame, int(y2)))

            if (ix2 - ix1) < 8 or (iy2 - iy1) < 8:
                continue

            crop = frame[iy1:iy2, ix1:ix2]
            crop_poses = self._estimate_whole_frame(crop)
            if not crop_poses:
                continue

            # Pick highest score detection in crop and map back to frame coords
            best_pose = max(crop_poses, key=lambda p: p.score)
            mapped_kpts = best_pose.keypoints.copy()
            mapped_kpts[:, 0] += ix1
            mapped_kpts[:, 1] += iy1

            poses.append(
                PersonPose(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    keypoints=mapped_kpts,
                    score=best_pose.score,
                    person_idx=p_idx,
                )
            )
        return poses
