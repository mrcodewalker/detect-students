"""Action classification interfaces and model backends (Keypoint & Crop)."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.action.track_buffer import ActionWindow
from src.utils.config import load_config

logger = logging.getLogger(__name__)

DEFAULT_ACTION_CLASSES = ["sitting", "standing", "raising_hand", "using_phone"]


class ActionClassifier(ABC):
    """Abstract base interface for temporal action classifiers."""

    def __init__(self, classes: Optional[List[str]] = None, device: str = "cpu") -> None:
        self.classes = classes or list(DEFAULT_ACTION_CLASSES)
        self.num_classes = len(self.classes)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.is_trained = False

    @abstractmethod
    def predict(self, window: ActionWindow) -> Tuple[str, float, Dict[str, float]]:
        """Predict action for a single temporal ActionWindow.

        Args:
            window: Temporal window containing observations.

        Returns:
            Tuple of (predicted_class_name, confidence_score, probabilities_dict).
        """
        pass

    @abstractmethod
    def predict_batch(
        self,
        windows: Sequence[ActionWindow],
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Predict action for a batch of ActionWindows.

        Args:
            windows: Sequence of temporal ActionWindows.

        Returns:
            List of (predicted_class_name, confidence_score, probabilities_dict).
        """
        pass


# =========================================================================
# Backend A: Keypoint-Sequence Temporal Model & Classifier
# =========================================================================

class KeypointTemporalModel(nn.Module):
    """Small temporal neural network operating on 17-keypoint sequences."""

    def __init__(
        self,
        num_classes: int = 4,
        input_features: int = 51,  # 17 keypoints * 3 (x, y, conf)
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.input_proj = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.temporal_attention = nn.Sequential(
            nn.Linear(gru_out_dim, 1),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, T, 51) or (B, T, 17, 3).

        Returns:
            Logits of shape (B, num_classes).
        """
        if x.dim() == 4:
            b, t, k, c = x.shape
            x = x.view(b, t, k * c)

        # 1. Project input features
        h = self.input_proj(x)  # (B, T, hidden_dim)

        # 2. Temporal modeling
        gru_out, _ = self.gru(h)  # (B, T, gru_out_dim)

        # 3. Temporal attention pooling
        weights = self.temporal_attention(gru_out)  # (B, T, 1)
        pooled = torch.sum(gru_out * weights, dim=1)  # (B, gru_out_dim)

        # 4. Classification logits
        logits = self.classifier(pooled)
        return logits


def compute_pose_geometric_probs(win: ActionWindow, classes: Sequence[str]) -> np.ndarray:
    """Anatomically accurate classroom action estimation from keypoints and bounding box trajectories."""
    num_classes = len(classes)
    t_steps = len(win.frame_indices)
    if win.keypoints is None or t_steps == 0:
        p = np.zeros(num_classes, dtype=np.float32)
        if "sitting" in classes:
            p[classes.index("sitting")] = 0.90
            rem = 0.10 / max(1, num_classes - 1)
            for i in range(num_classes):
                if classes[i] != "sitting":
                    p[i] = rem
        else:
            p.fill(1.0 / num_classes)
        return p

    kpts = win.keypoints  # (T, 17, 3)
    bboxes = win.bboxes   # (T, 4)

    hand_raised_scores = []
    standing_scores = []
    phone_scores = []

    for t in range(t_steps):
        box = bboxes[t]
        bw = max(1.0, box[2] - box[0])
        bh = max(1.0, box[3] - box[1])

        nose = kpts[t, 0]
        l_shldr, r_shldr = kpts[t, 5], kpts[t, 6]
        l_elbow, r_elbow = kpts[t, 7], kpts[t, 8]
        l_wrist, r_wrist = kpts[t, 9], kpts[t, 10]
        l_hip, r_hip = kpts[t, 11], kpts[t, 12]
        l_knee, r_knee = kpts[t, 13], kpts[t, 14]
        l_ankle, r_ankle = kpts[t, 15], kpts[t, 16]

        shldr_conf = max(l_shldr[2], r_shldr[2])
        shldr_y = min(l_shldr[1], r_shldr[1]) if (l_shldr[2] > 0.15 and r_shldr[2] > 0.15) else (
            l_shldr[1] if l_shldr[2] > 0.15 else r_shldr[1]
        )

        # 1. RAISING HAND
        l_up = (l_wrist[2] > 0.25) and (l_wrist[1] < shldr_y - 0.05 * bh)
        r_up = (r_wrist[2] > 0.25) and (r_wrist[1] < shldr_y - 0.05 * bh)
        if nose[2] > 0.25:
            l_up = l_up or ((l_wrist[2] > 0.25) and (l_wrist[1] < nose[1]))
            r_up = r_up or ((r_wrist[2] > 0.25) and (r_wrist[1] < nose[1]))

        # High elbow support when wrist is partially occluded
        if (l_elbow[2] > 0.35 and l_elbow[1] < shldr_y) or (r_elbow[2] > 0.35 and r_elbow[1] < shldr_y):
            if l_wrist[2] > 0.15 or r_wrist[2] > 0.15:
                l_up = l_up or (l_elbow[1] < shldr_y)
                r_up = r_up or (r_elbow[1] < shldr_y)

        hand_raised_scores.append(1.0 if (l_up or r_up) else 0.0)

        # 2. STANDING (Requires verified lower-body extension, not just narrow desk crop)
        hip_conf = max(l_hip[2], r_hip[2])
        knee_conf = max(l_knee[2], r_knee[2])
        ankle_conf = max(l_ankle[2], r_ankle[2])

        is_standing = False
        if hip_conf > 0.3 and knee_conf > 0.3:
            hip_y = (l_hip[1] + r_hip[1]) / 2
            knee_y = (l_knee[1] + r_knee[1]) / 2
            if knee_y - hip_y > 0.25 * bh:
                if ankle_conf > 0.25:
                    is_standing = True
                elif bh / bw >= 2.0:
                    is_standing = True
        standing_scores.append(1.0 if is_standing else 0.0)

        # 3. USING PHONE (Looking down at hands held close together near mid-torso/lap)
        is_phone = False
        if l_wrist[2] > 0.25 and r_wrist[2] > 0.25:
            wrist_dx = abs(l_wrist[0] - r_wrist[0])
            avg_wrist_y = (l_wrist[1] + r_wrist[1]) / 2
            if shldr_conf > 0.2 and (shldr_y < avg_wrist_y < shldr_y + 0.45 * bh):
                if wrist_dx < 0.18 * bw:
                    if nose[2] > 0.2 and (nose[1] >= shldr_y - 0.05 * bh):
                        is_phone = True
        phone_scores.append(1.0 if is_phone else 0.0)

    f_raise = float(np.mean(hand_raised_scores))
    f_stand = float(np.mean(standing_scores))
    f_phone = float(np.mean(phone_scores))

    raw_scores: Dict[str, float] = {
        "sitting": 3.0,
        "standing": 0.0,
        "raising_hand": 0.0,
        "using_phone": 0.0,
    }

    if f_raise >= 0.25:
        raw_scores["raising_hand"] = 4.5 + 5.0 * f_raise
        raw_scores["sitting"] = 0.2
    elif f_stand >= 0.4:
        raw_scores["standing"] = 3.5 + 4.0 * f_stand
        raw_scores["sitting"] = 0.3
    elif f_phone >= 0.4:
        raw_scores["using_phone"] = 3.0 + 3.0 * f_phone
        raw_scores["sitting"] = 0.5
    else:
        raw_scores["sitting"] = 4.0

    score_vec = np.array([raw_scores.get(c, 0.0) for c in classes], dtype=np.float32)
    exp_s = np.exp(score_vec - np.max(score_vec))
    probs = exp_s / np.sum(exp_s)
    return probs


class KeypointActionClassifier(ActionClassifier):
    """Classifier backend operating on 17-keypoint sequences."""

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        weights: Optional[Union[str, Path]] = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = "cpu",
    ) -> None:
        super().__init__(classes=classes, device=device)
        self.model = KeypointTemporalModel(
            num_classes=self.num_classes,
            input_features=51,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        ).to(self.device)

        if weights is not None and Path(weights).is_file():
            self.load_checkpoint(weights)
        else:
            self.is_trained = False
            self.model.eval()

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load trained model weights from checkpoint file."""
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.is_trained = True
        self.model.eval()
        logger.info("Loaded keypoint classifier weights from: %s", path)

    def _normalize_keypoints(self, keypoints: Optional[np.ndarray], bboxes: np.ndarray) -> np.ndarray:
        """Normalize keypoints relative to bounding box center and scale."""
        if keypoints is None:
            return np.zeros((len(bboxes), 17, 3), dtype=np.float32)

        t_steps = len(keypoints)
        norm_kpts = keypoints.copy()

        for t in range(t_steps):
            bx1, by1, bx2, by2 = bboxes[t]
            bw = max(1.0, bx2 - bx1)
            bh = max(1.0, by2 - by1)
            # Center coordinates
            norm_kpts[t, :, 0] = (norm_kpts[t, :, 0] - bx1) / bw
            norm_kpts[t, :, 1] = (norm_kpts[t, :, 1] - by1) / bh

        return norm_kpts

    def _compute_geometric_probs(self, win: ActionWindow) -> np.ndarray:
        """Compute rule-based anatomical posture probabilities from keypoint trajectories."""
        return compute_pose_geometric_probs(win, self.classes)

    def predict(self, window: ActionWindow) -> Tuple[str, float, Dict[str, float]]:
        """Predict action for a single ActionWindow."""
        results = self.predict_batch([window])
        return results[0]

    def predict_batch(
        self,
        windows: Sequence[ActionWindow],
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Predict action for a batch of ActionWindows."""
        if len(windows) == 0:
            return []

        # If trained model weights are available, run NN inference
        nn_probs = None
        if self.is_trained:
            batch_arrays = []
            for win in windows:
                norm_kpts = self._normalize_keypoints(win.keypoints, win.bboxes)
                batch_arrays.append(norm_kpts)

            x_tensor = torch.tensor(np.stack(batch_arrays, axis=0), dtype=torch.float32, device=self.device)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x_tensor)
                nn_probs = F.softmax(logits, dim=-1).cpu().numpy()

        results: List[Tuple[str, float, Dict[str, float]]] = []
        for i in range(len(windows)):
            geo_p = self._compute_geometric_probs(windows[i])

            if self.is_trained and nn_probs is not None:
                p = 0.7 * nn_probs[i] + 0.3 * geo_p
            else:
                p = geo_p

            best_idx = int(np.argmax(p))
            best_label = self.classes[best_idx]
            best_conf = float(p[best_idx])
            prob_dict = {self.classes[c]: float(p[c]) for c in range(self.num_classes)}
            results.append((best_label, best_conf, prob_dict))

        return results


# =========================================================================
# Backend B: Crop-based Image/Video Backbone Classifier
# =========================================================================

class CropTemporalModel(nn.Module):
    """Pretrained lightweight image backbone (MobileNetV3-Small) + Temporal Head."""

    def __init__(
        self,
        num_classes: int = 4,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        import torchvision.models as models

        weights = "DEFAULT" if pretrained else None
        if backbone_name == "mobilenet_v3_small":
            mobilenet = models.mobilenet_v3_small(weights=weights)
            # Use features + avgpool
            self.backbone_features = mobilenet.features
            self.backbone_pool = mobilenet.avgpool
            feat_dim = 576
        else:
            # Fallback to mobilenet_v3_small
            mobilenet = models.mobilenet_v3_small(weights=weights)
            self.backbone_features = mobilenet.features
            self.backbone_pool = mobilenet.avgpool
            feat_dim = 576

        self.temporal_gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        gru_out_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, T, C, H, W) normalized to [0, 1].

        Returns:
            Logits of shape (B, num_classes).
        """
        b, t, c, h, w = x.shape
        # Fold batch and temporal dimensions for CNN forward pass
        x_flat = x.view(b * t, c, h, w)
        feats = self.backbone_features(x_flat)
        pooled = self.backbone_pool(feats).flatten(1)  # (B*T, feat_dim)

        # Unfold back to (B, T, feat_dim)
        feat_dim = pooled.shape[-1]
        t_seq = pooled.view(b, t, feat_dim)

        gru_out, _ = self.temporal_gru(t_seq)  # (B, T, gru_out_dim)
        # Average pooling over time
        temporal_feat = torch.mean(gru_out, dim=1)  # (B, gru_out_dim)

        logits = self.classifier(temporal_feat)
        return logits


class CropActionClassifier(ActionClassifier):
    """Classifier backend operating on sequence of cropped person images."""

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        weights: Optional[Union[str, Path]] = None,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        device: str = "cpu",
    ) -> None:
        super().__init__(classes=classes, device=device)
        self.model = CropTemporalModel(
            num_classes=self.num_classes,
            backbone_name=backbone,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(self.device)

        if weights is not None and Path(weights).is_file():
            self.load_checkpoint(weights)
        else:
            self.is_trained = False
            self.model.eval()

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load trained model weights from checkpoint file."""
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.is_trained = True
        self.model.eval()
        logger.info("Loaded crop classifier weights from: %s", path)

    def _prepare_crops(self, crops: Optional[np.ndarray], window_size: int = 16) -> np.ndarray:
        """Prepare crops array of shape (T, 3, 112, 112) normalized to [0, 1]."""
        if crops is None or len(crops) == 0:
            return np.zeros((window_size, 3, 112, 112), dtype=np.float32)

        # crops: (T, H, W, 3) -> (T, 3, H, W)
        t_crops = np.transpose(crops, (0, 3, 1, 2)).astype(np.float32) / 255.0
        return t_crops

    def predict(self, window: ActionWindow) -> Tuple[str, float, Dict[str, float]]:
        """Predict action for a single ActionWindow."""
        results = self.predict_batch([window])
        return results[0]

    def predict_batch(
        self,
        windows: Sequence[ActionWindow],
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Predict action for a batch of ActionWindows."""
        if len(windows) == 0:
            return []

        batch_arrays = []
        for win in windows:
            c = self._prepare_crops(win.crops, window_size=len(win.frame_indices))
            batch_arrays.append(c)

        x_tensor = torch.tensor(np.stack(batch_arrays, axis=0), dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        results: List[Tuple[str, float, Dict[str, float]]] = []
        for i in range(len(windows)):
            p = probs[i]
            best_idx = int(np.argmax(p))
            best_label = self.classes[best_idx]
            best_conf = float(p[best_idx])
            prob_dict = {self.classes[c]: float(p[c]) for c in range(self.num_classes)}
            results.append((best_label, best_conf, prob_dict))

        return results


# =========================================================================
# Backend C: Pretrained Video Backbone (Google S3D on Kinetics-400)
# =========================================================================

class PretrainedS3DActionClassifier(ActionClassifier):
    """Deep Learning video action classifier using Google's pretrained S3D on Kinetics-400."""

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        weights: str = "DEFAULT",
        device: str = "cpu",
    ) -> None:
        super().__init__(classes=classes, device=device)
        import torchvision.models.video as vmodels

        logger.info("Initializing Google S3D video action backbone (Kinetics-400 pretrained)...")
        w = getattr(vmodels.S3D_Weights, weights) if isinstance(weights, str) and hasattr(vmodels.S3D_Weights, weights) else vmodels.S3D_Weights.DEFAULT
        self.model = vmodels.s3d(weights=w).to(self.device)
        self.model.eval()
        self.is_trained = True

        # Mapping from Kinetics-400 categories to classroom target actions
        self.kinetics_mapping = {
            "raising_hand": [2, 256, 303, 333],  # answering questions, pumping fist, sign language, stretching arm
            "using_phone": [355, 373, 226],      # texting, using computer, playing controller
            "sitting": [264, 265, 396, 98, 91, 351],  # reading, writing, drawing, dining, tapping pen
            "standing": [254, 354, 377, 192, 75],     # presenting, testifying, waiting in line, marching
        }

    def _prepare_crops_s3d(self, crops: Optional[np.ndarray], target_t: int = 16) -> torch.Tensor:
        """Prepare crops tensor (1, 3, T, 224, 224) normalized for S3D."""
        if crops is None or len(crops) == 0:
            return torch.zeros((1, 3, target_t, 224, 224), dtype=torch.float32, device=self.device)

        # crops: (T, H, W, 3) -> (1, 3, T, H, W)
        t_crops = torch.tensor(crops, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0) / 255.0
        # Interpolate spatial resolution to 224x224
        t_resized = F.interpolate(t_crops, size=(target_t, 224, 224), mode="trilinear", align_corners=False)

        mean = torch.tensor([0.43216, 0.394666, 0.37645], device=t_crops.device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], device=t_crops.device).view(1, 3, 1, 1, 1)
        normalized = (t_resized - mean) / std
        return normalized.to(self.device)

    def predict(self, window: ActionWindow) -> Tuple[str, float, Dict[str, float]]:
        """Predict action for a single ActionWindow."""
        results = self.predict_batch([window])
        return results[0]

    def predict_batch(
        self,
        windows: Sequence[ActionWindow],
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Predict action for a batch of ActionWindows."""
        if len(windows) == 0:
            return []

        results: List[Tuple[str, float, Dict[str, float]]] = []
        for win in windows:
            geo_p = compute_pose_geometric_probs(win, self.classes)

            # Check if valid crops are present
            has_valid_crops = (
                win.crops is not None
                and len(win.crops) > 0
                and win.crops.shape[-1] == 3
                and win.crops.shape[1] > 1
            )

            if has_valid_crops:
                x_tensor = self._prepare_crops_s3d(win.crops, target_t=16)
                with torch.no_grad():
                    logits = self.model(x_tensor)
                    k_probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                s3d_scores = np.zeros(self.num_classes, dtype=np.float32)
                for c_idx, c_name in enumerate(self.classes):
                    k_indices = self.kinetics_mapping.get(c_name, [])
                    if k_indices:
                        s3d_scores[c_idx] = float(np.sum(k_probs[k_indices]))
                    else:
                        s3d_scores[c_idx] = 0.01

                sum_s = np.sum(s3d_scores)
                if sum_s > 0:
                    s3d_p = s3d_scores / sum_s
                else:
                    s3d_p = np.full(self.num_classes, 1.0 / self.num_classes, dtype=np.float32)

                # Blend 40% S3D Deep 3D Video features + 60% Pose Geometric Prior
                p = 0.4 * s3d_p + 0.6 * geo_p
            else:
                p = geo_p

            best_idx = int(np.argmax(p))
            best_label = self.classes[best_idx]
            best_conf = float(p[best_idx])
            prob_dict = {self.classes[c]: float(p[c]) for c in range(self.num_classes)}
            results.append((best_label, best_conf, prob_dict))

        return results


# =========================================================================
# Factory function
# =========================================================================

def create_action_classifier(
    config: Union[str, Path, Dict[str, Any]] = "configs/action.yaml",
    backend: Optional[str] = None,
    **kwargs: Any,
) -> ActionClassifier:
    """Create an ActionClassifier instance according to configuration.

    Args:
        config: Path to YAML config file or dictionary.
        backend: Optional override: 'keypoint', 'crop', or 'pretrained_s3d'.

    Returns:
        Instance of KeypointActionClassifier, CropActionClassifier, or PretrainedS3DActionClassifier.
    """
    if isinstance(config, (str, Path)):
        cfg = load_config(config)
    else:
        cfg = dict(config)

    classes = cfg.get("classes", DEFAULT_ACTION_CLASSES)
    selected_backend = backend or cfg.get("default_backend", "keypoint")

    if selected_backend == "keypoint":
        model_cfg = cfg.get("keypoint_model", {})
        merged = {
            "classes": classes,
            "weights": model_cfg.get("weights"),
            "hidden_dim": model_cfg.get("hidden_dim", 64),
            "num_layers": model_cfg.get("num_layers", 2),
            "dropout": model_cfg.get("dropout", 0.2),
            "device": model_cfg.get("device", "cpu"),
            **kwargs,
        }
        return KeypointActionClassifier(**merged)
    elif selected_backend == "crop":
        model_cfg = cfg.get("crop_model", {})
        merged = {
            "classes": classes,
            "weights": model_cfg.get("weights"),
            "backbone": model_cfg.get("backbone", "mobilenet_v3_small"),
            "pretrained": model_cfg.get("pretrained", True),
            "hidden_dim": model_cfg.get("hidden_dim", 128),
            "dropout": model_cfg.get("dropout", 0.3),
            "device": model_cfg.get("device", "cpu"),
            **kwargs,
        }
        return CropActionClassifier(**merged)
    elif selected_backend in ["pretrained_s3d", "s3d"]:
        model_cfg = cfg.get("pretrained_s3d_model", {})
        merged = {
            "classes": classes,
            "weights": model_cfg.get("weights", "DEFAULT"),
            "device": model_cfg.get("device", "cpu"),
            **kwargs,
        }
        return PretrainedS3DActionClassifier(**merged)
    else:
        raise ValueError(
            f"Unknown action classifier backend '{selected_backend}'. "
            f"Expected 'keypoint', 'crop', or 'pretrained_s3d'."
        )

