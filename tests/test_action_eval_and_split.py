"""Unit tests for action smoothing, video-level splitting, and action evaluation (Prompt 13)."""

from pathlib import Path

import numpy as np
import pytest

from src.action.smoothing import ActionSmoother
from src.eval.action_eval import (
    compute_action_metrics,
    evaluate_gt_tracks,
    evaluate_pred_tracks,
    plot_confusion_matrix,
)
from src.utils.split import (
    DataLeakageError,
    split_by_video,
    validate_video_splits,
)

CLASSES = ["sitting", "standing", "raising_hand", "using_phone"]


# =========================================================================
# 1. Action Temporal Smoothing Tests
# =========================================================================

def test_majority_vote_smoothing():
    """Test majority vote smoothing window and output."""
    smoother = ActionSmoother(method="majority_vote", window_len=3, classes=CLASSES)
    tid = 1

    # First prediction: "sitting"
    lbl, conf = smoother.update(tid, "sitting", 0.9)
    assert lbl == "sitting"

    # Second prediction: "standing" -> 1 sitting, 1 standing -> sitting has higher conf (0.9 vs 0.7)
    lbl, conf = smoother.update(tid, "standing", 0.7)
    assert lbl == "sitting"

    # Third prediction: "standing" -> 1 sitting, 2 standing -> majority is standing
    lbl, conf = smoother.update(tid, "standing", 0.8)
    assert lbl == "standing"


def test_ema_hysteresis_smoothing():
    """Test EMA probability smoothing with hysteresis switching threshold."""
    smoother = ActionSmoother(
        method="ema",
        alpha=0.4,
        hysteresis_threshold=0.6,
        classes=CLASSES,
    )
    tid = 2

    # Initial state: strong "sitting"
    probs_sitting = {"sitting": 0.9, "standing": 0.05, "raising_hand": 0.03, "using_phone": 0.02}
    lbl, conf = smoother.update(tid, "sitting", 0.9, probs=probs_sitting)
    assert lbl == "sitting"

    # Transient noise: 1 frame of weak "standing" (p=0.45)
    probs_noise = {"sitting": 0.4, "standing": 0.45, "raising_hand": 0.1, "using_phone": 0.05}
    lbl, conf = smoother.update(tid, "standing", 0.45, probs=probs_noise)
    # Hysteresis threshold is 0.6, so standing (EMA ~ 0.4*0.45 + 0.6*0.05 = 0.21) does NOT switch label!
    assert lbl == "sitting"

    # Sustained strong "standing"
    probs_standing = {"sitting": 0.05, "standing": 0.92, "raising_hand": 0.02, "using_phone": 0.01}
    for _ in range(5):
        lbl, conf = smoother.update(tid, "standing", 0.92, probs=probs_standing)

    assert lbl == "standing"
    assert conf >= 0.6


# =========================================================================
# 2. Video Splitting and Leakage Prevention Tests
# =========================================================================

def test_split_by_video_disjoint():
    """Test video-level split partitions without overlap."""
    videos = [f"video_{i:02d}.mp4" for i in range(20)]
    splits = split_by_video(videos, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert len(train_set) > 0
    assert len(val_set) > 0
    assert len(test_set) > 0
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set) + len(val_set) + len(test_set) == len(videos)


def test_validate_video_splits_raises_on_leakage():
    """PROVE that an error is raised if any video appears in more than one split."""
    # Video 'session_A.mp4' appears in both train and val splits
    leaking_splits = {
        "train": ["session_A.mp4", "session_B.mp4", "session_C.mp4"],
        "val": ["session_A.mp4", "session_D.mp4"],
        "test": ["session_E.mp4"],
    }

    with pytest.raises(DataLeakageError) as exc_info:
        validate_video_splits(leaking_splits)

    # Verify error message clearly identifies the leaking video
    assert "DATA LEAKAGE DETECTED" in str(exc_info.value)
    assert "session_A.mp4" in str(exc_info.value)


# =========================================================================
# 3. Action Evaluation Tests (Both Modes)
# =========================================================================

def test_evaluate_gt_tracks_mode():
    """Test Mode 1: Evaluation on ground truth tracks."""
    y_true = ["sitting", "sitting", "standing", "raising_hand"]
    y_pred = ["sitting", "standing", "standing", "raising_hand"]

    metrics = evaluate_gt_tracks(y_true, y_pred, classes=CLASSES)
    assert metrics.accuracy == 0.75  # 3 / 4
    assert metrics.precision_per_class["sitting"] == 1.0  # 1 / 1
    assert metrics.recall_per_class["sitting"] == 0.5     # 1 / 2
    assert metrics.f1_per_class["raising_hand"] == 1.0
    assert metrics.confusion_matrix.shape == (len(CLASSES), len(CLASSES))


def test_evaluate_pred_tracks_mode():
    """Test Mode 2: Evaluation on predicted tracks matched by bounding box IoU."""
    gt_obs = [
        {"frame_idx": 0, "bbox": (10.0, 10.0, 100.0, 100.0), "track_id": 1, "action": "sitting"},
        {"frame_idx": 0, "bbox": (200.0, 200.0, 300.0, 300.0), "track_id": 2, "action": "standing"},
    ]
    pred_obs = [
        # Box 1 matches GT 1 with high IoU
        {"frame_idx": 0, "bbox": (12.0, 11.0, 98.0, 102.0), "track_id": 101, "action": "sitting"},
        # Box 2 matches GT 2 with high IoU
        {"frame_idx": 0, "bbox": (202.0, 198.0, 298.0, 301.0), "track_id": 102, "action": "standing"},
        # Box 3 false positive
        {"frame_idx": 0, "bbox": (500.0, 500.0, 600.0, 600.0), "track_id": 103, "action": "using_phone"},
    ]

    metrics = evaluate_pred_tracks(gt_obs, pred_obs, classes=CLASSES, iou_threshold=0.5)
    assert metrics.accuracy == 1.0
    assert metrics.support_per_class["sitting"] == 1
    assert metrics.support_per_class["standing"] == 1


def test_plot_confusion_matrix(tmp_path: Path):
    """Test confusion matrix image generation and file output."""
    cm = np.array([
        [10, 2, 0, 1],
        [1, 12, 1, 0],
        [0, 1, 8, 2],
        [1, 0, 1, 9],
    ], dtype=np.int64)

    out_file = tmp_path / "confusion_matrix.png"
    saved = plot_confusion_matrix(cm, CLASSES, out_file)

    assert saved.is_file()
    assert saved.stat().st_size > 1000

