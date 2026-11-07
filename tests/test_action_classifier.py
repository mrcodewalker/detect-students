"""Unit tests for action classifier backends (Prompt 12)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.action.classifier import (
    CropActionClassifier,
    KeypointActionClassifier,
    PretrainedS3DActionClassifier,
    create_action_classifier,
    DEFAULT_ACTION_CLASSES,
)
from src.action.track_buffer import ActionWindow


@pytest.fixture
def sample_action_window() -> ActionWindow:
    """Create a realistic dummy ActionWindow."""
    t_len = 16
    frame_indices = list(range(t_len))
    bboxes = np.tile([100.0, 100.0, 200.0, 300.0], (t_len, 1)).astype(np.float32)
    crops = np.random.randint(0, 255, (t_len, 112, 112, 3), dtype=np.uint8)
    keypoints = np.random.rand(t_len, 17, 3).astype(np.float32)
    scores = np.ones((t_len,), dtype=np.float32)

    return ActionWindow(
        track_id=1,
        start_frame=0,
        end_frame=t_len - 1,
        frame_indices=frame_indices,
        bboxes=bboxes,
        crops=crops,
        keypoints=keypoints,
        scores=scores,
    )


def test_keypoint_action_classifier_predict(sample_action_window: ActionWindow):
    """Test KeypointActionClassifier interface and prediction format."""
    classifier = KeypointActionClassifier(device="cpu")
    label, conf, probs = classifier.predict(sample_action_window)

    assert label in DEFAULT_ACTION_CLASSES
    assert 0.0 <= conf <= 1.0
    assert len(probs) == len(DEFAULT_ACTION_CLASSES)
    assert pytest.approx(sum(probs.values()), rel=1e-3) == 1.0


def test_crop_action_classifier_predict(sample_action_window: ActionWindow):
    """Test CropActionClassifier interface and prediction format."""
    classifier = CropActionClassifier(device="cpu", pretrained=False)
    label, conf, probs = classifier.predict(sample_action_window)

    assert label in DEFAULT_ACTION_CLASSES
    assert 0.0 <= conf <= 1.0
    assert len(probs) == len(DEFAULT_ACTION_CLASSES)
    assert pytest.approx(sum(probs.values()), rel=1e-3) == 1.0


def test_classifier_factory(sample_action_window: ActionWindow):
    """Test factory creation with different backends."""
    clf_kpt = create_action_classifier(backend="keypoint", device="cpu")
    assert isinstance(clf_kpt, KeypointActionClassifier)

    clf_crop = create_action_classifier(backend="crop", pretrained=False, device="cpu")
    assert isinstance(clf_crop, CropActionClassifier)

    batch_res = clf_kpt.predict_batch([sample_action_window, sample_action_window])
    assert len(batch_res) == 2
    for label, conf, probs in batch_res:
        assert label in DEFAULT_ACTION_CLASSES
        assert 0.0 <= conf <= 1.0


def test_load_checkpoint_roundtrip(tmp_path: Path, sample_action_window: ActionWindow):
    """Test saving model weights and loading into a fresh classifier instance."""
    clf = KeypointActionClassifier(device="cpu")
    ckpt_file = tmp_path / "test_ckpt.pt"
    torch.save(clf.model.state_dict(), ckpt_file)

    clf_new = KeypointActionClassifier(device="cpu", weights=ckpt_file)
    assert clf_new.is_trained is True

    label, conf, probs = clf_new.predict(sample_action_window)
    assert label in DEFAULT_ACTION_CLASSES


def test_pretrained_s3d_action_classifier(sample_action_window: ActionWindow):
    """Test Pretrained Google S3D video action classifier interface."""
    clf_s3d = create_action_classifier(backend="pretrained_s3d", device="cpu")
    assert isinstance(clf_s3d, PretrainedS3DActionClassifier)
    assert clf_s3d.is_trained is True

    label, conf, probs = clf_s3d.predict(sample_action_window)
    assert label in DEFAULT_ACTION_CLASSES
    assert 0.0 <= conf <= 1.0
    assert len(probs) == len(DEFAULT_ACTION_CLASSES)

