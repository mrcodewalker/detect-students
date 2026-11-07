"""Unit tests for SORT, ByteTrack, BoT-SORT, and OC-SORT tracker adapters."""

import sys
from pathlib import Path
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.detection.detection import Detection
from src.tracking.bot_sort import BoTSORTTracker
from src.tracking.byte_tracker import ByteTracker
from src.tracking.oc_sort import OCSORTTracker
from src.tracking.sort_tracker import SORTTracker


def test_sort_tracker_id_persistence() -> None:
    """Verify SORTTracker tracks an object across consecutive frames with stable ID."""
    tracker = SORTTracker(max_age=2, min_hits=1, iou_threshold=0.3)

    d0 = [Detection(bbox=(100.0, 100.0, 150.0, 200.0), score=0.9, frame_index=0)]
    t0 = tracker.update(d0)
    assert len(t0) == 1
    track_id = t0[0][0]

    d1 = [Detection(bbox=(102.0, 102.0, 152.0, 202.0), score=0.88, frame_index=1)]
    t1 = tracker.update(d1)
    assert len(t1) == 1
    assert t1[0][0] == track_id

    tracker.reset()
    assert len(tracker.trackers) == 0
    assert tracker.frame_count == 0


def test_bytetrack_two_stage_association_with_low_score() -> None:
    """Verify ByteTrack associates low-score detection in 2nd stage to prevent track loss."""
    tracker = ByteTracker(
        track_thresh=0.5,
        match_thresh=0.8,
        low_thresh=0.1,
        low_match_thresh=0.6,
        confirm_thresh=1,
    )

    d0 = [Detection(bbox=(50.0, 50.0, 100.0, 150.0), score=0.85, frame_index=0)]
    t0 = tracker.update(d0)
    assert len(t0) == 1
    init_id = t0[0][0]

    d1 = [Detection(bbox=(52.0, 51.0, 102.0, 151.0), score=0.25, frame_index=1)]
    t1 = tracker.update(d1)
    assert len(t1) == 1
    assert t1[0][0] == init_id

    tracker.reset()
    assert len(tracker.tracked_stracks) == 0
    assert tracker.frame_id == 0


def test_botsort_tracker_id_persistence() -> None:
    """Verify BoTSORTTracker tracks an object across frames with direct (w, h) Kalman estimation."""
    tracker = BoTSORTTracker(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.5,
        track_buffer=10,
        match_thresh=0.8,
    )

    d0 = [Detection(bbox=(80.0, 80.0, 130.0, 190.0), score=0.9, frame_index=0)]
    t0 = tracker.update(d0)
    assert len(t0) == 1
    track_id = t0[0][0]

    d1 = [Detection(bbox=(82.0, 81.0, 132.0, 191.0), score=0.87, frame_index=1)]
    t1 = tracker.update(d1)
    assert len(t1) == 1
    assert t1[0][0] == track_id

    tracker.reset()
    assert len(tracker.tracked_stracks) == 0


def test_ocsort_tracker_id_persistence() -> None:
    """Verify OCSORTTracker tracks an object across frames with observation-centric smoothing."""
    tracker = OCSORTTracker(
        det_thresh=0.5,
        max_age=10,
        min_hits=1,
        iou_threshold=0.3,
        inertia=0.2,
    )

    d0 = [Detection(bbox=(200.0, 150.0, 260.0, 270.0), score=0.91, frame_index=0)]
    t0 = tracker.update(d0)
    assert len(t0) == 1
    track_id = t0[0][0]

    d1 = [Detection(bbox=(203.0, 152.0, 263.0, 272.0), score=0.89, frame_index=1)]
    t1 = tracker.update(d1)
    assert len(t1) == 1
    assert t1[0][0] == track_id

    tracker.reset()
    assert len(tracker.trackers) == 0


def test_all_four_trackers_from_configs() -> None:
    """Verify loading all 4 trackers from their YAML configuration files."""
    sort = SORTTracker.from_config("configs/tracker_sort.yaml")
    assert sort.iou_threshold == 0.3

    byte = ByteTracker.from_config("configs/tracker_bytetrack.yaml")
    assert byte.track_thresh == 0.5

    bot = BoTSORTTracker.from_config("configs/tracker_botsort.yaml")
    assert bot.track_high_thresh == 0.5
    assert bot.cmc_method == "none"

    oc = OCSORTTracker.from_config("configs/tracker_ocsort.yaml")
    assert oc.det_thresh == 0.5
    assert oc.inertia == 0.2
