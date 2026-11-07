"""Unit tests for Tracker abstraction, DummyTracker, and MOTChallenge export."""

import sys
from pathlib import Path
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.detection.detection import Detection
from src.eval.mot_export import export_mot_format, read_mot_format
from src.tracking.base import DummyTracker


def test_dummy_tracker_assigns_ids_and_resets() -> None:
    """Verify DummyTracker assigns incrementing IDs and resets properly."""
    tracker = DummyTracker(start_id=1)

    # Frame 0: 2 detections
    dets_frame_0 = [
        Detection(bbox=(10.0, 20.0, 50.0, 100.0), score=0.9, frame_index=0),
        Detection(bbox=(60.0, 30.0, 120.0, 150.0), score=0.85, frame_index=0),
    ]
    tracks_0 = tracker.update(dets_frame_0)
    assert len(tracks_0) == 2
    assert tracks_0[0] == (1, (10.0, 20.0, 50.0, 100.0), 0.9)
    assert tracks_0[1] == (2, (60.0, 30.0, 120.0, 150.0), 0.85)

    # Frame 1: 1 detection -> should continue incrementing to ID 3
    dets_frame_1 = [
        Detection(bbox=(15.0, 25.0, 55.0, 105.0), score=0.92, frame_index=1),
    ]
    tracks_1 = tracker.update(dets_frame_1)
    assert len(tracks_1) == 1
    assert tracks_1[0][0] == 3

    # Reset tracker
    tracker.reset()
    tracks_reset = tracker.update(dets_frame_1)
    assert len(tracks_reset) == 1
    assert tracks_reset[0][0] == 1


def test_mot_export_1_based_frame_numbering(tmp_path: Path) -> None:
    """Verify export_mot_format writes 1-based frame numbers and correct columns."""
    out_file = tmp_path / "mot_output.txt"

    records = [
        (0, 1, (10.0, 20.0, 60.0, 120.0), 0.95),  # 0-based frame 0 -> 1-based frame 1
        (1, 1, (12.0, 22.0, 62.0, 122.0), 0.93),  # 0-based frame 1 -> 1-based frame 2
        (2, 2, (100.0, 150.0, 180.0, 300.0), 0.88),  # 0-based frame 2 -> 1-based frame 3
    ]

    export_mot_format(records, out_file, is_zero_based=True)

    lines = [line.strip() for line in out_file.read_text().splitlines() if line.strip()]
    assert len(lines) == 3

    # Line 1: frame 1, id 1, x=10.00, y=20.00, w=50.00, h=100.00, conf=0.9500, -1, -1, -1
    assert lines[0] == "1,1,10.00,20.00,50.00,100.00,0.9500,-1,-1,-1"
    # Line 2: frame 2
    assert lines[1] == "2,1,12.00,22.00,50.00,100.00,0.9300,-1,-1,-1"
    # Line 3: frame 3, id 2, w=80.00, h=150.00
    assert lines[2] == "3,2,100.00,150.00,80.00,150.00,0.8800,-1,-1,-1"


def test_mot_export_round_trip(tmp_path: Path) -> None:
    """Verify round-trip export and read preserves frames, IDs, boxes, and confidence."""
    out_file = tmp_path / "round_trip.txt"

    original_records = [
        (0, 42, (15.5, 30.2, 75.5, 150.2), 0.8912),
        (5, 77, (200.0, 100.0, 280.0, 260.0), 0.9450),
    ]

    # Export with 0-based input
    export_mot_format(original_records, out_file, is_zero_based=True)

    # Read back to 0-based representation
    loaded_records = read_mot_format(out_file, to_zero_based=True)
    assert len(loaded_records) == 2

    for orig, loaded in zip(original_records, loaded_records):
        orig_frame, orig_id, orig_bbox, orig_conf = orig
        assert loaded.frame == orig_frame
        assert loaded.track_id == orig_id
        for val_orig, val_loaded in zip(orig_bbox, loaded.bbox):
            assert val_loaded == pytest.approx(val_orig, abs=0.01)
        assert loaded.conf == pytest.approx(orig_conf, abs=0.001)

