"""Unit tests for spatio-temporal track action buffer (Prompt 11)."""

import numpy as np
import pytest

from src.action.track_buffer import (
    ActionWindow,
    FrameObservation,
    SingleTrackBuffer,
    TrackActionBuffer,
)


def test_invalid_parameters():
    """Verify ValueError raised for illegal buffer configuration."""
    with pytest.raises(ValueError):
        TrackActionBuffer(window_size=0)
    with pytest.raises(ValueError):
        TrackActionBuffer(window_size=16, stride=0)
    with pytest.raises(ValueError):
        TrackActionBuffer(window_size=16, max_missing_frames=-1)


def test_track_buffer_creation():
    """Verify creation and population of track buffers."""
    buf_mgr = TrackActionBuffer(window_size=8, max_missing_frames=10, stride=2)
    assert len(buf_mgr) == 0

    frame_dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox_1 = (100.0, 100.0, 200.0, 300.0)
    kpts_1 = np.ones((17, 3), dtype=np.float32)

    # Add track 1
    buf_mgr.update_track(track_id=1, frame_idx=0, bbox=bbox_1, score=0.9, frame=frame_dummy, keypoints=kpts_1)
    assert len(buf_mgr) == 1
    assert 1 in buf_mgr
    assert 2 not in buf_mgr

    # Add track 2
    bbox_2 = (300.0, 150.0, 400.0, 350.0)
    buf_mgr.update_track(track_id=2, frame_idx=0, bbox=bbox_2, score=0.85)
    assert len(buf_mgr) == 2
    assert 2 in buf_mgr

    # Reset
    buf_mgr.reset()
    assert len(buf_mgr) == 0


def test_track_buffer_expiry():
    """Verify tracks inactive for > K frames are pruned."""
    # K = 5 frames
    buf_mgr = TrackActionBuffer(window_size=4, max_missing_frames=5, stride=2)

    # Track 1 appears at frame 0, 1, 2, 3
    for f in range(4):
        buf_mgr.update_track(track_id=1, frame_idx=f, bbox=(10.0, 10.0, 50.0, 50.0))

    assert 1 in buf_mgr

    # At frame 8: elapsed since frame 3 is 8 - 3 = 5 frames.
    # 5 <= K (5), so track 1 is NOT expired yet.
    expired = buf_mgr.remove_expired(current_frame_idx=8)
    assert expired == []
    assert 1 in buf_mgr

    # At frame 9: elapsed since frame 3 is 9 - 3 = 6 frames > 5.
    # Track 1 must expire and be removed!
    expired = buf_mgr.remove_expired(current_frame_idx=9)
    assert expired == [1]
    assert 1 not in buf_mgr
    assert len(buf_mgr) == 0


def test_sliding_window_timing():
    """Verify window emission timing adheres strictly to window_size N and stride S."""
    # N = 4 frames, S = 2 frames
    buf_mgr = TrackActionBuffer(window_size=4, max_missing_frames=20, stride=2)
    tid = 10

    # Frames 0, 1, 2: Buffer length < N (4), not ready
    for f in range(3):
        buf_mgr.update_track(track_id=tid, frame_idx=f, bbox=(20.0, 20.0, 60.0, 80.0))
        assert not buf_mgr.is_window_ready(tid, current_frame_idx=f)
        assert buf_mgr.get_window(tid, current_frame_idx=f) is None

    # Frame 3: Buffer has 4 frames -> First window is ready!
    buf_mgr.update_track(track_id=tid, frame_idx=3, bbox=(20.0, 20.0, 60.0, 80.0))
    assert buf_mgr.is_window_ready(tid, current_frame_idx=3)

    win1 = buf_mgr.get_window(tid, current_frame_idx=3)
    assert win1 is not None
    assert win1.track_id == tid
    assert win1.start_frame == 0
    assert win1.end_frame == 3
    assert win1.frame_indices == [0, 1, 2, 3]
    assert win1.bboxes.shape == (4, 4)

    # Frame 4: 1 frame since last emitted (at frame 3). 4 - 3 = 1 < S (2) -> NOT ready
    buf_mgr.update_track(track_id=tid, frame_idx=4, bbox=(20.0, 20.0, 60.0, 80.0))
    assert not buf_mgr.is_window_ready(tid, current_frame_idx=4)
    assert buf_mgr.get_window(tid, current_frame_idx=4) is None

    # Frame 5: 2 frames since last emitted (5 - 3 = 2 == S) -> READY!
    buf_mgr.update_track(track_id=tid, frame_idx=5, bbox=(20.0, 20.0, 60.0, 80.0))
    assert buf_mgr.is_window_ready(tid, current_frame_idx=5)

    win2 = buf_mgr.get_window(tid, current_frame_idx=5)
    assert win2 is not None
    assert win2.start_frame == 2
    assert win2.end_frame == 5
    assert win2.frame_indices == [2, 3, 4, 5]

    # Frame 6: 6 - 5 = 1 < 2 -> NOT ready
    buf_mgr.update_track(track_id=tid, frame_idx=6, bbox=(20.0, 20.0, 60.0, 80.0))
    assert not buf_mgr.is_window_ready(tid, current_frame_idx=6)

    # Frame 7: 7 - 5 = 2 == S -> READY!
    buf_mgr.update_track(track_id=tid, frame_idx=7, bbox=(20.0, 20.0, 60.0, 80.0))
    assert buf_mgr.is_window_ready(tid, current_frame_idx=7)
    win3 = buf_mgr.get_window(tid, current_frame_idx=7)
    assert win3 is not None
    assert win3.start_frame == 4
    assert win3.end_frame == 7


def test_step_method_with_crops_and_keypoints():
    """Verify step method handles crop resizing and keypoint packing."""
    buf_mgr = TrackActionBuffer(window_size=3, max_missing_frames=10, stride=2, crop_size=(64, 64))

    frame_dummy = np.zeros((200, 300, 3), dtype=np.uint8)
    frame_dummy[20:80, 20:80] = 255  # Distinct white patch

    kpts_sample = np.ones((17, 3), dtype=np.float32)

    # Frame 0
    windows_0 = buf_mgr.step(
        frame_idx=0,
        tracks=[(1, (20.0, 20.0, 80.0, 80.0), 0.95)],
        frame=frame_dummy,
        keypoints_map={1: kpts_sample},
    )
    assert len(windows_0) == 0  # not full window

    # Frame 1
    windows_1 = buf_mgr.step(
        frame_idx=1,
        tracks=[(1, (20.0, 20.0, 80.0, 80.0), 0.92)],
        frame=frame_dummy,
        keypoints_map={1: kpts_sample},
    )
    assert len(windows_1) == 0

    # Frame 2: N = 3 frames full -> Ready!
    windows_2 = buf_mgr.step(
        frame_idx=2,
        tracks=[(1, (20.0, 20.0, 80.0, 80.0), 0.90)],
        frame=frame_dummy,
        keypoints_map={1: kpts_sample},
    )
    assert len(windows_2) == 1
    win = windows_2[0]
    assert win.track_id == 1
    assert win.start_frame == 0
    assert win.end_frame == 2
    assert win.crops is not None
    assert win.crops.shape == (3, 64, 64, 3)
    assert win.keypoints is not None
    assert win.keypoints.shape == (3, 17, 3)
