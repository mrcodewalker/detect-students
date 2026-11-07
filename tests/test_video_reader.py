"""Unit tests for video reader abstraction and FileVideoSource."""

import sys
from pathlib import Path
import pytest
import cv2
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.io.video_reader import FileVideoSource, RTSPVideoSource


def _create_synthetic_video(file_path: Path, num_frames: int = 15, width: int = 64, height: int = 64, fps: float = 30.0) -> None:
    """Helper to generate a small synthetic video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"Could not create video writer for {file_path}"

    for i in range(num_frames):
        # Generate distinct solid color frame
        color_val = int((i * 15) % 256)
        frame = np.full((height, width, 3), color_val, dtype=np.uint8)
        writer.write(frame)

    writer.release()


def test_file_video_source_read_synthetic(tmp_path: Path) -> None:
    """Test reading frames from a generated synthetic video."""
    video_path = tmp_path / "test_sample.mp4"
    total_frames = 15
    width, height = 64, 64
    nominal_fps = 30.0

    _create_synthetic_video(video_path, num_frames=total_frames, width=width, height=height, fps=nominal_fps)

    with FileVideoSource(video_path) as source:
        assert source.resolution == (width, height)
        assert abs(source.fps - nominal_fps) < 1.0

        frames_read = 0
        last_timestamp = -1.0

        for frame_idx, timestamp_sec, frame in source:
            assert frame_idx == frames_read
            assert timestamp_sec > last_timestamp
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (height, width, 3)
            assert frame.dtype == np.uint8

            last_timestamp = timestamp_sec
            frames_read += 1

        assert frames_read == total_frames


def test_file_video_source_missing_file(tmp_path: Path) -> None:
    """Test that FileVideoSource raises FileNotFoundError on nonexistent files."""
    missing_path = tmp_path / "does_not_exist.mp4"
    with pytest.raises(FileNotFoundError):
        FileVideoSource(missing_path)


def test_rtsp_video_source_stub() -> None:
    """Test that RTSPVideoSource raises NotImplementedError as a placeholder stub."""
    with pytest.raises(NotImplementedError):
        RTSPVideoSource("rtsp://127.0.0.1:554/live")

