"""Video ingestion abstractions and OpenCV-based readers."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np


class VideoSource(ABC):
    """Abstract base class for video inputs (files, streams, cameras)."""

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Iterate over video frames.

        Yields:
            Tuple of (frame_index: int, timestamp_seconds: float, frame: np.ndarray).
        """
        pass

    @abstractmethod
    def read(self) -> Optional[Tuple[int, float, np.ndarray]]:
        """Read the next frame from the video source.

        Returns:
            Tuple of (frame_index, timestamp_seconds, frame) or None if stream ended.
        """
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal or observed frames per second."""
        pass

    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """Video resolution as (width, height)."""
        pass

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Total number of frames (-1 if unknown or live stream)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release underlying video capture resources."""
        pass

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class FileVideoSource(VideoSource):
    """Reads video files from local storage using OpenCV with safe fallback mechanisms."""

    DEFAULT_FALLBACK_FPS: float = 25.0

    def __init__(
        self,
        file_path: Union[str, Path],
        default_fps: float = DEFAULT_FALLBACK_FPS,
    ) -> None:
        """Initialize FileVideoSource.

        Args:
            file_path: Path to the target video file.
            default_fps: Fallback FPS if video container has invalid or variable frame rate metadata.

        Raises:
            FileNotFoundError: If the specified video file does not exist.
            ValueError: If OpenCV cannot open the video file.
        """
        self.path = Path(file_path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {self.path}")

        self._default_fps = default_fps if default_fps > 0 else self.DEFAULT_FALLBACK_FPS
        self._cap = cv2.VideoCapture(str(self.path))

        if not self._cap.isOpened():
            raise ValueError(f"OpenCV failed to open video file: {self.path}")

        # Extract container metadata
        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if raw_fps is None or np.isnan(raw_fps) or raw_fps <= 0:
            self._fps = self._default_fps
        else:
            self._fps = float(raw_fps)

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolution: Tuple[int, int] = (width, height)

        raw_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_count = raw_count if raw_count >= 0 else -1

        self._current_index: int = 0
        self._last_timestamp: float = -1.0

    @property
    def fps(self) -> float:
        """Frames per second (container metadata or fallback)."""
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        """Video frame dimensions as (width, height)."""
        return self._resolution

    @property
    def frame_count(self) -> int:
        """Total frame count from container metadata (-1 if unknown)."""
        return self._frame_count

    def read(self) -> Optional[Tuple[int, float, np.ndarray]]:
        """Read the next valid frame from the video file.

        Returns:
            Tuple of (frame_index, timestamp_seconds, frame) or None at end of stream.
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        # Determine timestamp safely (handles variable FPS or missing POS_MSEC)
        pos_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec is not None and not np.isnan(pos_msec) and pos_msec > 0:
            timestamp = float(pos_msec) / 1000.0
        elif self._fps > 0:
            timestamp = float(self._current_index) / self._fps
        else:
            timestamp = float(self._current_index) / self._default_fps

        # Ensure monotonically non-decreasing timestamps
        if timestamp <= self._last_timestamp:
            step = 1.0 / self._fps if self._fps > 0 else 1.0 / self._default_fps
            timestamp = self._last_timestamp + step

        frame_idx = self._current_index
        self._current_index += 1
        self._last_timestamp = timestamp

        return frame_idx, timestamp, frame

    def __iter__(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Iterate over frames until EOF or read failure."""
        while True:
            item = self.read()
            if item is None:
                break
            yield item

    def close(self) -> None:
        """Release the video capture handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class RTSPVideoSource(VideoSource):
    """RTSP Network Stream Video Source (Stub)."""

    def __init__(self, rtsp_url: str) -> None:
        """Initialize RTSP stream source.

        Args:
            rtsp_url: RTSP streaming endpoint URL.

        Raises:
            NotImplementedError: RTSP streaming is not yet supported.
        """
        super().__init__()
        raise NotImplementedError("RTSPVideoSource is a stub and will be implemented in future work.")

    def read(self) -> Optional[Tuple[int, float, np.ndarray]]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        raise NotImplementedError

    @property
    def fps(self) -> float:
        raise NotImplementedError

    @property
    def resolution(self) -> Tuple[int, int]:
        raise NotImplementedError

    @property
    def frame_count(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        pass

