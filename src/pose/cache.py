"""Disk caching utilities for 2D pose estimation results per video."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.pose.base import PoseDetection


def get_default_cache_path(
    video_path: Union[str, Path],
    cache_dir: Union[str, Path] = "results/cache/pose",
) -> Path:
    """Derive standard pose cache file path for a given video.

    Args:
        video_path: Path to the input video.
        cache_dir: Directory where cache files are saved.

    Returns:
        Path pointing to the <video_stem>_pose.json cache file.
    """
    video_stem = Path(video_path).stem
    return Path(cache_dir) / f"{video_stem}_pose.json"


def save_pose_cache(
    cache_path: Union[str, Path],
    frames_poses: Dict[int, List[PoseDetection]],
    video_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save frame-by-frame pose detections to disk in JSON format.

    Args:
        cache_path: Target cache file path.
        frames_poses: Mapping from frame_index to list of PoseDetection.
        video_path: Optional original video path for traceability.

    Returns:
        Resolved Path of saved cache file.
    """
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized_frames: Dict[str, List[Dict[str, Any]]] = {}
    total_poses = 0
    for frame_idx, poses in frames_poses.items():
        serialized_frames[str(frame_idx)] = [p.to_dict() for p in poses]
        total_poses += len(poses)

    payload = {
        "video_path": str(video_path) if video_path else "",
        "total_frames": len(frames_poses),
        "total_poses": total_poses,
        "frames": serialized_frames,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path


def load_pose_cache(
    cache_path: Union[str, Path],
) -> Optional[Dict[int, List[PoseDetection]]]:
    """Load cached pose detections from disk if present.

    Args:
        cache_path: Target cache file path.

    Returns:
        Mapping from frame_index to list of PoseDetection, or None if not found.
    """
    path = Path(cache_path)
    if not path.is_file():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        frames_dict = data.get("frames", {})
        result: Dict[int, List[PoseDetection]] = {}

        for frame_key, raw_poses in frames_dict.items():
            frame_idx = int(frame_key)
            result[frame_idx] = [PoseDetection.from_dict(p) for p in raw_poses]

        return result
    except Exception:
        # Corrupted cache file
        return None


def has_pose_cache(cache_path: Union[str, Path]) -> bool:
    """Check whether a valid pose cache file exists."""
    path = Path(cache_path)
    return path.is_file() and path.stat().st_size > 0

