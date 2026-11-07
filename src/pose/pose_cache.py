"""Disk caching utilities for 2D human pose keypoints per video sequence."""

import json
from pathlib import Path
from typing import Dict, List, Union

from src.pose.pose_types import PersonPose


class PoseCacheManager:
    """Manages serialization and retrieval of computed pose keypoints to disk."""

    def __init__(self, cache_dir: Union[str, Path] = "data/processed/pose_cache") -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, video_path: Union[str, Path]) -> Path:
        """Derive deterministic cache file path from input video filename."""
        video_name = Path(video_path).stem
        return self.cache_dir / f"{video_name}_pose_cache.json"

    def has_cache(self, video_path: Union[str, Path]) -> bool:
        """Check whether precomputed pose cache exists on disk."""
        return self.get_cache_path(video_path).is_file()

    def save(
        self,
        video_path: Union[str, Path],
        frame_poses: Dict[int, List[PersonPose]],
    ) -> Path:
        """Save frame pose predictions to JSON cache.

        Args:
            video_path: Source video path.
            frame_poses: Dictionary mapping frame_index to list of PersonPose.

        Returns:
            Path to the written cache file.
        """
        cache_file = self.get_cache_path(video_path)
        payload = {
            "video_name": Path(video_path).name,
            "frames": {
                str(f_idx): [p.to_dict() for p in poses]
                for f_idx, poses in frame_poses.items()
            },
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return cache_file

    def load(self, video_path: Union[str, Path]) -> Dict[int, List[PersonPose]]:
        """Load cached pose predictions from disk.

        Args:
            video_path: Source video path.

        Returns:
            Dictionary mapping frame_index (int) to List[PersonPose].
        """
        cache_file = self.get_cache_path(video_path)
        if not cache_file.is_file():
            raise FileNotFoundError(f"Pose cache not found: {cache_file}")

        with open(cache_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        raw_frames = payload.get("frames", {})
        reconstructed: Dict[int, List[PersonPose]] = {}
        for f_str, p_list in raw_frames.items():
            f_idx = int(f_str)
            reconstructed[f_idx] = [PersonPose.from_dict(item) for item in p_list]
        return reconstructed
