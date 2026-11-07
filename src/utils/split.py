"""Data splitting utilities enforcing video-level isolation to prevent frame-level leakage."""

from typing import Any, Dict, List, Sequence, Set, Union
import numpy as np


class DataLeakageError(ValueError):
    """Raised when data leakage occurs across video/session splits."""
    pass


def validate_video_splits(splits: Dict[str, Sequence[str]]) -> None:
    """Verify that every video ID appears in exactly one split.

    Args:
        splits: Dictionary mapping split names (e.g. 'train', 'val', 'test')
            to sequences of video identifiers.

    Raises:
        DataLeakageError: If any video identifier appears in multiple splits.
    """
    seen_videos: Dict[str, str] = {}
    overlapping: Dict[str, List[str]] = {}

    for split_name, video_ids in splits.items():
        for vid in video_ids:
            vid_str = str(vid)
            if vid_str in seen_videos:
                prior_split = seen_videos[vid_str]
                if vid_str not in overlapping:
                    overlapping[vid_str] = [prior_split, split_name]
                else:
                    overlapping[vid_str].append(split_name)
            else:
                seen_videos[vid_str] = split_name

    if overlapping:
        err_msg = (
            f"DATA LEAKAGE DETECTED: {len(overlapping)} video(s) appear in multiple splits! "
            f"Details: {overlapping}. "
            "Data MUST be split strictly by video/session, NEVER by frame."
        )
        raise DataLeakageError(err_msg)


def split_by_video(
    video_ids: Sequence[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    shuffle: bool = True,
) -> Dict[str, List[str]]:
    """Split videos into train, validation, and test subsets with zero cross-split overlap.

    Args:
        video_ids: Sequence of unique video/session identifiers.
        train_ratio: Proportion of videos allocated to training.
        val_ratio: Proportion of videos allocated to validation.
        test_ratio: Proportion of videos allocated to testing.
        seed: Random seed for deterministic reproducibility.
        shuffle: Whether to shuffle video IDs before partitioning.

    Returns:
        Dictionary mapping 'train', 'val', 'test' to lists of video identifiers.

    Raises:
        ValueError: If ratios do not sum to 1.0 or are negative.
        DataLeakageError: If verification detects any overlapping video.
    """
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative.")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not (0.999 <= ratio_sum <= 1.001):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.4f}")

    # Remove duplicates from input list
    unique_videos = list(dict.fromkeys([str(v) for v in video_ids]))
    n = len(unique_videos)

    if n == 0:
        return {"train": [], "val": [], "test": []}

    indices = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # Ensure remaining goes to test
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits: Dict[str, List[str]] = {
        "train": [unique_videos[i] for i in train_idx],
        "val": [unique_videos[i] for i in val_idx],
        "test": [unique_videos[i] for i in test_idx],
    }

    # Strict validation check
    validate_video_splits(splits)
    return splits

