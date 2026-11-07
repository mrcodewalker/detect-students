"""Unit tests for FPS and per-stage latency benchmarking tool."""

import sys
from pathlib import Path
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.benchmark_fps import benchmark_single_tracker, get_system_metadata


def test_system_metadata_retrieval() -> None:
    """Verify hardware and environment metadata is queried properly."""
    meta = get_system_metadata()
    assert "gpu_name" in meta
    assert "driver_version" in meta
    assert "torch_version" in meta
    assert "cuda_available" in meta
    assert "cpu_platform" in meta


def test_benchmark_single_tracker_execution() -> None:
    """Verify benchmark runs on sample video and outputs mean, p95, and stage breakdown."""
    sample_video = Path("data/raw/test_sample_classroom.mp4")
    if not sample_video.is_file():
        pytest.skip("Sample classroom video not available for benchmark test.")

    res = benchmark_single_tracker(
        video_path=sample_video,
        tracker_name="bytetrack",
        warmup_frames=2,
        max_frames=6,
    )

    assert res["tracker_name"] == "bytetrack"
    assert "hardware" in res
    assert "config_used" in res
    assert "metrics" in res

    m = res["metrics"]
    for stage in ["decode", "detect", "track", "end_to_end"]:
        assert stage in m
        assert "mean_ms" in m[stage]
        assert "p95_ms" in m[stage]
        assert m[stage]["mean_ms"] >= 0.0
        assert m[stage]["p95_ms"] >= 0.0

    assert "fps" in m["end_to_end"]
    assert m["end_to_end"]["fps"] >= 0.0

