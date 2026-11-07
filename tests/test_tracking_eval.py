"""Unit tests for multi-object tracking evaluation engine and CLI."""

import sys
from pathlib import Path
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.eval.mot_export import MOTRecord, export_mot_format
from src.eval.tracking_evaluator import evaluate_sequences
from scripts.run_tracking_eval import main as run_eval_main


def test_synthetic_perfect_tracking() -> None:
    """Verify that perfect tracking yields 1.0 across all accuracy metrics and 0 ID switches."""
    # 3 frames, 2 targets moving linearly
    gt_records = [
        MOTRecord(frame=1, track_id=1, bbox=(10.0, 10.0, 30.0, 30.0), conf=1.0),
        MOTRecord(frame=1, track_id=2, bbox=(100.0, 100.0, 120.0, 120.0), conf=1.0),
        MOTRecord(frame=2, track_id=1, bbox=(12.0, 12.0, 32.0, 32.0), conf=1.0),
        MOTRecord(frame=2, track_id=2, bbox=(102.0, 102.0, 122.0, 122.0), conf=1.0),
        MOTRecord(frame=3, track_id=1, bbox=(14.0, 14.0, 34.0, 34.0), conf=1.0),
        MOTRecord(frame=3, track_id=2, bbox=(104.0, 104.0, 124.0, 124.0), conf=1.0),
    ]

    # Tracker outputs identical predictions
    trk_records = [
        MOTRecord(frame=1, track_id=1, bbox=(10.0, 10.0, 30.0, 30.0), conf=0.95),
        MOTRecord(frame=1, track_id=2, bbox=(100.0, 100.0, 120.0, 120.0), conf=0.92),
        MOTRecord(frame=2, track_id=1, bbox=(12.0, 12.0, 32.0, 32.0), conf=0.94),
        MOTRecord(frame=2, track_id=2, bbox=(102.0, 102.0, 122.0, 122.0), conf=0.91),
        MOTRecord(frame=3, track_id=1, bbox=(14.0, 14.0, 34.0, 34.0), conf=0.96),
        MOTRecord(frame=3, track_id=2, bbox=(104.0, 104.0, 124.0, 124.0), conf=0.93),
    ]

    metrics = evaluate_sequences(gt_records, trk_records)

    assert metrics["IDF1"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["HOTA"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["MOTA"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["DetA"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["AssA"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["IDSW"] == 0
    assert metrics["Frag"] == 0


def test_synthetic_forced_id_swap() -> None:
    """Verify that a forced identity switch produces exactly 1 ID switch."""
    # 2 frames, 1 continuous ground truth student
    gt_records = [
        MOTRecord(frame=1, track_id=1, bbox=(50.0, 50.0, 80.0, 80.0), conf=1.0),
        MOTRecord(frame=2, track_id=1, bbox=(55.0, 55.0, 85.0, 85.0), conf=1.0),
    ]

    # Tracker assigns ID 10 in frame 1, but mistakenly swaps to ID 20 in frame 2
    trk_records = [
        MOTRecord(frame=1, track_id=10, bbox=(50.0, 50.0, 80.0, 80.0), conf=0.9),
        MOTRecord(frame=2, track_id=20, bbox=(55.0, 55.0, 85.0, 85.0), conf=0.9),
    ]

    metrics = evaluate_sequences(gt_records, trk_records)

    assert metrics["IDSW"] == 1
    assert metrics["IDF1"] < 1.0


def test_run_tracking_eval_cli_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the full evaluation CLI produces summary CSVs and comparison table."""
    gt_dir = tmp_path / "gt"
    trackers_dir = tmp_path / "trackers"
    eval_out_dir = tmp_path / "eval_out"

    gt_dir.mkdir(parents=True)
    trk1_dir = trackers_dir / "tracker_a"
    trk2_dir = trackers_dir / "tracker_b"
    trk1_dir.mkdir(parents=True)
    trk2_dir.mkdir(parents=True)

    # 1. Ground truth sequence
    raw_gt = [(1, 1, (10.0, 10.0, 40.0, 40.0), 1.0), (2, 1, (12.0, 12.0, 42.0, 42.0), 1.0)]
    export_mot_format(raw_gt, gt_dir / "seq1.txt", is_zero_based=False)

    # 2. Tracker A (Perfect)
    raw_trkA = [(1, 1, (10.0, 10.0, 40.0, 40.0), 0.9), (2, 1, (12.0, 12.0, 42.0, 42.0), 0.9)]
    export_mot_format(raw_trkA, trk1_dir / "seq1.txt", is_zero_based=False)

    # 3. Tracker B (1 ID Swap)
    raw_trkB = [(1, 1, (10.0, 10.0, 40.0, 40.0), 0.9), (2, 2, (12.0, 12.0, 42.0, 42.0), 0.9)]
    export_mot_format(raw_trkB, trk2_dir / "seq1.txt", is_zero_based=False)

    # Run CLI main
    test_args = [
        "run_tracking_eval.py",
        "--gt-folder",
        str(gt_dir),
        "--trackers-folder",
        str(trackers_dir),
        "--output-dir",
        str(eval_out_dir),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    ret_code = run_eval_main()
    assert ret_code == 0

    # Verify generated files
    assert (eval_out_dir / "tracker_a_summary.csv").is_file()
    assert (eval_out_dir / "tracker_b_summary.csv").is_file()
    assert (eval_out_dir / "tracking_comparison.csv").is_file()

    comp_content = (eval_out_dir / "tracking_comparison.csv").read_text(encoding="utf-8")
    assert "tracker_a" in comp_content
    assert "tracker_b" in comp_content

