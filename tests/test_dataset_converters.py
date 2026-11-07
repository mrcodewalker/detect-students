"""Unit tests for dataset converters (DanceTrack, PoseTrack21, CVAT MOT 1.1)."""

import json
import sys
from pathlib import Path
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.convert_dataset import convert_cvat_mot, convert_dancetrack, convert_posetrack21
from src.eval.mot_export import read_mot_format


def test_convert_dancetrack_sample(tmp_path: Path) -> None:
    """Verify DanceTrack raw gt.txt is converted to project MOTChallenge format."""
    seq_dir = tmp_path / "dancetrack0001"
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir(parents=True)

    # DanceTrack format: frame, id, x, y, w, h, 1, 1, 1
    sample_lines = [
        "1,1,100.0,150.0,50.0,120.0,1,1,1",
        "2,1,103.0,152.0,50.0,120.0,1,1,1",
        "2,2,300.0,200.0,45.0,110.0,1,1,1",
    ]
    (gt_dir / "gt.txt").write_text("\n".join(sample_lines), encoding="utf-8")
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=dancetrack0001\n", encoding="utf-8")

    out_dir = tmp_path / "out_dancetrack"
    out_gt = convert_dancetrack(seq_dir, out_dir)

    assert out_gt.is_file()
    assert (out_dir / "seqinfo.ini").is_file()

    records = read_mot_format(out_gt, to_zero_based=False)
    assert len(records) == 3
    assert records[0].frame == 1
    assert records[0].track_id == 1
    assert records[0].bbox == (100.0, 150.0, 150.0, 270.0)


def test_convert_posetrack21_sample(tmp_path: Path) -> None:
    """Verify PoseTrack21 COCO-style JSON is split into MOT boxes and keypoints JSON."""
    json_path = tmp_path / "sample_posetrack.json"

    # Synthetic PoseTrack21 JSON structure
    mock_json = {
        "images": [
            {"id": 101, "file_name": "images/000001.jpg", "frame_id": 1},
            {"id": 102, "file_name": "images/000002.jpg", "frame_id": 2},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 101,
                "track_id": 1,
                "category_id": 1,
                "bbox": [50.0, 60.0, 40.0, 100.0],
                "score": 0.95,
                "keypoints": [55.0, 65.0, 2, 60.0, 70.0, 2] + [0, 0, 0] * 15,
            },
            {
                "id": 2,
                "image_id": 102,
                "track_id": 1,
                "category_id": 1,
                "bbox": [52.0, 62.0, 40.0, 100.0],
                "score": 0.94,
                "keypoints": [57.0, 67.0, 2, 62.0, 72.0, 2] + [0, 0, 0] * 15,
            },
        ],
    }
    json_path.write_text(json.dumps(mock_json), encoding="utf-8")

    out_dir = tmp_path / "out_posetrack"
    out_gt, out_kpts = convert_posetrack21(json_path, out_dir)

    assert out_gt.is_file()
    assert out_kpts.is_file()

    # Verify MOT bounding boxes
    records = read_mot_format(out_gt, to_zero_based=False)
    assert len(records) == 2
    assert records[0].frame == 1
    assert records[0].track_id == 1
    assert records[0].bbox == (50.0, 60.0, 90.0, 160.0)

    # Verify keypoints JSON
    kpts_data = json.loads(out_kpts.read_text(encoding="utf-8"))
    assert len(kpts_data) == 2
    assert kpts_data[0]["frame"] == 1
    assert kpts_data[0]["track_id"] == 1
    assert len(kpts_data[0]["keypoints"]) == 51  # 17 joints * 3


def test_convert_cvat_mot_sample(tmp_path: Path) -> None:
    """Verify CVAT exported MOT 1.1 file is normalized to project MOTChallenge format."""
    cvat_gt = tmp_path / "cvat_gt.txt"

    # CVAT MOT 1.1: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility
    cvat_lines = [
        "1, 5, 20.0, 30.0, 40.0, 80.0, 1.0, 1, 1.0",
        "2, 5, 22.0, 32.0, 40.0, 80.0, 1.0, 1, 1.0",
        "2, 6, 120.0, 80.0, 45.0, 90.0, 1.0, 2, 1.0",  # class_id 2 (e.g. distractor/teacher)
    ]
    cvat_gt.write_text("\n".join(cvat_lines), encoding="utf-8")

    out_dir = tmp_path / "out_cvat"
    # Convert filtering for student class_id == 1
    out_gt = convert_cvat_mot(cvat_gt, out_dir, target_class_id=1)

    assert out_gt.is_file()
    records = read_mot_format(out_gt, to_zero_based=False)
    assert len(records) == 2  # class_id 2 was filtered out
    assert records[0].track_id == 5
    assert records[1].track_id == 5
    assert records[0].frame == 1
    assert records[1].frame == 2

