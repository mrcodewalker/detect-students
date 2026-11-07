"""Dataset conversion tools for DanceTrack, PoseTrack21, and CVAT MOT 1.1 annotations.

Supported formats:
1. DanceTrack: Converts DanceTrack sequence directories (gt/gt.txt) to standard MOTChallenge format.
2. PoseTrack21: Parses COCO/PoseTrack JSON annotations, exports bounding boxes and track IDs to
   standard MOT format, and saves 2D pose keypoints into a separate JSON file.
3. CVAT MOT 1.1: Normalizes CVAT-exported MOT 1.1 annotation files (gt.txt) into clean project MOT format.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.eval.mot_export import MOTRecord, export_mot_format, read_mot_format


def convert_dancetrack(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> Path:
    """Convert DanceTrack annotations to project MOTChallenge format.

    Expected input structure:
        dancetrack_seq/
            gt/
                gt.txt
            seqinfo.ini (optional)

    Args:
        input_path: Path to DanceTrack sequence folder or directly to gt.txt.
        output_dir: Target directory where normalized gt.txt will be stored.

    Returns:
        Path to the converted gt.txt file.
    """
    in_p = Path(input_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate gt.txt
    if in_p.is_file():
        gt_file = in_p
    elif (in_p / "gt" / "gt.txt").is_file():
        gt_file = in_p / "gt" / "gt.txt"
    elif (in_p / "gt.txt").is_file():
        gt_file = in_p / "gt.txt"
    else:
        raise FileNotFoundError(f"Could not locate gt.txt in DanceTrack path: {in_p}")

    records = read_mot_format(gt_file, to_zero_based=False)
    out_gt = out_dir / "gt.txt"
    export_mot_format(
        [(r.frame, r.track_id, r.bbox, r.conf) for r in records],
        out_gt,
        is_zero_based=False,
    )

    # Copy seqinfo.ini if present
    seqinfo_candidates = [
        in_p / "seqinfo.ini",
        gt_file.parent.parent / "seqinfo.ini",
    ]
    for s in seqinfo_candidates:
        if s.is_file():
            shutil.copy2(s, out_dir / "seqinfo.ini")
            break

    return out_gt


def convert_posetrack21(
    input_json_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """Convert PoseTrack21 JSON annotations to MOT format (boxes + IDs) and separate keypoints JSON.

    Expected input:
        PoseTrack21 COCO-style JSON annotation file containing 'images' and 'annotations'.
        Each annotation entry contains:
            - image_id / frame_id
            - track_id
            - bbox: [x, y, w, h]
            - keypoints: [x1, y1, v1, x2, y2, v2, ...] (17 keypoints)

    Args:
        input_json_path: Path to the PoseTrack21 annotation JSON file.
        output_dir: Target directory for converted assets.

    Returns:
        Tuple of (path_to_mot_gt_txt, path_to_keypoints_json).
    """
    json_path = Path(input_json_path).resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"PoseTrack21 JSON annotation file not found: {json_path}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build image_id to frame index mapping
    images = data.get("images", [])
    img_to_frame: Dict[int, int] = {}
    for img in images:
        img_id = img["id"]
        # Handle 1-based frame_id or image index
        frame_idx = img.get("frame_id", img.get("image_id", None))
        if frame_idx is None:
            # Fallback to file_name suffix if numeric or sequential order
            stem = Path(img.get("file_name", "")).stem
            frame_idx = int(stem) if stem.isdigit() else img_id
        img_to_frame[img_id] = int(frame_idx)

    annotations = data.get("annotations", [])
    mot_records: List[Tuple[int, int, Tuple[float, float, float, float], float]] = []
    keypoints_records: List[Dict[str, Any]] = []

    for ann in annotations:
        # Ignore non-person annotations if specified
        if "category_id" in ann and ann["category_id"] != 1:
            continue

        image_id = ann["image_id"]
        frame_num = img_to_frame.get(image_id, image_id)
        track_id = int(ann.get("track_id", -1))
        if track_id < 0:
            # Skip unassociated/ignored background detections
            continue

        bbox_raw = ann.get("bbox", [])
        if len(bbox_raw) >= 4:
            x, y, w, h = [float(v) for v in bbox_raw[:4]]
            bbox_xyxy = (x, y, x + w, y + h)
            conf = float(ann.get("score", 1.0))
            mot_records.append((frame_num, track_id, bbox_xyxy, conf))

        # Extract 2D pose keypoints
        kpts = ann.get("keypoints", [])
        if kpts:
            keypoints_records.append(
                {
                    "frame": frame_num,
                    "track_id": track_id,
                    "bbox": [x, y, w, h],
                    "keypoints": kpts,
                }
            )

    # Sort MOT records by frame then ID
    mot_records.sort(key=lambda x: (x[0], x[1]))
    keypoints_records.sort(key=lambda x: (x["frame"], x["track_id"]))

    out_gt_txt = out_dir / "gt.txt"
    export_mot_format(mot_records, out_gt_txt, is_zero_based=False)

    out_kpts_json = out_dir / "keypoints.json"
    with open(out_kpts_json, "w", encoding="utf-8") as f:
        json.dump(keypoints_records, f, indent=2)

    return out_gt_txt, out_kpts_json


def convert_cvat_mot(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    target_class_id: Optional[int] = None,
) -> Path:
    """Normalize CVAT-exported 'MOT 1.1' annotations into standard project MOT format.

    CVAT 'MOT 1.1' raw format specification:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility

    Args:
        input_path: Path to CVAT gt.txt or folder containing gt/gt.txt.
        output_dir: Destination folder for cleaned gt.txt.
        target_class_id: Optional class ID filter (e.g. 1 for person/student).

    Returns:
        Path to normalized gt.txt file.
    """
    in_p = Path(input_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_p.is_file():
        gt_file = in_p
    elif (in_p / "gt" / "gt.txt").is_file():
        gt_file = in_p / "gt" / "gt.txt"
    elif (in_p / "gt.txt").is_file():
        gt_file = in_p / "gt.txt"
    else:
        raise FileNotFoundError(f"Could not locate CVAT gt.txt in: {in_p}")

    records: List[Tuple[int, int, Tuple[float, float, float, float], float]] = []

    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue

            parts = [p.strip() for p in clean.split(",")]
            if len(parts) < 6:
                continue

            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0

            # Filter by class_id if specified in CVAT MOT 1.1 format
            if target_class_id is not None and len(parts) > 7:
                cls_id = int(float(parts[7]))
                if cls_id != target_class_id:
                    continue

            bbox_xyxy = (x, y, x + w, y + h)
            records.append((frame, track_id, bbox_xyxy, conf))

    # Sort deterministically
    records.sort(key=lambda r: (r[0], r[1]))

    out_gt = out_dir / "gt.txt"
    export_mot_format(records, out_gt, is_zero_based=False)
    return out_gt


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dataset converters for DanceTrack, PoseTrack21, and CVAT MOT 1.1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        choices=["dancetrack", "posetrack21", "cvat"],
        help="Type of input dataset to convert.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input dataset file or folder.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Target folder for converted outputs.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Optional class ID filter for CVAT MOT 1.1.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI execution entrypoint."""
    args = parse_args()
    dataset_type = args.dataset.lower()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)

    print("==================================================")
    print("Dataset Conversion Utility")
    print("==================================================")
    print(f"Dataset Type: {dataset_type.upper()}")
    print(f"Input Path  : {in_path}")
    print(f"Output Dir  : {out_dir}")
    print("--------------------------------------------------")

    try:
        if dataset_type == "dancetrack":
            out_file = convert_dancetrack(in_path, out_dir)
            print(f"Successfully converted DanceTrack -> {out_file}")
        elif dataset_type == "posetrack21":
            mot_file, kpt_file = convert_posetrack21(in_path, out_dir)
            print(f"Successfully converted PoseTrack21:")
            print(f"  - MOT BBoxes   : {mot_file}")
            print(f"  - Pose Keypoints: {kpt_file}")
        elif dataset_type == "cvat":
            out_file = convert_cvat_mot(in_path, out_dir, target_class_id=args.class_id)
            print(f"Successfully converted CVAT MOT 1.1 -> {out_file}")
        else:
            print(f"Error: Unknown dataset type: {dataset_type}", file=sys.stderr)
            return 1

        print("==================================================")
        return 0
    except Exception as e:
        print(f"Conversion Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

