"""CLI tool to run multi-object tracking (SORT / ByteTrack) on video and export MOT format."""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.detection.detection import Detection
from src.eval.mot_export import export_mot_format
from src.io.video_reader import FileVideoSource
from src.tracking.base import Tracker
from src.tracking.bot_sort import BoTSORTTracker
from src.tracking.byte_tracker import ByteTracker
from src.tracking.oc_sort import OCSORTTracker
from src.tracking.sort_tracker import SORTTracker


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multi-object tracking (SORT / ByteTrack) on classroom video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "video_path",
        type=str,
        nargs="?",
        default=None,
        help="Input video file path (positional).",
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        dest="video_flag",
        default=None,
        help="Input video file path (flag).",
    )
    parser.add_argument(
        "--tracker",
        "-t",
        type=str,
        default="bytetrack",
        choices=["sort", "bytetrack", "botsort", "ocsort"],
        help="Target tracking algorithm ('sort', 'bytetrack', 'botsort', 'ocsort').",
    )
    parser.add_argument(
        "--detections",
        "-d",
        type=str,
        default=None,
        help="Optional path to precomputed detection text file (frame,x1,y1,x2,y2,score).",
    )
    parser.add_argument(
        "--tracker-config",
        type=str,
        default=None,
        help="Optional path to tracker YAML config file.",
    )
    parser.add_argument(
        "--detector-config",
        type=str,
        default="configs/detector.yaml",
        help="Detector YAML config file (used if --detections is not provided).",
    )
    parser.add_argument(
        "--output-txt",
        "-o",
        type=str,
        default=None,
        help="Output MOTChallenge txt file path.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Optional path to save annotated tracking video.",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=None,
        help="Limit processed frames.",
    )
    return parser.parse_args()


def load_precomputed_detections(txt_path: Union[str, Path]) -> Dict[int, List[Detection]]:
    """Load precomputed detections from text file grouped by frame index."""
    path = Path(txt_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Detections file not found: {path}")

    frame_dets: Dict[int, List[Detection]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            parts = [p.strip() for p in clean.split(",")]
            if len(parts) >= 6:
                frame_idx = int(float(parts[0]))
                x1 = float(parts[1])
                y1 = float(parts[2])
                x2 = float(parts[3])
                y2 = float(parts[4])
                score = float(parts[5])
                frame_dets[frame_idx].append(
                    Detection(bbox=(x1, y1, x2, y2), score=score, frame_index=frame_idx)
                )
    return frame_dets


def create_tracker(tracker_name: str, config_path: Optional[str]) -> Tracker:
    """Factory helper to instantiate tracker from name and config."""
    name = tracker_name.lower()
    if name == "sort":
        cfg_file = config_path or "configs/tracker_sort.yaml"
        if Path(cfg_file).is_file():
            return SORTTracker.from_config(cfg_file)
        return SORTTracker()
    elif name == "bytetrack":
        cfg_file = config_path or "configs/tracker_bytetrack.yaml"
        if Path(cfg_file).is_file():
            return ByteTracker.from_config(cfg_file)
        return ByteTracker()
    elif name == "botsort":
        cfg_file = config_path or "configs/tracker_botsort.yaml"
        if Path(cfg_file).is_file():
            return BoTSORTTracker.from_config(cfg_file)
        return BoTSORTTracker()
    elif name == "ocsort":
        cfg_file = config_path or "configs/tracker_ocsort.yaml"
        if Path(cfg_file).is_file():
            return OCSORTTracker.from_config(cfg_file)
        return OCSORTTracker()
    else:
        raise ValueError(f"Unsupported tracker: {tracker_name}")


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate distinct pseudo-random color for visualization based on track ID."""
    r = int((track_id * 37) % 256)
    g = int((track_id * 89) % 256)
    b = int((track_id * 151) % 256)
    # Ensure minimum brightness
    return (max(40, b), max(40, g), max(40, r))


def main() -> int:
    """Execute tracking pipeline."""
    args = parse_args()
    input_video = args.video_path or args.video_flag

    if not input_video:
        print("Error: Input video path required. Example: python scripts/run_tracking.py <video> -t bytetrack", file=sys.stderr)
        return 1

    tracker_name = args.tracker.lower()
    default_out_txt = f"results/{tracker_name}_tracks_mot.txt"
    out_txt = args.output_txt or default_out_txt

    print("==================================================")
    print("Classroom Multi-Object Tracking")
    print("==================================================")
    print(f"Input Video    : {input_video}")
    print(f"Tracker Engine : {tracker_name.upper()}")
    print(f"Output MOT TXT : {out_txt}")
    if args.output_video:
        print(f"Output Video   : {args.output_video}")

    tracker = create_tracker(tracker_name, args.tracker_config)

    # Prepare detections source
    detector = None
    precomputed_dets: Optional[Dict[int, List[Detection]]] = None
    if args.detections:
        print(f"Detections In  : {args.detections} (Precomputed)")
        precomputed_dets = load_precomputed_detections(args.detections)
    else:
        print(f"Detector Config: {args.detector_config} (Live YOLO Detection)")
        from src.detection.yolo_detector import YOLODetector
        detector = YOLODetector.from_config(args.detector_config)

    source = FileVideoSource(input_video)
    width, height = source.resolution
    fps = source.fps

    # Setup video writer if requested
    video_writer: Optional[cv2.VideoWriter] = None
    if args.output_video:
        out_vid_path = Path(args.output_video).resolve()
        out_vid_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (width, height))

    records: List[Tuple[int, int, Tuple[float, float, float, float], float]] = []
    track_trajectories: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    unique_track_ids = set()

    total_frames = 0
    start_time = time.perf_counter()

    try:
        for frame_idx, _, frame in source:
            if args.max_frames and total_frames >= args.max_frames:
                break

            # 1. Obtain detections
            if precomputed_dets is not None:
                current_dets = precomputed_dets.get(frame_idx, [])
            else:
                assert detector is not None
                current_dets = detector.detect(frame, frame_idx)

            # 2. Run tracker update
            tracks = tracker.update(current_dets, frame=frame)

            # 3. Collect tracking records
            for track_id, bbox, score in tracks:
                unique_track_ids.add(track_id)
                records.append((frame_idx, track_id, bbox, score))

            # 4. Optional video annotation
            if video_writer is not None:
                annotated = frame.copy()
                for track_id, bbox, score in tracks:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    color = get_track_color(track_id)

                    # Update and draw trajectory line
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    track_trajectories[track_id].append((cx, cy))
                    if len(track_trajectories[track_id]) > 30:
                        track_trajectories[track_id].pop(0)

                    pts = track_trajectories[track_id]
                    for p in range(1, len(pts)):
                        cv2.line(annotated, pts[p - 1], pts[p], color, 2)

                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    # Label text
                    label = f"ID:{track_id} ({score:.2f})"
                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        annotated,
                        (x1, max(0, y1 - label_size[1] - 4)),
                        (x1 + label_size[0], max(label_size[1] + 4, y1)),
                        color,
                        cv2.FILLED,
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(label_size[1] + 2, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                video_writer.write(annotated)

            total_frames += 1

    finally:
        source.close()
        if video_writer is not None:
            video_writer.release()

    # Export MOTChallenge format
    export_mot_format(records, out_txt, is_zero_based=True)

    elapsed = time.perf_counter() - start_time
    measured_fps = total_frames / elapsed if elapsed > 0 else 0.0

    print("--------------------------------------------------")
    print(f"Processed Frames   : {total_frames}")
    print(f"Total Unique Tracks: {len(unique_track_ids)}")
    print(f"Total Track Records: {len(records)}")
    print(f"Elapsed Time       : {elapsed:.2f} s")
    print(f"Processing Speed   : {measured_fps:.2f} FPS")
    print(f"MOT Output Saved   : {out_txt}")
    if args.output_video:
        print(f"Annotated Video    : {args.output_video}")
    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
