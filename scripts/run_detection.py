"""CLI tool to execute YOLO student detection on video, save text detections and annotated video."""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.detection.yolo_detector import YOLODetector
from src.io.video_reader import FileVideoSource
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLO person detection on classroom video.",
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
        "--config",
        "-c",
        type=str,
        default="configs/detector.yaml",
        help="Path to detector YAML configuration file.",
    )
    parser.add_argument(
        "--output-txt",
        "-o",
        type=str,
        default="results/detections.txt",
        help="Path to save detection results text file.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Optional path to save annotated output video.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Override model weights path.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override confidence threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device (e.g. 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=None,
        help="Limit number of processed frames.",
    )
    return parser.parse_args()


def get_hardware_info(detector_device: str) -> str:
    """Query and format host machine and GPU hardware description."""
    try:
        import torch

        if torch.cuda.is_available() and detector_device != "cpu":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM) | PyTorch CUDA: {torch.version.cuda}"
    except Exception:
        pass
    return f"CPU Execution (Target Device: {detector_device})"


def main() -> int:
    """Run detection pipeline on input video."""
    args = parse_args()
    input_video = args.video_path or args.video_flag

    if not input_video:
        print("Error: Input video path required. Example: python scripts/run_detection.py <video_path>", file=sys.stderr)
        return 1

    # Load configuration
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.is_file() else {}

    # Apply CLI overrides
    model_cfg = cfg.get("model", cfg)
    if args.weights:
        model_cfg["weights"] = args.weights
    if args.conf is not None:
        model_cfg["conf"] = args.conf
    if args.device:
        model_cfg["device"] = args.device

    print("==================================================")
    print("Classroom Student Detection (YOLO)")
    print("==================================================")
    print(f"Input Video   : {input_video}")
    print(f"Weights       : {model_cfg.get('weights', 'yolov8n.pt')}")
    print(f"Confidence    : {model_cfg.get('conf', 0.35)}")
    print(f"Output TXT    : {args.output_txt}")
    if args.output_video:
        print(f"Output Video  : {args.output_video}")

    # Initialize detector and video source
    detector = YOLODetector(
        weights=model_cfg.get("weights", "yolov8n.pt"),
        conf=model_cfg.get("conf", 0.35),
        iou=model_cfg.get("iou", 0.45),
        imgsz=model_cfg.get("imgsz", 640),
        device=model_cfg.get("device", "cuda:0"),
        half=model_cfg.get("half", False),
        classes=model_cfg.get("classes", [0]),
    )

    hw_info = get_hardware_info(detector.device)
    print(f"Active Device : {detector.device} [{hw_info}]")
    print("--------------------------------------------------")

    source = FileVideoSource(input_video)
    width, height = source.resolution
    fps = source.fps

    # Setup text output writer
    out_txt_path = Path(args.output_txt).resolve()
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_file = open(out_txt_path, "w", encoding="utf-8")
    txt_file.write("# frame_index,x1,y1,x2,y2,score\n")

    # Setup optional video writer
    video_writer: Optional[cv2.VideoWriter] = None
    if args.output_video:
        out_vid_path = Path(args.output_video).resolve()
        out_vid_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (width, height))

    total_frames = 0
    total_detections = 0
    start_time = time.perf_counter()

    try:
        for frame_idx, _, frame in source:
            if args.max_frames and total_frames >= args.max_frames:
                break

            # Run detection
            detections = detector.detect(frame, frame_idx)
            total_detections += len(detections)

            # Record detections
            for det in detections:
                txt_file.write(
                    f"{det.frame_index},{det.x1:.2f},{det.y1:.2f},{det.x2:.2f},{det.y2:.2f},{det.score:.4f}\n"
                )

            # Draw annotations if video output is requested
            if video_writer is not None:
                annotated = frame.copy()
                for det in detections:
                    pt1 = (int(det.x1), int(det.y1))
                    pt2 = (int(det.x2), int(det.y2))
                    cv2.rectangle(annotated, pt1, pt2, (0, 255, 0), 2)
                    label = f"Student {det.score:.2f}"
                    cv2.putText(
                        annotated,
                        label,
                        (pt1[0], max(15, pt1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                video_writer.write(annotated)

            total_frames += 1

    finally:
        source.close()
        txt_file.close()
        if video_writer is not None:
            video_writer.release()

    elapsed = time.perf_counter() - start_time
    measured_fps = total_frames / elapsed if elapsed > 0 else 0.0

    print("--------------------------------------------------")
    print(f"Processed Frames   : {total_frames}")
    print(f"Total Detections   : {total_detections}")
    print(f"Elapsed Time       : {elapsed:.2f} s")
    print(f"Measured Speed     : {measured_fps:.2f} FPS")
    print(f"Hardware & Device  : {hw_info}")
    print(f"Detections Saved To: {out_txt_path}")
    if args.output_video:
        print(f"Annotated Video To : {args.output_video}")
    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())

