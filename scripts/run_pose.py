"""CLI script to run 2D pose estimation on video, cache keypoints, and export overlay video."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.io.video_reader import FileVideoSource
from src.pose.base import PoseDetection
from src.pose.cache import get_default_cache_path, has_pose_cache, load_pose_cache, save_pose_cache
from src.pose.visualizer import draw_pose_on_frame
from src.pose.yolo_pose import YOLOPoseEstimator
from src.utils.config import load_config, merge_configs

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_pose")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 17-keypoint 2D pose estimation on video with caching and overlay generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default="data/raw/test_sample_classroom.mp4",
        help="Input video file path.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/pose_overlay.mp4",
        help="Output annotated video file path.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/pose.yaml",
        help="Path to pose configuration YAML file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device override ('cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["whole_frame", "crop"],
        help="Inference mode override ('whole_frame' or 'crop').",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputing poses, bypassing existing disk cache.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable saving computed keypoints to disk cache.",
    )
    return parser.parse_args()


def measure_memory_usage() -> Dict[str, float]:
    """Measure current process memory and GPU VRAM usage."""
    import psutil
    res = {}
    process = psutil.Process()
    res["ram_mb"] = process.memory_info().rss / (1024 * 1024)

    try:
        import torch
        if torch.cuda.is_available():
            res["vram_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            res["vram_current_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            res["vram_peak_mb"] = 0.0
            res["vram_current_mb"] = 0.0
    except Exception:
        res["vram_peak_mb"] = 0.0
        res["vram_current_mb"] = 0.0

    return res


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        logger.error("Input video not found: %s", video_path)
        sys.exit(1)

    # Load configuration
    cfg = {}
    if Path(args.config).is_file():
        cfg = load_config(args.config)

    model_cfg = cfg.get("model", {})
    cache_cfg = cfg.get("cache", {})

    if args.device:
        model_cfg["device"] = args.device
    if args.mode:
        model_cfg["mode"] = args.mode

    cache_dir = cache_cfg.get("dir", "results/cache/pose")
    cache_enabled = cache_cfg.get("enabled", True) and not args.no_cache
    cache_file = get_default_cache_path(video_path, cache_dir)

    frames_poses: Dict[int, List[PoseDetection]] = {}
    cache_loaded = False

    # Check cache
    if cache_enabled and not args.force_recompute and has_pose_cache(cache_file):
        logger.info("Found existing pose cache at: %s", cache_file)
        cached = load_pose_cache(cache_file)
        if cached is not None:
            frames_poses = cached
            cache_loaded = True
            logger.info("Loaded poses for %d frames from cache.", len(frames_poses))

    # Initialize video reader
    video_source = FileVideoSource(str(video_path))
    total_frames = video_source.frame_count
    fps = video_source.fps
    width, height = video_source.resolution

    logger.info(
        "Opened video: %s (%d frames, %dx%d, %.2f FPS)",
        video_path.name,
        total_frames,
        width,
        height,
        fps,
    )

    # Setup video writer for overlay
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Initialize model if needed
    estimator: Optional[YOLOPoseEstimator] = None
    if not cache_loaded:
        estimator = YOLOPoseEstimator(
            weights=model_cfg.get("weights", "yolov8n-pose.pt"),
            conf=model_cfg.get("conf", 0.35),
            iou=model_cfg.get("iou", 0.45),
            kpt_conf=model_cfg.get("kpt_conf", 0.30),
            imgsz=model_cfg.get("imgsz", 640),
            device=model_cfg.get("device", "cpu"),
            half=model_cfg.get("half", False),
            mode=model_cfg.get("mode", "whole_frame"),
        )

    kpt_conf_thresh = model_cfg.get("kpt_conf", 0.30)

    start_time = time.perf_counter()
    processed_count = 0

    logger.info("Processing video frames...")
    try:
        for frame_idx, timestamp, frame in video_source:
            if cache_loaded:
                poses = frames_poses.get(frame_idx, [])
            else:
                assert estimator is not None
                poses = estimator.estimate(frame, frame_index=frame_idx)
                frames_poses[frame_idx] = poses

            # Render overlay onto frame
            annotated = draw_pose_on_frame(
                frame,
                poses,
                kpt_conf_thresh=kpt_conf_thresh,
                draw_box=True,
                draw_labels=True,
            )

            # Draw frame number and inference stats
            info_text = f"Frame: {frame_idx + 1}/{total_frames} | Persons: {len(poses)}"
            cv2.putText(
                annotated,
                info_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated)
            processed_count += 1
    finally:
        video_source.close()
        writer.release()
    elapsed_time = time.perf_counter() - start_time
    inference_fps = processed_count / elapsed_time if elapsed_time > 0 else 0.0

    # Save cache if computed
    if cache_enabled and not cache_loaded:
        save_path = save_pose_cache(cache_file, frames_poses, video_path=str(video_path))
        logger.info("Saved pose cache to: %s", save_path)

    # Memory metrics
    mem_info = measure_memory_usage()

    print("\n" + "=" * 60)
    print("POSE ESTIMATION EXECUTION REPORT")
    print("=" * 60)
    print(f"Video Source      : {video_path}")
    print(f"Frames Processed  : {processed_count}")
    print(f"Elapsed Time      : {elapsed_time:.3f} s")
    print(f"Throughput (FPS)  : {inference_fps:.2f} FPS")
    print(f"Cache Status      : {'LOADED FROM CACHE' if cache_loaded else 'COMPUTED & CACHED'}")
    print(f"Peak VRAM Used    : {mem_info['vram_peak_mb']:.2f} MB (GPU)")
    print(f"Current VRAM Used : {mem_info['vram_current_mb']:.2f} MB (GPU)")
    print(f"Process RAM Used  : {mem_info['ram_mb']:.2f} MB (System)")
    print(f"Overlay Video Out : {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
