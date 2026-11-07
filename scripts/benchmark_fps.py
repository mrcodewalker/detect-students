"""Benchmark per-stage latency (decode, detect, track) and end-to-end FPS on classroom video."""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tabulate import tabulate

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.detection.yolo_detector import YOLODetector
from src.io.video_reader import FileVideoSource
from src.tracking.base import Tracker
from src.tracking.bot_sort import BoTSORTTracker
from src.tracking.byte_tracker import ByteTracker
from src.tracking.oc_sort import OCSORTTracker
from src.tracking.sort_tracker import SORTTracker
from src.utils.config import load_config


def sync_gpu() -> None:
    """Synchronize CUDA GPU streams if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_system_metadata() -> Dict[str, Any]:
    """Retrieve host GPU, driver, and PyTorch runtime metadata."""
    gpu_name = "None"
    driver_version = "Unknown"

    # Query through PyTorch first
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    # Query nvidia-smi for physical GPU and driver details
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        parts = res.stdout.strip().split(",")
        if len(parts) >= 2:
            driver_version = parts[0].strip()
            if gpu_name == "None":
                gpu_name = parts[1].strip()
    except Exception:
        pass

    return {
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda) if torch.cuda.is_available() else "N/A",
        "cpu_platform": platform.platform(),
        "python_version": sys.version.split()[0],
    }


def create_tracker(tracker_name: str, config_path: Optional[str]) -> Tuple[Tracker, Dict[str, Any]]:
    """Factory helper to instantiate tracker and load its configuration."""
    name = tracker_name.lower()
    if name == "sort":
        cfg_file = config_path or "configs/tracker_sort.yaml"
        cfg = load_config(cfg_file) if Path(cfg_file).is_file() else {}
        return (SORTTracker.from_config(cfg) if cfg else SORTTracker(), cfg)
    elif name == "bytetrack":
        cfg_file = config_path or "configs/tracker_bytetrack.yaml"
        cfg = load_config(cfg_file) if Path(cfg_file).is_file() else {}
        return (ByteTracker.from_config(cfg) if cfg else ByteTracker(), cfg)
    elif name == "botsort":
        cfg_file = config_path or "configs/tracker_botsort.yaml"
        cfg = load_config(cfg_file) if Path(cfg_file).is_file() else {}
        return (BoTSORTTracker.from_config(cfg) if cfg else BoTSORTTracker(), cfg)
    elif name == "ocsort":
        cfg_file = config_path or "configs/tracker_ocsort.yaml"
        cfg = load_config(cfg_file) if Path(cfg_file).is_file() else {}
        return (OCSORTTracker.from_config(cfg) if cfg else OCSORTTracker(), cfg)
    else:
        raise ValueError(f"Unsupported tracker: {tracker_name}")


def benchmark_single_tracker(
    video_path: Union[str, Path],
    tracker_name: str,
    detector_config_path: str = "configs/detector.yaml",
    tracker_config_path: Optional[str] = None,
    warmup_frames: int = 5,
    max_frames: Optional[int] = 50,
) -> Dict[str, Any]:
    """Execute latency benchmark on a video for a single tracker.

    Args:
        video_path: Path to target input video.
        tracker_name: Name of tracker ('sort', 'bytetrack', 'botsort', 'ocsort').
        detector_config_path: Path to detector YAML config.
        tracker_config_path: Optional path to tracker YAML config.
        warmup_frames: Number of initial frames excluded from measurements.
        max_frames: Total limit of frames to process.

    Returns:
        Dictionary containing latency metrics, metadata, and configurations.
    """
    sys_meta = get_system_metadata()

    # Load configurations
    det_cfg = load_config(detector_config_path) if Path(detector_config_path).is_file() else {}
    detector = YOLODetector.from_config(det_cfg)
    tracker, trk_cfg = create_tracker(tracker_name, tracker_config_path)

    source = FileVideoSource(video_path)

    latencies_decode: List[float] = []
    latencies_detect: List[float] = []
    latencies_track: List[float] = []
    latencies_end_to_end: List[float] = []

    frame_count = 0
    measured_count = 0

    try:
        while True:
            if max_frames and frame_count >= max_frames:
                break

            # 1. Stage: Decode
            sync_gpu()
            t_dec_start = time.perf_counter()
            ret = source.read()
            sync_gpu()
            t_dec_end = time.perf_counter()

            if ret is None:
                break

            frame_idx, timestamp_sec, frame = ret
            decode_ms = (t_dec_end - t_dec_start) * 1000.0

            # 2. Stage: Detect
            sync_gpu()
            t_det_start = time.perf_counter()
            detections = detector.detect(frame, frame_idx)
            sync_gpu()
            t_det_end = time.perf_counter()
            detect_ms = (t_det_end - t_det_start) * 1000.0

            # 3. Stage: Track
            sync_gpu()
            t_trk_start = time.perf_counter()
            tracks = tracker.update(detections, frame=frame)
            sync_gpu()
            t_trk_end = time.perf_counter()
            track_ms = (t_trk_end - t_trk_start) * 1000.0

            end_to_end_ms = (t_trk_end - t_dec_start) * 1000.0

            # Only record after warm-up phase
            if frame_count >= warmup_frames:
                latencies_decode.append(decode_ms)
                latencies_detect.append(detect_ms)
                latencies_track.append(track_ms)
                latencies_end_to_end.append(end_to_end_ms)
                measured_count += 1

            frame_count += 1
    finally:
        source.close()

    def calc_stats(lat_list: List[float]) -> Dict[str, float]:
        if not lat_list:
            return {"mean_ms": 0.0, "p95_ms": 0.0, "std_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        arr = np.array(lat_list, dtype=np.float64)
        return {
            "mean_ms": float(np.mean(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
        }

    stats_decode = calc_stats(latencies_decode)
    stats_detect = calc_stats(latencies_detect)
    stats_track = calc_stats(latencies_track)
    stats_e2e = calc_stats(latencies_end_to_end)

    total_measured_time_sec = sum(latencies_end_to_end) / 1000.0 if latencies_end_to_end else 0.0
    end_to_end_fps = float(measured_count / total_measured_time_sec) if total_measured_time_sec > 0 else 0.0

    return {
        "tracker_name": tracker_name,
        "hardware": sys_meta,
        "config_used": {
            "detector": det_cfg,
            "tracker": trk_cfg,
            "input_video": str(video_path),
            "total_frames_processed": frame_count,
            "warmup_frames_excluded": warmup_frames,
            "measured_frames": measured_count,
        },
        "metrics": {
            "decode": stats_decode,
            "detect": stats_detect,
            "track": stats_track,
            "end_to_end": {
                **stats_e2e,
                "fps": end_to_end_fps,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark latency and FPS breakdown across MOT pipeline stages.",
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
        choices=["sort", "bytetrack", "botsort", "ocsort", "all"],
        help="Target tracker or 'all' to benchmark all four trackers.",
    )
    parser.add_argument(
        "--detector-config",
        "-c",
        type=str,
        default="configs/detector.yaml",
        help="Path to detector YAML configuration file.",
    )
    parser.add_argument(
        "--tracker-config",
        type=str,
        default=None,
        help="Optional path to tracker YAML configuration file.",
    )
    parser.add_argument(
        "--warmup-frames",
        "-w",
        type=int,
        default=5,
        help="Number of initial warm-up frames to exclude from latency statistics.",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=50,
        help="Maximum frames to process in benchmark.",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        type=str,
        default=None,
        help="Path to save output benchmark JSON file.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI execution entrypoint."""
    args = parse_args()
    input_video = args.video_path or args.video_flag

    if not input_video:
        print("Error: Input video path required. Example: python scripts/benchmark_fps.py data/raw/video.mp4 -t all", file=sys.stderr)
        return 1

    target_trackers = ["sort", "bytetrack", "botsort", "ocsort"] if args.tracker.lower() == "all" else [args.tracker.lower()]

    print("==================================================")
    print("Classroom MOT Pipeline FPS & Latency Benchmark")
    print("==================================================")
    print(f"Input Video   : {input_video}")
    print(f"Trackers      : {', '.join(t.upper() for t in target_trackers)}")
    print(f"Warm-up Frames: {args.warmup_frames} (excluded from stats)")
    print(f"Max Frames    : {args.max_frames}")
    print("==================================================")

    all_results: Dict[str, Any] = {}
    summary_table_rows: List[Dict[str, Any]] = []

    for trk_name in target_trackers:
        print(f"\n>>> Running benchmark for: {trk_name.upper()}...")
        res = benchmark_single_tracker(
            video_path=input_video,
            tracker_name=trk_name,
            detector_config_path=args.detector_config,
            tracker_config_path=args.tracker_config,
            warmup_frames=args.warmup_frames,
            max_frames=args.max_frames,
        )
        all_results[trk_name] = res

        m = res["metrics"]
        summary_table_rows.append(
            {
                "Tracker": trk_name.upper(),
                "Decode Mean (ms)": f"{m['decode']['mean_ms']:.2f}",
                "Decode P95 (ms)": f"{m['decode']['p95_ms']:.2f}",
                "Detect Mean (ms)": f"{m['detect']['mean_ms']:.2f}",
                "Detect P95 (ms)": f"{m['detect']['p95_ms']:.2f}",
                "Track Mean (ms)": f"{m['track']['mean_ms']:.2f}",
                "Track P95 (ms)": f"{m['track']['p95_ms']:.2f}",
                "E2E Latency Mean (ms)": f"{m['end_to_end']['mean_ms']:.2f}",
                "E2E Latency P95 (ms)": f"{m['end_to_end']['p95_ms']:.2f}",
                "End-to-End FPS": f"{m['end_to_end']['fps']:.2f}",
            }
        )

        # Save individual JSON if specific tracker requested
        if len(target_trackers) == 1:
            out_json = Path(args.output_json or f"results/benchmark_{trk_name}.json").resolve()
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Results saved to: {out_json}")

    # Display comparison table
    print("\n==================================================")
    print("Pipeline Latency Breakdown & Throughput (Summary)")
    print("==================================================")
    print(tabulate(summary_table_rows, headers="keys", tablefmt="github"))
    print("==================================================")

    # Save aggregated JSON if 'all' was run
    if len(target_trackers) > 1:
        out_json = Path(args.output_json or "results/benchmark_all_trackers.json").resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll benchmark results saved to: {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

