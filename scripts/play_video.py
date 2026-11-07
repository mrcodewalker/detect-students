"""CLI tool to inspect and play video metadata and frame timestamps."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow running directly from anywhere
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.io.video_reader import FileVideoSource


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect video metadata and display frame timestamps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "video_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the input video file (positional).",
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        dest="video_flag",
        default=None,
        help="Path to the input video file (flag).",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=None,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--gui",
        "-g",
        action="store_true",
        default=True,
        help="Play video in an interactive OpenCV desktop GUI window (Space: Pause, Q: Exit).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_false",
        dest="gui",
        help="Disable desktop GUI and only print timestamps to console.",
    )
    return parser.parse_args()


def main() -> int:
    """Main CLI execution routine."""
    args = parse_args()
    target_path = args.video_path or args.video_flag

    if not target_path:
        print("Error: Video path must be provided. Usage: python scripts/play_video.py <path_to_video>", file=sys.stderr)
        return 1

    try:
        with FileVideoSource(target_path) as source:
            width, height = source.resolution
            print("==================================================")
            print("Video Source Metadata")
            print("==================================================")
            print(f"File Path    : {source.path}")
            print(f"FPS          : {source.fps:.2f}")
            print(f"Resolution   : {width} x {height}")
            print(f"Frame Count  : {source.frame_count if source.frame_count >= 0 else 'Unknown'}")
            print("==================================================")
            print(f"Reading up to {args.max_frames} frame timestamps:")
            print("--------------------------------------------------")

            import cv2
            count = 0
            delay_ms = max(1, int(1000.0 / source.fps)) if source.fps > 0 else 33

            if args.gui:
                cv2.namedWindow("Video Player (Space: Pause/Play, Q: Exit)", cv2.WINDOW_NORMAL)

            try:
                for frame_idx, timestamp_sec, frame in source:
                    count += 1
                    if args.gui:
                        cv2.imshow("Video Player (Space: Pause/Play, Q: Exit)", frame)
                        key = cv2.waitKey(delay_ms) & 0xFF
                        if key == ord("q") or key == 27:  # Q or ESC
                            print(f"\nPlayback interrupted by user at frame {frame_idx}.")
                            break
                        elif key == ord(" "):  # Space to pause
                            print("\nPlayback paused. Press any key to continue...")
                            cv2.waitKey(0)
                    else:
                        print(f"Frame {frame_idx:5d} | Timestamp: {timestamp_sec:8.3f}s")

                    if args.max_frames and count >= args.max_frames:
                        break
            finally:
                if args.gui:
                    cv2.destroyAllWindows()

            print("--------------------------------------------------")
            print(f"Processed {count} frame(s).")
            print("==================================================")
            return 0
    except FileNotFoundError as e:
        print(f"File Error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"Format/Decode Error: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"Unexpected Error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

