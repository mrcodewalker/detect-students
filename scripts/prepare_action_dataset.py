"""Extract, auto-annotate, augment, and split spatio-temporal action windows from classroom videos."""

import argparse
from collections import defaultdict
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
import sys
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.action.track_buffer import TrackActionBuffer
from src.detection.yolo_detector import YOLODetector
from src.io.video_reader import FileVideoSource
from src.pose.cache import get_default_cache_path, has_pose_cache, load_pose_cache, save_pose_cache
from src.pose.matching import match_pose_to_detections
from src.pose.yolo_pose import YOLOPoseEstimator
from src.tracking.byte_tracker import ByteTracker
from src.utils.split import split_by_video

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("prepare_dataset")


def classify_window_robust(kpts: Optional[np.ndarray], bboxes: np.ndarray) -> Tuple[str, float]:
    """Anatomically accurate classroom action estimation from keypoints and bboxes."""
    if kpts is None or len(bboxes) == 0:
        return "sitting", 0.90

    t_steps = len(bboxes)
    hand_raised_scores = []
    standing_scores = []
    phone_scores = []

    for t in range(t_steps):
        box = bboxes[t]
        bw = max(1.0, box[2] - box[0])
        bh = max(1.0, box[3] - box[1])

        nose = kpts[t, 0]
        l_sh = kpts[t, 5]
        r_sh = kpts[t, 6]
        l_el = kpts[t, 7]
        r_el = kpts[t, 8]
        l_wr = kpts[t, 9]
        r_wr = kpts[t, 10]
        l_hip = kpts[t, 11]
        r_hip = kpts[t, 12]
        l_knee = kpts[t, 13]
        r_knee = kpts[t, 14]
        l_ankle = kpts[t, 15]
        r_ankle = kpts[t, 16]

        shldr_conf = max(l_sh[2], r_sh[2])
        shldr_y = min(l_sh[1], r_sh[1]) if (l_sh[2] > 0.15 and r_sh[2] > 0.15) else (
            l_sh[1] if l_sh[2] > 0.15 else r_sh[1]
        )

        # 1. Raising Hand: wrist or elbow clearly raised above shoulder/nose line
        l_up = (l_wr[2] > 0.25) and (l_wr[1] < shldr_y - 0.05 * bh)
        r_up = (r_wr[2] > 0.25) and (r_wr[1] < shldr_y - 0.05 * bh)
        if nose[2] > 0.25:
            l_up = l_up or ((l_wr[2] > 0.25) and (l_wr[1] < nose[1]))
            r_up = r_up or ((r_wr[2] > 0.25) and (r_wr[1] < nose[1]))
        if (l_el[2] > 0.35 and l_el[1] < shldr_y) or (r_el[2] > 0.35 and r_el[1] < shldr_y):
            if l_wr[2] > 0.15 or r_wr[2] > 0.15:
                l_up = l_up or (l_el[1] < shldr_y)
                r_up = r_up or (r_el[1] < shldr_y)
        hand_raised_scores.append(1.0 if (l_up or r_up) else 0.0)

        # 2. Standing: Requires verified lower-body extension
        hip_conf = max(l_hip[2], r_hip[2])
        knee_conf = max(l_knee[2], r_knee[2])
        ankle_conf = max(l_ankle[2], r_ankle[2])
        is_standing = False
        if hip_conf > 0.3 and knee_conf > 0.3:
            hip_y = (l_hip[1] + r_hip[1]) / 2
            knee_y = (l_knee[1] + r_knee[1]) / 2
            if knee_y - hip_y > 0.25 * bh:
                if ankle_conf > 0.25 or (bh / bw >= 2.0):
                    is_standing = True
        standing_scores.append(1.0 if is_standing else 0.0)

        # 3. Using phone: looking down at hands held close together near mid-torso/lap
        is_phone = False
        if l_wr[2] > 0.25 and r_wr[2] > 0.25:
            wrist_dx = abs(l_wr[0] - r_wr[0])
            avg_wrist_y = (l_wr[1] + r_wr[1]) / 2
            if shldr_conf > 0.2 and (shldr_y < avg_wrist_y < shldr_y + 0.45 * bh):
                if wrist_dx < 0.18 * bw:
                    if nose[2] > 0.2 and (nose[1] >= shldr_y - 0.05 * bh):
                        is_phone = True
        phone_scores.append(1.0 if is_phone else 0.0)

    f_raise = float(np.mean(hand_raised_scores))
    f_stand = float(np.mean(standing_scores))
    f_phone = float(np.mean(phone_scores))

    if f_raise >= 0.25:
        return "raising_hand", 0.85 + 0.15 * f_raise
    elif f_stand >= 0.4:
        return "standing", 0.80 + 0.15 * f_stand
    elif f_phone >= 0.4:
        return "using_phone", 0.75 + 0.20 * f_phone
    else:
        return "sitting", 0.90


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare labeled action windows dataset from videos.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Folder containing raw classroom videos.")
    parser.add_argument("--output-dir", type=str, default="data/processed/action_windows", help="Destination folder for window .npz files.")
    parser.add_argument("--window-size", type=int, default=16, help="Window size N.")
    parser.add_argument("--stride", type=int, default=4, help="Sliding step S.")
    parser.add_argument("--max-frames-per-video", type=int, default=300, help="Max frames to process per video.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)

    classes = ["sitting", "standing", "raising_hand", "using_phone"]
    for c in classes:
        (out_dir / c).mkdir(parents=True, exist_ok=True)

    videos = list(raw_dir.glob("*.mp4"))
    if not videos:
        logger.error("No MP4 videos found in %s", raw_dir)
        return

    logger.info("Found %d video(s) for window extraction: %s", len(videos), [v.name for v in videos])

    detector = YOLODetector("yolov8n.pt", conf=0.35, device="cpu")
    pose_estimator = YOLOPoseEstimator("yolov8n-pose.pt", conf=0.35, device="cpu", mode="whole_frame")

    total_extracted: Dict[str, int] = defaultdict(int)

    for vid in videos:
        logger.info("Processing video: %s", vid.name)
        source = FileVideoSource(str(vid))
        tracker = ByteTracker.from_config("configs/tracker_bytetrack.yaml")
        buffer = TrackActionBuffer(window_size=args.window_size, stride=args.stride, max_missing_frames=30)

        # Pose cache check
        pose_cache_file = get_default_cache_path(vid, "results/cache/pose")
        cached_poses = load_pose_cache(pose_cache_file) if has_pose_cache(pose_cache_file) else None
        poses_to_save: Dict[int, Any] = {}

        count = 0
        try:
            for f_idx, ts, frame in source:
                count += 1
                if count > args.max_frames_per_video:
                    break

                dets = detector.detect(frame, frame_index=f_idx)

                # Pose
                if cached_poses and f_idx in cached_poses:
                    poses = cached_poses[f_idx]
                else:
                    poses = pose_estimator.estimate(frame, frame_index=f_idx)
                    poses_to_save[f_idx] = poses

                tracks = tracker.update(dets, frame=frame)

                track_kpts: Dict[int, Any] = {}
                if tracks and poses:
                    matched = match_pose_to_detections([t[1] for t in tracks], poses, iou_threshold=0.3)
                    for (tid, _, _), p in zip(tracks, matched):
                        if p is not None:
                            track_kpts[tid] = p.keypoints

                ready_windows = buffer.step(f_idx, tracks, frame=frame, keypoints_map=track_kpts)

                for win in ready_windows:
                    if win.keypoints is None:
                        continue

                    label, conf = classify_window_robust(win.keypoints, win.bboxes)
                    if conf < 0.65:
                        continue

                    # Save window
                    sample_name = f"{vid.stem}_t{win.track_id}_f{win.start_frame}_{win.end_frame}.npz"
                    sample_path = out_dir / label / sample_name

                    np.savez_compressed(
                        sample_path,
                        keypoints=win.keypoints,
                        crops=win.crops if win.crops is not None else np.zeros((1, 1, 1, 3), dtype=np.uint8),
                        bboxes=win.bboxes,
                        track_id=win.track_id,
                        video_id=vid.stem,
                        confidence=conf,
                    )
                    total_extracted[label] += 1

                    # Add horizontal-flip data augmentation for robustness
                    kpts_flipped = win.keypoints.copy()
                    # Flip X relative to bbox width
                    for step_idx in range(len(kpts_flipped)):
                        bx1, _, bx2, _ = win.bboxes[step_idx]
                        kpts_flipped[step_idx, :, 0] = bx2 - (kpts_flipped[step_idx, :, 0] - bx1)
                        # Swap left/right COCO indices
                        swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
                        for p1, p2 in swap_pairs:
                            kpts_flipped[step_idx, [p1, p2]] = kpts_flipped[step_idx, [p2, p1]]

                    flip_name = f"{vid.stem}_t{win.track_id}_f{win.start_frame}_{win.end_frame}_flip.npz"
                    crops_flipped = np.flip(win.crops, axis=2).copy() if win.crops is not None else np.zeros((1, 1, 1, 3), dtype=np.uint8)
                    np.savez_compressed(
                        out_dir / label / flip_name,
                        keypoints=kpts_flipped,
                        crops=crops_flipped,
                        bboxes=win.bboxes,
                        track_id=win.track_id,
                        video_id=vid.stem,
                        confidence=conf,
                    )
                    total_extracted[label] += 1

        finally:
            source.close()
            if poses_to_save and not cached_poses:
                save_pose_cache(pose_cache_file, poses_to_save, video_path=str(vid))

    print("\n" + "=" * 60)
    print("ACTION DATASET PREPARATION SUMMARY")
    print("=" * 60)
    for c in classes:
        print(f"Class '{c:<14}': {total_extracted[c]:>4} samples extracted")
    print(f"Total Samples Generated: {sum(total_extracted.values())}")
    print(f"Saved into Directory   : {out_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
