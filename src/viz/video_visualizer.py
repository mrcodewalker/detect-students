"""Visualization utilities for rendering bounding boxes, track IDs, actions, and HUD."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.pose.base import COCO_SKELETON_EDGES, PoseDetection


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Generate a distinct, deterministic RGB/BGR color for a track ID."""
    # Use golden ratio hue distribution
    hue = int((track_id * 137.508) % 180)
    hsv_color = np.uint8([[[hue, 200, 240]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])


ACTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "sitting": (0, 200, 0),       # Green
    "standing": (255, 165, 0),    # Blue/Cyan
    "raising_hand": (0, 165, 255), # Orange
    "using_phone": (0, 0, 255),   # Red
}


class PipelineVisualizer:
    """Renders classroom tracking, action classifications, and pose estimations."""

    def __init__(
        self,
        draw_boxes: bool = True,
        draw_ids: bool = True,
        draw_actions: bool = True,
        draw_poses: bool = True,
        draw_hud: bool = True,
        line_thickness: int = 2,
    ) -> None:
        self.draw_boxes = draw_boxes
        self.draw_ids = draw_ids
        self.draw_actions = draw_actions
        self.draw_poses = draw_poses
        self.draw_hud = draw_hud
        self.line_thickness = line_thickness

    def draw_frame(
        self,
        frame: np.ndarray,
        tracks: Sequence[Tuple[int, Tuple[float, float, float, float], float]],
        actions: Optional[Dict[int, Tuple[str, float]]] = None,
        poses: Optional[Sequence[PoseDetection]] = None,
        frame_idx: int = 0,
        total_frames: int = 0,
        timestamp: float = 0.0,
        fps: float = 25.0,
    ) -> np.ndarray:
        """Render annotations onto video frame.

        Args:
            frame: Input BGR image (H, W, 3).
            tracks: Sequence of (track_id, bbox_xyxy, score).
            actions: Optional mapping track_id -> (smoothed_action_label, confidence).
            poses: Optional sequence of PoseDetection instances.
            frame_idx: Current frame index.
            total_frames: Total video frame count.
            timestamp: Elapsed timestamp in seconds.
            fps: Video frames per second.

        Returns:
            Annotated frame array.
        """
        canvas = frame.copy()
        actions = actions or {}

        # 1. Draw 2D Poses if enabled
        if self.draw_poses and poses is not None:
            for pose in poses:
                kpts = pose.keypoints
                for p1, p2 in COCO_SKELETON_EDGES:
                    x1, y1, c1 = kpts[p1]
                    x2, y2, c2 = kpts[p2]
                    if c1 >= 0.3 and c2 >= 0.3:
                        pt1 = (int(round(x1)), int(round(y1)))
                        pt2 = (int(round(x2)), int(round(y2)))
                        cv2.line(canvas, pt1, pt2, (230, 230, 230), 1, cv2.LINE_AA)

                for k_idx in range(17):
                    x, y, c = kpts[k_idx]
                    if c >= 0.3:
                        pt = (int(round(x)), int(round(y)))
                        cv2.circle(canvas, pt, 3, (0, 255, 0), -1, cv2.LINE_AA)

        # 2. Draw Bounding Boxes and Track / Action Labels
        for track_id, bbox, score in tracks:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = get_color_for_id(track_id)

            if self.draw_boxes:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.line_thickness)

            # Build label text
            label_parts = []
            if self.draw_ids:
                label_parts.append(f"ID:{track_id}")

            action_color = color
            if self.draw_actions and track_id in actions:
                act_label, act_conf = actions[track_id]
                label_parts.append(f"{act_label} ({act_conf:.2f})")
                if act_label in ACTION_COLORS:
                    action_color = ACTION_COLORS[act_label]

            if label_parts:
                label_text = " | ".join(label_parts)
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )

                label_y1 = max(0, y1 - text_h - baseline - 6)
                label_y2 = y1
                label_x2 = min(canvas.shape[1], x1 + text_w + 10)

                # Draw filled background pill
                cv2.rectangle(canvas, (x1, label_y1), (label_x2, label_y2), action_color, -1)
                # Draw text in white
                cv2.putText(
                    canvas,
                    label_text,
                    (x1 + 4, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        # 3. Draw Top HUD Banner
        if self.draw_hud:
            h, w = canvas.shape[:2]
            hud_bg = canvas[:40, :].copy()
            cv2.rectangle(canvas, (0, 0), (w, 40), (20, 20, 20), -1)
            cv2.addWeighted(canvas[:40, :], 0.7, hud_bg, 0.3, 0, canvas[:40, :])

            hud_text = (
                f"Time: {timestamp:.2f}s | Frame: {frame_idx + 1}/{total_frames} | "
                f"Students Tracked: {len(tracks)}"
            )
            cv2.putText(
                canvas,
                hud_text,
                (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return canvas

