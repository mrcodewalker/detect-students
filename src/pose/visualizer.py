"""Visualization utilities for rendering 17 COCO pose keypoints and skeleton limbs."""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.pose.base import COCO_SKELETON_EDGES, PoseDetection


# Palette definitions for distinct visual appearance
# Color tuple in BGR: (B, G, R)
COLOR_LEFT_LIMB = (255, 128, 0)     # Light Blue / Orange in BGR: Blue=255, Green=128, Red=0
COLOR_RIGHT_LIMB = (0, 215, 255)    # Gold / Amber: Blue=0, Green=215, Red=255
COLOR_CENTER_LIMB = (180, 105, 255) # Pink / Magenta
COLOR_KEYPOINT = (0, 255, 0)        # Green
COLOR_BBOX = (255, 0, 0)            # Blue


def get_edge_color(p1: int, p2: int) -> Tuple[int, int, int]:
    """Assign color based on whether the limb is on the left, right, or center body."""
    # COCO left indices: 1, 3, 5, 7, 9, 11, 13, 15
    # COCO right indices: 2, 4, 6, 8, 10, 12, 14, 16
    is_left = (p1 % 2 == 1) and (p2 % 2 == 1)
    is_right = (p1 % 2 == 0 and p1 > 0) and (p2 % 2 == 0 and p2 > 0)

    if is_left:
        return COLOR_LEFT_LIMB
    elif is_right:
        return COLOR_RIGHT_LIMB
    else:
        return COLOR_CENTER_LIMB


def draw_pose_on_frame(
    frame: np.ndarray,
    poses: Sequence[PoseDetection],
    kpt_conf_thresh: float = 0.3,
    draw_box: bool = True,
    draw_labels: bool = True,
    line_thickness: int = 2,
    circle_radius: int = 4,
) -> np.ndarray:
    """Render 2D pose skeletons and bounding boxes onto an image.

    Args:
        frame: BGR image frame (H, W, 3).
        poses: Sequence of PoseDetection instances.
        kpt_conf_thresh: Minimum keypoint confidence required to draw.
        draw_box: Whether to draw detection bounding box.
        draw_labels: Whether to display score text.
        line_thickness: Thickness of skeleton limb lines.
        circle_radius: Radius of keypoint circles.

    Returns:
        New image array with rendered poses.
    """
    canvas = frame.copy()

    for idx, pose in enumerate(poses):
        kpts = pose.keypoints  # (17, 3)

        # 1. Draw bounding box if requested
        if draw_box:
            x1, y1, x2, y2 = [int(v) for v in pose.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BBOX, max(1, line_thickness))
            if draw_labels:
                label = f"Person {idx + 1} ({pose.score:.2f})"
                cv2.putText(
                    canvas,
                    label,
                    (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 2. Draw skeleton bones / limb connections
        for p1, p2 in COCO_SKELETON_EDGES:
            x1, y1, c1 = kpts[p1]
            x2, y2, c2 = kpts[p2]

            if c1 >= kpt_conf_thresh and c2 >= kpt_conf_thresh:
                pt1 = (int(round(x1)), int(round(y1)))
                pt2 = (int(round(x2)), int(round(y2)))
                color = get_edge_color(p1, p2)
                cv2.line(canvas, pt1, pt2, color, line_thickness, cv2.LINE_AA)

        # 3. Draw keypoint joint circles
        for k_idx in range(17):
            x, y, conf = kpts[k_idx]
            if conf >= kpt_conf_thresh:
                pt = (int(round(x)), int(round(y)))
                cv2.circle(canvas, pt, circle_radius, COLOR_KEYPOINT, -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, circle_radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas

