"""Evaluation metrics and confusion matrix generation for action recognition.

Supports two evaluation modes:
1. Mode 'gt_tracks':
   Evaluates action classification in isolation against clean ground-truth track IDs/windows.
   Input needed:
   - y_true: Sequence of ground-truth action class names.
   - y_pred: Sequence of predicted action class names.

2. Mode 'pred_tracks':
   Evaluates end-to-end detection + tracking + action classification.
   Matches predicted track bounding boxes to ground-truth boxes per frame using IoU matching.
   Input needed:
   - gt_observations: Sequence of dicts with keys ('frame_idx', 'bbox', 'track_id', 'action').
   - pred_observations: Sequence of dicts with keys ('frame_idx', 'bbox', 'track_id', 'action').
   - iou_threshold: Minimum bounding box IoU (e.g. 0.5) to consider a spatial match.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.pose.matching import compute_iou_matrix


@dataclass
class ActionEvaluationMetrics:
    """Action classification evaluation summary.

    Attributes:
        classes: Ordered list of action class names.
        precision_per_class: Mapping class -> precision.
        recall_per_class: Mapping class -> recall.
        f1_per_class: Mapping class -> F1 score.
        support_per_class: Mapping class -> number of ground truth instances.
        macro_f1: Macro-averaged F1 score.
        accuracy: Overall classification accuracy.
        confusion_matrix: (C, C) matrix where row = ground truth, col = prediction.
    """

    classes: List[str]
    precision_per_class: Dict[str, float]
    recall_per_class: Dict[str, float]
    f1_per_class: Dict[str, float]
    support_per_class: Dict[str, int]
    macro_f1: float
    accuracy: float
    confusion_matrix: np.ndarray


def compute_action_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    classes: Sequence[str],
) -> ActionEvaluationMetrics:
    """Compute per-class precision, recall, F1, accuracy, and confusion matrix.

    Args:
        y_true: Ground truth action labels.
        y_pred: Predicted action labels.
        classes: Target class names.

    Returns:
        ActionEvaluationMetrics dataclass instance.
    """
    class_list = list(classes)
    c_map = {c: i for i, c in enumerate(class_list)}
    num_classes = len(class_list)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for yt, yp in zip(y_true, y_pred):
        if yt in c_map and yp in c_map:
            cm[c_map[yt], c_map[yp]] += 1

    precision_dict: Dict[str, float] = {}
    recall_dict: Dict[str, float] = {}
    f1_dict: Dict[str, float] = {}
    support_dict: Dict[str, int] = {}

    total_tp = 0
    total_samples = int(np.sum(cm))

    for idx, c in enumerate(class_list):
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp
        support = int(np.sum(cm[idx, :]))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precision_dict[c] = prec
        recall_dict[c] = rec
        f1_dict[c] = f1
        support_dict[c] = support
        total_tp += tp

    macro_f1 = float(np.mean(list(f1_dict.values()))) if f1_dict else 0.0
    accuracy = float(total_tp / total_samples) if total_samples > 0 else 0.0

    return ActionEvaluationMetrics(
        classes=class_list,
        precision_per_class=precision_dict,
        recall_per_class=recall_dict,
        f1_per_class=f1_dict,
        support_per_class=support_dict,
        macro_f1=macro_f1,
        accuracy=accuracy,
        confusion_matrix=cm,
    )


def evaluate_gt_tracks(
    gt_labels: Sequence[str],
    pred_labels: Sequence[str],
    classes: Sequence[str],
) -> ActionEvaluationMetrics:
    """Mode 1: Evaluate action predictions on Ground-Truth tracks in isolation.

    Input needed:
        gt_labels: Ground truth action labels per track/window.
        pred_labels: Predicted action labels for the same tracks/windows.
        classes: Ordered list of action classes.
    """
    if len(gt_labels) != len(pred_labels):
        raise ValueError(f"Lengths must match: len(gt_labels)={len(gt_labels)}, len(pred_labels)={len(pred_labels)}")

    return compute_action_metrics(gt_labels, pred_labels, classes)


def evaluate_pred_tracks(
    gt_observations: Sequence[Dict[str, Any]],
    pred_observations: Sequence[Dict[str, Any]],
    classes: Sequence[str],
    iou_threshold: float = 0.5,
) -> ActionEvaluationMetrics:
    """Mode 2: Evaluate end-to-end detection, tracking, and action classification.

    Input needed:
        gt_observations: List of dicts, each containing:
            {'frame_idx': int, 'bbox': (x1, y1, x2, y2), 'track_id': int, 'action': str}
        pred_observations: List of dicts, each containing:
            {'frame_idx': int, 'bbox': (x1, y1, x2, y2), 'track_id': int, 'action': str}
        classes: Ordered list of action classes.
        iou_threshold: Spatial IoU threshold for matching predicted box to GT box.
    """
    # Group observations by frame_idx
    from collections import defaultdict
    gt_by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    pred_by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for g in gt_observations:
        gt_by_frame[int(g["frame_idx"])].append(g)
    for p in pred_observations:
        pred_by_frame[int(p["frame_idx"])].append(p)

    all_frames = set(gt_by_frame.keys()).union(set(pred_by_frame.keys()))
    matched_gt_labels: List[str] = []
    matched_pred_labels: List[str] = []

    for f_idx in all_frames:
        gts = gt_by_frame.get(f_idx, [])
        preds = pred_by_frame.get(f_idx, [])

        if not gts or not preds:
            continue

        gt_boxes = [g["bbox"] for g in gts]
        pred_boxes = [p["bbox"] for p in preds]

        iou_matrix = compute_iou_matrix(gt_boxes, pred_boxes)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                matched_gt_labels.append(gts[r]["action"])
                matched_pred_labels.append(preds[c]["action"])

    return compute_action_metrics(matched_gt_labels, matched_pred_labels, classes)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: Sequence[str],
    output_path: Union[str, Path],
    title: str = "Action Recognition Confusion Matrix",
    normalize: bool = True,
) -> Path:
    """Render and save a confusion matrix plot to disk.

    Args:
        cm: Confusion matrix array of shape (C, C).
        classes: Ordered class names.
        output_path: File path to save output image (e.g. .png).
        title: Title of plot.
        normalize: If True, display percentages normalized by true label count.

    Returns:
        Resolved Path to saved image file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cm_display = cm.astype(np.float32)
    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm_display, row_sums, where=row_sums > 0, out=np.zeros_like(cm_display))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    cax = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14, pad=15)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)

    # Annotate cells
    thresh = cm_display.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            val_raw = int(cm[i, j])
            if normalize:
                pct = cm_display[i, j] * 100.0
                cell_text = f"{val_raw}\n({pct:.1f}%)"
            else:
                cell_text = f"{val_raw}"

            ax.text(
                j,
                i,
                cell_text,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=9,
            )

    ax.set_ylabel("Ground Truth Class", fontsize=12, labelpad=10)
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.tight_layout()

    fig.savefig(out)
    plt.close(fig)
    return out

