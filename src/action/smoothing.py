"""Temporal smoothing for per-track action predictions (Majority Vote & EMA with Hysteresis)."""

from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np


class TrackSmoothingState:
    """Maintains temporal prediction history for a single track ID."""

    def __init__(
        self,
        method: str = "ema",
        window_len: int = 7,
        alpha: float = 0.4,
        hysteresis_threshold: float = 0.6,
        classes: Optional[List[str]] = None,
    ) -> None:
        self.method = method.lower()
        self.window_len = int(window_len)
        self.alpha = float(alpha)
        self.hysteresis_threshold = float(hysteresis_threshold)
        self.classes = classes or []

        # Majority vote state
        self.history: Deque[Tuple[str, float]] = deque(maxlen=self.window_len)

        # EMA state
        self.ema_probs: Optional[np.ndarray] = None
        self.current_label: Optional[str] = None
        self.current_conf: float = 0.0

    def update(
        self,
        label: str,
        conf: float,
        probs: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, float]:
        """Update state with new frame/window prediction and return smoothed (label, conf)."""
        if self.method == "majority_vote":
            return self._update_majority_vote(label, conf)
        elif self.method == "ema":
            return self._update_ema(label, conf, probs)
        else:
            raise ValueError(f"Unknown smoothing method '{self.method}'. Expected 'majority_vote' or 'ema'.")

    def _update_majority_vote(self, label: str, conf: float) -> Tuple[str, float]:
        self.history.append((label, conf))

        # Count frequencies
        counts: Dict[str, int] = Counter()
        conf_sum: Dict[str, float] = Counter()

        for l, c in self.history:
            counts[l] += 1
            conf_sum[l] += c

        # Find most frequent label (tie-break by cumulative confidence)
        best_label = max(counts.keys(), key=lambda k: (counts[k], conf_sum[k]))
        smoothed_conf = conf_sum[best_label] / counts[best_label]

        self.current_label = best_label
        self.current_conf = float(smoothed_conf)
        return best_label, float(smoothed_conf)

    def _update_ema(
        self,
        label: str,
        conf: float,
        probs: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, float]:
        if probs is None:
            # Fallback to majority vote if no probability distribution provided
            return self._update_majority_vote(label, conf)

        if not self.classes:
            self.classes = list(probs.keys())

        # Construct vector
        prob_vec = np.array([probs.get(c, 0.0) for c in self.classes], dtype=np.float32)

        if self.ema_probs is None:
            self.ema_probs = prob_vec
            self.current_label = label
            self.current_conf = conf
            return label, conf

        # EMA update: P_t = alpha * P_new + (1 - alpha) * P_{t-1}
        self.ema_probs = self.alpha * prob_vec + (1.0 - self.alpha) * self.ema_probs

        best_idx = int(np.argmax(self.ema_probs))
        candidate_label = self.classes[best_idx]
        candidate_prob = float(self.ema_probs[best_idx])

        # Hysteresis switching logic:
        # Only switch away from current_label if candidate_prob exceeds hysteresis_threshold
        if self.current_label is None:
            self.current_label = candidate_label
            self.current_conf = candidate_prob
        elif candidate_label != self.current_label:
            if candidate_prob >= self.hysteresis_threshold:
                self.current_label = candidate_label
                self.current_conf = candidate_prob
            else:
                # Retain current label with its EMA probability
                curr_idx = self.classes.index(self.current_label) if self.current_label in self.classes else -1
                if curr_idx >= 0:
                    self.current_conf = float(self.ema_probs[curr_idx])
        else:
            self.current_conf = candidate_prob

        return self.current_label, self.current_conf


class ActionSmoother:
    """Manages temporal smoothing across multiple track IDs."""

    def __init__(
        self,
        method: str = "ema",
        window_len: int = 7,
        alpha: float = 0.4,
        hysteresis_threshold: float = 0.6,
        classes: Optional[List[str]] = None,
    ) -> None:
        self.method = method.lower()
        self.window_len = window_len
        self.alpha = alpha
        self.hysteresis_threshold = hysteresis_threshold
        self.classes = classes

        self.track_states: Dict[int, TrackSmoothingState] = {}

    def update(
        self,
        track_id: int,
        label: str,
        conf: float,
        probs: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, float]:
        """Update smoothing state for a track ID and return (smoothed_label, smoothed_conf)."""
        tid = int(track_id)
        if tid not in self.track_states:
            self.track_states[tid] = TrackSmoothingState(
                method=self.method,
                window_len=self.window_len,
                alpha=self.alpha,
                hysteresis_threshold=self.hysteresis_threshold,
                classes=self.classes,
            )

        return self.track_states[tid].update(label, conf, probs)

    def get_current(self, track_id: int) -> Optional[Tuple[str, float]]:
        """Get current smoothed prediction for a track ID."""
        tid = int(track_id)
        if tid in self.track_states and self.track_states[tid].current_label is not None:
            state = self.track_states[tid]
            return state.current_label, state.current_conf
        return None

    def reset(self, track_id: Optional[int] = None) -> None:
        """Reset state for a specific track or all tracks."""
        if track_id is not None:
            self.track_states.pop(int(track_id), None)
        else:
            self.track_states.clear()

