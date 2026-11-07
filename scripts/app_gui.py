"""Modern Desktop GUI Application for Classroom Student Tracking & Action Recognition.

Built with CustomTkinter, OpenCV, and PyTorch.
Features:
- Video file selector with auto-discovery in data/raw/ and custom file dialog.
- Configurable tracker (ByteTrack, SORT, BoT-SORT, OC-SORT) and Pose estimation toggle.
- Real-time video canvas playback with bounding boxes, IDs, 17 keypoint skeletons, and action pills.
- Live analytics dashboard (active students, current action breakdown, FPS).
- Export shortcuts for annotated video and CSV summaries.
"""

import csv
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.action.classifier import create_action_classifier, DEFAULT_ACTION_CLASSES
from src.action.smoothing import ActionSmoother
from src.action.track_buffer import ActionWindow, TrackActionBuffer
from src.detection.yolo_detector import YOLODetector
from src.io.video_reader import FileVideoSource
from src.pose.base import PoseDetection
from src.pose.cache import get_default_cache_path, has_pose_cache, load_pose_cache, save_pose_cache
from src.pose.matching import match_pose_to_detections
from src.pose.yolo_pose import YOLOPoseEstimator
from src.tracking.bot_sort import BoTSORTTracker
from src.tracking.byte_tracker import ByteTracker
from src.tracking.oc_sort import OCSORTTracker
from src.tracking.sort_tracker import SORTTracker
from src.utils.config import load_config
from src.viz.video_visualizer import PipelineVisualizer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("app_gui")

# Set theme and appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ClassroomVisionApp(ctk.CTk):
    """Modern Desktop GUI for Student Detection, Tracking, and Action Recognition."""

    def __init__(self) -> None:
        super().__init__()

        self.title("Classroom AI Vision - Student Tracking & Action Recognition")
        self.geometry("1360x860")
        self.minsize(1100, 750)

        # State variables
        self.is_running = False
        self.is_paused = False
        self.worker_thread: Optional[threading.Thread] = None
        self.frame_queue: Queue = Queue(maxsize=4)

        self.current_video_path: Optional[str] = None
        self.last_output_video: Optional[str] = None
        self.last_output_csv: Optional[str] = None
        self.last_summary_csv: Optional[str] = None

        self._build_ui()
        self._refresh_video_list()
        self._check_frame_queue()

    def _build_ui(self) -> None:
        """Construct responsive layout with Left Sidebar and Right Main Canvas."""
        # Main grid layout (2 columns)
        self.grid_columnconfigure(0, weight=0, minsize=360)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -------------------------------------------------------------
        # Left Sidebar: Controls & Settings
        # -------------------------------------------------------------
        sidebar = ctk.CTkScrollableFrame(self, width=350, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Title / Brand
        brand_label = ctk.CTkLabel(
            sidebar,
            text="🎓 Classroom AI Vision",
            font=ctk.CTkFont(size=20, weight="bold"),
            anchor="w",
        )
        brand_label.pack(fill="x", padx=16, pady=(16, 4))

        subtitle_label = ctk.CTkLabel(
            sidebar,
            text="Real-time Tracking & Behavior Monitoring",
            font=ctk.CTkFont(size=12),
            text_color="gray70",
            anchor="w",
        )
        subtitle_label.pack(fill="x", padx=16, pady=(0, 16))

        # --- Section 1: Video Selection ---
        sec1_label = ctk.CTkLabel(sidebar, text="1. VIDEO INPUT SOURCE", font=ctk.CTkFont(size=12, weight="bold"), text_color="#3b8ed0", anchor="w")
        sec1_label.pack(fill="x", padx=16, pady=(8, 4))

        self.video_dropdown = ctk.CTkOptionMenu(
            sidebar,
            values=["Scanning data/raw/..."],
            command=self._on_video_selected,
            height=32,
        )
        self.video_dropdown.pack(fill="x", padx=16, pady=4)

        btn_browse = ctk.CTkButton(
            sidebar,
            text="📂 Browse Custom Video...",
            command=self._on_browse_file,
            fg_color="#2b2b2b",
            hover_color="#3a3a3a",
            height=30,
        )
        btn_browse.pack(fill="x", padx=16, pady=4)

        self.lbl_video_info = ctk.CTkLabel(
            sidebar,
            text="Resolution: -- | Frames: -- | FPS: --",
            font=ctk.CTkFont(size=11),
            text_color="gray65",
            anchor="w",
        )
        self.lbl_video_info.pack(fill="x", padx=16, pady=(2, 12))

        # --- Section 2: Pipeline Parameters ---
        sec2_label = ctk.CTkLabel(sidebar, text="2. TRACKER & POSE MODEL", font=ctk.CTkFont(size=12, weight="bold"), text_color="#3b8ed0", anchor="w")
        sec2_label.pack(fill="x", padx=16, pady=(8, 4))

        # Tracker algorithm
        tracker_box = ctk.CTkFrame(sidebar, fg_color="transparent")
        tracker_box.pack(fill="x", padx=16, pady=2)
        ctk.CTkLabel(tracker_box, text="Tracker:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.tracker_var = ctk.StringVar(value="bytetrack")
        self.tracker_menu = ctk.CTkOptionMenu(
            tracker_box,
            variable=self.tracker_var,
            values=["bytetrack", "sort", "botsort", "ocsort"],
            width=140,
            height=28,
        )
        self.tracker_menu.pack(side="right")

        # Pose Estimation toggle
        self.pose_var = ctk.BooleanVar(value=True)
        self.chk_pose = ctk.CTkCheckBox(
            sidebar,
            text="Enable 17-Keypoint Pose Estimation",
            variable=self.pose_var,
            font=ctk.CTkFont(size=12),
        )
        self.chk_pose.pack(fill="x", padx=16, pady=6)

        # Detection Confidence Slider
        self.lbl_conf = ctk.CTkLabel(
            sidebar,
            text="Detection Confidence: 0.35",
            font=ctk.CTkFont(size=11),
            text_color="gray70",
            anchor="w",
        )
        self.lbl_conf.pack(fill="x", padx=16, pady=(6, 0))
        self.conf_slider = ctk.CTkSlider(
            sidebar,
            from_=0.15,
            to=0.85,
            number_of_steps=14,
            command=self._on_conf_slider_change,
        )
        self.conf_slider.set(0.35)
        self.conf_slider.pack(fill="x", padx=16, pady=4)

        # --- Section 3: Action Recognition ---
        sec3_label = ctk.CTkLabel(sidebar, text="3. ACTION CLASSIFIER & SMOOTHING", font=ctk.CTkFont(size=12, weight="bold"), text_color="#3b8ed0", anchor="w")
        sec3_label.pack(fill="x", padx=16, pady=(12, 4))

        action_box = ctk.CTkFrame(sidebar, fg_color="transparent")
        action_box.pack(fill="x", padx=16, pady=2)
        ctk.CTkLabel(action_box, text="Backend:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.backend_var = ctk.StringVar(value="keypoint")
        self.backend_menu = ctk.CTkOptionMenu(
            action_box,
            variable=self.backend_var,
            values=["keypoint", "crop", "pretrained_s3d"],
            width=140,
            height=28,
        )
        self.backend_menu.pack(side="right")

        smooth_box = ctk.CTkFrame(sidebar, fg_color="transparent")
        smooth_box.pack(fill="x", padx=16, pady=2)
        ctk.CTkLabel(smooth_box, text="Smoothing:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.smooth_var = ctk.StringVar(value="ema")
        self.smooth_menu = ctk.CTkOptionMenu(
            smooth_box,
            variable=self.smooth_var,
            values=["ema", "majority_vote"],
            width=140,
            height=28,
        )
        self.smooth_menu.pack(side="right")

        # --- Section 4: Execution Controls ---
        sec4_label = ctk.CTkLabel(sidebar, text="4. EXECUTION CONTROLS", font=ctk.CTkFont(size=12, weight="bold"), text_color="#3b8ed0", anchor="w")
        sec4_label.pack(fill="x", padx=16, pady=(16, 4))

        self.btn_run = ctk.CTkButton(
            sidebar,
            text="▶ START TRACKING PIPELINE",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#28a745",
            hover_color="#218838",
            height=38,
            command=self._on_start_stop,
        )
        self.btn_run.pack(fill="x", padx=16, pady=6)

        btn_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=2)
        btn_row.grid_columnconfigure((0, 1), weight=1)

        self.btn_pause = ctk.CTkButton(
            btn_row,
            text="⏸ Pause",
            state="disabled",
            fg_color="#e0a800",
            hover_color="#c69500",
            height=32,
            command=self._on_pause_toggle,
        )
        self.btn_pause.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        self.btn_stop = ctk.CTkButton(
            btn_row,
            text="⏹ Stop",
            state="disabled",
            fg_color="#dc3545",
            hover_color="#c82333",
            height=32,
            command=self._stop_processing,
        )
        self.btn_stop.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(sidebar)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", padx=16, pady=(12, 4))

        self.lbl_progress = ctk.CTkLabel(sidebar, text="Status: Ready", font=ctk.CTkFont(size=11), text_color="gray70", anchor="w")
        self.lbl_progress.pack(fill="x", padx=16, pady=(0, 12))

        # --- Section 5: Exports ---
        sec5_label = ctk.CTkLabel(sidebar, text="5. EXPORT & RESULTS", font=ctk.CTkFont(size=12, weight="bold"), text_color="#3b8ed0", anchor="w")
        sec5_label.pack(fill="x", padx=16, pady=(8, 4))

        self.btn_open_video = ctk.CTkButton(
            sidebar,
            text="🎬 Open Annotated Video (.mp4)",
            command=self._open_result_video,
            fg_color="#1f538d",
            hover_color="#14375e",
            height=30,
        )
        self.btn_open_video.pack(fill="x", padx=16, pady=3)

        self.btn_open_csv = ctk.CTkButton(
            sidebar,
            text="📊 Open Action Log CSV",
            command=self._open_result_csv,
            fg_color="#2b2b2b",
            hover_color="#3a3a3a",
            height=30,
        )
        self.btn_open_csv.pack(fill="x", padx=16, pady=3)

        self.btn_open_summary = ctk.CTkButton(
            sidebar,
            text="📋 Open Duration Summary CSV",
            command=self._open_result_summary,
            fg_color="#2b2b2b",
            hover_color="#3a3a3a",
            height=30,
        )
        self.btn_open_summary.pack(fill="x", padx=16, pady=3)

        # -------------------------------------------------------------
        # Right Main Panel: Video Canvas & Real-Time Stats
        # -------------------------------------------------------------
        main_panel = ctk.CTkFrame(self, corner_radius=0, fg_color="#181818")
        main_panel.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        main_panel.grid_rowconfigure(0, weight=1)
        main_panel.grid_rowconfigure(1, weight=0)
        main_panel.grid_columnconfigure(0, weight=1)

        # Video Canvas Container
        canvas_card = ctk.CTkFrame(main_panel, fg_color="#101010", corner_radius=8)
        canvas_card.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 8))
        canvas_card.grid_rowconfigure(0, weight=1)
        canvas_card.grid_columnconfigure(0, weight=1)

        self.video_canvas = ctk.CTkLabel(
            canvas_card,
            text="🎬 Select a video from the sidebar and click\n[ START TRACKING PIPELINE ] to begin.",
            font=ctk.CTkFont(size=16),
            text_color="gray60",
        )
        self.video_canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Bottom Analytics Dashboard Card
        stats_card = ctk.CTkFrame(main_panel, fg_color="#202020", corner_radius=8)
        stats_card.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        stats_card.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        self.kpi_students = self._create_kpi_card(stats_card, 0, "Students Active", "0")
        self.kpi_sitting = self._create_kpi_card(stats_card, 1, "Sitting", "0", color="#28a745")
        self.kpi_raising = self._create_kpi_card(stats_card, 2, "Raising Hand", "0", color="#fd7e14")
        self.kpi_standing = self._create_kpi_card(stats_card, 3, "Standing", "0", color="#17a2b8")
        self.kpi_fps = self._create_kpi_card(stats_card, 4, "Processing Speed", "0.0 FPS")

    def _create_kpi_card(self, parent: ctk.CTkFrame, col: int, title: str, init_val: str, color: str = "#ffffff") -> ctk.CTkLabel:
        box = ctk.CTkFrame(parent, fg_color="#292929", corner_radius=6)
        box.grid(row=0, column=col, padx=6, pady=8, sticky="ew")

        ctk.CTkLabel(box, text=title, font=ctk.CTkFont(size=11), text_color="gray70").pack(pady=(4, 0))
        val_lbl = ctk.CTkLabel(box, text=init_val, font=ctk.CTkFont(size=18, weight="bold"), text_color=color)
        val_lbl.pack(pady=(0, 4))
        return val_lbl

    def _refresh_video_list(self) -> None:
        """Scan data/raw/ directory for video files."""
        raw_dir = Path("data/raw")
        videos: List[str] = []
        if raw_dir.is_dir():
            for ext in [".mp4", ".avi", ".mkv", ".mov"]:
                for vf in raw_dir.glob(f"*{ext}"):
                    videos.append(vf.name)

        if videos:
            self.video_dropdown.configure(values=videos)
            # Prefer real classroom video as default
            fav = "classroom_real_test.mp4" if "classroom_real_test.mp4" in videos else videos[0]
            self.video_dropdown.set(fav)
            self._on_video_selected(fav)
        else:
            self.video_dropdown.configure(values=["No videos in data/raw/"])

    def _on_video_selected(self, choice: str) -> None:
        v_path = Path("data/raw") / choice
        if v_path.is_file():
            self.current_video_path = str(v_path)
            self._update_video_info(self.current_video_path)

    def _on_browse_file(self) -> None:
        from tkinter import filedialog
        chosen = filedialog.askopenfilename(
            title="Select Classroom Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")],
        )
        if chosen and Path(chosen).is_file():
            self.current_video_path = chosen
            self.video_dropdown.set(Path(chosen).name)
            self._update_video_info(chosen)

    def _update_video_info(self, video_path: str) -> None:
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.lbl_video_info.configure(text=f"Res: {w}x{h} | Frames: {frames} | FPS: {fps:.1f}")
                cap.release()
        except Exception:
            pass

    def _on_conf_slider_change(self, val: float) -> None:
        if hasattr(self, "lbl_conf"):
            self.lbl_conf.configure(text=f"Detection Confidence: {val:.2f}")

    def _on_start_stop(self) -> None:
        if not self.is_running:
            self._start_processing()
        else:
            self._stop_processing()

    def _on_pause_toggle(self) -> None:
        if self.is_running:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.btn_pause.configure(text="▶ Resume", fg_color="#28a745")
                self.lbl_progress.configure(text="Status: Paused")
            else:
                self.btn_pause.configure(text="⏸ Pause", fg_color="#e0a800")
                self.lbl_progress.configure(text="Status: Processing...")

    def _start_processing(self) -> None:
        if not self.current_video_path or not Path(self.current_video_path).is_file():
            self.lbl_progress.configure(text="Error: Please select a valid video first.")
            return

        self.is_running = True
        self.is_paused = False
        self.btn_run.configure(text="⏹ STOP PIPELINE", fg_color="#dc3545", hover_color="#c82333")
        self.btn_pause.configure(state="normal", text="⏸ Pause", fg_color="#e0a800")
        self.btn_stop.configure(state="normal")
        self.lbl_progress.configure(text="Status: Initializing models...")

        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break

        # Spawn worker thread
        self.worker_thread = threading.Thread(target=self._pipeline_worker, daemon=True)
        self.worker_thread.start()

    def _stop_processing(self) -> None:
        self.is_running = False
        self.is_paused = False
        self.btn_run.configure(text="▶ START TRACKING PIPELINE", fg_color="#28a745", hover_color="#218838")
        self.btn_pause.configure(state="disabled", text="⏸ Pause", fg_color="#e0a800")
        self.btn_stop.configure(state="disabled")
        self.lbl_progress.configure(text="Status: Stopped")

    def _pipeline_worker(self) -> None:
        """Background thread executing full pipeline chain with live frame streaming."""
        video_path = Path(self.current_video_path)
        video_stem = video_path.stem

        out_video_path = Path("results") / f"{video_stem}_gui_output.mp4"
        out_csv_path = Path("results") / f"{video_stem}_gui_actions.csv"
        out_summary_path = Path("results") / f"{video_stem}_gui_summary.csv"
        out_video_path.parent.mkdir(parents=True, exist_ok=True)

        self.last_output_video = str(out_video_path)
        self.last_output_csv = str(out_csv_path)
        self.last_summary_csv = str(out_summary_path)

        # 1. Read video
        source = FileVideoSource(str(video_path))
        total_frames = source.frame_count
        fps = source.fps
        width, height = source.resolution
        frame_duration = 1.0 / fps if fps > 0 else 0.04

        # 2. Detector
        conf_val = float(self.conf_slider.get())
        detector = YOLODetector("yolov8n.pt", conf=conf_val, device="cpu")

        # 3. Tracker
        tracker_type = self.tracker_var.get()
        if tracker_type == "bytetrack":
            tracker = ByteTracker.from_config("configs/tracker_bytetrack.yaml")
        elif tracker_type == "sort":
            tracker = SORTTracker.from_config("configs/tracker_sort.yaml")
        elif tracker_type == "botsort":
            tracker = BoTSORTTracker.from_config("configs/tracker_botsort.yaml")
        else:
            tracker = OCSORTTracker.from_config("configs/tracker_ocsort.yaml")

        # 4. Pose Estimator
        pose_enabled = self.pose_var.get()
        pose_estimator = None
        pose_cache_dict: Dict[int, List[PoseDetection]] = {}
        if pose_enabled:
            cache_file = get_default_cache_path(video_path, "results/cache/pose")
            if has_pose_cache(cache_file):
                cached = load_pose_cache(cache_file)
                if cached:
                    pose_cache_dict = cached
            if not pose_cache_dict:
                pose_estimator = YOLOPoseEstimator("yolov8n-pose.pt", conf=conf_val, device="cpu", mode="whole_frame")

        # 5. Action Classifier & Buffer & Smoother
        backend = self.backend_var.get()
        classifier = create_action_classifier(backend=backend, device="cpu")
        action_buffer = TrackActionBuffer(window_size=16, stride=4, max_missing_frames=30)
        smoother = ActionSmoother(
            method=self.smooth_var.get(),
            window_len=7,
            alpha=0.4,
            hysteresis_threshold=0.6,
            classes=DEFAULT_ACTION_CLASSES,
        )

        visualizer = PipelineVisualizer(draw_boxes=True, draw_ids=True, draw_actions=True, draw_poses=pose_enabled, draw_hud=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        csv_rows: List[Tuple[float, int, str]] = []
        track_action_durations: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        track_total_time: Dict[int, float] = defaultdict(float)

        processed = 0
        start_time = time.perf_counter()

        try:
            for frame_idx, timestamp, frame in source:
                if not self.is_running:
                    break

                while self.is_paused and self.is_running:
                    time.sleep(0.05)

                # Detection
                detections = detector.detect(frame, frame_index=frame_idx)

                # Pose
                frame_poses: List[PoseDetection] = []
                if pose_enabled:
                    if frame_idx in pose_cache_dict:
                        frame_poses = pose_cache_dict[frame_idx]
                    elif pose_estimator is not None:
                        frame_poses = pose_estimator.estimate(frame, frame_index=frame_idx)

                # Tracking
                tracks = tracker.update(detections, frame=frame)

                # Match keypoints
                track_kpts_map: Dict[int, Any] = {}
                if pose_enabled and frame_poses and tracks:
                    track_boxes = [t[1] for t in tracks]
                    matched = match_pose_to_detections(track_boxes, frame_poses, iou_threshold=0.3)
                    for (tid, _, _), p in zip(tracks, matched):
                        if p is not None:
                            track_kpts_map[tid] = p.keypoints

                # Action Window update
                ready_windows = action_buffer.step(frame_idx, tracks, frame=frame, keypoints_map=track_kpts_map)
                if ready_windows:
                    preds = classifier.predict_batch(ready_windows)
                    for win, (pred_label, pred_conf, pred_probs) in zip(ready_windows, preds):
                        smoother.update(win.track_id, pred_label, pred_conf, probs=pred_probs)

                # Current smoothed actions
                current_actions: Dict[int, Tuple[str, float]] = {}
                act_counts: Dict[str, int] = defaultdict(int)

                for tid, bbox, score in tracks:
                    curr = smoother.get_current(tid)
                    if curr is not None:
                        act_lbl, act_c = curr
                    else:
                        act_lbl, act_c = "sitting", 0.7

                    current_actions[tid] = (act_lbl, act_c)
                    act_counts[act_lbl] += 1
                    csv_rows.append((round(timestamp, 3), tid, act_lbl))
                    track_action_durations[tid][act_lbl] += frame_duration
                    track_total_time[tid] += frame_duration

                # Draw annotation
                annotated = visualizer.draw_frame(
                    frame=frame,
                    tracks=tracks,
                    actions=current_actions,
                    poses=frame_poses if pose_enabled else None,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    timestamp=timestamp,
                    fps=fps,
                )
                writer.write(annotated)
                processed += 1

                elapsed = time.perf_counter() - start_time
                curr_fps = processed / elapsed if elapsed > 0 else 0.0

                # Send frame and stats to GUI queue
                payload = {
                    "frame": annotated,
                    "frame_idx": frame_idx,
                    "total_frames": total_frames,
                    "fps": curr_fps,
                    "students_count": len(tracks),
                    "act_counts": dict(act_counts),
                }

                if not self.frame_queue.full():
                    self.frame_queue.put(payload)

        finally:
            source.close()
            writer.release()

        # Save CSV outputs
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            cw = csv.writer(f)
            cw.writerow(["timestamp", "track_id", "action"])
            for r in csv_rows:
                cw.writerow(r)

        with open(out_summary_path, "w", newline="", encoding="utf-8") as f:
            cw = csv.writer(f)
            classes = DEFAULT_ACTION_CLASSES
            cw.writerow(["track_id", "total_duration_sec"] + [f"{c}_sec" for c in classes] + ["primary_action"])
            for tid in sorted(track_total_time.keys()):
                tot = track_total_time[tid]
                durs = [track_action_durations[tid].get(c, 0.0) for c in classes]
                prim = max(classes, key=lambda c: track_action_durations[tid].get(c, 0.0))
                cw.writerow([tid, round(tot, 2)] + [round(d, 2) for d in durs] + [prim])

        self.after(0, self._on_pipeline_completed, processed, total_frames)

    def _check_frame_queue(self) -> None:
        """Periodically consume annotated frames from worker and update GUI."""
        try:
            while not self.frame_queue.empty():
                data = self.frame_queue.get_nowait()
                bgr_frame = data["frame"]
                f_idx = data["frame_idx"]
                total = data["total_frames"]
                curr_fps = data["fps"]
                count = data["students_count"]
                act_counts = data["act_counts"]

                # Convert BGR to RGB and resize for canvas
                rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]

                # Max canvas bounds
                target_w, target_h = 860, 520
                scale = min(target_w / w, target_h / h)
                nw, nh = int(w * scale), int(h * scale)
                resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

                pil_img = Image.fromarray(resized)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(nw, nh))

                self.video_canvas.configure(image=ctk_img, text="")
                self.video_canvas.image = ctk_img  # type: ignore

                # Update progress
                progress = (f_idx + 1) / total if total > 0 else 0.0
                self.progress_bar.set(progress)
                self.lbl_progress.configure(text=f"Processing: {f_idx + 1}/{total} frames ({(progress*100):.1f}%)")

                # Update KPIs
                self.kpi_students.configure(text=str(count))
                self.kpi_sitting.configure(text=str(act_counts.get("sitting", 0)))
                self.kpi_raising.configure(text=str(act_counts.get("raising_hand", 0)))
                self.kpi_standing.configure(text=str(act_counts.get("standing", 0)))
                self.kpi_fps.configure(text=f"{curr_fps:.1f} FPS")

        except Exception:
            pass

        # Schedule next poll
        self.after(25, self._check_frame_queue)

    def _on_pipeline_completed(self, processed: int, total: int) -> None:
        self.is_running = False
        self.is_paused = False
        self.btn_run.configure(text="▶ START TRACKING PIPELINE", fg_color="#28a745", hover_color="#218838")
        self.btn_pause.configure(state="disabled", text="⏸ Pause", fg_color="#e0a800")
        self.btn_stop.configure(state="disabled")
        self.progress_bar.set(1.0)
        self.lbl_progress.configure(text=f"Status: Completed {processed}/{total} frames! Results ready.")

    def _open_result_video(self) -> None:
        p = self.last_output_video or "results/pipeline_output.mp4"
        if Path(p).is_file() and sys.platform == "win32":
            os.startfile(str(Path(p).resolve()))

    def _open_result_csv(self) -> None:
        p = self.last_output_csv or "results/pipeline_actions.csv"
        if Path(p).is_file() and sys.platform == "win32":
            os.startfile(str(Path(p).resolve()))

    def _open_result_summary(self) -> None:
        p = self.last_summary_csv or "results/pipeline_summary.csv"
        if Path(p).is_file() and sys.platform == "win32":
            os.startfile(str(Path(p).resolve()))


def main() -> None:
    app = ClassroomVisionApp()
    app.mainloop()


if __name__ == "__main__":
    main()

