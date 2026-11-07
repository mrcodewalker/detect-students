# Real-Time Multi-Object Tracking and Action Recognition of Students in Classroom Video

## Project Overview
This repository contains the codebase for an undergraduate thesis focused on **real-time multi-object tracking (MOT) and action recognition of students** in classroom video recorded from a single fixed camera.

### Core Objectives
1. **Real-Time Detection & Tracking**: Reliably detect and track multiple students under classroom conditions (occlusions, dense seating, varying illumination) using a single stationary viewpoint.
2. **Action Recognition**: Recognize individual and group classroom actions (e.g., listening, writing, raising hand, discussing, standing) over temporal tracklets.
3. **Efficiency & Reproducibility**: Run in real-time on a single GPU (12 GB VRAM target), driven by modular YAML configurations with fixed random seeds for deterministic evaluation.

---

## Directory Structure & Folder Meaning

```
detect-students/
├── configs/          # YAML configuration files (model backbones, tracking params, action classifiers, training/inference hyperparameters)
├── data/             # Dataset directory (excluded from git tracking, except documentation)
│   ├── raw/          # Original, unprocessed classroom video recordings and external benchmarks
│   ├── processed/    # Preprocessed frames, extracted keypoint sequences, crops, and metadata splits
│   └── README.md     # Instructions on data placement, ethical access, and manual setup
├── src/              # Core source package
│   ├── io/           # Video ingestion (camera streams, video files), frame decoders, and output writers
│   ├── detection/    # Object detectors (bounding box estimation for students)
│   ├── tracking/     # Multi-object tracking algorithms (Kalman filter, ReID association, tracklet management)
│   ├── pose/         # 2D human pose estimation / keypoint extractors for fine-grained action features
│   ├── action/       # Action recognition models (spatio-temporal action classifiers, sequence modeling)
│   ├── eval/         # Evaluation routines and metric calculations (MOTA, MOTP, IDF1, HOTA, action mAP / Accuracy)
│   ├── viz/          # Visualization utilities (bounding boxes, track IDs, action labels, dashboard overlays)
│   └── utils/        # Common utilities (seed fixing, YAML config parsers, logging, hardware checks)
├── scripts/          # Executable CLI entrypoints for data processing, training, evaluation, and live demo runs
├── tests/            # Automated unit and integration tests
├── results/          # Experiment outputs, metric summaries, and exported inference videos
├── requirements.txt  # Pinned dependencies with verified licenses
├── .gitignore        # Version control ignore rules (data, weights, artifacts, caches)
└── README.md         # Project documentation and architectural overview
```

---

## Technical Constraints & Design Principles
- **Runtime Environment**: Python 3.10+, PyTorch.
- **Hardware Target**: 1 GPU with 12 GB VRAM.
- **Configuration**: Fully config-driven via YAML files.
- **CLI-driven**: Every operational step (preprocessing, training, evaluation, inference) is runnable from the command line.
- **Data Policy**: No automatic dataset downloads; datasets requiring signed agreements are handled via documented manual placement.
