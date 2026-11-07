# Data Directory Structure and Dataset Conversion Guide

This folder contains raw datasets, video recordings, and preprocessed assets for classroom multi-object tracking and action recognition.

## Folder Organization

- `data/raw/`:
  - **Purpose**: Destination for original, unprocessed classroom video recordings and external benchmark datasets.
  - **Usage**: Place raw video files (e.g., `.mp4`, `.avi`) or benchmark archives here.
  - **Notice**: Never commit raw video files or benchmark archives to Git. Some datasets require formal signed access requests and must be obtained manually. No automatic downloads are performed by the codebase.

- `data/processed/`:
  - **Purpose**: Output directory for converted annotations and preprocessed artifacts formatted for the project pipeline.
  - **Contents**: Standardized MOTChallenge ground truth (`gt.txt`), extracted keypoints (`keypoints.json`), and sequence metadata (`seqinfo.ini`).

---

## Supported Dataset Formats & Expected Folder Layouts

### 1. DanceTrack Benchmark
DanceTrack is formatted in MOTChallenge-style sequence folders. Place raw sequences in `data/raw/dancetrack/`:

```text
data/raw/dancetrack/
└── dancetrack0001/
    ├── gt/
    │   └── gt.txt          # Raw annotations: <frame>,<id>,<x>,<y>,<w>,<h>,1,1,1
    ├── img1/               # Extracted frames (optional for tracking eval)
    │   ├── 00000001.jpg
    │   └── ...
    └── seqinfo.ini         # Sequence metadata (FPS, resolution, length)
```

**Conversion Command**:
```powershell
py -3.11 scripts/convert_dataset.py --dataset dancetrack --input data/raw/dancetrack/dancetrack0001 --output-dir data/processed/dancetrack0001
```

---

### 2. PoseTrack21 Benchmark
PoseTrack21 uses COCO-style JSON annotation files for each video clip containing bounding boxes, track IDs, and 17 2D human pose keypoints:

```text
data/raw/posetrack21/
└── posetrack_data/
    ├── train/
    │   ├── 000001_bonn_train.json
    │   └── ...
    └── val/
        ├── 000342_mpii_val.json
        └── ...
```

**JSON Annotation Schema expected**:
- `images`: `[{"id": int, "file_name": str, "frame_id": int}, ...]`
- `annotations`: `[{"image_id": int, "track_id": int, "bbox": [x, y, w, h], "keypoints": [x1, y1, v1, ... 17 joints]}, ...]`

**Conversion Output**:
- `gt.txt`: Standard MOTChallenge bounding boxes and track IDs (`<frame>,<id>,<x>,<y>,<w>,<h>,<conf>,-1,-1,-1`).
- `keypoints.json`: 2D human pose keypoints mapped to corresponding `frame` and `track_id`.

**Conversion Command**:
```powershell
py -3.11 scripts/convert_dataset.py --dataset posetrack21 --input data/raw/posetrack21/posetrack_data/val/000342_mpii_val.json --output-dir data/processed/posetrack_000342
```

---

### 3. Custom Classroom Video (CVAT Exported as "MOT 1.1")
Classroom videos annotated using [CVAT](https://github.com/cvat-ai/cvat) and exported using the **"MOT 1.1"** format export option:

```text
data/raw/classroom_cvat/
└── classroom_session_01/
    ├── gt/
    │   └── gt.txt          # CVAT MOT 1.1: <frame>,<id>,<x>,<y>,<w>,<h>,<conf>,<class_id>,<vis>
    └── seqinfo.ini         # Optional CVAT sequence info
```

**Conversion Command**:
```powershell
py -3.11 scripts/convert_dataset.py --dataset cvat --input data/raw/classroom_cvat/classroom_session_01 --output-dir data/processed/classroom_session_01
```

---

## Guidelines for Real Annotation Files
If your export format from CVAT or PoseTrack21 contains custom category names, 0-based frame indexing, or custom column ordering, please provide a 2-3 line sample snippet from the real annotation file so the conversion logic can be validated against the exact schema without guessing.
