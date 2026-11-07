# Data Directory Structure

This folder contains dataset files, raw videos, and preprocessed assets for classroom multi-object tracking and action recognition.

## Folder Organization

- `data/raw/`:
  - **Purpose**: Destination for original, unprocessed classroom video recordings and external benchmark datasets.
  - **Usage**: Place raw video files (e.g., `.mp4`, `.avi`) or benchmark archives here.
  - **Notice**: Never commit raw video files or benchmark archives to Git. Some datasets require formal signed access requests and must be obtained and placed here manually.

- `data/processed/`:
  - **Purpose**: Output directory for preprocessed artifacts.
  - **Contents**: Extracted frames, aligned bounding boxes, pose keypoint sequences, processed tracklets, and train/val/test split manifests.
  - **Notice**: Automatically generated or preprocessed data is ignored by version control to preserve disk space and privacy.

## Guidelines
1. No dataset is downloaded automatically by the codebase.
2. Ensure compliance with data protection and ethical approval agreements for classroom recordings.
