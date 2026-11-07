"""Training script for student action classification backends (Keypoint & Crop)."""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.action.classifier import (
    CropTemporalModel,
    KeypointTemporalModel,
    DEFAULT_ACTION_CLASSES,
)
from src.utils.config import load_config
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_action")


class LabeledActionDataset(Dataset):
    """Dataset loading window files (.npz) organized by class folders."""

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        backend: str = "keypoint",
        window_size: int = 16,
    ) -> None:
        self.backend = backend
        self.window_size = window_size
        if backend == "crop":
            # Exclude dummy-crop augmented files
            self.samples = [s for s in samples if not s[0].name.endswith("_flip.npz")]
        else:
            self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        data = np.load(file_path)

        if self.backend == "keypoint":
            kpts = data.get("keypoints", np.zeros((self.window_size, 17, 3), dtype=np.float32))
            x = torch.tensor(kpts, dtype=torch.float32)
        else:
            crops = data.get("crops", np.zeros((self.window_size, 112, 112, 3), dtype=np.uint8))
            # (T, H, W, 3) -> (T, 3, H, W) in [0, 1]
            c_norm = np.transpose(crops, (0, 3, 1, 2)).astype(np.float32) / 255.0
            x = torch.tensor(c_norm, dtype=torch.float32)

        return x, label


class SyntheticDryRunDataset(Dataset):
    """Synthetic in-memory dataset for dry-run verification."""

    def __init__(
        self,
        backend: str = "keypoint",
        num_samples: int = 32,
        num_classes: int = 4,
        window_size: int = 16,
    ) -> None:
        self.backend = backend
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.window_size = window_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = idx % self.num_classes
        if self.backend == "keypoint":
            # (T, 17, 3)
            x = torch.randn(self.window_size, 17, 3, dtype=torch.float32)
        else:
            # (T, 3, 112, 112)
            x = torch.rand(self.window_size, 3, 112, 112, dtype=torch.float32)
        return x, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train temporal student action classifier (Keypoint or Crop).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to labeled dataset directory containing class subfolders with .npz windows.",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="keypoint",
        choices=["keypoint", "crop"],
        help="Classifier backend architecture ('keypoint' or 'crop').",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/action.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results/checkpoints/action",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute a fast synthetic dry-run to verify the training pipeline.",
    )
    return parser.parse_args()


def train_action_model(
    backend: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    classes: List[str],
    epochs: int,
    lr: float,
    weight_decay: float,
    output_dir: Path,
    device: torch.device,
) -> Path:
    """Execute training loop with logging and checkpoint saving."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train.log"

    # Setup file logging
    file_handler = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s]: %(message)s"))
    logger.addHandler(file_handler)

    logger.info("Initializing model architecture for backend: %s", backend)
    if backend == "keypoint":
        model: nn.Module = KeypointTemporalModel(
            num_classes=num_classes,
            input_features=51,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
        )
    else:
        model = CropTemporalModel(
            num_classes=num_classes,
            backbone_name="mobilenet_v3_small",
            pretrained=True,
            hidden_dim=128,
            num_layers=1,
            dropout=0.3,
        )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_checkpoint_path = output_dir / f"{backend}_best.pt"

    logger.info("Starting training for %d epochs on device: %s", epochs, device)
    for epoch in range(1, epochs + 1):
        # 1. Train epoch
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == y).sum().item()
            train_total += len(y)

        train_loss_avg = train_loss / train_total if train_total > 0 else 0.0
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # 2. Validation epoch
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item() * len(y)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == y).sum().item()
                val_total += len(y)

        val_loss_avg = val_loss / val_total if val_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        logger.info(
            "Epoch [%d/%d] | Train Loss: %.4f, Train Acc: %.2f%% | Val Loss: %.4f, Val Acc: %.2f%%",
            epoch,
            epochs,
            train_loss_avg,
            train_acc * 100.0,
            val_loss_avg,
            val_acc * 100.0,
        )

        # Save checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "classes": classes,
                    "backend": backend,
                },
                best_checkpoint_path,
            )

    logger.info("Training complete. Best checkpoint saved at: %s (Val Acc: %.2f%%)", best_checkpoint_path, best_val_acc * 100.0)
    logger.removeHandler(file_handler)
    return best_checkpoint_path


def main() -> None:
    args = parse_args()

    # Load config defaults
    cfg = {}
    if Path(args.config).is_file():
        cfg = load_config(args.config)

    train_cfg = cfg.get("training", {})
    classes = cfg.get("classes", DEFAULT_ACTION_CLASSES)
    num_classes = len(classes)

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    epochs = args.epochs if args.epochs is not None else (2 if args.dry_run else train_cfg.get("epochs", 10))
    batch_size = args.batch_size if args.batch_size is not None else (8 if args.dry_run else train_cfg.get("batch_size", 16))
    lr = args.lr if args.lr is not None else train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    output_dir = Path(args.output_dir)

    # Set fixed random seed
    set_seed(seed)
    logger.info("Fixed random seed set to %d for deterministic execution.", seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dry_run:
        logger.info("=== RUNNING DRY-RUN ON SYNTHETIC RANDOM DATA ===")
        train_ds = SyntheticDryRunDataset(backend=args.backend, num_samples=32, num_classes=num_classes)
        val_ds = SyntheticDryRunDataset(backend=args.backend, num_samples=16, num_classes=num_classes)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        ckpt_path = train_action_model(
            backend=args.backend,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            classes=classes,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            output_dir=output_dir,
            device=device,
        )

        print("\n" + "=" * 60)
        print("DRY-RUN TRAINING EXECUTION SUCCESSFUL")
        print("=" * 60)
        print(f"Backend          : {args.backend}")
        print(f"Classes          : {classes}")
        print(f"Epochs Executed  : {epochs}")
        print(f"Checkpoint Saved : {ckpt_path}")
        print(f"Log File Saved   : {output_dir / 'train.log'}")
        print("=" * 60 + "\n")
        return

    # Real training path
    if not args.data_dir or not Path(args.data_dir).is_dir():
        logger.error("Data directory not found or not specified. Pass --data-dir or run with --dry-run.")
        sys.exit(1)

    data_path = Path(args.data_dir)
    samples: List[Tuple[Path, int]] = []
    for cls_idx, cls_name in enumerate(classes):
        cls_folder = data_path / cls_name
        if cls_folder.is_dir():
            for npz_file in cls_folder.glob("*.npz"):
                samples.append((npz_file, cls_idx))

    if not samples:
        logger.error("No .npz window files found in %s subfolders.", data_path)
        sys.exit(1)

    logger.info("Found %d labeled samples across %d classes.", len(samples), len(classes))
    # Simple split
    split_idx = int(0.8 * len(samples))
    np.random.shuffle(samples)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_ds = LabeledActionDataset(train_samples, backend=args.backend)
    val_ds = LabeledActionDataset(val_samples, backend=args.backend)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_action_model(
        backend=args.backend,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        classes=classes,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        output_dir=output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()

