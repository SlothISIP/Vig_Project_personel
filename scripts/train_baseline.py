#!/usr/bin/env python3
"""Baseline training script for defect detection."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.vision.models.swin_transformer import create_swin_tiny
from src.vision.preprocessing.dataset import MVTecDataset, CombinedMVTecDataset
from src.vision.preprocessing.transforms import DefectDetectionTransforms
from src.vision.training.trainer import DefectDetectionTrainer
from src.core.config import get_settings
from src.core.logging import setup_logging, get_logger
from src.core.constants import CHECKPOINTS_DIR, RAW_DATA_DIR

# Setup logging
setup_logging(level="INFO", log_format="text")
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Train defect detection model")
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["bottle"],
        help="MVTec categories to train on",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="swin_tiny",
        choices=["swin_tiny", "swin_small", "swin_base"],
        help="Model architecture",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--image-size", type=int, default=224, help="Input image size"
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Augmentation level",
    )
    parser.add_argument(
        "--pretrained", action="store_true", default=True, help="Use pretrained weights"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINTS_DIR,
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume from checkpoint"
    )

    return parser.parse_args()


def create_data_loaders(args):
    """Create training and validation data loaders."""
    logger.info("Creating data loaders...")

    # Check if data exists
    data_dir = RAW_DATA_DIR / "mvtec_ad"
    if not data_dir.exists():
        logger.error(f"Dataset not found at {data_dir}")
        logger.error("Please download MVTec AD dataset first:")
        logger.error("  python scripts/download_mvtec.py")
        sys.exit(1)

    # Create transforms
    train_transform = DefectDetectionTransforms.get_train_transforms(
        image_size=args.image_size,
        augmentation_level=args.augmentation,
    )
    val_transform = DefectDetectionTransforms.get_val_transforms(
        image_size=args.image_size
    )

    # Create datasets
    if len(args.categories) == 1:
        # Single category
        category = args.categories[0]
        logger.info(f"Training on single category: {category}")

        train_dataset = MVTecDataset(
            data_dir=data_dir,
            category=category,
            split="train",
            transform=train_transform,
        )
        val_dataset = MVTecDataset(
            data_dir=data_dir,
            category=category,
            split="test",
            transform=val_transform,
        )
    else:
        # Multiple categories
        logger.info(f"Training on {len(args.categories)} categories: {args.categories}")

        train_dataset = CombinedMVTecDataset(
            data_dir=data_dir,
            categories=args.categories,
            split="train",
            transform=train_transform,
        )
        val_dataset = CombinedMVTecDataset(
            data_dir=data_dir,
            categories=args.categories,
            split="test",
            transform=val_transform,
        )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
    )

    return train_loader, val_loader


def create_model(args):
    """Create model."""
    logger.info(f"Creating model: {args.model}")

    if args.model == "swin_tiny":
        model = create_swin_tiny(num_classes=2, pretrained=args.pretrained)
    elif args.model == "swin_small":
        from src.vision.models.swin_transformer import create_swin_small

        model = create_swin_small(num_classes=2, pretrained=args.pretrained)
    elif args.model == "swin_base":
        from src.vision.models.swin_transformer import create_swin_base

        model = create_swin_base(num_classes=2, pretrained=args.pretrained)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    return model


def main():
    """Main training function."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("DEFECT DETECTION TRAINING")
    logger.info("=" * 80)
    logger.info(f"Categories: {args.categories}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Augmentation: {args.augmentation}")
    logger.info(f"Pretrained: {args.pretrained}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(args)

    # Create model
    model = create_model(args)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    # Create trainer
    trainer = DefectDetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=5,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, metrics = trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from epoch {start_epoch}")
        logger.info(f"Previous metrics: {metrics}")

    # Train
    logger.info("\nStarting training...\n")
    history = trainer.train(num_epochs=args.epochs, save_best_only=True)

    # Save final model
    final_checkpoint = args.checkpoint_dir / "final_model.pth"
    trainer.save_checkpoint(
        epoch=args.epochs,
        metrics={"final": True},
        filename="final_model.pth",
    )
    logger.info(f"\nFinal model saved to: {final_checkpoint}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
