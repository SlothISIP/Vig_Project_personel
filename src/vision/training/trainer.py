"""Training pipeline for defect detection models."""

from typing import Dict, Optional
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.vision.training.metrics import MetricsCalculator, AverageMeter
from src.core.logging import get_logger
from src.core.constants import CHECKPOINTS_DIR

logger = get_logger(__name__)


class DefectDetectionTrainer:
    """
    Trainer for defect detection models.

    Handles training loop, validation, checkpointing, and metrics tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
        early_stopping_patience: int = 5,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.early_stopping_patience = early_stopping_patience

        if checkpoint_dir is None:
            checkpoint_dir = CHECKPOINTS_DIR
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.patience_counter = 0

        self.metrics_calc = MetricsCalculator()

        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        all_predictions = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, (images, targets) in enumerate(pbar):
            batch_size = images.size(0)
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calculate accuracy
            _, predictions = outputs.max(1)
            correct = predictions.eq(targets).sum().item()
            accuracy = correct / batch_size

            # Update meters
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy, batch_size)

            # Collect predictions for metrics
            all_predictions.append(predictions)
            all_targets.append(targets)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss_meter.avg:.4f}",
                    "acc": f"{acc_meter.avg:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

        # Calculate epoch metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        metrics = {
            "train_loss": loss_meter.avg,
            "train_acc": acc_meter.avg,
            "train_precision": self.metrics_calc.calculate_precision(
                all_predictions, all_targets
            ),
            "train_recall": self.metrics_calc.calculate_recall(all_predictions, all_targets),
            "train_f1": self.metrics_calc.calculate_f1(all_predictions, all_targets),
        }

        return metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        all_predictions = []
        all_targets = []
        all_probabilities = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

        for images, targets in pbar:
            batch_size = images.size(0)
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # Calculate accuracy
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
            correct = predictions.eq(targets).sum().item()
            accuracy = correct / batch_size

            # Update meters
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy, batch_size)

            # Collect predictions
            all_predictions.append(predictions)
            all_targets.append(targets)
            all_probabilities.append(probabilities)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss_meter.avg:.4f}",
                    "acc": f"{acc_meter.avg:.4f}",
                }
            )

        # Calculate epoch metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        all_probabilities = torch.cat(all_probabilities)

        metrics = {
            "val_loss": loss_meter.avg,
            "val_acc": acc_meter.avg,
            "val_precision": self.metrics_calc.calculate_precision(
                all_predictions, all_targets
            ),
            "val_recall": self.metrics_calc.calculate_recall(all_predictions, all_targets),
            "val_f1": self.metrics_calc.calculate_f1(all_predictions, all_targets),
            "val_roc_auc": self.metrics_calc.calculate_roc_auc(
                all_probabilities, all_targets
            ),
        }

        return metrics

    def train(self, num_epochs: int, save_best_only: bool = True) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            save_best_only: Only save best model if True

        Returns:
            History of metrics
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}

            # Update history
            for key in history:
                if key in epoch_metrics:
                    history[key].append(epoch_metrics[key])

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Print summary
            epoch_time = time.time() - epoch_start_time
            logger.info(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            logger.info(
                f"Train - Loss: {train_metrics['train_loss']:.4f}, "
                f"Acc: {train_metrics['train_acc']:.4f}, "
                f"F1: {train_metrics['train_f1']:.4f}"
            )
            logger.info(
                f"Val   - Loss: {val_metrics['val_loss']:.4f}, "
                f"Acc: {val_metrics['val_acc']:.4f}, "
                f"F1: {val_metrics['val_f1']:.4f}"
            )

            # Save checkpoint
            is_best = val_metrics["val_acc"] > self.best_val_acc

            if is_best:
                self.best_val_acc = val_metrics["val_acc"]
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0

                # Save best model
                self.save_checkpoint(
                    epoch=epoch,
                    metrics=epoch_metrics,
                    filename="best_model.pth",
                )
                logger.info(f"✓ New best model saved (acc: {self.best_val_acc:.4f})")

            elif not save_best_only:
                # Save epoch checkpoint
                self.save_checkpoint(
                    epoch=epoch,
                    metrics=epoch_metrics,
                    filename=f"epoch_{epoch:03d}.pth",
                )

            # Early stopping
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(patience: {self.early_stopping_patience})"
                )
                break

        logger.info("Training complete!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")

        return history

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        filename: str = "checkpoint.pth",
    ):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Current metrics
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

        return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
