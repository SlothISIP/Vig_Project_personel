"""Base model interface for vision models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn


class BaseVisionModel(ABC, nn.Module):
    """Abstract base class for all vision models."""

    def __init__(self):
        """Initialize base model."""
        super().__init__()
        self.model_name: Optional[str] = None
        self.num_classes: Optional[int] = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output logits of shape (B, num_classes)
        """
        pass

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.load_state_dict(checkpoint)

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
            optimizer: Optimizer state (optional)
            epoch: Current epoch (optional)
            metrics: Training metrics (optional)
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_name": self.model_name,
            "num_classes": self.num_classes,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if epoch is not None:
            checkpoint["epoch"] = epoch

        if metrics is not None:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, checkpoint_path)

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        # To be implemented by subclasses if needed
        pass

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        # To be implemented by subclasses if needed
        pass
