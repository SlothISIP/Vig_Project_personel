"""Swin Transformer model for defect detection."""

from typing import Optional
import torch
import torch.nn as nn
import timm

from src.vision.models.base import BaseVisionModel
from src.core.logging import get_logger

logger = get_logger(__name__)


class SwinTransformer(BaseVisionModel):
    """
    Swin Transformer model for defect detection.

    This model uses a pretrained Swin Transformer backbone from timm
    with a custom classification head.
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        num_classes: int = 2,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        """
        Initialize Swin Transformer model.

        Args:
            model_name: Name of the Swin model variant
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            drop_rate: Dropout rate for classifier
            drop_path_rate: Drop path rate for backbone
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        logger.info(f"Initializing {model_name} (pretrained={pretrained})")

        # Load pretrained Swin Transformer backbone
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove default classifier
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        logger.info(f"Backbone feature dimension: {self.feature_dim}")

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(drop_rate),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes),
        )

        # Initialize classifier weights
        self._init_classifier()

        logger.info(
            f"Model initialized with {self.count_parameters():,} trainable parameters"
        )

    def _init_classifier(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output logits of shape (B, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Classify
        logits = self.classifier(features)

        return logits

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning."""
        logger.info("Freezing backbone parameters")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        logger.info("Unfreezing backbone parameters")
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get attention weights from the last Swin block.

        Returns:
            Attention weights if available, None otherwise
        """
        # This would require modifying the forward pass to cache attention
        # For now, return None as placeholder
        logger.warning("Attention extraction not yet implemented")
        return None


def create_swin_tiny(num_classes: int = 2, pretrained: bool = True) -> SwinTransformer:
    """Create Swin-Tiny model."""
    return SwinTransformer(
        model_name="swin_tiny_patch4_window7_224",
        num_classes=num_classes,
        pretrained=pretrained,
    )


def create_swin_small(num_classes: int = 2, pretrained: bool = True) -> SwinTransformer:
    """Create Swin-Small model."""
    return SwinTransformer(
        model_name="swin_small_patch4_window7_224",
        num_classes=num_classes,
        pretrained=pretrained,
    )


def create_swin_base(num_classes: int = 2, pretrained: bool = True) -> SwinTransformer:
    """Create Swin-Base model."""
    return SwinTransformer(
        model_name="swin_base_patch4_window7_224",
        num_classes=num_classes,
        pretrained=pretrained,
    )
