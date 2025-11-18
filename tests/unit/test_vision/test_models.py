"""Unit tests for vision models."""

import pytest
import torch

from src.vision.models.swin_transformer import (
    SwinTransformer,
    create_swin_tiny,
    create_swin_small,
)


class TestSwinTransformer:
    """Tests for Swin Transformer model."""

    def test_model_creation(self):
        """Test model can be created."""
        model = SwinTransformer(
            model_name="swin_tiny_patch4_window7_224",
            num_classes=2,
            pretrained=False,
        )

        assert model is not None
        assert model.num_classes == 2
        assert model.model_name == "swin_tiny_patch4_window7_224"

    def test_forward_pass(self):
        """Test forward pass works."""
        model = create_swin_tiny(num_classes=2, pretrained=False)
        model.eval()

        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (batch_size, 2)

    def test_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        model = create_swin_tiny(num_classes=2, pretrained=False)
        model.eval()

        for batch_size in [1, 2, 4, 8]:
            input_tensor = torch.randn(batch_size, 3, 224, 224)

            with torch.no_grad():
                output = model(input_tensor)

            assert output.shape == (batch_size, 2)

    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        model = create_swin_tiny(num_classes=2, pretrained=False)

        num_params = sum(p.numel() for p in model.parameters())

        # Swin-Tiny should have around 28M parameters
        assert num_params > 20_000_000
        assert num_params < 35_000_000

    def test_freeze_backbone(self):
        """Test backbone freezing."""
        model = create_swin_tiny(num_classes=2, pretrained=False)

        # Initially, backbone should be trainable
        backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
        assert backbone_trainable

        # Freeze backbone
        model.freeze_backbone()

        # After freezing, backbone should not be trainable
        backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
        assert not backbone_trainable

        # Classifier should still be trainable
        classifier_trainable = any(p.requires_grad for p in model.classifier.parameters())
        assert classifier_trainable

    def test_unfreeze_backbone(self):
        """Test backbone unfreezing."""
        model = create_swin_tiny(num_classes=2, pretrained=False)

        # Freeze then unfreeze
        model.freeze_backbone()
        model.unfreeze_backbone()

        # Backbone should be trainable again
        backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
        assert backbone_trainable

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test model works on CUDA."""
        model = create_swin_tiny(num_classes=2, pretrained=False)
        model = model.cuda()

        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224).cuda()

        with torch.no_grad():
            output = model(input_tensor)

        assert output.is_cuda
        assert output.shape == (batch_size, 2)
