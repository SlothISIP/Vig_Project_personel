#!/usr/bin/env python3
"""Test script to verify the setup."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.vision.models.swin_transformer import create_swin_tiny
from src.core.logging import setup_logging, get_logger
from src.core.config import get_settings

# Setup logging
setup_logging(level="INFO", log_format="text")
logger = get_logger(__name__)


def test_model_creation():
    """Test model creation."""
    logger.info("=" * 60)
    logger.info("Testing model creation...")
    logger.info("=" * 60)

    try:
        # Create model
        model = create_swin_tiny(num_classes=2, pretrained=False)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        logger.info(f"✓ Model created successfully")
        logger.info(f"✓ Total parameters: {num_params:,}")

        # Test forward pass
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        logger.info(f"✓ Model moved to device: {device}")

        # Dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)

        logger.info(f"✓ Forward pass successful")
        logger.info(f"  Input shape: {dummy_input.shape}")
        logger.info(f"  Output shape: {output.shape}")

        assert output.shape == (batch_size, 2), "Output shape mismatch!"

        logger.info("✓ All model tests passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Model test failed: {e}")
        return False


def test_config():
    """Test configuration system."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing configuration system...")
    logger.info("=" * 60)

    try:
        settings = get_settings()

        logger.info(f"✓ Settings loaded")
        logger.info(f"  Environment: {settings.environment}")
        logger.info(f"  Model name: {settings.model.name}")
        logger.info(f"  Batch size: {settings.training.batch_size}")
        logger.info(f"  Learning rate: {settings.training.learning_rate}")

        logger.info("✓ All config tests passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Config test failed: {e}")
        return False


def test_transforms():
    """Test data transforms."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing data transforms...")
    logger.info("=" * 60)

    try:
        import numpy as np
        from src.vision.preprocessing.transforms import DefectDetectionTransforms

        # Create transforms
        train_transform = DefectDetectionTransforms.get_train_transforms()
        val_transform = DefectDetectionTransforms.get_val_transforms()

        logger.info(f"✓ Transforms created")

        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Apply transforms
        transformed_train = train_transform(image=dummy_image)["image"]
        transformed_val = val_transform(image=dummy_image)["image"]

        logger.info(f"✓ Transforms applied successfully")
        logger.info(f"  Original shape: {dummy_image.shape}")
        logger.info(f"  Transformed shape (train): {transformed_train.shape}")
        logger.info(f"  Transformed shape (val): {transformed_val.shape}")

        assert transformed_train.shape == (3, 224, 224), "Transform shape mismatch!"

        logger.info("✓ All transform tests passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Transform test failed: {e}")
        return False


def check_cuda():
    """Check CUDA availability."""
    logger.info("\n" + "=" * 60)
    logger.info("Checking CUDA...")
    logger.info("=" * 60)

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        logger.info(f"✓ CUDA is available")
        logger.info(f"  Device count: {torch.cuda.device_count()}")
        logger.info(f"  Current device: {torch.cuda.current_device()}")
        logger.info(f"  Device name: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠ CUDA is not available (CPU only)")

    return cuda_available


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("DIGITAL TWIN FACTORY - SETUP VERIFICATION")
    logger.info("=" * 80)

    # Run tests
    results = {}
    results["CUDA"] = check_cuda()
    results["Config"] = test_config()
    results["Transforms"] = test_transforms()
    results["Model"] = test_model_creation()

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 80)

    if all_passed:
        logger.info("\n🎉 All tests passed! Setup is complete.")
        logger.info("\nNext steps:")
        logger.info("  1. Download MVTec AD dataset")
        logger.info("  2. Run: python scripts/train_simple.py")
        return 0
    else:
        logger.error("\n❌ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
