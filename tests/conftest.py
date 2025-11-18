"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dummy_image_tensor():
    """Create dummy image tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def dummy_batch():
    """Create dummy batch of images."""
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 2, (batch_size,))
    return images, labels
