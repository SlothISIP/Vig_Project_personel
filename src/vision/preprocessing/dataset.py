"""PyTorch datasets for defect detection."""

from pathlib import Path
from typing import Optional, Tuple, Callable, List
import cv2
import numpy as np
from torch.utils.data import Dataset

from src.core.constants import RAW_DATA_DIR
from src.core.exceptions import DatasetNotFoundError
from src.core.logging import get_logger

logger = get_logger(__name__)


class MVTecDataset(Dataset):
    """
    MVTec Anomaly Detection dataset.

    Dataset structure:
        mvtec_ad/
            category/
                train/
                    good/
                        *.png
                test/
                    good/
                        *.png
                    defect_type/
                        *.png
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        category: str = "bottle",
        split: str = "train",
        transform: Optional[Callable] = None,
        binary_classification: bool = True,
    ):
        """
        Initialize MVTec dataset.

        Args:
            data_dir: Root data directory (defaults to RAW_DATA_DIR/mvtec_ad)
            category: Product category (e.g., "bottle", "cable")
            split: "train" or "test"
            transform: Albumentations transform pipeline
            binary_classification: If True, binary good/defect classification.
                                   If False, multi-class defect type classification.
        """
        if data_dir is None:
            data_dir = RAW_DATA_DIR / "mvtec_ad"

        self.data_dir = Path(data_dir)
        self.category = category
        self.split = split
        self.transform = transform
        self.binary_classification = binary_classification

        # Dataset path
        self.dataset_path = self.data_dir / category / split

        if not self.dataset_path.exists():
            raise DatasetNotFoundError(
                f"Dataset not found at {self.dataset_path}. "
                f"Please download MVTec AD dataset first."
            )

        # Load samples
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}

        self._load_samples()

        logger.info(
            f"Loaded {len(self.samples)} samples from {category}/{split} "
            f"({len(self.class_to_idx)} classes)"
        )

    def _load_samples(self):
        """Load image paths and labels."""
        if self.split == "train":
            # Training only has 'good' samples
            good_dir = self.dataset_path / "good"

            if not good_dir.exists():
                raise DatasetNotFoundError(f"Good samples not found at {good_dir}")

            self.class_to_idx = {"good": 0}

            for img_path in sorted(good_dir.glob("*.png")):
                self.samples.append((img_path, 0))

        else:  # test
            # Test has 'good' and defect samples
            class_idx = 0

            for class_dir in sorted(self.dataset_path.iterdir()):
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name

                # For binary classification, map all defects to class 1
                if self.binary_classification:
                    if class_name == "good":
                        label = 0
                        if "good" not in self.class_to_idx:
                            self.class_to_idx["good"] = 0
                    else:
                        label = 1
                        if "defect" not in self.class_to_idx:
                            self.class_to_idx["defect"] = 1
                else:
                    # Multi-class: each defect type gets unique label
                    if class_name not in self.class_to_idx:
                        self.class_to_idx[class_name] = class_idx
                        class_idx += 1
                    label = self.class_to_idx[class_name]

                # Load all images from this class
                for img_path in sorted(class_dir.glob("*.png")):
                    self.samples.append((img_path, label))

        # Create reverse mapping
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get item by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, label)
        """
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path))

        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

    def get_class_counts(self) -> dict[str, int]:
        """
        Get sample count per class.

        Returns:
            Dictionary mapping class names to counts
        """
        counts = {class_name: 0 for class_name in self.class_to_idx.keys()}

        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            counts[class_name] += 1

        return counts


class CombinedMVTecDataset(Dataset):
    """
    Combined MVTec dataset with multiple categories.

    Useful for training on all categories simultaneously.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        categories: Optional[List[str]] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        binary_classification: bool = True,
    ):
        """
        Initialize combined dataset.

        Args:
            data_dir: Root data directory
            categories: List of categories to include
            split: "train" or "test"
            transform: Transform pipeline
            binary_classification: Binary or multi-class classification
        """
        if categories is None:
            # Default: all MVTec AD categories
            categories = [
                "bottle",
                "cable",
                "capsule",
                "carpet",
                "grid",
                "hazelnut",
                "leather",
                "metal_nut",
                "pill",
                "screw",
                "tile",
                "toothbrush",
                "transistor",
                "wood",
                "zipper",
            ]

        self.categories = categories
        self.datasets = []

        # Create dataset for each category
        for category in categories:
            try:
                dataset = MVTecDataset(
                    data_dir=data_dir,
                    category=category,
                    split=split,
                    transform=transform,
                    binary_classification=binary_classification,
                )
                self.datasets.append(dataset)
            except DatasetNotFoundError as e:
                logger.warning(f"Skipping category {category}: {e}")

        # Calculate total length
        self.lengths = [len(ds) for ds in self.datasets]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)

        logger.info(
            f"Combined dataset: {len(self)} samples from {len(self.datasets)} categories"
        )

    def __len__(self) -> int:
        """Get total dataset size."""
        return sum(self.lengths)

    def __getitem__(self, idx: int):
        """Get item from appropriate sub-dataset."""
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1

        # Get local index within that dataset
        local_idx = idx - self.cumulative_lengths[dataset_idx]

        # Get item from sub-dataset
        return self.datasets[dataset_idx][local_idx]
