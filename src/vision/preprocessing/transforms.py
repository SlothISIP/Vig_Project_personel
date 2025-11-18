"""Data augmentation and preprocessing transforms."""

from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.core.constants import IMAGENET_MEAN, IMAGENET_STD, DEFAULT_IMAGE_SIZE


class DefectDetectionTransforms:
    """
    Augmentation pipeline for defect detection.

    Provides separate transforms for training and validation/testing.
    """

    @staticmethod
    def get_train_transforms(
        image_size: int = DEFAULT_IMAGE_SIZE,
        augmentation_level: str = "medium",
    ) -> A.Compose:
        """
        Get training augmentation pipeline.

        Args:
            image_size: Target image size
            augmentation_level: "light", "medium", or "heavy"

        Returns:
            Albumentations compose pipeline
        """
        base_transforms = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]

        if augmentation_level in ["medium", "heavy"]:
            base_transforms.extend(
                [
                    A.Rotate(limit=45, p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                    ),
                ]
            )

        if augmentation_level == "medium":
            base_transforms.extend(
                [
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(
                                brightness_limit=0.2, contrast_limit=0.2, p=1.0
                            ),
                            A.HueSaturationValue(
                                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0
                            ),
                        ],
                        p=0.3,
                    ),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                ]
            )

        if augmentation_level == "heavy":
            base_transforms.extend(
                [
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(
                                brightness_limit=0.3, contrast_limit=0.3, p=1.0
                            ),
                            A.HueSaturationValue(
                                hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1.0
                            ),
                            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
                        ],
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                            A.MotionBlur(blur_limit=7, p=1.0),
                        ],
                        p=0.3,
                    ),
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=32,
                        max_width=32,
                        fill_value=0,
                        p=0.3,
                    ),
                ]
            )

        # Normalization and tensor conversion (always applied)
        base_transforms.extend(
            [
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )

        return A.Compose(base_transforms)

    @staticmethod
    def get_val_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
        """
        Get validation/test augmentation pipeline.

        Args:
            image_size: Target image size

        Returns:
            Albumentations compose pipeline
        """
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def get_test_time_augmentation(
        image_size: int = DEFAULT_IMAGE_SIZE,
        n_augmentations: int = 5,
    ) -> list[A.Compose]:
        """
        Get test-time augmentation pipelines.

        Args:
            image_size: Target image size
            n_augmentations: Number of augmented versions

        Returns:
            List of augmentation pipelines
        """
        base_pipeline = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )

        augmented_pipelines = []

        for i in range(n_augmentations - 1):
            if i == 0:
                pipeline = A.Compose(
                    [
                        A.Resize(image_size, image_size),
                        A.HorizontalFlip(p=1.0),
                        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                        ToTensorV2(),
                    ]
                )
            elif i == 1:
                pipeline = A.Compose(
                    [
                        A.Resize(image_size, image_size),
                        A.VerticalFlip(p=1.0),
                        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                        ToTensorV2(),
                    ]
                )
            elif i == 2:
                pipeline = A.Compose(
                    [
                        A.Resize(image_size, image_size),
                        A.Rotate(limit=15, p=1.0),
                        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                        ToTensorV2(),
                    ]
                )
            else:
                pipeline = A.Compose(
                    [
                        A.Resize(image_size, image_size),
                        A.RandomBrightnessContrast(p=1.0),
                        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                        ToTensorV2(),
                    ]
                )

            augmented_pipelines.append(pipeline)

        return [base_pipeline] + augmented_pipelines
