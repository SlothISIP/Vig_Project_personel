"""Evaluation metrics for defect detection."""

from typing import Dict
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


class MetricsCalculator:
    """Calculate various classification metrics."""

    @staticmethod
    def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate accuracy.

        Args:
            predictions: Predicted class indices
            targets: Ground truth labels

        Returns:
            Accuracy score
        """
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        return accuracy_score(targets, predictions)

    @staticmethod
    def calculate_precision(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        average: str = "binary",
    ) -> float:
        """
        Calculate precision.

        Args:
            predictions: Predicted class indices
            targets: Ground truth labels
            average: Averaging method ('binary', 'micro', 'macro', 'weighted')

        Returns:
            Precision score
        """
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        return precision_score(targets, predictions, average=average, zero_division=0)

    @staticmethod
    def calculate_recall(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        average: str = "binary",
    ) -> float:
        """
        Calculate recall.

        Args:
            predictions: Predicted class indices
            targets: Ground truth labels
            average: Averaging method

        Returns:
            Recall score
        """
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        return recall_score(targets, predictions, average=average, zero_division=0)

    @staticmethod
    def calculate_f1(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        average: str = "binary",
    ) -> float:
        """
        Calculate F1 score.

        Args:
            predictions: Predicted class indices
            targets: Ground truth labels
            average: Averaging method

        Returns:
            F1 score
        """
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        return f1_score(targets, predictions, average=average, zero_division=0)

    @staticmethod
    def calculate_confusion_matrix(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> np.ndarray:
        """
        Calculate confusion matrix.

        Args:
            predictions: Predicted class indices
            targets: Ground truth labels

        Returns:
            Confusion matrix
        """
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        return confusion_matrix(targets, predictions)

    @staticmethod
    def calculate_roc_auc(
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        multi_class: str = "ovr",
    ) -> float:
        """
        Calculate ROC AUC score.

        Args:
            probabilities: Predicted probabilities (not logits!)
            targets: Ground truth labels
            multi_class: Multi-class strategy ('ovr' or 'ovo')

        Returns:
            ROC AUC score
        """
        probabilities = probabilities.cpu().numpy()
        targets = targets.cpu().numpy()

        try:
            if probabilities.shape[1] == 2:
                # Binary classification: use positive class probability
                return roc_auc_score(targets, probabilities[:, 1])
            else:
                # Multi-class
                return roc_auc_score(
                    targets,
                    probabilities,
                    multi_class=multi_class,
                    average="weighted",
                )
        except ValueError:
            # Handle edge cases (e.g., single class in batch)
            return 0.0

    @staticmethod
    def calculate_all_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Calculate all metrics at once.

        Args:
            predictions: Predicted class indices
            targets: Ground truth labels
            probabilities: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        calc = MetricsCalculator()

        metrics = {
            "accuracy": calc.calculate_accuracy(predictions, targets),
            "precision": calc.calculate_precision(predictions, targets),
            "recall": calc.calculate_recall(predictions, targets),
            "f1": calc.calculate_f1(predictions, targets),
            "roc_auc": calc.calculate_roc_auc(probabilities, targets),
        }

        return metrics


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update statistics.

        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
