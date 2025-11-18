"""XGBoost-based Failure Prediction Model.

Uses XGBoost to predict equipment failures based on sensor features.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import pickle

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FailurePrediction:
    """Result of failure prediction."""

    failure_probability: float
    is_failure_predicted: bool
    confidence: float
    risk_level: str  # "low", "medium", "high"
    recommended_action: str


class XGBoostFailurePredictor:
    """
    XGBoost-based machine failure predictor.

    Predicts binary failure (0=normal, 1=failure) using sensor features.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        threshold: float = 0.5,
    ):
        """
        Initialize XGBoost failure predictor.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            threshold: Classification threshold
        """
        if xgb is None:
            raise ImportError("xgboost is not installed. Install with: pip install xgboost")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.threshold = threshold

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: Optional[List[str]] = None
        self.is_trained = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train the failure prediction model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=normal, 1=failure)
            feature_names: Optional feature names
            validation_split: Proportion of data for validation

        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training XGBoost model with {len(X)} samples")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names

        # Create model
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )

        # Train
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self.is_trained = True

        # Evaluate
        metrics = self.evaluate(X_val, y_val)

        logger.info(
            f"Model trained - Accuracy: {metrics['accuracy']:.3f}, "
            f"F1: {metrics['f1']:.3f}, AUC: {metrics['roc_auc']:.3f}"
        )

        return metrics

    def predict(self, X: np.ndarray) -> FailurePrediction:
        """
        Predict failure for a single sample.

        Args:
            X: Feature vector

        Returns:
            FailurePrediction object
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet")

        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Predict probability
        prob = self.model.predict_proba(X)[0, 1]  # Probability of failure

        # Binary prediction
        is_failure = prob >= self.threshold

        # Confidence (distance from threshold)
        confidence = abs(prob - self.threshold)

        # Risk level
        if prob < 0.3:
            risk_level = "low"
            action = "Continue normal operation"
        elif prob < 0.7:
            risk_level = "medium"
            action = "Schedule inspection within 48 hours"
        else:
            risk_level = "high"
            action = "Immediate maintenance required"

        return FailurePrediction(
            failure_probability=float(prob),
            is_failure_predicted=bool(is_failure),
            confidence=float(confidence),
            risk_level=risk_level,
            recommended_action=action,
        )

    def predict_batch(self, X: np.ndarray) -> List[FailurePrediction]:
        """
        Predict failures for multiple samples.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            List of FailurePrediction objects
        """
        predictions = []

        for i in range(len(X)):
            pred = self.predict(X[i])
            predictions.append(pred)

        return predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Dictionary of metrics
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet")

        # Predictions
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]

        # Metrics
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_prob)),
        }

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics["true_negatives"] = int(cm[0, 0])
        metrics["false_positives"] = int(cm[0, 1])
        metrics["false_negatives"] = int(cm[1, 0])
        metrics["true_positives"] = int(cm[1, 1])

        return metrics

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.

        Returns:
            Dictionary of feature_name -> importance
        """
        if not self.is_trained or self.model is None:
            return None

        importances = self.model.feature_importances_

        if self.feature_names is not None:
            return dict(zip(self.feature_names, importances))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}

    def save(self, filepath: Path) -> None:
        """
        Save model to file.

        Args:
            filepath: Path to save model
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Cannot save untrained model")

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Path) -> None:
        """
        Load model from file.

        Args:
            filepath: Path to model file
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.threshold = model_data["threshold"]
        self.n_estimators = model_data["n_estimators"]
        self.max_depth = model_data["max_depth"]
        self.learning_rate = model_data["learning_rate"]

        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")


def train_example_model() -> XGBoostFailurePredictor:
    """
    Train an example failure prediction model.

    Returns:
        Trained XGBoostFailurePredictor
    """
    from src.predictive.feature_engineering import create_synthetic_failure_dataset

    # Create synthetic data
    X, y = create_synthetic_failure_dataset(
        num_samples=2000, num_features=30, failure_rate=0.15
    )

    # Create feature names
    feature_names = [f"sensor_{i}_feature_{j}" for i in range(5) for j in range(6)]

    # Train model
    model = XGBoostFailurePredictor(
        n_estimators=100, max_depth=5, learning_rate=0.1, threshold=0.5
    )

    metrics = model.train(X, y, feature_names=feature_names, validation_split=0.2)

    logger.info("Example model trained:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value}")

    return model
