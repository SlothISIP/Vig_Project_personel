"""Time-Series based Predictive Models.

LSTM-based models for time-series prediction and RUL estimation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RULPrediction:
    """Remaining Useful Life prediction result."""

    rul_hours: float
    confidence_interval: Tuple[float, float]  # (lower, upper)
    health_score: float  # 0-1, current health
    trend: str  # "improving", "stable", "degrading"


class TimeSeriesDataset(Dataset):
    """Dataset for time-series sequences."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset.

        Args:
            sequences: (n_samples, sequence_length, n_features)
            targets: (n_samples,) target values
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class LSTMModel(nn.Module):
    """LSTM-based time-series prediction model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            (batch_size, 1) predictions
        """
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Take last timestep
        last_out = lstm_out[:, -1, :]

        # Fully connected
        out = self.fc(last_out)

        return out


class TimeSeriesPredictor:
    """
    General time-series predictor using LSTM.

    Can be used for various time-series prediction tasks.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
    ):
        """
        Initialize predictor.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Model
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Loss
        self.criterion = nn.MSELoss()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.is_trained = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> List[float]:
        """
        Train the model.

        Args:
            X: Sequences (n_samples, sequence_length, n_features)
            y: Targets (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split

        Returns:
            List of validation losses
        """
        logger.info(f"Training LSTM model with {len(X)} samples")

        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Training loop
        val_losses = []

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0

            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                # Forward
                predictions = self.model(sequences)
                loss = self.criterion(predictions, targets)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device).unsqueeze(1)

                    predictions = self.model(sequences)
                    loss = self.criterion(predictions, targets)

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        self.is_trained = True
        logger.info("Training completed")

        return val_losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Sequences (n_samples, sequence_length, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")

        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy().squeeze()

        return predictions

    def save(self, filepath: Path) -> None:
        """Save model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
            },
            filepath,
        )

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Path) -> None:
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.input_size = checkpoint["input_size"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.learning_rate = checkpoint["learning_rate"]

        # Recreate model
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Recreate optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")


class RULPredictor:
    """
    Remaining Useful Life (RUL) predictor.

    Estimates how many hours until equipment failure.
    """

    def __init__(self, sequence_length: int = 50, input_size: int = 5):
        """
        Initialize RUL predictor.

        Args:
            sequence_length: Length of input sequences
            input_size: Number of sensor features
        """
        self.sequence_length = sequence_length
        self.input_size = input_size

        self.predictor = TimeSeriesPredictor(
            input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2
        )

    def train(
        self, sequences: np.ndarray, rul_targets: np.ndarray, epochs: int = 50
    ) -> List[float]:
        """
        Train RUL predictor.

        Args:
            sequences: (n_samples, sequence_length, n_features)
            rul_targets: (n_samples,) RUL in hours
            epochs: Number of epochs

        Returns:
            Validation losses
        """
        return self.predictor.train(sequences, rul_targets, epochs=epochs)

    def predict_rul(self, sequence: np.ndarray) -> RULPrediction:
        """
        Predict RUL for a sequence.

        Args:
            sequence: (sequence_length, n_features)

        Returns:
            RULPrediction object
        """
        # Reshape if needed
        if sequence.ndim == 2:
            sequence = sequence.reshape(1, *sequence.shape)

        # Predict
        rul = self.predictor.predict(sequence)[0]

        # Calculate health score (normalized)
        # Assume max RUL is 1000 hours
        max_rul = 1000.0
        health_score = min(1.0, max(0.0, rul / max_rul))

        # Confidence interval (simplified - assume ±20%)
        confidence_lower = max(0, rul * 0.8)
        confidence_upper = rul * 1.2

        # Trend detection (simplified)
        # In practice, would compare with previous predictions
        if rul > 500:
            trend = "stable"
        elif rul > 100:
            trend = "degrading"
        else:
            trend = "critical"

        return RULPrediction(
            rul_hours=float(rul),
            confidence_interval=(float(confidence_lower), float(confidence_upper)),
            health_score=float(health_score),
            trend=trend,
        )

    def save(self, filepath: Path) -> None:
        """Save model."""
        self.predictor.save(filepath)

    def load(self, filepath: Path) -> None:
        """Load model."""
        self.predictor.load(filepath)


def create_synthetic_rul_dataset(
    num_samples: int = 1000, sequence_length: int = 50, num_features: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic RUL dataset for testing.

    Args:
        num_samples: Number of samples
        sequence_length: Length of sequences
        num_features: Number of features

    Returns:
        (sequences, rul_targets)
    """
    sequences = []
    rul_targets = []

    for _ in range(num_samples):
        # Simulate degradation
        initial_rul = np.random.uniform(100, 1000)
        current_rul = initial_rul

        # Generate sequence with degrading trend
        sequence = []
        for t in range(sequence_length):
            # Features degrade over time
            degradation = (sequence_length - t) / sequence_length
            features = np.random.randn(num_features) + degradation * 2
            sequence.append(features)

            # RUL decreases
            current_rul -= np.random.uniform(1, 5)

        sequences.append(sequence)
        rul_targets.append(max(0, current_rul))

    X = np.array(sequences)
    y = np.array(rul_targets)

    logger.info(
        f"Created synthetic RUL dataset: {num_samples} samples, "
        f"sequence_length={sequence_length}, features={num_features}"
    )

    return X, y
