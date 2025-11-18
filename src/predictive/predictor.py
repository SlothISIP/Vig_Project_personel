"""Integrated Predictive Maintenance System.

Combines multiple prediction models for comprehensive maintenance recommendations.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any
from pathlib import Path

import numpy as np

from src.predictive.feature_engineering import FeatureExtractor
from src.predictive.models.xgboost_predictor import (
    XGBoostFailurePredictor,
    FailurePrediction,
)
from src.predictive.models.time_series_predictor import RULPredictor, RULPrediction
from src.digital_twin.simulation.sensor import SensorReading
from src.core.logging import get_logger

logger = get_logger(__name__)


class MaintenanceUrgency(Enum):
    """Maintenance urgency levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MaintenanceRecommendation:
    """Comprehensive maintenance recommendation."""

    machine_id: str
    timestamp: datetime

    # Failure prediction
    failure_probability: float
    failure_risk_level: str

    # RUL prediction
    remaining_useful_life_hours: Optional[float]
    health_score: float

    # Recommendation
    urgency: MaintenanceUrgency
    recommended_action: str
    estimated_downtime_hours: float
    maintenance_window: Optional[tuple[datetime, datetime]]

    # Additional info
    confidence: float
    contributing_factors: List[str]


class PredictiveMaintenanceSystem:
    """
    Integrated predictive maintenance system.

    Combines:
    - Feature extraction from sensors
    - XGBoost failure prediction
    - LSTM RUL prediction
    - Comprehensive recommendations
    """

    def __init__(
        self,
        failure_predictor: Optional[XGBoostFailurePredictor] = None,
        rul_predictor: Optional[RULPredictor] = None,
        feature_window_size: int = 100,
    ):
        """
        Initialize predictive maintenance system.

        Args:
            failure_predictor: XGBoost failure predictor (optional)
            rul_predictor: RUL predictor (optional)
            feature_window_size: Window size for feature extraction
        """
        self.failure_predictor = failure_predictor
        self.rul_predictor = rul_predictor
        self.feature_extractor = FeatureExtractor(window_size=feature_window_size)

        # Machine-specific extractors
        self._machine_extractors: Dict[str, FeatureExtractor] = {}

    def add_sensor_readings(
        self, machine_id: str, readings: List[SensorReading]
    ) -> None:
        """
        Add sensor readings for a machine.

        Args:
            machine_id: Machine identifier
            readings: List of sensor readings
        """
        # Get or create extractor for this machine
        if machine_id not in self._machine_extractors:
            self._machine_extractors[machine_id] = FeatureExtractor(
                window_size=self.feature_extractor.window_size
            )

        extractor = self._machine_extractors[machine_id]
        extractor.add_readings(readings)

    def predict_maintenance(
        self, machine_id: str, sensor_ids: Optional[List[str]] = None
    ) -> Optional[MaintenanceRecommendation]:
        """
        Generate comprehensive maintenance recommendation.

        Args:
            machine_id: Machine to predict for
            sensor_ids: Sensor IDs to use (None for all)

        Returns:
            MaintenanceRecommendation or None if insufficient data
        """
        if machine_id not in self._machine_extractors:
            logger.warning(f"No data for machine {machine_id}")
            return None

        extractor = self._machine_extractors[machine_id]

        # Extract features
        feature_vector = extractor.extract_feature_vector(sensor_ids)

        if feature_vector is None:
            logger.warning(f"Insufficient data for machine {machine_id}")
            return None

        # Failure prediction
        failure_pred: Optional[FailurePrediction] = None
        if self.failure_predictor and self.failure_predictor.is_trained:
            try:
                failure_pred = self.failure_predictor.predict(feature_vector)
            except Exception as e:
                logger.error(f"Failure prediction error: {e}")

        # RUL prediction
        rul_pred: Optional[RULPrediction] = None
        rul_hours: Optional[float] = None
        health_score = 1.0

        if self.rul_predictor and self.rul_predictor.predictor.is_trained:
            try:
                # For RUL, we need a sequence
                # Simplification: use recent features as proxy
                # In practice, would need actual time-series data
                buffer_status = extractor.get_buffer_status()
                if buffer_status:
                    # Get first sensor's readings as sequence
                    first_sensor = list(buffer_status.keys())[0]
                    readings = extractor._reading_buffer[first_sensor]

                    if len(readings) >= 10:
                        # Create simple sequence from values
                        values = np.array([r.value for r in readings[-50:]])
                        # Reshape to (1, seq_len, 1)
                        seq = values.reshape(1, -1, 1)

                        # Pad if needed
                        if seq.shape[1] < 50:
                            pad_width = ((0, 0), (50 - seq.shape[1], 0), (0, 0))
                            seq = np.pad(seq, pad_width, mode="edge")

                        rul_pred = self.rul_predictor.predict_rul(seq[0])
                        rul_hours = rul_pred.rul_hours
                        health_score = rul_pred.health_score

            except Exception as e:
                logger.error(f"RUL prediction error: {e}")

        # Combine predictions
        urgency, action, downtime = self._determine_urgency(
            failure_pred, rul_pred
        )

        # Maintenance window
        maintenance_window = None
        if urgency in [MaintenanceUrgency.HIGH, MaintenanceUrgency.CRITICAL]:
            now = datetime.now()
            # Critical: within 24 hours
            # High: within 48 hours
            hours = 24 if urgency == MaintenanceUrgency.CRITICAL else 48
            maintenance_window = (now, now + timedelta(hours=hours))

        # Contributing factors
        factors = self._identify_factors(extractor, sensor_ids)

        # Confidence (average of available predictions)
        confidence = 0.0
        count = 0
        if failure_pred:
            confidence += failure_pred.confidence
            count += 1
        if rul_pred:
            # Use health score as proxy for confidence
            confidence += health_score
            count += 1

        if count > 0:
            confidence /= count

        return MaintenanceRecommendation(
            machine_id=machine_id,
            timestamp=datetime.now(),
            failure_probability=(
                failure_pred.failure_probability if failure_pred else 0.0
            ),
            failure_risk_level=(
                failure_pred.risk_level if failure_pred else "unknown"
            ),
            remaining_useful_life_hours=rul_hours,
            health_score=health_score,
            urgency=urgency,
            recommended_action=action,
            estimated_downtime_hours=downtime,
            maintenance_window=maintenance_window,
            confidence=confidence,
            contributing_factors=factors,
        )

    def _determine_urgency(
        self,
        failure_pred: Optional[FailurePrediction],
        rul_pred: Optional[RULPrediction],
    ) -> tuple[MaintenanceUrgency, str, float]:
        """
        Determine maintenance urgency based on predictions.

        Returns:
            (urgency, recommended_action, estimated_downtime_hours)
        """
        # Start with defaults
        urgency = MaintenanceUrgency.LOW
        action = "Continue monitoring"
        downtime = 0.0

        # Consider failure prediction
        if failure_pred:
            if failure_pred.failure_probability > 0.9:
                urgency = MaintenanceUrgency.CRITICAL
                action = "Immediate shutdown and maintenance required"
                downtime = 8.0
            elif failure_pred.failure_probability > 0.7:
                urgency = MaintenanceUrgency.HIGH
                action = "Schedule urgent maintenance within 24 hours"
                downtime = 4.0
            elif failure_pred.failure_probability > 0.5:
                urgency = MaintenanceUrgency.MEDIUM
                action = "Schedule maintenance within 48 hours"
                downtime = 2.0

        # Consider RUL prediction
        if rul_pred:
            if rul_pred.rul_hours < 24:
                if urgency.value < MaintenanceUrgency.CRITICAL.value:
                    urgency = MaintenanceUrgency.CRITICAL
                    action = "Critical: Less than 24 hours to failure"
                    downtime = 8.0
            elif rul_pred.rul_hours < 72:
                if urgency.value < MaintenanceUrgency.HIGH.value:
                    urgency = MaintenanceUrgency.HIGH
                    action = "High priority: Less than 3 days to failure"
                    downtime = 4.0
            elif rul_pred.rul_hours < 168:  # 1 week
                if urgency.value < MaintenanceUrgency.MEDIUM.value:
                    urgency = MaintenanceUrgency.MEDIUM
                    action = "Medium priority: Less than 1 week to failure"
                    downtime = 2.0

        return urgency, action, downtime

    def _identify_factors(
        self, extractor: FeatureExtractor, sensor_ids: Optional[List[str]]
    ) -> List[str]:
        """Identify contributing factors from sensor data."""
        factors = []

        all_features = extractor.extract_all_features()

        for sensor_id, features in all_features.items():
            # High std deviation
            if features.std > features.mean * 0.5:
                factors.append(f"{sensor_id}: High variability detected")

            # Strong upward trend
            if features.trend_slope > features.mean * 0.1:
                factors.append(f"{sensor_id}: Increasing trend")

            # Many peaks
            if features.num_peaks > 10:
                factors.append(f"{sensor_id}: Frequent oscillations")

        return factors[:5]  # Return top 5

    def get_machine_status(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a machine.

        Args:
            machine_id: Machine identifier

        Returns:
            Status dictionary or None
        """
        if machine_id not in self._machine_extractors:
            return None

        extractor = self._machine_extractors[machine_id]

        buffer_status = extractor.get_buffer_status()
        all_features = extractor.extract_all_features()

        return {
            "machine_id": machine_id,
            "sensors_active": len(buffer_status),
            "total_readings": sum(buffer_status.values()),
            "sensor_status": buffer_status,
            "feature_summary": {
                sensor_id: {
                    "mean": features.mean,
                    "std": features.std,
                    "trend_slope": features.trend_slope,
                }
                for sensor_id, features in all_features.items()
            },
        }

    def save_models(self, directory: Path) -> None:
        """
        Save all models.

        Args:
            directory: Directory to save models
        """
        directory.mkdir(parents=True, exist_ok=True)

        if self.failure_predictor and self.failure_predictor.is_trained:
            self.failure_predictor.save(directory / "failure_predictor.pkl")

        if self.rul_predictor and self.rul_predictor.predictor.is_trained:
            self.rul_predictor.save(directory / "rul_predictor.pth")

        logger.info(f"Models saved to {directory}")

    def load_models(self, directory: Path) -> None:
        """
        Load models from directory.

        Args:
            directory: Directory containing models
        """
        failure_path = directory / "failure_predictor.pkl"
        if failure_path.exists():
            if self.failure_predictor is None:
                self.failure_predictor = XGBoostFailurePredictor()
            self.failure_predictor.load(failure_path)

        rul_path = directory / "rul_predictor.pth"
        if rul_path.exists():
            if self.rul_predictor is None:
                self.rul_predictor = RULPredictor()
            self.rul_predictor.load(rul_path)

        logger.info(f"Models loaded from {directory}")
