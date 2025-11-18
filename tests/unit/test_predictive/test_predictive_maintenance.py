"""Tests for predictive maintenance models."""

import pytest
import numpy as np
from pathlib import Path

from src.predictive.feature_engineering import (
    FeatureExtractor,
    extract_statistical_features,
    create_synthetic_failure_dataset,
)
from src.predictive.models.xgboost_predictor import XGBoostFailurePredictor
from src.predictive.models.time_series_predictor import (
    TimeSeriesPredictor,
    RULPredictor,
    create_synthetic_rul_dataset,
)
from src.predictive.predictor import (
    PredictiveMaintenanceSystem,
    MaintenanceUrgency,
)
from src.digital_twin.simulation.sensor import (
    IoTSensor,
    SensorType,
    SensorReading,
)


class TestFeatureExtraction:
    """Tests for feature extraction."""

    def test_extract_statistical_features(self):
        """Test statistical feature extraction."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        features = extract_statistical_features(values)

        assert "mean" in features
        assert "std" in features
        assert "min" in features
        assert "max" in features
        assert features["mean"] == 3.0
        assert features["min"] == 1.0
        assert features["max"] == 5.0

    def test_feature_extractor(self):
        """Test feature extractor."""
        extractor = FeatureExtractor(window_size=10)

        # Create sensor readings
        for i in range(15):
            reading = SensorReading(
                sensor_id="temp_01",
                sensor_type=SensorType.TEMPERATURE,
                value=float(65 + i),
                unit="°C",
                timestamp=None,
            )
            extractor.add_reading(reading)

        # Extract features
        features = extractor.extract_features("temp_01")

        assert features is not None
        assert features.mean > 0
        assert features.trend_slope > 0  # Should have upward trend

    def test_feature_vector(self):
        """Test feature vector extraction."""
        extractor = FeatureExtractor(window_size=20)

        # Add readings for two sensors
        for i in range(25):
            reading1 = SensorReading(
                sensor_id="temp_01",
                sensor_type=SensorType.TEMPERATURE,
                value=float(65 + i * 0.1),
                unit="°C",
                timestamp=None,
            )
            reading2 = SensorReading(
                sensor_id="vib_01",
                sensor_type=SensorType.VIBRATION,
                value=float(2.5 + i * 0.01),
                unit="mm/s",
                timestamp=None,
            )

            extractor.add_reading(reading1)
            extractor.add_reading(reading2)

        # Extract vector
        vector = extractor.extract_feature_vector(["temp_01", "vib_01"])

        assert vector is not None
        assert len(vector) == 20  # 2 sensors * 10 features each


class TestXGBoostPredictor:
    """Tests for XGBoost failure predictor."""

    def test_create_predictor(self):
        """Test predictor creation."""
        model = XGBoostFailurePredictor(n_estimators=10)

        assert model.n_estimators == 10
        assert not model.is_trained

    def test_train_predictor(self):
        """Test training predictor."""
        # Create small dataset
        X, y = create_synthetic_failure_dataset(
            num_samples=200, num_features=10, failure_rate=0.2
        )

        model = XGBoostFailurePredictor(n_estimators=10)
        metrics = model.train(X, y, validation_split=0.2)

        assert model.is_trained
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert metrics["accuracy"] > 0.5  # Should be better than random

    def test_predict(self):
        """Test making predictions."""
        X, y = create_synthetic_failure_dataset(num_samples=200, num_features=10)

        model = XGBoostFailurePredictor(n_estimators=10)
        model.train(X, y, validation_split=0.2)

        # Predict single sample
        prediction = model.predict(X[0])

        assert prediction.failure_probability >= 0
        assert prediction.failure_probability <= 1
        assert prediction.risk_level in ["low", "medium", "high"]


class TestRULPredictor:
    """Tests for RUL predictor."""

    def test_create_rul_predictor(self):
        """Test RUL predictor creation."""
        model = RULPredictor(sequence_length=20, input_size=3)

        assert model.sequence_length == 20
        assert model.input_size == 3

    def test_train_rul_predictor(self):
        """Test training RUL predictor."""
        # Create small dataset
        X, y = create_synthetic_rul_dataset(
            num_samples=100, sequence_length=20, num_features=3
        )

        model = RULPredictor(sequence_length=20, input_size=3)
        val_losses = model.train(X, y, epochs=5)

        assert len(val_losses) == 5
        assert all(loss > 0 for loss in val_losses)

    def test_predict_rul(self):
        """Test RUL prediction."""
        X, y = create_synthetic_rul_dataset(
            num_samples=100, sequence_length=20, num_features=3
        )

        model = RULPredictor(sequence_length=20, input_size=3)
        model.train(X, y, epochs=5)

        # Predict
        prediction = model.predict_rul(X[0])

        assert prediction.rul_hours >= 0
        assert 0 <= prediction.health_score <= 1
        assert prediction.trend in ["stable", "degrading", "critical"]


class TestPredictiveMaintenanceSystem:
    """Tests for integrated predictive maintenance system."""

    def test_create_system(self):
        """Test system creation."""
        system = PredictiveMaintenanceSystem()

        assert system is not None

    def test_add_sensor_readings(self):
        """Test adding sensor readings."""
        system = PredictiveMaintenanceSystem()

        readings = [
            SensorReading(
                sensor_id="temp_01",
                sensor_type=SensorType.TEMPERATURE,
                value=65.0,
                unit="°C",
                timestamp=None,
            )
        ]

        system.add_sensor_readings("Machine_01", readings)

        status = system.get_machine_status("Machine_01")
        assert status is not None
        assert status["machine_id"] == "Machine_01"

    def test_predict_with_trained_models(self):
        """Test prediction with trained models."""
        # Train small models
        X_failure, y_failure = create_synthetic_failure_dataset(
            num_samples=200, num_features=30
        )
        failure_model = XGBoostFailurePredictor(n_estimators=10)
        failure_model.train(X_failure, y_failure)

        # Create system
        system = PredictiveMaintenanceSystem(failure_predictor=failure_model)

        # Add readings
        machine_id = "Test_Machine"
        for i in range(150):
            readings = [
                SensorReading(
                    sensor_id=f"sensor_{j}",
                    sensor_type=SensorType.TEMPERATURE,
                    value=float(65 + i * 0.1 + j),
                    unit="°C",
                    timestamp=None,
                )
                for j in range(5)
            ]
            system.add_sensor_readings(machine_id, readings)

        # Get recommendation
        recommendation = system.predict_maintenance(machine_id)

        # Should get a recommendation
        assert recommendation is not None
        assert recommendation.machine_id == machine_id
        assert isinstance(recommendation.urgency, MaintenanceUrgency)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
