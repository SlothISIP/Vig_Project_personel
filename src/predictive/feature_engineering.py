"""Feature Engineering for Predictive Maintenance.

Extract meaningful features from sensor data for machine learning models.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

from src.digital_twin.simulation.sensor import SensorReading, SensorType
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimeSeriesFeatures:
    """Container for time-series features extracted from sensor data."""

    # Statistical features
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    iqr: float
    skewness: float
    kurtosis: float

    # Trend features
    trend_slope: float
    trend_intercept: float

    # Frequency features (if applicable)
    peak_frequency: Optional[float] = None
    num_peaks: int = 0

    # Change features
    rate_of_change: float = 0.0
    cumulative_sum: float = 0.0


def extract_statistical_features(values: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical features from a time series.

    Args:
        values: Array of sensor values

    Returns:
        Dictionary of feature name -> value
    """
    if len(values) == 0:
        return {}

    features = {}

    # Basic statistics
    features["mean"] = float(np.mean(values))
    features["std"] = float(np.std(values))
    features["min"] = float(np.min(values))
    features["max"] = float(np.max(values))
    features["median"] = float(np.median(values))
    features["range"] = features["max"] - features["min"]

    # Quartiles
    q25, q75 = np.percentile(values, [25, 75])
    features["q25"] = float(q25)
    features["q75"] = float(q75)
    features["iqr"] = float(q75 - q25)

    # Shape statistics
    features["skewness"] = float(stats.skew(values))
    features["kurtosis"] = float(stats.kurtosis(values))

    # Trend
    if len(values) > 1:
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        features["trend_slope"] = float(slope)
        features["trend_intercept"] = float(intercept)
        features["trend_r_squared"] = float(r_value**2)
    else:
        features["trend_slope"] = 0.0
        features["trend_intercept"] = float(values[0])
        features["trend_r_squared"] = 0.0

    # Rate of change
    if len(values) > 1:
        diff = np.diff(values)
        features["rate_of_change_mean"] = float(np.mean(diff))
        features["rate_of_change_std"] = float(np.std(diff))
        features["rate_of_change_max"] = float(np.max(np.abs(diff)))
    else:
        features["rate_of_change_mean"] = 0.0
        features["rate_of_change_std"] = 0.0
        features["rate_of_change_max"] = 0.0

    # Cumulative sum
    features["cumulative_sum"] = float(np.sum(values))

    # Peak detection
    peaks, _ = find_peaks(values)
    features["num_peaks"] = len(peaks)

    # Zero crossings (mean crossings)
    mean_centered = values - np.mean(values)
    zero_crossings = np.where(np.diff(np.sign(mean_centered)))[0]
    features["zero_crossings"] = len(zero_crossings)

    return features


class FeatureExtractor:
    """
    Extract features from sensor readings for predictive models.

    Converts raw sensor data into meaningful features for ML models.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize feature extractor.

        Args:
            window_size: Number of readings to use for feature extraction
        """
        self.window_size = window_size
        self._reading_buffer: Dict[str, List[SensorReading]] = {}

    def add_reading(self, reading: SensorReading) -> None:
        """
        Add a sensor reading to the buffer.

        Args:
            reading: Sensor reading to add
        """
        sensor_id = reading.sensor_id

        if sensor_id not in self._reading_buffer:
            self._reading_buffer[sensor_id] = []

        self._reading_buffer[sensor_id].append(reading)

        # Keep only the last window_size readings
        if len(self._reading_buffer[sensor_id]) > self.window_size:
            self._reading_buffer[sensor_id].pop(0)

    def add_readings(self, readings: List[SensorReading]) -> None:
        """
        Add multiple sensor readings.

        Args:
            readings: List of sensor readings
        """
        for reading in readings:
            self.add_reading(reading)

    def extract_features(self, sensor_id: str) -> Optional[TimeSeriesFeatures]:
        """
        Extract features for a specific sensor.

        Args:
            sensor_id: ID of sensor to extract features for

        Returns:
            TimeSeriesFeatures or None if insufficient data
        """
        if sensor_id not in self._reading_buffer:
            return None

        readings = self._reading_buffer[sensor_id]

        if len(readings) < 10:  # Minimum required
            return None

        # Extract values
        values = np.array([r.value for r in readings])

        # Statistical features
        stats_features = extract_statistical_features(values)

        # Create features object
        features = TimeSeriesFeatures(
            mean=stats_features["mean"],
            std=stats_features["std"],
            min=stats_features["min"],
            max=stats_features["max"],
            median=stats_features["median"],
            q25=stats_features["q25"],
            q75=stats_features["q75"],
            iqr=stats_features["iqr"],
            skewness=stats_features["skewness"],
            kurtosis=stats_features["kurtosis"],
            trend_slope=stats_features["trend_slope"],
            trend_intercept=stats_features["trend_intercept"],
            num_peaks=int(stats_features["num_peaks"]),
            rate_of_change=stats_features["rate_of_change_mean"],
            cumulative_sum=stats_features["cumulative_sum"],
        )

        return features

    def extract_all_features(self) -> Dict[str, TimeSeriesFeatures]:
        """
        Extract features for all sensors in the buffer.

        Returns:
            Dictionary of sensor_id -> features
        """
        features = {}

        for sensor_id in self._reading_buffer.keys():
            sensor_features = self.extract_features(sensor_id)
            if sensor_features is not None:
                features[sensor_id] = sensor_features

        return features

    def extract_feature_vector(
        self, sensor_ids: Optional[List[str]] = None
    ) -> Optional[np.ndarray]:
        """
        Extract a feature vector suitable for ML models.

        Args:
            sensor_ids: List of sensor IDs to include (None for all)

        Returns:
            Feature vector as numpy array, or None if insufficient data
        """
        if sensor_ids is None:
            sensor_ids = list(self._reading_buffer.keys())

        feature_values = []

        for sensor_id in sensor_ids:
            features = self.extract_features(sensor_id)

            if features is None:
                return None

            # Convert features to vector
            sensor_vector = [
                features.mean,
                features.std,
                features.min,
                features.max,
                features.median,
                features.iqr,
                features.skewness,
                features.kurtosis,
                features.trend_slope,
                features.rate_of_change,
            ]

            feature_values.extend(sensor_vector)

        return np.array(feature_values)

    def get_feature_names(
        self, sensor_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get feature names corresponding to feature vector.

        Args:
            sensor_ids: List of sensor IDs

        Returns:
            List of feature names
        """
        if sensor_ids is None:
            sensor_ids = list(self._reading_buffer.keys())

        feature_names = []

        for sensor_id in sensor_ids:
            sensor_features = [
                f"{sensor_id}_mean",
                f"{sensor_id}_std",
                f"{sensor_id}_min",
                f"{sensor_id}_max",
                f"{sensor_id}_median",
                f"{sensor_id}_iqr",
                f"{sensor_id}_skewness",
                f"{sensor_id}_kurtosis",
                f"{sensor_id}_trend_slope",
                f"{sensor_id}_rate_of_change",
            ]
            feature_names.extend(sensor_features)

        return feature_names

    def reset(self) -> None:
        """Clear all buffered readings."""
        self._reading_buffer.clear()

    def get_buffer_status(self) -> Dict[str, int]:
        """
        Get status of reading buffers.

        Returns:
            Dictionary of sensor_id -> number of readings
        """
        return {
            sensor_id: len(readings)
            for sensor_id, readings in self._reading_buffer.items()
        }


def create_synthetic_failure_dataset(
    num_samples: int = 1000, num_features: int = 10, failure_rate: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic dataset for testing failure prediction models.

    Args:
        num_samples: Number of samples to generate
        num_features: Number of features per sample
        failure_rate: Proportion of samples that are failures

    Returns:
        (X, y) where X is features and y is labels (0=normal, 1=failure)
    """
    # Generate normal samples
    num_normal = int(num_samples * (1 - failure_rate))
    num_failure = num_samples - num_normal

    # Normal condition: features centered around 0
    X_normal = np.random.randn(num_normal, num_features)

    # Failure condition: features with higher mean and variance
    X_failure = np.random.randn(num_failure, num_features) * 2.0 + 3.0

    # Combine
    X = np.vstack([X_normal, X_failure])
    y = np.array([0] * num_normal + [1] * num_failure)

    # Shuffle
    indices = np.random.permutation(num_samples)
    X = X[indices]
    y = y[indices]

    logger.info(
        f"Created synthetic dataset: {num_samples} samples, "
        f"{num_features} features, {failure_rate:.1%} failure rate"
    )

    return X, y
