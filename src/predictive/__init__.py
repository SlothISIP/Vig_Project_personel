"""Predictive Maintenance Module.

Machine learning models for predicting equipment failures and maintenance needs.
"""

from src.predictive.feature_engineering import (
    FeatureExtractor,
    TimeSeriesFeatures,
    extract_statistical_features,
)
from src.predictive.models.xgboost_predictor import (
    XGBoostFailurePredictor,
)
from src.predictive.models.time_series_predictor import (
    TimeSeriesPredictor,
    RULPredictor,
)
from src.predictive.predictor import (
    PredictiveMaintenanceSystem,
    MaintenanceRecommendation,
)

__all__ = [
    # Feature Engineering
    "FeatureExtractor",
    "TimeSeriesFeatures",
    "extract_statistical_features",
    # Models
    "XGBoostFailurePredictor",
    "TimeSeriesPredictor",
    "RULPredictor",
    # System
    "PredictiveMaintenanceSystem",
    "MaintenanceRecommendation",
]
