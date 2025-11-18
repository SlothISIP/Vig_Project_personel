"""Train Predictive Maintenance Models.

Train and evaluate XGBoost and LSTM models for predictive maintenance.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predictive.feature_engineering import create_synthetic_failure_dataset
from src.predictive.models.xgboost_predictor import XGBoostFailurePredictor
from src.predictive.models.time_series_predictor import (
    RULPredictor,
    create_synthetic_rul_dataset,
)
from src.predictive.predictor import PredictiveMaintenanceSystem
from src.core.logging import get_logger

logger = get_logger(__name__)


def train_failure_predictor(save_path: Optional[Path] = None) -> XGBoostFailurePredictor:
    """
    Train XGBoost failure predictor.

    Args:
        save_path: Path to save model (optional)

    Returns:
        Trained model
    """
    logger.info("=" * 80)
    logger.info("TRAINING XGBOOST FAILURE PREDICTOR")
    logger.info("=" * 80)

    # Create synthetic dataset
    X, y = create_synthetic_failure_dataset(
        num_samples=5000, num_features=30, failure_rate=0.15
    )

    # Feature names
    feature_names = [
        f"sensor_{i // 6}_feature_{i % 6}" for i in range(30)
    ]

    # Train model
    model = XGBoostFailurePredictor(
        n_estimators=150, max_depth=6, learning_rate=0.05, threshold=0.5
    )

    metrics = model.train(X, y, feature_names=feature_names, validation_split=0.2)

    # Print metrics
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1']:.3f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['true_negatives']:4d}  FP: {metrics['false_positives']:4d}")
    print(f"  FN: {metrics['false_negatives']:4d}  TP: {metrics['true_positives']:4d}")

    # Feature importance
    importance = model.get_feature_importance()
    if importance:
        print(f"\nTop 10 Important Features:")
        sorted_features = sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for feat, imp in sorted_features:
            print(f"  {feat:<30} {imp:.4f}")

    print("=" * 80)

    # Save if requested
    if save_path:
        model.save(save_path)
        print(f"\nModel saved to {save_path}")

    return model


def train_rul_predictor(save_path: Optional[Path] = None) -> RULPredictor:
    """
    Train RUL predictor.

    Args:
        save_path: Path to save model (optional)

    Returns:
        Trained model
    """
    logger.info("=" * 80)
    logger.info("TRAINING RUL PREDICTOR (LSTM)")
    logger.info("=" * 80)

    # Create synthetic dataset
    X, y = create_synthetic_rul_dataset(
        num_samples=2000, sequence_length=50, num_features=5
    )

    # Train model
    model = RULPredictor(sequence_length=50, input_size=5)

    val_losses = model.train(X, y, epochs=30)

    # Print results
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Best Validation Loss: {min(val_losses):.4f}")

    # Test prediction
    test_sequence = X[0]
    prediction = model.predict_rul(test_sequence)

    print(f"\nSample Prediction:")
    print(f"  RUL:        {prediction.rul_hours:.1f} hours")
    print(f"  Confidence: {prediction.confidence_interval}")
    print(f"  Health:     {prediction.health_score:.2f}")
    print(f"  Trend:      {prediction.trend}")
    print("=" * 80)

    # Save if requested
    if save_path:
        model.save(save_path)
        print(f"\nModel saved to {save_path}")

    return model


def demo_integrated_system():
    """Demonstrate integrated predictive maintenance system."""
    logger.info("=" * 80)
    logger.info("INTEGRATED PREDICTIVE MAINTENANCE DEMO")
    logger.info("=" * 80)

    # Train models
    failure_model = train_failure_predictor()
    rul_model = train_rul_predictor()

    # Create integrated system
    system = PredictiveMaintenanceSystem(
        failure_predictor=failure_model,
        rul_predictor=rul_model,
        feature_window_size=100,
    )

    # Simulate sensor readings
    from src.digital_twin.simulation.sensor import create_standard_sensor_network

    machine_id = "Demo_Machine_01"
    sensor_network = create_standard_sensor_network(machine_id)

    # Generate readings
    print("\nGenerating sensor readings...")
    for i in range(150):
        readings = sensor_network.read_all()
        system.add_sensor_readings(machine_id, readings)

    # Get recommendation
    print("\n" + "=" * 80)
    print("MAINTENANCE RECOMMENDATION")
    print("=" * 80)

    recommendation = system.predict_maintenance(machine_id)

    if recommendation:
        print(f"Machine: {recommendation.machine_id}")
        print(f"Timestamp: {recommendation.timestamp}")
        print(f"\nFailure Prediction:")
        print(f"  Probability: {recommendation.failure_probability:.2%}")
        print(f"  Risk Level: {recommendation.failure_risk_level}")
        print(f"\nRUL Prediction:")
        if recommendation.remaining_useful_life_hours:
            print(f"  RUL: {recommendation.remaining_useful_life_hours:.1f} hours")
        print(f"  Health Score: {recommendation.health_score:.2%}")
        print(f"\nRecommendation:")
        print(f"  Urgency: {recommendation.urgency.value.upper()}")
        print(f"  Action: {recommendation.recommended_action}")
        print(f"  Estimated Downtime: {recommendation.estimated_downtime_hours} hours")

        if recommendation.contributing_factors:
            print(f"\nContributing Factors:")
            for factor in recommendation.contributing_factors:
                print(f"  - {factor}")

        print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train Predictive Maintenance Models"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["failure", "rul", "both", "demo"],
        default="demo",
        help="Model to train",
    )

    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("models/predictive"),
        help="Directory to save models",
    )

    args = parser.parse_args()

    # Create save directory
    args.save_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.model == "failure":
            train_failure_predictor(args.save_dir / "failure_predictor.pkl")

        elif args.model == "rul":
            train_rul_predictor(args.save_dir / "rul_predictor.pth")

        elif args.model == "both":
            train_failure_predictor(args.save_dir / "failure_predictor.pkl")
            print("\n\n")
            train_rul_predictor(args.save_dir / "rul_predictor.pth")

        elif args.model == "demo":
            demo_integrated_system()

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
