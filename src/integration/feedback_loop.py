"""
Vision-Digital Twin Feedback Loop Integration.

This module creates a closed-loop system connecting:
1. Vision AI (defect detection + XAI)
2. Digital Twin (factory simulation)
3. RL Scheduling (optimization)

Key Innovation:
- Defect detection results feed back into Digital Twin
- Digital Twin uses defect patterns to update machine health models
- RL agent learns from this integrated environment
- Enables predictive quality control and proactive maintenance

Architecture:
    Vision AI ──> Defect Detection ──> Explainability
         │                              │
         v                              v
    Feedback Loop  <── Root Cause ──< Pattern Analysis
         │
         v
    Digital Twin ──> Machine Health Update
         │
         v
    RL Agent ────> Optimal Scheduling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import threading

from src.digital_twin.simulation.simulator import FactorySimulator, SimulationConfig
from src.digital_twin.simulation.production_line import (
    ProductionLine,
    WorkStation,
    Product,
    ProductStatus,
)
from src.digital_twin.state.machine_state import MachineStatus
from src.digital_twin.events.event_bus import get_event_bus
from src.digital_twin.events.event_types import Event, EventType
from src.vision.explainability.defect_explainer import (
    DefectExplainer,
    ExplanationResult,
    DefectType,
    DefectSeverity,
    RootCauseHypothesis,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


class FeedbackType(Enum):
    """Types of feedback signals."""

    DEFECT_DETECTED = "defect_detected"
    QUALITY_UPDATE = "quality_update"
    HEALTH_DEGRADATION = "health_degradation"
    MAINTENANCE_REQUIRED = "maintenance_required"
    PROCESS_ADJUSTMENT = "process_adjustment"
    PATTERN_LEARNED = "pattern_learned"


@dataclass
class DefectFeedback:
    """Feedback data from defect detection."""

    timestamp: datetime
    product_id: str
    station_id: str
    is_defect: bool
    defect_probability: float
    defect_type: DefectType
    severity: DefectSeverity
    affected_area: float
    root_causes: List[RootCauseHypothesis]
    explanation_result: Optional[ExplanationResult] = None


@dataclass
class QualityMetrics:
    """Aggregated quality metrics for feedback."""

    station_id: str
    window_size: int  # Number of samples
    defect_rate: float
    defect_rate_trend: float  # Positive = increasing
    most_common_defect: DefectType
    average_severity: float
    health_score_impact: float


@dataclass
class StationHealthModel:
    """Health model for a manufacturing station."""

    station_id: str
    base_health: float = 1.0
    current_health: float = 1.0
    defect_impact_factor: float = 0.02  # Health loss per defect
    recovery_rate: float = 0.001  # Health recovery per good product
    maintenance_threshold: float = 0.6
    critical_threshold: float = 0.3

    # Historical data
    defect_history: deque = field(default_factory=lambda: deque(maxlen=100))
    health_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    def update_from_defect(self, defect_feedback: DefectFeedback) -> float:
        """
        Update health based on defect feedback.

        Returns:
            New health score
        """
        if defect_feedback.is_defect:
            # Calculate health impact based on severity
            severity_multiplier = {
                DefectSeverity.MINOR: 0.5,
                DefectSeverity.MODERATE: 1.0,
                DefectSeverity.MAJOR: 2.0,
                DefectSeverity.CRITICAL: 3.0,
            }.get(defect_feedback.severity, 1.0)

            # Area-weighted impact
            area_factor = 1.0 + defect_feedback.affected_area / 100.0

            # Total health loss
            health_loss = (
                self.defect_impact_factor *
                severity_multiplier *
                area_factor *
                defect_feedback.defect_probability
            )

            self.current_health = max(0.1, self.current_health - health_loss)

            # Record defect
            self.defect_history.append({
                "timestamp": defect_feedback.timestamp,
                "type": defect_feedback.defect_type.value,
                "severity": defect_feedback.severity.value,
                "impact": health_loss,
            })
        else:
            # Good product - slight recovery
            self.current_health = min(
                self.base_health,
                self.current_health + self.recovery_rate
            )

        # Record health
        self.health_history.append({
            "timestamp": datetime.now(),
            "health": self.current_health,
        })

        return self.current_health

    def needs_maintenance(self) -> bool:
        """Check if maintenance is needed."""
        return self.current_health < self.maintenance_threshold

    def is_critical(self) -> bool:
        """Check if station is in critical state."""
        return self.current_health < self.critical_threshold

    def perform_maintenance(self) -> None:
        """Reset health after maintenance."""
        self.current_health = self.base_health

    def get_defect_rate(self, window: int = 50) -> float:
        """Calculate recent defect rate."""
        if len(self.defect_history) == 0:
            return 0.0

        recent = list(self.defect_history)[-window:]
        return len(recent) / window if recent else 0.0


class VisionDigitalTwinBridge:
    """
    Bridge component connecting Vision AI to Digital Twin.

    Translates defect detection results into Digital Twin updates.
    """

    def __init__(
        self,
        station_mapping: Dict[str, str] = None,
    ):
        """
        Initialize bridge.

        Args:
            station_mapping: Map from product location to station ID
                           (default: inspection station)
        """
        self.station_mapping = station_mapping or {}
        self.default_station = "inspection"

        # Statistics
        self.total_products_inspected = 0
        self.total_defects_found = 0

    def translate_to_feedback(
        self,
        product_id: str,
        explanation: ExplanationResult,
        station_id: Optional[str] = None,
    ) -> DefectFeedback:
        """
        Translate explanation result to feedback format.

        Args:
            product_id: ID of inspected product
            explanation: Vision AI explanation result
            station_id: Override station ID

        Returns:
            DefectFeedback for Digital Twin
        """
        # Determine responsible station
        if station_id is None:
            # Try to infer from root causes
            if explanation.root_cause_hypotheses:
                primary_cause = explanation.root_cause_hypotheses[0]
                station_id = primary_cause.related_station or self.default_station
            else:
                station_id = self.default_station

        self.total_products_inspected += 1
        if explanation.is_defect:
            self.total_defects_found += 1

        return DefectFeedback(
            timestamp=datetime.now(),
            product_id=product_id,
            station_id=station_id,
            is_defect=explanation.is_defect,
            defect_probability=explanation.defect_probability,
            defect_type=explanation.defect_type,
            severity=explanation.severity,
            affected_area=explanation.affected_area_percent,
            root_causes=explanation.root_cause_hypotheses,
            explanation_result=explanation,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "total_inspected": self.total_products_inspected,
            "total_defects": self.total_defects_found,
            "defect_rate": (
                self.total_defects_found / self.total_products_inspected
                if self.total_products_inspected > 0
                else 0.0
            ),
        }


class DigitalTwinFeedbackController:
    """
    Controller that applies feedback to Digital Twin state.

    Updates machine health, triggers maintenance, adjusts parameters.
    """

    def __init__(
        self,
        production_lines: List[ProductionLine],
    ):
        """
        Initialize controller.

        Args:
            production_lines: Production lines to control
        """
        self.production_lines = {line.line_id: line for line in production_lines}

        # Health models for each station
        self.health_models: Dict[str, StationHealthModel] = {}
        for line in production_lines:
            for station_id, station in line.stations.items():
                self.health_models[station_id] = StationHealthModel(
                    station_id=station_id,
                    base_health=station.machine_state.health_score,
                )

        # Feedback queue for batch processing
        self.feedback_queue: deque = deque(maxlen=1000)

        # Event bus for notifications
        self.event_bus = get_event_bus()

        # Callbacks
        self._maintenance_callbacks: List[Callable] = []
        self._quality_callbacks: List[Callable] = []

        logger.info(
            f"DigitalTwinFeedbackController initialized for "
            f"{len(self.health_models)} stations"
        )

    def process_feedback(self, feedback: DefectFeedback) -> Dict[str, Any]:
        """
        Process defect feedback and update Digital Twin.

        Args:
            feedback: Defect feedback from Vision AI

        Returns:
            Dict with update results
        """
        results = {
            "station_updated": feedback.station_id,
            "health_before": None,
            "health_after": None,
            "maintenance_triggered": False,
            "actions": [],
        }

        # Get health model
        if feedback.station_id not in self.health_models:
            logger.warning(f"Unknown station: {feedback.station_id}")
            return results

        health_model = self.health_models[feedback.station_id]
        results["health_before"] = health_model.current_health

        # Update health model
        new_health = health_model.update_from_defect(feedback)
        results["health_after"] = new_health

        # Update actual station state
        self._update_station_state(feedback.station_id, new_health)

        # Check maintenance requirements
        if health_model.is_critical():
            results["maintenance_triggered"] = True
            results["actions"].append("CRITICAL: Immediate maintenance required")
            self._trigger_maintenance_alert(feedback.station_id, "critical")

        elif health_model.needs_maintenance():
            results["actions"].append("WARNING: Schedule maintenance soon")
            self._trigger_maintenance_alert(feedback.station_id, "warning")

        # Process root causes and generate actions
        if feedback.root_causes:
            for hypothesis in feedback.root_causes[:2]:  # Top 2 causes
                for action in hypothesis.recommended_actions:
                    results["actions"].append(f"RECOMMENDED: {action}")

        # Store in queue
        self.feedback_queue.append(feedback)

        # Publish event
        self._publish_feedback_event(feedback, results)

        return results

    def _update_station_state(self, station_id: str, health: float) -> None:
        """Update station state in production line."""
        for line in self.production_lines.values():
            if station_id in line.stations:
                station = line.stations[station_id]

                # Update machine state health
                station.machine_state.health_score = health

                # Update status based on health
                if health < 0.3:
                    station.machine_state.update_status(MachineStatus.ERROR)
                elif health < 0.6:
                    station.machine_state.update_status(MachineStatus.WARNING)
                elif station.machine_state.status == MachineStatus.WARNING:
                    station.machine_state.update_status(MachineStatus.IDLE)

                # Adjust defect rate based on health degradation
                # Lower health = higher effective defect rate
                health_factor = 2.0 - health  # 1.0 at full health, 2.0 at zero
                station.defect_rate = min(0.3, station.defect_rate * health_factor)

                break

    def _trigger_maintenance_alert(self, station_id: str, severity: str) -> None:
        """Trigger maintenance alert."""
        for callback in self._maintenance_callbacks:
            try:
                callback(station_id, severity)
            except Exception as e:
                logger.error(f"Maintenance callback error: {e}")

    def _publish_feedback_event(
        self,
        feedback: DefectFeedback,
        results: Dict[str, Any],
    ) -> None:
        """Publish feedback processing event."""
        event = Event(
            event_id=f"fb_{datetime.now().timestamp()}",
            event_type=EventType.QUALITY_CHECK,
            source="FeedbackController",
            timestamp=datetime.now(),
            data={
                "product_id": feedback.product_id,
                "station_id": feedback.station_id,
                "is_defect": feedback.is_defect,
                "defect_type": feedback.defect_type.value,
                "health_after": results["health_after"],
                "maintenance_triggered": results["maintenance_triggered"],
            },
        )
        self.event_bus.publish(event)

    def register_maintenance_callback(self, callback: Callable) -> None:
        """Register callback for maintenance alerts."""
        self._maintenance_callbacks.append(callback)

    def register_quality_callback(self, callback: Callable) -> None:
        """Register callback for quality updates."""
        self._quality_callbacks.append(callback)

    def get_quality_metrics(self, station_id: str) -> Optional[QualityMetrics]:
        """Get quality metrics for a station."""
        if station_id not in self.health_models:
            return None

        health_model = self.health_models[station_id]
        recent_defects = list(health_model.defect_history)

        if not recent_defects:
            return QualityMetrics(
                station_id=station_id,
                window_size=0,
                defect_rate=0.0,
                defect_rate_trend=0.0,
                most_common_defect=DefectType.UNKNOWN,
                average_severity=0.0,
                health_score_impact=0.0,
            )

        # Calculate metrics
        defect_rate = health_model.get_defect_rate()

        # Calculate trend (compare recent vs older)
        if len(recent_defects) >= 20:
            recent_count = len([d for d in recent_defects[-10:]])
            older_count = len([d for d in recent_defects[-20:-10]])
            trend = (recent_count - older_count) / 10.0
        else:
            trend = 0.0

        # Most common defect type
        type_counts: Dict[str, int] = {}
        severity_sum = 0.0
        impact_sum = 0.0

        for defect in recent_defects:
            dtype = defect.get("type", "unknown")
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

            severity_map = {
                "minor": 1, "moderate": 2, "major": 3, "critical": 4
            }
            severity_sum += severity_map.get(defect.get("severity", "minor"), 1)
            impact_sum += defect.get("impact", 0)

        most_common = max(type_counts, key=type_counts.get) if type_counts else "unknown"
        try:
            most_common_type = DefectType(most_common)
        except ValueError:
            most_common_type = DefectType.UNKNOWN

        return QualityMetrics(
            station_id=station_id,
            window_size=len(recent_defects),
            defect_rate=defect_rate,
            defect_rate_trend=trend,
            most_common_defect=most_common_type,
            average_severity=severity_sum / len(recent_defects) if recent_defects else 0,
            health_score_impact=impact_sum,
        )

    def get_all_health_scores(self) -> Dict[str, float]:
        """Get health scores for all stations."""
        return {
            station_id: model.current_health
            for station_id, model in self.health_models.items()
        }


class IntegratedFeedbackLoop:
    """
    Complete integrated feedback loop system.

    Orchestrates the entire Vision AI -> Digital Twin -> RL cycle.
    """

    def __init__(
        self,
        defect_explainer: DefectExplainer,
        production_lines: List[ProductionLine],
        enable_async: bool = True,
    ):
        """
        Initialize integrated feedback loop.

        Args:
            defect_explainer: Vision AI explainer
            production_lines: Digital Twin production lines
            enable_async: Enable asynchronous processing
        """
        self.explainer = defect_explainer
        self.production_lines = production_lines
        self.enable_async = enable_async

        # Initialize components
        self.bridge = VisionDigitalTwinBridge()
        self.controller = DigitalTwinFeedbackController(production_lines)

        # Statistics
        self.loop_iterations = 0
        self.total_processing_time_ms = 0.0

        # Async executor
        if enable_async:
            self.executor = ThreadPoolExecutor(max_workers=2)
        else:
            self.executor = None

        # Thread safety
        self._lock = threading.Lock()

        logger.info("IntegratedFeedbackLoop initialized")

    def process_inspection(
        self,
        product_id: str,
        image: torch.Tensor,
        original_image: Optional[np.ndarray] = None,
    ) -> Tuple[ExplanationResult, Dict[str, Any]]:
        """
        Process a product inspection through the full feedback loop.

        Args:
            product_id: Product being inspected
            image: Preprocessed image tensor
            original_image: Original image for visualization

        Returns:
            Tuple of (ExplanationResult, feedback_results)
        """
        import time
        start = time.time()

        # Step 1: Vision AI explanation
        explanation = self.explainer.explain(
            image,
            original_image=original_image,
            return_overlay=original_image is not None,
        )

        # Step 2: Translate to feedback
        feedback = self.bridge.translate_to_feedback(
            product_id=product_id,
            explanation=explanation,
        )

        # Step 3: Update Digital Twin
        with self._lock:
            results = self.controller.process_feedback(feedback)

        # Update statistics
        self.loop_iterations += 1
        self.total_processing_time_ms += (time.time() - start) * 1000

        return explanation, results

    def process_inspection_async(
        self,
        product_id: str,
        image: torch.Tensor,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Process inspection asynchronously.

        Args:
            product_id: Product being inspected
            image: Preprocessed image tensor
            callback: Function to call with results
        """
        if not self.enable_async or self.executor is None:
            # Fallback to sync
            result = self.process_inspection(product_id, image)
            if callback:
                callback(result)
            return

        def _async_process():
            try:
                result = self.process_inspection(product_id, image)
                if callback:
                    callback(result)
            except Exception as e:
                logger.error(f"Async inspection error: {e}")
                if callback:
                    callback((None, {"error": str(e)}))

        self.executor.submit(_async_process)

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health_scores = self.controller.get_all_health_scores()

        return {
            "loop_iterations": self.loop_iterations,
            "avg_processing_time_ms": (
                self.total_processing_time_ms / self.loop_iterations
                if self.loop_iterations > 0
                else 0.0
            ),
            "bridge_stats": self.bridge.get_statistics(),
            "station_health": health_scores,
            "min_health": min(health_scores.values()) if health_scores else 1.0,
            "avg_health": np.mean(list(health_scores.values())) if health_scores else 1.0,
            "stations_needing_maintenance": [
                station_id
                for station_id, model in self.controller.health_models.items()
                if model.needs_maintenance()
            ],
            "critical_stations": [
                station_id
                for station_id, model in self.controller.health_models.items()
                if model.is_critical()
            ],
        }

    def get_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.bridge.get_statistics(),
            "station_metrics": {},
        }

        for station_id in self.controller.health_models:
            metrics = self.controller.get_quality_metrics(station_id)
            if metrics:
                report["station_metrics"][station_id] = {
                    "defect_rate": metrics.defect_rate,
                    "trend": metrics.defect_rate_trend,
                    "most_common_defect": metrics.most_common_defect.value,
                    "avg_severity": metrics.average_severity,
                    "health_impact": metrics.health_score_impact,
                }

        return report

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.loop_iterations = 0
        self.total_processing_time_ms = 0.0
        self.bridge.total_products_inspected = 0
        self.bridge.total_defects_found = 0

    def shutdown(self) -> None:
        """Shutdown the feedback loop."""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("IntegratedFeedbackLoop shutdown complete")


def create_feedback_loop(
    model: torch.nn.Module,
    production_lines: List[ProductionLine],
    **explainer_kwargs,
) -> IntegratedFeedbackLoop:
    """
    Factory function to create a complete feedback loop system.

    Args:
        model: Trained defect detection model
        production_lines: Digital Twin production lines
        **explainer_kwargs: Arguments for DefectExplainer

    Returns:
        Configured IntegratedFeedbackLoop
    """
    explainer = DefectExplainer(model, **explainer_kwargs)
    return IntegratedFeedbackLoop(explainer, production_lines)
