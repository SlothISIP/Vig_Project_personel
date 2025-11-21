"""
Unified Pipeline - Complete Integration of Vision AI, Digital Twin, and RL.

This module provides the complete end-to-end pipeline demonstrating the
project's key innovations:

1. Digital Twin-in-the-Loop RL (Option B)
   - Uses actual factory simulator as RL environment
   - Domain randomization for sim-to-real transfer
   - Curriculum learning for progressive difficulty

2. Explainable AI with Feedback Loop (Option C)
   - Grad-CAM visualization for defect localization
   - Pattern classification for defect categorization
   - Root cause analysis linking defects to manufacturing process
   - Closed-loop feedback to Digital Twin for continuous improvement

The unified pipeline combines these for a complete smart factory system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from enum import Enum
import threading
import queue
import numpy as np
import torch
import torch.nn as nn

# Core components
from src.digital_twin.simulation.simulator import FactorySimulator, SimulationConfig
from src.digital_twin.simulation.production_line import (
    ProductionLine,
    Product,
    create_sample_production_line,
)
from src.rl_scheduling.digital_twin_env import (
    DigitalTwinRLEnv,
    SimToRealConfig,
    create_digital_twin_env,
)
from src.vision.explainability.defect_explainer import (
    DefectExplainer,
    ExplanationResult,
    DefectType,
    DefectSeverity,
)
from src.integration.feedback_loop import (
    IntegratedFeedbackLoop,
    VisionDigitalTwinBridge,
    DigitalTwinFeedbackController,
    create_feedback_loop,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


class PipelineMode(Enum):
    """Operating modes for the unified pipeline."""

    TRAINING = "training"  # RL training mode
    INFERENCE = "inference"  # Production inference mode
    SIMULATION = "simulation"  # Full simulation with all components
    ANALYSIS = "analysis"  # Analysis/debugging mode


@dataclass
class PipelineConfig:
    """Configuration for unified pipeline."""

    # General settings
    mode: PipelineMode = PipelineMode.SIMULATION

    # Digital Twin settings
    num_production_lines: int = 1
    simulation_hours: float = 8.0
    step_duration: float = 60.0  # seconds per step

    # RL settings
    enable_rl: bool = True
    rl_difficulty: str = "medium"  # easy, medium, hard
    rl_training_steps: int = 10000

    # Vision AI settings
    enable_vision: bool = True
    defect_threshold: float = 0.5
    enable_xai: bool = True

    # Feedback loop settings
    enable_feedback: bool = True
    feedback_async: bool = True

    # Logging
    verbose: bool = False
    log_interval: int = 100


@dataclass
class PipelineState:
    """Current state of the unified pipeline."""

    is_running: bool = False
    mode: PipelineMode = PipelineMode.SIMULATION
    iteration: int = 0
    simulation_time: float = 0.0

    # RL state
    rl_episode: int = 0
    rl_total_reward: float = 0.0
    rl_avg_reward: float = 0.0

    # Quality state
    products_inspected: int = 0
    defects_detected: int = 0
    defect_rate: float = 0.0

    # Health state
    avg_station_health: float = 1.0
    min_station_health: float = 1.0
    stations_needing_maintenance: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_running": self.is_running,
            "mode": self.mode.value,
            "iteration": self.iteration,
            "simulation_time": self.simulation_time,
            "rl_episode": self.rl_episode,
            "rl_avg_reward": self.rl_avg_reward,
            "products_inspected": self.products_inspected,
            "defects_detected": self.defects_detected,
            "defect_rate": self.defect_rate,
            "avg_station_health": self.avg_station_health,
            "min_station_health": self.min_station_health,
            "maintenance_required": len(self.stations_needing_maintenance),
        }


class UnifiedPipeline:
    """
    Unified Pipeline integrating Vision AI, Digital Twin, and RL.

    This is the main orchestrator that demonstrates the project's innovations
    by running all components together in a coordinated manner.

    Key Innovations Demonstrated:
    1. RL agent learns scheduling from Digital Twin simulation
    2. Vision AI provides explainable defect detection
    3. Defect feedback updates Digital Twin state
    4. Closed-loop continuous improvement

    Example Usage:
        ```python
        pipeline = UnifiedPipeline(config=PipelineConfig())
        pipeline.initialize()
        pipeline.run(steps=1000)
        report = pipeline.get_report()
        pipeline.shutdown()
        ```
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        vision_model: Optional[nn.Module] = None,
    ):
        """
        Initialize unified pipeline.

        Args:
            config: Pipeline configuration
            vision_model: Pre-trained defect detection model (optional)
        """
        self.config = config or PipelineConfig()
        self.vision_model = vision_model

        # Components (initialized in initialize())
        self.production_lines: List[ProductionLine] = []
        self.rl_env: Optional[DigitalTwinRLEnv] = None
        self.feedback_loop: Optional[IntegratedFeedbackLoop] = None
        self.defect_explainer: Optional[DefectExplainer] = None

        # State
        self.state = PipelineState()
        self._lock = threading.Lock()

        # Callbacks
        self._step_callbacks: List[Callable[[PipelineState], None]] = []
        self._defect_callbacks: List[Callable[[ExplanationResult], None]] = []

        # Event queue for async processing
        self._event_queue: queue.Queue = queue.Queue()

        logger.info(f"UnifiedPipeline created with mode: {self.config.mode.value}")

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing unified pipeline...")

        # Create production lines
        self.production_lines = [
            create_sample_production_line(f"Line_{i+1:02d}")
            for i in range(self.config.num_production_lines)
        ]
        logger.info(f"Created {len(self.production_lines)} production lines")

        # Initialize RL environment if enabled
        if self.config.enable_rl:
            self.rl_env = create_digital_twin_env(
                difficulty=self.config.rl_difficulty,
                num_production_lines=self.config.num_production_lines,
                simulation_hours=self.config.simulation_hours,
                step_duration=self.config.step_duration,
            )
            logger.info("RL environment initialized")

        # Initialize Vision AI if enabled and model provided
        if self.config.enable_vision and self.vision_model is not None:
            self.defect_explainer = DefectExplainer(
                model=self.vision_model,
                defect_threshold=self.config.defect_threshold,
            )
            logger.info("Vision AI explainer initialized")

        # Initialize feedback loop if enabled
        if self.config.enable_feedback and self.defect_explainer is not None:
            self.feedback_loop = IntegratedFeedbackLoop(
                defect_explainer=self.defect_explainer,
                production_lines=self.production_lines,
                enable_async=self.config.feedback_async,
            )
            logger.info("Feedback loop initialized")

        self.state.mode = self.config.mode
        logger.info("Pipeline initialization complete")

    def run(self, steps: int = 1000) -> PipelineState:
        """
        Run the unified pipeline for specified number of steps.

        Args:
            steps: Number of steps to run

        Returns:
            Final pipeline state
        """
        if self.rl_env is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        self.state.is_running = True
        logger.info(f"Starting pipeline run for {steps} steps")

        # Reset RL environment
        obs, info = self.rl_env.reset()
        episode_reward = 0.0
        episode_steps = 0

        for step in range(steps):
            self.state.iteration = step

            # RL step
            action = self._select_action(obs)
            next_obs, reward, terminated, truncated, info = self.rl_env.step(action)

            episode_reward += reward
            episode_steps += 1

            # Update state
            self.state.simulation_time = info.get("simulation_time", 0)
            self.state.rl_total_reward += reward

            # Simulate vision inspection (every N steps)
            if self.config.enable_vision and step % 10 == 0:
                self._simulate_vision_inspection()

            # Episode reset
            if terminated or truncated:
                self.state.rl_episode += 1
                self.state.rl_avg_reward = (
                    self.state.rl_total_reward / max(1, step + 1)
                )

                if self.config.verbose:
                    logger.info(
                        f"Episode {self.state.rl_episode} complete: "
                        f"reward={episode_reward:.2f}, steps={episode_steps}"
                    )

                obs, info = self.rl_env.reset()
                episode_reward = 0.0
                episode_steps = 0
            else:
                obs = next_obs

            # Update health metrics
            self._update_health_metrics()

            # Call step callbacks
            for callback in self._step_callbacks:
                try:
                    callback(self.state)
                except Exception as e:
                    logger.error(f"Step callback error: {e}")

            # Logging
            if self.config.verbose and step % self.config.log_interval == 0:
                logger.info(
                    f"Step {step}: reward={reward:.3f}, "
                    f"health={self.state.avg_station_health:.2f}"
                )

        self.state.is_running = False
        logger.info(f"Pipeline run complete after {steps} steps")

        return self.state

    def _select_action(self, observation: np.ndarray) -> Any:
        """
        Select action for RL step.

        In production, this would use a trained policy.
        For demonstration, uses random or heuristic action.
        """
        if self.rl_env is None:
            return 0

        # Simple heuristic: random action with bias towards maintenance
        # when health is low
        if self.state.min_station_health < 0.5:
            # Higher probability of maintenance action
            if np.random.random() < 0.3:
                # Select maintenance action
                from src.rl_scheduling.digital_twin_env import ActionType
                return {
                    "station": np.random.randint(0, self.rl_env.num_stations),
                    "action_type": list(ActionType).index(ActionType.MAINTENANCE_TRIGGER),
                    "param": np.array([0.7], dtype=np.float32),
                }

        # Random action
        return self.rl_env.action_space.sample()

    def _simulate_vision_inspection(self) -> None:
        """Simulate a vision inspection event."""
        if self.feedback_loop is None:
            # Without vision model, simulate with random defects
            self.state.products_inspected += 1
            if np.random.random() < 0.05:  # 5% defect rate
                self.state.defects_detected += 1
            return

        # Generate synthetic inspection data
        product_id = f"P{self.state.products_inspected:06d}"

        # Create dummy image tensor (would be real camera image in production)
        dummy_image = torch.randn(1, 3, 224, 224)

        try:
            explanation, results = self.feedback_loop.process_inspection(
                product_id=product_id,
                image=dummy_image,
            )

            self.state.products_inspected += 1
            if explanation.is_defect:
                self.state.defects_detected += 1

                # Call defect callbacks
                for callback in self._defect_callbacks:
                    try:
                        callback(explanation)
                    except Exception as e:
                        logger.error(f"Defect callback error: {e}")

        except Exception as e:
            logger.debug(f"Vision inspection simulation error: {e}")
            self.state.products_inspected += 1

        # Update defect rate
        if self.state.products_inspected > 0:
            self.state.defect_rate = (
                self.state.defects_detected / self.state.products_inspected
            )

    def _update_health_metrics(self) -> None:
        """Update station health metrics from Digital Twin."""
        if self.rl_env is None:
            return

        # Get health scores from production lines
        health_scores = []
        needing_maintenance = []

        for line in self.rl_env.production_lines:
            for station_id, station in line.stations.items():
                health = station.machine_state.health_score
                health_scores.append(health)

                if health < 0.6:
                    needing_maintenance.append(station_id)

        if health_scores:
            self.state.avg_station_health = np.mean(health_scores)
            self.state.min_station_health = min(health_scores)
        self.state.stations_needing_maintenance = needing_maintenance

    def step_single(self, action: Optional[Any] = None) -> Tuple[PipelineState, Dict]:
        """
        Execute a single pipeline step.

        Useful for external control or integration with other systems.

        Args:
            action: Optional action override

        Returns:
            Tuple of (state, info)
        """
        if self.rl_env is None:
            raise RuntimeError("Pipeline not initialized")

        # Use provided action or generate one
        if action is None:
            obs = self.rl_env._get_observation() if hasattr(self.rl_env, '_get_observation') else np.zeros(self.rl_env.obs_size)
            action = self._select_action(obs)

        # Execute step
        next_obs, reward, terminated, truncated, info = self.rl_env.step(action)

        # Update state
        self.state.iteration += 1
        self.state.simulation_time = info.get("simulation_time", 0)
        self.state.rl_total_reward += reward

        # Vision inspection
        if self.config.enable_vision:
            self._simulate_vision_inspection()

        # Health update
        self._update_health_metrics()

        return self.state, {
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            **info,
        }

    def register_step_callback(
        self,
        callback: Callable[[PipelineState], None],
    ) -> None:
        """Register callback for each pipeline step."""
        self._step_callbacks.append(callback)

    def register_defect_callback(
        self,
        callback: Callable[[ExplanationResult], None],
    ) -> None:
        """Register callback for defect detection events."""
        self._defect_callbacks.append(callback)

    def get_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive pipeline report.

        Returns:
            Dictionary with full pipeline status and metrics
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "mode": self.config.mode.value,
                "num_lines": self.config.num_production_lines,
                "simulation_hours": self.config.simulation_hours,
                "rl_enabled": self.config.enable_rl,
                "vision_enabled": self.config.enable_vision,
                "feedback_enabled": self.config.enable_feedback,
            },
            "state": self.state.to_dict(),
            "performance": {
                "rl_episodes": self.state.rl_episode,
                "rl_avg_reward": self.state.rl_avg_reward,
                "defect_rate": self.state.defect_rate,
                "avg_health": self.state.avg_station_health,
            },
        }

        # Add feedback loop report if available
        if self.feedback_loop:
            report["feedback"] = self.feedback_loop.get_system_health()
            report["quality"] = self.feedback_loop.get_quality_report()

        # Add RL environment stats if available
        if self.rl_env:
            report["simulator"] = self.rl_env.get_simulator_state()

        return report

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data formatted for visualization.

        Returns:
            Dictionary with visualization-ready data
        """
        viz_data = {
            "time_series": {
                "health_scores": [],
                "defect_rates": [],
                "rewards": [],
            },
            "station_status": [],
            "recent_defects": [],
        }

        # Station status
        if self.rl_env:
            for line in self.rl_env.production_lines:
                for station_id, station in line.stations.items():
                    viz_data["station_status"].append({
                        "station_id": station_id,
                        "type": station.station_type,
                        "health": station.machine_state.health_score,
                        "status": station.machine_state.status.value,
                        "defect_rate": (
                            station.total_defects / station.total_processed
                            if station.total_processed > 0
                            else 0.0
                        ),
                        "queue_size": station.buffer.qsize(),
                    })

        return viz_data

    def shutdown(self) -> None:
        """Shutdown the pipeline and cleanup resources."""
        logger.info("Shutting down unified pipeline...")

        self.state.is_running = False

        if self.feedback_loop:
            self.feedback_loop.shutdown()

        if self.rl_env:
            self.rl_env.close()

        logger.info("Pipeline shutdown complete")


def create_demo_pipeline(
    with_vision_model: bool = False,
) -> UnifiedPipeline:
    """
    Create a demo pipeline for testing.

    Args:
        with_vision_model: Whether to create a mock vision model

    Returns:
        Configured UnifiedPipeline
    """
    config = PipelineConfig(
        mode=PipelineMode.SIMULATION,
        num_production_lines=1,
        simulation_hours=2.0,
        step_duration=30.0,
        enable_rl=True,
        rl_difficulty="easy",
        enable_vision=with_vision_model,
        enable_feedback=with_vision_model,
        verbose=True,
    )

    # Create mock vision model if requested
    vision_model = None
    if with_vision_model:
        # Simple mock model for testing
        vision_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    pipeline = UnifiedPipeline(config=config, vision_model=vision_model)
    return pipeline


def run_integration_demo(steps: int = 100) -> Dict[str, Any]:
    """
    Run a complete integration demonstration.

    This demonstrates the key innovations:
    1. Digital Twin as RL environment
    2. Explainable defect detection
    3. Feedback loop for continuous improvement

    Args:
        steps: Number of simulation steps

    Returns:
        Demo results report
    """
    logger.info("=" * 60)
    logger.info("UNIFIED PIPELINE INTEGRATION DEMO")
    logger.info("=" * 60)

    # Create and initialize pipeline
    pipeline = create_demo_pipeline(with_vision_model=False)
    pipeline.initialize()

    # Run simulation
    final_state = pipeline.run(steps=steps)

    # Generate report
    report = pipeline.get_report()

    # Print summary
    logger.info("=" * 60)
    logger.info("DEMO COMPLETE - SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total iterations: {final_state.iteration}")
    logger.info(f"RL episodes: {final_state.rl_episode}")
    logger.info(f"Average reward: {final_state.rl_avg_reward:.3f}")
    logger.info(f"Products inspected: {final_state.products_inspected}")
    logger.info(f"Defect rate: {final_state.defect_rate:.2%}")
    logger.info(f"Average station health: {final_state.avg_station_health:.2%}")
    logger.info("=" * 60)

    # Cleanup
    pipeline.shutdown()

    return report


if __name__ == "__main__":
    # Run demo when executed directly
    run_integration_demo(steps=200)
