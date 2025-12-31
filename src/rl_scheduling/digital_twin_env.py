"""
Digital Twin-in-the-Loop RL Environment

This module provides a Gymnasium environment that wraps the actual
Digital Twin FactorySimulator, enabling:
1. Realistic simulation-based training
2. Sim-to-Real transfer capabilities
3. Multi-fidelity learning (fast abstract + high-fidelity DT)

Innovation: Unlike typical RL scheduling envs that use simplified models,
this uses the full discrete-event simulator as the environment.
"""

from typing import Callable, Dict, List, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from src.digital_twin.simulation.simulator import FactorySimulator, SimulationConfig
from src.digital_twin.simulation.production_line import (
    ProductionLine,
    WorkStation,
    Product,
    ProductStatus,
    create_sample_production_line,
)
from src.digital_twin.state.machine_state import MachineStatus
from src.core.logging import get_logger

logger = get_logger(__name__)


class ActionType(Enum):
    """Types of actions the RL agent can take."""
    ASSIGN_JOB = "assign_job"
    PRIORITY_ADJUSTMENT = "priority_adjust"
    MAINTENANCE_TRIGGER = "maintenance"
    SPEED_ADJUSTMENT = "speed_adjust"
    NO_OP = "no_op"


@dataclass
class SimToRealConfig:
    """Configuration for Sim-to-Real transfer."""

    # Domain randomization ranges
    processing_time_noise: float = 0.2  # ±20% variation
    defect_rate_noise: float = 0.3  # ±30% variation
    arrival_rate_noise: float = 0.25  # ±25% variation

    # Curriculum learning
    initial_difficulty: float = 0.3  # Start easy
    difficulty_increment: float = 0.1  # Increase per milestone
    max_difficulty: float = 1.0

    # Reality gap compensation
    action_delay_steps: int = 0  # Simulate action execution delay
    observation_noise: float = 0.05  # Sensor noise simulation

    # Domain adaptation
    enable_domain_randomization: bool = True
    num_envs_for_averaging: int = 5


class DigitalTwinRLEnv(gym.Env):
    """
    Gymnasium environment that uses the Digital Twin simulator directly.

    This is the core innovation - the RL agent learns scheduling policies
    by interacting with a high-fidelity discrete-event simulation.

    Key Features:
    1. Real Digital Twin Integration: Uses actual FactorySimulator
    2. Hierarchical Action Space: Job assignment + parameter tuning
    3. Rich Observation Space: Full factory state + predictive metrics
    4. Domain Randomization: For robust Sim-to-Real transfer
    5. Multi-objective Rewards: Throughput, quality, efficiency

    State Space (shape: (N,)):
        - Per-station features: status, health, queue, defect_rate, utilization
        - Global metrics: throughput, yield, WIP, bottleneck indicator
        - Temporal features: time-of-day, remaining_capacity

    Action Space:
        - Discrete: Station selection × Action type
        - Continuous: Processing speed multiplier, maintenance threshold
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        num_production_lines: int = 1,
        simulation_hours: float = 8.0,
        step_duration: float = 60.0,  # seconds per RL step
        sim_to_real_config: Optional[SimToRealConfig] = None,
        use_hierarchical_actions: bool = True,
        reward_weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        external_production_lines: Optional[List[ProductionLine]] = None,
    ):
        """
        Initialize Digital Twin RL Environment.

        Args:
            num_production_lines: Number of production lines to simulate
            simulation_hours: Total episode duration in hours
            step_duration: Simulation time per RL step (seconds)
            sim_to_real_config: Configuration for Sim-to-Real transfer
            use_hierarchical_actions: Use hierarchical action space
            reward_weights: Custom reward component weights
            seed: Random seed for reproducibility
            external_production_lines: External production lines to use (for integration)
                                       If provided, enables closed-loop with feedback systems
        """
        super().__init__()

        self.num_lines = num_production_lines
        self.simulation_hours = simulation_hours
        self.step_duration = step_duration
        self.sim_to_real = sim_to_real_config or SimToRealConfig()
        self.use_hierarchical_actions = use_hierarchical_actions
        self._external_lines = external_production_lines  # Store for reset()
        self._on_reset_callbacks: List[Callable[[], None]] = []  # Reset callbacks

        # Reward weights
        self.reward_weights = reward_weights or {
            "throughput": 1.0,
            "quality": 2.0,  # Quality is critical
            "efficiency": 0.5,
            "stability": 0.3,
            "maintenance": 0.2,
        }

        self.rng = np.random.default_rng(seed)

        # Create or use external production lines (CRITICAL for integration)
        if external_production_lines is not None:
            self._validate_external_lines(external_production_lines, num_production_lines)
            self.production_lines = external_production_lines
            self.num_lines = len(external_production_lines)
        else:
            self.production_lines: List[ProductionLine] = []
            self._create_production_lines()

        # Calculate observation and action space dimensions
        self.num_stations = sum(
            len(line.stations) for line in self.production_lines
        )

        # Station features: 8 per station
        # - status (one-hot 4), health, queue_size, defect_rate, utilization
        station_features = self.num_stations * 8

        # Global features: 10
        # - throughput_rate, yield_rate, wip_count, bottleneck_idx
        # - time_ratio, products_completed, products_defective
        # - avg_processing_time, avg_wait_time, overall_health
        global_features = 10

        # Predictive features: 4 (from maintenance prediction)
        # - predicted_failure_prob, time_to_maintenance
        # - quality_trend, throughput_trend
        predictive_features = 4

        self.obs_size = station_features + global_features + predictive_features

        # Observation space
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        # Action space
        if use_hierarchical_actions:
            # Hierarchical: [station_idx, action_type, action_param]
            # Station selection + action type + continuous parameter
            self.action_space = spaces.Dict({
                "station": spaces.Discrete(self.num_stations + 1),  # +1 for global action
                "action_type": spaces.Discrete(len(ActionType)),
                "param": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            })
        else:
            # Flat discrete action space
            # Actions: per-station maintenance triggers + global actions
            self.action_space = spaces.Discrete(
                self.num_stations * len(ActionType) + 1  # +1 for no-op
            )

        # Internal state
        self.simulator: Optional[FactorySimulator] = None
        self.current_step = 0
        self.episode_rewards: List[float] = []

        # Metrics tracking
        self.prev_completed = 0
        self.prev_defective = 0
        self.prev_health_scores: Dict[str, float] = {}

        # Curriculum learning state
        self.current_difficulty = self.sim_to_real.initial_difficulty
        self.episodes_completed = 0

        logger.info(
            f"DigitalTwinRLEnv initialized: {self.num_lines} lines, "
            f"{self.num_stations} stations, obs_size={self.obs_size}"
        )

    def _validate_external_lines(
        self,
        lines: List[ProductionLine],
        requested_num_lines: int,
    ) -> None:
        """Validate external production lines for integration."""
        if not lines:
            raise ValueError(
                "external_production_lines cannot be empty. "
                "Provide at least one ProductionLine or use None."
            )

        if len(lines) != requested_num_lines:
            logger.warning(
                f"num_production_lines={requested_num_lines} ignored; "
                f"using {len(lines)} external lines"
            )

        for i, line in enumerate(lines):
            if not hasattr(line, 'line_id'):
                raise ValueError(f"Line at index {i} missing 'line_id'")

            if not hasattr(line, 'stations') or not line.stations:
                raise ValueError(f"Line '{getattr(line, 'line_id', i)}' has no stations")

            for station_id, station in line.stations.items():
                required_attrs = ['machine_state', 'buffer', 'station_type']
                for attr in required_attrs:
                    if not hasattr(station, attr):
                        raise ValueError(f"Station '{station_id}' missing '{attr}'")

                if not hasattr(station.machine_state, 'health_score'):
                    raise ValueError(f"Station '{station_id}' missing 'health_score'")

        logger.info(f"Validated {len(lines)} external production lines")

    def register_reset_callback(self, callback: Callable[[], None]) -> None:
        """Register callback to be called after environment reset."""
        self._on_reset_callbacks.append(callback)

    def _create_production_lines(self) -> None:
        """Create production lines with optional domain randomization."""
        self.production_lines = []

        for i in range(self.num_lines):
            line = create_sample_production_line(f"Line_{i+1:02d}")

            # Apply domain randomization if enabled
            if self.sim_to_real.enable_domain_randomization:
                self._apply_domain_randomization(line)

            self.production_lines.append(line)

    def _apply_domain_randomization(self, line: ProductionLine) -> None:
        """Apply domain randomization to production line parameters."""
        for station in line.stations.values():
            # Randomize processing time
            noise_factor = 1.0 + self.rng.uniform(
                -self.sim_to_real.processing_time_noise,
                self.sim_to_real.processing_time_noise
            ) * self.current_difficulty
            station.processing_time_mean *= noise_factor

            # Randomize defect rate
            defect_noise = 1.0 + self.rng.uniform(
                -self.sim_to_real.defect_rate_noise,
                self.sim_to_real.defect_rate_noise
            ) * self.current_difficulty
            station.defect_rate *= defect_noise
            station.defect_rate = min(0.2, max(0.001, station.defect_rate))

    def _reset_production_lines_state(self) -> None:
        """
        Reset production line state without replacing the objects.

        This is CRITICAL for integration - external production lines are shared
        with the feedback loop. We reset state but keep object references intact
        so feedback updates continue to affect this environment.
        """
        for line in self.production_lines:
            # Clear product tracking
            line.products.clear()
            line.completed_products.clear()
            line.defective_products.clear()

            # Reset station state
            for station in line.stations.values():
                # Reset machine state but preserve degradation from feedback
                # Only reset counters, not health_score (feedback manages health)
                station.total_processed = 0
                station.total_defects = 0
                station.downtime_events = 0
                station.products_in_process.clear()

                # Clear buffer
                while not station.buffer.empty():
                    try:
                        station.buffer.get_nowait()
                    except Exception:
                        break

                # Reset sensors
                station.sensor_network.reset_all()

        logger.debug("Production lines state reset (objects preserved for integration)")

    def _create_simulator(self) -> FactorySimulator:
        """Create and configure the factory simulator."""
        # Apply arrival rate randomization
        base_arrival_rate = 60.0  # seconds
        if self.sim_to_real.enable_domain_randomization:
            noise = self.rng.uniform(
                -self.sim_to_real.arrival_rate_noise,
                self.sim_to_real.arrival_rate_noise
            ) * self.current_difficulty
            arrival_rate = base_arrival_rate * (1.0 + noise)
        else:
            arrival_rate = base_arrival_rate

        config = SimulationConfig(
            duration=self.simulation_hours * 3600,
            product_arrival_rate=arrival_rate,
            product_types=["ProductA", "ProductB", "ProductC"],
            verbose=False,
        )

        return FactorySimulator(config, self.production_lines)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial state observation
            info: Additional information
        """
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Only recreate production lines if NOT using external lines
        # External lines are shared with feedback loop - don't replace them!
        if self._external_lines is None:
            self._create_production_lines()
        else:
            # Reset external lines state but keep the same objects (CRITICAL for integration)
            self._reset_production_lines_state()

        # Create fresh simulator using current production_lines
        self.simulator = self._create_simulator()

        # Initialize tracking
        self.current_step = 0
        self.episode_rewards = []
        self.prev_completed = 0
        self.prev_defective = 0

        # Store initial health scores
        self.prev_health_scores = {}
        for line in self.production_lines:
            for station_id, station in line.stations.items():
                self.prev_health_scores[station_id] = station.machine_state.health_score

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        # Execute reset callbacks (for state synchronization)
        for callback in self._on_reset_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Reset callback error: {e}")

        logger.debug(f"Environment reset, difficulty: {self.current_difficulty:.2f}")

        return observation, info

    def step(
        self,
        action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: The action to take

        Returns:
            observation: New state observation
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut short
            info: Additional information
        """
        if self.simulator is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Parse and execute action
        action_info = self._execute_action(action)

        # Run simulator for step_duration
        self.simulator.step(duration=self.step_duration)

        # Calculate reward
        reward = self._calculate_reward(action_info)
        self.episode_rewards.append(reward)

        # Check termination
        self.current_step += 1
        max_steps = int(self.simulation_hours * 3600 / self.step_duration)

        terminated = self.simulator.simulation_time >= self.simulator.config.duration
        truncated = self.current_step >= max_steps

        # Episode completion bonuses
        if terminated or truncated:
            reward += self._calculate_episode_bonus()
            self._update_curriculum()

        # Get observation
        observation = self._get_observation()
        info = self._get_info()
        info.update(action_info)

        return observation, reward, terminated, truncated, info

    def _execute_action(self, action: Any) -> Dict:
        """Execute the given action and return info."""
        info = {
            "action_valid": True,
            "action_type": None,
            "target_station": None,
            "param_value": None,
        }

        if self.use_hierarchical_actions:
            # Parse hierarchical action
            station_idx = action["station"]
            action_type_idx = action["action_type"]
            param = action["param"][0]

            action_type = list(ActionType)[action_type_idx]
            info["action_type"] = action_type.value
            info["param_value"] = float(param)

            if station_idx < self.num_stations:
                # Get target station
                station = self._get_station_by_index(station_idx)
                if station:
                    info["target_station"] = station.station_id
                    self._apply_station_action(station, action_type, param)
            else:
                # Global action (no specific station)
                self._apply_global_action(action_type, param)
        else:
            # Flat discrete action
            if action == self.action_space.n - 1:
                # No-op
                info["action_type"] = ActionType.NO_OP.value
            else:
                station_idx = action // len(ActionType)
                action_type_idx = action % len(ActionType)
                action_type = list(ActionType)[action_type_idx]

                info["action_type"] = action_type.value
                station = self._get_station_by_index(station_idx)
                if station:
                    info["target_station"] = station.station_id
                    self._apply_station_action(station, action_type, 0.5)

        return info

    def _get_station_by_index(self, idx: int) -> Optional[WorkStation]:
        """Get station by flat index across all production lines."""
        current_idx = 0
        for line in self.production_lines:
            for station in line.stations.values():
                if current_idx == idx:
                    return station
                current_idx += 1
        return None

    def _apply_station_action(
        self,
        station: WorkStation,
        action_type: ActionType,
        param: float
    ) -> None:
        """Apply an action to a specific station."""
        if action_type == ActionType.MAINTENANCE_TRIGGER:
            # Trigger preventive maintenance if health below threshold
            threshold = 0.3 + param * 0.5  # 0.3 - 0.8 threshold
            if station.machine_state.health_score < threshold:
                station.perform_maintenance()
                logger.debug(f"Maintenance triggered on {station.station_id}")

        elif action_type == ActionType.SPEED_ADJUSTMENT:
            # Adjust processing speed (affects quality tradeoff)
            speed_factor = 0.8 + param * 0.4  # 0.8x - 1.2x
            station.processing_time_mean /= speed_factor
            # Speed increase raises defect rate
            station.defect_rate *= (1.0 + (speed_factor - 1.0) * 0.5)

        elif action_type == ActionType.PRIORITY_ADJUSTMENT:
            # Adjust station priority (affects product routing)
            # In this simplified version, just log it
            pass

    def _apply_global_action(self, action_type: ActionType, param: float) -> None:
        """Apply a global action affecting all stations."""
        if action_type == ActionType.MAINTENANCE_TRIGGER:
            # Global maintenance check
            for line in self.production_lines:
                for station in line.stations.values():
                    if station.machine_state.health_score < 0.5:
                        station.perform_maintenance()

    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector from current simulator state.

        Returns:
            Normalized observation vector
        """
        obs = []

        # Per-station features
        for line in self.production_lines:
            for station in line.stations.values():
                # Status one-hot (4 values: IDLE, RUNNING, WARNING, ERROR)
                status = station.machine_state.status
                status_onehot = [0.0] * 4
                status_map = {
                    MachineStatus.IDLE: 0,
                    MachineStatus.RUNNING: 1,
                    MachineStatus.WARNING: 2,
                    MachineStatus.ERROR: 3,
                }
                if status in status_map:
                    status_onehot[status_map[status]] = 1.0
                obs.extend(status_onehot)

                # Health score (0-1)
                obs.append(station.machine_state.health_score)

                # Queue size (normalized)
                queue_size = station.buffer.qsize() / 10.0  # Assume max 10
                obs.append(min(1.0, queue_size))

                # Defect rate (normalized, assumed max 0.2)
                if station.total_processed > 0:
                    actual_defect_rate = station.total_defects / station.total_processed
                else:
                    actual_defect_rate = station.defect_rate
                obs.append(min(1.0, actual_defect_rate / 0.2))

                # Utilization
                utilization = len(station.products_in_process) / station.capacity
                obs.append(utilization)

        # Global features
        stats = self.simulator.get_statistics()

        # Throughput rate (products per hour, normalized)
        if stats["simulation_time"] > 0:
            throughput_rate = stats["total_products_completed"] / (stats["simulation_time"] / 3600)
            obs.append(min(1.0, throughput_rate / 100.0))  # Assume max 100/hr
        else:
            obs.append(0.0)

        # Yield rate
        obs.append(stats["overall_yield"])

        # WIP count (normalized)
        wip = stats["total_products_introduced"] - stats["total_products_completed"] - stats["total_products_defective"]
        obs.append(min(1.0, wip / 50.0))  # Assume max 50 WIP

        # Bottleneck indicator (station with largest queue)
        max_queue = 0
        bottleneck_idx = 0
        idx = 0
        for line in self.production_lines:
            for station in line.stations.values():
                q_size = station.buffer.qsize()
                if q_size > max_queue:
                    max_queue = q_size
                    bottleneck_idx = idx
                idx += 1
        obs.append(bottleneck_idx / max(1, self.num_stations))

        # Time ratio (progress through episode)
        duration = stats.get("duration", 1.0) or 1.0  # Prevent division by zero
        time_ratio = stats["simulation_time"] / duration
        obs.append(min(1.0, time_ratio))

        # Products completed (normalized)
        obs.append(min(1.0, stats["total_products_completed"] / 500.0))

        # Products defective (normalized)
        obs.append(min(1.0, stats["total_products_defective"] / 50.0))

        # Average processing time estimate
        avg_proc_time = 0.0
        num_stations = 0
        for line in self.production_lines:
            for station in line.stations.values():
                avg_proc_time += station.processing_time_mean
                num_stations += 1
        if num_stations > 0:
            avg_proc_time /= num_stations
        obs.append(min(1.0, avg_proc_time / 60.0))  # Normalized to 60s max

        # Average wait time estimate (based on queue sizes)
        total_queue = sum(
            station.buffer.qsize()
            for line in self.production_lines
            for station in line.stations.values()
        )
        obs.append(min(1.0, total_queue * avg_proc_time / 600.0))

        # Overall health (average across stations)
        health_sum = sum(
            station.machine_state.health_score
            for line in self.production_lines
            for station in line.stations.values()
        )
        obs.append(health_sum / max(1, self.num_stations))

        # Predictive features
        # Predicted failure probability (based on health trends)
        health_degradation = 0.0
        for station_id, prev_health in self.prev_health_scores.items():
            current_health = self._get_station_health(station_id)
            health_degradation += max(0, prev_health - current_health)
        failure_prob = min(1.0, health_degradation * 10)
        obs.append(failure_prob)

        # Time to maintenance (estimate)
        min_health = 1.0
        for line in self.production_lines:
            for station in line.stations.values():
                min_health = min(min_health, station.machine_state.health_score)
        time_to_maint = min_health  # Lower health = closer to maintenance
        obs.append(time_to_maint)

        # Quality trend (recent defect rate change)
        current_defects = stats["total_products_defective"]
        defect_change = current_defects - self.prev_defective
        self.prev_defective = current_defects
        quality_trend = -min(1.0, defect_change / 10.0)  # Negative = getting worse
        obs.append(quality_trend)

        # Throughput trend
        current_completed = stats["total_products_completed"]
        completed_change = current_completed - self.prev_completed
        self.prev_completed = current_completed
        throughput_trend = min(1.0, completed_change / 10.0)
        obs.append(throughput_trend)

        # Add observation noise for sim-to-real robustness
        obs_array = np.array(obs, dtype=np.float32)
        if self.sim_to_real.enable_domain_randomization:
            noise = self.rng.normal(0, self.sim_to_real.observation_noise, obs_array.shape)
            obs_array += noise.astype(np.float32)

        # Clip to valid range
        obs_array = np.clip(obs_array, -1.0, 1.0)

        # Ensure correct size (pad if necessary)
        if len(obs_array) < self.obs_size:
            obs_array = np.pad(obs_array, (0, self.obs_size - len(obs_array)))
        elif len(obs_array) > self.obs_size:
            obs_array = obs_array[:self.obs_size]

        return obs_array

    def _get_station_health(self, station_id: str) -> float:
        """Get health score for a station by ID."""
        for line in self.production_lines:
            if station_id in line.stations:
                return line.stations[station_id].machine_state.health_score
        return 1.0

    def _calculate_reward(self, action_info: Dict) -> float:
        """
        Calculate multi-objective reward.

        Components:
        - Throughput: Products completed this step
        - Quality: Inverse of defect rate
        - Efficiency: Machine utilization
        - Stability: Low variance in queue sizes
        - Maintenance: Proactive maintenance bonus
        """
        stats = self.simulator.get_statistics()
        reward = 0.0

        # Throughput reward
        new_completed = stats["total_products_completed"] - self.prev_completed
        throughput_reward = new_completed * 0.5
        reward += self.reward_weights["throughput"] * throughput_reward

        # Quality reward
        yield_rate = stats["overall_yield"]
        quality_reward = yield_rate * 2.0 - 1.0  # -1 to +1 range
        reward += self.reward_weights["quality"] * quality_reward

        # Efficiency reward
        total_utilization = 0.0
        for line in self.production_lines:
            for station in line.stations.values():
                total_utilization += len(station.products_in_process) / station.capacity
        avg_utilization = total_utilization / max(1, self.num_stations)
        efficiency_reward = avg_utilization - 0.5  # Centered at 50%
        reward += self.reward_weights["efficiency"] * efficiency_reward

        # Stability reward (low queue variance)
        queue_sizes = [
            station.buffer.qsize()
            for line in self.production_lines
            for station in line.stations.values()
        ]
        if len(queue_sizes) > 1:
            queue_variance = np.var(queue_sizes)
            stability_reward = -min(1.0, queue_variance / 10.0)
        else:
            stability_reward = 0.0
        reward += self.reward_weights["stability"] * stability_reward

        # Maintenance reward (proactive maintenance when needed)
        if action_info.get("action_type") == ActionType.MAINTENANCE_TRIGGER.value:
            target = action_info.get("target_station")
            if target:
                # Get station health before action
                prev_health = self.prev_health_scores.get(target, 1.0)
                if prev_health < 0.7:  # Maintenance was needed
                    maintenance_reward = 0.5 * (0.7 - prev_health)
                else:  # Unnecessary maintenance
                    maintenance_reward = -0.2
                reward += self.reward_weights["maintenance"] * maintenance_reward

        # Update health score tracking
        for line in self.production_lines:
            for station_id, station in line.stations.items():
                self.prev_health_scores[station_id] = station.machine_state.health_score

        return reward

    def _calculate_episode_bonus(self) -> float:
        """Calculate bonus/penalty at episode end."""
        stats = self.simulator.get_statistics()
        bonus = 0.0

        # High yield bonus
        if stats["overall_yield"] > 0.95:
            bonus += 10.0
        elif stats["overall_yield"] > 0.90:
            bonus += 5.0

        # Throughput target bonus
        target_throughput = 100  # products per episode
        if stats["total_products_completed"] >= target_throughput:
            bonus += 5.0

        # Low defect bonus
        if stats["total_products_defective"] < 5:
            bonus += 3.0

        return bonus

    def _update_curriculum(self) -> None:
        """Update curriculum learning difficulty."""
        self.episodes_completed += 1

        # Increase difficulty every 10 episodes
        if self.episodes_completed % 10 == 0:
            self.current_difficulty = min(
                self.sim_to_real.max_difficulty,
                self.current_difficulty + self.sim_to_real.difficulty_increment
            )
            logger.info(f"Curriculum difficulty increased to {self.current_difficulty:.2f}")

    def _get_info(self) -> Dict:
        """Get additional information about current state."""
        stats = self.simulator.get_statistics()

        return {
            "simulation_time": stats["simulation_time"],
            "step": self.current_step,
            "products_completed": stats["total_products_completed"],
            "products_defective": stats["total_products_defective"],
            "yield_rate": stats["overall_yield"],
            "difficulty": self.current_difficulty,
            "episode_reward_sum": sum(self.episode_rewards),
        }

    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment state."""
        if self.simulator is None:
            return None

        stats = self.simulator.get_statistics()

        output = []
        output.append(f"\n{'='*60}")
        output.append(f"Digital Twin RL Environment - Step {self.current_step}")
        output.append(f"{'='*60}")
        output.append(f"Simulation Time: {stats['simulation_time']:.1f}s / {stats['duration']:.1f}s")
        output.append(f"Products: Completed={stats['total_products_completed']}, "
                     f"Defective={stats['total_products_defective']}")
        output.append(f"Yield: {stats['overall_yield']:.2%}")
        output.append(f"Episode Reward: {sum(self.episode_rewards):.2f}")
        output.append("")

        for line_id, line_stats in stats.get("production_lines", {}).items():
            output.append(f"Production Line: {line_id}")
            for station_id, station_stats in line_stats.get("stations", {}).items():
                output.append(
                    f"  {station_id}: "
                    f"Health={station_stats['health_score']:.2f}, "
                    f"Defects={station_stats['total_defects']}, "
                    f"Queue={station_stats['buffer_size']}"
                )

        output.append(f"{'='*60}\n")

        result = "\n".join(output)
        if mode == "human":
            print(result)
        return result

    def close(self) -> None:
        """Clean up resources."""
        self.simulator = None
        self.production_lines = []

    def get_simulator_state(self) -> Dict:
        """
        Get full simulator state for analysis.

        Useful for:
        - Debugging
        - Visualization
        - Sim-to-Real comparison
        """
        if self.simulator is None:
            return {}
        return self.simulator.get_statistics()


def create_digital_twin_env(
    difficulty: str = "easy",
    **kwargs
) -> DigitalTwinRLEnv:
    """
    Factory function to create DigitalTwinRLEnv with preset configurations.

    Args:
        difficulty: "easy", "medium", "hard"
        **kwargs: Additional arguments passed to DigitalTwinRLEnv

    Returns:
        Configured environment
    """
    configs = {
        "easy": SimToRealConfig(
            processing_time_noise=0.1,
            defect_rate_noise=0.1,
            arrival_rate_noise=0.1,
            initial_difficulty=0.2,
            enable_domain_randomization=False,
        ),
        "medium": SimToRealConfig(
            processing_time_noise=0.2,
            defect_rate_noise=0.2,
            arrival_rate_noise=0.2,
            initial_difficulty=0.5,
            enable_domain_randomization=True,
        ),
        "hard": SimToRealConfig(
            processing_time_noise=0.3,
            defect_rate_noise=0.3,
            arrival_rate_noise=0.3,
            initial_difficulty=0.8,
            enable_domain_randomization=True,
            observation_noise=0.1,
        ),
    }

    config = configs.get(difficulty, configs["medium"])
    return DigitalTwinRLEnv(sim_to_real_config=config, **kwargs)
