"""
RL Policy Inference Module
Deploys trained RL agent for real-time production scheduling
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from ray.rllib.algorithms import Algorithm
import pickle

from .state import SchedulingStateRL, SchedulingAction
from .environment import ProductionSchedulingEnv


class RLSchedulingPolicy:
    """Inference interface for trained RL scheduling policy"""

    def __init__(
        self, checkpoint_path: str, use_ray: bool = True, device: str = "cpu"
    ):
        """
        Initialize policy from checkpoint

        Args:
            checkpoint_path: Path to Ray RLlib checkpoint
            use_ray: Whether to use Ray for inference (faster) or standalone
            device: Device for computation ('cpu' or 'cuda')
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.use_ray = use_ray
        self.device = device

        if not self.checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

        self._load_policy()

    def _load_policy(self):
        """Load policy from checkpoint"""
        if self.use_ray:
            # Load full Ray algorithm
            import ray

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, num_cpus=2)

            self.algorithm = Algorithm.from_checkpoint(str(self.checkpoint_path))
            self.policy = self.algorithm.get_policy()
        else:
            # Load policy state only (lightweight)
            policy_state_path = self.checkpoint_path / "policy_state.pkl"
            if policy_state_path.exists():
                with open(policy_state_path, "rb") as f:
                    self.policy_state = pickle.load(f)
            else:
                raise ValueError("Policy state file not found for standalone mode")

        print(f"✓ Policy loaded from: {self.checkpoint_path}")

    def predict_action(
        self, state: SchedulingStateRL, explore: bool = False
    ) -> Tuple[int, Dict]:
        """
        Predict best action for given state

        Args:
            state: Current scheduling state
            explore: Whether to use exploration (stochastic policy)

        Returns:
            action: Action index
            info: Additional information (action probabilities, value, etc.)
        """
        # Convert state to observation
        obs = state.to_observation()

        if self.use_ray:
            # Use Ray RLlib policy
            action = self.algorithm.compute_single_action(obs, explore=explore)

            # Get action probabilities and value
            policy_output = self.policy.compute_actions(
                np.array([obs]), explore=explore
            )
            action_probs = policy_output[2].get("action_dist_inputs", None)

            info = {
                "action_probs": action_probs,
                "explore": explore,
            }
        else:
            # Standalone inference (simplified)
            # This requires implementing a lightweight inference path
            # For now, return random action
            action = np.random.randint(0, state.num_machines * 10)
            info = {"standalone_mode": True}

        return action, info

    def predict_job_machine_assignment(
        self, state: SchedulingStateRL
    ) -> Optional[SchedulingAction]:
        """
        Predict which job to assign to which machine

        Args:
            state: Current scheduling state

        Returns:
            SchedulingAction or None if no assignment recommended
        """
        action_idx, info = self.predict_action(state, explore=False)

        # Decode action
        num_machines = len(state.machines)
        num_pending = len(state.pending_jobs)

        # Check for no-op
        if action_idx >= num_pending * num_machines:
            return None

        job_idx = action_idx // num_machines
        machine_idx = action_idx % num_machines

        # Validate indices
        if job_idx >= num_pending or machine_idx >= num_machines:
            return None

        job = state.pending_jobs[job_idx]
        machine = state.machines[machine_idx]

        # Check if assignment is valid
        if not machine.is_available:
            return None

        return SchedulingAction(
            job_id=job.job_id, machine_id=machine.machine_id, priority_boost=0.0
        )

    def batch_predict(
        self, states: List[SchedulingStateRL], explore: bool = False
    ) -> List[Tuple[int, Dict]]:
        """
        Predict actions for multiple states in batch

        Args:
            states: List of scheduling states
            explore: Whether to use exploration

        Returns:
            List of (action, info) tuples
        """
        if not self.use_ray:
            # Fallback to sequential prediction
            return [self.predict_action(state, explore) for state in states]

        # Batch inference with Ray
        observations = np.array([state.to_observation() for state in states])
        actions = self.algorithm.compute_actions(observations, explore=explore)

        results = []
        for action in actions:
            results.append((action, {"batch_mode": True}))

        return results

    def evaluate_state_value(self, state: SchedulingStateRL) -> float:
        """
        Estimate value of current state (expected future reward)

        Args:
            state: Current scheduling state

        Returns:
            Estimated value
        """
        if not self.use_ray:
            return 0.0  # Not available in standalone mode

        obs = state.to_observation()
        policy_output = self.policy.compute_actions(np.array([obs]))

        # Extract value from policy output
        if len(policy_output) > 3 and "vf_preds" in policy_output[3]:
            value = float(policy_output[3]["vf_preds"][0])
            return value

        return 0.0

    def get_action_probabilities(
        self, state: SchedulingStateRL
    ) -> Optional[np.ndarray]:
        """
        Get probability distribution over actions

        Args:
            state: Current scheduling state

        Returns:
            Action probabilities array or None
        """
        if not self.use_ray:
            return None

        obs = state.to_observation()
        policy_output = self.policy.compute_actions(np.array([obs]))

        if len(policy_output) > 2 and "action_dist_inputs" in policy_output[2]:
            # For discrete actions, this is logits
            logits = policy_output[2]["action_dist_inputs"]
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            return probs[0]

        return None

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "algorithm"):
            self.algorithm.stop()


class DynamicScheduler:
    """
    Dynamic production scheduler using RL policy
    Integrates with existing scheduling system
    """

    def __init__(self, policy: RLSchedulingPolicy, fallback_enabled: bool = True):
        """
        Initialize dynamic scheduler

        Args:
            policy: Trained RL policy
            fallback_enabled: Use rule-based fallback if RL fails
        """
        self.policy = policy
        self.fallback_enabled = fallback_enabled
        self.stats = {
            "rl_decisions": 0,
            "fallback_decisions": 0,
            "invalid_actions": 0,
        }

    def schedule_next_job(
        self, state: SchedulingStateRL
    ) -> Optional[SchedulingAction]:
        """
        Decide next job to schedule using RL policy

        Args:
            state: Current factory state

        Returns:
            Scheduling action or None
        """
        try:
            # Get RL recommendation
            action = self.policy.predict_job_machine_assignment(state)

            if action is not None:
                self.stats["rl_decisions"] += 1
                return action
            else:
                # RL returned no-op or invalid action
                self.stats["invalid_actions"] += 1

        except Exception as e:
            print(f"RL policy error: {e}")
            self.stats["invalid_actions"] += 1

        # Fallback to rule-based scheduling
        if self.fallback_enabled:
            return self._fallback_schedule(state)

        return None

    def _fallback_schedule(
        self, state: SchedulingStateRL
    ) -> Optional[SchedulingAction]:
        """
        Simple rule-based fallback scheduler

        Priority-based scheduling with earliest deadline first
        """
        if not state.pending_jobs or not any(
            m.is_available for m in state.machines
        ):
            return None

        # Find highest priority job with nearest deadline
        sorted_jobs = sorted(
            state.pending_jobs,
            key=lambda j: (-j.priority, j.deadline, j.arrival_time),
        )

        # Find available machine with lowest queue
        available_machines = [m for m in state.machines if m.is_available]
        if not available_machines:
            return None

        selected_machine = min(available_machines, key=lambda m: m.queue_length)
        selected_job = sorted_jobs[0]

        self.stats["fallback_decisions"] += 1

        return SchedulingAction(
            job_id=selected_job.job_id,
            machine_id=selected_machine.machine_id,
            priority_boost=0.0,
        )

    def get_statistics(self) -> Dict:
        """Get scheduler performance statistics"""
        total = sum(self.stats.values())
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "rl_decision_rate": self.stats["rl_decisions"] / total,
            "fallback_rate": self.stats["fallback_decisions"] / total,
            "invalid_rate": self.stats["invalid_actions"] / total,
        }

    def reset_statistics(self):
        """Reset statistics counters"""
        for key in self.stats:
            self.stats[key] = 0
