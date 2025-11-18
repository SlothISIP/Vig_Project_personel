"""
Reward Function for RL-based Production Scheduling
Defines the reward signal that guides the RL agent
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from .state import SchedulingStateRL, JobStateRL


@dataclass
class RewardConfig:
    """Configuration for reward function weights"""

    # Positive rewards
    job_completion_reward: float = 10.0
    early_completion_bonus: float = 5.0
    utilization_reward_weight: float = 2.0
    throughput_reward_weight: float = 3.0

    # Negative rewards (penalties)
    tardiness_penalty_weight: float = -15.0
    waiting_time_penalty_weight: float = -0.1
    idle_machine_penalty: float = -1.0
    queue_imbalance_penalty: float = -2.0
    machine_failure_risk_penalty: float = -5.0

    # Bonus multipliers
    high_priority_multiplier: float = 1.5
    deadline_urgency_multiplier: float = 2.0


class RewardCalculator:
    """Calculate rewards for the scheduling environment"""

    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()

    def calculate_reward(
        self,
        prev_state: SchedulingStateRL,
        action: Dict,
        new_state: SchedulingStateRL,
        info: Dict,
    ) -> float:
        """
        Calculate reward for the state transition

        Args:
            prev_state: Previous environment state
            action: Action taken (job-machine assignment)
            new_state: Resulting state after action
            info: Additional information about the transition

        Returns:
            Total reward value
        """
        reward = 0.0

        # 1. Job completion rewards
        num_completed = len(new_state.completed_jobs) - len(prev_state.completed_jobs)
        if num_completed > 0:
            # Base completion reward
            reward += self.config.job_completion_reward * num_completed

            # Early completion bonus
            for job in new_state.completed_jobs[-num_completed:]:
                if not job.is_overdue:
                    time_margin = job.deadline - new_state.current_time
                    if time_margin > 0:
                        bonus = self.config.early_completion_bonus * (
                            time_margin / job.total_processing_time
                        )
                        reward += bonus

                    # High priority bonus
                    if job.priority >= 8:
                        reward += (
                            self.config.job_completion_reward
                            * self.config.high_priority_multiplier
                        )

        # 2. Tardiness penalties
        new_overdue = new_state.num_overdue_jobs - prev_state.num_overdue_jobs
        if new_overdue > 0:
            penalty = self.config.tardiness_penalty_weight * new_overdue
            reward += penalty

        # Ongoing tardiness penalty
        for job in new_state.pending_jobs + new_state.active_jobs:
            if job.is_overdue:
                tardiness = new_state.current_time - job.deadline
                penalty = self.config.tardiness_penalty_weight * (tardiness / 100.0)
                reward += penalty

                # Severe penalty for high-priority overdue jobs
                if job.priority >= 8:
                    reward += penalty * self.config.deadline_urgency_multiplier

        # 3. Utilization reward
        utilization_improvement = new_state.total_utilization - prev_state.total_utilization
        reward += self.config.utilization_reward_weight * utilization_improvement

        # Penalty for low overall utilization
        if new_state.total_utilization < 0.5:
            reward += self.config.idle_machine_penalty * (0.5 - new_state.total_utilization)

        # 4. Waiting time penalty
        total_waiting_delta = new_state.total_waiting_time - prev_state.total_waiting_time
        reward += self.config.waiting_time_penalty_weight * (total_waiting_delta / 60.0)

        # 5. Queue balance reward/penalty
        queue_lengths = [m.queue_length for m in new_state.machines]
        if len(queue_lengths) > 0:
            queue_std = np.std(queue_lengths)
            avg_queue = np.mean(queue_lengths)

            # Penalty for imbalanced queues
            if avg_queue > 0:
                imbalance_ratio = queue_std / (avg_queue + 1)
                reward += self.config.queue_imbalance_penalty * imbalance_ratio

        # 6. Machine health and failure risk
        for machine in new_state.machines:
            # Penalty for using machines with high failure risk
            if machine.failure_probability > 0.3 and not machine.is_available:
                risk_penalty = (
                    self.config.machine_failure_risk_penalty * machine.failure_probability
                )
                reward += risk_penalty

            # Penalty for poor health machines
            if machine.health_score < 0.7 and not machine.is_available:
                health_penalty = self.config.machine_failure_risk_penalty * (
                    0.7 - machine.health_score
                )
                reward += health_penalty

        # 7. Throughput reward (jobs completed per time unit)
        time_delta = new_state.current_time - prev_state.current_time
        if time_delta > 0 and num_completed > 0:
            throughput = num_completed / time_delta
            reward += self.config.throughput_reward_weight * throughput * 100

        # 8. Action quality (if applicable)
        if "action_valid" in info and not info["action_valid"]:
            reward -= 5.0  # Penalty for invalid actions

        return reward

    def calculate_episode_bonus(self, final_state: SchedulingStateRL, info: Dict) -> float:
        """
        Calculate bonus/penalty at the end of an episode

        Args:
            final_state: Final state of the episode
            info: Episode information

        Returns:
            Bonus reward
        """
        bonus = 0.0

        # Completion rate bonus
        total_jobs = (
            len(final_state.completed_jobs)
            + len(final_state.active_jobs)
            + len(final_state.pending_jobs)
        )
        if total_jobs > 0:
            completion_rate = len(final_state.completed_jobs) / total_jobs
            bonus += 20.0 * completion_rate

        # On-time delivery rate bonus
        if len(final_state.completed_jobs) > 0:
            on_time_jobs = sum(1 for job in final_state.completed_jobs if not job.is_overdue)
            on_time_rate = on_time_jobs / len(final_state.completed_jobs)
            bonus += 30.0 * on_time_rate

        # Average utilization bonus
        if final_state.total_utilization > 0.7:
            bonus += 10.0 * (final_state.total_utilization - 0.7)

        # Penalty for many overdue jobs
        if final_state.num_overdue_jobs > 5:
            bonus -= 50.0

        return bonus

    def get_reward_breakdown(
        self,
        prev_state: SchedulingStateRL,
        action: Dict,
        new_state: SchedulingStateRL,
        info: Dict,
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components for debugging/analysis

        Returns:
            Dictionary with reward component names and values
        """
        breakdown = {}

        # Calculate individual components
        num_completed = len(new_state.completed_jobs) - len(prev_state.completed_jobs)

        breakdown["completion_reward"] = self.config.job_completion_reward * num_completed
        breakdown["tardiness_penalty"] = self.config.tardiness_penalty_weight * (
            new_state.num_overdue_jobs - prev_state.num_overdue_jobs
        )
        breakdown["utilization_reward"] = self.config.utilization_reward_weight * (
            new_state.total_utilization - prev_state.total_utilization
        )
        breakdown["waiting_penalty"] = self.config.waiting_time_penalty_weight * (
            new_state.total_waiting_time - prev_state.total_waiting_time
        )

        # Queue balance
        queue_lengths = [m.queue_length for m in new_state.machines]
        if len(queue_lengths) > 0 and np.mean(queue_lengths) > 0:
            queue_std = np.std(queue_lengths)
            avg_queue = np.mean(queue_lengths)
            imbalance_ratio = queue_std / (avg_queue + 1)
            breakdown["queue_balance_penalty"] = (
                self.config.queue_imbalance_penalty * imbalance_ratio
            )
        else:
            breakdown["queue_balance_penalty"] = 0.0

        breakdown["total"] = sum(breakdown.values())

        return breakdown
