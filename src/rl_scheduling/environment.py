"""
Production Scheduling Gymnasium Environment for RL Training
Custom environment for learning optimal scheduling policies
"""

from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datetime import datetime, timedelta
import random

from .state import (
    SchedulingStateRL,
    MachineStateRL,
    JobStateRL,
    SchedulingAction,
)
from .reward import RewardCalculator, RewardConfig


class ProductionSchedulingEnv(gym.Env):
    """
    Gymnasium environment for production scheduling

    State Space:
        - Machine states (availability, utilization, queue, health)
        - Job states (priority, deadlines, processing times)
        - Global metrics (utilization, queues, overdue count)

    Action Space:
        - Discrete: Which (job, machine) pair to schedule
        - Or Continuous: Job selection weight + machine selection weight

    Reward:
        - Positive: Job completion, high utilization, on-time delivery
        - Negative: Tardiness, idle time, queue imbalance, machine failures
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        num_machines: int = 3,
        max_jobs_per_episode: int = 50,
        episode_duration: int = 2000,  # minutes
        reward_config: Optional[RewardConfig] = None,
        use_discrete_actions: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.num_machines = num_machines
        self.max_jobs_per_episode = max_jobs_per_episode
        self.episode_duration = episode_duration
        self.use_discrete_actions = use_discrete_actions

        # Reward calculator
        self.reward_calculator = RewardCalculator(reward_config)

        # Define action space
        if use_discrete_actions:
            # Action: index of (job, machine) pair
            # Max actions = max_pending_jobs × num_machines + 1 (no-op)
            max_actions = 10 * num_machines + 1
            self.action_space = spaces.Discrete(max_actions)
        else:
            # Continuous: [job_selection_weights, machine_selection_weights]
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(20,), dtype=np.float32  # 10 jobs + 10 machines
            )

        # Define observation space
        obs_size = SchedulingStateRL.get_observation_space_size(
            num_machines=num_machines, max_pending_jobs=10
        )
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        # Internal state
        self.state: Optional[SchedulingStateRL] = None
        self.current_step = 0
        self.rng = np.random.default_rng(seed)

        # Job generation parameters
        self.job_arrival_rate = 0.05  # probability per time step
        self.next_job_id = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            random.seed(seed)
            np.random.seed(seed)

        # Initialize machines
        machines = []
        for i in range(self.num_machines):
            machine = MachineStateRL(
                machine_id=f"M{i+1:03d}",
                is_available=True,
                current_utilization=0.0,
                queue_length=0,
                processing_job_id=None,
                estimated_completion_time=None,
                failure_probability=self.rng.random() * 0.1,  # Low initial failure risk
                health_score=0.9 + self.rng.random() * 0.1,  # High initial health
            )
            machines.append(machine)

        # Initialize with some pending jobs
        initial_jobs = self._generate_initial_jobs(5)

        # Create initial state
        self.state = SchedulingStateRL(
            machines=machines,
            pending_jobs=initial_jobs,
            active_jobs=[],
            completed_jobs=[],
            current_time=0.0,
            time_step=0,
            total_utilization=0.0,
            average_queue_length=0.0,
            num_overdue_jobs=0,
            total_waiting_time=0.0,
        )

        self.current_step = 0
        self.next_job_id = len(initial_jobs)

        observation = self.state.to_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step

        Args:
            action: Action to take (job-machine assignment index)

        Returns:
            observation: New state observation
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        prev_state = self._copy_state(self.state)

        # Decode and execute action
        action_info = self._execute_action(action)

        # Advance simulation time
        time_delta = 5.0  # 5 minutes per step
        self._advance_time(time_delta)

        # Generate new jobs stochastically
        if self.rng.random() < self.job_arrival_rate:
            new_job = self._generate_job()
            if new_job and len(self.state.pending_jobs) < 20:
                self.state.pending_jobs.append(new_job)

        # Update state metrics
        self._update_metrics()

        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            prev_state, action_info, self.state, action_info
        )

        # Check termination conditions
        self.current_step += 1
        terminated = self.state.current_time >= self.episode_duration
        truncated = self.current_step >= 500  # Max steps per episode

        # Episode bonus
        if terminated or truncated:
            episode_bonus = self.reward_calculator.calculate_episode_bonus(
                self.state, self._get_info()
            )
            reward += episode_bonus

        observation = self.state.to_observation()
        info = self._get_info()
        info.update(action_info)

        return observation, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> Dict:
        """Execute the action and return info"""
        info = {"action_valid": False, "job_assigned": None, "machine_assigned": None}

        if self.use_discrete_actions:
            # Decode discrete action
            if action >= len(self.state.pending_jobs) * self.num_machines:
                # No-op action
                info["action_valid"] = True
                return info

            job_idx = action // self.num_machines
            machine_idx = action % self.num_machines

            # Check validity
            if job_idx >= len(self.state.pending_jobs):
                return info  # Invalid action

            job = self.state.pending_jobs[job_idx]
            machine = self.state.machines[machine_idx]

            if not machine.is_available:
                return info  # Machine not available

            # Assign job to machine
            self._assign_job_to_machine(job, machine)
            info["action_valid"] = True
            info["job_assigned"] = job.job_id
            info["machine_assigned"] = machine.machine_id

        return info

    def _assign_job_to_machine(self, job: JobStateRL, machine: MachineStateRL):
        """Assign a job to a machine"""
        # Remove from pending
        if job in self.state.pending_jobs:
            self.state.pending_jobs.remove(job)

        # Add to active
        if job not in self.state.active_jobs:
            self.state.active_jobs.append(job)

        # Update machine
        machine.is_available = False
        machine.processing_job_id = job.job_id
        machine.queue_length += 1
        machine.estimated_completion_time = job.total_processing_time

    def _advance_time(self, delta: float):
        """Advance simulation time by delta minutes"""
        self.state.current_time += delta
        self.state.time_step += 1

        # Update job waiting times
        for job in self.state.pending_jobs:
            job.waiting_time += delta
            if self.state.current_time > job.deadline and not job.is_overdue:
                job.is_overdue = True

        # Process active jobs
        for job in self.state.active_jobs[:]:
            job.total_processing_time -= delta

            # Check if job completed
            if job.total_processing_time <= 0:
                # Find machine processing this job
                for machine in self.state.machines:
                    if machine.processing_job_id == job.job_id:
                        machine.is_available = True
                        machine.processing_job_id = None
                        machine.queue_length = max(0, machine.queue_length - 1)
                        machine.estimated_completion_time = None
                        break

                # Move to completed
                self.state.active_jobs.remove(job)
                self.state.completed_jobs.append(job)

        # Update machine utilization
        for machine in self.state.machines:
            if not machine.is_available:
                machine.current_utilization = min(1.0, machine.current_utilization + 0.01)
            else:
                machine.current_utilization = max(0.0, machine.current_utilization - 0.02)

            # Gradual health degradation
            if not machine.is_available:
                machine.health_score = max(0.5, machine.health_score - 0.0001)
                machine.failure_probability = min(0.5, machine.failure_probability + 0.0001)

    def _update_metrics(self):
        """Update aggregate state metrics"""
        # Total utilization
        if len(self.state.machines) > 0:
            self.state.total_utilization = np.mean(
                [m.current_utilization for m in self.state.machines]
            )
            self.state.average_queue_length = np.mean(
                [m.queue_length for m in self.state.machines]
            )

        # Overdue jobs
        self.state.num_overdue_jobs = sum(
            1 for job in self.state.pending_jobs + self.state.active_jobs if job.is_overdue
        )

        # Total waiting time
        self.state.total_waiting_time = sum(job.waiting_time for job in self.state.pending_jobs)

    def _generate_initial_jobs(self, count: int) -> List[JobStateRL]:
        """Generate initial set of jobs"""
        jobs = []
        for _ in range(count):
            job = self._generate_job()
            if job:
                jobs.append(job)
        return jobs

    def _generate_job(self) -> Optional[JobStateRL]:
        """Generate a new job with random attributes"""
        job_id = f"J{self.next_job_id:04d}"
        self.next_job_id += 1

        # Random attributes
        priority = int(self.rng.integers(1, 11))
        processing_time = float(self.rng.integers(20, 200))  # 20-200 minutes
        deadline_margin = float(self.rng.integers(100, 500))  # deadline = now + margin
        deadline = self.state.current_time + deadline_margin if self.state else deadline_margin

        job = JobStateRL(
            job_id=job_id,
            priority=priority,
            arrival_time=self.state.current_time if self.state else 0.0,
            deadline=deadline,
            remaining_tasks=1,
            current_task_idx=0,
            total_processing_time=processing_time,
            waiting_time=0.0,
            is_overdue=False,
        )

        return job

    def _copy_state(self, state: SchedulingStateRL) -> SchedulingStateRL:
        """Create a copy of the state"""
        import copy

        return copy.deepcopy(state)

    def _get_info(self) -> Dict:
        """Get additional information about current state"""
        return {
            "time": self.state.current_time,
            "step": self.state.time_step,
            "utilization": self.state.total_utilization,
            "num_pending": len(self.state.pending_jobs),
            "num_active": len(self.state.active_jobs),
            "num_completed": len(self.state.completed_jobs),
            "num_overdue": self.state.num_overdue_jobs,
            "avg_queue": self.state.average_queue_length,
        }

    def render(self):
        """Render the environment (optional)"""
        if self.state is None:
            return

        print(f"\n=== Time: {self.state.current_time:.1f} min ===")
        print(
            f"Utilization: {self.state.total_utilization:.2%} | "
            f"Pending: {len(self.state.pending_jobs)} | "
            f"Active: {len(self.state.active_jobs)} | "
            f"Completed: {len(self.state.completed_jobs)} | "
            f"Overdue: {self.state.num_overdue_jobs}"
        )
        print("\nMachines:")
        for machine in self.state.machines:
            status = "BUSY" if not machine.is_available else "IDLE"
            print(
                f"  {machine.machine_id}: {status} | "
                f"Util: {machine.current_utilization:.2%} | "
                f"Queue: {machine.queue_length} | "
                f"Health: {machine.health_score:.2%}"
            )

    def close(self):
        """Clean up resources"""
        pass
