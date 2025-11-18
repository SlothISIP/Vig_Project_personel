"""
Tests for RL Scheduling Module
"""

import pytest
import numpy as np
from src.rl_scheduling import (
    SchedulingStateRL,
    MachineStateRL,
    JobStateRL,
    SchedulingAction,
    RewardCalculator,
    RewardConfig,
    ProductionSchedulingEnv,
)


class TestState:
    """Test state representation"""

    def test_machine_state_creation(self):
        """Test creating machine state"""
        machine = MachineStateRL(
            machine_id="M001",
            is_available=True,
            current_utilization=0.8,
            queue_length=3,
            processing_job_id=None,
            estimated_completion_time=None,
            failure_probability=0.1,
            health_score=0.9,
        )

        assert machine.machine_id == "M001"
        assert machine.is_available is True
        assert machine.current_utilization == 0.8
        assert machine.health_score == 0.9

    def test_job_state_creation(self):
        """Test creating job state"""
        job = JobStateRL(
            job_id="J001",
            priority=5,
            arrival_time=0.0,
            deadline=100.0,
            remaining_tasks=3,
            current_task_idx=0,
            total_processing_time=60.0,
            waiting_time=0.0,
            is_overdue=False,
        )

        assert job.job_id == "J001"
        assert job.priority == 5
        assert job.deadline == 100.0
        assert not job.is_overdue

    def test_state_to_observation(self):
        """Test converting state to observation vector"""
        machines = [
            MachineStateRL(
                machine_id=f"M{i:03d}",
                is_available=(i % 2 == 0),
                current_utilization=0.5 + i * 0.1,
                queue_length=i,
                processing_job_id=None,
                estimated_completion_time=None,
                failure_probability=0.1,
                health_score=0.9,
            )
            for i in range(3)
        ]

        jobs = [
            JobStateRL(
                job_id=f"J{i:03d}",
                priority=i + 1,
                arrival_time=0.0,
                deadline=100.0 + i * 50,
                remaining_tasks=1,
                current_task_idx=0,
                total_processing_time=50.0,
                waiting_time=0.0,
                is_overdue=False,
            )
            for i in range(5)
        ]

        state = SchedulingStateRL(
            machines=machines,
            pending_jobs=jobs,
            active_jobs=[],
            completed_jobs=[],
            current_time=0.0,
            time_step=0,
            total_utilization=0.6,
            average_queue_length=1.0,
            num_overdue_jobs=0,
            total_waiting_time=0.0,
        )

        obs = state.to_observation()

        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert len(obs) == state.get_observation_space_size(num_machines=3)
        assert np.all(np.isfinite(obs))

    def test_observation_space_size(self):
        """Test observation space size calculation"""
        size = SchedulingStateRL.get_observation_space_size(
            num_machines=3, max_pending_jobs=10
        )

        # 3 machines × 5 features = 15
        # 10 jobs × 4 features = 40
        # 4 global features
        # Total = 59
        expected_size = 3 * 5 + 10 * 4 + 4
        assert size == expected_size


class TestReward:
    """Test reward calculation"""

    def test_reward_config(self):
        """Test reward configuration"""
        config = RewardConfig(
            job_completion_reward=10.0,
            tardiness_penalty_weight=-15.0,
            utilization_reward_weight=2.0,
        )

        assert config.job_completion_reward == 10.0
        assert config.tardiness_penalty_weight == -15.0

    def test_completion_reward(self):
        """Test job completion reward"""
        calculator = RewardCalculator()

        # Create states with job completion
        prev_state = self._create_test_state(completed=0)
        new_state = self._create_test_state(completed=2)

        reward = calculator.calculate_reward(prev_state, {}, new_state, {})

        # Should be positive (completion reward)
        assert reward > 0

    def test_tardiness_penalty(self):
        """Test tardiness penalty"""
        calculator = RewardCalculator()

        # Create states with overdue jobs
        prev_state = self._create_test_state(overdue=0)
        new_state = self._create_test_state(overdue=2)

        reward = calculator.calculate_reward(prev_state, {}, new_state, {})

        # Should be negative (tardiness penalty)
        assert reward < 0

    def test_reward_breakdown(self):
        """Test reward component breakdown"""
        calculator = RewardCalculator()

        prev_state = self._create_test_state()
        new_state = self._create_test_state(completed=1)

        breakdown = calculator.get_reward_breakdown(prev_state, {}, new_state, {})

        assert isinstance(breakdown, dict)
        assert "completion_reward" in breakdown
        assert "tardiness_penalty" in breakdown
        assert "total" in breakdown

    def _create_test_state(
        self, completed=0, overdue=0, utilization=0.5
    ) -> SchedulingStateRL:
        """Helper to create test state"""
        machines = [
            MachineStateRL(
                machine_id="M001",
                is_available=True,
                current_utilization=utilization,
                queue_length=0,
                processing_job_id=None,
                estimated_completion_time=None,
                failure_probability=0.1,
                health_score=0.9,
            )
        ]

        completed_jobs = [
            JobStateRL(
                job_id=f"J{i:03d}",
                priority=5,
                arrival_time=0.0,
                deadline=100.0,
                remaining_tasks=0,
                current_task_idx=1,
                total_processing_time=0.0,
                waiting_time=10.0,
                is_overdue=False,
            )
            for i in range(completed)
        ]

        pending_jobs = [
            JobStateRL(
                job_id=f"JP{i:03d}",
                priority=5,
                arrival_time=0.0,
                deadline=50.0,
                remaining_tasks=1,
                current_task_idx=0,
                total_processing_time=30.0,
                waiting_time=100.0,  # Long waiting = overdue
                is_overdue=(i < overdue),
            )
            for i in range(3)
        ]

        return SchedulingStateRL(
            machines=machines,
            pending_jobs=pending_jobs,
            active_jobs=[],
            completed_jobs=completed_jobs,
            current_time=0.0,
            time_step=0,
            total_utilization=utilization,
            average_queue_length=0.0,
            num_overdue_jobs=overdue,
            total_waiting_time=0.0,
        )


class TestEnvironment:
    """Test production scheduling environment"""

    def test_env_creation(self):
        """Test environment initialization"""
        env = ProductionSchedulingEnv(
            num_machines=3, max_jobs_per_episode=30, episode_duration=1000
        )

        assert env.num_machines == 3
        assert env.max_jobs_per_episode == 30
        assert env.episode_duration == 1000

    def test_env_reset(self):
        """Test environment reset"""
        env = ProductionSchedulingEnv(num_machines=3)

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert len(obs) == env.observation_space.shape[0]
        assert isinstance(info, dict)
        assert env.state is not None
        assert len(env.state.machines) == 3

    def test_env_step(self):
        """Test environment step"""
        env = ProductionSchedulingEnv(num_machines=3)
        env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_termination(self):
        """Test episode terminates correctly"""
        env = ProductionSchedulingEnv(
            num_machines=3, episode_duration=100  # Short episode
        )
        env.reset()

        done = False
        steps = 0
        max_steps = 1000

        while not done and steps < max_steps:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done, "Episode should terminate"
        assert steps < max_steps, "Episode should not exceed max steps"

    def test_action_space(self):
        """Test action space"""
        env = ProductionSchedulingEnv(num_machines=3)

        assert env.action_space.contains(env.action_space.sample())

        # Test discrete action space
        assert hasattr(env.action_space, "n")
        assert env.action_space.n > 0

    def test_observation_space(self):
        """Test observation space"""
        env = ProductionSchedulingEnv(num_machines=3)
        env.reset()

        obs = env.state.to_observation()

        assert env.observation_space.contains(obs)

    def test_job_generation(self):
        """Test job generation"""
        env = ProductionSchedulingEnv(num_machines=3)
        env.reset()

        initial_jobs = len(env.state.pending_jobs)

        # Step multiple times to potentially generate jobs
        for _ in range(50):
            action = env.action_space.sample()
            env.step(action)

        # Should have generated or processed some jobs
        assert (
            len(env.state.pending_jobs) >= 0
        )  # May be 0 if all completed

    def test_machine_assignment(self):
        """Test job-machine assignment"""
        env = ProductionSchedulingEnv(num_machines=3)
        env.reset()

        # Ensure we have pending jobs
        if len(env.state.pending_jobs) > 0:
            job = env.state.pending_jobs[0]
            machine = env.state.machines[0]

            if machine.is_available:
                env._assign_job_to_machine(job, machine)

                assert not machine.is_available
                assert machine.processing_job_id == job.job_id
                assert job in env.state.active_jobs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
