"""Tests for scheduling module."""

import pytest
from datetime import datetime, timedelta

from src.scheduling.models import (
    Job,
    Task,
    Machine,
    Schedule,
    ScheduleAssignment,
    MachineAvailability,
    TaskStatus,
    create_sample_job,
    create_sample_machines,
)
from src.scheduling.solvers.job_shop_solver import JobShopSolver, SolverConfig
from src.scheduling.optimizer import ScheduleOptimizer, OptimizationObjective
from src.scheduling.scheduler import ProductionScheduler


class TestModels:
    """Tests for data models."""

    def test_create_machine(self):
        """Test machine creation."""
        machine = Machine(
            machine_id="M1",
            machine_type="assembly",
            capabilities=["assemble", "inspect"],
        )

        assert machine.machine_id == "M1"
        assert machine.can_execute("assemble")
        assert machine.can_execute("inspect")
        assert not machine.can_execute("weld")

    def test_machine_reservation(self):
        """Test machine reservation."""
        machine = Machine(machine_id="M1", machine_type="assembly")

        until = datetime.now() + timedelta(hours=2)
        machine.reserve("Job1", until)

        assert machine.availability == MachineAvailability.BUSY
        assert machine.current_job_id == "Job1"
        assert machine.available_from == until

        machine.release()

        assert machine.availability == MachineAvailability.AVAILABLE
        assert machine.current_job_id is None

    def test_create_task(self):
        """Test task creation."""
        task = Task(
            task_id="T1",
            task_type="assemble",
            duration=30,
            predecessor_task_ids=["T0"],
        )

        assert task.task_id == "T1"
        assert task.duration == 30
        assert task.status == TaskStatus.PENDING

    def test_task_can_start(self):
        """Test task prerequisite checking."""
        task = Task(
            task_id="T2",
            task_type="inspect",
            duration=15,
            predecessor_task_ids=["T1"],
        )

        assert not task.can_start(set())
        assert not task.can_start({"T0"})
        assert task.can_start({"T1"})
        assert task.can_start({"T1", "T0"})

    def test_create_job(self):
        """Test job creation."""
        job = create_sample_job("Job1", num_tasks=3, task_duration=20)

        assert job.job_id == "Job1"
        assert len(job.tasks) == 3
        assert job.get_total_duration() == 60  # 3 * 20

    def test_job_critical_path(self):
        """Test critical path calculation."""
        job = create_sample_job("Job1", num_tasks=4, task_duration=25)

        # Sequential tasks should have critical path = sum of durations
        critical_path = job.get_critical_path_length()
        assert critical_path == 100  # 4 * 25

    def test_schedule_assignment(self):
        """Test schedule assignment."""
        now = datetime.now()
        assignment = ScheduleAssignment(
            task_id="T1",
            job_id="Job1",
            machine_id="M1",
            start_time=now,
            end_time=now + timedelta(minutes=30),
        )

        assert assignment.get_duration_minutes() == 30

    def test_assignment_overlap(self):
        """Test assignment overlap detection."""
        now = datetime.now()

        assignment1 = ScheduleAssignment(
            task_id="T1",
            job_id="Job1",
            machine_id="M1",
            start_time=now,
            end_time=now + timedelta(minutes=30),
        )

        assignment2 = ScheduleAssignment(
            task_id="T2",
            job_id="Job2",
            machine_id="M1",
            start_time=now + timedelta(minutes=15),
            end_time=now + timedelta(minutes=45),
        )

        assignment3 = ScheduleAssignment(
            task_id="T3",
            job_id="Job3",
            machine_id="M2",  # Different machine
            start_time=now + timedelta(minutes=15),
            end_time=now + timedelta(minutes=45),
        )

        assert assignment1.overlaps_with(assignment2)
        assert not assignment1.overlaps_with(assignment3)

    def test_schedule_creation(self):
        """Test schedule creation."""
        schedule = Schedule()

        assert len(schedule.assignments) == 0
        assert schedule.get_makespan() is None

    def test_schedule_makespan(self):
        """Test schedule makespan calculation."""
        schedule = Schedule()

        now = datetime.now()

        schedule.add_assignment(
            ScheduleAssignment(
                task_id="T1",
                job_id="Job1",
                machine_id="M1",
                start_time=now,
                end_time=now + timedelta(minutes=30),
            )
        )

        schedule.add_assignment(
            ScheduleAssignment(
                task_id="T2",
                job_id="Job2",
                machine_id="M2",
                start_time=now + timedelta(minutes=10),
                end_time=now + timedelta(minutes=50),
            )
        )

        makespan = schedule.get_makespan()
        assert makespan == 50  # Total time from start to end

    def test_schedule_utilization(self):
        """Test machine utilization calculation."""
        schedule = Schedule()

        now = datetime.now()

        # Add two 30-minute tasks on M1 within 100 minute makespan
        schedule.add_assignment(
            ScheduleAssignment(
                task_id="T1",
                job_id="Job1",
                machine_id="M1",
                start_time=now,
                end_time=now + timedelta(minutes=30),
            )
        )

        schedule.add_assignment(
            ScheduleAssignment(
                task_id="T2",
                job_id="Job2",
                machine_id="M1",
                start_time=now + timedelta(minutes=70),
                end_time=now + timedelta(minutes=100),
            )
        )

        utilization = schedule.get_machine_utilization("M1")
        assert utilization == 0.6  # 60 minutes / 100 minutes


class TestJobShopSolver:
    """Tests for job shop solver."""

    def test_create_solver(self):
        """Test solver creation."""
        config = SolverConfig(max_time_seconds=60)
        solver = JobShopSolver(config=config)

        assert solver.config.max_time_seconds == 60

    def test_solve_simple_problem(self):
        """Test solving a simple problem."""
        jobs = [
            create_sample_job("Job1", num_tasks=2, task_duration=20),
            create_sample_job("Job2", num_tasks=2, task_duration=25),
        ]

        machines = create_sample_machines(num_machines=2)

        config = SolverConfig(max_time_seconds=30)
        solver = JobShopSolver(config=config)

        schedule = solver.solve(jobs, machines)

        # Should find a solution
        assert schedule is not None
        assert len(schedule.assignments) == 4  # 2 jobs * 2 tasks
        assert not schedule.has_conflicts()


class TestScheduleOptimizer:
    """Tests for schedule optimizer."""

    def test_create_optimizer(self):
        """Test optimizer creation."""
        optimizer = ScheduleOptimizer()

        assert optimizer is not None

    def test_optimize_schedule(self):
        """Test schedule optimization."""
        jobs = [
            create_sample_job("Job1", num_tasks=2, task_duration=20),
            create_sample_job("Job2", num_tasks=2, task_duration=30),
        ]

        machines = create_sample_machines(num_machines=2)

        optimizer = ScheduleOptimizer(
            solver_config=SolverConfig(max_time_seconds=30)
        )

        schedule = optimizer.optimize(
            jobs, machines, objective=OptimizationObjective.MINIMIZE_MAKESPAN
        )

        assert schedule is not None

    def test_evaluate_schedule(self):
        """Test schedule evaluation."""
        jobs = [create_sample_job("Job1", num_tasks=2, task_duration=20)]
        machines = create_sample_machines(num_machines=1)

        optimizer = ScheduleOptimizer(
            solver_config=SolverConfig(max_time_seconds=30)
        )
        schedule = optimizer.optimize(jobs, machines)

        metrics = optimizer.evaluate_schedule(schedule, machines)

        assert "makespan" in metrics
        assert "average_utilization" in metrics
        assert "total_assignments" in metrics


class TestProductionScheduler:
    """Tests for production scheduler."""

    def test_create_scheduler(self):
        """Test scheduler creation."""
        machines = create_sample_machines(num_machines=2)
        scheduler = ProductionScheduler(machines=machines)

        assert len(scheduler.machines) == 2

    def test_schedule_jobs(self):
        """Test scheduling jobs."""
        machines = create_sample_machines(num_machines=2)
        scheduler = ProductionScheduler(
            machines=machines, config=SolverConfig(max_time_seconds=30)
        )

        jobs = [
            create_sample_job("Job1", num_tasks=2, task_duration=20),
            create_sample_job("Job2", num_tasks=2, task_duration=25),
        ]

        result = scheduler.schedule_jobs(jobs)

        assert result.success
        assert result.schedule is not None
        assert "makespan" in result.metrics

    def test_get_machine_status(self):
        """Test getting machine status."""
        machines = create_sample_machines(num_machines=2)
        scheduler = ProductionScheduler(machines=machines)

        status = scheduler.get_machine_status("M1")

        assert status["machine_id"] == "M1"
        assert "availability" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
