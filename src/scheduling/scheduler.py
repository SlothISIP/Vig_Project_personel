"""Production Scheduler - High-level scheduling interface.

Provides easy-to-use interface for production scheduling.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

from src.scheduling.models import Job, Machine, Schedule
from src.scheduling.solvers.job_shop_solver import JobShopSolver, SolverConfig
from src.scheduling.optimizer import ScheduleOptimizer, OptimizationObjective
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SchedulingResult:
    """Result of a scheduling operation."""

    success: bool
    schedule: Optional[Schedule]
    message: str
    metrics: Dict[str, any]


class ProductionScheduler:
    """
    High-level production scheduler.

    Provides simple interface for scheduling operations.
    """

    def __init__(self, machines: List[Machine], config: Optional[SolverConfig] = None):
        """
        Initialize production scheduler.

        Args:
            machines: Available machines
            config: Solver configuration
        """
        self.machines = machines
        self.config = config or SolverConfig()
        self.optimizer = ScheduleOptimizer(solver_config=self.config)

        # Current schedule
        self.current_schedule: Optional[Schedule] = None

        # Schedule history for tracking multiple schedules
        self.schedule_history: List[Schedule] = []

        logger.info(
            f"Production scheduler initialized with {len(machines)} machines"
        )

    def schedule_jobs(
        self,
        jobs: List[Job],
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_MAKESPAN,
    ) -> SchedulingResult:
        """
        Schedule a list of jobs.

        Args:
            jobs: Jobs to schedule
            objective: Optimization objective

        Returns:
            SchedulingResult
        """
        logger.info(f"Scheduling {len(jobs)} jobs")

        # Optimize
        schedule = self.optimizer.optimize(jobs, self.machines, objective=objective)

        if schedule:
            # Store as current schedule
            self.current_schedule = schedule

            # Add to history
            self.schedule_history.append(schedule)

            # Evaluate
            metrics = self.optimizer.evaluate_schedule(schedule, self.machines)

            return SchedulingResult(
                success=True,
                schedule=schedule,
                message=f"Successfully scheduled {len(jobs)} jobs",
                metrics=metrics,
            )
        else:
            return SchedulingResult(
                success=False,
                schedule=None,
                message="Failed to find feasible schedule",
                metrics={},
            )

    def add_new_jobs(self, new_jobs: List[Job]) -> SchedulingResult:
        """
        Add new jobs to existing schedule.

        Args:
            new_jobs: New jobs to add

        Returns:
            SchedulingResult with updated schedule
        """
        logger.info(f"Adding {len(new_jobs)} new jobs to schedule")

        if self.current_schedule is None:
            # No existing schedule, just schedule new jobs
            return self.schedule_jobs(new_jobs)

        # Re-schedule
        new_schedule = self.optimizer.reschedule(
            self.current_schedule, new_jobs, self.machines
        )

        if new_schedule:
            self.current_schedule = new_schedule

            metrics = self.optimizer.evaluate_schedule(new_schedule, self.machines)

            return SchedulingResult(
                success=True,
                schedule=new_schedule,
                message=f"Successfully re-scheduled with {len(new_jobs)} new jobs",
                metrics=metrics,
            )
        else:
            return SchedulingResult(
                success=False,
                schedule=self.current_schedule,
                message="Failed to re-schedule, keeping current schedule",
                metrics={},
            )

    def get_current_schedule(self) -> Optional[Schedule]:
        """Get current schedule."""
        return self.current_schedule

    def get_all_schedules(self) -> List[Dict[str, any]]:
        """
        Get all schedules in history.

        Returns:
            List of schedule dictionaries with metadata
        """
        schedules = []

        for schedule in self.schedule_history:
            # Get unique machine IDs from assignments
            machine_ids = list(set(a.machine_id for a in schedule.assignments))

            # Extract unique job IDs from assignments
            job_ids = list(set(a.job_id for a in schedule.assignments))

            schedule_dict = {
                "schedule_id": schedule.schedule_id,
                "created_at": schedule.created_at.isoformat(),
                "makespan_minutes": schedule.get_makespan(),
                "objective_value": schedule.objective_value,
                "solver_time_seconds": schedule.solver_time_seconds,
                "utilization": schedule.get_average_utilization(machine_ids) if machine_ids else 0.0,
                "jobs": [
                    {
                        "job_id": job_id,
                        "status": "scheduled",
                        "assignments": [
                            {
                                "task_id": a.task_id,
                                "machine_id": a.machine_id,
                                "start_time": a.start_time.isoformat(),
                                "end_time": a.end_time.isoformat(),
                            }
                            for a in schedule.assignments if a.job_id == job_id
                        ]
                    }
                    for job_id in job_ids
                ],
                "assignments": [
                    {
                        "task_id": a.task_id,
                        "job_id": a.job_id,
                        "machine_id": a.machine_id,
                        "start_time": a.start_time.isoformat(),
                        "end_time": a.end_time.isoformat(),
                    }
                    for a in schedule.assignments
                ],
            }
            schedules.append(schedule_dict)

        return schedules

    def get_machine_status(self, machine_id: str) -> Dict[str, any]:
        """
        Get status of a specific machine.

        Args:
            machine_id: Machine ID

        Returns:
            Status dictionary
        """
        machine = None
        for m in self.machines:
            if m.machine_id == machine_id:
                machine = m
                break

        if machine is None:
            return {"error": "Machine not found"}

        status = {
            "machine_id": machine_id,
            "machine_type": machine.machine_type,
            "availability": machine.availability.value,
            "current_job": machine.current_job_id,
        }

        # Add schedule info if available
        if self.current_schedule:
            assignments = self.current_schedule.get_assignments_for_machine(machine_id)
            status["scheduled_tasks"] = len(assignments)
            status["utilization"] = self.current_schedule.get_machine_utilization(
                machine_id
            )

            # Next task
            now = datetime.now()
            future_assignments = [a for a in assignments if a.start_time > now]
            if future_assignments:
                next_task = min(future_assignments, key=lambda a: a.start_time)
                status["next_task"] = {
                    "task_id": next_task.task_id,
                    "job_id": next_task.job_id,
                    "start_time": next_task.start_time.isoformat(),
                }

        return status

    def get_job_status(self, job_id: str) -> Dict[str, any]:
        """
        Get status of a specific job.

        Args:
            job_id: Job ID

        Returns:
            Status dictionary
        """
        if self.current_schedule is None:
            return {"error": "No schedule available"}

        assignments = self.current_schedule.get_assignments_for_job(job_id)

        if not assignments:
            return {"error": "Job not found in schedule"}

        # Calculate progress
        now = datetime.now()
        completed = [a for a in assignments if a.end_time <= now]
        in_progress = [a for a in assignments if a.start_time <= now < a.end_time]
        future = [a for a in assignments if a.start_time > now]

        status = {
            "job_id": job_id,
            "total_tasks": len(assignments),
            "completed_tasks": len(completed),
            "in_progress_tasks": len(in_progress),
            "future_tasks": len(future),
            "progress_percent": (len(completed) / len(assignments) * 100)
            if assignments
            else 0,
        }

        # Estimated completion
        if assignments:
            last_assignment = max(assignments, key=lambda a: a.end_time)
            status["estimated_completion"] = last_assignment.end_time.isoformat()

        return status

    def get_overall_status(self) -> Dict[str, any]:
        """
        Get overall scheduling status.

        Returns:
            Status dictionary
        """
        if self.current_schedule is None:
            return {
                "status": "no_schedule",
                "message": "No schedule has been created yet",
            }

        stats = self.current_schedule.get_statistics()

        # Add machine statuses
        machine_statuses = {}
        for machine in self.machines:
            machine_statuses[machine.machine_id] = {
                "availability": machine.availability.value,
                "utilization": self.current_schedule.get_machine_utilization(
                    machine.machine_id
                ),
            }

        stats["machines"] = machine_statuses

        return stats

    def export_schedule(self, filepath: str) -> None:
        """
        Export current schedule to file.

        Args:
            filepath: Path to save schedule
        """
        if self.current_schedule is None:
            logger.warning("No schedule to export")
            return

        # Export as JSON
        import json

        data = {
            "schedule_id": self.current_schedule.schedule_id,
            "created_at": self.current_schedule.created_at.isoformat(),
            "assignments": [
                {
                    "task_id": a.task_id,
                    "job_id": a.job_id,
                    "machine_id": a.machine_id,
                    "start_time": a.start_time.isoformat(),
                    "end_time": a.end_time.isoformat(),
                }
                for a in self.current_schedule.assignments
            ],
            "statistics": self.current_schedule.get_statistics(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Schedule exported to {filepath}")


def create_demo_scheduler() -> ProductionScheduler:
    """
    Create a demo scheduler with sample data.

    Returns:
        Configured ProductionScheduler
    """
    from src.scheduling.models import create_sample_machines, create_sample_job

    # Create machines
    machines = create_sample_machines(num_machines=3)

    # Create scheduler
    config = SolverConfig(max_time_seconds=60, log_search_progress=False)
    scheduler = ProductionScheduler(machines=machines, config=config)

    # Create and schedule sample jobs
    jobs = [
        create_sample_job("Job1", num_tasks=4, task_duration=25),
        create_sample_job("Job2", num_tasks=3, task_duration=30),
        create_sample_job("Job3", num_tasks=5, task_duration=20),
    ]

    result = scheduler.schedule_jobs(jobs)

    if result.success:
        logger.info("Demo scheduler created with sample schedule")
        logger.info(f"Metrics: {result.metrics}")
    else:
        logger.warning("Failed to create sample schedule")

    return scheduler
