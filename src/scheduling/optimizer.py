"""Schedule Optimization and Re-scheduling.

Dynamic schedule optimization with multiple objectives.
"""

from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from src.scheduling.models import Job, Machine, Schedule, ScheduleAssignment
from src.scheduling.solvers.job_shop_solver import JobShopSolver, SolverConfig
from src.core.logging import get_logger

logger = get_logger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives."""

    MINIMIZE_MAKESPAN = "minimize_makespan"
    MINIMIZE_TARDINESS = "minimize_tardiness"
    MAXIMIZE_UTILIZATION = "maximize_utilization"
    MINIMIZE_CHANGEOVER = "minimize_changeover"


class ScheduleOptimizer:
    """
    Schedule optimizer with multi-objective optimization.

    Provides dynamic re-scheduling capabilities.
    """

    def __init__(self, solver_config: Optional[SolverConfig] = None):
        """
        Initialize optimizer.

        Args:
            solver_config: Configuration for the solver
        """
        self.solver_config = solver_config or SolverConfig()
        self.solver = JobShopSolver(config=self.solver_config)

    def optimize(
        self,
        jobs: List[Job],
        machines: List[Machine],
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_MAKESPAN,
    ) -> Optional[Schedule]:
        """
        Optimize schedule with specified objective.

        Args:
            jobs: Jobs to schedule
            machines: Available machines
            objective: Optimization objective

        Returns:
            Optimized schedule
        """
        logger.info(f"Optimizing schedule with objective: {objective.value}")

        # Currently only supports makespan minimization
        # Future: Add other objectives
        if objective == OptimizationObjective.MINIMIZE_MAKESPAN:
            return self.solver.solve(jobs, machines)
        else:
            logger.warning(
                f"Objective {objective.value} not yet implemented, "
                f"using makespan minimization"
            )
            return self.solver.solve(jobs, machines)

    def reschedule(
        self,
        current_schedule: Schedule,
        new_jobs: List[Job],
        machines: List[Machine],
        current_time: Optional[datetime] = None,
    ) -> Optional[Schedule]:
        """
        Re-schedule with new jobs while preserving in-progress tasks.

        Args:
            current_schedule: Current schedule
            new_jobs: New jobs to add
            machines: Available machines
            current_time: Current time (defaults to now)

        Returns:
            Updated schedule
        """
        if current_time is None:
            current_time = datetime.now()

        logger.info(
            f"Re-scheduling: {len(new_jobs)} new jobs, "
            f"{len(current_schedule.assignments)} existing assignments"
        )

        # Identify in-progress and future tasks
        in_progress = []
        future_assignments = []

        for assignment in current_schedule.assignments:
            if assignment.end_time <= current_time:
                # Completed - ignore
                continue
            elif assignment.start_time <= current_time < assignment.end_time:
                # In progress - must preserve
                in_progress.append(assignment)
            else:
                # Future - can re-schedule
                future_assignments.append(assignment)

        # Extract jobs from future assignments
        future_job_ids = set(a.job_id for a in future_assignments)

        # Combine with new jobs
        # Simplified: just solve with new jobs
        # In practice, would need to handle in-progress constraints
        all_jobs = new_jobs

        # Solve
        new_schedule = self.solver.solve(all_jobs, machines)

        if new_schedule:
            # Add back in-progress assignments
            for assignment in in_progress:
                new_schedule.add_assignment(assignment)

        return new_schedule

    def evaluate_schedule(
        self, schedule: Schedule, machines: List[Machine]
    ) -> Dict[str, float]:
        """
        Evaluate schedule quality on multiple metrics.

        Args:
            schedule: Schedule to evaluate
            machines: Machine list

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Makespan
        metrics["makespan"] = schedule.get_makespan() or 0

        # Utilization
        machine_ids = [m.machine_id for m in machines]
        metrics["average_utilization"] = schedule.get_average_utilization(machine_ids)

        # Individual machine utilizations
        for machine_id in machine_ids:
            metrics[f"utilization_{machine_id}"] = schedule.get_machine_utilization(
                machine_id
            )

        # Number of assignments
        metrics["total_assignments"] = len(schedule.assignments)

        # Conflicts
        metrics["has_conflicts"] = 1.0 if schedule.has_conflicts() else 0.0

        return metrics

    def compare_schedules(
        self, schedule1: Schedule, schedule2: Schedule, machines: List[Machine]
    ) -> Dict[str, any]:
        """
        Compare two schedules.

        Args:
            schedule1: First schedule
            schedule2: Second schedule
            machines: Machine list

        Returns:
            Comparison dictionary
        """
        metrics1 = self.evaluate_schedule(schedule1, machines)
        metrics2 = self.evaluate_schedule(schedule2, machines)

        comparison = {
            "schedule1": metrics1,
            "schedule2": metrics2,
            "differences": {},
        }

        # Calculate differences
        for key in metrics1.keys():
            if key in metrics2:
                diff = metrics2[key] - metrics1[key]
                pct_change = (
                    (diff / metrics1[key] * 100) if metrics1[key] != 0 else 0.0
                )
                comparison["differences"][key] = {
                    "absolute": diff,
                    "percent": pct_change,
                }

        return comparison
