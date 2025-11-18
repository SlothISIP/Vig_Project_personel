"""Job Shop Scheduling Solver using OR-Tools.

Solves flexible job shop scheduling problems with CP-SAT.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time

try:
    from ortools.sat.python import cp_model
except ImportError:
    cp_model = None

from src.scheduling.models import (
    Job,
    Task,
    Machine,
    Schedule,
    ScheduleAssignment,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SolverConfig:
    """Configuration for the scheduling solver."""

    max_time_seconds: int = 300  # Maximum solving time
    num_search_workers: int = 4  # Parallel search workers
    optimize_makespan: bool = True  # Minimize makespan
    optimize_tardiness: bool = False  # Minimize tardiness
    optimize_utilization: bool = False  # Maximize utilization

    # Logging
    log_search_progress: bool = False


class JobShopSolver:
    """
    Job Shop Scheduling solver using OR-Tools CP-SAT.

    Solves flexible job shop problems where:
    - Jobs consist of ordered tasks
    - Tasks can be assigned to multiple possible machines
    - Objective is to minimize makespan (or other objectives)
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        """
        Initialize solver.

        Args:
            config: Solver configuration
        """
        if cp_model is None:
            raise ImportError(
                "OR-Tools is not installed. Install with: pip install ortools"
            )

        self.config = config or SolverConfig()
        self.model: Optional[cp_model.CpModel] = None
        self.solver: Optional[cp_model.CpSolver] = None

    def solve(
        self,
        jobs: List[Job],
        machines: List[Machine],
        horizon: Optional[int] = None,
    ) -> Optional[Schedule]:
        """
        Solve job shop scheduling problem.

        Args:
            jobs: List of jobs to schedule
            machines: List of available machines
            horizon: Time horizon in minutes (auto-calculated if None)

        Returns:
            Optimized schedule, or None if no solution found
        """
        start_time = time.time()

        logger.info(
            f"Solving job shop problem: {len(jobs)} jobs, {len(machines)} machines"
        )

        # Calculate horizon if not provided
        if horizon is None:
            horizon = sum(job.get_total_duration() for job in jobs) * 2
            logger.info(f"Auto-calculated horizon: {horizon} minutes")

        # Create model
        self.model = cp_model.CpModel()

        # Create variables
        task_vars, machine_vars = self._create_variables(jobs, machines, horizon)

        # Add constraints
        self._add_constraints(jobs, machines, task_vars, machine_vars)

        # Add objective
        makespan_var = self._add_objective(jobs, task_vars)

        # Solve
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = self.config.max_time_seconds
        self.solver.parameters.num_search_workers = self.config.num_search_workers

        if self.config.log_search_progress:
            self.solver.parameters.log_search_progress = True

        status = self.solver.Solve(self.model)

        solve_time = time.time() - start_time

        # Extract solution
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            schedule = self._extract_solution(
                jobs, task_vars, machine_vars, makespan_var
            )
            schedule.solver_time_seconds = solve_time
            schedule.objective_value = float(self.solver.ObjectiveValue())

            logger.info(
                f"Solution found in {solve_time:.2f}s - "
                f"Makespan: {self.solver.ObjectiveValue()} min, "
                f"Status: {'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'}"
            )

            return schedule
        else:
            logger.warning(f"No solution found. Status: {status}")
            return None

    def _create_variables(
        self, jobs: List[Job], machines: List[Machine], horizon: int
    ) -> Tuple[Dict, Dict]:
        """
        Create CP-SAT variables.

        Returns:
            (task_vars, machine_vars) dictionaries
        """
        task_vars = {}  # (job_id, task_id) -> IntervalVar
        machine_vars = {}  # (job_id, task_id, machine_id) -> IntervalVar

        for job in jobs:
            for task in job.tasks:
                # Create start and end variables
                start_var = self.model.NewIntVar(
                    0, horizon, f"start_{job.job_id}_{task.task_id}"
                )
                end_var = self.model.NewIntVar(
                    0, horizon, f"end_{job.job_id}_{task.task_id}"
                )

                # Create interval variable for the task
                interval_var = self.model.NewIntervalVar(
                    start_var,
                    task.duration,
                    end_var,
                    f"interval_{job.job_id}_{task.task_id}",
                )

                task_vars[(job.job_id, task.task_id)] = {
                    "start": start_var,
                    "end": end_var,
                    "interval": interval_var,
                }

                # Create machine assignment variables
                # For simplicity, assume all machines can handle all tasks
                for machine in machines:
                    if machine.can_execute(task.task_type):
                        # Optional interval for this machine
                        machine_presence = self.model.NewBoolVar(
                            f"presence_{job.job_id}_{task.task_id}_{machine.machine_id}"
                        )

                        machine_interval = self.model.NewOptionalIntervalVar(
                            start_var,
                            task.duration,
                            end_var,
                            machine_presence,
                            f"machine_interval_{job.job_id}_{task.task_id}_{machine.machine_id}",
                        )

                        machine_vars[(job.job_id, task.task_id, machine.machine_id)] = {
                            "presence": machine_presence,
                            "interval": machine_interval,
                        }

        return task_vars, machine_vars

    def _add_constraints(
        self,
        jobs: List[Job],
        machines: List[Machine],
        task_vars: Dict,
        machine_vars: Dict,
    ) -> None:
        """Add scheduling constraints."""

        # 1. Precedence constraints (task dependencies)
        for job in jobs:
            for task in job.tasks:
                for pred_task_id in task.predecessor_task_ids:
                    # Find predecessor task
                    pred_task = job.get_task(pred_task_id)
                    if pred_task:
                        # Current task must start after predecessor ends
                        self.model.Add(
                            task_vars[(job.job_id, task.task_id)]["start"]
                            >= task_vars[(job.job_id, pred_task_id)]["end"]
                        )

        # 2. Machine assignment constraints (each task assigned to exactly one machine)
        for job in jobs:
            for task in job.tasks:
                # Exactly one machine must be selected
                machine_presences = [
                    machine_vars[(job.job_id, task.task_id, machine.machine_id)][
                        "presence"
                    ]
                    for machine in machines
                    if (job.job_id, task.task_id, machine.machine_id) in machine_vars
                ]

                if machine_presences:
                    self.model.Add(sum(machine_presences) == 1)

        # 3. No overlap constraints (no two tasks on same machine at same time)
        for machine in machines:
            machine_intervals = []

            for job in jobs:
                for task in job.tasks:
                    key = (job.job_id, task.task_id, machine.machine_id)
                    if key in machine_vars:
                        machine_intervals.append(machine_vars[key]["interval"])

            if machine_intervals:
                self.model.AddNoOverlap(machine_intervals)

    def _add_objective(self, jobs: List[Job], task_vars: Dict) -> cp_model.IntVar:
        """
        Add optimization objective.

        Returns:
            Makespan variable
        """
        # Makespan = maximum end time of all tasks
        makespan_var = self.model.NewIntVar(0, 10000, "makespan")

        # Makespan must be >= all task end times
        for job in jobs:
            for task in job.tasks:
                self.model.Add(
                    makespan_var >= task_vars[(job.job_id, task.task_id)]["end"]
                )

        # Minimize makespan
        self.model.Minimize(makespan_var)

        return makespan_var

    def _extract_solution(
        self,
        jobs: List[Job],
        task_vars: Dict,
        machine_vars: Dict,
        makespan_var: cp_model.IntVar,
    ) -> Schedule:
        """Extract solution from solved model."""
        schedule = Schedule()

        # Reference start time (now)
        base_time = datetime.now()

        for job in jobs:
            for task in job.tasks:
                # Get start and end times
                start_min = self.solver.Value(
                    task_vars[(job.job_id, task.task_id)]["start"]
                )
                end_min = self.solver.Value(
                    task_vars[(job.job_id, task.task_id)]["end"]
                )

                # Find assigned machine
                assigned_machine = None
                for key, var_dict in machine_vars.items():
                    if key[0] == job.job_id and key[1] == task.task_id:
                        if self.solver.Value(var_dict["presence"]):
                            assigned_machine = key[2]
                            break

                if assigned_machine:
                    # Create assignment
                    assignment = ScheduleAssignment(
                        task_id=task.task_id,
                        job_id=job.job_id,
                        machine_id=assigned_machine,
                        start_time=base_time + timedelta(minutes=start_min),
                        end_time=base_time + timedelta(minutes=end_min),
                    )

                    schedule.add_assignment(assignment)

                    # Update task
                    task.schedule(
                        machine_id=assigned_machine,
                        start_time=assignment.start_time,
                        end_time=assignment.end_time,
                    )

        return schedule


def solve_simple_example() -> Optional[Schedule]:
    """
    Solve a simple example problem.

    Returns:
        Optimized schedule
    """
    from src.scheduling.models import create_sample_job, create_sample_machines

    # Create jobs
    jobs = [
        create_sample_job("Job1", num_tasks=3, task_duration=20),
        create_sample_job("Job2", num_tasks=3, task_duration=30),
        create_sample_job("Job3", num_tasks=2, task_duration=25),
    ]

    # Create machines
    machines = create_sample_machines(num_machines=2)

    # Solve
    solver = JobShopSolver(
        config=SolverConfig(max_time_seconds=60, log_search_progress=False)
    )

    schedule = solver.solve(jobs, machines)

    if schedule:
        stats = schedule.get_statistics()
        logger.info("Schedule statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    return schedule
