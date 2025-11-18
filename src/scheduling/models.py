"""Data Models for Production Scheduling.

Defines Job, Task, Machine, and Schedule data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Tuple
import uuid

from src.core.logging import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Status of a task."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class MachineAvailability(Enum):
    """Machine availability status."""

    AVAILABLE = "available"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    DOWN = "down"


@dataclass
class Machine:
    """
    Production machine that can execute tasks.

    Represents a physical machine with capabilities and availability.
    """

    machine_id: str
    machine_type: str
    capabilities: List[str] = field(default_factory=list)
    availability: MachineAvailability = MachineAvailability.AVAILABLE

    # Performance characteristics
    speed_factor: float = 1.0  # 1.0 = normal, >1.0 = faster, <1.0 = slower
    setup_time: int = 0  # Time required to setup for a new job (minutes)

    # Current state
    current_job_id: Optional[str] = None
    available_from: Optional[datetime] = None

    def can_execute(self, task_type: str) -> bool:
        """
        Check if machine can execute a task type.

        Args:
            task_type: Type of task

        Returns:
            True if machine has the capability
        """
        return task_type in self.capabilities or len(self.capabilities) == 0

    def is_available_at(self, time: datetime) -> bool:
        """
        Check if machine is available at a specific time.

        Args:
            time: Time to check

        Returns:
            True if available
        """
        if self.availability != MachineAvailability.AVAILABLE:
            return False

        if self.available_from is None:
            return True

        return time >= self.available_from

    def reserve(self, job_id: str, until: datetime) -> None:
        """
        Reserve machine for a job.

        Args:
            job_id: Job identifier
            until: Time until which machine is reserved
        """
        self.availability = MachineAvailability.BUSY
        self.current_job_id = job_id
        self.available_from = until

    def release(self) -> None:
        """Release machine from current job."""
        self.availability = MachineAvailability.AVAILABLE
        self.current_job_id = None


@dataclass
class Task:
    """
    A single task within a job.

    Tasks are atomic units of work that must be executed on a machine.
    """

    task_id: str
    task_type: str
    duration: int  # Duration in minutes
    required_capability: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING

    # Dependencies
    predecessor_task_ids: List[str] = field(default_factory=list)

    # Scheduling info
    assigned_machine_id: Optional[str] = None
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None

    # Priority (higher = more important)
    priority: int = 0

    def can_start(self, completed_task_ids: set) -> bool:
        """
        Check if task can start given completed tasks.

        Args:
            completed_task_ids: Set of completed task IDs

        Returns:
            True if all predecessors are completed
        """
        return all(pid in completed_task_ids for pid in self.predecessor_task_ids)

    def schedule(
        self, machine_id: str, start_time: datetime, end_time: datetime
    ) -> None:
        """
        Schedule this task.

        Args:
            machine_id: Machine to assign to
            start_time: Scheduled start time
            end_time: Scheduled end time
        """
        self.assigned_machine_id = machine_id
        self.scheduled_start = start_time
        self.scheduled_end = end_time
        self.status = TaskStatus.SCHEDULED

    def start_execution(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.actual_start = datetime.now()

    def complete_execution(self) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.actual_end = datetime.now()


@dataclass
class Job:
    """
    A production job consisting of multiple tasks.

    Jobs represent complete manufacturing orders that need to be scheduled.
    """

    job_id: str
    product_type: str
    tasks: List[Task] = field(default_factory=list)
    priority: int = 0  # Higher = more important

    # Due date
    due_date: Optional[datetime] = None
    release_date: Optional[datetime] = None  # Earliest start time

    # Metadata
    customer: Optional[str] = None
    order_quantity: int = 1

    def add_task(self, task: Task) -> None:
        """Add a task to this job."""
        self.tasks.append(task)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_total_duration(self) -> int:
        """Get total duration of all tasks (minutes)."""
        return sum(task.duration for task in self.tasks)

    def get_completed_tasks(self) -> set:
        """Get set of completed task IDs."""
        return {
            task.task_id
            for task in self.tasks
            if task.status == TaskStatus.COMPLETED
        }

    def is_completed(self) -> bool:
        """Check if all tasks are completed."""
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks)

    def get_critical_path_length(self) -> int:
        """
        Calculate critical path length (longest sequence).

        Returns:
            Critical path duration in minutes
        """
        # Build dependency graph
        task_durations = {task.task_id: task.duration for task in self.tasks}
        task_predecessors = {
            task.task_id: task.predecessor_task_ids for task in self.tasks
        }

        # Calculate earliest finish times
        earliest_finish = {}

        def calculate_earliest_finish(task_id: str) -> int:
            if task_id in earliest_finish:
                return earliest_finish[task_id]

            predecessors = task_predecessors.get(task_id, [])

            if not predecessors:
                earliest_finish[task_id] = task_durations[task_id]
            else:
                max_predecessor_finish = max(
                    calculate_earliest_finish(pred) for pred in predecessors
                )
                earliest_finish[task_id] = (
                    max_predecessor_finish + task_durations[task_id]
                )

            return earliest_finish[task_id]

        # Calculate for all tasks
        for task in self.tasks:
            calculate_earliest_finish(task.task_id)

        # Return maximum
        return max(earliest_finish.values()) if earliest_finish else 0


@dataclass
class ScheduleAssignment:
    """
    An assignment of a task to a machine at a specific time.

    Represents one entry in the schedule.
    """

    task_id: str
    job_id: str
    machine_id: str
    start_time: datetime
    end_time: datetime

    def get_duration_minutes(self) -> int:
        """Get duration in minutes."""
        return int((self.end_time - self.start_time).total_seconds() / 60)

    def overlaps_with(self, other: "ScheduleAssignment") -> bool:
        """
        Check if this assignment overlaps with another.

        Args:
            other: Other assignment

        Returns:
            True if assignments overlap in time on the same machine
        """
        if self.machine_id != other.machine_id:
            return False

        return not (
            self.end_time <= other.start_time or self.start_time >= other.end_time
        )


@dataclass
class Schedule:
    """
    Complete production schedule.

    Contains all assignments and provides analysis methods.
    """

    schedule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assignments: List[ScheduleAssignment] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    # Metadata
    objective_value: Optional[float] = None  # Optimization objective value
    solver_time_seconds: Optional[float] = None

    def add_assignment(self, assignment: ScheduleAssignment) -> None:
        """Add an assignment to the schedule."""
        self.assignments.append(assignment)

    def get_assignments_for_machine(self, machine_id: str) -> List[ScheduleAssignment]:
        """Get all assignments for a specific machine."""
        return [a for a in self.assignments if a.machine_id == machine_id]

    def get_assignments_for_job(self, job_id: str) -> List[ScheduleAssignment]:
        """Get all assignments for a specific job."""
        return [a for a in self.assignments if a.job_id == job_id]

    def get_makespan(self) -> Optional[int]:
        """
        Get schedule makespan (total time from start to finish).

        Returns:
            Makespan in minutes, or None if schedule is empty
        """
        if not self.assignments:
            return None

        earliest_start = min(a.start_time for a in self.assignments)
        latest_end = max(a.end_time for a in self.assignments)

        return int((latest_end - earliest_start).total_seconds() / 60)

    def get_machine_utilization(self, machine_id: str) -> float:
        """
        Calculate utilization for a machine.

        Args:
            machine_id: Machine ID

        Returns:
            Utilization ratio (0-1)
        """
        machine_assignments = self.get_assignments_for_machine(machine_id)

        if not machine_assignments:
            return 0.0

        # Total assigned time
        total_assigned = sum(a.get_duration_minutes() for a in machine_assignments)

        # Total available time (makespan)
        makespan = self.get_makespan()

        if makespan is None or makespan == 0:
            return 0.0

        return min(1.0, total_assigned / makespan)

    def get_average_utilization(self, machine_ids: List[str]) -> float:
        """
        Calculate average utilization across machines.

        Args:
            machine_ids: List of machine IDs

        Returns:
            Average utilization (0-1)
        """
        if not machine_ids:
            return 0.0

        utilizations = [self.get_machine_utilization(mid) for mid in machine_ids]
        return sum(utilizations) / len(utilizations)

    def has_conflicts(self) -> bool:
        """
        Check if schedule has any conflicts (overlapping assignments).

        Returns:
            True if conflicts exist
        """
        for i, assignment1 in enumerate(self.assignments):
            for assignment2 in self.assignments[i + 1 :]:
                if assignment1.overlaps_with(assignment2):
                    logger.warning(
                        f"Conflict detected: {assignment1.task_id} overlaps with "
                        f"{assignment2.task_id} on machine {assignment1.machine_id}"
                    )
                    return True

        return False

    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive schedule statistics."""
        machine_ids = list(set(a.machine_id for a in self.assignments))

        return {
            "schedule_id": self.schedule_id,
            "total_assignments": len(self.assignments),
            "makespan_minutes": self.get_makespan(),
            "num_machines": len(machine_ids),
            "average_utilization": self.get_average_utilization(machine_ids),
            "has_conflicts": self.has_conflicts(),
            "objective_value": self.objective_value,
            "solver_time_seconds": self.solver_time_seconds,
            "created_at": self.created_at.isoformat(),
        }


def create_sample_job(
    job_id: str, num_tasks: int = 3, task_duration: int = 30
) -> Job:
    """
    Create a sample job for testing.

    Args:
        job_id: Job identifier
        num_tasks: Number of tasks
        task_duration: Duration per task (minutes)

    Returns:
        Sample job
    """
    job = Job(job_id=job_id, product_type="Sample_Product", priority=1)

    for i in range(num_tasks):
        task = Task(
            task_id=f"{job_id}_T{i+1}",
            task_type="processing",
            duration=task_duration,
            predecessor_task_ids=[f"{job_id}_T{i}"] if i > 0 else [],
        )
        job.add_task(task)

    return job


def create_sample_machines(num_machines: int = 3) -> List[Machine]:
    """
    Create sample machines for testing.

    Args:
        num_machines: Number of machines

    Returns:
        List of machines
    """
    machines = []

    for i in range(num_machines):
        machine = Machine(
            machine_id=f"M{i+1}",
            machine_type="general",
            capabilities=["processing"],
            speed_factor=1.0 + (i * 0.1),  # Slightly different speeds
        )
        machines.append(machine)

    return machines
