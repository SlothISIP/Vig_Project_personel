"""
Production Scheduling Environment State Representation
Defines how the factory state is represented for RL agents
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime


@dataclass
class MachineStateRL:
    """Machine state for RL environment"""

    machine_id: str
    is_available: bool
    current_utilization: float  # 0-1
    queue_length: int
    processing_job_id: Optional[str]
    estimated_completion_time: Optional[float]  # minutes from now
    failure_probability: float  # 0-1 from predictive maintenance
    health_score: float  # 0-1


@dataclass
class JobStateRL:
    """Job state for RL environment"""

    job_id: str
    priority: int  # 1-10
    arrival_time: float  # timestamp
    deadline: float  # timestamp
    remaining_tasks: int
    current_task_idx: int
    total_processing_time: float  # minutes
    waiting_time: float  # minutes
    is_overdue: bool


@dataclass
class SchedulingStateRL:
    """Complete state representation for RL scheduling environment"""

    machines: List[MachineStateRL]
    pending_jobs: List[JobStateRL]
    active_jobs: List[JobStateRL]
    completed_jobs: List[JobStateRL]
    current_time: float  # minutes from episode start
    time_step: int

    # Aggregate metrics
    total_utilization: float
    average_queue_length: float
    num_overdue_jobs: int
    total_waiting_time: float

    def to_observation(self) -> np.ndarray:
        """
        Convert state to numerical observation vector for RL agent

        Observation space:
        - Machine features: [is_available, utilization, queue_length, failure_prob, health] × num_machines
        - Job features (top N): [priority_norm, time_to_deadline, completion_ratio, waiting_time_norm] × N
        - Global features: [total_util, avg_queue, num_overdue, time_ratio]
        """
        obs_components = []

        # Machine features (5 features × num_machines)
        for machine in self.machines:
            machine_features = [
                1.0 if machine.is_available else 0.0,
                machine.current_utilization,
                min(machine.queue_length / 10.0, 1.0),  # normalize queue length
                machine.failure_probability,
                machine.health_score,
            ]
            obs_components.extend(machine_features)

        # Top N pending jobs features (4 features × N jobs)
        N = 10  # Consider top 10 pending jobs
        sorted_jobs = sorted(
            self.pending_jobs, key=lambda j: (j.is_overdue, -j.priority, j.deadline)
        )[:N]

        for i in range(N):
            if i < len(sorted_jobs):
                job = sorted_jobs[i]
                time_to_deadline = max(job.deadline - self.current_time, 0)
                job_features = [
                    job.priority / 10.0,  # normalize priority
                    min(time_to_deadline / 1000.0, 1.0),  # normalize deadline
                    (
                        job.current_task_idx / job.remaining_tasks
                        if job.remaining_tasks > 0
                        else 1.0
                    ),
                    min(job.waiting_time / 500.0, 1.0),  # normalize waiting time
                ]
            else:
                # Padding for missing jobs
                job_features = [0.0, 0.0, 0.0, 0.0]

            obs_components.extend(job_features)

        # Global features
        global_features = [
            self.total_utilization,
            min(self.average_queue_length / 10.0, 1.0),
            min(self.num_overdue_jobs / 20.0, 1.0),
            min(self.current_time / 10000.0, 1.0),  # normalize time
        ]
        obs_components.extend(global_features)

        return np.array(obs_components, dtype=np.float32)

    @staticmethod
    def get_observation_space_size(num_machines: int, max_pending_jobs: int = 10) -> int:
        """Calculate observation space size"""
        machine_features = num_machines * 5
        job_features = max_pending_jobs * 4
        global_features = 4
        return machine_features + job_features + global_features


@dataclass
class SchedulingAction:
    """Action representation: Assign job to machine"""

    job_id: str
    machine_id: str
    priority_boost: float = 0.0  # -1.0 to 1.0, adjust job priority


def state_from_dict(state_dict: Dict) -> SchedulingStateRL:
    """Convert dictionary representation to SchedulingStateRL"""
    machines = [
        MachineStateRL(
            machine_id=m["machine_id"],
            is_available=m["is_available"],
            current_utilization=m["utilization"],
            queue_length=m["queue_length"],
            processing_job_id=m.get("processing_job_id"),
            estimated_completion_time=m.get("estimated_completion_time"),
            failure_probability=m.get("failure_probability", 0.0),
            health_score=m.get("health_score", 1.0),
        )
        for m in state_dict["machines"]
    ]

    def job_from_dict(j: Dict) -> JobStateRL:
        return JobStateRL(
            job_id=j["job_id"],
            priority=j["priority"],
            arrival_time=j["arrival_time"],
            deadline=j["deadline"],
            remaining_tasks=j["remaining_tasks"],
            current_task_idx=j["current_task_idx"],
            total_processing_time=j["total_processing_time"],
            waiting_time=j["waiting_time"],
            is_overdue=j.get("is_overdue", False),
        )

    return SchedulingStateRL(
        machines=machines,
        pending_jobs=[job_from_dict(j) for j in state_dict.get("pending_jobs", [])],
        active_jobs=[job_from_dict(j) for j in state_dict.get("active_jobs", [])],
        completed_jobs=[job_from_dict(j) for j in state_dict.get("completed_jobs", [])],
        current_time=state_dict["current_time"],
        time_step=state_dict["time_step"],
        total_utilization=state_dict.get("total_utilization", 0.0),
        average_queue_length=state_dict.get("average_queue_length", 0.0),
        num_overdue_jobs=state_dict.get("num_overdue_jobs", 0),
        total_waiting_time=state_dict.get("total_waiting_time", 0.0),
    )
