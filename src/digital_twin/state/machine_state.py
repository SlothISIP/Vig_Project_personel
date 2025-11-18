"""Machine state management for digital twin."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


class MachineStatus(Enum):
    """Machine status enumeration."""

    IDLE = "idle"
    RUNNING = "running"
    WARNING = "warning"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class MachineState:
    """
    State of a single machine in the factory.

    Represents the current operational state of a manufacturing machine.
    """

    machine_id: str
    machine_type: str
    status: MachineStatus = MachineStatus.IDLE
    current_job_id: Optional[str] = None
    health_score: float = 1.0  # 0.0 to 1.0
    cycle_count: int = 0
    defect_count: int = 0
    last_defect_time: Optional[datetime] = None
    last_maintenance: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_status(self, new_status: MachineStatus) -> None:
        """Update machine status."""
        self.status = new_status
        self.updated_at = datetime.now()

    def increment_cycle(self) -> None:
        """Increment cycle count."""
        self.cycle_count += 1
        self.updated_at = datetime.now()

    def report_defect(self) -> None:
        """Report a defect detected."""
        self.defect_count += 1
        self.last_defect_time = datetime.now()
        self.updated_at = datetime.now()

        # Update health score based on defect rate
        if self.cycle_count > 0:
            defect_rate = self.defect_count / self.cycle_count
            self.health_score = max(0.0, 1.0 - defect_rate)

            # Trigger warning if health score drops
            if self.health_score < 0.7:
                self.status = MachineStatus.WARNING

    def perform_maintenance(self) -> None:
        """Perform maintenance on machine."""
        self.last_maintenance = datetime.now()
        self.health_score = 1.0
        self.defect_count = 0
        self.status = MachineStatus.IDLE
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "machine_id": self.machine_id,
            "machine_type": self.machine_type,
            "status": self.status.value,
            "current_job_id": self.current_job_id,
            "health_score": self.health_score,
            "cycle_count": self.cycle_count,
            "defect_count": self.defect_count,
            "defect_rate": (
                self.defect_count / self.cycle_count if self.cycle_count > 0 else 0.0
            ),
            "last_defect_time": (
                self.last_defect_time.isoformat() if self.last_defect_time else None
            ),
            "last_maintenance": (
                self.last_maintenance.isoformat() if self.last_maintenance else None
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class FactoryState:
    """
    Overall state of the factory.

    Maintains state of all machines and production line.
    """

    factory_id: str
    machines: Dict[str, MachineState] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_machine(self, machine: MachineState) -> None:
        """Add a machine to the factory."""
        self.machines[machine.machine_id] = machine
        self.updated_at = datetime.now()

    def get_machine(self, machine_id: str) -> Optional[MachineState]:
        """Get machine by ID."""
        return self.machines.get(machine_id)

    def update_machine_status(self, machine_id: str, status: MachineStatus) -> None:
        """Update machine status."""
        if machine_id in self.machines:
            self.machines[machine_id].update_status(status)
            self.updated_at = datetime.now()

    def get_overall_health(self) -> float:
        """Calculate overall factory health score."""
        if not self.machines:
            return 1.0

        total_health = sum(m.health_score for m in self.machines.values())
        return total_health / len(self.machines)

    def get_active_machines(self) -> list[MachineState]:
        """Get all active (running) machines."""
        return [
            m for m in self.machines.values() if m.status == MachineStatus.RUNNING
        ]

    def get_machines_by_status(self, status: MachineStatus) -> list[MachineState]:
        """Get machines by status."""
        return [m for m in self.machines.values() if m.status == status]

    def get_statistics(self) -> Dict[str, Any]:
        """Get factory statistics."""
        total_machines = len(self.machines)
        if total_machines == 0:
            return {
                "total_machines": 0,
                "overall_health": 1.0,
                "status_breakdown": {},
            }

        # Status breakdown
        status_counts = {}
        for status in MachineStatus:
            count = len(self.get_machines_by_status(status))
            if count > 0:
                status_counts[status.value] = count

        # Total metrics
        total_cycles = sum(m.cycle_count for m in self.machines.values())
        total_defects = sum(m.defect_count for m in self.machines.values())

        return {
            "total_machines": total_machines,
            "overall_health": self.get_overall_health(),
            "status_breakdown": status_counts,
            "total_cycles": total_cycles,
            "total_defects": total_defects,
            "overall_defect_rate": (
                total_defects / total_cycles if total_cycles > 0 else 0.0
            ),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "factory_id": self.factory_id,
            "machines": {
                machine_id: machine.to_dict()
                for machine_id, machine in self.machines.items()
            },
            "statistics": self.get_statistics(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
