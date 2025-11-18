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


class MachineStateManager:
    """
    Machine State Manager for API integration.

    Provides a simplified interface for managing machine states
    with support for dynamic properties (temperature, vibration, etc.)
    """

    def __init__(self, factory_id: str = "Factory_01"):
        """Initialize machine state manager."""
        self.factory_state = FactoryState(factory_id=factory_id)
        self._machine_properties: Dict[str, Dict[str, Any]] = {}

    def add_machine(
        self,
        machine_id: str,
        machine_type: str,
        initial_state: str = "idle",
    ) -> None:
        """
        Add a new machine to the factory.

        Args:
            machine_id: Unique machine identifier
            machine_type: Type/model of machine
            initial_state: Initial operating state
        """
        # Map string state to MachineStatus enum
        status_map = {
            "idle": MachineStatus.IDLE,
            "running": MachineStatus.RUNNING,
            "warning": MachineStatus.WARNING,
            "error": MachineStatus.ERROR,
            "maintenance": MachineStatus.MAINTENANCE,
            "offline": MachineStatus.OFFLINE,
        }

        status = status_map.get(initial_state.lower(), MachineStatus.IDLE)

        machine = MachineState(
            machine_id=machine_id,
            machine_type=machine_type,
            status=status,
        )

        self.factory_state.add_machine(machine)

        # Initialize dynamic properties
        self._machine_properties[machine_id] = {
            "temperature": 70.0,  # Default temperature
            "vibration": 2.0,     # Default vibration
            "pressure": 90.0,     # Default pressure
            "speed": 1000.0,      # Default speed
            "defect_rate": 0.0,
            "state": initial_state,
        }

    def get_machine_state(self, machine_id: str):
        """
        Get machine state with dynamic properties.

        Returns a state object with both MachineState attributes
        and dynamic properties (temperature, vibration, etc.)
        """
        machine = self.factory_state.get_machine(machine_id)
        if not machine:
            return None

        # Create a wrapper object with all properties
        class MachineStateWrapper:
            def __init__(self, machine_state, properties):
                self._machine_state = machine_state
                self._properties = properties

            def __getattr__(self, name):
                # First try to get from machine_state
                if hasattr(self._machine_state, name):
                    return getattr(self._machine_state, name)
                # Then try properties
                if name in self._properties:
                    return self._properties[name]
                # Special mapping for 'state' attribute
                if name == 'state':
                    return self._properties.get('state', self._machine_state.status.value)
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

            def __setattr__(self, name, value):
                if name.startswith('_'):
                    super().__setattr__(name, value)
                elif hasattr(self._machine_state, name):
                    setattr(self._machine_state, name, value)
                else:
                    self._properties[name] = value

        props = self._machine_properties.get(machine_id, {})
        return MachineStateWrapper(machine, props)

    def get_all_machines(self) -> Dict[str, Any]:
        """Get all machines with their states."""
        result = {}
        for machine_id in self.factory_state.machines.keys():
            result[machine_id] = self.get_machine_state(machine_id)
        return result

    def update_machine_state(
        self,
        machine_id: str,
        **kwargs
    ) -> None:
        """
        Update machine state properties.

        Args:
            machine_id: Machine to update
            **kwargs: Properties to update (temperature, vibration, state, etc.)
        """
        machine = self.factory_state.get_machine(machine_id)
        if not machine:
            return

        # Update MachineState properties
        if 'state' in kwargs or 'status' in kwargs:
            status_str = kwargs.get('state', kwargs.get('status', 'idle'))
            status_map = {
                "idle": MachineStatus.IDLE,
                "running": MachineStatus.RUNNING,
                "warning": MachineStatus.WARNING,
                "error": MachineStatus.ERROR,
                "maintenance": MachineStatus.MAINTENANCE,
                "offline": MachineStatus.OFFLINE,
            }
            if isinstance(status_str, str):
                machine.update_status(status_map.get(status_str.lower(), MachineStatus.IDLE))

        # Update dynamic properties
        if machine_id not in self._machine_properties:
            self._machine_properties[machine_id] = {}

        for key, value in kwargs.items():
            if key not in ['state', 'status']:
                self._machine_properties[machine_id][key] = value
