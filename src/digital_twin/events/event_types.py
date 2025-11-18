"""Event types for digital twin."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class EventType(Enum):
    """Digital twin event types."""

    # Machine events
    MACHINE_STATUS_CHANGED = "machine_status_changed"
    MACHINE_CYCLE_COMPLETED = "machine_cycle_completed"
    MACHINE_STARTED = "machine_started"
    MACHINE_STOPPED = "machine_stopped"

    # Defect events
    DEFECT_DETECTED = "defect_detected"
    DEFECT_RESOLVED = "defect_resolved"

    # Maintenance events
    MAINTENANCE_SCHEDULED = "maintenance_scheduled"
    MAINTENANCE_STARTED = "maintenance_started"
    MAINTENANCE_COMPLETED = "maintenance_completed"

    # Job events
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"

    # Alert events
    ALERT_RAISED = "alert_raised"
    ALERT_CLEARED = "alert_cleared"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Event:
    """Base event class."""

    event_type: EventType
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DefectDetectedEvent(Event):
    """Event raised when defect is detected."""

    def __init__(
        self,
        machine_id: str,
        defect_type: str,
        confidence: float,
        image_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            event_type=EventType.DEFECT_DETECTED,
            source=f"machine:{machine_id}",
            data={
                "machine_id": machine_id,
                "defect_type": defect_type,
                "confidence": confidence,
                "image_path": image_path,
                **kwargs,
            },
        )


@dataclass
class MachineStatusChangedEvent(Event):
    """Event raised when machine status changes."""

    def __init__(
        self,
        machine_id: str,
        old_status: str,
        new_status: str,
        **kwargs,
    ):
        super().__init__(
            event_type=EventType.MACHINE_STATUS_CHANGED,
            source=f"machine:{machine_id}",
            data={
                "machine_id": machine_id,
                "old_status": old_status,
                "new_status": new_status,
                **kwargs,
            },
        )


@dataclass
class AlertRaisedEvent(Event):
    """Event raised when alert is triggered."""

    def __init__(
        self,
        machine_id: str,
        message: str,
        severity: AlertSeverity,
        **kwargs,
    ):
        super().__init__(
            event_type=EventType.ALERT_RAISED,
            source=f"machine:{machine_id}",
            data={
                "machine_id": machine_id,
                "message": message,
                "severity": severity.value,
                **kwargs,
            },
        )
