"""Event bus for digital twin events."""

from typing import Callable, Dict, List
from collections import defaultdict
import uuid

from src.digital_twin.events.event_types import Event, EventType
from src.core.logging import get_logger

logger = get_logger(__name__)


class EventBus:
    """
    Simple event bus for digital twin.

    Allows subscribing to events and publishing events to subscribers.
    """

    def __init__(self):
        """Initialize event bus."""
        self._handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_history: List[Event] = []
        self._max_history_size: int = 1000

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Callback function to handle the event
        """
        self._handlers[event_type].append(handler)
        logger.info(f"Subscribed handler to {event_type.value}")

    def unsubscribe(
        self, event_type: EventType, handler: Callable[[Event], None]
    ) -> None:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Type of event
            handler: Handler to remove
        """
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            logger.info(f"Unsubscribed handler from {event_type.value}")

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        # Generate event ID if not present
        if event.event_id is None:
            event.event_id = str(uuid.uuid4())

        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)

        # Log event
        logger.info(
            f"Event published: {event.event_type.value} from {event.source}"
        )

        # Call all handlers for this event type
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Error in event handler for {event.event_type.value}: {e}"
                )

    def get_event_history(
        self, event_type: Optional[EventType] = None, limit: int = 100
    ) -> List[Event]:
        """
        Get event history.

        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return

        Returns:
            List of events
        """
        events = self._event_history

        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
        logger.info("Event history cleared")


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get global event bus instance (singleton)."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
