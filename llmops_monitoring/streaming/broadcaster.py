"""
Event broadcaster for real-time streaming.

Broadcasts events to WebSocket connections as they're written.
"""

from typing import Optional

from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.streaming.connection_manager import ConnectionManager
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class EventBroadcaster:
    """
    Broadcasts events to WebSocket connections.

    Integrates with MonitoringWriter to stream events in real-time
    as they're written to storage.

    Example:
        ```python
        from llmops_monitoring.streaming import EventBroadcaster

        broadcaster = EventBroadcaster()
        await broadcaster.broadcast_event(event)
        ```
    """

    _instance: Optional["EventBroadcaster"] = None

    def __init__(self):
        """Initialize event broadcaster."""
        if EventBroadcaster._instance is not None:
            raise RuntimeError("EventBroadcaster is a singleton. Use get_instance()")

        self.connection_manager = ConnectionManager()
        self.enabled = False

        EventBroadcaster._instance = self

    @classmethod
    def get_instance(cls) -> "EventBroadcaster":
        """
        Get or create the singleton instance.

        Returns:
            EventBroadcaster instance
        """
        if cls._instance is None:
            cls._instance = cls()

        return cls._instance

    def enable(self):
        """Enable event broadcasting."""
        self.enabled = True
        logger.info("Event broadcasting enabled")

    def disable(self):
        """Disable event broadcasting."""
        self.enabled = False
        logger.info("Event broadcasting disabled")

    async def broadcast_event(self, event: MetricEvent):
        """
        Broadcast an event to all connected WebSocket clients.

        Args:
            event: MetricEvent to broadcast
        """
        if not self.enabled:
            return

        try:
            await self.connection_manager.broadcast_event(event)
        except Exception as e:
            logger.error(f"Error broadcasting event: {e}")

    async def broadcast_events(self, events: list[MetricEvent]):
        """
        Broadcast multiple events.

        Args:
            events: List of MetricEvents to broadcast
        """
        if not self.enabled:
            return

        for event in events:
            await self.broadcast_event(event)

    def get_connection_manager(self) -> ConnectionManager:
        """
        Get the connection manager.

        Returns:
            ConnectionManager instance
        """
        return self.connection_manager

    def get_stats(self):
        """
        Get broadcaster statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "enabled": self.enabled,
            **self.connection_manager.get_stats()
        }
