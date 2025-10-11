"""
Base abstraction for storage backends.

Extension Point: Implement StorageBackend to add new storage systems.
"""

from abc import ABC, abstractmethod
from typing import List

from llmops_monitoring.schema.events import MetricEvent


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    Backends handle persistence of metric events. Users can implement
    custom backends for different storage systems.

    Example:
        class RedisBackend(StorageBackend):
            async def write_event(self, event):
                await redis.rpush("events", event.json())

            async def write_batch(self, events):
                await redis.rpush("events", *[e.json() for e in events])
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the backend (create tables, connections, etc.).

        Called once when the monitoring system starts.
        """
        pass

    @abstractmethod
    async def write_event(self, event: MetricEvent) -> None:
        """
        Write a single event to storage.

        Args:
            event: The metric event to store

        Raises:
            Exception: If write fails (caller handles retries)
        """
        pass

    @abstractmethod
    async def write_batch(self, events: List[MetricEvent]) -> None:
        """
        Write multiple events in a batch (more efficient).

        Args:
            events: List of metric events to store

        Raises:
            Exception: If write fails (caller handles retries)
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Clean up resources (close connections, flush buffers, etc.).

        Called when the monitoring system shuts down.
        """
        pass

    async def health_check(self) -> bool:
        """
        Check if the backend is healthy and can accept writes.

        Override for custom health check logic.

        Returns:
            True if backend is healthy, False otherwise
        """
        return True

    def supports_batch_writes(self) -> bool:
        """
        Indicate if this backend benefits from batch writes.

        Returns:
            True if batch writes are more efficient than individual writes
        """
        return True
