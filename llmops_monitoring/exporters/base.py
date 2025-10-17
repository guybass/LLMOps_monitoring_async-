"""
Base interface for metrics exporters.

Extension Point: Implement MetricsExporter to add new export targets
(Prometheus, Datadog, New Relic, CloudWatch, etc.)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from llmops_monitoring.schema.events import MetricEvent


class MetricsExporter(ABC):
    """
    Abstract base class for metrics exporters.

    Exporters receive MetricEvents and expose them to external monitoring systems.

    Extension Point: Implement this interface to add new exporter types.

    Example:
        class DatadogExporter(MetricsExporter):
            async def initialize(self):
                # Set up Datadog client
                pass

            def record_event(self, event: MetricEvent):
                # Send to Datadog
                pass

            async def shutdown(self):
                # Close connections
                pass
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the exporter.

        Called once during MonitoringWriter startup.
        Use this to:
        - Set up HTTP servers
        - Initialize clients
        - Validate configuration
        - Start background threads

        Raises:
            Exception: If initialization fails
        """
        pass

    @abstractmethod
    def record_event(self, event: MetricEvent) -> None:
        """
        Record a metric event.

        Called for each event after it's successfully written to storage.
        Should be fast and non-blocking (use async for I/O).

        Args:
            event: The metric event to record

        Note:
            This method is called synchronously from the writer's flush loop.
            Keep it fast! For I/O-bound operations, queue the event and
            process it in a background task.
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Gracefully shut down the exporter.

        Called during MonitoringWriter.stop().
        Use this to:
        - Stop HTTP servers
        - Flush pending metrics
        - Close connections
        - Clean up resources
        """
        pass

    def health_check(self) -> Dict[str, Any]:
        """
        Check health of the exporter.

        Returns:
            Dictionary with health status. Should include:
            - "healthy": bool
            - Additional status info

        Example:
            {
                "healthy": True,
                "server_running": True,
                "metrics_exported": 12345
            }
        """
        return {"healthy": True}

    @property
    @abstractmethod
    def exporter_type(self) -> str:
        """
        Return the exporter type identifier.

        Examples: "prometheus", "datadog", "cloudwatch", "newrelic"
        """
        pass


class ExporterRegistry:
    """
    Registry for managing metric exporters.

    Extension Point: Register custom exporters here.
    """

    _exporters: Dict[str, type[MetricsExporter]] = {}

    @classmethod
    def register(cls, name: str, exporter_class: type[MetricsExporter]) -> None:
        """
        Register a new exporter type.

        Args:
            name: Unique identifier for the exporter (e.g., "prometheus")
            exporter_class: The exporter class (not instance)

        Example:
            ExporterRegistry.register("prometheus", PrometheusExporter)
        """
        cls._exporters[name] = exporter_class

    @classmethod
    def get(cls, name: str) -> Optional[type[MetricsExporter]]:
        """Get an exporter class by name."""
        return cls._exporters.get(name)

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered exporter names."""
        return list(cls._exporters.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered exporters (mainly for testing)."""
        cls._exporters.clear()
