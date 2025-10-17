"""
Prometheus metrics exporter.

Exposes LLM monitoring metrics via HTTP endpoint for scraping by Prometheus.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

from llmops_monitoring.exporters.base import MetricsExporter, ExporterRegistry
from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class PrometheusExporter(MetricsExporter):
    """
    Prometheus exporter for LLM monitoring metrics.

    Exposes metrics via HTTP endpoint in Prometheus format.

    Metrics exposed:
    - llm_operations_total (Counter): Total operations by operation_name, model, type
    - llm_errors_total (Counter): Total errors by operation_name, error_type
    - llm_operation_duration_seconds (Histogram): Operation latency distribution
    - llm_text_characters_total (Counter): Total characters processed
    - llm_cost_usd (Histogram): Cost per operation distribution
    - llm_queue_size (Gauge): Current queue size
    - llm_buffer_size (Gauge): Current buffer size

    Example:
        config = MonitorConfig(
            extensions={
                "prometheus": {
                    "enabled": True,
                    "port": 8000,
                    "host": "0.0.0.0"
                }
            }
        )
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Prometheus exporter.

        Args:
            config: Configuration dict with keys:
                - port (int): HTTP server port (default: 8000)
                - host (str): HTTP server host (default: "0.0.0.0")
                - path (str): Metrics endpoint path (default: "/metrics")
                - include_labels (List[str]): Labels to include (default: all)
        """
        self.config = config
        self.port = config.get("port", 8000)
        self.host = config.get("host", "0.0.0.0")
        self.path = config.get("path", "/metrics")
        self.include_labels = config.get("include_labels", ["operation_name", "model", "operation_type"])

        # Check if prometheus_client is available
        self._prometheus_available = self._check_prometheus_client()
        if not self._prometheus_available:
            logger.warning(
                "prometheus_client not installed. "
                "Install with: pip install 'llamonitor-async[prometheus]'"
            )
            return

        # Import prometheus_client
        from prometheus_client import Counter, Histogram, Gauge, start_http_server

        self._start_http_server = start_http_server
        self._http_server = None
        self._server_thread = None

        # Initialize metrics
        self._operations_counter = Counter(
            "llm_operations_total",
            "Total number of LLM operations",
            labelnames=["operation_name", "model", "operation_type"]
        )

        self._errors_counter = Counter(
            "llm_errors_total",
            "Total number of errors in LLM operations",
            labelnames=["operation_name", "error_type"]
        )

        self._duration_histogram = Histogram(
            "llm_operation_duration_seconds",
            "Duration of LLM operations in seconds",
            labelnames=["operation_name", "model"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )

        self._characters_counter = Counter(
            "llm_text_characters_total",
            "Total number of text characters processed",
            labelnames=["operation_name"]
        )

        self._cost_histogram = Histogram(
            "llm_cost_usd",
            "Cost of LLM operations in USD",
            labelnames=["operation_name", "model"],
            buckets=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        )

        self._queue_size_gauge = Gauge(
            "llm_queue_size",
            "Current size of the event queue"
        )

        self._buffer_size_gauge = Gauge(
            "llm_buffer_size",
            "Current size of the event buffer"
        )

        self._metrics_exported = 0
        logger.info(f"PrometheusExporter initialized (will serve on {self.host}:{self.port})")

    def _check_prometheus_client(self) -> bool:
        """Check if prometheus_client is installed."""
        try:
            import prometheus_client
            return True
        except ImportError:
            return False

    async def initialize(self) -> None:
        """Start the Prometheus HTTP server."""
        if not self._prometheus_available:
            raise RuntimeError(
                "PrometheusExporter requires prometheus_client. "
                "Install with: pip install 'llamonitor-async[prometheus]'"
            )

        try:
            # Start HTTP server in a background thread
            # prometheus_client's start_http_server is blocking, so we run it in a thread
            logger.info(f"Starting Prometheus HTTP server on {self.host}:{self.port}")
            self._start_http_server(self.port, addr=self.host)
            logger.info(f"✓ Prometheus metrics available at http://{self.host}:{self.port}/metrics")

        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning(
                    f"Port {self.port} already in use. "
                    f"Metrics may already be exposed or choose a different port."
                )
            else:
                raise

    def record_event(self, event: MetricEvent) -> None:
        """
        Record a metric event to Prometheus.

        Updates all relevant metrics based on the event data.

        Args:
            event: The metric event to record
        """
        if not self._prometheus_available:
            return

        try:
            # Extract labels
            operation_name = event.operation_name
            model = self._extract_model(event)
            operation_type = event.operation_type

            # Update operations counter
            self._operations_counter.labels(
                operation_name=operation_name,
                model=model,
                operation_type=operation_type
            ).inc()

            # Update duration histogram
            if event.duration_ms is not None:
                duration_seconds = event.duration_ms / 1000.0
                self._duration_histogram.labels(
                    operation_name=operation_name,
                    model=model
                ).observe(duration_seconds)

            # Update character counter
            if event.text_metrics and event.text_metrics.char_count:
                self._characters_counter.labels(
                    operation_name=operation_name
                ).inc(event.text_metrics.char_count)

            # Update cost histogram
            cost = self._extract_cost(event)
            if cost is not None and cost > 0:
                self._cost_histogram.labels(
                    operation_name=operation_name,
                    model=model
                ).observe(cost)

            # Update error counter
            if event.error:
                error_type = event.error_type or "unknown"
                self._errors_counter.labels(
                    operation_name=operation_name,
                    error_type=error_type
                ).inc()

            self._metrics_exported += 1

        except Exception as e:
            logger.debug(f"Error recording Prometheus metric: {e}")

    def update_queue_metrics(self, queue_size: int, buffer_size: int) -> None:
        """
        Update queue and buffer size gauges.

        Args:
            queue_size: Current queue size
            buffer_size: Current buffer size
        """
        if not self._prometheus_available:
            return

        try:
            self._queue_size_gauge.set(queue_size)
            self._buffer_size_gauge.set(buffer_size)
        except Exception as e:
            logger.debug(f"Error updating queue metrics: {e}")

    async def shutdown(self) -> None:
        """Shut down the Prometheus exporter."""
        logger.info("Shutting down Prometheus exporter")
        # prometheus_client HTTP server doesn't have a clean shutdown method
        # The metrics will remain in memory but the server will stop when process ends
        logger.info(f"✓ Prometheus exporter shut down ({self._metrics_exported} metrics exported)")

    def health_check(self) -> Dict[str, Any]:
        """Check health of the Prometheus exporter."""
        return {
            "healthy": self._prometheus_available,
            "server_running": self._prometheus_available,
            "metrics_exported": self._metrics_exported,
            "endpoint": f"http://{self.host}:{self.port}/metrics"
        }

    @property
    def exporter_type(self) -> str:
        return "prometheus"

    def _extract_model(self, event: MetricEvent) -> str:
        """Extract model name from event custom attributes."""
        if event.custom_attributes and "model" in event.custom_attributes:
            return event.custom_attributes["model"]
        return "unknown"

    def _extract_cost(self, event: MetricEvent) -> Optional[float]:
        """Extract cost from event custom attributes."""
        if not event.custom_attributes:
            return None

        # Check for estimated_cost_usd (from cost collector)
        if "estimated_cost_usd" in event.custom_attributes:
            return event.custom_attributes["estimated_cost_usd"]

        # Check for cost_usd (custom)
        if "cost_usd" in event.custom_attributes:
            return event.custom_attributes["cost_usd"]

        return None


# Register the exporter
ExporterRegistry.register("prometheus", PrometheusExporter)
