"""
Datadog metrics exporter for monitoring data.

Exports metrics to Datadog for monitoring and alerting.
"""

import time
from typing import Dict, Optional, List
from datetime import datetime

from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.exporters.base import MetricsExporter
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class DatadogExporter(MetricsExporter):
    """
    Datadog metrics exporter.

    Sends metrics to Datadog using the DogStatsD protocol or HTTP API.

    Features:
    - Operation counts with tags
    - Error tracking
    - Latency distribution
    - Character counts
    - Cost tracking
    - Custom tags support

    Example:
        ```python
        from llmops_monitoring.schema.config import DatadogConfig

        config = DatadogConfig(
            enabled=True,
            api_key="your-api-key",
            app_key="your-app-key",
            site="datadoghq.com",  # or datadoghq.eu
            use_statsd=False  # Use HTTP API
        )

        exporter = DatadogExporter(config)
        await exporter.initialize()
        ```
    """

    def __init__(self, config: Dict):
        """
        Initialize Datadog exporter.

        Args:
            config: Datadog configuration dictionary
        """
        self.config = config
        self.api_key = config.get("api_key", "")
        self.app_key = config.get("app_key", "")
        self.site = config.get("site", "datadoghq.com")
        self.use_statsd = config.get("use_statsd", False)
        self.statsd_host = config.get("statsd_host", "localhost")
        self.statsd_port = config.get("statsd_port", 8125)
        self.namespace = config.get("namespace", "llmops")

        self.datadog_client = None
        self.statsd_client = None
        self._datadog_available = self._check_datadog_available()

        # Metrics cache for aggregation
        self.operation_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.total_characters: Dict[str, int] = {}
        self.total_cost: Dict[str, float] = {}
        self.latencies: Dict[str, List[float]] = {}

    def _check_datadog_available(self) -> bool:
        """Check if Datadog client is available."""
        if self.use_statsd:
            try:
                from datadog import statsd
                return True
            except ImportError:
                return False
        else:
            try:
                from datadog_api_client import ApiClient, Configuration
                from datadog_api_client.v2.api.metrics_api import MetricsApi
                return True
            except ImportError:
                return False

    async def initialize(self) -> None:
        """Initialize Datadog client."""
        if not self._datadog_available:
            if self.use_statsd:
                raise RuntimeError(
                    "Datadog StatsD client not available. "
                    "Install with: pip install 'llamonitor-async[datadog]'"
                )
            else:
                raise RuntimeError(
                    "Datadog API client not available. "
                    "Install with: pip install 'llamonitor-async[datadog]'"
                )

        if self.use_statsd:
            # Initialize DogStatsD client
            from datadog import initialize, statsd

            options = {
                'statsd_host': self.statsd_host,
                'statsd_port': self.statsd_port,
            }
            initialize(**options)
            self.statsd_client = statsd

            logger.info(f"Datadog StatsD exporter initialized (host={self.statsd_host}:{self.statsd_port})")

        else:
            # Initialize HTTP API client
            from datadog_api_client import ApiClient, Configuration
            from datadog_api_client.v2.api.metrics_api import MetricsApi

            configuration = Configuration()
            configuration.api_key["apiKeyAuth"] = self.api_key
            configuration.api_key["appKeyAuth"] = self.app_key
            configuration.server_variables["site"] = self.site

            self.datadog_client = ApiClient(configuration)
            self.metrics_api = MetricsApi(self.datadog_client)

            logger.info(f"Datadog HTTP API exporter initialized (site={self.site})")

    def record_event(self, event: MetricEvent) -> None:
        """
        Record a metric event for later submission.

        Args:
            event: Metric event to record
        """
        # Create tag key
        tag_key = self._create_tag_key(event)

        # Update operation count
        if tag_key not in self.operation_counts:
            self.operation_counts[tag_key] = 0
        self.operation_counts[tag_key] += 1

        # Update error count
        if event.error:
            if tag_key not in self.error_counts:
                self.error_counts[tag_key] = 0
            self.error_counts[tag_key] += 1

        # Update character count
        if event.text_metrics and event.text_metrics.char_count:
            if tag_key not in self.total_characters:
                self.total_characters[tag_key] = 0
            self.total_characters[tag_key] += event.text_metrics.char_count

        # Update cost
        if event.custom_attributes:
            cost = event.custom_attributes.get("estimated_cost_usd", 0)
            if cost:
                if tag_key not in self.total_cost:
                    self.total_cost[tag_key] = 0.0
                self.total_cost[tag_key] += cost

        # Record latency
        if event.duration_ms is not None:
            if tag_key not in self.latencies:
                self.latencies[tag_key] = []
            self.latencies[tag_key].append(event.duration_ms)

    def _create_tag_key(self, event: MetricEvent) -> str:
        """Create a unique key from event tags."""
        model = event.custom_attributes.get("model", "unknown") if event.custom_attributes else "unknown"
        return f"{event.operation_name}:{model}:{event.operation_type}"

    def _parse_tags(self, tag_key: str) -> List[str]:
        """Parse tag key back into Datadog tags."""
        parts = tag_key.split(":")
        if len(parts) >= 3:
            operation_name = parts[0]
            model = parts[1]
            operation_type = parts[2]
            return [
                f"operation_name:{operation_name}",
                f"model:{model}",
                f"operation_type:{operation_type}"
            ]
        return []

    async def submit_metrics(self) -> None:
        """Submit accumulated metrics to Datadog."""
        if self.use_statsd:
            await self._submit_via_statsd()
        else:
            await self._submit_via_api()

        # Clear caches after submission
        self._clear_caches()

    async def _submit_via_statsd(self) -> None:
        """Submit metrics using DogStatsD."""
        # Operation counts
        for tag_key, count in self.operation_counts.items():
            tags = self._parse_tags(tag_key)
            self.statsd_client.increment(
                f"{self.namespace}.operations.count",
                value=count,
                tags=tags
            )

        # Error counts
        for tag_key, count in self.error_counts.items():
            tags = self._parse_tags(tag_key)
            self.statsd_client.increment(
                f"{self.namespace}.errors.count",
                value=count,
                tags=tags
            )

        # Character counts
        for tag_key, chars in self.total_characters.items():
            tags = self._parse_tags(tag_key)
            self.statsd_client.gauge(
                f"{self.namespace}.characters.total",
                value=chars,
                tags=tags
            )

        # Costs
        for tag_key, cost in self.total_cost.items():
            tags = self._parse_tags(tag_key)
            self.statsd_client.gauge(
                f"{self.namespace}.cost.usd",
                value=cost,
                tags=tags
            )

        # Latencies (distribution)
        for tag_key, latencies in self.latencies.items():
            tags = self._parse_tags(tag_key)
            for latency in latencies:
                self.statsd_client.histogram(
                    f"{self.namespace}.operation.duration.ms",
                    value=latency,
                    tags=tags
                )

        logger.debug("Submitted metrics to Datadog via StatsD")

    async def _submit_via_api(self) -> None:
        """Submit metrics using HTTP API."""
        from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
        from datadog_api_client.v2.model.metric_payload import MetricPayload
        from datadog_api_client.v2.model.metric_point import MetricPoint
        from datadog_api_client.v2.model.metric_resource import MetricResource
        from datadog_api_client.v2.model.metric_series import MetricSeries

        current_time = int(time.time())
        series = []

        # Operation counts
        for tag_key, count in self.operation_counts.items():
            tags = self._parse_tags(tag_key)
            series.append(MetricSeries(
                metric=f"{self.namespace}.operations.count",
                type=MetricIntakeType.COUNT,
                points=[MetricPoint(timestamp=current_time, value=float(count))],
                tags=tags
            ))

        # Error counts
        for tag_key, count in self.error_counts.items():
            tags = self._parse_tags(tag_key)
            series.append(MetricSeries(
                metric=f"{self.namespace}.errors.count",
                type=MetricIntakeType.COUNT,
                points=[MetricPoint(timestamp=current_time, value=float(count))],
                tags=tags
            ))

        # Character counts
        for tag_key, chars in self.total_characters.items():
            tags = self._parse_tags(tag_key)
            series.append(MetricSeries(
                metric=f"{self.namespace}.characters.total",
                type=MetricIntakeType.GAUGE,
                points=[MetricPoint(timestamp=current_time, value=float(chars))],
                tags=tags
            ))

        # Costs
        for tag_key, cost in self.total_cost.items():
            tags = self._parse_tags(tag_key)
            series.append(MetricSeries(
                metric=f"{self.namespace}.cost.usd",
                type=MetricIntakeType.GAUGE,
                points=[MetricPoint(timestamp=current_time, value=cost)],
                tags=tags
            ))

        # Latencies (avg per tag)
        for tag_key, latencies in self.latencies.items():
            if latencies:
                tags = self._parse_tags(tag_key)
                avg_latency = sum(latencies) / len(latencies)
                series.append(MetricSeries(
                    metric=f"{self.namespace}.operation.duration.ms",
                    type=MetricIntakeType.GAUGE,
                    points=[MetricPoint(timestamp=current_time, value=avg_latency)],
                    tags=tags
                ))

        if series:
            payload = MetricPayload(series=series)

            try:
                response = self.metrics_api.submit_metrics(body=payload)
                logger.debug(f"Submitted {len(series)} metric series to Datadog API")
            except Exception as e:
                logger.error(f"Error submitting metrics to Datadog: {e}")

    def _clear_caches(self) -> None:
        """Clear metric caches after submission."""
        self.operation_counts.clear()
        self.error_counts.clear()
        self.total_characters.clear()
        self.total_cost.clear()
        self.latencies.clear()

    async def shutdown(self) -> None:
        """Shutdown Datadog exporter."""
        # Submit any remaining metrics
        await self.submit_metrics()

        if self.datadog_client:
            self.datadog_client.close()

        logger.info("Datadog exporter shutdown complete")

    def health_check(self) -> Dict:
        """Check exporter health."""
        return {
            "healthy": self._datadog_available,
            "use_statsd": self.use_statsd,
            "namespace": self.namespace,
            "cached_metrics": {
                "operations": len(self.operation_counts),
                "errors": len(self.error_counts),
                "characters": len(self.total_characters),
                "costs": len(self.total_cost),
                "latencies": len(self.latencies)
            }
        }
