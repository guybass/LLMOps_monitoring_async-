"""
Tests for Prometheus exporter

These tests verify Prometheus exporter functionality without requiring
the prometheus_client library (uses mocking).
"""

import asyncio
import logging
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch, MagicMock, call
from uuid import uuid4

import pytest

from llmops_monitoring.schema.events import MetricEvent, TextMetrics
from llmops_monitoring.exporters.prometheus import PrometheusExporter


# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@pytest.fixture
def prometheus_config():
    """Create Prometheus configuration for testing."""
    return {
        "enabled": True,
        "port": 8000,
        "host": "0.0.0.0",
        "path": "/metrics",
        "include_labels": ["operation_name", "model", "operation_type"]
    }


@pytest.fixture
def sample_event():
    """Create a sample metric event for testing."""
    return MetricEvent(
        event_id=uuid4(),
        schema_version="1.0.0",
        session_id="test-session",
        trace_id="test-trace",
        span_id="test-span",
        parent_span_id=None,
        operation_name="test_operation",
        operation_type="completion",
        timestamp=datetime.utcnow(),
        duration_ms=250.5,
        text_metrics=TextMetrics(
            char_count=1000,
            word_count=200,
            byte_size=1500,
            line_count=50
        ),
        custom_attributes={
            "model": "gpt-4",
            "estimated_cost_usd": 0.015
        }
    )


@pytest.fixture
def error_event():
    """Create a sample error event for testing."""
    return MetricEvent(
        event_id=uuid4(),
        schema_version="1.0.0",
        session_id="test-session",
        trace_id="test-trace",
        span_id="test-span-error",
        parent_span_id=None,
        operation_name="failing_operation",
        operation_type="completion",
        timestamp=datetime.utcnow(),
        duration_ms=100.0,
        error=True,
        error_type="ValueError",
        error_message="Simulated error",
        custom_attributes={
            "model": "claude-3-opus"
        }
    )


class TestPrometheusExporter:
    """Test suite for Prometheus exporter."""

    def test_initialization_without_prometheus_client(self, prometheus_config):
        """Test initialization when prometheus_client is not installed."""
        logger.info("Testing initialization without prometheus_client...")

        with patch.object(PrometheusExporter, '_check_prometheus_client', return_value=False):
            exporter = PrometheusExporter(prometheus_config)

            assert exporter.port == 8000
            assert exporter.host == "0.0.0.0"
            assert exporter._prometheus_available is False

        logger.info("✓ Initialization without prometheus_client test passed")

    def test_initialization_with_prometheus_client(self, prometheus_config):
        """Test initialization with prometheus_client installed."""
        logger.info("Testing initialization with prometheus_client...")

        mock_counter = Mock()
        mock_histogram = Mock()
        mock_gauge = Mock()
        mock_start_http_server = Mock()

        with patch('llmops_monitoring.exporters.prometheus.Counter', mock_counter), \
             patch('llmops_monitoring.exporters.prometheus.Histogram', mock_histogram), \
             patch('llmops_monitoring.exporters.prometheus.Gauge', mock_gauge), \
             patch('llmops_monitoring.exporters.prometheus.start_http_server', mock_start_http_server):

            exporter = PrometheusExporter(prometheus_config)

            assert exporter.port == 8000
            assert exporter.host == "0.0.0.0"
            assert exporter.path == "/metrics"
            assert exporter._prometheus_available is True

            # Verify metrics were created
            assert mock_counter.call_count == 3  # operations, errors, characters
            assert mock_histogram.call_count == 2  # duration, cost
            assert mock_gauge.call_count == 2  # queue_size, buffer_size

        logger.info("✓ Initialization with prometheus_client test passed")

    @pytest.mark.asyncio
    async def test_initialize_raises_without_prometheus_client(self, prometheus_config):
        """Test that initialize() raises error when prometheus_client not installed."""
        logger.info("Testing initialize() without prometheus_client...")

        with patch.object(PrometheusExporter, '_check_prometheus_client', return_value=False):
            exporter = PrometheusExporter(prometheus_config)

            with pytest.raises(RuntimeError, match="requires prometheus_client"):
                await exporter.initialize()

        logger.info("✓ Initialize requirement test passed")

    @pytest.mark.asyncio
    async def test_initialize_starts_http_server(self, prometheus_config):
        """Test that initialize() starts the HTTP server."""
        logger.info("Testing HTTP server startup...")

        mock_start_http_server = Mock()
        mock_counter = Mock()
        mock_histogram = Mock()
        mock_gauge = Mock()

        with patch('llmops_monitoring.exporters.prometheus.Counter', mock_counter), \
             patch('llmops_monitoring.exporters.prometheus.Histogram', mock_histogram), \
             patch('llmops_monitoring.exporters.prometheus.Gauge', mock_gauge), \
             patch('llmops_monitoring.exporters.prometheus.start_http_server', mock_start_http_server):

            exporter = PrometheusExporter(prometheus_config)
            await exporter.initialize()

            # Verify HTTP server was started
            mock_start_http_server.assert_called_once_with(8000, addr="0.0.0.0")

        logger.info("✓ HTTP server startup test passed")

    @pytest.mark.asyncio
    async def test_initialize_handles_port_in_use(self, prometheus_config):
        """Test that initialize() handles port already in use."""
        logger.info("Testing port already in use handling...")

        mock_start_http_server = Mock(side_effect=OSError("Address already in use"))
        mock_counter = Mock()
        mock_histogram = Mock()
        mock_gauge = Mock()

        with patch('llmops_monitoring.exporters.prometheus.Counter', mock_counter), \
             patch('llmops_monitoring.exporters.prometheus.Histogram', mock_histogram), \
             patch('llmops_monitoring.exporters.prometheus.Gauge', mock_gauge), \
             patch('llmops_monitoring.exporters.prometheus.start_http_server', mock_start_http_server):

            exporter = PrometheusExporter(prometheus_config)

            # Should not raise, just log warning
            await exporter.initialize()

        logger.info("✓ Port in use handling test passed")

    def test_record_event_without_prometheus_client(self, prometheus_config, sample_event):
        """Test record_event when prometheus_client not installed."""
        logger.info("Testing record_event without prometheus_client...")

        with patch.object(PrometheusExporter, '_check_prometheus_client', return_value=False):
            exporter = PrometheusExporter(prometheus_config)

            # Should not raise error
            exporter.record_event(sample_event)

        logger.info("✓ Record event without prometheus_client test passed")

    def test_record_event_updates_metrics(self, prometheus_config, sample_event):
        """Test that record_event updates all relevant metrics."""
        logger.info("Testing metric updates on record_event...")

        # Create mock metrics
        mock_operations_counter = Mock()
        mock_operations_counter.labels = Mock(return_value=Mock(inc=Mock()))

        mock_duration_histogram = Mock()
        mock_duration_histogram.labels = Mock(return_value=Mock(observe=Mock()))

        mock_characters_counter = Mock()
        mock_characters_counter.labels = Mock(return_value=Mock(inc=Mock()))

        mock_cost_histogram = Mock()
        mock_cost_histogram.labels = Mock(return_value=Mock(observe=Mock()))

        mock_errors_counter = Mock()
        mock_queue_gauge = Mock()
        mock_buffer_gauge = Mock()

        with patch('llmops_monitoring.exporters.prometheus.Counter') as mock_counter_class, \
             patch('llmops_monitoring.exporters.prometheus.Histogram') as mock_histogram_class, \
             patch('llmops_monitoring.exporters.prometheus.Gauge') as mock_gauge_class, \
             patch('llmops_monitoring.exporters.prometheus.start_http_server'):

            # Set up Counter mock to return different counters
            mock_counter_class.side_effect = [
                mock_operations_counter,
                mock_errors_counter,
                mock_characters_counter
            ]

            # Set up Histogram mock to return different histograms
            mock_histogram_class.side_effect = [
                mock_duration_histogram,
                mock_cost_histogram
            ]

            # Set up Gauge mock
            mock_gauge_class.side_effect = [
                mock_queue_gauge,
                mock_buffer_gauge
            ]

            exporter = PrometheusExporter(prometheus_config)
            exporter.record_event(sample_event)

            # Verify operations counter was incremented
            mock_operations_counter.labels.assert_called_once_with(
                operation_name="test_operation",
                model="gpt-4",
                operation_type="completion"
            )
            mock_operations_counter.labels().inc.assert_called_once()

            # Verify duration histogram was observed
            mock_duration_histogram.labels.assert_called_once_with(
                operation_name="test_operation",
                model="gpt-4"
            )
            mock_duration_histogram.labels().observe.assert_called_once_with(0.2505)

            # Verify characters counter was incremented
            mock_characters_counter.labels.assert_called_once_with(
                operation_name="test_operation"
            )
            mock_characters_counter.labels().inc.assert_called_once_with(1000)

            # Verify cost histogram was observed
            mock_cost_histogram.labels.assert_called_once_with(
                operation_name="test_operation",
                model="gpt-4"
            )
            mock_cost_histogram.labels().observe.assert_called_once_with(0.015)

        logger.info("✓ Metric updates test passed")

    def test_record_event_with_error(self, prometheus_config, error_event):
        """Test that record_event handles error events correctly."""
        logger.info("Testing error event recording...")

        mock_operations_counter = Mock()
        mock_operations_counter.labels = Mock(return_value=Mock(inc=Mock()))

        mock_errors_counter = Mock()
        mock_errors_counter.labels = Mock(return_value=Mock(inc=Mock()))

        mock_duration_histogram = Mock()
        mock_duration_histogram.labels = Mock(return_value=Mock(observe=Mock()))

        mock_characters_counter = Mock()
        mock_cost_histogram = Mock()
        mock_queue_gauge = Mock()
        mock_buffer_gauge = Mock()

        with patch('llmops_monitoring.exporters.prometheus.Counter') as mock_counter_class, \
             patch('llmops_monitoring.exporters.prometheus.Histogram') as mock_histogram_class, \
             patch('llmops_monitoring.exporters.prometheus.Gauge') as mock_gauge_class, \
             patch('llmops_monitoring.exporters.prometheus.start_http_server'):

            mock_counter_class.side_effect = [
                mock_operations_counter,
                mock_errors_counter,
                mock_characters_counter
            ]

            mock_histogram_class.side_effect = [
                mock_duration_histogram,
                mock_cost_histogram
            ]

            mock_gauge_class.side_effect = [
                mock_queue_gauge,
                mock_buffer_gauge
            ]

            exporter = PrometheusExporter(prometheus_config)
            exporter.record_event(error_event)

            # Verify error counter was incremented
            mock_errors_counter.labels.assert_called_once_with(
                operation_name="failing_operation",
                error_type="ValueError"
            )
            mock_errors_counter.labels().inc.assert_called_once()

        logger.info("✓ Error event recording test passed")

    def test_update_queue_metrics(self, prometheus_config):
        """Test update_queue_metrics updates gauges."""
        logger.info("Testing queue metrics update...")

        mock_queue_gauge = Mock()
        mock_queue_gauge.set = Mock()

        mock_buffer_gauge = Mock()
        mock_buffer_gauge.set = Mock()

        mock_operations_counter = Mock()
        mock_errors_counter = Mock()
        mock_characters_counter = Mock()
        mock_duration_histogram = Mock()
        mock_cost_histogram = Mock()

        with patch('llmops_monitoring.exporters.prometheus.Counter') as mock_counter_class, \
             patch('llmops_monitoring.exporters.prometheus.Histogram') as mock_histogram_class, \
             patch('llmops_monitoring.exporters.prometheus.Gauge') as mock_gauge_class, \
             patch('llmops_monitoring.exporters.prometheus.start_http_server'):

            mock_counter_class.side_effect = [
                mock_operations_counter,
                mock_errors_counter,
                mock_characters_counter
            ]

            mock_histogram_class.side_effect = [
                mock_duration_histogram,
                mock_cost_histogram
            ]

            mock_gauge_class.side_effect = [
                mock_queue_gauge,
                mock_buffer_gauge
            ]

            exporter = PrometheusExporter(prometheus_config)
            exporter.update_queue_metrics(queue_size=42, buffer_size=15)

            # Verify gauges were updated
            mock_queue_gauge.set.assert_called_once_with(42)
            mock_buffer_gauge.set.assert_called_once_with(15)

        logger.info("✓ Queue metrics update test passed")

    def test_update_queue_metrics_without_prometheus_client(self, prometheus_config):
        """Test update_queue_metrics when prometheus_client not installed."""
        logger.info("Testing queue metrics update without prometheus_client...")

        with patch.object(PrometheusExporter, '_check_prometheus_client', return_value=False):
            exporter = PrometheusExporter(prometheus_config)

            # Should not raise error
            exporter.update_queue_metrics(queue_size=10, buffer_size=5)

        logger.info("✓ Queue metrics update without prometheus_client test passed")

    @pytest.mark.asyncio
    async def test_shutdown(self, prometheus_config):
        """Test shutdown method."""
        logger.info("Testing shutdown...")

        mock_counter = Mock()
        mock_histogram = Mock()
        mock_gauge = Mock()

        with patch('llmops_monitoring.exporters.prometheus.Counter', mock_counter), \
             patch('llmops_monitoring.exporters.prometheus.Histogram', mock_histogram), \
             patch('llmops_monitoring.exporters.prometheus.Gauge', mock_gauge), \
             patch('llmops_monitoring.exporters.prometheus.start_http_server'):

            exporter = PrometheusExporter(prometheus_config)
            exporter._metrics_exported = 100

            await exporter.shutdown()

            # Verify shutdown completed (mainly checking it doesn't raise)
            assert exporter._metrics_exported == 100

        logger.info("✓ Shutdown test passed")

    def test_health_check(self, prometheus_config):
        """Test health_check returns correct status."""
        logger.info("Testing health_check...")

        mock_counter = Mock()
        mock_histogram = Mock()
        mock_gauge = Mock()

        with patch('llmops_monitoring.exporters.prometheus.Counter', mock_counter), \
             patch('llmops_monitoring.exporters.prometheus.Histogram', mock_histogram), \
             patch('llmops_monitoring.exporters.prometheus.Gauge', mock_gauge), \
             patch('llmops_monitoring.exporters.prometheus.start_http_server'):

            exporter = PrometheusExporter(prometheus_config)
            exporter._metrics_exported = 42

            health = exporter.health_check()

            assert health["healthy"] is True
            assert health["server_running"] is True
            assert health["metrics_exported"] == 42
            assert health["endpoint"] == "http://0.0.0.0:8000/metrics"

        logger.info("✓ Health check test passed")

    def test_health_check_when_unavailable(self, prometheus_config):
        """Test health_check when prometheus_client not available."""
        logger.info("Testing health_check when prometheus_client unavailable...")

        with patch.object(PrometheusExporter, '_check_prometheus_client', return_value=False):
            exporter = PrometheusExporter(prometheus_config)

            health = exporter.health_check()

            assert health["healthy"] is False
            assert health["server_running"] is False
            assert health["metrics_exported"] == 0

        logger.info("✓ Health check when unavailable test passed")

    def test_exporter_type(self, prometheus_config):
        """Test exporter_type property."""
        logger.info("Testing exporter_type property...")

        with patch.object(PrometheusExporter, '_check_prometheus_client', return_value=False):
            exporter = PrometheusExporter(prometheus_config)

            assert exporter.exporter_type == "prometheus"

        logger.info("✓ Exporter type test passed")

    def test_extract_model_from_custom_attributes(self, prometheus_config):
        """Test model extraction from custom attributes."""
        logger.info("Testing model extraction...")

        mock_counter = Mock()
        mock_histogram = Mock()
        mock_gauge = Mock()

        with patch('llmops_monitoring.exporters.prometheus.Counter', mock_counter), \
             patch('llmops_monitoring.exporters.prometheus.Histogram', mock_histogram), \
             patch('llmops_monitoring.exporters.prometheus.Gauge', mock_gauge), \
             patch('llmops_monitoring.exporters.prometheus.start_http_server'):

            exporter = PrometheusExporter(prometheus_config)

            # Test with model in custom attributes
            event = MetricEvent(
                event_id=uuid4(),
                schema_version="1.0.0",
                session_id="test",
                trace_id="test",
                span_id="test",
                operation_name="test",
                operation_type="test",
                timestamp=datetime.utcnow(),
                custom_attributes={"model": "gpt-4"}
            )
            model = exporter._extract_model(event)
            assert model == "gpt-4"

            # Test without model in custom attributes
            event_no_model = MetricEvent(
                event_id=uuid4(),
                schema_version="1.0.0",
                session_id="test",
                trace_id="test",
                span_id="test",
                operation_name="test",
                operation_type="test",
                timestamp=datetime.utcnow()
            )
            model = exporter._extract_model(event_no_model)
            assert model == "unknown"

        logger.info("✓ Model extraction test passed")

    def test_extract_cost_from_custom_attributes(self, prometheus_config):
        """Test cost extraction from custom attributes."""
        logger.info("Testing cost extraction...")

        mock_counter = Mock()
        mock_histogram = Mock()
        mock_gauge = Mock()

        with patch('llmops_monitoring.exporters.prometheus.Counter', mock_counter), \
             patch('llmops_monitoring.exporters.prometheus.Histogram', mock_histogram), \
             patch('llmops_monitoring.exporters.prometheus.Gauge', mock_gauge), \
             patch('llmops_monitoring.exporters.prometheus.start_http_server'):

            exporter = PrometheusExporter(prometheus_config)

            # Test with estimated_cost_usd
            event1 = MetricEvent(
                event_id=uuid4(),
                schema_version="1.0.0",
                session_id="test",
                trace_id="test",
                span_id="test",
                operation_name="test",
                operation_type="test",
                timestamp=datetime.utcnow(),
                custom_attributes={"estimated_cost_usd": 0.123}
            )
            cost = exporter._extract_cost(event1)
            assert cost == 0.123

            # Test with cost_usd
            event2 = MetricEvent(
                event_id=uuid4(),
                schema_version="1.0.0",
                session_id="test",
                trace_id="test",
                span_id="test",
                operation_name="test",
                operation_type="test",
                timestamp=datetime.utcnow(),
                custom_attributes={"cost_usd": 0.456}
            )
            cost = exporter._extract_cost(event2)
            assert cost == 0.456

            # Test without cost
            event3 = MetricEvent(
                event_id=uuid4(),
                schema_version="1.0.0",
                session_id="test",
                trace_id="test",
                span_id="test",
                operation_name="test",
                operation_type="test",
                timestamp=datetime.utcnow()
            )
            cost = exporter._extract_cost(event3)
            assert cost is None

        logger.info("✓ Cost extraction test passed")

    def test_custom_config_values(self):
        """Test custom configuration values."""
        logger.info("Testing custom configuration...")

        custom_config = {
            "enabled": True,
            "port": 9090,
            "host": "127.0.0.1",
            "path": "/custom-metrics",
            "include_labels": ["operation_name", "model"]
        }

        with patch.object(PrometheusExporter, '_check_prometheus_client', return_value=False):
            exporter = PrometheusExporter(custom_config)

            assert exporter.port == 9090
            assert exporter.host == "127.0.0.1"
            assert exporter.path == "/custom-metrics"
            assert exporter.include_labels == ["operation_name", "model"]

        logger.info("✓ Custom configuration test passed")


def run_tests():
    """Run all tests (for manual execution)."""
    logger.info("=" * 60)
    logger.info("Running Prometheus Exporter Tests")
    logger.info("=" * 60)

    # Run with pytest
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
