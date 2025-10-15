"""
Tests for MySQL backend

These tests verify MySQL backend functionality without requiring
a real MySQL database (uses mocking).
"""

import asyncio
import logging
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from uuid import uuid4

import pytest

from llmops_monitoring.schema.config import StorageConfig
from llmops_monitoring.schema.events import MetricEvent, TextMetrics
from llmops_monitoring.transport.backends.mysql import MySQLBackend


# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@pytest.fixture
def mysql_config():
    """Create MySQL storage configuration for testing."""
    return StorageConfig(
        backend="mysql",
        connection_string="mysql://test:test@localhost:3306/test_monitoring",
        table_name="test_metric_events",
        pool_size=5,
        batch_size=10
    )


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
        operation_type="test",
        timestamp=datetime.utcnow(),
        duration_ms=100.5,
        text_metrics=TextMetrics(
            char_count=1000,
            word_count=200,
            byte_size=1500,
            line_count=50
        ),
        custom_attributes={"test": "value"}
    )


class TestMySQLBackend:
    """Test suite for MySQL backend."""

    def test_initialization(self, mysql_config):
        """Test MySQL backend initialization."""
        logger.info("Testing MySQL backend initialization...")
        backend = MySQLBackend(mysql_config)

        assert backend.config == mysql_config
        assert backend.pool is None
        logger.info("✓ Initialization test passed")

    def test_check_aiomysql_not_available(self, mysql_config):
        """Test when aiomysql is not installed."""
        logger.info("Testing aiomysql availability check...")

        with patch.object(MySQLBackend, '_check_aiomysql', return_value=False):
            backend = MySQLBackend(mysql_config)
            assert backend._aiomysql_available is False

        logger.info("✓ aiomysql check test passed")

    @pytest.mark.asyncio
    async def test_initialize_requires_aiomysql(self, mysql_config):
        """Test that initialize() requires aiomysql."""
        logger.info("Testing aiomysql requirement...")

        with patch.object(MySQLBackend, '_check_aiomysql', return_value=False):
            backend = MySQLBackend(mysql_config)

            with pytest.raises(RuntimeError, match="requires aiomysql"):
                await backend.initialize()

        logger.info("✓ aiomysql requirement test passed")

    @pytest.mark.asyncio
    async def test_initialize_with_mock_pool(self, mysql_config):
        """Test initialization with mocked connection pool."""
        logger.info("Testing initialization with mock pool...")

        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock()
        mock_pool.close = Mock()
        mock_pool.wait_closed = AsyncMock()

        # Mock the context manager
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(1,))
        mock_conn.cursor = Mock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock()
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock()

        mock_pool.acquire.return_value = mock_conn

        with patch('aiomysql.create_pool', AsyncMock(return_value=mock_pool)):
            backend = MySQLBackend(mysql_config)
            await backend.initialize()

            assert backend.pool is not None
            logger.info("✓ Mock pool initialization test passed")

            # Test health check
            health = await backend.health_check()
            assert health is True
            logger.info("✓ Health check test passed")

            # Close
            await backend.close()
            logger.info("✓ Close test passed")

    @pytest.mark.asyncio
    async def test_write_batch_with_mock(self, mysql_config, sample_event):
        """Test batch write with mocked MySQL."""
        logger.info("Testing batch write with mock...")

        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()

        mock_cursor.execute = AsyncMock()
        mock_cursor.executemany = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(1,))
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock()

        mock_conn.cursor = Mock(return_value=mock_cursor)
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock()

        mock_pool.acquire.return_value = mock_conn

        backend = MySQLBackend(mysql_config)
        backend.pool = mock_pool

        # Write batch
        events = [sample_event]
        await backend.write_batch(events)

        # Verify executemany was called
        mock_cursor.executemany.assert_called_once()
        mock_conn.commit.assert_called_once()

        logger.info("✓ Batch write test passed")

    def test_connection_string_parsing(self, mysql_config):
        """Test connection string parsing."""
        logger.info("Testing connection string parsing...")
        backend = MySQLBackend(mysql_config)

        # Test URL format
        params = backend._parse_connection_string()

        assert params["host"] == "localhost"
        assert params["port"] == 3306
        assert params["user"] == "test"
        assert params["password"] == "test"
        assert params["db"] == "test_monitoring"
        assert params["charset"] == "utf8mb4"

        logger.info("✓ Connection string parsing test passed")

    def test_event_to_record_conversion(self, mysql_config, sample_event):
        """Test event to tuple conversion."""
        logger.info("Testing event to record conversion...")
        backend = MySQLBackend(mysql_config)

        record = backend._event_to_record(sample_event)

        # Verify tuple has correct number of fields (25)
        assert len(record) == 25
        assert record[0] == str(sample_event.event_id)
        assert record[1] == sample_event.schema_version
        assert record[2] == sample_event.session_id
        assert record[10] == 1000  # text_char_count

        logger.info("✓ Event to record conversion test passed")

    def test_insert_query_generation(self, mysql_config):
        """Test INSERT query generation."""
        logger.info("Testing INSERT query generation...")
        backend = MySQLBackend(mysql_config)

        query = backend._get_insert_query()

        assert "INSERT INTO" in query
        assert "test_metric_events" in query
        assert "VALUES" in query
        assert "%s" in query  # MySQL placeholders

        logger.info("✓ INSERT query generation test passed")

    def test_supports_batch_writes(self, mysql_config):
        """Test batch write support flag."""
        logger.info("Testing batch write support...")
        backend = MySQLBackend(mysql_config)

        assert backend.supports_batch_writes() is True

        logger.info("✓ Batch write support test passed")


def run_tests():
    """Run all tests (for manual execution)."""
    logger.info("=" * 60)
    logger.info("Running MySQL Backend Tests")
    logger.info("=" * 60)

    # Run with pytest
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
