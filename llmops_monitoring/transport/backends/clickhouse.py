"""
ClickHouse storage backend for analytics workloads.

Stores events in ClickHouse for fast analytical queries.
ClickHouse is optimized for OLAP and time-series data.
"""

import asyncio
import json
import logging
from typing import List, Optional
from urllib.parse import urlparse

from llmops_monitoring.schema.config import StorageConfig
from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.transport.backends.base import StorageBackend
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class ClickHouseBackend(StorageBackend):
    """
    ClickHouse storage backend.

    Features:
    - Column-oriented storage for fast analytics
    - Automatic table creation with optimal schema
    - Partitioning by date
    - MergeTree engine for fast queries
    - Batch inserts for efficiency
    - Compression

    Example:
        ```python
        from llmops_monitoring import MonitorConfig
        from llmops_monitoring.schema.config import StorageConfig

        config = MonitorConfig(
            storage=StorageConfig(
                backend="clickhouse",
                connection_string="clickhouse://default:@localhost:9000/monitoring",
                table_name="metric_events",
                batch_size=1000
            )
        )
        ```
    """

    def __init__(self, config: StorageConfig):
        self.config = config
        self.client = None
        self._clickhouse_available = self._check_clickhouse_driver()

    async def initialize(self) -> None:
        """Initialize ClickHouse connection and create tables."""
        if not self._clickhouse_available:
            raise RuntimeError(
                "ClickHouse backend requires clickhouse-driver. "
                "Install with: pip install 'llamonitor-async[clickhouse]'"
            )

        from clickhouse_driver import Client

        # Parse connection string
        conn_params = self._parse_connection_string()

        # Create client
        self.client = Client(**conn_params)

        # Create tables
        await self._create_tables()

        logger.info("Initialized ClickHouse backend")

    async def write_event(self, event: MetricEvent) -> None:
        """Write a single event to ClickHouse."""
        await self.write_batch([event])

    async def write_batch(self, events: List[MetricEvent]) -> None:
        """Write multiple events in a batch."""
        if not events:
            return

        # Prepare records
        records = [self._event_to_record(event) for event in events]

        # Execute insert in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            None,
            self._execute_insert,
            records
        )

        logger.debug(f"Wrote {len(events)} events to ClickHouse")

    def _execute_insert(self, records: List[tuple]):
        """Execute batch insert (runs in thread pool)."""
        query = self._get_insert_query()
        self.client.execute(query, records)

    async def close(self) -> None:
        """Close ClickHouse connection."""
        if self.client:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.disconnect
            )
        logger.info("Closed ClickHouse backend")

    async def health_check(self) -> bool:
        """Check if ClickHouse is accessible."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.execute,
                "SELECT 1"
            )
            return result == [(1,)]
        except Exception as e:
            logger.error(f"ClickHouse health check failed: {e}")
            return False

    async def _create_tables(self) -> None:
        """Create metric_events table with optimal schema."""
        database = self.config.schema_name or "default"
        table_name = self.config.table_name

        # Create database if it doesn't exist
        create_db_sql = f"CREATE DATABASE IF NOT EXISTS {database}"

        # Create table with MergeTree engine
        # Partitioned by date for efficient time-range queries
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {database}.{table_name} (
            -- Event identification
            event_id String,
            schema_version String,

            -- Hierarchical tracking
            session_id String,
            trace_id String,
            span_id String,
            parent_span_id Nullable(String),

            -- Operation metadata
            operation_name String,
            operation_type String,
            timestamp DateTime64(3),
            duration_ms Nullable(Float64),

            -- Text metrics
            text_char_count Nullable(UInt32),
            text_word_count Nullable(UInt32),
            text_byte_size Nullable(UInt32),
            text_line_count Nullable(UInt32),

            -- Image metrics
            image_count Nullable(UInt32),
            image_total_pixels Nullable(UInt64),
            image_file_size_bytes Nullable(UInt64),
            image_width Nullable(UInt32),
            image_height Nullable(UInt32),
            image_format Nullable(String),

            -- Error tracking
            error Nullable(String),
            error_type Nullable(String),

            -- Custom attributes (JSON)
            custom_attributes String,

            -- Metadata
            created_at DateTime DEFAULT now()
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (session_id, trace_id, timestamp)
        SETTINGS index_granularity = 8192
        """

        # Execute in thread pool
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.client.execute,
            create_db_sql
        )

        await asyncio.get_event_loop().run_in_executor(
            None,
            self.client.execute,
            create_table_sql
        )

        logger.info(f"Created ClickHouse table {database}.{table_name}")

    def _event_to_record(self, event: MetricEvent) -> tuple:
        """Convert event to tuple for INSERT."""
        text_metrics = event.text_metrics
        image_metrics = event.image_metrics

        return (
            str(event.event_id),
            event.schema_version,
            event.session_id,
            event.trace_id,
            event.span_id,
            event.parent_span_id,
            event.operation_name,
            event.operation_type,
            event.timestamp,
            event.duration_ms,
            text_metrics.char_count if text_metrics else None,
            text_metrics.word_count if text_metrics else None,
            text_metrics.byte_size if text_metrics else None,
            text_metrics.line_count if text_metrics else None,
            image_metrics.count if image_metrics else None,
            image_metrics.total_pixels if image_metrics else None,
            image_metrics.file_size_bytes if image_metrics else None,
            image_metrics.width if image_metrics else None,
            image_metrics.height if image_metrics else None,
            image_metrics.format if image_metrics else None,
            event.error,
            event.error_type,
            json.dumps(event.custom_attributes) if event.custom_attributes else "{}"
        )

    def _get_insert_query(self) -> str:
        """Get INSERT query for events."""
        database = self.config.schema_name or "default"
        table_name = self.config.table_name

        return f"""
        INSERT INTO {database}.{table_name} (
            event_id, schema_version, session_id, trace_id, span_id, parent_span_id,
            operation_name, operation_type, timestamp, duration_ms,
            text_char_count, text_word_count, text_byte_size, text_line_count,
            image_count, image_total_pixels, image_file_size_bytes, image_width, image_height, image_format,
            error, error_type, custom_attributes
        ) VALUES
        """

    def _parse_connection_string(self) -> dict:
        """
        Parse connection string into ClickHouse client parameters.

        Supports formats:
        - clickhouse://user:password@host:port/database
        - clickhouse://host:port/database
        """
        conn_str = self.config.connection_string

        if conn_str.startswith("clickhouse://"):
            parsed = urlparse(conn_str)

            return {
                "host": parsed.hostname or "localhost",
                "port": parsed.port or 9000,
                "user": parsed.username or "default",
                "password": parsed.password or "",
                "database": parsed.path.lstrip("/") or "default",
                "compression": True,  # Enable compression
                "connect_timeout": 10,
                "send_receive_timeout": 300
            }

        # Fallback to localhost
        return {
            "host": "localhost",
            "port": 9000,
            "user": "default",
            "password": "",
            "database": "default",
            "compression": True
        }

    def _check_clickhouse_driver(self) -> bool:
        """Check if clickhouse-driver is available."""
        try:
            import clickhouse_driver
            return True
        except ImportError:
            return False

    def supports_batch_writes(self) -> bool:
        return True
