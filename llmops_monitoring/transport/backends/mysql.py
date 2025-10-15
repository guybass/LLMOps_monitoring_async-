"""
MySQL storage backend for production deployments.

Stores events in MySQL with proper indexing for efficient queries.
"""

import asyncio
import logging
from typing import List, Optional
import json

from llmops_monitoring.schema.config import StorageConfig
from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.transport.backends.base import StorageBackend
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class MySQLBackend(StorageBackend):
    """
    MySQL storage backend.

    Features:
    - Connection pooling
    - Automatic schema creation
    - Efficient batch inserts
    - Indexed for hierarchical queries

    Extension Point: Customize schema, add materialized views

    Example:
        ```python
        from llmops_monitoring import MonitorConfig
        from llmops_monitoring.schema.config import StorageConfig

        config = MonitorConfig(
            storage=StorageConfig(
                backend="mysql",
                connection_string="mysql://user:pass@localhost:3306/monitoring",
                table_name="metric_events",
                pool_size=10
            )
        )
        ```
    """

    def __init__(self, config: StorageConfig):
        self.config = config
        self.pool: Optional[any] = None
        self._aiomysql_available = self._check_aiomysql()

    async def initialize(self) -> None:
        """Initialize connection pool and create tables."""
        if not self._aiomysql_available:
            raise RuntimeError(
                "MySQL backend requires aiomysql. "
                "Install with: pip install 'llamonitor-async[mysql]' or pip install aiomysql"
            )

        import aiomysql

        # Parse connection string for aiomysql
        conn_params = self._parse_connection_string()

        # Create connection pool
        self.pool = await aiomysql.create_pool(
            minsize=1,
            maxsize=self.config.pool_size,
            **conn_params
        )

        # Create tables
        await self._create_tables()

        logger.info("Initialized MySQL backend")

    async def write_event(self, event: MetricEvent) -> None:
        """Write a single event to MySQL."""
        await self.write_batch([event])

    async def write_batch(self, events: List[MetricEvent]) -> None:
        """Write multiple events in a batch."""
        if not events:
            return

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Prepare records
                records = [self._event_to_record(event) for event in events]

                # Batch insert
                await cursor.executemany(
                    self._get_insert_query(),
                    records
                )
                await conn.commit()

        logger.debug(f"Wrote {len(events)} events to MySQL")

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
        logger.info("Closed MySQL backend")

    async def health_check(self) -> bool:
        """Check if MySQL is accessible."""
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    await cursor.fetchone()
            return True
        except Exception as e:
            logger.error(f"MySQL health check failed: {e}")
            return False

    async def _create_tables(self) -> None:
        """Create metric_events table and indexes."""
        database = self.config.schema_name or ""
        table_name = self.config.table_name

        # If database is specified, use it
        table_ref = f"`{database}`.`{table_name}`" if database else f"`{table_name}`"

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_ref} (
            event_id CHAR(36) PRIMARY KEY,
            schema_version VARCHAR(20) NOT NULL,

            -- Hierarchical tracking
            session_id VARCHAR(255) NOT NULL,
            trace_id VARCHAR(255) NOT NULL,
            span_id VARCHAR(255) NOT NULL,
            parent_span_id VARCHAR(255),

            -- Operation metadata
            operation_name VARCHAR(255) NOT NULL,
            operation_type VARCHAR(50) NOT NULL,
            timestamp DATETIME(6) NOT NULL,
            duration_ms DOUBLE,

            -- Text metrics
            text_char_count INT,
            text_word_count INT,
            text_byte_size INT,
            text_line_count INT,
            text_custom_metrics JSON,

            -- Image metrics
            image_count INT,
            image_total_pixels BIGINT,
            image_file_size_bytes BIGINT,
            image_width INT,
            image_height INT,
            image_format VARCHAR(20),
            image_custom_metrics JSON,

            -- Error tracking
            error TEXT,
            error_type VARCHAR(255),

            -- Custom attributes
            custom_attributes JSON,

            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Indexes
            INDEX idx_session (session_id, timestamp),
            INDEX idx_trace (trace_id, timestamp),
            INDEX idx_span (span_id),
            INDEX idx_parent_span (parent_span_id),
            INDEX idx_timestamp (timestamp),
            INDEX idx_operation (operation_name, timestamp)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(create_table_sql)
                await conn.commit()

        logger.info(f"Created table {table_ref} with indexes")

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
            json.dumps(text_metrics.custom_metrics) if text_metrics and text_metrics.custom_metrics else None,
            image_metrics.count if image_metrics else None,
            image_metrics.total_pixels if image_metrics else None,
            image_metrics.file_size_bytes if image_metrics else None,
            image_metrics.width if image_metrics else None,
            image_metrics.height if image_metrics else None,
            image_metrics.format if image_metrics else None,
            json.dumps(image_metrics.custom_metrics) if image_metrics and image_metrics.custom_metrics else None,
            event.error,
            event.error_type,
            json.dumps(event.custom_attributes) if event.custom_attributes else None,
        )

    def _get_insert_query(self) -> str:
        """Get INSERT query for events."""
        database = self.config.schema_name or ""
        table_name = self.config.table_name
        table_ref = f"`{database}`.`{table_name}`" if database else f"`{table_name}`"

        return f"""
        INSERT INTO {table_ref} (
            event_id, schema_version, session_id, trace_id, span_id, parent_span_id,
            operation_name, operation_type, timestamp, duration_ms,
            text_char_count, text_word_count, text_byte_size, text_line_count, text_custom_metrics,
            image_count, image_total_pixels, image_file_size_bytes, image_width, image_height, image_format, image_custom_metrics,
            error, error_type, custom_attributes
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

    def _parse_connection_string(self) -> dict:
        """
        Parse connection string into aiomysql parameters.

        Supports formats:
        - mysql://user:password@host:port/database
        - mysql://user:password@host/database
        - Direct dict with keys: host, port, user, password, db
        """
        conn_str = self.config.connection_string

        # If it's already a dict, return it
        if isinstance(conn_str, dict):
            return conn_str

        # Parse URL format
        if conn_str.startswith("mysql://"):
            # Remove mysql:// prefix
            conn_str = conn_str[8:]

            # Split user:pass@host:port/db
            if "@" in conn_str:
                auth, rest = conn_str.split("@", 1)
                user, password = auth.split(":", 1) if ":" in auth else (auth, "")
            else:
                user, password = "root", ""
                rest = conn_str

            # Split host:port/db
            if "/" in rest:
                host_port, db = rest.split("/", 1)
            else:
                host_port, db = rest, "monitoring"

            # Split host:port
            if ":" in host_port:
                host, port = host_port.split(":", 1)
                port = int(port)
            else:
                host, port = host_port, 3306

            return {
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "db": db,
                "autocommit": False,
                "charset": "utf8mb4"
            }

        # Fallback to localhost
        return {
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "",
            "db": "monitoring",
            "autocommit": False,
            "charset": "utf8mb4"
        }

    def _check_aiomysql(self) -> bool:
        """Check if aiomysql is available."""
        try:
            import aiomysql
            return True
        except ImportError:
            return False

    def supports_batch_writes(self) -> bool:
        return True
