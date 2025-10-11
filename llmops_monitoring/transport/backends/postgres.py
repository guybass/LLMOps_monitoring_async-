"""
PostgreSQL storage backend for production deployments.

Stores events in PostgreSQL with proper indexing for efficient queries.
"""

import asyncio
import logging
from typing import List, Optional
import json

from llmops_monitoring.schema.config import StorageConfig
from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.transport.backends.base import StorageBackend


logger = logging.getLogger(__name__)


class PostgresBackend(StorageBackend):
    """
    PostgreSQL storage backend.

    Features:
    - Connection pooling
    - Automatic schema creation
    - Efficient batch inserts
    - Indexed for hierarchical queries

    Extension Point: Customize schema, add materialized views
    """

    def __init__(self, config: StorageConfig):
        self.config = config
        self.pool: Optional[any] = None
        self._asyncpg_available = self._check_asyncpg()

    async def initialize(self) -> None:
        """Initialize connection pool and create tables."""
        if not self._asyncpg_available:
            raise RuntimeError(
                "PostgreSQL backend requires asyncpg. "
                "Install with: pip install 'llmops-monitoring[postgres]'"
            )

        import asyncpg

        # Create connection pool
        self.pool = await asyncpg.create_pool(
            self.config.connection_string,
            min_size=1,
            max_size=self.config.pool_size
        )

        # Create tables
        await self._create_tables()

        logger.info("Initialized PostgreSQL backend")

    async def write_event(self, event: MetricEvent) -> None:
        """Write a single event to PostgreSQL."""
        await self.write_batch([event])

    async def write_batch(self, events: List[MetricEvent]) -> None:
        """Write multiple events in a batch."""
        if not events:
            return

        async with self.pool.acquire() as conn:
            # Prepare records
            records = [self._event_to_record(event) for event in events]

            # Batch insert
            await conn.executemany(
                self._get_insert_query(),
                records
            )

        logger.debug(f"Wrote {len(events)} events to PostgreSQL")

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
        logger.info("Closed PostgreSQL backend")

    async def health_check(self) -> bool:
        """Check if PostgreSQL is accessible."""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False

    async def _create_tables(self) -> None:
        """Create metric_events table and indexes."""
        schema = f"{self.config.schema_name}." if self.config.schema_name else ""
        table_name = f"{schema}{self.config.table_name}"

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            event_id UUID PRIMARY KEY,
            schema_version VARCHAR(20) NOT NULL,

            -- Hierarchical tracking
            session_id VARCHAR(255) NOT NULL,
            trace_id VARCHAR(255) NOT NULL,
            span_id VARCHAR(255) NOT NULL,
            parent_span_id VARCHAR(255),

            -- Operation metadata
            operation_name VARCHAR(255) NOT NULL,
            operation_type VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            duration_ms FLOAT,

            -- Text metrics
            text_char_count INTEGER,
            text_word_count INTEGER,
            text_byte_size INTEGER,
            text_line_count INTEGER,
            text_custom_metrics JSONB,

            -- Image metrics
            image_count INTEGER,
            image_total_pixels BIGINT,
            image_file_size_bytes BIGINT,
            image_width INTEGER,
            image_height INTEGER,
            image_format VARCHAR(20),
            image_custom_metrics JSONB,

            -- Error tracking
            error TEXT,
            error_type VARCHAR(255),

            -- Custom attributes
            custom_attributes JSONB,

            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # Create indexes for common queries
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_session ON {table_name}(session_id, timestamp DESC);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_trace ON {table_name}(trace_id, timestamp DESC);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_span ON {table_name}(span_id);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_parent_span ON {table_name}(parent_span_id);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_timestamp ON {table_name}(timestamp DESC);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_operation ON {table_name}(operation_name, timestamp DESC);",
        ]

        async with self.pool.acquire() as conn:
            await conn.execute(create_table_sql)
            for index_sql in indexes:
                await conn.execute(index_sql)

        logger.info(f"Created table {table_name} with indexes")

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
            json.dumps(text_metrics.custom_metrics) if text_metrics else None,
            image_metrics.count if image_metrics else None,
            image_metrics.total_pixels if image_metrics else None,
            image_metrics.file_size_bytes if image_metrics else None,
            image_metrics.width if image_metrics else None,
            image_metrics.height if image_metrics else None,
            image_metrics.format if image_metrics else None,
            json.dumps(image_metrics.custom_metrics) if image_metrics else None,
            event.error,
            event.error_type,
            json.dumps(event.custom_attributes),
        )

    def _get_insert_query(self) -> str:
        """Get INSERT query for events."""
        schema = f"{self.config.schema_name}." if self.config.schema_name else ""
        table_name = f"{schema}{self.config.table_name}"

        return f"""
        INSERT INTO {table_name} (
            event_id, schema_version, session_id, trace_id, span_id, parent_span_id,
            operation_name, operation_type, timestamp, duration_ms,
            text_char_count, text_word_count, text_byte_size, text_line_count, text_custom_metrics,
            image_count, image_total_pixels, image_file_size_bytes, image_width, image_height, image_format, image_custom_metrics,
            error, error_type, custom_attributes
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25)
        """

    def _check_asyncpg(self) -> bool:
        """Check if asyncpg is available."""
        try:
            import asyncpg
            return True
        except ImportError:
            return False

    def supports_batch_writes(self) -> bool:
        return True
