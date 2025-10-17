"""
PostgreSQL query backend for reading and aggregating stored events.

Provides efficient SQL-based querying with aggregations.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from llmops_monitoring.schema.config import StorageConfig
from llmops_monitoring.schema.events import MetricEvent, TextMetrics, ImageMetrics
from llmops_monitoring.transport.backends.query_base import QueryBackend, QueryFilter


logger = logging.getLogger(__name__)


class PostgresQueryBackend(QueryBackend):
    """
    Query backend for PostgreSQL.

    Features:
    - Efficient SQL queries with proper indexing
    - Aggregations using native SQL functions
    - JSON field querying for custom attributes
    - Percentile calculations
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize PostgreSQL query backend.

        Args:
            config: Storage configuration with connection_string
        """
        self.config = config
        self.pool = None
        self._asyncpg_available = self._check_asyncpg()

    def _check_asyncpg(self) -> bool:
        """Check if asyncpg is available."""
        try:
            import asyncpg
            return True
        except ImportError:
            return False

    async def initialize(self):
        """Initialize connection pool."""
        if not self._asyncpg_available:
            raise RuntimeError(
                "PostgreSQL query backend requires asyncpg. "
                "Install with: pip install 'llamonitor-async[postgres]'"
            )

        import asyncpg

        self.pool = await asyncpg.create_pool(
            self.config.connection_string,
            min_size=1,
            max_size=self.config.pool_size
        )

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()

    def _get_table_name(self) -> str:
        """Get full table name with schema."""
        schema = f"{self.config.schema_name}." if self.config.schema_name else ""
        return f"{schema}{self.config.table_name}"

    def _record_to_event(self, record: Dict) -> MetricEvent:
        """Convert database record to MetricEvent."""
        # Parse text metrics
        text_metrics = None
        if record.get("text_char_count") is not None:
            text_metrics = TextMetrics(
                char_count=record.get("text_char_count"),
                word_count=record.get("text_word_count"),
                byte_size=record.get("text_byte_size"),
                line_count=record.get("text_line_count"),
                custom_metrics=json.loads(record.get("text_custom_metrics") or "{}")
            )

        # Parse image metrics
        image_metrics = None
        if record.get("image_count") is not None:
            image_metrics = ImageMetrics(
                count=record.get("image_count"),
                total_pixels=record.get("image_total_pixels"),
                file_size_bytes=record.get("image_file_size_bytes"),
                width=record.get("image_width"),
                height=record.get("image_height"),
                format=record.get("image_format"),
                custom_metrics=json.loads(record.get("image_custom_metrics") or "{}")
            )

        # Parse custom attributes
        custom_attributes = json.loads(record.get("custom_attributes") or "{}")

        return MetricEvent(
            event_id=UUID(record["event_id"]),
            schema_version=record["schema_version"],
            session_id=record["session_id"],
            trace_id=record["trace_id"],
            span_id=record["span_id"],
            parent_span_id=record.get("parent_span_id"),
            operation_name=record["operation_name"],
            operation_type=record["operation_type"],
            timestamp=record["timestamp"],
            duration_ms=record.get("duration_ms"),
            error=bool(record.get("error")),
            error_type=record.get("error_type"),
            error_message=None,  # Not stored in current schema
            text_metrics=text_metrics,
            image_metrics=image_metrics,
            custom_attributes=custom_attributes
        )

    async def query_events(self, filter: QueryFilter) -> List[MetricEvent]:
        """Query events with filtering."""
        table_name = self._get_table_name()

        # Build WHERE clause
        where_clauses = []
        params = []
        param_idx = 1

        if filter.session_id:
            where_clauses.append(f"session_id = ${param_idx}")
            params.append(filter.session_id)
            param_idx += 1

        if filter.trace_id:
            where_clauses.append(f"trace_id = ${param_idx}")
            params.append(filter.trace_id)
            param_idx += 1

        if filter.operation_name:
            where_clauses.append(f"operation_name = ${param_idx}")
            params.append(filter.operation_name)
            param_idx += 1

        if filter.operation_type:
            where_clauses.append(f"operation_type = ${param_idx}")
            params.append(filter.operation_type)
            param_idx += 1

        if filter.start_time:
            where_clauses.append(f"timestamp >= ${param_idx}")
            params.append(filter.start_time)
            param_idx += 1

        if filter.end_time:
            where_clauses.append(f"timestamp <= ${param_idx}")
            params.append(filter.end_time)
            param_idx += 1

        if filter.error_only:
            where_clauses.append("error IS NOT NULL")

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            SELECT * FROM {table_name}
            {where_sql}
            ORDER BY timestamp DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([filter.limit, filter.offset])

        async with self.pool.acquire() as conn:
            records = await conn.fetch(query, *params)

        events = [self._record_to_event(dict(r)) for r in records]
        return events

    async def get_sessions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get list of sessions with summary stats."""
        table_name = self._get_table_name()

        where_clauses = []
        params = []
        param_idx = 1

        if start_time:
            where_clauses.append(f"timestamp >= ${param_idx}")
            params.append(start_time)
            param_idx += 1

        if end_time:
            where_clauses.append(f"timestamp <= ${param_idx}")
            params.append(end_time)
            param_idx += 1

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            SELECT
                session_id,
                MIN(timestamp) as start_time,
                MAX(timestamp) as end_time,
                COUNT(*) as event_count,
                COUNT(DISTINCT operation_name) as operation_count,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count,
                SUM(duration_ms) as total_duration_ms
            FROM {table_name}
            {where_sql}
            GROUP BY session_id
            ORDER BY start_time DESC
            LIMIT ${param_idx}
        """
        params.append(limit)

        async with self.pool.acquire() as conn:
            records = await conn.fetch(query, *params)

        return [dict(r) for r in records]

    async def get_traces(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get list of traces for a session."""
        table_name = self._get_table_name()

        query = f"""
            SELECT
                trace_id,
                MIN(timestamp) as start_time,
                MAX(timestamp) as end_time,
                COUNT(*) as event_count,
                COUNT(DISTINCT span_id) as span_count,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count,
                SUM(duration_ms) as total_duration_ms
            FROM {table_name}
            WHERE session_id = $1
            GROUP BY trace_id
            ORDER BY start_time DESC
            LIMIT $2
        """

        async with self.pool.acquire() as conn:
            records = await conn.fetch(query, session_id, limit)

        return [dict(r) for r in records]

    async def aggregate_by_operation(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics by operation name."""
        table_name = self._get_table_name()

        where_clauses = []
        params = []
        param_idx = 1

        if start_time:
            where_clauses.append(f"timestamp >= ${param_idx}")
            params.append(start_time)
            param_idx += 1

        if end_time:
            where_clauses.append(f"timestamp <= ${param_idx}")
            params.append(end_time)
            param_idx += 1

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            SELECT
                operation_name,
                COUNT(*) as count,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count,
                AVG(duration_ms) as avg_duration_ms,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) as p50_duration_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_duration_ms,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99_duration_ms,
                SUM(text_char_count) as total_characters,
                SUM(CAST(custom_attributes->>'estimated_cost_usd' AS FLOAT)) as estimated_cost_usd
            FROM {table_name}
            {where_sql}
            GROUP BY operation_name
            ORDER BY count DESC
        """

        async with self.pool.acquire() as conn:
            records = await conn.fetch(query, *params)

        return [dict(r) for r in records]

    async def aggregate_by_model(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics by model."""
        table_name = self._get_table_name()

        where_clauses = []
        params = []
        param_idx = 1

        if start_time:
            where_clauses.append(f"timestamp >= ${param_idx}")
            params.append(start_time)
            param_idx += 1

        if end_time:
            where_clauses.append(f"timestamp <= ${param_idx}")
            params.append(end_time)
            param_idx += 1

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            SELECT
                COALESCE(custom_attributes->>'model', 'unknown') as model,
                COUNT(*) as count,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count,
                AVG(duration_ms) as avg_duration_ms,
                SUM(text_char_count) as total_characters,
                SUM(CAST(custom_attributes->>'estimated_cost_usd' AS FLOAT)) as estimated_cost_usd
            FROM {table_name}
            {where_sql}
            GROUP BY custom_attributes->>'model'
            ORDER BY count DESC
        """

        async with self.pool.acquire() as conn:
            records = await conn.fetch(query, *params)

        return [dict(r) for r in records]

    async def aggregate_costs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        group_by: str = "model"
    ) -> List[Dict[str, Any]]:
        """Aggregate cost metrics."""
        table_name = self._get_table_name()

        where_clauses = []
        params = []
        param_idx = 1

        if start_time:
            where_clauses.append(f"timestamp >= ${param_idx}")
            params.append(start_time)
            param_idx += 1

        if end_time:
            where_clauses.append(f"timestamp <= ${param_idx}")
            params.append(end_time)
            param_idx += 1

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        # Determine grouping field
        if group_by == "model":
            group_field = "COALESCE(custom_attributes->>'model', 'unknown')"
        elif group_by == "operation":
            group_field = "operation_name"
        elif group_by == "session":
            group_field = "session_id"
        elif group_by == "day":
            group_field = "DATE(timestamp)"
        else:
            raise ValueError(f"Invalid group_by: {group_by}")

        query = f"""
            SELECT
                {group_field} as {group_by},
                SUM(CAST(custom_attributes->>'estimated_cost_usd' AS FLOAT)) as total_cost_usd,
                COUNT(*) as operation_count,
                SUM(CAST(custom_attributes->>'estimated_cost_usd' AS FLOAT)) / COUNT(*) as avg_cost_per_operation_usd
            FROM {table_name}
            {where_sql}
            GROUP BY {group_field}
            ORDER BY total_cost_usd DESC
        """

        async with self.pool.acquire() as conn:
            records = await conn.fetch(query, *params)

        return [dict(r) for r in records]

    async def get_summary_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get overall summary statistics."""
        table_name = self._get_table_name()

        where_clauses = []
        params = []
        param_idx = 1

        if start_time:
            where_clauses.append(f"timestamp >= ${param_idx}")
            params.append(start_time)
            param_idx += 1

        if end_time:
            where_clauses.append(f"timestamp <= ${param_idx}")
            params.append(end_time)
            param_idx += 1

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            SELECT
                COUNT(*) as total_events,
                COUNT(DISTINCT session_id) as total_sessions,
                COUNT(DISTINCT operation_name) as total_operations,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as total_errors,
                CAST(SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS FLOAT) / NULLIF(COUNT(*), 0) as error_rate,
                AVG(duration_ms) as avg_duration_ms,
                SUM(text_char_count) as total_characters,
                SUM(CAST(custom_attributes->>'estimated_cost_usd' AS FLOAT)) as total_cost_usd
            FROM {table_name}
            {where_sql}
        """

        async with self.pool.acquire() as conn:
            record = await conn.fetchrow(query, *params)

        return dict(record) if record else {}

    async def health_check(self) -> bool:
        """Check if the query backend is healthy."""
        try:
            if not self.pool:
                return False

            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL query health check failed: {e}")
            return False
