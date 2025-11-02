"""
ClickHouse query backend for fast analytical queries.

Provides high-performance querying and aggregation using ClickHouse.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
from uuid import UUID

from llmops_monitoring.schema.config import StorageConfig
from llmops_monitoring.schema.events import MetricEvent, TextMetrics, ImageMetrics
from llmops_monitoring.transport.backends.query_base import QueryBackend, QueryFilter
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class ClickHouseQueryBackend(QueryBackend):
    """
    Query backend for ClickHouse.

    Features:
    - Ultra-fast aggregations using column-oriented storage
    - Efficient time-range queries with partitioning
    - Percentile calculations using quantile functions
    - JSON field querying for custom attributes
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize ClickHouse query backend.

        Args:
            config: Storage configuration with connection_string
        """
        self.config = config
        self.client = None
        self._clickhouse_available = self._check_clickhouse_driver()

    def _check_clickhouse_driver(self) -> bool:
        """Check if clickhouse-driver is available."""
        try:
            import clickhouse_driver
            return True
        except ImportError:
            return False

    async def initialize(self):
        """Initialize ClickHouse connection."""
        if not self._clickhouse_available:
            raise RuntimeError(
                "ClickHouse query backend requires clickhouse-driver. "
                "Install with: pip install 'llamonitor-async[clickhouse]'"
            )

        from clickhouse_driver import Client

        conn_params = self._parse_connection_string()
        self.client = Client(**conn_params)

        logger.info("Initialized ClickHouse query backend")

    async def close(self):
        """Close ClickHouse connection."""
        if self.client:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.disconnect
            )

    def _get_table_name(self) -> str:
        """Get full table name with database."""
        database = self.config.schema_name or "default"
        return f"{database}.{self.config.table_name}"

    def _parse_connection_string(self) -> dict:
        """Parse connection string into ClickHouse client parameters."""
        conn_str = self.config.connection_string

        if conn_str.startswith("clickhouse://"):
            parsed = urlparse(conn_str)

            return {
                "host": parsed.hostname or "localhost",
                "port": parsed.port or 9000,
                "user": parsed.username or "default",
                "password": parsed.password or "",
                "database": parsed.path.lstrip("/") or "default",
                "compression": True
            }

        return {
            "host": "localhost",
            "port": 9000,
            "user": "default",
            "password": "",
            "database": "default",
            "compression": True
        }

    def _record_to_event(self, record: tuple, columns: List[str]) -> MetricEvent:
        """Convert ClickHouse record to MetricEvent."""
        # Create dict from record
        data = dict(zip(columns, record))

        # Parse text metrics
        text_metrics = None
        if data.get("text_char_count") is not None:
            text_metrics = TextMetrics(
                char_count=data.get("text_char_count"),
                word_count=data.get("text_word_count"),
                byte_size=data.get("text_byte_size"),
                line_count=data.get("text_line_count"),
                custom_metrics={}
            )

        # Parse image metrics
        image_metrics = None
        if data.get("image_count") is not None:
            image_metrics = ImageMetrics(
                count=data.get("image_count"),
                total_pixels=data.get("image_total_pixels"),
                file_size_bytes=data.get("image_file_size_bytes"),
                width=data.get("image_width"),
                height=data.get("image_height"),
                format=data.get("image_format"),
                custom_metrics={}
            )

        # Parse custom attributes
        custom_attrs = data.get("custom_attributes", "{}")
        custom_attributes = json.loads(custom_attrs) if custom_attrs else {}

        return MetricEvent(
            event_id=UUID(data["event_id"]),
            schema_version=data["schema_version"],
            session_id=data["session_id"],
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            operation_name=data["operation_name"],
            operation_type=data["operation_type"],
            timestamp=data["timestamp"],
            duration_ms=data.get("duration_ms"),
            error=bool(data.get("error")),
            error_type=data.get("error_type"),
            error_message=None,
            text_metrics=text_metrics,
            image_metrics=image_metrics,
            custom_attributes=custom_attributes
        )

    async def query_events(self, filter: QueryFilter) -> List[MetricEvent]:
        """Query events with filtering."""
        table_name = self._get_table_name()

        # Build WHERE clause
        where_clauses = []
        params = {}

        if filter.session_id:
            where_clauses.append("session_id = %(session_id)s")
            params["session_id"] = filter.session_id

        if filter.trace_id:
            where_clauses.append("trace_id = %(trace_id)s")
            params["trace_id"] = filter.trace_id

        if filter.operation_name:
            where_clauses.append("operation_name = %(operation_name)s")
            params["operation_name"] = filter.operation_name

        if filter.operation_type:
            where_clauses.append("operation_type = %(operation_type)s")
            params["operation_type"] = filter.operation_type

        if filter.start_time:
            where_clauses.append("timestamp >= %(start_time)s")
            params["start_time"] = filter.start_time

        if filter.end_time:
            where_clauses.append("timestamp <= %(end_time)s")
            params["end_time"] = filter.end_time

        if filter.error_only:
            where_clauses.append("error IS NOT NULL")

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            SELECT * FROM {table_name}
            {where_sql}
            ORDER BY timestamp DESC
            LIMIT %(limit)s OFFSET %(offset)s
        """
        params["limit"] = filter.limit
        params["offset"] = filter.offset

        # Execute query in thread pool
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self.client.execute,
            query,
            params,
            True  # with_column_types
        )

        if not result:
            return []

        rows, columns_with_types = result
        columns = [col[0] for col in columns_with_types]

        # Convert to events
        events = [self._record_to_event(row, columns) for row in rows]
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
        params = {}

        if start_time:
            where_clauses.append("timestamp >= %(start_time)s")
            params["start_time"] = start_time

        if end_time:
            where_clauses.append("timestamp <= %(end_time)s")
            params["end_time"] = end_time

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            SELECT
                session_id,
                min(timestamp) as start_time,
                max(timestamp) as end_time,
                count() as event_count,
                uniq(operation_name) as operation_count,
                countIf(error IS NOT NULL) as error_count,
                sum(duration_ms) as total_duration_ms
            FROM {table_name}
            {where_sql}
            GROUP BY session_id
            ORDER BY start_time DESC
            LIMIT %(limit)s
        """
        params["limit"] = limit

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self.client.execute,
            query,
            params,
            True
        )

        if not result:
            return []

        rows, columns_with_types = result
        columns = [col[0] for col in columns_with_types]

        return [dict(zip(columns, row)) for row in rows]

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
                min(timestamp) as start_time,
                max(timestamp) as end_time,
                count() as event_count,
                uniq(span_id) as span_count,
                countIf(error IS NOT NULL) as error_count,
                sum(duration_ms) as total_duration_ms
            FROM {table_name}
            WHERE session_id = %(session_id)s
            GROUP BY trace_id
            ORDER BY start_time DESC
            LIMIT %(limit)s
        """

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self.client.execute,
            query,
            {"session_id": session_id, "limit": limit},
            True
        )

        if not result:
            return []

        rows, columns_with_types = result
        columns = [col[0] for col in columns_with_types]

        return [dict(zip(columns, row)) for row in rows]

    async def aggregate_by_operation(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics by operation name."""
        table_name = self._get_table_name()

        where_clauses = []
        params = {}

        if start_time:
            where_clauses.append("timestamp >= %(start_time)s")
            params["start_time"] = start_time

        if end_time:
            where_clauses.append("timestamp <= %(end_time)s")
            params["end_time"] = end_time

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        # ClickHouse has native percentile functions (quantile)
        query = f"""
            SELECT
                operation_name,
                count() as count,
                countIf(error IS NOT NULL) as error_count,
                avg(duration_ms) as avg_duration_ms,
                quantile(0.5)(duration_ms) as p50_duration_ms,
                quantile(0.95)(duration_ms) as p95_duration_ms,
                quantile(0.99)(duration_ms) as p99_duration_ms,
                sum(text_char_count) as total_characters,
                sum(JSONExtractFloat(custom_attributes, 'estimated_cost_usd')) as estimated_cost_usd
            FROM {table_name}
            {where_sql}
            GROUP BY operation_name
            ORDER BY count DESC
        """

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self.client.execute,
            query,
            params,
            True
        )

        if not result:
            return []

        rows, columns_with_types = result
        columns = [col[0] for col in columns_with_types]

        return [dict(zip(columns, row)) for row in rows]

    async def aggregate_by_model(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics by model."""
        table_name = self._get_table_name()

        where_clauses = []
        params = {}

        if start_time:
            where_clauses.append("timestamp >= %(start_time)s")
            params["start_time"] = start_time

        if end_time:
            where_clauses.append("timestamp <= %(end_time)s")
            params["end_time"] = end_time

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            SELECT
                ifNull(JSONExtractString(custom_attributes, 'model'), 'unknown') as model,
                count() as count,
                countIf(error IS NOT NULL) as error_count,
                avg(duration_ms) as avg_duration_ms,
                sum(text_char_count) as total_characters,
                sum(JSONExtractFloat(custom_attributes, 'estimated_cost_usd')) as estimated_cost_usd
            FROM {table_name}
            {where_sql}
            GROUP BY model
            ORDER BY count DESC
        """

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self.client.execute,
            query,
            params,
            True
        )

        if not result:
            return []

        rows, columns_with_types = result
        columns = [col[0] for col in columns_with_types]

        return [dict(zip(columns, row)) for row in rows]

    async def aggregate_costs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        group_by: str = "model"
    ) -> List[Dict[str, Any]]:
        """Aggregate cost metrics."""
        table_name = self._get_table_name()

        where_clauses = []
        params = {}

        if start_time:
            where_clauses.append("timestamp >= %(start_time)s")
            params["start_time"] = start_time

        if end_time:
            where_clauses.append("timestamp <= %(end_time)s")
            params["end_time"] = end_time

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        # Determine grouping field
        if group_by == "model":
            group_field = "ifNull(JSONExtractString(custom_attributes, 'model'), 'unknown')"
        elif group_by == "operation":
            group_field = "operation_name"
        elif group_by == "session":
            group_field = "session_id"
        elif group_by == "day":
            group_field = "toDate(timestamp)"
        else:
            raise ValueError(f"Invalid group_by: {group_by}")

        query = f"""
            SELECT
                {group_field} as `{group_by}`,
                sum(JSONExtractFloat(custom_attributes, 'estimated_cost_usd')) as total_cost_usd,
                count() as operation_count,
                total_cost_usd / operation_count as avg_cost_per_operation_usd
            FROM {table_name}
            {where_sql}
            GROUP BY `{group_by}`
            ORDER BY total_cost_usd DESC
        """

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self.client.execute,
            query,
            params,
            True
        )

        if not result:
            return []

        rows, columns_with_types = result
        columns = [col[0] for col in columns_with_types]

        return [dict(zip(columns, row)) for row in rows]

    async def get_summary_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get overall summary statistics."""
        table_name = self._get_table_name()

        where_clauses = []
        params = {}

        if start_time:
            where_clauses.append("timestamp >= %(start_time)s")
            params["start_time"] = start_time

        if end_time:
            where_clauses.append("timestamp <= %(end_time)s")
            params["end_time"] = end_time

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            SELECT
                count() as total_events,
                uniq(session_id) as total_sessions,
                uniq(operation_name) as total_operations,
                countIf(error IS NOT NULL) as total_errors,
                total_errors / total_events as error_rate,
                avg(duration_ms) as avg_duration_ms,
                sum(text_char_count) as total_characters,
                sum(JSONExtractFloat(custom_attributes, 'estimated_cost_usd')) as total_cost_usd
            FROM {table_name}
            {where_sql}
        """

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self.client.execute,
            query,
            params,
            True
        )

        if not result or not result[0]:
            return {}

        rows, columns_with_types = result
        columns = [col[0] for col in columns_with_types]

        return dict(zip(columns, rows[0]))

    async def health_check(self) -> bool:
        """Check if the query backend is healthy."""
        try:
            if not self.client:
                return False

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.execute,
                "SELECT 1"
            )
            return result == [(1,)]
        except Exception as e:
            logger.error(f"ClickHouse query health check failed: {e}")
            return False
