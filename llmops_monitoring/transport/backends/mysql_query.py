"""
MySQL query backend for reading and aggregating stored events.

Provides efficient SQL-based querying with aggregations using MySQL-specific syntax.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from llmops_monitoring.schema.config import StorageConfig
from llmops_monitoring.schema.events import MetricEvent, TextMetrics, ImageMetrics
from llmops_monitoring.transport.backends.query_base import QueryBackend, QueryFilter
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class MySQLQueryBackend(QueryBackend):
    """
    Query backend for MySQL.

    Features:
    - Efficient SQL queries with proper indexing
    - Aggregations using native MySQL functions
    - JSON field querying for custom attributes
    - Percentile calculations using ordered aggregates
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize MySQL query backend.

        Args:
            config: Storage configuration with connection_string
        """
        self.config = config
        self.pool = None
        self._aiomysql_available = self._check_aiomysql()

    def _check_aiomysql(self) -> bool:
        """Check if aiomysql is available."""
        try:
            import aiomysql
            return True
        except ImportError:
            return False

    async def initialize(self):
        """Initialize connection pool."""
        if not self._aiomysql_available:
            raise RuntimeError(
                "MySQL query backend requires aiomysql. "
                "Install with: pip install 'llamonitor-async[mysql]'"
            )

        import aiomysql

        # Parse connection string
        conn_params = self._parse_connection_string()

        self.pool = await aiomysql.create_pool(
            minsize=1,
            maxsize=self.config.pool_size,
            **conn_params
        )

    async def close(self):
        """Close connection pool."""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()

    def _get_table_name(self) -> str:
        """Get full table name with database."""
        database = self.config.schema_name or ""
        table_name = self.config.table_name
        return f"`{database}`.`{table_name}`" if database else f"`{table_name}`"

    def _parse_connection_string(self) -> dict:
        """Parse connection string into aiomysql parameters."""
        conn_str = self.config.connection_string

        if isinstance(conn_str, dict):
            return conn_str

        if conn_str.startswith("mysql://"):
            conn_str = conn_str[8:]

            if "@" in conn_str:
                auth, rest = conn_str.split("@", 1)
                user, password = auth.split(":", 1) if ":" in auth else (auth, "")
            else:
                user, password = "root", ""
                rest = conn_str

            if "/" in rest:
                host_port, db = rest.split("/", 1)
            else:
                host_port, db = rest, "monitoring"

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

        return {
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "",
            "db": "monitoring",
            "autocommit": False,
            "charset": "utf8mb4"
        }

    def _record_to_event(self, record: Dict) -> MetricEvent:
        """Convert database record to MetricEvent."""
        # Parse text metrics
        text_metrics = None
        if record.get("text_char_count") is not None:
            text_custom_metrics = record.get("text_custom_metrics")
            text_metrics = TextMetrics(
                char_count=record.get("text_char_count"),
                word_count=record.get("text_word_count"),
                byte_size=record.get("text_byte_size"),
                line_count=record.get("text_line_count"),
                custom_metrics=json.loads(text_custom_metrics) if text_custom_metrics else {}
            )

        # Parse image metrics
        image_metrics = None
        if record.get("image_count") is not None:
            image_custom_metrics = record.get("image_custom_metrics")
            image_metrics = ImageMetrics(
                count=record.get("image_count"),
                total_pixels=record.get("image_total_pixels"),
                file_size_bytes=record.get("image_file_size_bytes"),
                width=record.get("image_width"),
                height=record.get("image_height"),
                format=record.get("image_format"),
                custom_metrics=json.loads(image_custom_metrics) if image_custom_metrics else {}
            )

        # Parse custom attributes
        custom_attrs = record.get("custom_attributes")
        custom_attributes = json.loads(custom_attrs) if custom_attrs else {}

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
        params = []

        if filter.session_id:
            where_clauses.append("session_id = %s")
            params.append(filter.session_id)

        if filter.trace_id:
            where_clauses.append("trace_id = %s")
            params.append(filter.trace_id)

        if filter.operation_name:
            where_clauses.append("operation_name = %s")
            params.append(filter.operation_name)

        if filter.operation_type:
            where_clauses.append("operation_type = %s")
            params.append(filter.operation_type)

        if filter.start_time:
            where_clauses.append("timestamp >= %s")
            params.append(filter.start_time)

        if filter.end_time:
            where_clauses.append("timestamp <= %s")
            params.append(filter.end_time)

        if filter.error_only:
            where_clauses.append("error IS NOT NULL")

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            SELECT * FROM {table_name}
            {where_sql}
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
        """
        params.extend([filter.limit, filter.offset])

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, params)
                records = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

        # Convert to dictionaries
        events = []
        for record in records:
            record_dict = dict(zip(columns, record))
            events.append(self._record_to_event(record_dict))

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

        if start_time:
            where_clauses.append("timestamp >= %s")
            params.append(start_time)

        if end_time:
            where_clauses.append("timestamp <= %s")
            params.append(end_time)

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
            LIMIT %s
        """
        params.append(limit)

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, params)
                records = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

        return [dict(zip(columns, record)) for record in records]

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
            WHERE session_id = %s
            GROUP BY trace_id
            ORDER BY start_time DESC
            LIMIT %s
        """

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, (session_id, limit))
                records = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

        return [dict(zip(columns, record)) for record in records]

    async def aggregate_by_operation(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics by operation name."""
        table_name = self._get_table_name()

        where_clauses = []
        params = []

        if start_time:
            where_clauses.append("timestamp >= %s")
            params.append(start_time)

        if end_time:
            where_clauses.append("timestamp <= %s")
            params.append(end_time)

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        # MySQL doesn't have PERCENTILE_CONT, so we use a subquery approach
        query = f"""
            SELECT
                operation_name,
                COUNT(*) as count,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count,
                AVG(duration_ms) as avg_duration_ms,
                (SELECT duration_ms FROM {table_name} t2
                 WHERE t2.operation_name = t1.operation_name AND duration_ms IS NOT NULL
                 ORDER BY duration_ms LIMIT 1 OFFSET GREATEST(0, FLOOR(COUNT(*) * 0.5))) as p50_duration_ms,
                (SELECT duration_ms FROM {table_name} t2
                 WHERE t2.operation_name = t1.operation_name AND duration_ms IS NOT NULL
                 ORDER BY duration_ms LIMIT 1 OFFSET GREATEST(0, FLOOR(COUNT(*) * 0.95))) as p95_duration_ms,
                (SELECT duration_ms FROM {table_name} t2
                 WHERE t2.operation_name = t1.operation_name AND duration_ms IS NOT NULL
                 ORDER BY duration_ms LIMIT 1 OFFSET GREATEST(0, FLOOR(COUNT(*) * 0.99))) as p99_duration_ms,
                SUM(text_char_count) as total_characters,
                SUM(CAST(JSON_UNQUOTE(JSON_EXTRACT(custom_attributes, '$.estimated_cost_usd')) AS DECIMAL(10,6))) as estimated_cost_usd
            FROM {table_name} t1
            {where_sql}
            GROUP BY operation_name
            ORDER BY count DESC
        """

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, params)
                records = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

        return [dict(zip(columns, record)) for record in records]

    async def aggregate_by_model(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics by model."""
        table_name = self._get_table_name()

        where_clauses = []
        params = []

        if start_time:
            where_clauses.append("timestamp >= %s")
            params.append(start_time)

        if end_time:
            where_clauses.append("timestamp <= %s")
            params.append(end_time)

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            SELECT
                COALESCE(JSON_UNQUOTE(JSON_EXTRACT(custom_attributes, '$.model')), 'unknown') as model,
                COUNT(*) as count,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count,
                AVG(duration_ms) as avg_duration_ms,
                SUM(text_char_count) as total_characters,
                SUM(CAST(JSON_UNQUOTE(JSON_EXTRACT(custom_attributes, '$.estimated_cost_usd')) AS DECIMAL(10,6))) as estimated_cost_usd
            FROM {table_name}
            {where_sql}
            GROUP BY JSON_UNQUOTE(JSON_EXTRACT(custom_attributes, '$.model'))
            ORDER BY count DESC
        """

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, params)
                records = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

        return [dict(zip(columns, record)) for record in records]

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

        if start_time:
            where_clauses.append("timestamp >= %s")
            params.append(start_time)

        if end_time:
            where_clauses.append("timestamp <= %s")
            params.append(end_time)

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        # Determine grouping field
        if group_by == "model":
            group_field = "COALESCE(JSON_UNQUOTE(JSON_EXTRACT(custom_attributes, '$.model')), 'unknown')"
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
                {group_field} as `{group_by}`,
                SUM(CAST(JSON_UNQUOTE(JSON_EXTRACT(custom_attributes, '$.estimated_cost_usd')) AS DECIMAL(10,6))) as total_cost_usd,
                COUNT(*) as operation_count,
                SUM(CAST(JSON_UNQUOTE(JSON_EXTRACT(custom_attributes, '$.estimated_cost_usd')) AS DECIMAL(10,6))) / COUNT(*) as avg_cost_per_operation_usd
            FROM {table_name}
            {where_sql}
            GROUP BY {group_field}
            ORDER BY total_cost_usd DESC
        """

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, params)
                records = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

        return [dict(zip(columns, record)) for record in records]

    async def get_summary_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get overall summary statistics."""
        table_name = self._get_table_name()

        where_clauses = []
        params = []

        if start_time:
            where_clauses.append("timestamp >= %s")
            params.append(start_time)

        if end_time:
            where_clauses.append("timestamp <= %s")
            params.append(end_time)

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
                SUM(CAST(JSON_UNQUOTE(JSON_EXTRACT(custom_attributes, '$.estimated_cost_usd')) AS DECIMAL(10,6))) as total_cost_usd
            FROM {table_name}
            {where_sql}
        """

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, params)
                record = await cursor.fetchone()
                columns = [desc[0] for desc in cursor.description]

        return dict(zip(columns, record)) if record else {}

    async def health_check(self) -> bool:
        """Check if the query backend is healthy."""
        try:
            if not self.pool:
                return False

            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    await cursor.fetchone()
            return True
        except Exception as e:
            logger.error(f"MySQL query health check failed: {e}")
            return False
