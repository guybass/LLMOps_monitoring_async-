"""
Aggregation service for querying and analyzing monitoring data.

Provides high-level API for accessing stored events and metrics.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from llmops_monitoring.schema.config import MonitorConfig, StorageConfig
from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.transport.backends.query_base import QueryBackend, QueryFilter
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class AggregationService:
    """
    Service for querying and aggregating monitoring data.

    This service creates the appropriate query backend based on
    configuration and provides convenient methods for data access.

    Example:
        ```python
        from llmops_monitoring import MonitorConfig
        from llmops_monitoring.aggregation import AggregationService

        config = MonitorConfig.for_local_dev()
        service = AggregationService(config)
        await service.initialize()

        # Query events
        events = await service.query_events(session_id="my-session")

        # Get summary stats
        stats = await service.get_summary_stats()

        await service.close()
        ```
    """

    def __init__(self, config: Optional[MonitorConfig] = None):
        """
        Initialize aggregation service.

        Args:
            config: Monitoring configuration (uses defaults if None)
        """
        self.config = config or MonitorConfig.for_local_dev()
        self.query_backend: Optional[QueryBackend] = None

    async def initialize(self) -> None:
        """Initialize the query backend."""
        backend_type = self.config.storage.backend

        if backend_type == "parquet":
            from llmops_monitoring.transport.backends.parquet_query import ParquetQueryBackend
            self.query_backend = ParquetQueryBackend(self.config.storage)

        elif backend_type == "postgres":
            from llmops_monitoring.transport.backends.postgres_query import PostgresQueryBackend
            self.query_backend = PostgresQueryBackend(self.config.storage)
            await self.query_backend.initialize()

        elif backend_type == "mysql":
            from llmops_monitoring.transport.backends.mysql_query import MySQLQueryBackend
            self.query_backend = MySQLQueryBackend(self.config.storage)
            await self.query_backend.initialize()

        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

        logger.info(f"Initialized aggregation service with {backend_type} backend")

    async def close(self) -> None:
        """Close the query backend and cleanup resources."""
        if self.query_backend and hasattr(self.query_backend, 'close'):
            await self.query_backend.close()
        logger.info("Closed aggregation service")

    # Query Methods

    async def query_events(
        self,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        operation_name: Optional[str] = None,
        operation_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        error_only: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> List[MetricEvent]:
        """
        Query events with filtering.

        Args:
            session_id: Filter by session ID
            trace_id: Filter by trace ID
            operation_name: Filter by operation name
            operation_type: Filter by operation type
            start_time: Filter events after this time
            end_time: Filter events before this time
            error_only: Only return events with errors
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching metric events
        """
        filter = QueryFilter(
            session_id=session_id,
            trace_id=trace_id,
            operation_name=operation_name,
            operation_type=operation_type,
            start_time=start_time,
            end_time=end_time,
            error_only=error_only,
            limit=limit,
            offset=offset
        )

        return await self.query_backend.query_events(filter)

    async def get_sessions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get list of sessions with summary stats.

        Args:
            start_time: Filter sessions after this time
            end_time: Filter sessions before this time
            limit: Maximum number of sessions

        Returns:
            List of session summaries
        """
        return await self.query_backend.get_sessions(start_time, end_time, limit)

    async def get_session_details(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific session.

        Args:
            session_id: Session ID to get details for

        Returns:
            Dictionary with session details including traces and events
        """
        # Get session summary
        sessions = await self.get_sessions()
        session = next((s for s in sessions if s["session_id"] == session_id), None)

        if not session:
            return {"error": "Session not found"}

        # Get traces for this session
        traces = await self.query_backend.get_traces(session_id)

        # Get recent events
        events = await self.query_events(session_id=session_id, limit=50)

        return {
            "session": session,
            "traces": traces,
            "events": [
                {
                    "event_id": str(e.event_id),
                    "operation_name": e.operation_name,
                    "operation_type": e.operation_type,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "duration_ms": e.duration_ms,
                    "error": e.error,
                    "error_type": e.error_type,
                }
                for e in events
            ]
        }

    async def get_traces(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get list of traces for a session.

        Args:
            session_id: Session ID to get traces for
            limit: Maximum number of traces

        Returns:
            List of trace summaries
        """
        return await self.query_backend.get_traces(session_id, limit)

    # Aggregation Methods

    async def aggregate_by_operation(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Aggregate metrics by operation name.

        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time

        Returns:
            List of operation aggregations with metrics
        """
        return await self.query_backend.aggregate_by_operation(start_time, end_time)

    async def aggregate_by_model(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Aggregate metrics by model.

        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time

        Returns:
            List of model aggregations with metrics
        """
        return await self.query_backend.aggregate_by_model(start_time, end_time)

    async def aggregate_costs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        group_by: str = "model"
    ) -> List[Dict[str, Any]]:
        """
        Aggregate cost metrics.

        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            group_by: Field to group by (model, operation, session, day)

        Returns:
            List of cost aggregations
        """
        return await self.query_backend.aggregate_costs(start_time, end_time, group_by)

    async def get_summary_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get overall summary statistics.

        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time

        Returns:
            Summary statistics dictionary
        """
        return await self.query_backend.get_summary_stats(start_time, end_time)

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of the aggregation service.

        Returns:
            Health status dictionary
        """
        backend_healthy = await self.query_backend.health_check() if self.query_backend else False

        return {
            "service_healthy": backend_healthy,
            "backend_type": self.config.storage.backend,
            "backend_healthy": backend_healthy
        }


# Convenience function for quick setup
async def create_aggregation_service(config: Optional[MonitorConfig] = None) -> AggregationService:
    """
    Create and initialize an aggregation service.

    Args:
        config: Monitoring configuration (uses defaults if None)

    Returns:
        Initialized AggregationService instance

    Example:
        ```python
        service = await create_aggregation_service(
            MonitorConfig.for_production("postgresql://...")
        )
        stats = await service.get_summary_stats()
        await service.close()
        ```
    """
    service = AggregationService(config)
    await service.initialize()
    return service
