"""
Base abstraction for query backends.

Extension Point: Implement QueryBackend to add querying capabilities to storage systems.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from llmops_monitoring.schema.events import MetricEvent


class AggregationPeriod(str, Enum):
    """Time periods for aggregation."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class QueryFilter:
    """Filter criteria for querying events."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        operation_name: Optional[str] = None,
        operation_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        error_only: bool = False,
        limit: int = 1000,
        offset: int = 0
    ):
        """
        Initialize query filter.

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
        """
        self.session_id = session_id
        self.trace_id = trace_id
        self.operation_name = operation_name
        self.operation_type = operation_type
        self.start_time = start_time
        self.end_time = end_time
        self.error_only = error_only
        self.limit = limit
        self.offset = offset


class QueryBackend(ABC):
    """
    Abstract base class for query backends.

    Query backends provide read access to stored metric events with
    filtering, aggregation, and analytics capabilities.

    Example:
        class MyQueryBackend(QueryBackend):
            async def query_events(self, filter: QueryFilter) -> List[MetricEvent]:
                # Query implementation
                return events
    """

    @abstractmethod
    async def query_events(self, filter: QueryFilter) -> List[MetricEvent]:
        """
        Query events with filtering.

        Args:
            filter: Query filter criteria

        Returns:
            List of matching metric events
        """
        pass

    @abstractmethod
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
            List of session summaries with keys:
                - session_id: str
                - start_time: datetime
                - end_time: datetime
                - event_count: int
                - operation_count: int
                - error_count: int
                - total_duration_ms: float
        """
        pass

    @abstractmethod
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
            List of trace summaries with keys:
                - trace_id: str
                - start_time: datetime
                - end_time: datetime
                - span_count: int
                - error_count: int
                - total_duration_ms: float
        """
        pass

    @abstractmethod
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
            List of operation aggregations with keys:
                - operation_name: str
                - count: int
                - error_count: int
                - avg_duration_ms: float
                - p50_duration_ms: float
                - p95_duration_ms: float
                - p99_duration_ms: float
                - total_characters: int (if available)
                - estimated_cost_usd: float (if available)
        """
        pass

    @abstractmethod
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
            List of model aggregations with keys:
                - model: str
                - count: int
                - error_count: int
                - avg_duration_ms: float
                - total_characters: int (if available)
                - estimated_cost_usd: float (if available)
        """
        pass

    @abstractmethod
    async def aggregate_costs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        group_by: str = "model"  # model, operation, session, day
    ) -> List[Dict[str, Any]]:
        """
        Aggregate cost metrics.

        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            group_by: Field to group by (model, operation, session, day)

        Returns:
            List of cost aggregations with keys varying by group_by:
                - {group_by}: str (the grouping field value)
                - total_cost_usd: float
                - operation_count: int
                - avg_cost_per_operation_usd: float
        """
        pass

    @abstractmethod
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
            Summary statistics with keys:
                - total_events: int
                - total_sessions: int
                - total_operations: int
                - total_errors: int
                - error_rate: float (0-1)
                - avg_duration_ms: float
                - total_characters: int
                - total_cost_usd: float
        """
        pass

    async def health_check(self) -> bool:
        """
        Check if the query backend is healthy.

        Returns:
            True if backend is healthy and can execute queries
        """
        return True
