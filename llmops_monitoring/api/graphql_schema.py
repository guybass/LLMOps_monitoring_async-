"""
GraphQL schema definition for monitoring data.

Provides flexible querying capabilities using Strawberry GraphQL.
"""

from datetime import datetime
from typing import List, Optional
import strawberry
from strawberry.types import Info


# GraphQL Types

@strawberry.type
class TextMetrics:
    """Text content metrics."""
    char_count: Optional[int] = None
    word_count: Optional[int] = None
    byte_size: Optional[int] = None
    line_count: Optional[int] = None


@strawberry.type
class ImageMetrics:
    """Image content metrics."""
    count: Optional[int] = None
    total_pixels: Optional[int] = None
    file_size_bytes: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None


@strawberry.type
class MetricEvent:
    """Individual metric event."""
    event_id: str
    schema_version: str
    session_id: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str
    operation_type: str
    timestamp: datetime
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    text_metrics: Optional[TextMetrics] = None
    image_metrics: Optional[ImageMetrics] = None


@strawberry.type
class SessionSummary:
    """Summary of a monitoring session."""
    session_id: str
    start_time: datetime
    end_time: datetime
    event_count: int
    operation_count: int
    error_count: int
    total_duration_ms: float


@strawberry.type
class TraceSummary:
    """Summary of a trace within a session."""
    trace_id: str
    start_time: datetime
    end_time: datetime
    event_count: int
    span_count: int
    error_count: int
    total_duration_ms: float


@strawberry.type
class OperationMetrics:
    """Aggregated metrics for an operation."""
    operation_name: str
    count: int
    error_count: int
    avg_duration_ms: float
    p50_duration_ms: Optional[float] = None
    p95_duration_ms: Optional[float] = None
    p99_duration_ms: Optional[float] = None
    total_characters: int
    estimated_cost_usd: float


@strawberry.type
class ModelMetrics:
    """Aggregated metrics for a model."""
    model: str
    count: int
    error_count: int
    avg_duration_ms: float
    total_characters: int
    estimated_cost_usd: float


@strawberry.type
class CostMetrics:
    """Cost analytics metrics."""
    group_key: str  # model, operation, session, or day
    total_cost_usd: float
    operation_count: int
    avg_cost_per_operation_usd: float


@strawberry.type
class SummaryStats:
    """Overall summary statistics."""
    total_events: int
    total_sessions: int
    total_operations: int
    total_errors: int
    error_rate: float
    avg_duration_ms: float
    total_characters: int
    total_cost_usd: float


# Input Types for filtering

@strawberry.input
class EventFilter:
    """Filter criteria for querying events."""
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    operation_name: Optional[str] = None
    operation_type: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_only: bool = False
    limit: int = 100
    offset: int = 0


@strawberry.input
class TimeRangeFilter:
    """Time range filter for aggregations."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@strawberry.input
class CostFilter:
    """Filter for cost analytics."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    group_by: str = "model"  # model, operation, session, day


# Queries

@strawberry.type
class Query:
    """Root query type."""

    @strawberry.field
    async def events(self, info: Info, filter: Optional[EventFilter] = None) -> List[MetricEvent]:
        """Query metric events with optional filtering."""
        from llmops_monitoring.api.graphql_resolvers import resolve_events
        return await resolve_events(info, filter)

    @strawberry.field
    async def event(self, info: Info, event_id: str) -> Optional[MetricEvent]:
        """Get a single event by ID."""
        from llmops_monitoring.api.graphql_resolvers import resolve_event
        return await resolve_event(info, event_id)

    @strawberry.field
    async def sessions(
        self,
        info: Info,
        filter: Optional[TimeRangeFilter] = None,
        limit: int = 100
    ) -> List[SessionSummary]:
        """Get list of sessions with summary stats."""
        from llmops_monitoring.api.graphql_resolvers import resolve_sessions
        return await resolve_sessions(info, filter, limit)

    @strawberry.field
    async def session(self, info: Info, session_id: str) -> Optional[SessionSummary]:
        """Get a single session by ID."""
        from llmops_monitoring.api.graphql_resolvers import resolve_session
        return await resolve_session(info, session_id)

    @strawberry.field
    async def traces(
        self,
        info: Info,
        session_id: str,
        limit: int = 100
    ) -> List[TraceSummary]:
        """Get traces for a specific session."""
        from llmops_monitoring.api.graphql_resolvers import resolve_traces
        return await resolve_traces(info, session_id, limit)

    @strawberry.field
    async def operations(
        self,
        info: Info,
        filter: Optional[TimeRangeFilter] = None
    ) -> List[OperationMetrics]:
        """Get aggregated metrics by operation."""
        from llmops_monitoring.api.graphql_resolvers import resolve_operations
        return await resolve_operations(info, filter)

    @strawberry.field
    async def models(
        self,
        info: Info,
        filter: Optional[TimeRangeFilter] = None
    ) -> List[ModelMetrics]:
        """Get aggregated metrics by model."""
        from llmops_monitoring.api.graphql_resolvers import resolve_models
        return await resolve_models(info, filter)

    @strawberry.field
    async def costs(
        self,
        info: Info,
        filter: Optional[CostFilter] = None
    ) -> List[CostMetrics]:
        """Get cost analytics."""
        from llmops_monitoring.api.graphql_resolvers import resolve_costs
        return await resolve_costs(info, filter)

    @strawberry.field
    async def summary(
        self,
        info: Info,
        filter: Optional[TimeRangeFilter] = None
    ) -> SummaryStats:
        """Get overall summary statistics."""
        from llmops_monitoring.api.graphql_resolvers import resolve_summary
        return await resolve_summary(info, filter)


# Subscriptions for real-time data

@strawberry.type
class Subscription:
    """Root subscription type for real-time updates."""

    @strawberry.subscription
    async def event_stream(
        self,
        info: Info,
        session_id: Optional[str] = None,
        operation_name: Optional[str] = None,
        error_only: bool = False
    ) -> MetricEvent:
        """Subscribe to real-time event stream."""
        from llmops_monitoring.api.graphql_resolvers import subscribe_events
        async for event in subscribe_events(info, session_id, operation_name, error_only):
            yield event


# Root schema

schema = strawberry.Schema(
    query=Query,
    subscription=Subscription
)
