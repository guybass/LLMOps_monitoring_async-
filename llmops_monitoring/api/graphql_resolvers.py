"""
GraphQL resolvers for monitoring data.

Resolvers connect GraphQL queries to the AggregationService.
"""

import asyncio
from typing import List, Optional, AsyncGenerator
from strawberry.types import Info

from llmops_monitoring.aggregation import AggregationService
from llmops_monitoring.api.graphql_schema import (
    MetricEvent, SessionSummary, TraceSummary,
    OperationMetrics, ModelMetrics, CostMetrics, SummaryStats,
    EventFilter, TimeRangeFilter, CostFilter,
    TextMetrics, ImageMetrics
)
from llmops_monitoring.streaming.broadcaster import EventBroadcaster
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


def get_aggregation_service(info: Info) -> AggregationService:
    """Get AggregationService from GraphQL context."""
    return info.context["aggregation_service"]


def convert_event(event) -> MetricEvent:
    """Convert backend event to GraphQL MetricEvent."""
    text_metrics = None
    if event.text_metrics:
        text_metrics = TextMetrics(
            char_count=event.text_metrics.char_count,
            word_count=event.text_metrics.word_count,
            byte_size=event.text_metrics.byte_size,
            line_count=event.text_metrics.line_count
        )

    image_metrics = None
    if event.image_metrics:
        image_metrics = ImageMetrics(
            count=event.image_metrics.count,
            total_pixels=event.image_metrics.total_pixels,
            file_size_bytes=event.image_metrics.file_size_bytes,
            width=event.image_metrics.width,
            height=event.image_metrics.height,
            format=event.image_metrics.format
        )

    return MetricEvent(
        event_id=str(event.event_id),
        schema_version=event.schema_version,
        session_id=event.session_id,
        trace_id=event.trace_id,
        span_id=event.span_id,
        parent_span_id=event.parent_span_id,
        operation_name=event.operation_name,
        operation_type=event.operation_type,
        timestamp=event.timestamp,
        duration_ms=event.duration_ms,
        error=event.error,
        error_type=event.error_type,
        text_metrics=text_metrics,
        image_metrics=image_metrics
    )


# Query Resolvers

async def resolve_events(info: Info, filter: Optional[EventFilter]) -> List[MetricEvent]:
    """Resolve events query."""
    service = get_aggregation_service(info)

    if filter:
        events = await service.query_events(
            session_id=filter.session_id,
            trace_id=filter.trace_id,
            operation_name=filter.operation_name,
            operation_type=filter.operation_type,
            start_time=filter.start_time,
            end_time=filter.end_time,
            error_only=filter.error_only,
            limit=filter.limit,
            offset=filter.offset
        )
    else:
        events = await service.query_events()

    return [convert_event(e) for e in events]


async def resolve_event(info: Info, event_id: str) -> Optional[MetricEvent]:
    """Resolve single event by ID."""
    service = get_aggregation_service(info)

    # Query with limit 1 and filter by event_id
    # Note: Most backends don't support event_id filtering directly,
    # so we'll fetch recent events and filter in memory
    events = await service.query_events(limit=1000)

    for event in events:
        if str(event.event_id) == event_id:
            return convert_event(event)

    return None


async def resolve_sessions(
    info: Info,
    filter: Optional[TimeRangeFilter],
    limit: int
) -> List[SessionSummary]:
    """Resolve sessions query."""
    service = get_aggregation_service(info)

    start_time = filter.start_time if filter else None
    end_time = filter.end_time if filter else None

    sessions = await service.get_sessions(
        start_time=start_time,
        end_time=end_time,
        limit=limit
    )

    return [
        SessionSummary(
            session_id=s["session_id"],
            start_time=s["start_time"],
            end_time=s["end_time"],
            event_count=s["event_count"],
            operation_count=s["operation_count"],
            error_count=s["error_count"],
            total_duration_ms=s.get("total_duration_ms", 0)
        )
        for s in sessions
    ]


async def resolve_session(info: Info, session_id: str) -> Optional[SessionSummary]:
    """Resolve single session by ID."""
    sessions = await resolve_sessions(info, None, 1000)

    for session in sessions:
        if session.session_id == session_id:
            return session

    return None


async def resolve_traces(
    info: Info,
    session_id: str,
    limit: int
) -> List[TraceSummary]:
    """Resolve traces for a session."""
    service = get_aggregation_service(info)

    traces = await service.get_traces(session_id, limit)

    return [
        TraceSummary(
            trace_id=t["trace_id"],
            start_time=t["start_time"],
            end_time=t["end_time"],
            event_count=t["event_count"],
            span_count=t["span_count"],
            error_count=t["error_count"],
            total_duration_ms=t.get("total_duration_ms", 0)
        )
        for t in traces
    ]


async def resolve_operations(
    info: Info,
    filter: Optional[TimeRangeFilter]
) -> List[OperationMetrics]:
    """Resolve operation metrics."""
    service = get_aggregation_service(info)

    start_time = filter.start_time if filter else None
    end_time = filter.end_time if filter else None

    operations = await service.aggregate_by_operation(start_time, end_time)

    return [
        OperationMetrics(
            operation_name=op["operation_name"],
            count=op["count"],
            error_count=op.get("error_count", 0),
            avg_duration_ms=op.get("avg_duration_ms", 0),
            p50_duration_ms=op.get("p50_duration_ms"),
            p95_duration_ms=op.get("p95_duration_ms"),
            p99_duration_ms=op.get("p99_duration_ms"),
            total_characters=op.get("total_characters", 0),
            estimated_cost_usd=op.get("estimated_cost_usd", 0)
        )
        for op in operations
    ]


async def resolve_models(
    info: Info,
    filter: Optional[TimeRangeFilter]
) -> List[ModelMetrics]:
    """Resolve model metrics."""
    service = get_aggregation_service(info)

    start_time = filter.start_time if filter else None
    end_time = filter.end_time if filter else None

    models = await service.aggregate_by_model(start_time, end_time)

    return [
        ModelMetrics(
            model=m["model"],
            count=m["count"],
            error_count=m.get("error_count", 0),
            avg_duration_ms=m.get("avg_duration_ms", 0),
            total_characters=m.get("total_characters", 0),
            estimated_cost_usd=m.get("estimated_cost_usd", 0)
        )
        for m in models
    ]


async def resolve_costs(
    info: Info,
    filter: Optional[CostFilter]
) -> List[CostMetrics]:
    """Resolve cost metrics."""
    service = get_aggregation_service(info)

    if not filter:
        filter = CostFilter()

    costs = await service.aggregate_costs(
        start_time=filter.start_time,
        end_time=filter.end_time,
        group_by=filter.group_by
    )

    return [
        CostMetrics(
            group_key=c[filter.group_by],
            total_cost_usd=c.get("total_cost_usd", 0),
            operation_count=c.get("operation_count", 0),
            avg_cost_per_operation_usd=c.get("avg_cost_per_operation_usd", 0)
        )
        for c in costs
    ]


async def resolve_summary(
    info: Info,
    filter: Optional[TimeRangeFilter]
) -> SummaryStats:
    """Resolve summary statistics."""
    service = get_aggregation_service(info)

    start_time = filter.start_time if filter else None
    end_time = filter.end_time if filter else None

    stats = await service.get_summary_stats(start_time, end_time)

    return SummaryStats(
        total_events=stats.get("total_events", 0),
        total_sessions=stats.get("total_sessions", 0),
        total_operations=stats.get("total_operations", 0),
        total_errors=stats.get("total_errors", 0),
        error_rate=stats.get("error_rate", 0),
        avg_duration_ms=stats.get("avg_duration_ms", 0),
        total_characters=stats.get("total_characters", 0),
        total_cost_usd=stats.get("total_cost_usd", 0)
    )


# Subscription Resolvers

async def subscribe_events(
    info: Info,
    session_id: Optional[str],
    operation_name: Optional[str],
    error_only: bool
) -> AsyncGenerator[MetricEvent, None]:
    """Subscribe to real-time event stream."""
    broadcaster = EventBroadcaster.get_instance()

    if not broadcaster or not broadcaster.enabled:
        logger.warning("Event broadcaster not enabled for GraphQL subscriptions")
        return

    # Create a queue for this subscription
    queue = asyncio.Queue()

    # Register callback with broadcaster
    async def callback(event):
        # Apply filters
        if session_id and event.session_id != session_id:
            return
        if operation_name and event.operation_name != operation_name:
            return
        if error_only and not event.error:
            return

        await queue.put(event)

    # Note: This is a simplified implementation
    # In production, you'd want to register/unregister callbacks properly
    # with the broadcaster's connection manager

    try:
        while True:
            event = await queue.get()
            yield convert_event(event)
    except asyncio.CancelledError:
        logger.debug("GraphQL subscription cancelled")
