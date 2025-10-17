"""
FastAPI server for monitoring data aggregation and querying.

Provides REST API endpoints for accessing stored events and metrics.
"""

from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from llmops_monitoring.schema.config import MonitorConfig
from llmops_monitoring.aggregation.service import AggregationService
from llmops_monitoring.api.models import (
    QueryEventsRequest,
    EventResponse,
    SessionSummaryResponse,
    SessionDetailsResponse,
    TraceSummaryResponse,
    OperationMetricsResponse,
    ModelMetricsResponse,
    CostAggregationResponse,
    SummaryStatsResponse,
    HealthResponse,
    ErrorResponse,
    TextMetricsResponse,
    ImageMetricsResponse,
)
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


# Global aggregation service instance
_aggregation_service: Optional[AggregationService] = None


def get_aggregation_service() -> AggregationService:
    """Get the global aggregation service instance."""
    if _aggregation_service is None:
        raise RuntimeError("Aggregation service not initialized")
    return _aggregation_service


def create_api_server(config: Optional[MonitorConfig] = None):
    """
    Create and configure FastAPI application.

    Args:
        config: Monitoring configuration (uses defaults if None)

    Returns:
        FastAPI application instance
    """
    try:
        from fastapi import FastAPI, Query, HTTPException, Path, WebSocket, WebSocketDisconnect
        from fastapi.responses import JSONResponse
    except ImportError:
        raise RuntimeError(
            "FastAPI is required for the API server. "
            "Install with: pip install 'llamonitor-async[api]'"
        )

    # Lifespan context manager for startup/shutdown
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifespan (startup/shutdown)."""
        global _aggregation_service

        # Startup
        logger.info("Starting aggregation API server...")
        _aggregation_service = AggregationService(config)
        await _aggregation_service.initialize()
        logger.info("Aggregation service initialized")

        yield

        # Shutdown
        logger.info("Shutting down aggregation API server...")
        await _aggregation_service.close()
        logger.info("Aggregation service closed")

    # Create FastAPI app
    app = FastAPI(
        title="LLMOps Monitoring API",
        description="REST API for querying and aggregating LLM monitoring data",
        version="1.0.0",
        lifespan=lifespan
    )

    # Health Check Endpoint
    @app.get(
        "/api/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Check API health status"
    )
    async def health_check():
        """Check if the API server and backend are healthy."""
        service = get_aggregation_service()
        health = await service.health_check()

        return HealthResponse(
            status="healthy" if health["service_healthy"] else "unhealthy",
            service_healthy=health["service_healthy"],
            backend_type=health["backend_type"],
            backend_healthy=health["backend_healthy"],
            timestamp=datetime.utcnow()
        )

    # Events Endpoints
    @app.get(
        "/api/v1/events",
        response_model=List[EventResponse],
        tags=["Events"],
        summary="Query events with filters"
    )
    async def query_events(
        session_id: Optional[str] = Query(None, description="Filter by session ID"),
        trace_id: Optional[str] = Query(None, description="Filter by trace ID"),
        operation_name: Optional[str] = Query(None, description="Filter by operation name"),
        operation_type: Optional[str] = Query(None, description="Filter by operation type"),
        start_time: Optional[datetime] = Query(None, description="Filter events after this time"),
        end_time: Optional[datetime] = Query(None, description="Filter events before this time"),
        error_only: bool = Query(False, description="Only return events with errors"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
        offset: int = Query(0, ge=0, description="Number of results to skip")
    ):
        """
        Query events with filtering options.

        Returns a list of events matching the specified criteria.
        """
        try:
            service = get_aggregation_service()
            events = await service.query_events(
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

            # Convert events to response model
            return [
                EventResponse(
                    event_id=str(e.event_id),
                    schema_version=e.schema_version,
                    session_id=e.session_id,
                    trace_id=e.trace_id,
                    span_id=e.span_id,
                    parent_span_id=e.parent_span_id,
                    operation_name=e.operation_name,
                    operation_type=e.operation_type,
                    timestamp=e.timestamp,
                    duration_ms=e.duration_ms,
                    error=e.error,
                    error_type=e.error_type,
                    error_message=e.error_message,
                    text_metrics=TextMetricsResponse(**e.text_metrics.model_dump()) if e.text_metrics else None,
                    image_metrics=ImageMetricsResponse(**e.image_metrics.model_dump()) if e.image_metrics else None,
                    custom_attributes=e.custom_attributes or {}
                )
                for e in events
            ]

        except Exception as e:
            logger.error(f"Error querying events: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # Sessions Endpoints
    @app.get(
        "/api/v1/sessions",
        response_model=List[SessionSummaryResponse],
        tags=["Sessions"],
        summary="Get list of sessions"
    )
    async def get_sessions(
        start_time: Optional[datetime] = Query(None, description="Filter sessions after this time"),
        end_time: Optional[datetime] = Query(None, description="Filter sessions before this time"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum number of results")
    ):
        """
        Get list of sessions with summary statistics.

        Returns sessions ordered by start time (most recent first).
        """
        try:
            service = get_aggregation_service()
            sessions = await service.get_sessions(start_time, end_time, limit)

            return [SessionSummaryResponse(**s) for s in sessions]

        except Exception as e:
            logger.error(f"Error getting sessions: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/api/v1/sessions/{session_id}",
        response_model=SessionDetailsResponse,
        tags=["Sessions"],
        summary="Get session details"
    )
    async def get_session_details(
        session_id: str = Path(..., description="Session ID to retrieve")
    ):
        """
        Get detailed information about a specific session.

        Includes session summary, traces, and recent events.
        """
        try:
            service = get_aggregation_service()
            details = await service.get_session_details(session_id)

            if "error" in details:
                raise HTTPException(status_code=404, detail=details["error"])

            return SessionDetailsResponse(
                session=SessionSummaryResponse(**details["session"]),
                traces=[TraceSummaryResponse(**t) for t in details["traces"]],
                events=details["events"]
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting session details: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/api/v1/sessions/{session_id}/traces",
        response_model=List[TraceSummaryResponse],
        tags=["Sessions"],
        summary="Get traces for a session"
    )
    async def get_session_traces(
        session_id: str = Path(..., description="Session ID"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum number of results")
    ):
        """
        Get list of traces for a specific session.

        Returns traces ordered by start time (most recent first).
        """
        try:
            service = get_aggregation_service()
            traces = await service.get_traces(session_id, limit)

            return [TraceSummaryResponse(**t) for t in traces]

        except Exception as e:
            logger.error(f"Error getting traces: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # Metrics Endpoints
    @app.get(
        "/api/v1/metrics/summary",
        response_model=SummaryStatsResponse,
        tags=["Metrics"],
        summary="Get summary statistics"
    )
    async def get_summary_stats(
        start_time: Optional[datetime] = Query(None, description="Filter events after this time"),
        end_time: Optional[datetime] = Query(None, description="Filter events before this time")
    ):
        """
        Get overall summary statistics for the monitoring data.

        Includes total events, sessions, operations, errors, costs, and more.
        """
        try:
            service = get_aggregation_service()
            stats = await service.get_summary_stats(start_time, end_time)

            return SummaryStatsResponse(**stats)

        except Exception as e:
            logger.error(f"Error getting summary stats: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/api/v1/metrics/operations",
        response_model=List[OperationMetricsResponse],
        tags=["Metrics"],
        summary="Get metrics grouped by operation"
    )
    async def get_operation_metrics(
        start_time: Optional[datetime] = Query(None, description="Filter events after this time"),
        end_time: Optional[datetime] = Query(None, description="Filter events before this time")
    ):
        """
        Get metrics aggregated by operation name.

        Includes counts, error rates, latency percentiles, costs, and more.
        """
        try:
            service = get_aggregation_service()
            metrics = await service.aggregate_by_operation(start_time, end_time)

            return [OperationMetricsResponse(**m) for m in metrics]

        except Exception as e:
            logger.error(f"Error getting operation metrics: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/api/v1/metrics/models",
        response_model=List[ModelMetricsResponse],
        tags=["Metrics"],
        summary="Get metrics grouped by model"
    )
    async def get_model_metrics(
        start_time: Optional[datetime] = Query(None, description="Filter events after this time"),
        end_time: Optional[datetime] = Query(None, description="Filter events before this time")
    ):
        """
        Get metrics aggregated by model.

        Includes counts, error rates, latency, costs, and more.
        """
        try:
            service = get_aggregation_service()
            metrics = await service.aggregate_by_model(start_time, end_time)

            return [ModelMetricsResponse(**m) for m in metrics]

        except Exception as e:
            logger.error(f"Error getting model metrics: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/api/v1/metrics/costs",
        response_model=List[CostAggregationResponse],
        tags=["Metrics"],
        summary="Get cost metrics"
    )
    async def get_cost_metrics(
        start_time: Optional[datetime] = Query(None, description="Filter events after this time"),
        end_time: Optional[datetime] = Query(None, description="Filter events before this time"),
        group_by: str = Query("model", description="Group by field (model, operation, session, day)")
    ):
        """
        Get cost metrics aggregated by specified field.

        Supports grouping by model, operation, session, or day.
        """
        try:
            service = get_aggregation_service()
            costs = await service.aggregate_costs(start_time, end_time, group_by)

            # Convert to response format
            return [
                CostAggregationResponse(
                    group_value=str(c[group_by]),
                    total_cost_usd=float(c["total_cost_usd"] or 0),
                    operation_count=int(c["operation_count"]),
                    avg_cost_per_operation_usd=float(c["avg_cost_per_operation_usd"] or 0)
                )
                for c in costs
            ]

        except Exception as e:
            logger.error(f"Error getting cost metrics: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # WebSocket Streaming Endpoints
    @app.websocket("/api/v1/stream")
    async def websocket_stream(websocket: WebSocket):
        """
        WebSocket endpoint for real-time event streaming.

        Connect to this endpoint to receive events in real-time as they're written.
        """
        from llmops_monitoring.streaming.broadcaster import EventBroadcaster

        broadcaster = EventBroadcaster.get_instance()
        connection_manager = broadcaster.get_connection_manager()

        connection_id = None
        try:
            connection_id = await connection_manager.connect(websocket)
            logger.info(f"WebSocket client connected: {connection_id}")

            # Keep connection alive and wait for disconnect
            while True:
                # Receive messages from client (if any)
                try:
                    data = await websocket.receive_text()
                    # Echo back for heartbeat
                    await websocket.send_json({"type": "ping", "echo": data})
                except:
                    break

        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if connection_id:
                await connection_manager.disconnect(connection_id)

    @app.websocket("/api/v1/stream/sessions/{session_id}")
    async def websocket_stream_session(websocket: WebSocket, session_id: str):
        """
        WebSocket endpoint for session-specific event streaming.

        Only events for the specified session will be streamed.
        """
        from llmops_monitoring.streaming.broadcaster import EventBroadcaster

        broadcaster = EventBroadcaster.get_instance()
        connection_manager = broadcaster.get_connection_manager()

        connection_id = None
        try:
            # Connect with session filter
            connection_id = await connection_manager.connect(
                websocket,
                filters={"session_id": session_id}
            )
            logger.info(f"WebSocket client connected to session {session_id}: {connection_id}")

            # Keep connection alive
            while True:
                try:
                    data = await websocket.receive_text()
                    await websocket.send_json({"type": "ping", "echo": data})
                except:
                    break

        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected from session {session_id}: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if connection_id:
                await connection_manager.disconnect(connection_id)

    @app.websocket("/api/v1/stream/operations/{operation_name}")
    async def websocket_stream_operation(websocket: WebSocket, operation_name: str):
        """
        WebSocket endpoint for operation-specific event streaming.

        Only events for the specified operation will be streamed.
        """
        from llmops_monitoring.streaming.broadcaster import EventBroadcaster

        broadcaster = EventBroadcaster.get_instance()
        connection_manager = broadcaster.get_connection_manager()

        connection_id = None
        try:
            # Connect with operation filter
            connection_id = await connection_manager.connect(
                websocket,
                filters={"operation_name": operation_name}
            )
            logger.info(f"WebSocket client connected to operation {operation_name}: {connection_id}")

            # Keep connection alive
            while True:
                try:
                    data = await websocket.receive_text()
                    await websocket.send_json({"type": "ping", "echo": data})
                except:
                    break

        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected from operation {operation_name}: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if connection_id:
                await connection_manager.disconnect(connection_id)

    return app


def run_api_server(
    config: Optional[MonitorConfig] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False
):
    """
    Run the API server with Uvicorn.

    Args:
        config: Monitoring configuration
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development

    Example:
        ```python
        from llmops_monitoring import MonitorConfig
        from llmops_monitoring.api import run_api_server

        config = MonitorConfig.for_local_dev()
        run_api_server(config, port=8080)
        ```
    """
    try:
        import uvicorn
    except ImportError:
        raise RuntimeError(
            "Uvicorn is required to run the API server. "
            "Install with: pip install 'llamonitor-async[api]'"
        )

    app = create_api_server(config)

    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"API documentation available at http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port, reload=reload)
