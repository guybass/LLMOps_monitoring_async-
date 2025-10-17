"""
Pydantic models for API requests and responses.

These models define the structure of data exchanged via the REST API.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID


# Request Models

class QueryEventsRequest(BaseModel):
    """Request model for querying events."""
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    operation_name: Optional[str] = None
    operation_type: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_only: bool = False
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


# Response Models

class TextMetricsResponse(BaseModel):
    """Response model for text metrics."""
    char_count: Optional[int] = None
    word_count: Optional[int] = None
    byte_size: Optional[int] = None
    line_count: Optional[int] = None


class ImageMetricsResponse(BaseModel):
    """Response model for image metrics."""
    count: Optional[int] = None
    total_pixels: Optional[int] = None
    file_size_bytes: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None


class EventResponse(BaseModel):
    """Response model for a metric event."""
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
    error: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    text_metrics: Optional[TextMetricsResponse] = None
    image_metrics: Optional[ImageMetricsResponse] = None
    custom_attributes: Dict[str, Any] = {}


class SessionSummaryResponse(BaseModel):
    """Response model for session summary."""
    session_id: str
    start_time: datetime
    end_time: datetime
    event_count: int
    operation_count: int
    error_count: int
    total_duration_ms: Optional[float] = None


class TraceSummaryResponse(BaseModel):
    """Response model for trace summary."""
    trace_id: str
    start_time: datetime
    end_time: datetime
    event_count: int
    span_count: int
    error_count: int
    total_duration_ms: Optional[float] = None


class OperationMetricsResponse(BaseModel):
    """Response model for operation metrics."""
    operation_name: str
    count: int
    error_count: int
    avg_duration_ms: Optional[float] = None
    p50_duration_ms: Optional[float] = None
    p95_duration_ms: Optional[float] = None
    p99_duration_ms: Optional[float] = None
    total_characters: Optional[int] = None
    estimated_cost_usd: Optional[float] = None


class ModelMetricsResponse(BaseModel):
    """Response model for model metrics."""
    model: str
    count: int
    error_count: int
    avg_duration_ms: Optional[float] = None
    total_characters: Optional[int] = None
    estimated_cost_usd: Optional[float] = None


class CostAggregationResponse(BaseModel):
    """Response model for cost aggregation."""
    group_value: str
    total_cost_usd: float
    operation_count: int
    avg_cost_per_operation_usd: float


class SummaryStatsResponse(BaseModel):
    """Response model for summary statistics."""
    total_events: int
    total_sessions: int
    total_operations: int
    total_errors: int
    error_rate: float
    avg_duration_ms: float
    total_characters: int
    total_cost_usd: float


class SessionDetailsResponse(BaseModel):
    """Response model for detailed session information."""
    session: SessionSummaryResponse
    traces: List[TraceSummaryResponse]
    events: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    service_healthy: bool
    backend_type: str
    backend_healthy: bool
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: Optional[str] = None
