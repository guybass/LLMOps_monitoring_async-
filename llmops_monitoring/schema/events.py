"""
Event data models for metric collection.

These schemas are versioned to support backward compatibility
as the system evolves ("air conditioning space").
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TextMetrics(BaseModel):
    """Metrics for text-based content.

    Users can extend this via custom_metrics field.
    """
    char_count: Optional[int] = None
    word_count: Optional[int] = None
    byte_size: Optional[int] = None
    line_count: Optional[int] = None
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "char_count": 1500,
                "word_count": 250,
                "byte_size": 1500,
                "line_count": 10,
                "custom_metrics": {"language": "en"}
            }
        }


class ImageMetrics(BaseModel):
    """Metrics for image-based content.

    Users can extend this via custom_metrics field.
    """
    count: Optional[int] = None
    total_pixels: Optional[int] = None
    file_size_bytes: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "count": 2,
                "total_pixels": 2073600,  # 1920x1080
                "file_size_bytes": 524288,
                "format": "png"
            }
        }


class MetricEvent(BaseModel):
    """
    Core event model representing a single monitored operation.

    Extension Point: Add new metric types by adding new fields.
    Schema versioning ensures backward compatibility.
    """
    # Identity
    event_id: UUID = Field(default_factory=uuid4)
    schema_version: str = Field(default="1.0.0")

    # Hierarchical tracking
    session_id: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None

    # Operation metadata
    operation_name: str
    operation_type: str = "llm_call"  # llm_call, embedding, agent_step, etc.
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None

    # Metrics
    text_metrics: Optional[TextMetrics] = None
    image_metrics: Optional[ImageMetrics] = None

    # Extensibility
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)

    # Error tracking
    error: Optional[str] = None
    error_type: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "550e8400-e29b-41d4-a716-446655440000",
                "schema_version": "1.0.0",
                "session_id": "user-session-123",
                "trace_id": "trace-456",
                "span_id": "span-789",
                "parent_span_id": "span-788",
                "operation_name": "generate_response",
                "operation_type": "llm_call",
                "duration_ms": 1250.5,
                "text_metrics": {
                    "char_count": 1500,
                    "word_count": 250
                },
                "custom_attributes": {
                    "model": "gpt-4",
                    "temperature": 0.7
                }
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = self.model_dump()
        data["event_id"] = str(data["event_id"])
        data["timestamp"] = data["timestamp"].isoformat()
        return data
