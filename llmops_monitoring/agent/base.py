"""
Base data models for agent intelligence.

Defines schemas for agents, operations, handoffs, and coordination graphs.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class AgentType(Enum):
    """Type of agent in the system."""
    PRIMARY = "primary"          # Main coordinator
    SPECIALIST = "specialist"    # Domain-specific expert
    COORDINATOR = "coordinator"  # Orchestrates other agents
    FALLBACK = "fallback"       # Handles errors/edge cases
    UNKNOWN = "unknown"          # Auto-detected, type unclear


class HandoffQuality(Enum):
    """Quality level of an agent handoff."""
    EXCELLENT = "excellent"      # >0.9 score
    GOOD = "good"               # 0.7-0.9 score
    ACCEPTABLE = "acceptable"    # 0.5-0.7 score
    POOR = "poor"               # 0.3-0.5 score
    FAILED = "failed"           # <0.3 score


class Agent(BaseModel):
    """
    Agent definition.

    Represents an autonomous LLM-powered entity with specific role and capabilities.
    """
    agent_id: UUID = Field(default_factory=uuid4)
    agent_name: str
    agent_role: Optional[str] = None
    agent_type: AgentType = AgentType.UNKNOWN

    # Capabilities
    capabilities: List[str] = Field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    # Performance metrics (aggregated)
    total_invocations: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0

    # Relationships
    can_handoff_to: List[str] = Field(default_factory=list)  # Agent names
    reports_to: Optional[str] = None  # Coordinator agent name

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.total_invocations
        if total == 0:
            return 0.0
        return self.success_count / total

    class Config:
        json_schema_extra = {
            "example": {
                "agent_name": "customer_classifier",
                "agent_role": "classifier",
                "agent_type": "specialist",
                "capabilities": ["intent_classification", "sentiment_analysis"],
                "total_invocations": 1247,
                "success_count": 1198,
                "avg_latency_ms": 245.3,
                "can_handoff_to": ["support_specialist", "billing_agent"]
            }
        }


class AgentOperation(BaseModel):
    """
    Single agent operation.

    Links to underlying MetricEvent for full details.
    """
    operation_id: UUID = Field(default_factory=uuid4)
    event_id: UUID  # Links to MetricEvent

    # Agent identification
    agent_id: UUID
    agent_name: str

    # Parent operation (if this agent was invoked by another)
    parent_operation_id: Optional[UUID] = None
    parent_agent_id: Optional[UUID] = None
    parent_agent_name: Optional[str] = None

    # Handoff tracking
    is_handoff: bool = False
    handoff_reason: Optional[str] = None
    handoff_quality_score: Optional[float] = None

    # Context tracking
    input_context_hash: Optional[str] = None
    output_context_hash: Optional[str] = None
    context_similarity_to_parent: Optional[float] = None

    # Performance
    duration_ms: float
    success: bool
    error: Optional[str] = None

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "agent_name": "support_specialist",
                "parent_agent_name": "customer_classifier",
                "is_handoff": True,
                "handoff_reason": "Customer query classified as technical support",
                "handoff_quality_score": 0.92,
                "duration_ms": 1250.5,
                "success": True
            }
        }


class AgentHandoff(BaseModel):
    """
    Agent-to-agent handoff event.

    Tracks when one agent transfers control to another.
    """
    handoff_id: UUID = Field(default_factory=uuid4)
    session_id: str
    trace_id: str

    # Source and target
    from_agent_id: UUID
    from_agent_name: str
    from_operation_id: UUID

    to_agent_id: UUID
    to_agent_name: str
    to_operation_id: UUID

    # Handoff details
    handoff_timestamp: datetime = Field(default_factory=datetime.utcnow)
    handoff_reason: Optional[str] = None
    handoff_decision_process: Optional[Dict[str, Any]] = None

    # Quality metrics
    quality_score: float  # 0.0 to 1.0
    quality_level: HandoffQuality
    was_correct_agent: Optional[bool] = None
    handoff_efficiency: Optional[float] = None

    # Context passed
    context_size_bytes: Optional[int] = None
    context_summary: Optional[str] = None
    context_retention_score: Optional[float] = None  # How much info preserved

    # Latency
    handoff_latency_ms: float  # Time between operations

    class Config:
        json_schema_extra = {
            "example": {
                "from_agent_name": "customer_classifier",
                "to_agent_name": "support_specialist",
                "handoff_reason": "Technical support query detected",
                "quality_score": 0.87,
                "quality_level": "good",
                "was_correct_agent": True,
                "context_retention_score": 0.92,
                "handoff_latency_ms": 45.2
            }
        }


@dataclass
class GraphNode:
    """Node in coordination graph (represents an agent)."""
    agent_id: UUID
    agent_name: str
    agent_type: AgentType

    # Aggregated metrics
    invocation_count: int = 0
    total_latency_ms: float = 0.0
    success_count: int = 0
    failure_count: int = 0

    # Graph position (for visualization)
    position: Optional[Dict[str, float]] = None

    @property
    def avg_latency_ms(self) -> float:
        if self.invocation_count == 0:
            return 0.0
        return self.total_latency_ms / self.invocation_count

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total


@dataclass
class GraphEdge:
    """Edge in coordination graph (represents handoff between agents)."""
    from_agent_id: UUID
    from_agent_name: str
    to_agent_id: UUID
    to_agent_name: str

    # Aggregated metrics
    handoff_count: int = 0
    total_latency_ms: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    total_quality_score: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.handoff_count == 0:
            return 0.0
        return self.total_latency_ms / self.handoff_count

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    @property
    def avg_quality_score(self) -> float:
        if self.handoff_count == 0:
            return 0.0
        return self.total_quality_score / self.handoff_count


class CoordinationGraph(BaseModel):
    """
    Coordination graph for a session.

    Represents agent interactions as a directed graph.
    """
    graph_id: UUID = Field(default_factory=uuid4)
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Graph structure
    nodes: List[Dict[str, Any]] = Field(default_factory=list)  # GraphNode data
    edges: List[Dict[str, Any]] = Field(default_factory=list)  # GraphEdge data

    # Graph metrics
    total_agents: int = 0
    total_handoffs: int = 0
    max_depth: int = 0  # Longest chain
    bottleneck_agent_id: Optional[UUID] = None
    critical_path_ms: Optional[float] = None

    # Analysis
    avg_handoff_quality: float = 0.0
    avg_context_retention: float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_123",
                "total_agents": 4,
                "total_handoffs": 7,
                "max_depth": 3,
                "avg_handoff_quality": 0.85,
                "avg_context_retention": 0.91
            }
        }


class BottleneckInfo(BaseModel):
    """
    Information about a bottleneck agent.
    """
    agent_id: UUID
    agent_name: str

    # Severity
    severity: str  # "critical", "high", "medium", "low"
    bottleneck_score: float  # 0.0 to 1.0

    # Metrics causing bottleneck
    avg_queue_time_ms: float
    p95_latency_ms: float
    utilization: float  # Percentage of total traffic
    requests_delayed: int

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "agent_name": "database_retriever",
                "severity": "high",
                "bottleneck_score": 0.78,
                "avg_queue_time_ms": 1250.0,
                "p95_latency_ms": 3500.0,
                "utilization": 0.73,
                "recommendations": [
                    "Enable caching for frequently accessed data",
                    "Consider horizontal scaling (deploy 2 more instances)",
                    "Optimize database query performance"
                ]
            }
        }


class Coalition(BaseModel):
    """
    Group of agents that collaborate on specific task types.
    """
    coalition_id: UUID = Field(default_factory=uuid4)
    coalition_name: str
    agent_names: List[str]

    # Task type this coalition handles
    task_type: str

    # Performance metrics
    total_tasks: int = 0
    successful_tasks: int = 0
    avg_total_latency_ms: float = 0.0
    avg_total_cost_usd: float = 0.0

    # Quality
    avg_user_satisfaction: Optional[float] = None
    avg_handoff_efficiency: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    class Config:
        json_schema_extra = {
            "example": {
                "coalition_name": "Customer Support Team",
                "agent_names": ["classifier", "knowledge_base", "responder"],
                "task_type": "customer_support",
                "total_tasks": 1523,
                "successful_tasks": 1447,
                "avg_total_latency_ms": 2150.0,
                "avg_total_cost_usd": 0.08
            }
        }
