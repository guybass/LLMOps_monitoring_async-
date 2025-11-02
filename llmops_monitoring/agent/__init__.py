"""
Multi-Agent Workflow Intelligence Layer.

Provides visibility into agent interactions, handoffs, and coordination.

Key Features:
- Automatic agent detection from traces
- Handoff quality scoring
- Context drift detection
- Coordination graph visualization
- Bottleneck identification
- Coalition analytics

Example:
    from llmops_monitoring.agent import (
        AgentDetector,
        HandoffAnalyzer,
        ContextDriftDetector,
        CoordinationGraphBuilder
    )

    # Detect agents from traces
    detector = AgentDetector()
    agents = await detector.detect_agents(session_id="session_123")

    # Analyze handoffs
    analyzer = HandoffAnalyzer()
    handoffs = await analyzer.analyze_handoffs(session_id="session_123")

    # Build coordination graph
    builder = CoordinationGraphBuilder()
    graph = await builder.build_graph(session_id="session_123")
"""

from llmops_monitoring.agent.base import (
    Agent,
    AgentOperation,
    AgentHandoff,
    CoordinationGraph,
    HandoffQuality,
    AgentType,
    BottleneckInfo,
    Coalition
)
from llmops_monitoring.agent.detector import AgentDetector
from llmops_monitoring.agent.handoff import HandoffAnalyzer
from llmops_monitoring.agent.context_drift import ContextDriftDetector, DriftAnalysis
from llmops_monitoring.agent.graph import CoordinationGraphBuilder
from llmops_monitoring.agent.bottleneck import BottleneckDetector
from llmops_monitoring.agent.analyzer import CoalitionAnalyzer
from llmops_monitoring.agent.service import AgentIntelligenceService


__all__ = [
    # Data models
    "Agent",
    "AgentOperation",
    "AgentHandoff",
    "CoordinationGraph",
    "HandoffQuality",
    "AgentType",
    "BottleneckInfo",
    "Coalition",
    "DriftAnalysis",

    # Analyzers
    "AgentDetector",
    "HandoffAnalyzer",
    "ContextDriftDetector",
    "CoordinationGraphBuilder",
    "BottleneckDetector",
    "CoalitionAnalyzer",

    # Services
    "AgentIntelligenceService",
]
