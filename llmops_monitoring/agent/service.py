"""
Agent Intelligence Service.

Orchestrates all agent analysis components:
- Agent detection and lifecycle tracking
- Handoff analysis
- Context drift detection
- Coordination graph building
- Bottleneck detection
- Coalition analysis

Runs in the background analyzing agent workflows from monitoring events.
"""

import asyncio
from typing import List, Dict, Optional, Set
from uuid import UUID
from datetime import datetime, timedelta
from collections import defaultdict

from llmops_monitoring.agent.detector import AgentDetector
from llmops_monitoring.agent.handoff import HandoffAnalyzer
from llmops_monitoring.agent.context_drift import ContextDriftDetector
from llmops_monitoring.agent.graph import CoordinationGraphBuilder
from llmops_monitoring.agent.bottleneck import BottleneckDetector
from llmops_monitoring.agent.analyzer import CoalitionAnalyzer
from llmops_monitoring.agent.base import (
    Agent,
    AgentOperation,
    AgentHandoff,
    CoordinationGraph,
    BottleneckInfo,
    Coalition
)
from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class AgentIntelligenceService:
    """
    Background service for agent intelligence analysis.

    Orchestrates multiple analyzers to provide comprehensive insights
    into multi-agent workflows.

    Usage:
        service = AgentIntelligenceService(storage_backend)
        await service.initialize()

        # Process events
        await service.process_event(event)

        # Get insights
        agents = await service.get_active_agents()
        bottlenecks = await service.get_bottlenecks(session_id)
        coalitions = await service.get_coalitions(session_id)
    """

    def __init__(self, storage_backend=None):
        """
        Initialize agent intelligence service.

        Args:
            storage_backend: Storage backend for persistence (optional)
        """
        self.storage = storage_backend

        # Core analyzers
        self.agent_detector = AgentDetector()
        self.handoff_analyzer = HandoffAnalyzer()
        self.context_drift_detector = ContextDriftDetector()
        self.coordination_builder = CoordinationGraphBuilder()
        self.bottleneck_detector = BottleneckDetector()
        self.coalition_analyzer = CoalitionAnalyzer()

        # In-memory state
        self.active_agents: Dict[UUID, Agent] = {}  # agent_id -> Agent
        self.session_agents: Dict[UUID, Set[UUID]] = defaultdict(set)  # session_id -> agent_ids
        self.session_events: Dict[UUID, List[MetricEvent]] = defaultdict(list)  # session_id -> events

        # Analysis cache
        self.coordination_graphs: Dict[UUID, CoordinationGraph] = {}  # session_id -> graph
        self.bottlenecks: Dict[UUID, List[BottleneckInfo]] = {}  # session_id -> bottlenecks
        self.coalitions: Dict[UUID, List[Coalition]] = {}  # session_id -> coalitions

        # Background tasks
        self._analysis_task: Optional[asyncio.Task] = None
        self._running = False

    async def initialize(self) -> None:
        """Initialize service and start background analysis."""
        logger.info("Initializing Agent Intelligence Service...")

        # Start background analysis loop
        self._running = True
        self._analysis_task = asyncio.create_task(self._analysis_loop())

        logger.info("Agent Intelligence Service initialized")

    async def shutdown(self) -> None:
        """Gracefully shutdown service."""
        logger.info("Shutting down Agent Intelligence Service...")

        self._running = False
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass

        logger.info("Agent Intelligence Service shutdown complete")

    async def process_event(self, event: MetricEvent) -> None:
        """
        Process a monitoring event for agent intelligence.

        Args:
            event: Metric event to analyze
        """
        # Add to session events
        self.session_events[event.session_id].append(event)

        # Detect agents
        detected_agent = self.agent_detector.detect_from_event(event)

        if detected_agent:
            # Track agent
            self.active_agents[detected_agent.agent_id] = detected_agent
            self.session_agents[event.session_id].add(detected_agent.agent_id)

            # Save to storage if available
            if self.storage and hasattr(self.storage, 'agent_storage'):
                try:
                    await self.storage.agent_storage.save_agent(detected_agent)
                except Exception as e:
                    logger.error(f"Failed to save agent: {e}")

            logger.debug(f"Detected agent: {detected_agent.name} ({detected_agent.agent_type})")

        # Detect handoffs
        handoffs = self.handoff_analyzer.detect_handoff(event, self.session_events[event.session_id])

        for handoff in handoffs:
            # Save to storage
            if self.storage and hasattr(self.storage, 'agent_storage'):
                try:
                    await self.storage.agent_storage.save_handoff(handoff)
                except Exception as e:
                    logger.error(f"Failed to save handoff: {e}")

            logger.debug(f"Detected handoff: {handoff.from_agent} -> {handoff.to_agent}")

        # Detect context drift
        drift = self.context_drift_detector.detect_drift(event, self.session_events[event.session_id])

        if drift and drift.drift_detected:
            logger.warning(
                f"Context drift detected in session {event.session_id}: "
                f"score={drift.drift_score:.2f}, reason={drift.reason}"
            )

    async def analyze_session(self, session_id: UUID) -> Dict[str, any]:
        """
        Perform comprehensive analysis of a session.

        Args:
            session_id: Session ID to analyze

        Returns:
            Dictionary with all analysis results
        """
        events = self.session_events.get(session_id, [])

        if not events:
            logger.warning(f"No events found for session {session_id}")
            return {}

        logger.info(f"Analyzing session {session_id} with {len(events)} events...")

        analysis = {
            "session_id": str(session_id),
            "event_count": len(events),
            "agents": [],
            "handoffs": [],
            "coordination_graph": None,
            "bottlenecks": [],
            "coalitions": [],
            "context_drift": []
        }

        # Get agents
        agent_ids = self.session_agents.get(session_id, set())
        analysis["agents"] = [
            self._agent_to_dict(self.active_agents[aid])
            for aid in agent_ids
            if aid in self.active_agents
        ]

        # Build coordination graph
        operations = []
        for event in events:
            agent_id = self._extract_agent_id_from_event(event)
            if agent_id:
                operation = AgentOperation(
                    operation_id=event.event_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    operation_type=event.event_type,
                    timestamp=event.timestamp,
                    duration_ms=event.latency_ms,
                    total_tokens=event.total_tokens
                )
                operations.append(operation)

        if operations:
            graph = self.coordination_builder.build_graph(operations)
            self.coordination_graphs[session_id] = graph

            # Save to storage
            if self.storage and hasattr(self.storage, 'agent_storage'):
                try:
                    await self.storage.agent_storage.save_coordination_graph(graph)
                except Exception as e:
                    logger.error(f"Failed to save coordination graph: {e}")

            analysis["coordination_graph"] = {
                "nodes": graph.nodes,
                "edges": len(graph.edges),
                "execution_paths": graph.execution_paths,
                "parallelism_degree": graph.parallelism_degree,
                "longest_path_ms": graph.longest_path_ms
            }

        # Detect bottlenecks
        bottlenecks = self.bottleneck_detector.detect_bottlenecks(operations)
        self.bottlenecks[session_id] = bottlenecks

        for bottleneck in bottlenecks:
            # Save to storage
            if self.storage and hasattr(self.storage, 'agent_storage'):
                try:
                    await self.storage.agent_storage.save_bottleneck(bottleneck)
                except Exception as e:
                    logger.error(f"Failed to save bottleneck: {e}")

            analysis["bottlenecks"].append({
                "agent_id": str(bottleneck.agent_id),
                "severity": bottleneck.severity,
                "avg_duration_ms": bottleneck.avg_duration_ms,
                "p95_duration_ms": bottleneck.p95_duration_ms,
                "recommendation": bottleneck.recommendation
            })

        # Detect coalitions
        coalitions = self.coalition_analyzer.detect_coalitions(operations, events)
        self.coalitions[session_id] = coalitions

        for coalition in coalitions:
            # Save to storage
            if self.storage and hasattr(self.storage, 'agent_storage'):
                try:
                    await self.storage.agent_storage.save_coalition(coalition)
                except Exception as e:
                    logger.error(f"Failed to save coalition: {e}")

            analysis["coalitions"].append({
                "coalition_id": str(coalition.coalition_id),
                "agent_ids": [str(aid) for aid in coalition.agent_ids],
                "coalition_type": coalition.coalition_type,
                "cohesion_score": coalition.cohesion_score,
                "total_interactions": coalition.total_interactions
            })

        # Context drift summary
        drift_events = []
        for i, event in enumerate(events):
            drift = self.context_drift_detector.detect_drift(event, events[:i])
            if drift and drift.drift_detected:
                drift_events.append({
                    "event_index": i,
                    "drift_score": drift.drift_score,
                    "reason": drift.reason,
                    "timestamp": event.timestamp.isoformat()
                })

        analysis["context_drift"] = drift_events

        logger.info(
            f"Session analysis complete: {len(agent_ids)} agents, "
            f"{len(bottlenecks)} bottlenecks, {len(coalitions)} coalitions"
        )

        return analysis

    async def get_active_agents(
        self,
        session_id: Optional[UUID] = None,
        agent_type: Optional[str] = None
    ) -> List[Agent]:
        """
        Get active agents.

        Args:
            session_id: Filter by session (optional)
            agent_type: Filter by type (optional)

        Returns:
            List of agents
        """
        agents = list(self.active_agents.values())

        if session_id:
            agent_ids = self.session_agents.get(session_id, set())
            agents = [a for a in agents if a.agent_id in agent_ids]

        if agent_type:
            agents = [a for a in agents if a.agent_type == agent_type]

        return agents

    async def get_coordination_graph(self, session_id: UUID) -> Optional[CoordinationGraph]:
        """Get coordination graph for session."""
        return self.coordination_graphs.get(session_id)

    async def get_bottlenecks(
        self,
        session_id: UUID,
        min_severity: str = "low"
    ) -> List[BottleneckInfo]:
        """
        Get bottlenecks for session.

        Args:
            session_id: Session ID
            min_severity: Minimum severity ('low', 'medium', 'high')

        Returns:
            List of bottlenecks
        """
        bottlenecks = self.bottlenecks.get(session_id, [])

        severity_order = {"low": 0, "medium": 1, "high": 2}
        min_level = severity_order.get(min_severity, 0)

        return [
            b for b in bottlenecks
            if severity_order.get(b.severity, 0) >= min_level
        ]

    async def get_coalitions(self, session_id: UUID) -> List[Coalition]:
        """Get coalitions for session."""
        return self.coalitions.get(session_id, [])

    async def get_session_summary(self, session_id: UUID) -> Dict[str, any]:
        """
        Get high-level summary of session.

        Args:
            session_id: Session ID

        Returns:
            Summary dictionary
        """
        events = self.session_events.get(session_id, [])
        agent_ids = self.session_agents.get(session_id, set())
        bottlenecks = self.bottlenecks.get(session_id, [])
        coalitions = self.coalitions.get(session_id, [])
        graph = self.coordination_graphs.get(session_id)

        # Calculate totals
        total_tokens = sum(e.total_tokens for e in events)
        total_cost = sum(e.cost_usd for e in events)
        total_latency = sum(e.latency_ms for e in events)

        summary = {
            "session_id": str(session_id),
            "agent_count": len(agent_ids),
            "event_count": len(events),
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "total_latency_ms": total_latency,
            "avg_latency_ms": total_latency / len(events) if events else 0,
            "bottleneck_count": len(bottlenecks),
            "high_severity_bottlenecks": len([b for b in bottlenecks if b.severity == "high"]),
            "coalition_count": len(coalitions),
            "has_coordination_graph": graph is not None,
            "parallelism_degree": graph.parallelism_degree if graph else 0
        }

        return summary

    async def _analysis_loop(self) -> None:
        """Background loop for periodic analysis."""
        logger.info("Starting agent intelligence analysis loop...")

        while self._running:
            try:
                # Wait 30 seconds between analyses
                await asyncio.sleep(30)

                # Analyze all active sessions
                for session_id in list(self.session_events.keys()):
                    try:
                        await self.analyze_session(session_id)
                    except Exception as e:
                        logger.error(f"Error analyzing session {session_id}: {e}")

                # Cleanup old sessions (older than 1 hour)
                await self._cleanup_old_sessions()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")

        logger.info("Agent intelligence analysis loop stopped")

    async def _cleanup_old_sessions(self, max_age_hours: int = 1) -> None:
        """Clean up old session data."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        sessions_to_remove = []

        for session_id, events in self.session_events.items():
            if not events:
                sessions_to_remove.append(session_id)
                continue

            # Check last event timestamp
            last_event = max(events, key=lambda e: e.timestamp)
            if last_event.timestamp < cutoff_time:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            self.session_events.pop(session_id, None)
            self.session_agents.pop(session_id, None)
            self.coordination_graphs.pop(session_id, None)
            self.bottlenecks.pop(session_id, None)
            self.coalitions.pop(session_id, None)

        if sessions_to_remove:
            logger.debug(f"Cleaned up {len(sessions_to_remove)} old sessions")

    def _extract_agent_id_from_event(self, event: MetricEvent) -> Optional[UUID]:
        """Extract agent ID from event metadata."""
        if hasattr(event, 'metadata') and event.metadata:
            if 'agent_id' in event.metadata:
                agent_id = event.metadata['agent_id']
                if isinstance(agent_id, UUID):
                    return agent_id
                try:
                    return UUID(agent_id)
                except:
                    pass

        # Try to detect from event
        agent = self.agent_detector.detect_from_event(event)
        return agent.agent_id if agent else None

    def _agent_to_dict(self, agent: Agent) -> Dict[str, any]:
        """Convert agent to dictionary."""
        return {
            "agent_id": str(agent.agent_id),
            "name": agent.name,
            "agent_type": agent.agent_type,
            "role": agent.role,
            "capabilities": agent.capabilities,
            "status": agent.status,
            "created_at": agent.created_at.isoformat(),
            "last_active": agent.last_active.isoformat() if agent.last_active else None,
            "total_operations": agent.total_operations,
            "total_tokens": agent.total_tokens,
            "total_cost": agent.total_cost
        }
