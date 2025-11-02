"""
Agent Detection from Traces.

Automatically identifies agents from operation traces and builds agent profiles.
"""

import hashlib
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4
from collections import defaultdict

from llmops_monitoring.agent.base import Agent, AgentOperation, AgentType
from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class AgentDetector:
    """
    Detects agents from operation traces.

    Uses operation names and parent-child relationships to identify
    distinct agents and their interactions.
    """

    def __init__(self):
        """Initialize detector."""
        self.agent_registry: Dict[str, Agent] = {}  # agent_name -> Agent
        self.operation_to_agent: Dict[str, str] = {}  # operation_name -> agent_name

    async def detect_agents(
        self,
        events: List[MetricEvent],
        auto_register: bool = True
    ) -> List[Agent]:
        """
        Detect agents from a list of events.

        Args:
            events: List of metric events
            auto_register: Automatically register detected agents

        Returns:
            List of detected agents
        """
        # Build operation -> agent mapping
        for event in events:
            agent_name = self._extract_agent_name(event)

            if agent_name and auto_register:
                if agent_name not in self.agent_registry:
                    # Create new agent
                    agent = self._create_agent_from_event(event, agent_name)
                    self.agent_registry[agent_name] = agent
                else:
                    # Update existing agent stats
                    self._update_agent_stats(agent_name, event)

                # Track operation -> agent mapping
                self.operation_to_agent[event.operation_name] = agent_name

        return list(self.agent_registry.values())

    def _extract_agent_name(self, event: MetricEvent) -> Optional[str]:
        """
        Extract agent name from event.

        Strategies:
        1. Check custom_attributes for 'agent_name'
        2. Use operation_name as agent name
        3. Detect from operation_type patterns

        Args:
            event: Metric event

        Returns:
            Agent name or None
        """
        # Strategy 1: Explicit agent_name in custom_attributes
        if event.custom_attributes and 'agent_name' in event.custom_attributes:
            return event.custom_attributes['agent_name']

        # Strategy 2: Use operation_name (most common)
        # This assumes operation_name represents the agent
        return event.operation_name

    def _create_agent_from_event(
        self,
        event: MetricEvent,
        agent_name: str
    ) -> Agent:
        """
        Create agent from event.

        Args:
            event: First event from this agent
            agent_name: Name for the agent

        Returns:
            New Agent instance
        """
        # Infer agent role from operation_type or name
        agent_role = self._infer_agent_role(event)
        agent_type = self._infer_agent_type(event)

        # Create agent
        agent = Agent(
            agent_id=uuid4(),
            agent_name=agent_name,
            agent_role=agent_role,
            agent_type=agent_type,
            total_invocations=1,
            success_count=1 if not event.error else 0,
            failure_count=1 if event.error else 0,
            avg_latency_ms=event.duration_ms or 0.0,
            avg_cost_usd=0.0  # Will be updated if cost data available
        )

        logger.debug(f"Created agent: {agent_name} (role={agent_role}, type={agent_type})")

        return agent

    def _update_agent_stats(self, agent_name: str, event: MetricEvent):
        """
        Update agent statistics from new event.

        Args:
            agent_name: Agent name
            event: New event from this agent
        """
        agent = self.agent_registry[agent_name]

        # Update invocation count
        agent.total_invocations += 1

        # Update success/failure
        if event.error:
            agent.failure_count += 1
        else:
            agent.success_count += 1

        # Update average latency (running average)
        if event.duration_ms:
            total_latency = agent.avg_latency_ms * (agent.total_invocations - 1)
            agent.avg_latency_ms = (total_latency + event.duration_ms) / agent.total_invocations

        # Update last seen
        agent.last_seen = event.timestamp

    def _infer_agent_role(self, event: MetricEvent) -> Optional[str]:
        """
        Infer agent role from event.

        Uses heuristics based on operation name and type.

        Args:
            event: Metric event

        Returns:
            Inferred role or None
        """
        name_lower = event.operation_name.lower()

        # Common role patterns
        if 'classif' in name_lower:
            return "classifier"
        elif 'retriev' in name_lower or 'search' in name_lower:
            return "retriever"
        elif 'generat' in name_lower or 'create' in name_lower:
            return "generator"
        elif 'orchestrat' in name_lower or 'coordinat' in name_lower:
            return "coordinator"
        elif 'validat' in name_lower or 'check' in name_lower:
            return "validator"
        elif 'respons' in name_lower or 'reply' in name_lower:
            return "responder"

        return event.operation_type

    def _infer_agent_type(self, event: MetricEvent) -> AgentType:
        """
        Infer agent type from event.

        Args:
            event: Metric event

        Returns:
            Inferred agent type
        """
        name_lower = event.operation_name.lower()

        # Coordinator patterns
        if 'orchestrat' in name_lower or 'coordinat' in name_lower or 'workflow' in name_lower:
            return AgentType.COORDINATOR

        # Specialist patterns
        if any(word in name_lower for word in ['specialist', 'expert', 'specific']):
            return AgentType.SPECIALIST

        # Fallback patterns
        if 'fallback' in name_lower or 'error' in name_lower or 'default' in name_lower:
            return AgentType.FALLBACK

        # Default to specialist for most agents
        return AgentType.SPECIALIST

    async def detect_agent_operations(
        self,
        events: List[MetricEvent]
    ) -> List[AgentOperation]:
        """
        Convert events to agent operations.

        Args:
            events: List of metric events

        Returns:
            List of agent operations
        """
        operations = []

        # First, ensure agents are detected
        await self.detect_agents(events)

        # Build parent lookup for handoff detection
        event_lookup = {event.span_id: event for event in events}

        for event in events:
            agent_name = self.operation_to_agent.get(event.operation_name)
            if not agent_name:
                continue

            agent = self.agent_registry[agent_name]

            # Check if this is a handoff (parent is different agent)
            is_handoff = False
            parent_agent_name = None
            parent_agent_id = None

            if event.parent_span_id:
                parent_event = event_lookup.get(event.parent_span_id)
                if parent_event:
                    parent_agent_name = self.operation_to_agent.get(parent_event.operation_name)
                    if parent_agent_name and parent_agent_name != agent_name:
                        is_handoff = True
                        parent_agent_id = self.agent_registry[parent_agent_name].agent_id

            # Create agent operation
            operation = AgentOperation(
                event_id=event.event_id,
                agent_id=agent.agent_id,
                agent_name=agent_name,
                parent_agent_name=parent_agent_name,
                parent_agent_id=parent_agent_id,
                is_handoff=is_handoff,
                duration_ms=event.duration_ms or 0.0,
                success=not bool(event.error),
                error=event.error,
                timestamp=event.timestamp
            )

            operations.append(operation)

        logger.info(f"Detected {len(operations)} agent operations ({sum(1 for op in operations if op.is_handoff)} handoffs)")

        return operations

    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """
        Get agent by name.

        Args:
            agent_name: Agent name

        Returns:
            Agent or None
        """
        return self.agent_registry.get(agent_name)

    def get_all_agents(self) -> List[Agent]:
        """
        Get all detected agents.

        Returns:
            List of all agents
        """
        return list(self.agent_registry.values())

    def build_agent_relationships(self, operations: List[AgentOperation]):
        """
        Build agent relationship graph.

        Updates can_handoff_to for each agent based on observed handoffs.

        Args:
            operations: List of agent operations
        """
        # Track handoffs: from_agent -> to_agent
        handoff_map: Dict[str, Set[str]] = defaultdict(set)

        for operation in operations:
            if operation.is_handoff and operation.parent_agent_name:
                handoff_map[operation.parent_agent_name].add(operation.agent_name)

        # Update agents
        for from_agent, to_agents in handoff_map.items():
            agent = self.get_agent(from_agent)
            if agent:
                agent.can_handoff_to = list(to_agents)
                logger.debug(f"Agent {from_agent} can hand off to: {to_agents}")
