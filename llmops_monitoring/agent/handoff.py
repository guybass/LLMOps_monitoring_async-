"""
Handoff Quality Analysis.

Evaluates the quality of agent-to-agent handoffs.
"""

from typing import Dict, List, Optional, Tuple
from uuid import UUID
from collections import defaultdict

from llmops_monitoring.agent.base import (
    AgentHandoff,
    AgentOperation,
    HandoffQuality
)
from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class HandoffAnalyzer:
    """
    Analyzes handoff quality between agents.

    Evaluates:
    - Correctness: Was the target agent appropriate?
    - Efficiency: Did it resolve faster than alternatives?
    - Context Transfer: Was information preserved?
    """

    def __init__(self):
        """Initialize analyzer."""
        self.handoff_history: List[AgentHandoff] = []

    async def analyze_handoffs(
        self,
        operations: List[AgentOperation],
        events: List[MetricEvent]
    ) -> List[AgentHandoff]:
        """
        Analyze handoffs from agent operations.

        Args:
            operations: List of agent operations
            events: Corresponding metric events

        Returns:
            List of analyzed handoffs
        """
        handoffs = []

        # Build event lookup
        event_lookup = {str(event.event_id): event for event in events}

        # Find all handoff operations
        for operation in operations:
            if not operation.is_handoff or not operation.parent_agent_name:
                continue

            # Get corresponding events
            current_event = event_lookup.get(str(operation.event_id))
            if not current_event:
                continue

            # Find parent operation
            parent_operation = self._find_parent_operation(operation, operations)
            if not parent_operation:
                continue

            parent_event = event_lookup.get(str(parent_operation.event_id))
            if not parent_event:
                continue

            # Create handoff record
            handoff = self._create_handoff(
                parent_operation, parent_event,
                operation, current_event
            )

            handoffs.append(handoff)

        self.handoff_history.extend(handoffs)
        logger.info(f"Analyzed {len(handoffs)} handoffs")

        return handoffs

    def _find_parent_operation(
        self,
        operation: AgentOperation,
        all_operations: List[AgentOperation]
    ) -> Optional[AgentOperation]:
        """
        Find parent operation.

        Args:
            operation: Current operation
            all_operations: All operations

        Returns:
            Parent operation or None
        """
        if not operation.parent_agent_id:
            return None

        for op in all_operations:
            if op.agent_id == operation.parent_agent_id:
                # Find the specific operation that spawned this one
                # For now, use the most recent one before this operation
                if op.timestamp < operation.timestamp:
                    return op

        return None

    def _create_handoff(
        self,
        parent_operation: AgentOperation,
        parent_event: MetricEvent,
        child_operation: AgentOperation,
        child_event: MetricEvent
    ) -> AgentHandoff:
        """
        Create handoff record from operations.

        Args:
            parent_operation: Parent agent operation
            parent_event: Parent event
            child_operation: Child agent operation
            child_event: Child event

        Returns:
            AgentHandoff record
        """
        # Calculate handoff latency (time between operations)
        time_delta = child_operation.timestamp - parent_operation.timestamp
        handoff_latency_ms = time_delta.total_seconds() * 1000

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            parent_operation, parent_event,
            child_operation, child_event
        )

        # Determine quality level
        quality_level = self._classify_quality(quality_score)

        # Evaluate correctness (did the child operation succeed?)
        was_correct = child_operation.success

        # Extract handoff reason from context if available
        handoff_reason = child_operation.handoff_reason or \
                        self._infer_handoff_reason(parent_operation, child_operation)

        # Create handoff
        handoff = AgentHandoff(
            session_id=parent_event.session_id,
            trace_id=parent_event.trace_id,
            from_agent_id=parent_operation.agent_id,
            from_agent_name=parent_operation.agent_name,
            from_operation_id=parent_operation.operation_id,
            to_agent_id=child_operation.agent_id,
            to_agent_name=child_operation.agent_name,
            to_operation_id=child_operation.operation_id,
            handoff_timestamp=child_operation.timestamp,
            handoff_reason=handoff_reason,
            quality_score=quality_score,
            quality_level=quality_level,
            was_correct_agent=was_correct,
            handoff_latency_ms=handoff_latency_ms
        )

        return handoff

    def _calculate_quality_score(
        self,
        parent_operation: AgentOperation,
        parent_event: MetricEvent,
        child_operation: AgentOperation,
        child_event: MetricEvent
    ) -> float:
        """
        Calculate handoff quality score (0.0 to 1.0).

        Components:
        - Correctness (40%): Did child operation succeed?
        - Efficiency (30%): Was latency reasonable?
        - Success chain (30%): Did subsequent operations succeed?

        Args:
            parent_operation: Parent operation
            parent_event: Parent event
            child_operation: Child operation
            child_event: Child event

        Returns:
            Quality score (0.0 to 1.0)
        """
        score_components = []

        # Correctness: Did child succeed?
        correctness_score = 1.0 if child_operation.success else 0.0
        score_components.append(("correctness", correctness_score, 0.4))

        # Efficiency: Was handoff fast?
        # Good handoff should be < 100ms, excellent < 50ms
        time_delta = child_operation.timestamp - parent_operation.timestamp
        handoff_latency_ms = time_delta.total_seconds() * 1000

        if handoff_latency_ms < 50:
            efficiency_score = 1.0
        elif handoff_latency_ms < 100:
            efficiency_score = 0.8
        elif handoff_latency_ms < 500:
            efficiency_score = 0.6
        elif handoff_latency_ms < 1000:
            efficiency_score = 0.4
        else:
            efficiency_score = 0.2

        score_components.append(("efficiency", efficiency_score, 0.3))

        # Success chain: Overall success
        # If both parent and child succeeded → perfect
        # If parent succeeded but child failed → poor handoff
        if parent_operation.success and child_operation.success:
            chain_score = 1.0
        elif parent_operation.success and not child_operation.success:
            chain_score = 0.3  # Bad handoff choice
        elif not parent_operation.success:
            chain_score = 0.5  # Parent failed, hard to evaluate
        else:
            chain_score = 0.5

        score_components.append(("success_chain", chain_score, 0.3))

        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)

        logger.debug(f"Handoff quality: {parent_operation.agent_name} → {child_operation.agent_name} = {total_score:.2f}")
        for component, score, weight in score_components:
            logger.debug(f"  {component}: {score:.2f} (weight: {weight})")

        return total_score

    def _classify_quality(self, score: float) -> HandoffQuality:
        """
        Classify quality score into level.

        Args:
            score: Quality score (0.0 to 1.0)

        Returns:
            HandoffQuality level
        """
        if score >= 0.9:
            return HandoffQuality.EXCELLENT
        elif score >= 0.7:
            return HandoffQuality.GOOD
        elif score >= 0.5:
            return HandoffQuality.ACCEPTABLE
        elif score >= 0.3:
            return HandoffQuality.POOR
        else:
            return HandoffQuality.FAILED

    def _infer_handoff_reason(
        self,
        parent_operation: AgentOperation,
        child_operation: AgentOperation
    ) -> str:
        """
        Infer handoff reason from agent names.

        Args:
            parent_operation: Parent operation
            child_operation: Child operation

        Returns:
            Inferred reason
        """
        return f"{parent_operation.agent_name} delegated to {child_operation.agent_name}"

    def calculate_handoff_statistics(
        self,
        from_agent: Optional[str] = None,
        to_agent: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Calculate handoff statistics.

        Args:
            from_agent: Filter by source agent (optional)
            to_agent: Filter by target agent (optional)

        Returns:
            Dictionary of statistics
        """
        # Filter handoffs
        handoffs = self.handoff_history

        if from_agent:
            handoffs = [h for h in handoffs if h.from_agent_name == from_agent]

        if to_agent:
            handoffs = [h for h in handoffs if h.to_agent_name == to_agent]

        if not handoffs:
            return {
                "total_handoffs": 0,
                "avg_quality_score": 0.0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0
            }

        # Calculate statistics
        total_handoffs = len(handoffs)
        avg_quality = sum(h.quality_score for h in handoffs) / total_handoffs
        success_count = sum(1 for h in handoffs if h.was_correct_agent)
        success_rate = success_count / total_handoffs
        avg_latency = sum(h.handoff_latency_ms for h in handoffs) / total_handoffs

        # Quality distribution
        quality_dist = defaultdict(int)
        for handoff in handoffs:
            quality_dist[handoff.quality_level.value] += 1

        return {
            "total_handoffs": total_handoffs,
            "avg_quality_score": avg_quality,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "quality_distribution": dict(quality_dist)
        }

    def get_low_quality_handoffs(
        self,
        threshold: float = 0.5
    ) -> List[AgentHandoff]:
        """
        Get handoffs below quality threshold.

        Args:
            threshold: Quality threshold (default: 0.5)

        Returns:
            List of low-quality handoffs
        """
        return [
            h for h in self.handoff_history
            if h.quality_score < threshold
        ]

    def generate_recommendations(
        self,
        handoffs: Optional[List[AgentHandoff]] = None
    ) -> List[str]:
        """
        Generate improvement recommendations.

        Args:
            handoffs: Handoffs to analyze (default: all)

        Returns:
            List of recommendations
        """
        if handoffs is None:
            handoffs = self.handoff_history

        recommendations = []

        if not handoffs:
            return recommendations

        # Analyze handoff patterns
        handoff_pairs = defaultdict(list)
        for handoff in handoffs:
            pair = (handoff.from_agent_name, handoff.to_agent_name)
            handoff_pairs[pair].append(handoff)

        # Find problematic pairs
        for (from_agent, to_agent), pair_handoffs in handoff_pairs.items():
            avg_quality = sum(h.quality_score for h in pair_handoffs) / len(pair_handoffs)

            if avg_quality < 0.5:
                recommendations.append(
                    f"⚠️ Handoffs from {from_agent} to {to_agent} have low quality "
                    f"(avg: {avg_quality:.2f}). Consider reviewing the handoff logic."
                )

            # Check latency
            avg_latency = sum(h.handoff_latency_ms for h in pair_handoffs) / len(pair_handoffs)
            if avg_latency > 500:
                recommendations.append(
                    f"⏱️ Handoffs from {from_agent} to {to_agent} are slow "
                    f"({avg_latency:.0f}ms). Consider optimizing the transition."
                )

        return recommendations
