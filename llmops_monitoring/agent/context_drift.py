"""
Context Drift Detection.

Detects when information is lost as tasks pass through agent chains.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from llmops_monitoring.agent.base import AgentOperation
from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class ContextEntity:
    """Represents an extracted entity from context."""
    entity_type: str  # "name", "number", "date", "identifier", etc.
    value: str
    position: int


@dataclass
class DriftAnalysis:
    """Analysis of context drift between two operations."""
    from_agent: str
    to_agent: str

    # Entities
    input_entities: List[ContextEntity]
    output_entities: List[ContextEntity]

    # Drift metrics
    entities_preserved: int
    entities_lost: int
    entities_added: int
    context_retention_score: float  # 0.0 to 1.0
    context_drift_score: float      # 0.0 to 1.0

    # Lost entities
    lost_entities: List[ContextEntity]

    @property
    def has_high_drift(self) -> bool:
        """Check if drift is concerning (>40% lost)."""
        return self.context_drift_score > 0.4


class ContextDriftDetector:
    """
    Detects context drift in agent handoffs.

    Analyzes how much information is preserved when control passes
    from one agent to another.
    """

    def __init__(self):
        """Initialize detector."""
        self.drift_history: List[DriftAnalysis] = []

    async def detect_drift(
        self,
        operations: List[AgentOperation],
        events: List[MetricEvent]
    ) -> List[DriftAnalysis]:
        """
        Detect context drift in agent chain.

        Args:
            operations: List of agent operations
            events: Corresponding metric events

        Returns:
            List of drift analyses
        """
        analyses = []

        # Build event lookup
        event_lookup = {str(event.event_id): event for event in events}

        # Analyze each handoff
        for i, operation in enumerate(operations):
            if not operation.is_handoff or not operation.parent_agent_name:
                continue

            # Find parent operation
            parent_operation = self._find_parent(operation, operations)
            if not parent_operation:
                continue

            # Get events
            parent_event = event_lookup.get(str(parent_operation.event_id))
            current_event = event_lookup.get(str(operation.event_id))

            if not parent_event or not current_event:
                continue

            # Analyze drift
            analysis = self._analyze_drift_between_operations(
                parent_operation, parent_event,
                operation, current_event
            )

            analyses.append(analysis)

        self.drift_history.extend(analyses)
        logger.info(f"Detected drift in {len(analyses)} handoffs")

        # Log high-drift cases
        high_drift = [a for a in analyses if a.has_high_drift]
        if high_drift:
            logger.warning(f"Found {len(high_drift)} handoffs with high context drift (>40%)")

        return analyses

    def _find_parent(
        self,
        operation: AgentOperation,
        all_operations: List[AgentOperation]
    ) -> Optional[AgentOperation]:
        """Find parent operation."""
        if not operation.parent_agent_id:
            return None

        # Find parent by agent_id and timestamp
        candidates = [
            op for op in all_operations
            if op.agent_id == operation.parent_agent_id
            and op.timestamp < operation.timestamp
        ]

        if candidates:
            # Return most recent parent
            return max(candidates, key=lambda op: op.timestamp)

        return None

    def _analyze_drift_between_operations(
        self,
        parent_operation: AgentOperation,
        parent_event: MetricEvent,
        child_operation: AgentOperation,
        child_event: MetricEvent
    ) -> DriftAnalysis:
        """
        Analyze drift between parent and child.

        Args:
            parent_operation: Parent operation
            parent_event: Parent event
            child_operation: Child operation
            child_event: Child event

        Returns:
            DriftAnalysis
        """
        # Extract text from events (simplified - in production would look at prompts/outputs)
        parent_text = self._extract_text_from_event(parent_event)
        child_text = self._extract_text_from_event(child_event)

        # Extract entities
        parent_entities = self._extract_entities(parent_text)
        child_entities = self._extract_entities(child_text)

        # Calculate preservation
        preserved, lost, added = self._calculate_entity_changes(
            parent_entities, child_entities
        )

        # Calculate scores
        total_input = len(parent_entities)
        if total_input > 0:
            retention_score = preserved / total_input
            drift_score = lost / total_input
        else:
            retention_score = 1.0
            drift_score = 0.0

        # Create analysis
        analysis = DriftAnalysis(
            from_agent=parent_operation.agent_name,
            to_agent=child_operation.agent_name,
            input_entities=parent_entities,
            output_entities=child_entities,
            entities_preserved=preserved,
            entities_lost=lost,
            entities_added=added,
            context_retention_score=retention_score,
            context_drift_score=drift_score,
            lost_entities=self._identify_lost_entities(parent_entities, child_entities)
        )

        if analysis.has_high_drift:
            logger.warning(
                f"High drift detected: {parent_operation.agent_name} → {child_operation.agent_name} "
                f"(retention: {retention_score:.2%}, lost {lost}/{total_input} entities)"
            )

        return analysis

    def _extract_text_from_event(self, event: MetricEvent) -> str:
        """
        Extract text content from event.

        In production, this would extract from prompt, output, etc.
        For now, uses operation_name as proxy.

        Args:
            event: Metric event

        Returns:
            Text content
        """
        text_parts = [event.operation_name]

        # Add custom attributes if available
        if event.custom_attributes:
            for key, value in event.custom_attributes.items():
                if isinstance(value, str):
                    text_parts.append(value)

        return " ".join(text_parts)

    def _extract_entities(self, text: str) -> List[ContextEntity]:
        """
        Extract key entities from text.

        Simple extraction using regex patterns.
        In production, would use NLP (spaCy, etc.)

        Args:
            text: Input text

        Returns:
            List of extracted entities
        """
        entities = []

        # Extract numbers (IDs, quantities, etc.)
        for match in re.finditer(r'\b\d+\b', text):
            entities.append(ContextEntity(
                entity_type="number",
                value=match.group(),
                position=match.start()
            ))

        # Extract emails
        for match in re.finditer(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            entities.append(ContextEntity(
                entity_type="email",
                value=match.group(),
                position=match.start()
            ))

        # Extract phone numbers
        for match in re.finditer(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
            entities.append(ContextEntity(
                entity_type="phone",
                value=match.group(),
                position=match.start()
            ))

        # Extract capitalized words (potential names/places)
        for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
            entities.append(ContextEntity(
                entity_type="proper_noun",
                value=match.group(),
                position=match.start()
            ))

        # Extract quoted strings (important context)
        for match in re.finditer(r'"([^"]*)"', text):
            entities.append(ContextEntity(
                entity_type="quoted_text",
                value=match.group(1),
                position=match.start()
            ))

        return entities

    def _calculate_entity_changes(
        self,
        input_entities: List[ContextEntity],
        output_entities: List[ContextEntity]
    ) -> Tuple[int, int, int]:
        """
        Calculate entity preservation, loss, and addition.

        Args:
            input_entities: Entities from input
            output_entities: Entities from output

        Returns:
            Tuple of (preserved, lost, added)
        """
        # Create sets of entity values for comparison
        input_values = {e.value for e in input_entities}
        output_values = {e.value for e in output_entities}

        # Calculate changes
        preserved = len(input_values & output_values)  # Intersection
        lost = len(input_values - output_values)       # In input but not output
        added = len(output_values - input_values)      # In output but not input

        return preserved, lost, added

    def _identify_lost_entities(
        self,
        input_entities: List[ContextEntity],
        output_entities: List[ContextEntity]
    ) -> List[ContextEntity]:
        """
        Identify which specific entities were lost.

        Args:
            input_entities: Input entities
            output_entities: Output entities

        Returns:
            List of lost entities
        """
        output_values = {e.value for e in output_entities}
        lost = [e for e in input_entities if e.value not in output_values]
        return lost

    def get_high_drift_handoffs(
        self,
        threshold: float = 0.4
    ) -> List[DriftAnalysis]:
        """
        Get handoffs with high drift.

        Args:
            threshold: Drift threshold (default: 0.4 = 40%)

        Returns:
            List of high-drift handoffs
        """
        return [
            analysis for analysis in self.drift_history
            if analysis.context_drift_score > threshold
        ]

    def calculate_drift_statistics(
        self,
        from_agent: Optional[str] = None,
        to_agent: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Calculate drift statistics.

        Args:
            from_agent: Filter by source agent
            to_agent: Filter by target agent

        Returns:
            Dictionary of statistics
        """
        analyses = self.drift_history

        if from_agent:
            analyses = [a for a in analyses if a.from_agent == from_agent]

        if to_agent:
            analyses = [a for a in analyses if a.to_agent == to_agent]

        if not analyses:
            return {
                "total_handoffs": 0,
                "avg_retention": 0.0,
                "avg_drift": 0.0,
                "high_drift_count": 0
            }

        total = len(analyses)
        avg_retention = sum(a.context_retention_score for a in analyses) / total
        avg_drift = sum(a.context_drift_score for a in analyses) / total
        high_drift = sum(1 for a in analyses if a.has_high_drift)

        return {
            "total_handoffs": total,
            "avg_retention": avg_retention,
            "avg_drift": avg_drift,
            "high_drift_count": high_drift,
            "high_drift_percentage": high_drift / total if total > 0 else 0.0
        }

    def generate_recommendations(
        self,
        analyses: Optional[List[DriftAnalysis]] = None
    ) -> List[str]:
        """
        Generate recommendations to reduce drift.

        Args:
            analyses: Drift analyses to process (default: all)

        Returns:
            List of recommendations
        """
        if analyses is None:
            analyses = self.drift_history

        recommendations = []

        # Analyze by agent pair
        from collections import defaultdict
        pair_analyses = defaultdict(list)

        for analysis in analyses:
            pair = (analysis.from_agent, analysis.to_agent)
            pair_analyses[pair].append(analysis)

        # Generate recommendations per pair
        for (from_agent, to_agent), pair_list in pair_analyses.items():
            avg_drift = sum(a.context_drift_score for a in pair_list) / len(pair_list)

            if avg_drift > 0.4:
                # Identify most commonly lost entity types
                lost_types = defaultdict(int)
                for analysis in pair_list:
                    for entity in analysis.lost_entities:
                        lost_types[entity.entity_type] += 1

                top_lost = sorted(lost_types.items(), key=lambda x: x[1], reverse=True)[:3]

                recommendations.append(
                    f"⚠️ High drift in {from_agent} → {to_agent} handoffs ({avg_drift:.1%} average drift). "
                    f"Most commonly lost: {', '.join(t for t, _ in top_lost)}. "
                    "Consider adding explicit context preservation in handoff prompt."
                )

        return recommendations
