"""
Bottleneck Detection and Analysis.

Identifies performance bottlenecks in agent coordination systems.
"""

from typing import Dict, List, Optional, Tuple
from uuid import UUID
from collections import defaultdict
import statistics

from llmops_monitoring.agent.base import (
    Agent,
    AgentOperation,
    AgentHandoff,
    BottleneckInfo,
    GraphNode,
    GraphEdge
)
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class BottleneckDetector:
    """
    Detects and analyzes bottlenecks in agent systems.

    Identifies agents that:
    - Process high volumes of traffic
    - Have high latency
    - Have low success rates
    - Cause downstream delays
    """

    def __init__(self):
        """Initialize detector."""
        self.bottleneck_history: List[BottleneckInfo] = []

    async def detect_bottlenecks(
        self,
        agents: List[Agent],
        operations: List[AgentOperation],
        handoffs: List[AgentHandoff],
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> List[BottleneckInfo]:
        """
        Detect bottlenecks in agent system.

        Args:
            agents: List of agents
            operations: List of agent operations
            handoffs: List of handoffs
            nodes: Graph nodes
            edges: Graph edges

        Returns:
            List of detected bottlenecks
        """
        bottlenecks = []

        # Analyze each agent for bottleneck characteristics
        for agent in agents:
            node = self._find_node_for_agent(agent.agent_id, nodes)
            if not node:
                continue

            # Calculate bottleneck score
            score = self._calculate_bottleneck_score(
                agent, node, edges, operations, handoffs
            )

            if score > 0.5:  # Significant bottleneck
                # Create bottleneck info
                bottleneck = self._create_bottleneck_info(
                    agent, node, edges, operations, handoffs, score
                )
                bottlenecks.append(bottleneck)

        # Sort by severity (score)
        bottlenecks.sort(key=lambda b: b.bottleneck_score, reverse=True)

        self.bottleneck_history.extend(bottlenecks)
        logger.info(f"Detected {len(bottlenecks)} bottlenecks")

        # Log critical bottlenecks
        critical = [b for b in bottlenecks if b.severity == "critical"]
        if critical:
            logger.warning(f"Found {len(critical)} CRITICAL bottlenecks")
            for b in critical:
                logger.warning(f"  - {b.agent_name}: score={b.bottleneck_score:.2f}, p95={b.p95_latency_ms:.0f}ms")

        return bottlenecks

    def _find_node_for_agent(
        self,
        agent_id: UUID,
        nodes: List[GraphNode]
    ) -> Optional[GraphNode]:
        """Find graph node for agent."""
        for node in nodes:
            if node.agent_id == agent_id:
                return node
        return None

    def _calculate_bottleneck_score(
        self,
        agent: Agent,
        node: GraphNode,
        edges: List[GraphEdge],
        operations: List[AgentOperation],
        handoffs: List[AgentHandoff]
    ) -> float:
        """
        Calculate comprehensive bottleneck score.

        Components:
        - Traffic utilization (30%): How much traffic flows through this agent
        - Latency impact (30%): How slow is this agent
        - Failure rate (20%): How often does it fail
        - Downstream impact (20%): Does it delay other agents

        Args:
            agent: Agent
            node: Graph node
            edges: All edges
            operations: All operations
            handoffs: All handoffs

        Returns:
            Bottleneck score (0.0 to 1.0)
        """
        components = []

        # 1. Traffic utilization
        utilization_score = self._calculate_utilization_score(agent, node, edges)
        components.append(("utilization", utilization_score, 0.30))

        # 2. Latency impact
        latency_score = self._calculate_latency_score(agent, node, operations)
        components.append(("latency", latency_score, 0.30))

        # 3. Failure rate
        failure_score = agent.failure_count / max(agent.total_invocations, 1)
        components.append(("failure_rate", failure_score, 0.20))

        # 4. Downstream impact
        downstream_score = self._calculate_downstream_impact(agent, edges, handoffs)
        components.append(("downstream", downstream_score, 0.20))

        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in components)

        logger.debug(f"Bottleneck score for {agent.agent_name}: {total_score:.2f}")
        for name, score, weight in components:
            logger.debug(f"  {name}: {score:.2f} (weight: {weight})")

        return total_score

    def _calculate_utilization_score(
        self,
        agent: Agent,
        node: GraphNode,
        edges: List[GraphEdge]
    ) -> float:
        """
        Calculate traffic utilization score.

        High utilization = potential bottleneck.

        Args:
            agent: Agent
            node: Graph node
            edges: All edges

        Returns:
            Utilization score (0.0 to 1.0)
        """
        # Calculate traffic through this node
        incoming = sum(
            e.handoff_count for e in edges
            if e.to_agent_id == agent.agent_id
        )
        outgoing = sum(
            e.handoff_count for e in edges
            if e.from_agent_id == agent.agent_id
        )

        total_traffic = incoming + outgoing + node.invocation_count

        # Calculate max traffic (busiest agent)
        all_traffic = []
        for node_iter in [node]:  # Would iterate all nodes in production
            node_traffic = node_iter.invocation_count
            all_traffic.append(node_traffic)

        max_traffic = max(all_traffic) if all_traffic else 1

        # Normalize
        utilization = total_traffic / max(max_traffic, 1)

        return min(utilization, 1.0)

    def _calculate_latency_score(
        self,
        agent: Agent,
        node: GraphNode,
        operations: List[AgentOperation]
    ) -> float:
        """
        Calculate latency impact score.

        High latency = potential bottleneck.

        Args:
            agent: Agent
            node: Graph node
            operations: All operations

        Returns:
            Latency score (0.0 to 1.0)
        """
        # Get operations for this agent
        agent_ops = [
            op for op in operations
            if op.agent_id == agent.agent_id
        ]

        if not agent_ops:
            return 0.0

        # Calculate p95 latency
        latencies = [op.duration_ms for op in agent_ops]
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index] if latencies else 0.0

        # Normalize to score
        # 0-500ms: low (0.0-0.3)
        # 500-2000ms: medium (0.3-0.7)
        # 2000-5000ms: high (0.7-0.9)
        # >5000ms: critical (0.9-1.0)

        if p95_latency < 500:
            score = p95_latency / 500 * 0.3
        elif p95_latency < 2000:
            score = 0.3 + ((p95_latency - 500) / 1500) * 0.4
        elif p95_latency < 5000:
            score = 0.7 + ((p95_latency - 2000) / 3000) * 0.2
        else:
            score = 0.9 + min((p95_latency - 5000) / 5000, 0.1)

        return min(score, 1.0)

    def _calculate_downstream_impact(
        self,
        agent: Agent,
        edges: List[GraphEdge],
        handoffs: List[AgentHandoff]
    ) -> float:
        """
        Calculate impact on downstream agents.

        If this agent delays, does it cause cascading delays?

        Args:
            agent: Agent
            edges: All edges
            handoffs: All handoffs

        Returns:
            Downstream impact score (0.0 to 1.0)
        """
        # Find handoffs from this agent
        outgoing_handoffs = [
            h for h in handoffs
            if h.from_agent_id == agent.agent_id
        ]

        if not outgoing_handoffs:
            return 0.0

        # Calculate average handoff latency
        avg_handoff_latency = statistics.mean(
            h.handoff_latency_ms for h in outgoing_handoffs
        )

        # High handoff latency = high downstream impact
        # 0-100ms: low impact
        # 100-500ms: medium impact
        # >500ms: high impact

        if avg_handoff_latency < 100:
            return avg_handoff_latency / 100 * 0.3
        elif avg_handoff_latency < 500:
            return 0.3 + ((avg_handoff_latency - 100) / 400) * 0.4
        else:
            return 0.7 + min((avg_handoff_latency - 500) / 500, 0.3)

    def _create_bottleneck_info(
        self,
        agent: Agent,
        node: GraphNode,
        edges: List[GraphEdge],
        operations: List[AgentOperation],
        handoffs: List[AgentHandoff],
        score: float
    ) -> BottleneckInfo:
        """
        Create detailed bottleneck information.

        Args:
            agent: Agent
            node: Graph node
            edges: All edges
            operations: All operations
            handoffs: All handoffs
            score: Bottleneck score

        Returns:
            BottleneckInfo
        """
        # Get agent operations
        agent_ops = [
            op for op in operations
            if op.agent_id == agent.agent_id
        ]

        # Calculate metrics
        latencies = [op.duration_ms for op in agent_ops] if agent_ops else [0.0]
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index] if latencies else 0.0

        # Calculate utilization
        incoming = sum(e.handoff_count for e in edges if e.to_agent_id == agent.agent_id)
        outgoing = sum(e.handoff_count for e in edges if e.from_agent_id == agent.agent_id)
        total_traffic = incoming + outgoing + node.invocation_count

        # Estimate utilization (simplified)
        utilization = min(total_traffic / max(node.invocation_count, 1) * 0.5, 1.0)

        # Queue time (simplified - would be measured in production)
        avg_queue_time = max(0, p95_latency - agent.avg_latency_ms)

        # Requests delayed (estimate)
        requests_delayed = int(agent.total_invocations * (1 - agent.success_rate))

        # Determine severity
        if score >= 0.8:
            severity = "critical"
        elif score >= 0.7:
            severity = "high"
        elif score >= 0.6:
            severity = "medium"
        else:
            severity = "low"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            agent, node, edges, operations, score, p95_latency, utilization
        )

        return BottleneckInfo(
            agent_id=agent.agent_id,
            agent_name=agent.agent_name,
            severity=severity,
            bottleneck_score=score,
            avg_queue_time_ms=avg_queue_time,
            p95_latency_ms=p95_latency,
            utilization=utilization,
            requests_delayed=requests_delayed,
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        agent: Agent,
        node: GraphNode,
        edges: List[GraphEdge],
        operations: List[AgentOperation],
        score: float,
        p95_latency: float,
        utilization: float
    ) -> List[str]:
        """
        Generate optimization recommendations.

        Args:
            agent: Agent
            node: Graph node
            edges: All edges
            operations: All operations
            score: Bottleneck score
            p95_latency: P95 latency
            utilization: Utilization

        Returns:
            List of recommendations
        """
        recommendations = []

        # High latency recommendations
        if p95_latency > 2000:
            recommendations.append(
                f"⚡ P95 latency is {p95_latency:.0f}ms. Consider optimizing agent logic or using caching."
            )

        if p95_latency > 1000 and agent.avg_latency_ms < p95_latency * 0.5:
            recommendations.append(
                "📊 High variance in latency. Investigate intermittent slowdowns or external dependencies."
            )

        # High utilization recommendations
        if utilization > 0.7:
            recommendations.append(
                f"🔄 High utilization ({utilization:.0%}). Consider horizontal scaling (add more instances)."
            )

        # High failure rate recommendations
        failure_rate = agent.failure_count / max(agent.total_invocations, 1)
        if failure_rate > 0.1:
            recommendations.append(
                f"❌ High failure rate ({failure_rate:.1%}). Review error handling and add retry logic."
            )

        # Handoff recommendations
        outgoing_edges = [e for e in edges if e.from_agent_id == agent.agent_id]
        if outgoing_edges:
            avg_handoff_quality = statistics.mean(e.avg_quality_score for e in outgoing_edges)
            if avg_handoff_quality < 0.7:
                recommendations.append(
                    f"🔀 Low handoff quality ({avg_handoff_quality:.2f}). Review handoff decision logic."
                )

        # General optimization
        if score > 0.8:
            recommendations.append(
                "🚨 CRITICAL bottleneck detected. This agent requires immediate optimization."
            )
            recommendations.append(
                "💡 Consider: (1) Caching frequently accessed data, (2) Adding circuit breakers, "
                "(3) Implementing request queuing, (4) Scaling horizontally."
            )

        # If no specific recommendations
        if not recommendations:
            recommendations.append(
                "✅ Agent is performing reasonably well. Monitor for future degradation."
            )

        return recommendations

    def get_critical_bottlenecks(self) -> List[BottleneckInfo]:
        """
        Get critical bottlenecks (score > 0.8).

        Returns:
            List of critical bottlenecks
        """
        return [
            b for b in self.bottleneck_history
            if b.bottleneck_score > 0.8
        ]

    def get_bottleneck_summary(self) -> Dict[str, any]:
        """
        Get summary of bottleneck analysis.

        Returns:
            Summary statistics
        """
        if not self.bottleneck_history:
            return {
                "total_bottlenecks": 0,
                "critical_count": 0,
                "high_count": 0,
                "avg_score": 0.0
            }

        critical_count = sum(1 for b in self.bottleneck_history if b.severity == "critical")
        high_count = sum(1 for b in self.bottleneck_history if b.severity == "high")
        avg_score = statistics.mean(b.bottleneck_score for b in self.bottleneck_history)

        return {
            "total_bottlenecks": len(self.bottleneck_history),
            "critical_count": critical_count,
            "high_count": high_count,
            "medium_count": sum(1 for b in self.bottleneck_history if b.severity == "medium"),
            "low_count": sum(1 for b in self.bottleneck_history if b.severity == "low"),
            "avg_score": avg_score,
            "max_score": max(b.bottleneck_score for b in self.bottleneck_history),
            "most_critical": max(
                self.bottleneck_history,
                key=lambda b: b.bottleneck_score
            ).agent_name if self.bottleneck_history else None
        }

    def analyze_trends(
        self,
        window_size: int = 10
    ) -> Dict[str, any]:
        """
        Analyze bottleneck trends over time.

        Args:
            window_size: Number of recent analyses to consider

        Returns:
            Trend analysis
        """
        if len(self.bottleneck_history) < 2:
            return {
                "trend": "insufficient_data",
                "message": "Need more data points for trend analysis"
            }

        recent = self.bottleneck_history[-window_size:]

        # Group by agent
        agent_scores: Dict[str, List[float]] = defaultdict(list)
        for bottleneck in recent:
            agent_scores[bottleneck.agent_name].append(bottleneck.bottleneck_score)

        # Analyze trends
        trends = {}
        for agent_name, scores in agent_scores.items():
            if len(scores) >= 2:
                # Simple trend: compare first half to second half
                mid = len(scores) // 2
                first_half_avg = statistics.mean(scores[:mid])
                second_half_avg = statistics.mean(scores[mid:])

                change = second_half_avg - first_half_avg

                if change > 0.1:
                    trend = "worsening"
                elif change < -0.1:
                    trend = "improving"
                else:
                    trend = "stable"

                trends[agent_name] = {
                    "trend": trend,
                    "change": change,
                    "current_score": scores[-1],
                    "avg_score": statistics.mean(scores)
                }

        return {
            "analyzed_agents": len(trends),
            "agent_trends": trends,
            "overall_trend": self._determine_overall_trend(trends)
        }

    def _determine_overall_trend(self, agent_trends: Dict[str, Dict]) -> str:
        """Determine overall system trend."""
        if not agent_trends:
            return "unknown"

        worsening = sum(1 for t in agent_trends.values() if t["trend"] == "worsening")
        improving = sum(1 for t in agent_trends.values() if t["trend"] == "improving")

        if worsening > improving:
            return "worsening"
        elif improving > worsening:
            return "improving"
        else:
            return "stable"
