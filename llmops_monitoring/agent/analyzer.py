"""
Coalition Analysis.

Analyzes groups of agents that work together and identifies optimal team compositions.
"""

from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID
from collections import defaultdict
import statistics

from llmops_monitoring.agent.base import (
    Agent,
    AgentOperation,
    AgentHandoff,
    Coalition
)
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class CoalitionAnalyzer:
    """
    Analyzes agent coalitions (teams).

    Identifies groups of agents that frequently collaborate
    and evaluates their effectiveness.
    """

    def __init__(self):
        """Initialize analyzer."""
        self.coalitions: List[Coalition] = []
        self.task_patterns: Dict[str, Set[str]] = {}  # task_type -> agent_names

    async def discover_coalitions(
        self,
        agents: List[Agent],
        operations: List[AgentOperation],
        handoffs: List[AgentHandoff],
        session_id: str
    ) -> List[Coalition]:
        """
        Discover agent coalitions from collaboration patterns.

        Args:
            agents: List of agents
            operations: List of operations
            handoffs: List of handoffs
            session_id: Session ID

        Returns:
            List of discovered coalitions
        """
        # Build collaboration graph
        collaboration_graph = self._build_collaboration_graph(handoffs)

        # Identify clusters of frequently collaborating agents
        clusters = self._identify_clusters(collaboration_graph, agents)

        # Create coalitions from clusters
        coalitions = []
        for cluster_agents in clusters:
            coalition = self._create_coalition(
                cluster_agents, operations, handoffs, session_id
            )
            coalitions.append(coalition)

        self.coalitions.extend(coalitions)
        logger.info(f"Discovered {len(coalitions)} agent coalitions")

        return coalitions

    def _build_collaboration_graph(
        self,
        handoffs: List[AgentHandoff]
    ) -> Dict[str, Dict[str, int]]:
        """
        Build collaboration graph showing agent interactions.

        Args:
            handoffs: List of handoffs

        Returns:
            Graph: agent_name -> {partner_name -> handoff_count}
        """
        graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for handoff in handoffs:
            from_agent = handoff.from_agent_name
            to_agent = handoff.to_agent_name

            # Bidirectional collaboration
            graph[from_agent][to_agent] += 1
            graph[to_agent][from_agent] += 1

        return graph

    def _identify_clusters(
        self,
        collaboration_graph: Dict[str, Dict[str, int]],
        agents: List[Agent],
        min_cluster_size: int = 2,
        min_collaboration_count: int = 2
    ) -> List[Set[str]]:
        """
        Identify clusters of collaborating agents.

        Uses simple connected components algorithm.

        Args:
            collaboration_graph: Collaboration graph
            agents: All agents
            min_cluster_size: Minimum cluster size
            min_collaboration_count: Minimum handoffs to consider collaboration

        Returns:
            List of agent clusters
        """
        # Filter weak connections
        strong_graph: Dict[str, Set[str]] = defaultdict(set)

        for agent_name, partners in collaboration_graph.items():
            for partner, count in partners.items():
                if count >= min_collaboration_count:
                    strong_graph[agent_name].add(partner)

        # Find connected components
        visited: Set[str] = set()
        clusters: List[Set[str]] = []

        for agent_name in strong_graph.keys():
            if agent_name in visited:
                continue

            # BFS to find cluster
            cluster = self._bfs_cluster(agent_name, strong_graph)
            visited.update(cluster)

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters

    def _bfs_cluster(
        self,
        start_agent: str,
        graph: Dict[str, Set[str]]
    ) -> Set[str]:
        """
        Find cluster using BFS.

        Args:
            start_agent: Starting agent
            graph: Collaboration graph

        Returns:
            Cluster of connected agents
        """
        cluster = {start_agent}
        queue = [start_agent]
        visited = {start_agent}

        while queue:
            current = queue.pop(0)

            for neighbor in graph.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    cluster.add(neighbor)
                    queue.append(neighbor)

        return cluster

    def _create_coalition(
        self,
        agent_names: Set[str],
        operations: List[AgentOperation],
        handoffs: List[AgentHandoff],
        session_id: str
    ) -> Coalition:
        """
        Create coalition from agent cluster.

        Args:
            agent_names: Agents in coalition
            operations: All operations
            handoffs: All handoffs
            session_id: Session ID

        Returns:
            Coalition
        """
        # Filter operations for this coalition
        coalition_ops = [
            op for op in operations
            if op.agent_name in agent_names
        ]

        # Filter handoffs within coalition
        coalition_handoffs = [
            h for h in handoffs
            if h.from_agent_name in agent_names and h.to_agent_name in agent_names
        ]

        # Infer task type from operations
        task_type = self._infer_task_type(coalition_ops, agent_names)

        # Calculate metrics
        total_tasks = len(set(op.event_id for op in coalition_ops))
        successful_tasks = len(set(
            op.event_id for op in coalition_ops if op.success
        ))

        # Calculate average latency (per task)
        task_latencies: Dict[UUID, float] = defaultdict(float)
        for op in coalition_ops:
            task_latencies[op.event_id] += op.duration_ms

        avg_total_latency = (
            statistics.mean(task_latencies.values())
            if task_latencies else 0.0
        )

        # Calculate handoff efficiency
        avg_handoff_efficiency = (
            statistics.mean(h.quality_score for h in coalition_handoffs)
            if coalition_handoffs else 0.0
        )

        # Create coalition name
        coalition_name = self._generate_coalition_name(agent_names, task_type)

        return Coalition(
            coalition_name=coalition_name,
            agent_names=list(agent_names),
            task_type=task_type,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            avg_total_latency_ms=avg_total_latency,
            avg_total_cost_usd=0.0,  # Would calculate from cost data
            avg_handoff_efficiency=avg_handoff_efficiency
        )

    def _infer_task_type(
        self,
        operations: List[AgentOperation],
        agent_names: Set[str]
    ) -> str:
        """
        Infer task type from operation patterns.

        Args:
            operations: Coalition operations
            agent_names: Agent names

        Returns:
            Inferred task type
        """
        # Check agent names for patterns
        names_lower = {name.lower() for name in agent_names}

        if any('support' in name or 'customer' in name for name in names_lower):
            return "customer_support"
        elif any('search' in name or 'retriev' in name for name in names_lower):
            return "information_retrieval"
        elif any('generat' in name or 'creat' in name for name in names_lower):
            return "content_generation"
        elif any('analy' in name or 'process' in name for name in names_lower):
            return "data_analysis"
        elif any('classif' in name or 'route' in name for name in names_lower):
            return "routing_classification"
        else:
            return "general_workflow"

    def _generate_coalition_name(
        self,
        agent_names: Set[str],
        task_type: str
    ) -> str:
        """
        Generate descriptive coalition name.

        Args:
            agent_names: Agent names
            task_type: Task type

        Returns:
            Coalition name
        """
        # Convert task type to readable name
        task_display = task_type.replace("_", " ").title()

        # Use first 2-3 agent names
        agents_list = sorted(list(agent_names))[:3]
        agents_str = ", ".join(agents_list)

        if len(agent_names) > 3:
            agents_str += f" (+{len(agent_names) - 3} more)"

        return f"{task_display} Team ({agents_str})"

    def analyze_coalition_performance(
        self,
        coalition: Coalition
    ) -> Dict[str, any]:
        """
        Analyze coalition performance in detail.

        Args:
            coalition: Coalition to analyze

        Returns:
            Performance analysis
        """
        analysis = {
            "coalition_name": coalition.coalition_name,
            "agent_count": len(coalition.agent_names),
            "success_rate": coalition.success_rate,
            "avg_latency_ms": coalition.avg_total_latency_ms,
            "avg_handoff_efficiency": coalition.avg_handoff_efficiency,
            "total_tasks": coalition.total_tasks
        }

        # Performance rating
        rating = self._rate_coalition_performance(coalition)
        analysis["performance_rating"] = rating

        # Recommendations
        recommendations = self._generate_coalition_recommendations(coalition)
        analysis["recommendations"] = recommendations

        return analysis

    def _rate_coalition_performance(self, coalition: Coalition) -> str:
        """
        Rate coalition performance.

        Args:
            coalition: Coalition

        Returns:
            Rating: "excellent", "good", "fair", "poor"
        """
        score = 0.0

        # Success rate (40%)
        score += coalition.success_rate * 0.4

        # Handoff efficiency (30%)
        score += coalition.avg_handoff_efficiency * 0.3

        # Latency (30%) - inverse, lower is better
        latency_score = max(0, 1.0 - (coalition.avg_total_latency_ms / 5000))
        score += latency_score * 0.3

        if score >= 0.85:
            return "excellent"
        elif score >= 0.70:
            return "good"
        elif score >= 0.50:
            return "fair"
        else:
            return "poor"

    def _generate_coalition_recommendations(
        self,
        coalition: Coalition
    ) -> List[str]:
        """
        Generate recommendations for coalition improvement.

        Args:
            coalition: Coalition

        Returns:
            List of recommendations
        """
        recommendations = []

        # Success rate recommendations
        if coalition.success_rate < 0.9:
            recommendations.append(
                f"📊 Success rate is {coalition.success_rate:.1%}. "
                "Review error handling and agent coordination."
            )

        # Handoff efficiency recommendations
        if coalition.avg_handoff_efficiency < 0.7:
            recommendations.append(
                f"🔀 Low handoff efficiency ({coalition.avg_handoff_efficiency:.2f}). "
                "Improve context passing between agents."
            )

        # Latency recommendations
        if coalition.avg_total_latency_ms > 3000:
            recommendations.append(
                f"⏱️ High average latency ({coalition.avg_total_latency_ms:.0f}ms). "
                "Consider parallelizing agent operations or optimizing slow agents."
            )

        # Team size recommendations
        if len(coalition.agent_names) > 5:
            recommendations.append(
                f"👥 Large team ({len(coalition.agent_names)} agents). "
                "Consider splitting into smaller, specialized sub-teams."
            )
        elif len(coalition.agent_names) == 2:
            recommendations.append(
                "👥 Small team (2 agents). Monitor for opportunities to add specialists."
            )

        if not recommendations:
            recommendations.append(
                "✅ Coalition is performing well. Continue monitoring for changes."
            )

        return recommendations

    def compare_coalitions(
        self,
        coalition_a: Coalition,
        coalition_b: Coalition
    ) -> Dict[str, any]:
        """
        Compare two coalitions.

        Args:
            coalition_a: First coalition
            coalition_b: Second coalition

        Returns:
            Comparison analysis
        """
        comparison = {
            "coalition_a": coalition_a.coalition_name,
            "coalition_b": coalition_b.coalition_name,
            "metrics": {}
        }

        # Success rate
        comparison["metrics"]["success_rate"] = {
            "a": coalition_a.success_rate,
            "b": coalition_b.success_rate,
            "winner": "a" if coalition_a.success_rate > coalition_b.success_rate else "b"
        }

        # Latency
        comparison["metrics"]["avg_latency_ms"] = {
            "a": coalition_a.avg_total_latency_ms,
            "b": coalition_b.avg_total_latency_ms,
            "winner": "a" if coalition_a.avg_total_latency_ms < coalition_b.avg_total_latency_ms else "b"
        }

        # Handoff efficiency
        comparison["metrics"]["handoff_efficiency"] = {
            "a": coalition_a.avg_handoff_efficiency,
            "b": coalition_b.avg_handoff_efficiency,
            "winner": "a" if coalition_a.avg_handoff_efficiency > coalition_b.avg_handoff_efficiency else "b"
        }

        # Overall winner
        a_wins = sum(
            1 for metric in comparison["metrics"].values()
            if metric["winner"] == "a"
        )
        comparison["overall_winner"] = "a" if a_wins > 1 else "b"

        return comparison

    def suggest_optimal_coalition(
        self,
        task_type: str,
        available_agents: List[Agent]
    ) -> Dict[str, any]:
        """
        Suggest optimal coalition for a task type.

        Based on historical performance data.

        Args:
            task_type: Type of task
            available_agents: Available agents

        Returns:
            Suggested coalition
        """
        # Find coalitions that handled this task type
        relevant_coalitions = [
            c for c in self.coalitions
            if c.task_type == task_type
        ]

        if not relevant_coalitions:
            return {
                "suggestion": "no_data",
                "message": f"No historical data for task type: {task_type}"
            }

        # Rank by performance
        ranked = sorted(
            relevant_coalitions,
            key=lambda c: (c.success_rate, -c.avg_total_latency_ms),
            reverse=True
        )

        best = ranked[0]

        # Check if agents are available
        available_names = {agent.agent_name for agent in available_agents}
        missing_agents = set(best.agent_names) - available_names

        if missing_agents:
            # Find alternative
            for coalition in ranked[1:]:
                missing = set(coalition.agent_names) - available_names
                if not missing:
                    best = coalition
                    break

        return {
            "task_type": task_type,
            "recommended_agents": best.agent_names,
            "expected_success_rate": best.success_rate,
            "expected_latency_ms": best.avg_total_latency_ms,
            "expected_handoff_efficiency": best.avg_handoff_efficiency,
            "based_on_tasks": best.total_tasks,
            "performance_rating": self._rate_coalition_performance(best),
            "missing_agents": list(missing_agents) if missing_agents else []
        }

    def get_coalition_summary(self) -> Dict[str, any]:
        """
        Get summary of all coalitions.

        Returns:
            Summary statistics
        """
        if not self.coalitions:
            return {
                "total_coalitions": 0,
                "avg_team_size": 0.0,
                "avg_success_rate": 0.0
            }

        total = len(self.coalitions)
        avg_team_size = statistics.mean(len(c.agent_names) for c in self.coalitions)
        avg_success_rate = statistics.mean(c.success_rate for c in self.coalitions)
        avg_handoff_efficiency = statistics.mean(
            c.avg_handoff_efficiency for c in self.coalitions
        )

        # Find best performing coalition
        best = max(self.coalitions, key=lambda c: c.success_rate)

        # Task type distribution
        task_types = defaultdict(int)
        for coalition in self.coalitions:
            task_types[coalition.task_type] += 1

        return {
            "total_coalitions": total,
            "avg_team_size": avg_team_size,
            "avg_success_rate": avg_success_rate,
            "avg_handoff_efficiency": avg_handoff_efficiency,
            "best_coalition": {
                "name": best.coalition_name,
                "success_rate": best.success_rate,
                "task_type": best.task_type
            },
            "task_type_distribution": dict(task_types)
        }

    def detect_coalition_issues(
        self,
        coalition: Coalition
    ) -> List[Dict[str, any]]:
        """
        Detect issues in coalition.

        Args:
            coalition: Coalition to analyze

        Returns:
            List of detected issues
        """
        issues = []

        # Low success rate
        if coalition.success_rate < 0.8:
            issues.append({
                "severity": "high",
                "issue": "low_success_rate",
                "message": f"Success rate is only {coalition.success_rate:.1%}",
                "recommendation": "Review agent error logs and improve error handling"
            })

        # High latency
        if coalition.avg_total_latency_ms > 5000:
            issues.append({
                "severity": "high",
                "issue": "high_latency",
                "message": f"Average latency is {coalition.avg_total_latency_ms:.0f}ms",
                "recommendation": "Optimize slow agents or introduce parallelization"
            })

        # Poor handoff efficiency
        if coalition.avg_handoff_efficiency < 0.6:
            issues.append({
                "severity": "medium",
                "issue": "poor_handoffs",
                "message": f"Handoff efficiency is {coalition.avg_handoff_efficiency:.2f}",
                "recommendation": "Improve context passing and handoff decision logic"
            })

        # Low task volume
        if coalition.total_tasks < 10:
            issues.append({
                "severity": "low",
                "issue": "insufficient_data",
                "message": f"Only {coalition.total_tasks} tasks analyzed",
                "recommendation": "Collect more data before making optimization decisions"
            })

        return issues
