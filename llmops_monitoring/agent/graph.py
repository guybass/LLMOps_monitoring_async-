"""
Coordination Graph Builder.

Builds visual representation of agent interactions as a directed graph.
"""

from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID
from collections import defaultdict, deque

from llmops_monitoring.agent.base import (
    Agent,
    AgentOperation,
    AgentHandoff,
    CoordinationGraph,
    GraphNode,
    GraphEdge
)
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class CoordinationGraphBuilder:
    """
    Builds coordination graphs from agent operations.

    Creates a directed graph showing:
    - Nodes: Agents with their metrics
    - Edges: Handoffs between agents with quality scores
    """

    def __init__(self):
        """Initialize graph builder."""
        pass

    async def build_graph(
        self,
        agents: List[Agent],
        operations: List[AgentOperation],
        handoffs: List[AgentHandoff],
        session_id: str
    ) -> CoordinationGraph:
        """
        Build coordination graph.

        Args:
            agents: List of agents
            operations: List of agent operations
            handoffs: List of handoffs
            session_id: Session ID for this graph

        Returns:
            CoordinationGraph
        """
        # Build nodes
        nodes = self._build_nodes(agents, operations)

        # Build edges
        edges = self._build_edges(handoffs)

        # Calculate graph metrics
        metrics = self._calculate_graph_metrics(nodes, edges, operations)

        # Create graph
        graph = CoordinationGraph(
            session_id=session_id,
            nodes=[self._node_to_dict(n) for n in nodes],
            edges=[self._edge_to_dict(e) for e in edges],
            total_agents=len(nodes),
            total_handoffs=sum(e.handoff_count for e in edges),
            max_depth=metrics['max_depth'],
            bottleneck_agent_id=metrics.get('bottleneck_agent_id'),
            critical_path_ms=metrics.get('critical_path_ms'),
            avg_handoff_quality=metrics.get('avg_handoff_quality', 0.0),
            avg_context_retention=0.0  # Would be calculated from drift analysis
        )

        logger.info(
            f"Built coordination graph: {len(nodes)} agents, {len(edges)} edges, "
            f"max depth: {metrics['max_depth']}"
        )

        return graph

    def _build_nodes(
        self,
        agents: List[Agent],
        operations: List[AgentOperation]
    ) -> List[GraphNode]:
        """
        Build graph nodes from agents and operations.

        Args:
            agents: List of agents
            operations: List of operations

        Returns:
            List of graph nodes
        """
        nodes = []

        # Create agent_id -> operations map
        agent_ops: Dict[UUID, List[AgentOperation]] = defaultdict(list)
        for op in operations:
            agent_ops[op.agent_id].append(op)

        # Build nodes
        for agent in agents:
            ops = agent_ops.get(agent.agent_id, [])

            # Calculate aggregated metrics
            invocation_count = len(ops)
            total_latency = sum(op.duration_ms for op in ops)
            success_count = sum(1 for op in ops if op.success)
            failure_count = sum(1 for op in ops if not op.success)

            node = GraphNode(
                agent_id=agent.agent_id,
                agent_name=agent.agent_name,
                agent_type=agent.agent_type,
                invocation_count=invocation_count,
                total_latency_ms=total_latency,
                success_count=success_count,
                failure_count=failure_count
            )

            nodes.append(node)

        return nodes

    def _build_edges(self, handoffs: List[AgentHandoff]) -> List[GraphEdge]:
        """
        Build graph edges from handoffs.

        Args:
            handoffs: List of handoffs

        Returns:
            List of graph edges
        """
        # Group handoffs by (from_agent, to_agent) pair
        edge_map: Dict[Tuple[UUID, UUID], List[AgentHandoff]] = defaultdict(list)

        for handoff in handoffs:
            key = (handoff.from_agent_id, handoff.to_agent_id)
            edge_map[key].append(handoff)

        # Build edges
        edges = []

        for (from_id, to_id), group_handoffs in edge_map.items():
            # Get representative handoff for names
            first = group_handoffs[0]

            # Calculate aggregated metrics
            handoff_count = len(group_handoffs)
            total_latency = sum(h.handoff_latency_ms for h in group_handoffs)
            success_count = sum(1 for h in group_handoffs if h.was_correct_agent)
            failure_count = handoff_count - success_count
            total_quality = sum(h.quality_score for h in group_handoffs)

            edge = GraphEdge(
                from_agent_id=from_id,
                from_agent_name=first.from_agent_name,
                to_agent_id=to_id,
                to_agent_name=first.to_agent_name,
                handoff_count=handoff_count,
                total_latency_ms=total_latency,
                success_count=success_count,
                failure_count=failure_count,
                total_quality_score=total_quality
            )

            edges.append(edge)

        return edges

    def _calculate_graph_metrics(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        operations: List[AgentOperation]
    ) -> Dict[str, any]:
        """
        Calculate graph-level metrics.

        Args:
            nodes: Graph nodes
            edges: Graph edges
            operations: All operations

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Max depth (longest chain)
        metrics['max_depth'] = self._calculate_max_depth(operations)

        # Critical path (longest execution time)
        metrics['critical_path_ms'] = self._calculate_critical_path(operations)

        # Bottleneck detection
        bottleneck = self._identify_bottleneck(nodes, edges)
        if bottleneck:
            metrics['bottleneck_agent_id'] = bottleneck.agent_id

        # Average handoff quality
        if edges:
            total_quality = sum(e.total_quality_score for e in edges)
            total_handoffs = sum(e.handoff_count for e in edges)
            metrics['avg_handoff_quality'] = total_quality / total_handoffs if total_handoffs > 0 else 0.0

        return metrics

    def _calculate_max_depth(self, operations: List[AgentOperation]) -> int:
        """
        Calculate maximum chain depth.

        Args:
            operations: List of operations

        Returns:
            Maximum depth
        """
        # Build parent-child map
        children: Dict[UUID, List[AgentOperation]] = defaultdict(list)

        for op in operations:
            if op.parent_operation_id:
                children[op.parent_operation_id].append(op)

        # Find roots (operations with no parent)
        roots = [op for op in operations if not op.parent_operation_id]

        if not roots:
            return 0

        # BFS to find max depth
        max_depth = 0

        for root in roots:
            depth = self._calculate_depth_from(root, children)
            max_depth = max(max_depth, depth)

        return max_depth

    def _calculate_depth_from(
        self,
        operation: AgentOperation,
        children: Dict[UUID, List[AgentOperation]]
    ) -> int:
        """
        Calculate depth from a starting operation.

        Args:
            operation: Starting operation
            children: Map of operation_id -> child operations

        Returns:
            Depth
        """
        queue = deque([(operation, 1)])
        max_depth = 1

        while queue:
            current, depth = queue.popleft()
            max_depth = max(max_depth, depth)

            for child in children.get(current.operation_id, []):
                queue.append((child, depth + 1))

        return max_depth

    def _calculate_critical_path(self, operations: List[AgentOperation]) -> float:
        """
        Calculate critical path (longest execution time).

        Args:
            operations: List of operations

        Returns:
            Critical path latency in ms
        """
        # Build dependency graph
        children: Dict[UUID, List[AgentOperation]] = defaultdict(list)

        for op in operations:
            if op.parent_operation_id:
                children[op.parent_operation_id].append(op)

        # Find roots
        roots = [op for op in operations if not op.parent_operation_id]

        if not roots:
            return 0.0

        # Calculate longest path
        max_path = 0.0

        for root in roots:
            path_length = self._longest_path_from(root, children)
            max_path = max(max_path, path_length)

        return max_path

    def _longest_path_from(
        self,
        operation: AgentOperation,
        children: Dict[UUID, List[AgentOperation]]
    ) -> float:
        """
        Calculate longest path from operation.

        Args:
            operation: Starting operation
            children: Child operations map

        Returns:
            Longest path length in ms
        """
        # Get all children
        child_ops = children.get(operation.operation_id, [])

        if not child_ops:
            # Leaf node
            return operation.duration_ms

        # Recursively calculate longest path through children
        max_child_path = max(
            self._longest_path_from(child, children)
            for child in child_ops
        )

        return operation.duration_ms + max_child_path

    def _identify_bottleneck(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> Optional[GraphNode]:
        """
        Identify bottleneck agent.

        Bottleneck is the agent with:
        - High utilization (lots of traffic)
        - High average latency
        - Low success rate

        Args:
            nodes: Graph nodes
            edges: Graph edges

        Returns:
            Bottleneck node or None
        """
        if not nodes:
            return None

        # Calculate bottleneck scores
        scored_nodes = []

        for node in nodes:
            score = self._calculate_bottleneck_score(node, edges)
            scored_nodes.append((score, node))

        # Return highest scoring node
        scored_nodes.sort(reverse=True)

        if scored_nodes and scored_nodes[0][0] > 0.5:
            return scored_nodes[0][1]

        return None

    def _calculate_bottleneck_score(
        self,
        node: GraphNode,
        edges: List[GraphEdge]
    ) -> float:
        """
        Calculate bottleneck score for a node.

        Args:
            node: Graph node
            edges: All edges

        Returns:
            Bottleneck score (0.0 to 1.0)
        """
        # Calculate traffic through this node
        incoming = sum(
            e.handoff_count for e in edges
            if e.to_agent_id == node.agent_id
        )
        outgoing = sum(
            e.handoff_count for e in edges
            if e.from_agent_id == node.agent_id
        )

        total_traffic = incoming + outgoing
        max_traffic = max(
            sum(e.handoff_count for e in edges if e.to_agent_id == n.agent_id) +
            sum(e.handoff_count for e in edges if e.from_agent_id == n.agent_id)
            for n in [node]  # Simplified
        )

        # Utilization score (0-1)
        utilization = total_traffic / max_traffic if max_traffic > 0 else 0.0

        # Latency score (higher latency = higher score)
        avg_latency = node.avg_latency_ms
        latency_score = min(avg_latency / 5000.0, 1.0)  # Normalize to 5s

        # Failure rate
        failure_rate = node.failure_count / (node.success_count + node.failure_count) \
                      if (node.success_count + node.failure_count) > 0 else 0.0

        # Combined score
        bottleneck_score = (
            utilization * 0.4 +
            latency_score * 0.4 +
            failure_rate * 0.2
        )

        return bottleneck_score

    def _node_to_dict(self, node: GraphNode) -> Dict[str, any]:
        """Convert node to dictionary."""
        return {
            "agent_id": str(node.agent_id),
            "agent_name": node.agent_name,
            "agent_type": node.agent_type.value,
            "invocation_count": node.invocation_count,
            "avg_latency_ms": node.avg_latency_ms,
            "success_rate": node.success_rate,
            "position": node.position
        }

    def _edge_to_dict(self, edge: GraphEdge) -> Dict[str, any]:
        """Convert edge to dictionary."""
        return {
            "from_agent_id": str(edge.from_agent_id),
            "from_agent_name": edge.from_agent_name,
            "to_agent_id": str(edge.to_agent_id),
            "to_agent_name": edge.to_agent_name,
            "handoff_count": edge.handoff_count,
            "avg_latency_ms": edge.avg_latency_ms,
            "success_rate": edge.success_rate,
            "avg_quality_score": edge.avg_quality_score
        }

    def export_for_visualization(
        self,
        graph: CoordinationGraph,
        format: str = "networkx"
    ) -> Dict[str, any]:
        """
        Export graph for visualization libraries.

        Args:
            graph: Coordination graph
            format: Export format ("networkx", "cytoscape", "d3")

        Returns:
            Graph data in requested format
        """
        if format == "networkx":
            return self._export_networkx(graph)
        elif format == "cytoscape":
            return self._export_cytoscape(graph)
        elif format == "d3":
            return self._export_d3(graph)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_networkx(self, graph: CoordinationGraph) -> Dict[str, any]:
        """Export for NetworkX."""
        return {
            "nodes": graph.nodes,
            "edges": graph.edges,
            "directed": True
        }

    def _export_cytoscape(self, graph: CoordinationGraph) -> Dict[str, any]:
        """Export for Cytoscape.js."""
        elements = []

        # Add nodes
        for node in graph.nodes:
            elements.append({
                "data": {
                    "id": node["agent_id"],
                    "label": node["agent_name"],
                    **node
                }
            })

        # Add edges
        for edge in graph.edges:
            elements.append({
                "data": {
                    "source": edge["from_agent_id"],
                    "target": edge["to_agent_id"],
                    "label": f"{edge['handoff_count']} handoffs",
                    **edge
                }
            })

        return {"elements": elements}

    def _export_d3(self, graph: CoordinationGraph) -> Dict[str, any]:
        """Export for D3.js."""
        return {
            "nodes": [
                {"id": node["agent_id"], "name": node["agent_name"], **node}
                for node in graph.nodes
            ],
            "links": [
                {
                    "source": edge["from_agent_id"],
                    "target": edge["to_agent_id"],
                    "value": edge["handoff_count"],
                    **edge
                }
                for edge in graph.edges
            ]
        }
