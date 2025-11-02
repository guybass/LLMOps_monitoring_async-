"""
Repository Topology Scanner.

Analyzes Python codebases to extract structure, dependencies, and call relationships.
Enables correlation with runtime token consumption for optimization insights.

Key Features:
- AST-based code parsing
- Module dependency graph construction
- Function call graph analysis
- Circular dependency detection
- Complexity metrics calculation
- Token consumption correlation
- Multiple visualization formats (D3.js, Cytoscape, GraphViz)

Example:
    from llmops_monitoring.topology import CodeParser, DependencyGraphBuilder

    # Parse repository
    parser = CodeParser()
    modules = parser.parse_repository("/path/to/repo")

    # Build dependency graph
    builder = DependencyGraphBuilder()
    topology = builder.build_repository_topology(modules)

    # Analyze
    print(f"Total modules: {topology.metrics.total_modules}")
    print(f"Circular dependencies: {topology.metrics.circular_dependencies}")
"""

from llmops_monitoring.topology.models import (
    ModuleInfo,
    FunctionInfo,
    ClassInfo,
    RepositoryTopology,
    TopologyMetrics,
    CircularDependency,
    ComponentUsage,
    HotspotAnalysis,
    TokenConsumption
)

from llmops_monitoring.topology.parser import CodeParser

from llmops_monitoring.topology.graph import DependencyGraphBuilder

from llmops_monitoring.topology.visualizer import TopologyVisualizer

from llmops_monitoring.topology.correlator import TokenConsumptionCorrelator


__all__ = [
    # Data Models
    "ModuleInfo",
    "FunctionInfo",
    "ClassInfo",
    "RepositoryTopology",
    "TopologyMetrics",
    "CircularDependency",
    "ComponentUsage",
    "HotspotAnalysis",
    "TokenConsumption",

    # Core Components
    "CodeParser",
    "DependencyGraphBuilder",
    "TopologyVisualizer",
    "TokenConsumptionCorrelator",
]

__version__ = "0.3.0"
