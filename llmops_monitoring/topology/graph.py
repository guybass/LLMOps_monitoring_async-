"""
Dependency graph builder using NetworkX.

Constructs module dependency graphs and call graphs from parsed code.
"""

from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque

from llmops_monitoring.topology.models import (
    ModuleInfo,
    DependencyEdge,
    CallEdge,
    CircularDependency,
    RepositoryTopology,
    TopologyMetrics
)
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


# NetworkX is optional - gracefully degrade if not available
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available. Install with: pip install networkx")


class DependencyGraphBuilder:
    """
    Build dependency graphs from parsed modules.

    Creates:
    - Module dependency graph
    - Function call graph
    - Detects circular dependencies
    - Calculates graph metrics
    """

    def __init__(self):
        """Initialize graph builder."""
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "NetworkX is required for topology analysis. "
                "Install with: pip install networkx"
            )

        self.module_graph: Optional[nx.DiGraph] = None
        self.call_graph: Optional[nx.DiGraph] = None

    def build_repository_topology(
        self,
        modules: Dict[str, ModuleInfo]
    ) -> RepositoryTopology:
        """
        Build complete repository topology.

        Args:
            modules: Dictionary of module_path -> ModuleInfo

        Returns:
            RepositoryTopology with all graphs and metrics
        """
        topology = RepositoryTopology()
        topology.modules = modules

        # Build module dependency graph
        logger.info("Building module dependency graph...")
        topology.module_dependencies = self.build_module_dependencies(modules)
        self.module_graph = self._dependencies_to_graph(topology.module_dependencies)

        # Build function call graph
        logger.info("Building function call graph...")
        topology.function_calls = self.build_call_graph(modules)
        self.call_graph = self._calls_to_graph(topology.function_calls)

        # Detect circular dependencies
        logger.info("Detecting circular dependencies...")
        topology.circular_deps = self.detect_circular_dependencies(topology.module_dependencies)

        # Calculate metrics
        logger.info("Calculating topology metrics...")
        topology.metrics = self.calculate_metrics(topology)

        logger.info(
            f"Topology complete: {topology.metrics.total_modules} modules, "
            f"{topology.metrics.total_functions} functions, "
            f"{topology.metrics.circular_dependencies} circular dependencies"
        )

        return topology

    def build_module_dependencies(
        self,
        modules: Dict[str, ModuleInfo]
    ) -> List[DependencyEdge]:
        """
        Build module dependency edges.

        Args:
            modules: Dictionary of modules

        Returns:
            List of dependency edges
        """
        edges = []
        module_set = set(modules.keys())

        for module_path, module_info in modules.items():
            for imp in module_info.imports:
                # Resolve import to module path
                target_module = self._resolve_import(imp.module_name, module_path, module_set)

                if target_module:
                    edge = DependencyEdge(
                        from_module=module_path,
                        to_module=target_module,
                        import_type=imp.import_type,
                        imported_names=imp.imported_names
                    )
                    edges.append(edge)

        # Deduplicate and aggregate
        edges = self._aggregate_edges(edges)

        logger.debug(f"Built {len(edges)} module dependency edges")

        return edges

    def build_call_graph(
        self,
        modules: Dict[str, ModuleInfo]
    ) -> List[CallEdge]:
        """
        Build function call graph.

        Args:
            modules: Dictionary of modules

        Returns:
            List of call edges
        """
        edges = []

        # Build function registry (qualified_name -> FunctionInfo)
        function_registry = {}
        for module_info in modules.values():
            for func in module_info.functions:
                function_registry[func.qualified_name] = func
            for cls in module_info.classes:
                for method in cls.methods:
                    function_registry[method.qualified_name] = method

        # Build call edges
        for qualified_name, func_info in function_registry.items():
            for call_name in func_info.calls:
                # Try to resolve call to qualified name
                callee = self._resolve_call(call_name, func_info, function_registry)

                if callee:
                    edge = CallEdge(
                        caller=qualified_name,
                        callee=callee,
                        line_number=func_info.line_start
                    )
                    edges.append(edge)

        logger.debug(f"Built {len(edges)} function call edges")

        return edges

    def detect_circular_dependencies(
        self,
        dependencies: List[DependencyEdge]
    ) -> List[CircularDependency]:
        """
        Detect circular dependencies using cycle detection.

        Args:
            dependencies: List of dependency edges

        Returns:
            List of circular dependency cycles
        """
        if not NETWORKX_AVAILABLE:
            return []

        # Build graph
        graph = self._dependencies_to_graph(dependencies)

        # Find all cycles
        try:
            cycles = list(nx.simple_cycles(graph))
        except nx.NetworkXNoCycle:
            cycles = []

        # Convert to CircularDependency objects
        circular_deps = []
        for cycle in cycles:
            circular_deps.append(CircularDependency(cycle=cycle))

        # Mark edges as circular
        for circ in circular_deps:
            for i in range(len(circ.cycle)):
                from_mod = circ.cycle[i]
                to_mod = circ.cycle[(i + 1) % len(circ.cycle)]

                # Find and mark edge
                for edge in dependencies:
                    if edge.from_module == from_mod and edge.to_module == to_mod:
                        edge.is_circular = True

        if circular_deps:
            logger.warning(f"Found {len(circular_deps)} circular dependency cycles")

        return circular_deps

    def calculate_metrics(
        self,
        topology: RepositoryTopology
    ) -> TopologyMetrics:
        """
        Calculate overall topology metrics.

        Args:
            topology: Repository topology

        Returns:
            Topology metrics
        """
        metrics = TopologyMetrics()

        # Count modules, functions, classes
        metrics.total_modules = len(topology.modules)

        total_functions = 0
        total_classes = 0
        total_lines = 0
        complexities = []

        for module_info in topology.modules.values():
            total_functions += len(module_info.functions)
            total_classes += len(module_info.classes)
            total_lines += module_info.line_count

            # Collect complexity scores
            if module_info.complexity_score > 0:
                complexities.append(module_info.complexity_score)

            for func in module_info.functions:
                complexities.append(func.complexity)

            for cls in module_info.classes:
                for method in cls.methods:
                    complexities.append(method.complexity)

        metrics.total_functions = total_functions
        metrics.total_classes = total_classes
        metrics.total_lines = total_lines

        # Complexity metrics
        if complexities:
            metrics.avg_complexity = sum(complexities) / len(complexities)
            metrics.max_complexity = max(complexities)

        # Dependency metrics
        metrics.total_dependencies = len(topology.module_dependencies)
        metrics.circular_dependencies = len(topology.circular_deps)

        # Calculate max depth using graph
        if self.module_graph:
            try:
                # Find longest path in DAG
                if nx.is_directed_acyclic_graph(self.module_graph):
                    metrics.max_depth = nx.dag_longest_path_length(self.module_graph)
                else:
                    # Has cycles, use approximation
                    metrics.max_depth = self._estimate_max_depth(self.module_graph)
            except:
                metrics.max_depth = 0

        # Call graph metrics
        metrics.total_calls = len(topology.function_calls)

        # Find most complex components
        metrics.most_complex_modules = self._find_most_complex_modules(topology.modules, top_n=10)
        metrics.most_complex_functions = self._find_most_complex_functions(topology.modules, top_n=10)

        # Find most imported modules
        metrics.most_imported_modules = self._find_most_imported_modules(topology.module_dependencies, top_n=10)
        metrics.most_dependent_modules = self._find_most_dependent_modules(topology.module_dependencies, top_n=10)

        return metrics

    def _resolve_import(
        self,
        import_name: str,
        from_module: str,
        available_modules: Set[str]
    ) -> Optional[str]:
        """
        Resolve import name to actual module path.

        Args:
            import_name: Name in import statement
            from_module: Module making the import
            available_modules: Set of available module paths

        Returns:
            Resolved module path or None
        """
        # Direct match
        if import_name in available_modules:
            return import_name

        # Check if it's a submodule of available modules
        for module in available_modules:
            if module.startswith(import_name + '.'):
                return module

        # Check relative imports (same package)
        from_package = '.'.join(from_module.split('.')[:-1])
        if from_package:
            relative_path = f"{from_package}.{import_name}"
            if relative_path in available_modules:
                return relative_path

        return None

    def _resolve_call(
        self,
        call_name: str,
        caller_func,
        function_registry: Dict[str, any]
    ) -> Optional[str]:
        """
        Resolve function call name to qualified name.

        Args:
            call_name: Name used in call
            caller_func: FunctionInfo of calling function
            function_registry: Registry of all functions

        Returns:
            Qualified name of called function or None
        """
        # Try direct match (for fully qualified calls)
        if call_name in function_registry:
            return call_name

        # Try in same module
        module_path = '.'.join(caller_func.qualified_name.split('.')[:-1])
        if caller_func.parent_class:
            # Remove class name
            module_path = '.'.join(module_path.split('.')[:-1])

        same_module_call = f"{module_path}.{call_name}"
        if same_module_call in function_registry:
            return same_module_call

        # Try in same class (for methods)
        if caller_func.parent_class:
            class_path = '.'.join(caller_func.qualified_name.split('.')[:-1])
            same_class_call = f"{class_path}.{call_name}"
            if same_class_call in function_registry:
                return same_class_call

        return None

    def _aggregate_edges(
        self,
        edges: List[DependencyEdge]
    ) -> List[DependencyEdge]:
        """Aggregate duplicate edges."""
        edge_map: Dict[Tuple[str, str], DependencyEdge] = {}

        for edge in edges:
            key = (edge.from_module, edge.to_module)
            if key in edge_map:
                # Merge
                edge_map[key].weight += 1
                edge_map[key].imported_names.extend(edge.imported_names)
            else:
                edge_map[key] = edge

        return list(edge_map.values())

    def _dependencies_to_graph(
        self,
        dependencies: List[DependencyEdge]
    ) -> nx.DiGraph:
        """Convert dependency edges to NetworkX graph."""
        graph = nx.DiGraph()

        for edge in dependencies:
            graph.add_edge(
                edge.from_module,
                edge.to_module,
                weight=edge.weight,
                import_type=edge.import_type.value,
                imported_names=edge.imported_names
            )

        return graph

    def _calls_to_graph(
        self,
        calls: List[CallEdge]
    ) -> nx.DiGraph:
        """Convert call edges to NetworkX graph."""
        graph = nx.DiGraph()

        for edge in calls:
            if graph.has_edge(edge.caller, edge.callee):
                # Increment call count
                graph[edge.caller][edge.callee]['call_count'] += 1
            else:
                graph.add_edge(
                    edge.caller,
                    edge.callee,
                    call_count=edge.call_count
                )

        return graph

    def _estimate_max_depth(self, graph: nx.DiGraph) -> int:
        """Estimate max depth for graphs with cycles."""
        try:
            # Use longest simple path as approximation
            lengths = []
            for source in graph.nodes():
                for target in graph.nodes():
                    if source != target:
                        try:
                            paths = list(nx.all_simple_paths(graph, source, target, cutoff=20))
                            if paths:
                                lengths.append(max(len(p) for p in paths))
                        except:
                            continue
            return max(lengths) if lengths else 0
        except:
            return 0

    def _find_most_complex_modules(
        self,
        modules: Dict[str, ModuleInfo],
        top_n: int = 10
    ) -> List[str]:
        """Find modules with highest complexity."""
        sorted_modules = sorted(
            modules.items(),
            key=lambda x: x[1].complexity_score,
            reverse=True
        )
        return [m[0] for m in sorted_modules[:top_n]]

    def _find_most_complex_functions(
        self,
        modules: Dict[str, ModuleInfo],
        top_n: int = 10
    ) -> List[str]:
        """Find functions with highest complexity."""
        all_functions = []

        for module_info in modules.values():
            for func in module_info.functions:
                all_functions.append((func.qualified_name, func.complexity))
            for cls in module_info.classes:
                for method in cls.methods:
                    all_functions.append((method.qualified_name, method.complexity))

        sorted_functions = sorted(all_functions, key=lambda x: x[1], reverse=True)
        return [f[0] for f in sorted_functions[:top_n]]

    def _find_most_imported_modules(
        self,
        dependencies: List[DependencyEdge],
        top_n: int = 10
    ) -> List[str]:
        """Find most imported modules."""
        import_counts = defaultdict(int)

        for edge in dependencies:
            import_counts[edge.to_module] += edge.weight

        sorted_imports = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)
        return [m[0] for m in sorted_imports[:top_n]]

    def _find_most_dependent_modules(
        self,
        dependencies: List[DependencyEdge],
        top_n: int = 10
    ) -> List[str]:
        """Find modules with most dependencies."""
        dependency_counts = defaultdict(int)

        for edge in dependencies:
            dependency_counts[edge.from_module] += 1

        sorted_deps = sorted(dependency_counts.items(), key=lambda x: x[1], reverse=True)
        return [m[0] for m in sorted_deps[:top_n]]

    def get_module_info(self, module_path: str) -> Optional[Dict[str, any]]:
        """Get detailed info about a module from graph."""
        if not self.module_graph or module_path not in self.module_graph:
            return None

        # In-degree (imported by)
        in_degree = self.module_graph.in_degree(module_path)

        # Out-degree (imports)
        out_degree = self.module_graph.out_degree(module_path)

        # Neighbors
        predecessors = list(self.module_graph.predecessors(module_path))
        successors = list(self.module_graph.successors(module_path))

        return {
            "module": module_path,
            "imported_by_count": in_degree,
            "imports_count": out_degree,
            "imported_by": predecessors,
            "imports": successors
        }
