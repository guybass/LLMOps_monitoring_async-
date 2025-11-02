"""
Data models for repository topology analysis.

Represents code structure, dependencies, and call relationships.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum


class ImportType(Enum):
    """Type of import statement."""
    IMPORT = "import"              # import module
    FROM_IMPORT = "from_import"    # from module import X
    RELATIVE = "relative"           # from . import X


class FunctionType(Enum):
    """Type of function."""
    FUNCTION = "function"           # def func()
    METHOD = "method"               # class method
    ASYNC_FUNCTION = "async_function"  # async def func()
    ASYNC_METHOD = "async_method"   # async class method
    LAMBDA = "lambda"               # lambda expression
    PROPERTY = "property"           # @property decorated
    STATIC_METHOD = "static_method" # @staticmethod
    CLASS_METHOD = "class_method"   # @classmethod


@dataclass
class ImportStatement:
    """Represents an import in code."""
    module_name: str
    imported_names: List[str] = field(default_factory=list)
    import_type: ImportType = ImportType.IMPORT
    alias: Optional[str] = None
    level: int = 0  # For relative imports (0 = absolute, 1 = ., 2 = .., etc.)
    line_number: int = 0


@dataclass
class FunctionInfo:
    """Information about a function or method."""
    function_id: UUID = field(default_factory=uuid4)
    name: str = ""
    qualified_name: str = ""  # Full path: module.Class.method
    function_type: FunctionType = FunctionType.FUNCTION

    # Location
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0

    # Signature
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)

    # Documentation
    docstring: Optional[str] = None

    # Metrics
    complexity: float = 0.0  # Cyclomatic complexity
    line_count: int = 0
    is_async: bool = False

    # Calls made by this function
    calls: List[str] = field(default_factory=list)  # Qualified names of called functions

    # Parent class (if method)
    parent_class: Optional[str] = None


@dataclass
class ClassInfo:
    """Information about a class."""
    class_id: UUID = field(default_factory=uuid4)
    name: str = ""
    qualified_name: str = ""

    # Location
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0

    # Inheritance
    base_classes: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)

    # Documentation
    docstring: Optional[str] = None

    # Members
    methods: List[FunctionInfo] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)

    # Metrics
    line_count: int = 0
    method_count: int = 0


@dataclass
class ModuleInfo:
    """Information about a Python module."""
    module_id: UUID = field(default_factory=uuid4)
    name: str = ""
    module_path: str = ""  # Dot-separated: package.subpackage.module
    file_path: str = ""    # Absolute file path

    # Imports
    imports: List[ImportStatement] = field(default_factory=list)

    # Code elements
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)

    # Documentation
    docstring: Optional[str] = None

    # Metrics
    line_count: int = 0
    complexity_score: float = 0.0
    import_count: int = 0

    # Dependencies (derived from imports)
    dependencies: Set[str] = field(default_factory=set)

    # Metadata
    last_modified: Optional[datetime] = None


@dataclass
class DependencyEdge:
    """Represents a dependency between modules."""
    from_module: str  # Module path (dot-separated)
    to_module: str    # Module path (dot-separated)
    import_type: ImportType = ImportType.IMPORT
    imported_names: List[str] = field(default_factory=list)
    is_circular: bool = False
    weight: int = 1  # Number of times this dependency appears


@dataclass
class CallEdge:
    """Represents a function call relationship."""
    caller: str       # Qualified function name
    callee: str       # Qualified function name
    call_count: int = 1
    line_number: int = 0


@dataclass
class CircularDependency:
    """Represents a circular dependency cycle."""
    cycle: List[str]  # List of module paths forming the cycle
    cycle_length: int = 0

    def __post_init__(self):
        self.cycle_length = len(self.cycle)


@dataclass
class TopologyMetrics:
    """Overall metrics for repository topology."""
    total_modules: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_lines: int = 0

    # Dependencies
    total_dependencies: int = 0
    circular_dependencies: int = 0
    max_depth: int = 0  # Maximum dependency depth

    # Complexity
    avg_complexity: float = 0.0
    max_complexity: float = 0.0

    # Function metrics
    avg_function_length: float = 0.0
    total_calls: int = 0

    # Most complex components
    most_complex_modules: List[str] = field(default_factory=list)
    most_complex_functions: List[str] = field(default_factory=list)

    # Dependency stats
    most_imported_modules: List[str] = field(default_factory=list)
    most_dependent_modules: List[str] = field(default_factory=list)


@dataclass
class RepositoryTopology:
    """Complete topology of a Python repository."""
    topology_id: UUID = field(default_factory=uuid4)

    # Repository info
    repository_path: str = ""
    commit_hash: Optional[str] = None
    scanned_at: datetime = field(default_factory=datetime.utcnow)

    # Code structure
    modules: Dict[str, ModuleInfo] = field(default_factory=dict)  # module_path -> ModuleInfo

    # Dependency graphs
    module_dependencies: List[DependencyEdge] = field(default_factory=list)
    function_calls: List[CallEdge] = field(default_factory=list)

    # Circular dependencies
    circular_deps: List[CircularDependency] = field(default_factory=list)

    # Metrics
    metrics: TopologyMetrics = field(default_factory=TopologyMetrics)

    # Visualization data (cached)
    graph_data: Optional[Dict[str, Any]] = None


@dataclass
class TokenConsumption:
    """Token consumption data for a code component."""
    component_id: UUID  # Function or module ID
    component_type: str  # "function", "module", "class"
    component_name: str  # Qualified name

    # Token stats
    total_tokens: int = 0
    total_calls: int = 0
    avg_tokens_per_call: float = 0.0

    # Cost (if available)
    total_cost_usd: float = 0.0

    # Time range
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


@dataclass
class ComponentUsage:
    """Combined topology + usage data for a component."""
    # From topology
    function_info: FunctionInfo

    # From monitoring
    token_consumption: TokenConsumption

    # Derived metrics
    tokens_per_line: float = 0.0
    cost_per_complexity: float = 0.0
    optimization_potential: float = 0.0  # Score: 0.0 to 1.0


@dataclass
class TopologySnapshot:
    """Snapshot of repository topology at a point in time."""
    snapshot_id: UUID = field(default_factory=uuid4)
    topology: RepositoryTopology
    token_data: List[TokenConsumption] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Comparison with previous snapshot (if available)
    changes_from_previous: Optional[Dict[str, Any]] = None


@dataclass
class HotspotAnalysis:
    """Analysis of token consumption hotspots in codebase."""
    # Top consumers
    top_functions_by_tokens: List[ComponentUsage] = field(default_factory=list)
    top_modules_by_tokens: List[str] = field(default_factory=list)

    # Patterns
    high_complexity_high_tokens: List[str] = field(default_factory=list)
    optimization_candidates: List[ComponentUsage] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Total impact
    total_tokens_analyzed: int = 0
    potential_savings: float = 0.0  # Percentage
