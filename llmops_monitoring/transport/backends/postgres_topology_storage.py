"""
PostgreSQL storage for repository topology data.

Stores code structure, dependencies, and links with token consumption metrics.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
import json

import asyncpg

from llmops_monitoring.topology.models import (
    ModuleInfo,
    FunctionInfo,
    ClassInfo,
    RepositoryTopology,
    DependencyEdge,
    CallEdge,
    CircularDependency,
    TopologyMetrics,
    TokenConsumption,
    ComponentUsage,
    HotspotAnalysis
)
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class PostgresTopologyStorage:
    """
    PostgreSQL storage for topology data.

    Tables:
    - topology_snapshots: Complete topology scans
    - code_modules: Module information
    - code_functions: Function/method information
    - code_classes: Class information
    - module_dependencies: Module dependency edges
    - function_calls: Function call edges
    - circular_dependencies: Detected cycles
    - component_token_consumption: Token usage per component
    """

    def __init__(self, pool: asyncpg.Pool, schema: str = "public"):
        """
        Initialize topology storage.

        Args:
            pool: AsyncPG connection pool
            schema: Database schema name
        """
        self.pool = pool
        self.schema = schema

    async def create_topology_tables(self) -> None:
        """Create all topology-related tables."""
        await self._create_topology_snapshots_table()
        await self._create_code_modules_table()
        await self._create_code_functions_table()
        await self._create_code_classes_table()
        await self._create_module_dependencies_table()
        await self._create_function_calls_table()
        await self._create_circular_dependencies_table()
        await self._create_component_token_consumption_table()

        logger.info(f"Created topology tables in schema '{self.schema}'")

    async def _create_topology_snapshots_table(self) -> None:
        """Create topology_snapshots table."""
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.topology_snapshots (
                    snapshot_id UUID PRIMARY KEY,
                    repository_path TEXT NOT NULL,
                    commit_hash TEXT,
                    scanned_at TIMESTAMP NOT NULL DEFAULT NOW(),

                    -- Metrics summary
                    total_modules INTEGER NOT NULL DEFAULT 0,
                    total_functions INTEGER NOT NULL DEFAULT 0,
                    total_classes INTEGER NOT NULL DEFAULT 0,
                    total_lines INTEGER NOT NULL DEFAULT 0,
                    total_dependencies INTEGER NOT NULL DEFAULT 0,
                    circular_dependencies INTEGER NOT NULL DEFAULT 0,
                    avg_complexity FLOAT DEFAULT 0.0,
                    max_complexity FLOAT DEFAULT 0.0,

                    -- Full topology data (JSONB for flexibility)
                    metrics_data JSONB,

                    -- Visualization cache
                    visualization_data JSONB,

                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_topology_snapshots_repo
                    ON {self.schema}.topology_snapshots(repository_path);
                CREATE INDEX IF NOT EXISTS idx_topology_snapshots_scanned
                    ON {self.schema}.topology_snapshots(scanned_at DESC);
                CREATE INDEX IF NOT EXISTS idx_topology_snapshots_commit
                    ON {self.schema}.topology_snapshots(commit_hash);
            """)

    async def _create_code_modules_table(self) -> None:
        """Create code_modules table."""
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.code_modules (
                    module_id UUID PRIMARY KEY,
                    snapshot_id UUID NOT NULL REFERENCES {self.schema}.topology_snapshots(snapshot_id) ON DELETE CASCADE,

                    -- Module identity
                    name TEXT NOT NULL,
                    module_path TEXT NOT NULL,
                    file_path TEXT NOT NULL,

                    -- Documentation
                    docstring TEXT,

                    -- Metrics
                    line_count INTEGER DEFAULT 0,
                    complexity_score FLOAT DEFAULT 0.0,
                    import_count INTEGER DEFAULT 0,
                    function_count INTEGER DEFAULT 0,
                    class_count INTEGER DEFAULT 0,

                    -- Dependencies (array of module paths)
                    dependencies TEXT[],

                    -- Metadata
                    last_modified TIMESTAMP,

                    -- Full module data
                    module_data JSONB,

                    created_at TIMESTAMP DEFAULT NOW(),

                    UNIQUE(snapshot_id, module_path)
                );

                CREATE INDEX IF NOT EXISTS idx_code_modules_snapshot
                    ON {self.schema}.code_modules(snapshot_id);
                CREATE INDEX IF NOT EXISTS idx_code_modules_path
                    ON {self.schema}.code_modules(module_path);
                CREATE INDEX IF NOT EXISTS idx_code_modules_complexity
                    ON {self.schema}.code_modules(complexity_score DESC);
            """)

    async def _create_code_functions_table(self) -> None:
        """Create code_functions table."""
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.code_functions (
                    function_id UUID PRIMARY KEY,
                    module_id UUID NOT NULL REFERENCES {self.schema}.code_modules(module_id) ON DELETE CASCADE,
                    snapshot_id UUID NOT NULL REFERENCES {self.schema}.topology_snapshots(snapshot_id) ON DELETE CASCADE,

                    -- Function identity
                    name TEXT NOT NULL,
                    qualified_name TEXT NOT NULL,
                    function_type TEXT NOT NULL,

                    -- Location
                    file_path TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,

                    -- Signature
                    parameters TEXT[],
                    return_type TEXT,
                    decorators TEXT[],

                    -- Documentation
                    docstring TEXT,

                    -- Metrics
                    complexity FLOAT DEFAULT 0.0,
                    line_count INTEGER DEFAULT 0,
                    is_async BOOLEAN DEFAULT FALSE,

                    -- Calls (array of qualified names)
                    calls TEXT[],

                    -- Parent (if method)
                    parent_class TEXT,

                    -- Full function data
                    function_data JSONB,

                    created_at TIMESTAMP DEFAULT NOW(),

                    UNIQUE(snapshot_id, qualified_name)
                );

                CREATE INDEX IF NOT EXISTS idx_code_functions_module
                    ON {self.schema}.code_functions(module_id);
                CREATE INDEX IF NOT EXISTS idx_code_functions_snapshot
                    ON {self.schema}.code_functions(snapshot_id);
                CREATE INDEX IF NOT EXISTS idx_code_functions_qualified
                    ON {self.schema}.code_functions(qualified_name);
                CREATE INDEX IF NOT EXISTS idx_code_functions_complexity
                    ON {self.schema}.code_functions(complexity DESC);
            """)

    async def _create_code_classes_table(self) -> None:
        """Create code_classes table."""
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.code_classes (
                    class_id UUID PRIMARY KEY,
                    module_id UUID NOT NULL REFERENCES {self.schema}.code_modules(module_id) ON DELETE CASCADE,
                    snapshot_id UUID NOT NULL REFERENCES {self.schema}.topology_snapshots(snapshot_id) ON DELETE CASCADE,

                    -- Class identity
                    name TEXT NOT NULL,
                    qualified_name TEXT NOT NULL,

                    -- Location
                    file_path TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,

                    -- Inheritance
                    base_classes TEXT[],
                    decorators TEXT[],

                    -- Documentation
                    docstring TEXT,

                    -- Metrics
                    line_count INTEGER DEFAULT 0,
                    method_count INTEGER DEFAULT 0,

                    -- Full class data
                    class_data JSONB,

                    created_at TIMESTAMP DEFAULT NOW(),

                    UNIQUE(snapshot_id, qualified_name)
                );

                CREATE INDEX IF NOT EXISTS idx_code_classes_module
                    ON {self.schema}.code_classes(module_id);
                CREATE INDEX IF NOT EXISTS idx_code_classes_snapshot
                    ON {self.schema}.code_classes(snapshot_id);
                CREATE INDEX IF NOT EXISTS idx_code_classes_qualified
                    ON {self.schema}.code_classes(qualified_name);
            """)

    async def _create_module_dependencies_table(self) -> None:
        """Create module_dependencies table."""
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.module_dependencies (
                    id SERIAL PRIMARY KEY,
                    snapshot_id UUID NOT NULL REFERENCES {self.schema}.topology_snapshots(snapshot_id) ON DELETE CASCADE,

                    -- Dependency edge
                    from_module TEXT NOT NULL,
                    to_module TEXT NOT NULL,

                    -- Import details
                    import_type TEXT NOT NULL,
                    imported_names TEXT[],

                    -- Properties
                    is_circular BOOLEAN DEFAULT FALSE,
                    weight INTEGER DEFAULT 1,

                    created_at TIMESTAMP DEFAULT NOW(),

                    UNIQUE(snapshot_id, from_module, to_module)
                );

                CREATE INDEX IF NOT EXISTS idx_module_dependencies_snapshot
                    ON {self.schema}.module_dependencies(snapshot_id);
                CREATE INDEX IF NOT EXISTS idx_module_dependencies_from
                    ON {self.schema}.module_dependencies(from_module);
                CREATE INDEX IF NOT EXISTS idx_module_dependencies_to
                    ON {self.schema}.module_dependencies(to_module);
                CREATE INDEX IF NOT EXISTS idx_module_dependencies_circular
                    ON {self.schema}.module_dependencies(is_circular) WHERE is_circular = TRUE;
            """)

    async def _create_function_calls_table(self) -> None:
        """Create function_calls table."""
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.function_calls (
                    id SERIAL PRIMARY KEY,
                    snapshot_id UUID NOT NULL REFERENCES {self.schema}.topology_snapshots(snapshot_id) ON DELETE CASCADE,

                    -- Call edge
                    caller TEXT NOT NULL,
                    callee TEXT NOT NULL,

                    -- Call details
                    call_count INTEGER DEFAULT 1,
                    line_number INTEGER,

                    created_at TIMESTAMP DEFAULT NOW(),

                    UNIQUE(snapshot_id, caller, callee)
                );

                CREATE INDEX IF NOT EXISTS idx_function_calls_snapshot
                    ON {self.schema}.function_calls(snapshot_id);
                CREATE INDEX IF NOT EXISTS idx_function_calls_caller
                    ON {self.schema}.function_calls(caller);
                CREATE INDEX IF NOT EXISTS idx_function_calls_callee
                    ON {self.schema}.function_calls(callee);
            """)

    async def _create_circular_dependencies_table(self) -> None:
        """Create circular_dependencies table."""
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.circular_dependencies (
                    id SERIAL PRIMARY KEY,
                    snapshot_id UUID NOT NULL REFERENCES {self.schema}.topology_snapshots(snapshot_id) ON DELETE CASCADE,

                    -- Cycle information
                    cycle TEXT[] NOT NULL,
                    cycle_length INTEGER NOT NULL,

                    created_at TIMESTAMP DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_circular_dependencies_snapshot
                    ON {self.schema}.circular_dependencies(snapshot_id);
            """)

    async def _create_component_token_consumption_table(self) -> None:
        """Create component_token_consumption table for linking topology with metrics."""
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.component_token_consumption (
                    id SERIAL PRIMARY KEY,

                    -- Link to topology
                    snapshot_id UUID REFERENCES {self.schema}.topology_snapshots(snapshot_id) ON DELETE CASCADE,
                    component_type TEXT NOT NULL,  -- 'function', 'module', 'class'
                    component_name TEXT NOT NULL,  -- Qualified name

                    -- Token statistics (aggregated from metric_events)
                    total_tokens INTEGER DEFAULT 0,
                    total_calls INTEGER DEFAULT 0,
                    avg_tokens_per_call FLOAT DEFAULT 0.0,

                    -- Cost data
                    total_cost_usd FLOAT DEFAULT 0.0,

                    -- Time range
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,

                    -- Derived insights
                    tokens_per_line FLOAT DEFAULT 0.0,
                    cost_per_complexity FLOAT DEFAULT 0.0,
                    optimization_potential FLOAT DEFAULT 0.0,  -- 0.0 to 1.0

                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),

                    UNIQUE(snapshot_id, component_type, component_name)
                );

                CREATE INDEX IF NOT EXISTS idx_component_token_snapshot
                    ON {self.schema}.component_token_consumption(snapshot_id);
                CREATE INDEX IF NOT EXISTS idx_component_token_name
                    ON {self.schema}.component_token_consumption(component_name);
                CREATE INDEX IF NOT EXISTS idx_component_token_total
                    ON {self.schema}.component_token_consumption(total_tokens DESC);
                CREATE INDEX IF NOT EXISTS idx_component_token_optimization
                    ON {self.schema}.component_token_consumption(optimization_potential DESC);
            """)

    async def save_topology_snapshot(self, topology: RepositoryTopology) -> UUID:
        """
        Save complete topology snapshot.

        Args:
            topology: Repository topology to save

        Returns:
            Snapshot ID
        """
        logger.info(f"Saving topology snapshot for {topology.repository_path}...")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Save snapshot metadata
                snapshot_id = topology.topology_id

                await conn.execute(f"""
                    INSERT INTO {self.schema}.topology_snapshots (
                        snapshot_id, repository_path, commit_hash, scanned_at,
                        total_modules, total_functions, total_classes, total_lines,
                        total_dependencies, circular_dependencies,
                        avg_complexity, max_complexity,
                        metrics_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (snapshot_id) DO UPDATE SET
                        repository_path = EXCLUDED.repository_path,
                        commit_hash = EXCLUDED.commit_hash,
                        scanned_at = EXCLUDED.scanned_at,
                        total_modules = EXCLUDED.total_modules,
                        total_functions = EXCLUDED.total_functions,
                        total_classes = EXCLUDED.total_classes,
                        total_lines = EXCLUDED.total_lines,
                        total_dependencies = EXCLUDED.total_dependencies,
                        circular_dependencies = EXCLUDED.circular_dependencies,
                        avg_complexity = EXCLUDED.avg_complexity,
                        max_complexity = EXCLUDED.max_complexity,
                        metrics_data = EXCLUDED.metrics_data,
                        updated_at = NOW()
                """,
                    snapshot_id,
                    topology.repository_path,
                    topology.commit_hash,
                    topology.scanned_at,
                    topology.metrics.total_modules,
                    topology.metrics.total_functions,
                    topology.metrics.total_classes,
                    topology.metrics.total_lines,
                    topology.metrics.total_dependencies,
                    topology.metrics.circular_dependencies,
                    topology.metrics.avg_complexity,
                    topology.metrics.max_complexity,
                    json.dumps(self._metrics_to_dict(topology.metrics))
                )

                # Save modules
                for module_path, module_info in topology.modules.items():
                    await self._save_module(conn, snapshot_id, module_info)

                # Save dependencies
                for dep in topology.module_dependencies:
                    await self._save_dependency(conn, snapshot_id, dep)

                # Save function calls
                for call in topology.function_calls:
                    await self._save_function_call(conn, snapshot_id, call)

                # Save circular dependencies
                for circ in topology.circular_deps:
                    await self._save_circular_dependency(conn, snapshot_id, circ)

        logger.info(f"Topology snapshot saved: {snapshot_id}")
        return snapshot_id

    async def _save_module(
        self,
        conn: asyncpg.Connection,
        snapshot_id: UUID,
        module: ModuleInfo
    ) -> None:
        """Save module information."""
        await conn.execute(f"""
            INSERT INTO {self.schema}.code_modules (
                module_id, snapshot_id, name, module_path, file_path,
                docstring, line_count, complexity_score, import_count,
                function_count, class_count, dependencies, last_modified,
                module_data
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (snapshot_id, module_path) DO UPDATE SET
                module_id = EXCLUDED.module_id,
                name = EXCLUDED.name,
                file_path = EXCLUDED.file_path,
                docstring = EXCLUDED.docstring,
                line_count = EXCLUDED.line_count,
                complexity_score = EXCLUDED.complexity_score,
                import_count = EXCLUDED.import_count,
                function_count = EXCLUDED.function_count,
                class_count = EXCLUDED.class_count,
                dependencies = EXCLUDED.dependencies,
                last_modified = EXCLUDED.last_modified,
                module_data = EXCLUDED.module_data
        """,
            module.module_id,
            snapshot_id,
            module.name,
            module.module_path,
            module.file_path,
            module.docstring,
            module.line_count,
            module.complexity_score,
            module.import_count,
            len(module.functions),
            len(module.classes),
            list(module.dependencies),
            module.last_modified,
            json.dumps(self._module_to_dict(module))
        )

        # Save functions
        for func in module.functions:
            await self._save_function(conn, snapshot_id, module.module_id, func)

        # Save classes
        for cls in module.classes:
            await self._save_class(conn, snapshot_id, module.module_id, cls)

    async def _save_function(
        self,
        conn: asyncpg.Connection,
        snapshot_id: UUID,
        module_id: UUID,
        func: FunctionInfo
    ) -> None:
        """Save function information."""
        await conn.execute(f"""
            INSERT INTO {self.schema}.code_functions (
                function_id, module_id, snapshot_id, name, qualified_name,
                function_type, file_path, line_start, line_end,
                parameters, return_type, decorators, docstring,
                complexity, line_count, is_async, calls, parent_class,
                function_data
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
            ON CONFLICT (snapshot_id, qualified_name) DO UPDATE SET
                function_id = EXCLUDED.function_id,
                module_id = EXCLUDED.module_id,
                name = EXCLUDED.name,
                function_type = EXCLUDED.function_type,
                file_path = EXCLUDED.file_path,
                line_start = EXCLUDED.line_start,
                line_end = EXCLUDED.line_end,
                parameters = EXCLUDED.parameters,
                return_type = EXCLUDED.return_type,
                decorators = EXCLUDED.decorators,
                docstring = EXCLUDED.docstring,
                complexity = EXCLUDED.complexity,
                line_count = EXCLUDED.line_count,
                is_async = EXCLUDED.is_async,
                calls = EXCLUDED.calls,
                parent_class = EXCLUDED.parent_class,
                function_data = EXCLUDED.function_data
        """,
            func.function_id,
            module_id,
            snapshot_id,
            func.name,
            func.qualified_name,
            func.function_type.value,
            func.file_path,
            func.line_start,
            func.line_end,
            func.parameters,
            func.return_type,
            func.decorators,
            func.docstring,
            func.complexity,
            func.line_count,
            func.is_async,
            func.calls,
            func.parent_class,
            json.dumps(self._function_to_dict(func))
        )

    async def _save_class(
        self,
        conn: asyncpg.Connection,
        snapshot_id: UUID,
        module_id: UUID,
        cls: ClassInfo
    ) -> None:
        """Save class information."""
        await conn.execute(f"""
            INSERT INTO {self.schema}.code_classes (
                class_id, module_id, snapshot_id, name, qualified_name,
                file_path, line_start, line_end, base_classes, decorators,
                docstring, line_count, method_count, class_data
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (snapshot_id, qualified_name) DO UPDATE SET
                class_id = EXCLUDED.class_id,
                module_id = EXCLUDED.module_id,
                name = EXCLUDED.name,
                file_path = EXCLUDED.file_path,
                line_start = EXCLUDED.line_start,
                line_end = EXCLUDED.line_end,
                base_classes = EXCLUDED.base_classes,
                decorators = EXCLUDED.decorators,
                docstring = EXCLUDED.docstring,
                line_count = EXCLUDED.line_count,
                method_count = EXCLUDED.method_count,
                class_data = EXCLUDED.class_data
        """,
            cls.class_id,
            module_id,
            snapshot_id,
            cls.name,
            cls.qualified_name,
            cls.file_path,
            cls.line_start,
            cls.line_end,
            cls.base_classes,
            cls.decorators,
            cls.docstring,
            cls.line_count,
            cls.method_count,
            json.dumps(self._class_to_dict(cls))
        )

        # Save methods as functions
        for method in cls.methods:
            await self._save_function(conn, snapshot_id, module_id, method)

    async def _save_dependency(
        self,
        conn: asyncpg.Connection,
        snapshot_id: UUID,
        dep: DependencyEdge
    ) -> None:
        """Save module dependency."""
        await conn.execute(f"""
            INSERT INTO {self.schema}.module_dependencies (
                snapshot_id, from_module, to_module, import_type,
                imported_names, is_circular, weight
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (snapshot_id, from_module, to_module) DO UPDATE SET
                import_type = EXCLUDED.import_type,
                imported_names = EXCLUDED.imported_names,
                is_circular = EXCLUDED.is_circular,
                weight = EXCLUDED.weight
        """,
            snapshot_id,
            dep.from_module,
            dep.to_module,
            dep.import_type.value,
            dep.imported_names,
            dep.is_circular,
            dep.weight
        )

    async def _save_function_call(
        self,
        conn: asyncpg.Connection,
        snapshot_id: UUID,
        call: CallEdge
    ) -> None:
        """Save function call."""
        await conn.execute(f"""
            INSERT INTO {self.schema}.function_calls (
                snapshot_id, caller, callee, call_count, line_number
            ) VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (snapshot_id, caller, callee) DO UPDATE SET
                call_count = EXCLUDED.call_count,
                line_number = EXCLUDED.line_number
        """,
            snapshot_id,
            call.caller,
            call.callee,
            call.call_count,
            call.line_number
        )

    async def _save_circular_dependency(
        self,
        conn: asyncpg.Connection,
        snapshot_id: UUID,
        circ: CircularDependency
    ) -> None:
        """Save circular dependency."""
        await conn.execute(f"""
            INSERT INTO {self.schema}.circular_dependencies (
                snapshot_id, cycle, cycle_length
            ) VALUES ($1, $2, $3)
        """,
            snapshot_id,
            circ.cycle,
            circ.cycle_length
        )

    async def get_latest_snapshot(self, repository_path: str) -> Optional[Dict[str, Any]]:
        """Get latest topology snapshot for a repository."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT * FROM {self.schema}.topology_snapshots
                WHERE repository_path = $1
                ORDER BY scanned_at DESC
                LIMIT 1
            """, repository_path)

            if row:
                return dict(row)
            return None

    async def save_token_consumption(
        self,
        snapshot_id: UUID,
        component_type: str,
        component_name: str,
        total_tokens: int,
        total_calls: int,
        avg_tokens_per_call: float,
        total_cost_usd: float = 0.0,
        first_seen: Optional[datetime] = None,
        last_seen: Optional[datetime] = None
    ) -> None:
        """
        Save or update token consumption data for a component.

        Args:
            snapshot_id: Topology snapshot ID
            component_type: 'function', 'module', or 'class'
            component_name: Qualified name
            total_tokens: Total tokens consumed
            total_calls: Total number of calls
            avg_tokens_per_call: Average tokens per call
            total_cost_usd: Total cost in USD
            first_seen: First observation timestamp
            last_seen: Last observation timestamp
        """
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self.schema}.component_token_consumption (
                    snapshot_id, component_type, component_name,
                    total_tokens, total_calls, avg_tokens_per_call,
                    total_cost_usd, first_seen, last_seen
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (snapshot_id, component_type, component_name) DO UPDATE SET
                    total_tokens = EXCLUDED.total_tokens,
                    total_calls = EXCLUDED.total_calls,
                    avg_tokens_per_call = EXCLUDED.avg_tokens_per_call,
                    total_cost_usd = EXCLUDED.total_cost_usd,
                    first_seen = COALESCE(EXCLUDED.first_seen, {self.schema}.component_token_consumption.first_seen),
                    last_seen = EXCLUDED.last_seen,
                    updated_at = NOW()
            """,
                snapshot_id,
                component_type,
                component_name,
                total_tokens,
                total_calls,
                avg_tokens_per_call,
                total_cost_usd,
                first_seen,
                last_seen
            )

    async def get_top_token_consumers(
        self,
        snapshot_id: UUID,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get top token consuming components."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT * FROM {self.schema}.component_token_consumption
                WHERE snapshot_id = $1
                ORDER BY total_tokens DESC
                LIMIT $2
            """, snapshot_id, limit)

            return [dict(row) for row in rows]

    def _metrics_to_dict(self, metrics: TopologyMetrics) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_modules": metrics.total_modules,
            "total_functions": metrics.total_functions,
            "total_classes": metrics.total_classes,
            "total_lines": metrics.total_lines,
            "total_dependencies": metrics.total_dependencies,
            "circular_dependencies": metrics.circular_dependencies,
            "max_depth": metrics.max_depth,
            "avg_complexity": metrics.avg_complexity,
            "max_complexity": metrics.max_complexity,
            "avg_function_length": metrics.avg_function_length,
            "total_calls": metrics.total_calls,
            "most_complex_modules": metrics.most_complex_modules,
            "most_complex_functions": metrics.most_complex_functions,
            "most_imported_modules": metrics.most_imported_modules,
            "most_dependent_modules": metrics.most_dependent_modules
        }

    def _module_to_dict(self, module: ModuleInfo) -> Dict[str, Any]:
        """Convert module to dictionary (lightweight)."""
        return {
            "name": module.name,
            "module_path": module.module_path,
            "file_path": module.file_path,
            "line_count": module.line_count,
            "complexity_score": module.complexity_score,
            "import_count": module.import_count
        }

    def _function_to_dict(self, func: FunctionInfo) -> Dict[str, Any]:
        """Convert function to dictionary (lightweight)."""
        return {
            "name": func.name,
            "qualified_name": func.qualified_name,
            "function_type": func.function_type.value,
            "complexity": func.complexity,
            "line_count": func.line_count,
            "is_async": func.is_async
        }

    def _class_to_dict(self, cls: ClassInfo) -> Dict[str, Any]:
        """Convert class to dictionary (lightweight)."""
        return {
            "name": cls.name,
            "qualified_name": cls.qualified_name,
            "line_count": cls.line_count,
            "method_count": cls.method_count
        }
