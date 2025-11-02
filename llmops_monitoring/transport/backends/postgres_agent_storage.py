"""
PostgreSQL storage extension for agent intelligence.

Adds tables and methods for storing agent-related data:
- Agents
- Agent Operations
- Agent Handoffs
- Coordination Graphs
- Bottlenecks
- Coalitions
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
import json

from llmops_monitoring.agent.base import (
    Agent,
    AgentOperation,
    AgentHandoff,
    CoordinationGraph,
    BottleneckInfo,
    Coalition
)


logger = logging.getLogger(__name__)


class PostgresAgentStorage:
    """
    PostgreSQL storage for agent intelligence data.

    Extends the base PostgresBackend with agent-specific tables and methods.
    """

    def __init__(self, pool, schema_name: Optional[str] = None):
        """
        Initialize agent storage.

        Args:
            pool: asyncpg connection pool
            schema_name: Optional schema name
        """
        self.pool = pool
        self.schema_name = schema_name
        self._schema_prefix = f"{schema_name}." if schema_name else ""

    async def create_agent_tables(self) -> None:
        """Create all agent-related tables."""
        await self._create_agents_table()
        await self._create_agent_operations_table()
        await self._create_agent_handoffs_table()
        await self._create_coordination_graphs_table()
        await self._create_bottlenecks_table()
        await self._create_coalitions_table()

        logger.info("Created agent intelligence tables in PostgreSQL")

    async def _create_agents_table(self) -> None:
        """Create agents table."""
        table_name = f"{self._schema_prefix}agents"

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            agent_id UUID PRIMARY KEY,
            agent_name VARCHAR(255) NOT NULL,
            agent_role VARCHAR(100),
            agent_type VARCHAR(50) NOT NULL,

            -- Capabilities
            capabilities JSONB DEFAULT '[]',
            input_schema JSONB,
            output_schema JSONB,

            -- Performance metrics
            total_invocations INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            avg_latency_ms FLOAT DEFAULT 0.0,
            avg_cost_usd DECIMAL(10, 6) DEFAULT 0.0,

            -- Relationships
            can_handoff_to JSONB DEFAULT '[]',
            reports_to VARCHAR(255),

            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Constraints
            UNIQUE(agent_name)
        );
        """

        # Create indexes
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_agents_name ON {table_name}(agent_name);",
            f"CREATE INDEX IF NOT EXISTS idx_agents_type ON {table_name}(agent_type);",
            f"CREATE INDEX IF NOT EXISTS idx_agents_last_seen ON {table_name}(last_seen DESC);",
        ]

        async with self.pool.acquire() as conn:
            await conn.execute(create_table_sql)
            for index_sql in indexes:
                await conn.execute(index_sql)

        logger.debug(f"Created agents table: {table_name}")

    async def _create_agent_operations_table(self) -> None:
        """Create agent_operations table."""
        table_name = f"{self._schema_prefix}agent_operations"

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            operation_id UUID PRIMARY KEY,
            event_id UUID NOT NULL,

            -- Agent identification
            agent_id UUID NOT NULL,
            agent_name VARCHAR(255) NOT NULL,

            -- Parent operation (for handoff tracking)
            parent_operation_id UUID,
            parent_agent_id UUID,
            parent_agent_name VARCHAR(255),

            -- Handoff tracking
            is_handoff BOOLEAN DEFAULT FALSE,
            handoff_reason TEXT,
            handoff_quality_score FLOAT,

            -- Context tracking
            input_context_hash VARCHAR(64),
            output_context_hash VARCHAR(64),
            context_similarity_to_parent FLOAT,

            -- Performance
            duration_ms FLOAT NOT NULL,
            success BOOLEAN NOT NULL,
            error TEXT,

            -- Metadata
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Foreign keys
            FOREIGN KEY (agent_id) REFERENCES {self._schema_prefix}agents(agent_id) ON DELETE CASCADE
        );
        """

        # Create indexes
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_agent_ops_agent ON {table_name}(agent_id, timestamp DESC);",
            f"CREATE INDEX IF NOT EXISTS idx_agent_ops_event ON {table_name}(event_id);",
            f"CREATE INDEX IF NOT EXISTS idx_agent_ops_parent ON {table_name}(parent_operation_id);",
            f"CREATE INDEX IF NOT EXISTS idx_agent_ops_handoff ON {table_name}(is_handoff) WHERE is_handoff = TRUE;",
            f"CREATE INDEX IF NOT EXISTS idx_agent_ops_timestamp ON {table_name}(timestamp DESC);",
        ]

        async with self.pool.acquire() as conn:
            await conn.execute(create_table_sql)
            for index_sql in indexes:
                await conn.execute(index_sql)

        logger.debug(f"Created agent_operations table: {table_name}")

    async def _create_agent_handoffs_table(self) -> None:
        """Create agent_handoffs table."""
        table_name = f"{self._schema_prefix}agent_handoffs"

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            handoff_id UUID PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            trace_id VARCHAR(255) NOT NULL,

            -- Source and target
            from_agent_id UUID NOT NULL,
            from_agent_name VARCHAR(255) NOT NULL,
            from_operation_id UUID NOT NULL,

            to_agent_id UUID NOT NULL,
            to_agent_name VARCHAR(255) NOT NULL,
            to_operation_id UUID NOT NULL,

            -- Handoff details
            handoff_timestamp TIMESTAMP NOT NULL,
            handoff_reason TEXT,
            handoff_decision_process JSONB,

            -- Quality metrics
            quality_score FLOAT NOT NULL,
            quality_level VARCHAR(20) NOT NULL,
            was_correct_agent BOOLEAN,
            handoff_efficiency FLOAT,

            -- Context passed
            context_size_bytes INTEGER,
            context_summary TEXT,
            context_retention_score FLOAT,

            -- Latency
            handoff_latency_ms FLOAT NOT NULL,

            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Foreign keys
            FOREIGN KEY (from_agent_id) REFERENCES {self._schema_prefix}agents(agent_id) ON DELETE CASCADE,
            FOREIGN KEY (to_agent_id) REFERENCES {self._schema_prefix}agents(agent_id) ON DELETE CASCADE,
            FOREIGN KEY (from_operation_id) REFERENCES {self._schema_prefix}agent_operations(operation_id) ON DELETE CASCADE,
            FOREIGN KEY (to_operation_id) REFERENCES {self._schema_prefix}agent_operations(operation_id) ON DELETE CASCADE
        );
        """

        # Create indexes
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_handoffs_session ON {table_name}(session_id, handoff_timestamp DESC);",
            f"CREATE INDEX IF NOT EXISTS idx_handoffs_trace ON {table_name}(trace_id);",
            f"CREATE INDEX IF NOT EXISTS idx_handoffs_from_agent ON {table_name}(from_agent_id, handoff_timestamp DESC);",
            f"CREATE INDEX IF NOT EXISTS idx_handoffs_to_agent ON {table_name}(to_agent_id, handoff_timestamp DESC);",
            f"CREATE INDEX IF NOT EXISTS idx_handoffs_quality ON {table_name}(quality_score);",
        ]

        async with self.pool.acquire() as conn:
            await conn.execute(create_table_sql)
            for index_sql in indexes:
                await conn.execute(index_sql)

        logger.debug(f"Created agent_handoffs table: {table_name}")

    async def _create_coordination_graphs_table(self) -> None:
        """Create coordination_graphs table."""
        table_name = f"{self._schema_prefix}coordination_graphs"

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            graph_id UUID PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,

            -- Graph structure (stored as JSON)
            nodes JSONB NOT NULL,
            edges JSONB NOT NULL,

            -- Graph metrics
            total_agents INTEGER NOT NULL,
            total_handoffs INTEGER NOT NULL,
            max_depth INTEGER DEFAULT 0,
            bottleneck_agent_id UUID,
            critical_path_ms FLOAT,

            -- Analysis results
            avg_handoff_quality FLOAT DEFAULT 0.0,
            avg_context_retention FLOAT DEFAULT 0.0,

            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Constraints
            UNIQUE(session_id)
        );
        """

        # Create indexes
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_graphs_session ON {table_name}(session_id);",
            f"CREATE INDEX IF NOT EXISTS idx_graphs_created ON {table_name}(created_at DESC);",
        ]

        async with self.pool.acquire() as conn:
            await conn.execute(create_table_sql)
            for index_sql in indexes:
                await conn.execute(index_sql)

        logger.debug(f"Created coordination_graphs table: {table_name}")

    async def _create_bottlenecks_table(self) -> None:
        """Create bottlenecks table."""
        table_name = f"{self._schema_prefix}bottlenecks"

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            bottleneck_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID NOT NULL,
            agent_name VARCHAR(255) NOT NULL,

            -- Severity
            severity VARCHAR(20) NOT NULL,
            bottleneck_score FLOAT NOT NULL,

            -- Metrics
            avg_queue_time_ms FLOAT NOT NULL,
            p95_latency_ms FLOAT NOT NULL,
            utilization FLOAT NOT NULL,
            requests_delayed INTEGER NOT NULL,

            -- Recommendations
            recommendations JSONB NOT NULL,

            -- Metadata
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id VARCHAR(255),

            -- Foreign keys
            FOREIGN KEY (agent_id) REFERENCES {self._schema_prefix}agents(agent_id) ON DELETE CASCADE
        );
        """

        # Create indexes
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_bottlenecks_agent ON {table_name}(agent_id, detected_at DESC);",
            f"CREATE INDEX IF NOT EXISTS idx_bottlenecks_severity ON {table_name}(severity, bottleneck_score DESC);",
            f"CREATE INDEX IF NOT EXISTS idx_bottlenecks_session ON {table_name}(session_id);",
        ]

        async with self.pool.acquire() as conn:
            await conn.execute(create_table_sql)
            for index_sql in indexes:
                await conn.execute(index_sql)

        logger.debug(f"Created bottlenecks table: {table_name}")

    async def _create_coalitions_table(self) -> None:
        """Create coalitions table."""
        table_name = f"{self._schema_prefix}coalitions"

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            coalition_id UUID PRIMARY KEY,
            coalition_name VARCHAR(255) NOT NULL,
            agent_names JSONB NOT NULL,

            -- Task type
            task_type VARCHAR(100) NOT NULL,

            -- Performance metrics
            total_tasks INTEGER DEFAULT 0,
            successful_tasks INTEGER DEFAULT 0,
            avg_total_latency_ms FLOAT DEFAULT 0.0,
            avg_total_cost_usd DECIMAL(10, 6) DEFAULT 0.0,

            -- Quality
            avg_user_satisfaction FLOAT,
            avg_handoff_efficiency FLOAT DEFAULT 0.0,

            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # Create indexes
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_coalitions_task_type ON {table_name}(task_type);",
            f"CREATE INDEX IF NOT EXISTS idx_coalitions_created ON {table_name}(created_at DESC);",
        ]

        async with self.pool.acquire() as conn:
            await conn.execute(create_table_sql)
            for index_sql in indexes:
                await conn.execute(index_sql)

        logger.debug(f"Created coalitions table: {table_name}")

    # ========== WRITE METHODS ==========

    async def save_agent(self, agent: Agent) -> None:
        """Save or update agent."""
        table_name = f"{self._schema_prefix}agents"

        query = f"""
        INSERT INTO {table_name} (
            agent_id, agent_name, agent_role, agent_type,
            capabilities, input_schema, output_schema,
            total_invocations, success_count, failure_count,
            avg_latency_ms, avg_cost_usd,
            can_handoff_to, reports_to,
            created_at, last_seen
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        ON CONFLICT (agent_name) DO UPDATE SET
            agent_role = EXCLUDED.agent_role,
            agent_type = EXCLUDED.agent_type,
            capabilities = EXCLUDED.capabilities,
            total_invocations = EXCLUDED.total_invocations,
            success_count = EXCLUDED.success_count,
            failure_count = EXCLUDED.failure_count,
            avg_latency_ms = EXCLUDED.avg_latency_ms,
            avg_cost_usd = EXCLUDED.avg_cost_usd,
            can_handoff_to = EXCLUDED.can_handoff_to,
            reports_to = EXCLUDED.reports_to,
            last_seen = EXCLUDED.last_seen
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                str(agent.agent_id),
                agent.agent_name,
                agent.agent_role,
                agent.agent_type.value,
                json.dumps(agent.capabilities),
                json.dumps(agent.input_schema) if agent.input_schema else None,
                json.dumps(agent.output_schema) if agent.output_schema else None,
                agent.total_invocations,
                agent.success_count,
                agent.failure_count,
                agent.avg_latency_ms,
                float(agent.avg_cost_usd),
                json.dumps(agent.can_handoff_to),
                agent.reports_to,
                agent.created_at,
                agent.last_seen
            )

        logger.debug(f"Saved agent: {agent.agent_name}")

    async def save_agent_operation(self, operation: AgentOperation) -> None:
        """Save agent operation."""
        table_name = f"{self._schema_prefix}agent_operations"

        query = f"""
        INSERT INTO {table_name} (
            operation_id, event_id, agent_id, agent_name,
            parent_operation_id, parent_agent_id, parent_agent_name,
            is_handoff, handoff_reason, handoff_quality_score,
            input_context_hash, output_context_hash, context_similarity_to_parent,
            duration_ms, success, error, timestamp
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                str(operation.operation_id),
                str(operation.event_id),
                str(operation.agent_id),
                operation.agent_name,
                str(operation.parent_operation_id) if operation.parent_operation_id else None,
                str(operation.parent_agent_id) if operation.parent_agent_id else None,
                operation.parent_agent_name,
                operation.is_handoff,
                operation.handoff_reason,
                operation.handoff_quality_score,
                operation.input_context_hash,
                operation.output_context_hash,
                operation.context_similarity_to_parent,
                operation.duration_ms,
                operation.success,
                operation.error,
                operation.timestamp
            )

        logger.debug(f"Saved agent operation: {operation.operation_id}")

    async def save_handoff(self, handoff: AgentHandoff) -> None:
        """Save agent handoff."""
        table_name = f"{self._schema_prefix}agent_handoffs"

        query = f"""
        INSERT INTO {table_name} (
            handoff_id, session_id, trace_id,
            from_agent_id, from_agent_name, from_operation_id,
            to_agent_id, to_agent_name, to_operation_id,
            handoff_timestamp, handoff_reason, handoff_decision_process,
            quality_score, quality_level, was_correct_agent, handoff_efficiency,
            context_size_bytes, context_summary, context_retention_score,
            handoff_latency_ms
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                str(handoff.handoff_id),
                handoff.session_id,
                handoff.trace_id,
                str(handoff.from_agent_id),
                handoff.from_agent_name,
                str(handoff.from_operation_id),
                str(handoff.to_agent_id),
                handoff.to_agent_name,
                str(handoff.to_operation_id),
                handoff.handoff_timestamp,
                handoff.handoff_reason,
                json.dumps(handoff.handoff_decision_process) if handoff.handoff_decision_process else None,
                handoff.quality_score,
                handoff.quality_level.value,
                handoff.was_correct_agent,
                handoff.handoff_efficiency,
                handoff.context_size_bytes,
                handoff.context_summary,
                handoff.context_retention_score,
                handoff.handoff_latency_ms
            )

        logger.debug(f"Saved handoff: {handoff.from_agent_name} → {handoff.to_agent_name}")

    async def save_coordination_graph(self, graph: CoordinationGraph) -> None:
        """Save coordination graph."""
        table_name = f"{self._schema_prefix}coordination_graphs"

        query = f"""
        INSERT INTO {table_name} (
            graph_id, session_id, nodes, edges,
            total_agents, total_handoffs, max_depth,
            bottleneck_agent_id, critical_path_ms,
            avg_handoff_quality, avg_context_retention
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (session_id) DO UPDATE SET
            nodes = EXCLUDED.nodes,
            edges = EXCLUDED.edges,
            total_agents = EXCLUDED.total_agents,
            total_handoffs = EXCLUDED.total_handoffs,
            max_depth = EXCLUDED.max_depth,
            bottleneck_agent_id = EXCLUDED.bottleneck_agent_id,
            critical_path_ms = EXCLUDED.critical_path_ms,
            avg_handoff_quality = EXCLUDED.avg_handoff_quality,
            avg_context_retention = EXCLUDED.avg_context_retention
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                str(graph.graph_id),
                graph.session_id,
                json.dumps(graph.nodes),
                json.dumps(graph.edges),
                graph.total_agents,
                graph.total_handoffs,
                graph.max_depth,
                str(graph.bottleneck_agent_id) if graph.bottleneck_agent_id else None,
                graph.critical_path_ms,
                graph.avg_handoff_quality,
                graph.avg_context_retention
            )

        logger.debug(f"Saved coordination graph for session: {graph.session_id}")

    async def save_bottleneck(self, bottleneck: BottleneckInfo, session_id: Optional[str] = None) -> None:
        """Save bottleneck detection."""
        table_name = f"{self._schema_prefix}bottlenecks"

        query = f"""
        INSERT INTO {table_name} (
            agent_id, agent_name, severity, bottleneck_score,
            avg_queue_time_ms, p95_latency_ms, utilization, requests_delayed,
            recommendations, session_id
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                str(bottleneck.agent_id),
                bottleneck.agent_name,
                bottleneck.severity,
                bottleneck.bottleneck_score,
                bottleneck.avg_queue_time_ms,
                bottleneck.p95_latency_ms,
                bottleneck.utilization,
                bottleneck.requests_delayed,
                json.dumps(bottleneck.recommendations),
                session_id
            )

        logger.debug(f"Saved bottleneck: {bottleneck.agent_name} (severity: {bottleneck.severity})")

    async def save_coalition(self, coalition: Coalition) -> None:
        """Save coalition."""
        table_name = f"{self._schema_prefix}coalitions"

        query = f"""
        INSERT INTO {table_name} (
            coalition_id, coalition_name, agent_names, task_type,
            total_tasks, successful_tasks,
            avg_total_latency_ms, avg_total_cost_usd,
            avg_user_satisfaction, avg_handoff_efficiency
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (coalition_id) DO UPDATE SET
            total_tasks = EXCLUDED.total_tasks,
            successful_tasks = EXCLUDED.successful_tasks,
            avg_total_latency_ms = EXCLUDED.avg_total_latency_ms,
            avg_total_cost_usd = EXCLUDED.avg_total_cost_usd,
            avg_user_satisfaction = EXCLUDED.avg_user_satisfaction,
            avg_handoff_efficiency = EXCLUDED.avg_handoff_efficiency,
            last_updated = CURRENT_TIMESTAMP
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                str(coalition.coalition_id),
                coalition.coalition_name,
                json.dumps(coalition.agent_names),
                coalition.task_type,
                coalition.total_tasks,
                coalition.successful_tasks,
                coalition.avg_total_latency_ms,
                float(coalition.avg_total_cost_usd),
                coalition.avg_user_satisfaction,
                coalition.avg_handoff_efficiency
            )

        logger.debug(f"Saved coalition: {coalition.coalition_name}")

    # ========== QUERY METHODS ==========

    async def get_agent(self, agent_name: str) -> Optional[Agent]:
        """Get agent by name."""
        table_name = f"{self._schema_prefix}agents"

        query = f"""
        SELECT * FROM {table_name} WHERE agent_name = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, agent_name)

        if not row:
            return None

        return self._row_to_agent(row)

    async def get_agents(
        self,
        agent_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Agent]:
        """Get all agents, optionally filtered by type."""
        table_name = f"{self._schema_prefix}agents"

        if agent_type:
            query = f"""
            SELECT * FROM {table_name}
            WHERE agent_type = $1
            ORDER BY last_seen DESC
            LIMIT $2
            """
            args = [agent_type, limit]
        else:
            query = f"""
            SELECT * FROM {table_name}
            ORDER BY last_seen DESC
            LIMIT $1
            """
            args = [limit]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)

        return [self._row_to_agent(row) for row in rows]

    async def get_agent_operations(
        self,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        is_handoff: Optional[bool] = None,
        limit: int = 100
    ) -> List[AgentOperation]:
        """Get agent operations with optional filters."""
        table_name = f"{self._schema_prefix}agent_operations"

        conditions = []
        args = []
        arg_count = 1

        if agent_name:
            conditions.append(f"agent_name = ${arg_count}")
            args.append(agent_name)
            arg_count += 1

        if session_id:
            # Need to join with events table to filter by session
            conditions.append(f"event_id IN (SELECT event_id FROM {self._schema_prefix}metric_events WHERE session_id = ${arg_count})")
            args.append(session_id)
            arg_count += 1

        if is_handoff is not None:
            conditions.append(f"is_handoff = ${arg_count}")
            args.append(is_handoff)
            arg_count += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        args.append(limit)

        query = f"""
        SELECT * FROM {table_name}
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT ${arg_count}
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)

        return [self._row_to_agent_operation(row) for row in rows]

    async def get_handoffs(
        self,
        session_id: Optional[str] = None,
        from_agent_name: Optional[str] = None,
        to_agent_name: Optional[str] = None,
        min_quality: Optional[float] = None,
        limit: int = 100
    ) -> List[AgentHandoff]:
        """Get handoffs with optional filters."""
        table_name = f"{self._schema_prefix}agent_handoffs"

        conditions = []
        args = []
        arg_count = 1

        if session_id:
            conditions.append(f"session_id = ${arg_count}")
            args.append(session_id)
            arg_count += 1

        if from_agent_name:
            conditions.append(f"from_agent_name = ${arg_count}")
            args.append(from_agent_name)
            arg_count += 1

        if to_agent_name:
            conditions.append(f"to_agent_name = ${arg_count}")
            args.append(to_agent_name)
            arg_count += 1

        if min_quality is not None:
            conditions.append(f"quality_score >= ${arg_count}")
            args.append(min_quality)
            arg_count += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        args.append(limit)

        query = f"""
        SELECT * FROM {table_name}
        WHERE {where_clause}
        ORDER BY handoff_timestamp DESC
        LIMIT ${arg_count}
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)

        return [self._row_to_handoff(row) for row in rows]

    async def get_coordination_graph(self, session_id: str) -> Optional[CoordinationGraph]:
        """Get coordination graph for a session."""
        table_name = f"{self._schema_prefix}coordination_graphs"

        query = f"""
        SELECT * FROM {table_name} WHERE session_id = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, session_id)

        if not row:
            return None

        return self._row_to_coordination_graph(row)

    async def get_bottlenecks(
        self,
        agent_name: Optional[str] = None,
        min_severity: Optional[str] = None,
        limit: int = 100
    ) -> List[BottleneckInfo]:
        """Get bottlenecks with optional filters."""
        table_name = f"{self._schema_prefix}bottlenecks"

        conditions = []
        args = []
        arg_count = 1

        if agent_name:
            conditions.append(f"agent_name = ${arg_count}")
            args.append(agent_name)
            arg_count += 1

        if min_severity:
            # Map severity to numeric for comparison
            severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            min_level = severity_order.get(min_severity, 1)
            conditions.append(f"CASE severity WHEN 'low' THEN 1 WHEN 'medium' THEN 2 WHEN 'high' THEN 3 WHEN 'critical' THEN 4 END >= {min_level}")

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        args.append(limit)

        query = f"""
        SELECT * FROM {table_name}
        WHERE {where_clause}
        ORDER BY bottleneck_score DESC, detected_at DESC
        LIMIT ${arg_count}
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)

        return [self._row_to_bottleneck(row) for row in rows]

    async def get_coalitions(
        self,
        task_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Coalition]:
        """Get coalitions with optional filters."""
        table_name = f"{self._schema_prefix}coalitions"

        if task_type:
            query = f"""
            SELECT * FROM {table_name}
            WHERE task_type = $1
            ORDER BY created_at DESC
            LIMIT $2
            """
            args = [task_type, limit]
        else:
            query = f"""
            SELECT * FROM {table_name}
            ORDER BY created_at DESC
            LIMIT $1
            """
            args = [limit]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)

        return [self._row_to_coalition(row) for row in rows]

    # ========== CONVERSION METHODS ==========

    def _row_to_agent(self, row: Dict[str, Any]) -> Agent:
        """Convert database row to Agent."""
        from llmops_monitoring.agent.base import AgentType

        return Agent(
            agent_id=UUID(row['agent_id']),
            agent_name=row['agent_name'],
            agent_role=row['agent_role'],
            agent_type=AgentType(row['agent_type']),
            capabilities=json.loads(row['capabilities']),
            input_schema=json.loads(row['input_schema']) if row['input_schema'] else None,
            output_schema=json.loads(row['output_schema']) if row['output_schema'] else None,
            total_invocations=row['total_invocations'],
            success_count=row['success_count'],
            failure_count=row['failure_count'],
            avg_latency_ms=row['avg_latency_ms'],
            avg_cost_usd=float(row['avg_cost_usd']),
            can_handoff_to=json.loads(row['can_handoff_to']),
            reports_to=row['reports_to'],
            created_at=row['created_at'],
            last_seen=row['last_seen']
        )

    def _row_to_agent_operation(self, row: Dict[str, Any]) -> AgentOperation:
        """Convert database row to AgentOperation."""
        return AgentOperation(
            operation_id=UUID(row['operation_id']),
            event_id=UUID(row['event_id']),
            agent_id=UUID(row['agent_id']),
            agent_name=row['agent_name'],
            parent_operation_id=UUID(row['parent_operation_id']) if row['parent_operation_id'] else None,
            parent_agent_id=UUID(row['parent_agent_id']) if row['parent_agent_id'] else None,
            parent_agent_name=row['parent_agent_name'],
            is_handoff=row['is_handoff'],
            handoff_reason=row['handoff_reason'],
            handoff_quality_score=row['handoff_quality_score'],
            input_context_hash=row['input_context_hash'],
            output_context_hash=row['output_context_hash'],
            context_similarity_to_parent=row['context_similarity_to_parent'],
            duration_ms=row['duration_ms'],
            success=row['success'],
            error=row['error'],
            timestamp=row['timestamp']
        )

    def _row_to_handoff(self, row: Dict[str, Any]) -> AgentHandoff:
        """Convert database row to AgentHandoff."""
        from llmops_monitoring.agent.base import HandoffQuality

        return AgentHandoff(
            handoff_id=UUID(row['handoff_id']),
            session_id=row['session_id'],
            trace_id=row['trace_id'],
            from_agent_id=UUID(row['from_agent_id']),
            from_agent_name=row['from_agent_name'],
            from_operation_id=UUID(row['from_operation_id']),
            to_agent_id=UUID(row['to_agent_id']),
            to_agent_name=row['to_agent_name'],
            to_operation_id=UUID(row['to_operation_id']),
            handoff_timestamp=row['handoff_timestamp'],
            handoff_reason=row['handoff_reason'],
            handoff_decision_process=json.loads(row['handoff_decision_process']) if row['handoff_decision_process'] else None,
            quality_score=row['quality_score'],
            quality_level=HandoffQuality(row['quality_level']),
            was_correct_agent=row['was_correct_agent'],
            handoff_efficiency=row['handoff_efficiency'],
            context_size_bytes=row['context_size_bytes'],
            context_summary=row['context_summary'],
            context_retention_score=row['context_retention_score'],
            handoff_latency_ms=row['handoff_latency_ms']
        )

    def _row_to_coordination_graph(self, row: Dict[str, Any]) -> CoordinationGraph:
        """Convert database row to CoordinationGraph."""
        return CoordinationGraph(
            graph_id=UUID(row['graph_id']),
            session_id=row['session_id'],
            nodes=json.loads(row['nodes']),
            edges=json.loads(row['edges']),
            total_agents=row['total_agents'],
            total_handoffs=row['total_handoffs'],
            max_depth=row['max_depth'],
            bottleneck_agent_id=UUID(row['bottleneck_agent_id']) if row['bottleneck_agent_id'] else None,
            critical_path_ms=row['critical_path_ms'],
            avg_handoff_quality=row['avg_handoff_quality'],
            avg_context_retention=row['avg_context_retention'],
            created_at=row['created_at']
        )

    def _row_to_bottleneck(self, row: Dict[str, Any]) -> BottleneckInfo:
        """Convert database row to BottleneckInfo."""
        return BottleneckInfo(
            agent_id=UUID(row['agent_id']),
            agent_name=row['agent_name'],
            severity=row['severity'],
            bottleneck_score=row['bottleneck_score'],
            avg_queue_time_ms=row['avg_queue_time_ms'],
            p95_latency_ms=row['p95_latency_ms'],
            utilization=row['utilization'],
            requests_delayed=row['requests_delayed'],
            recommendations=json.loads(row['recommendations'])
        )

    def _row_to_coalition(self, row: Dict[str, Any]) -> Coalition:
        """Convert database row to Coalition."""
        return Coalition(
            coalition_id=UUID(row['coalition_id']),
            coalition_name=row['coalition_name'],
            agent_names=json.loads(row['agent_names']),
            task_type=row['task_type'],
            total_tasks=row['total_tasks'],
            successful_tasks=row['successful_tasks'],
            avg_total_latency_ms=row['avg_total_latency_ms'],
            avg_total_cost_usd=float(row['avg_total_cost_usd']),
            avg_user_satisfaction=row['avg_user_satisfaction'],
            avg_handoff_efficiency=row['avg_handoff_efficiency']
        )
