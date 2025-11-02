# 🚀 LLMOps Monitoring - Complete Implementation Master Plan

## Overview
Implementing Multi-Agent Workflow Intelligence + Repository Topology Scanner

---

## ✅ COMPLETED (Phase 1A - PostgreSQL Storage)
- PostgreSQL schema for agent tables (6 tables, 950+ lines)
- PostgreSQL agent storage methods (write + query)
- Integration with main PostgresBackend

---

## 🎯 PART 1: MULTI-AGENT WORKFLOW INTELLIGENCE

### Phase 1B: Remaining Storage Backends

#### MySQL Agent Storage
- Similar structure to PostgreSQL
- Tables: agents, agent_operations, agent_handoffs, coordination_graphs, bottlenecks, coalitions
- Full CRUD methods

#### ClickHouse Agent Storage
- Optimized for analytics queries
- Columnar storage for high-performance aggregations
- Specialized indexes for time-series analysis

#### Parquet Agent Storage
- File-based append-only storage
- Efficient for batch processing
- Schema evolution support

### Phase 2: Agent Intelligence Service

#### Core Service (`llmops_monitoring/agent/service.py`)
```python
class AgentIntelligenceService:
    """Orchestrates all agent analysis components."""

    async def analyze_session(session_id: str):
        # 1. Fetch events from storage
        # 2. Detect agents (AgentDetector)
        # 3. Analyze handoffs (HandoffAnalyzer)
        # 4. Detect context drift (ContextDriftDetector)
        # 5. Build coordination graph (CoordinationGraphBuilder)
        # 6. Detect bottlenecks (BottleneckDetector)
        # 7. Discover coalitions (CoalitionAnalyzer)
        # 8. Store all results
```

#### Background Processing
- Async task queue for analysis
- Real-time vs batch processing modes
- Incremental updates

### Phase 3: API Layer

#### GraphQL Extensions
```graphql
type Agent { ... }
type AgentHandoff { ... }
type CoordinationGraph { ... }
type BottleneckInfo { ... }
type Coalition { ... }

type Query {
  agents(type: AgentType): [Agent!]!
  agent(name: String!): Agent
  handoffs(sessionId: String!, minQuality: Float): [AgentHandoff!]!
  coordinationGraph(sessionId: String!): CoordinationGraph
  bottlenecks(minSeverity: Severity): [BottleneckInfo!]!
  coalitions(taskType: String): [Coalition!]!
}
```

#### REST API Endpoints
```
GET  /api/agents
GET  /api/agents/{name}
GET  /api/handoffs?session_id={id}&min_quality={score}
GET  /api/coordination-graph/{session_id}
GET  /api/bottlenecks?min_severity={level}
GET  /api/coalitions?task_type={type}
```

### Phase 4: Decorator Integration

```python
@monitor_llm(
    operation_name="classify_query",
    agent_name="customer_classifier",        # NEW
    agent_role="classifier",                  # NEW
    agent_type="specialist",                  # NEW
    can_handoff_to=["support", "billing"]     # NEW
)
async def classify_customer_query(query: str):
    ...
```

### Phase 5: Examples

- **Example 14**: Basic multi-agent workflow with handoff tracking
- **Example 15**: Coordination graph visualization (D3.js/Cytoscape export)
- **Example 16**: Bottleneck detection and optimization recommendations
- **Example 17**: Coalition analysis and optimal team selection

---

## 🎯 PART 2: REPOSITORY TOPOLOGY SCANNER (NEW!)

### Research-Based Design

**Inspiration from:**
- **Pydeps**: Module dependency visualization
- **PyCG**: Static call graph generation (99.2% precision)
- **Code2graph**: Structure extraction for Deep Learning projects
- **NetworkX**: Graph construction and algorithms
- **Pyvis**: Interactive visualization

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Repository Topology Scanner                            │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ AST Parser   │  │ Import       │  │ Call Graph   │ │
│  │ (Python ast) │  │ Analyzer     │  │ Builder      │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                 │          │
│         └─────────────────┴─────────────────┘          │
│                          │                             │
│                          ▼                             │
│          ┌───────────────────────────────┐            │
│          │ Topology Graph Builder        │            │
│          │ (NetworkX DiGraph)            │            │
│          └───────────────┬───────────────┘            │
│                          │                             │
│         ┌────────────────┴────────────────┐           │
│         │                                 │           │
│         ▼                                 ▼           │
│  ┌──────────────┐              ┌──────────────┐      │
│  │ Visualization│              │ Storage      │      │
│  │ Exporter     │              │ (PostgreSQL) │      │
│  │ (D3, Cyto)   │              │              │      │
│  └──────────────┘              └──────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. AST Parser (`llmops_monitoring/topology/parser.py`)
```python
class CodeParser:
    """Parse Python code using AST."""

    def parse_file(filepath: str) -> ModuleInfo:
        # Extract: imports, classes, functions, methods
        # Track: decorators, docstrings, complexity

    def parse_repository(repo_path: str) -> RepositoryInfo:
        # Recursively parse all .py files
        # Build module tree
```

#### 2. Dependency Graph Builder (`llmops_monitoring/topology/graph.py`)
```python
class DependencyGraphBuilder:
    """Build module and function dependency graphs."""

    def build_module_graph() -> nx.DiGraph:
        # Nodes: Modules
        # Edges: Import relationships

    def build_call_graph() -> nx.DiGraph:
        # Nodes: Functions/Methods
        # Edges: Call relationships

    def detect_cycles() -> List[List[str]]:
        # Find circular dependencies

    def calculate_metrics():
        # Coupling, cohesion, complexity
```

#### 3. Call Graph Analyzer (`llmops_monitoring/topology/call_graph.py`)
```python
class CallGraphAnalyzer:
    """Analyze function call relationships."""

    def extract_calls(ast_node) -> List[Call]:
        # Static analysis of function calls

    def build_call_chain(entry_point: str) -> CallChain:
        # Trace execution paths
```

#### 4. Topology Storage (`postgres_topology_storage.py`)
```sql
-- New tables for topology data

CREATE TABLE code_modules (
    module_id UUID PRIMARY KEY,
    module_path VARCHAR(500) NOT NULL,
    module_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    line_count INTEGER,
    complexity_score FLOAT,
    last_scanned TIMESTAMP
);

CREATE TABLE code_functions (
    function_id UUID PRIMARY KEY,
    module_id UUID REFERENCES code_modules,
    function_name VARCHAR(255) NOT NULL,
    qualified_name VARCHAR(500) NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    complexity FLOAT,
    is_async BOOLEAN,
    decorators JSONB
);

CREATE TABLE function_calls (
    call_id UUID PRIMARY KEY,
    caller_function_id UUID REFERENCES code_functions,
    callee_function_id UUID REFERENCES code_functions,
    call_count INTEGER DEFAULT 1,  -- Updated from runtime data
    avg_duration_ms FLOAT,         -- From monitoring
    total_tokens INTEGER,          -- NEW: Token consumption!
    avg_tokens_per_call FLOAT
);

CREATE TABLE module_dependencies (
    dependency_id UUID PRIMARY KEY,
    from_module_id UUID REFERENCES code_modules,
    to_module_id UUID REFERENCES code_modules,
    import_type VARCHAR(50),  -- 'import', 'from_import'
    is_circular BOOLEAN DEFAULT FALSE
);

CREATE TABLE topology_snapshots (
    snapshot_id UUID PRIMARY KEY,
    repository_path VARCHAR(500),
    commit_hash VARCHAR(40),
    scanned_at TIMESTAMP,
    total_modules INTEGER,
    total_functions INTEGER,
    total_dependencies INTEGER,
    circular_dependencies INTEGER,
    graph_data JSONB  -- Full graph for visualization
);
```

#### 5. Token Consumption Integration
```python
class TokenTopologyCorrelator:
    """Correlate token consumption with code topology."""

    async def track_function_tokens(
        function_name: str,
        tokens: int,
        operation_id: UUID
    ):
        # Update function_calls table with token data

    async def get_high_token_components() -> List[ComponentUsage]:
        # Find code components consuming most tokens
        # Return: function_name, total_tokens, call_count, avg_tokens

    async def analyze_token_hotspots() -> HotspotAnalysis:
        # Identify:
        # - Most expensive functions
        # - Token consumption patterns
        # - Optimization opportunities
```

#### 6. Visualization Exporter
```python
class TopologyVisualizer:
    """Export topology for visualization."""

    def export_d3_force_graph() -> Dict:
        # D3.js force-directed graph format

    def export_cytoscape() -> Dict:
        # Cytoscape.js format

    def export_sankey() -> Dict:
        # Sankey diagram (for token flow)

    def export_treemap() -> Dict:
        # Treemap (for module hierarchy + token consumption)
```

### Phase 2T.1: Core Topology Scanner

**Files to Create:**
1. `llmops_monitoring/topology/__init__.py`
2. `llmops_monitoring/topology/parser.py` - AST parsing
3. `llmops_monitoring/topology/graph.py` - Graph building
4. `llmops_monitoring/topology/call_graph.py` - Call analysis
5. `llmops_monitoring/topology/models.py` - Data models
6. `llmops_monitoring/topology/analyzer.py` - Metrics & analysis

### Phase 2T.2: Storage Integration

**Files to Create:**
1. `llmops_monitoring/transport/backends/postgres_topology_storage.py`
2. Integration with existing PostgresBackend
3. Migration scripts

### Phase 2T.3: Token Correlation

**Files to Create:**
1. `llmops_monitoring/topology/correlator.py`
2. Update `@monitor_llm` decorator to track function names
3. Auto-link monitoring events to topology data

### Phase 2T.4: Visualization

**Files to Create:**
1. `llmops_monitoring/topology/visualizer.py`
2. `llmops_monitoring/examples/18_topology_scan.py`
3. `llmops_monitoring/examples/19_token_consumption_by_component.py`

### Phase 2T.5: API Integration

**GraphQL Extensions:**
```graphql
type CodeModule {
  moduleId: ID!
  modulePath: String!
  moduleName: String!
  lineCount: Int!
  complexityScore: Float!
  functions: [CodeFunction!]!
  dependencies: [ModuleDependency!]!
  tokenConsumption: TokenStats!
}

type CodeFunction {
  functionId: ID!
  functionName: String!
  qualifiedName: String!
  complexity: Float!
  totalTokens: Int!
  avgTokensPerCall: Float!
  callCount: Int!
}

type Query {
  topologySnapshot(commitHash: String): TopologySnapshot
  highTokenFunctions(limit: Int): [FunctionUsage!]!
  moduleGraph(format: String): GraphData!
  tokenHotspots: [ComponentUsage!]!
}
```

---

## 🎯 IMPLEMENTATION PRIORITY

### Week 1: Complete Storage Layer
- ✅ PostgreSQL (Done!)
- MySQL agent storage
- ClickHouse agent storage
- Parquet agent storage

### Week 2: Agent Intelligence Service + Examples
- AgentIntelligenceService orchestration
- 4 comprehensive examples
- Integration with @monitor_llm decorator

### Week 3: Repository Topology Scanner (NEW!)
- AST parser + graph builder
- PostgreSQL topology storage
- Basic visualization export

### Week 4: Token Correlation & Advanced Features
- Link monitoring events to code components
- Token consumption analysis
- Hotspot detection
- Optimization recommendations

### Week 5: API Layer + Documentation
- GraphQL schema extensions
- REST API endpoints
- Comprehensive documentation
- End-to-end tests

---

## 🎨 VISUALIZATION EXAMPLES

### Agent Coordination Graph
```javascript
// D3.js force-directed graph
{
  "nodes": [
    {"id": "classifier", "type": "specialist", "invocations": 1000},
    {"id": "retriever", "type": "specialist", "invocations": 750},
    {"id": "responder", "type": "primary", "invocations": 500}
  ],
  "edges": [
    {"from": "classifier", "to": "retriever", "handoffs": 600, "quality": 0.95},
    {"from": "retriever", "to": "responder", "handoffs": 500, "quality": 0.88}
  ]
}
```

### Repository Topology with Token Consumption
```javascript
// Treemap: size = token consumption
{
  "name": "repository",
  "children": [
    {
      "name": "llmops_monitoring",
      "children": [
        {
          "name": "instrumentation",
          "tokens": 125000,
          "children": [
            {"name": "decorators.monitor_llm", "tokens": 85000, "calls": 1500},
            {"name": "collectors.text", "tokens": 40000, "calls": 2000}
          ]
        },
        {
          "name": "agent",
          "tokens": 95000,
          "children": [
            {"name": "detector.detect_agents", "tokens": 45000, "calls": 500},
            {"name": "analyzer.analyze_handoffs", "tokens": 50000, "calls": 800}
          ]
        }
      ]
    }
  ]
}
```

---

## 📊 SUCCESS METRICS

### Agent Intelligence
- ✅ Store 1M+ agent operations with <50ms overhead
- ✅ Detect handoffs with 95%+ accuracy
- ✅ Identify bottlenecks within 100ms
- ✅ Build coordination graphs for 100+ agent systems

### Repository Topology
- ✅ Parse 10K+ file repositories in <60 seconds
- ✅ Detect circular dependencies
- ✅ Calculate complexity metrics
- ✅ Correlate 99%+ of function calls with monitoring data
- ✅ Identify top 10 token-consuming components

### Integration
- ✅ Zero-config auto-detection
- ✅ GraphQL + REST APIs for all data
- ✅ Real-time visualization updates
- ✅ <100ms query performance

---

## 🚀 ESTIMATED DELIVERABLES

**Total Lines of Code: ~8,500**
- Storage backends: ~2,000 lines
- Agent Intelligence Service: ~600 lines
- Repository Topology Scanner: ~2,500 lines
- API layer: ~800 lines
- Examples: ~1,200 lines
- Tests: ~1,200 lines
- Documentation: ~200 lines

**Implementation Time: 5 weeks**
- Full production-ready implementation
- Comprehensive tests
- Complete documentation
- All storage backends
- All visualization formats

---

## 🎯 NEXT STEPS (Immediate)

1. **Create Repository Topology Scanner** (Most exciting new feature!)
   - AST parser
   - Dependency graph builder
   - Call graph analyzer

2. **Complete Storage Backends**
   - MySQL agent storage
   - ClickHouse agent storage

3. **Build AgentIntelligenceService**
   - Orchestrate all analyzers
   - Background processing

4. **Create Examples**
   - Show real-world usage
   - Demonstrate value

Let's execute! 🚀
