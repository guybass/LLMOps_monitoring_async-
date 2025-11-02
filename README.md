# llamonitor-async 🦙📊

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/llamonitor-async.svg)](https://pypi.org/project/llamonitor-async/)
[![Downloads](https://static.pepy.tech/badge/llamonitor-async)](https://pepy.tech/project/llamonitor-async)
[![Downloads/Month](https://static.pepy.tech/badge/llamonitor-async/month)](https://pepy.tech/project/llamonitor-async)
[![Downloads/Week](https://static.pepy.tech/badge/llamonitor-async/week)](https://pepy.tech/project/llamonitor-async)

**Lightweight async monitoring for LLM applications** - flexible measurement strategies with pluggable storage.

A modern alternative to Langfuse/LangSmith with **pluggable measurement** (capacity OR tokens OR both), async-first architecture, and maximum extensibility.

## Documentation

📚 **[Complete Documentation](docs/README.md)** | 🚀 **[Quick Start Guide](docs/getting-started/QUICKSTART.md)** | 🧪 **[Testing Guide](docs/guides/TEST_GUIDE.md)** | 📊 **[Download Tracking](docs/guides/DOWNLOAD_TRACKING.md)**

### Publishing Guides
- **[Publishing to PyPI](docs/publishing/PUBLISH.md)** - Complete publication guide
- **[Upload Guide](docs/publishing/UPLOAD_GUIDE.md)** - Quick reference
- **[Pre-Publish Checklist](docs/publishing/PRE_PUBLISH_CHECKLIST.md)** - Step-by-step checklist

## Design Philosophy: "Leave Space for Air Conditioning"

Every component has clear extension points for future enhancements. Whether you need custom metric collectors, new storage backends, or specialized aggregation strategies, the architecture supports growth without breaking existing code.

## Features

- **Async-First**: Non-blocking metric collection with buffered batch writes
- **Hierarchical Tracking**: Automatic parent-child relationships across nested operations
- **Pluggable Measurement**: Choose capacity (chars/words/bytes), tokens (industry standard), or hybrid ✨ NEW!
- **Flexible Metrics**: Measure text (characters, words, bytes, tokens) and images (count, pixels, file size)
- **Built-in Cost Tracking**: Automatic cost calculation for 18+ major LLM models
- **Prometheus Exporter**: Real-time metrics export for monitoring and alerting
- **Anomaly Detection**: ML-based detection with Z-score, IQR, and error rate methods
- **Pluggable Storage**: Local Parquet, PostgreSQL, MySQL, ClickHouse (easily add more)
- **Simple API**: Single decorator for most use cases
- **Production-Ready**: Error handling, retries, graceful shutdown
- **Extensible**: Custom collectors, backends, and aggregation strategies

## Quick Start

### Installation

```bash
# Basic installation
pip install llamonitor-async

# With storage backends
pip install llamonitor-async[parquet]    # For local Parquet files
pip install llamonitor-async[postgres]   # For PostgreSQL
pip install llamonitor-async[mysql]      # For MySQL
pip install llamonitor-async[clickhouse] # For ClickHouse (analytics)

# With measurement strategies
pip install llamonitor-async[tokens]     # For token measurement (tiktoken)

# With exporters and integrations
pip install llamonitor-async[prometheus] # For Prometheus metrics
pip install llamonitor-async[datadog]    # For Datadog integration

# With API servers
pip install llamonitor-async[api]        # For REST API server
pip install llamonitor-async[graphql]    # For GraphQL API

# All features
pip install llamonitor-async[all]        # Everything
```

### Basic Usage

```python
import asyncio
from llamonitor import monitor_llm, initialize_monitoring, MonitorConfig

@monitor_llm(
    operation_name="generate_text",
    measure_text=True,  # Collect all text metrics
    custom_attributes={"model": "gpt-4"}
)
async def my_llm_function(prompt: str):
    # Your LLM call here
    return {"text": "Generated response..."}

async def main():
    # Initialize monitoring
    await initialize_monitoring(MonitorConfig.for_local_dev())

    # Use your decorated functions
    result = await my_llm_function("Hello!")

    # Events are automatically tracked and written asynchronously

if __name__ == "__main__":
    asyncio.run(main())
```

## Measurement Strategies ✨ NEW!

Choose how to measure text: **capacity** (chars/words/bytes), **tokens** (industry standard), or **both** (hybrid).

### Strategy Modes

**1. Auto Mode** (Recommended)
Automatically selects the best strategy based on the model:
```python
@monitor_llm(
    operation_name="my_llm_call",
    measurement="auto",  # Auto-select based on model
    measure_text=True,
    custom_attributes={"model": "gpt-4"}
)
async def my_llm_call(prompt: str):
    return await llm.generate(prompt)
```

**2. Token Mode** (Cost Tracking)
Accurate token counting using provider-specific tokenizers:
```python
@monitor_llm(
    operation_name="my_llm_call",
    measurement="token",  # Use token counting
    measure_text=True,
    custom_attributes={"model": "gpt-4"}
)
async def my_llm_call(prompt: str):
    return await llm.generate(prompt)

# Metrics collected: input_tokens, output_tokens, total_tokens
```

**3. Capacity Mode** (Performance Monitoring)
Fast, reliable character/word/byte counting:
```python
@monitor_llm(
    operation_name="my_llm_call",
    measurement="capacity",  # Use capacity metrics
    measure_text=True
)
async def my_llm_call(prompt: str):
    return await llm.generate(prompt)

# Metrics collected: char_count, word_count, byte_size, line_count
```

**4. Hybrid Mode** (Migration & Cross-Validation)
Collect BOTH capacity and token metrics simultaneously:
```python
@monitor_llm(
    operation_name="my_llm_call",
    measurement="hybrid",  # Collect both!
    measure_text=True,
    custom_attributes={"model": "gpt-4"}
)
async def my_llm_call(prompt: str):
    return await llm.generate(prompt)

# Metrics collected: char_count, word_count, byte_size, line_count,
#                    input_tokens, output_tokens, total_tokens
```

### Direct Usage (Without Decorator)

```python
from llmops_monitoring.measurement import measure_text

# Quick measurement
result = await measure_text(
    "Hello, world!",
    mode="token",
    context={"model": "gpt-4"}
)

print(f"Tokens: {result.total_tokens}")
print(f"Characters: {result.char_count}")
print(f"Reliability: {result.metadata.reliability.value}")
```

### Advanced Configuration

```python
@monitor_llm(
    operation_name="my_llm_call",
    measurement={
        "mode": "hybrid",
        "prefer_tokens": True,          # Prefer token measurement
        "fallback_enabled": True,       # Fall back to capacity if tokens fail
        "parallel_hybrid": True,        # Run both strategies in parallel
        "async_tokenization": True,     # Non-blocking tokenization
        "tokenization_timeout_ms": 100  # Timeout for token counting
    },
    measure_text=True
)
async def my_llm_call(prompt: str):
    return await llm.generate(prompt)
```

### Supported Tokenizers

- **OpenAI** (gpt-4, gpt-4-turbo, gpt-4o, gpt-3.5-turbo): Uses `tiktoken` (exact)
- **Anthropic** (claude-3-*): API-based (estimation fallback)
- **Google** (gemini-*): API-based (estimation fallback)
- **Meta** (llama-*): `sentencepiece` (planned)
- **Unknown models**: 4:1 character-to-token ratio estimation

### Installation

```bash
# Token measurement support
pip install llamonitor-async[tokens]  # Includes tiktoken

# Or install manually
pip install tiktoken  # For OpenAI models
```

### When to Use Each Strategy

| Use Case | Recommended Strategy | Why |
|----------|---------------------|-----|
| **Cost tracking** | `token` | Industry standard, billing accuracy |
| **Performance monitoring** | `capacity` | <1ms latency, always reliable |
| **Migration** | `hybrid` | Compare old vs new metrics |
| **Multi-provider apps** | `auto` | Adapts to each provider |
| **Debugging token counts** | `hybrid` | Cross-validate token counting |
| **Unknown models** | `capacity` | Universal support |

### Example: Migration from Capacity to Tokens

```python
# Week 1-2: Collect both metrics
@monitor_llm(measurement="hybrid", measure_text=True)
async def my_call(prompt: str):
    return await llm.generate(prompt)

# Analyze correlation in Grafana/Jupyter
# df = pd.read_parquet("./monitoring_data/**/*.parquet")
# df.plot.scatter(x='text_metrics.char_count', y='text_metrics.total_tokens')

# Week 3+: Switch to tokens only
@monitor_llm(measurement="token", measure_text=True)
async def my_call(prompt: str):
    return await llm.generate(prompt)
```

See `examples/13_measurement_strategies.py` for comprehensive examples.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                         │
│  @monitor_llm decorated functions/methods                   │
└───────────────────┬─────────────────────────────────────────┘
                    │ (async, non-blocking)
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              Instrumentation Layer                          │
│  • MetricCollectors (text, image, cost, custom)             │
│  • Context Management (session/trace/span)                  │
│  • Decorator Logic                                          │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│               Transport Layer                               │
│  • Async Queue (buffering)                                  │
│  • Background Worker (batching)                             │
│  • Retry Logic                                              │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ├─────────────────────┬───────────────────┐
                    ▼                     ▼                   ▼
         ┌─────────────────┐   ┌──────────────────┐  ┌──────────────┐
         │ Storage Backend │   │ Metrics Exporter │  │   Future     │
         │  • Parquet      │   │  • Prometheus    │  │ Integrations │
         │  • PostgreSQL   │   │  • Datadog (TBD) │  │              │
         │  • MySQL        │   │  • Custom        │  │              │
         └─────────────────┘   └──────────────────┘  └──────────────┘
```

## Configuration

### Environment Variables

```bash
LLMOPS_BACKEND=postgres
LLMOPS_CONNECTION_STRING=postgresql://user:pass@localhost/monitoring
LLMOPS_BATCH_SIZE=100
LLMOPS_FLUSH_INTERVAL_SECONDS=5.0
```

### Programmatic Configuration

```python
from llmops_monitoring import MonitorConfig
from llmops_monitoring.schema.config import StorageConfig

# Local development
config = MonitorConfig.for_local_dev()

# Production
config = MonitorConfig.for_production(
    "postgresql://user:pass@host:5432/monitoring"
)

# Custom
config = MonitorConfig(
    storage=StorageConfig(
        backend="parquet",
        output_dir="./my_data",
        batch_size=500,
        flush_interval_seconds=10.0
    ),
    max_queue_size=50000
)

await initialize_monitoring(config)
```

## Examples

### Hierarchical Tracking (Agentic Workflows)

```python
from llmops_monitoring.instrumentation.context import monitoring_session, monitoring_trace

@monitor_llm("orchestrator", operation_type="agent_workflow")
async def run_workflow(query: str):
    # All nested calls automatically tracked
    intent = await classify_intent(query)      # Child span
    knowledge = await search_kb(intent)        # Child span
    response = await generate_response(knowledge)  # Child span
    return response

@monitor_llm("classify_intent")
async def classify_intent(query: str):
    # Automatically linked to parent
    return await llm.classify(query)

# Use with session context
with monitoring_session("user-123"):
    with monitoring_trace("conversation-1"):
        result = await run_workflow("What is the weather?")
```

### Built-in Cost Tracking ✨ NEW!

Automatically track costs for major LLM providers:

```python
@monitor_llm(
    operation_name="my_llm_call",
    measure_text=True,
    collectors=["cost"],  # Enable cost tracking
    custom_attributes={
        "model": "gpt-4o-mini"  # Pricing lookup
    }
)
async def my_llm_call(prompt: str):
    # Your LLM API call here
    return {"text": "response..."}

# Query costs later
import pandas as pd
df = pd.read_parquet("./dev_monitoring_data/**/*.parquet")
df['cost'] = df['custom_attributes'].apply(lambda x: x.get('estimated_cost_usd'))
print(f"Total cost: ${df['cost'].sum():.6f}")
```

**Supported Models (18 total):**
- OpenAI: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo
- Anthropic: claude-3-opus, claude-3-sonnet, claude-3-5-sonnet, claude-3-haiku
- Google: gemini-1.5-pro, gemini-1.5-flash, gemini-1.0-pro
- Meta: llama-3-8b, llama-3-70b
- Mistral: mixtral-8x7b, mistral-small, mistral-medium, mistral-large

### Prometheus Metrics Export ✨ NEW!

Expose metrics to Prometheus for monitoring and alerting:

```python
from llmops_monitoring import initialize_monitoring, MonitorConfig
from llmops_monitoring.schema.config import PrometheusConfig

# Configure with Prometheus exporter
config = MonitorConfig.for_local_dev()
config.extensions["prometheus"] = PrometheusConfig(
    enabled=True,
    port=8000,
    host="0.0.0.0"
).model_dump()

await initialize_monitoring(config)

# Metrics available at http://localhost:8000/metrics
```

**Available Metrics:**
- `llm_operations_total` (Counter): Total operations by operation_name, model, type
- `llm_errors_total` (Counter): Total errors by operation_name, error_type
- `llm_operation_duration_seconds` (Histogram): Operation latency distribution
- `llm_text_characters_total` (Counter): Total characters processed
- `llm_cost_usd` (Histogram): Cost per operation distribution
- `llm_queue_size` (Gauge): Current queue size
- `llm_buffer_size` (Gauge): Current buffer size

**Prometheus Scrape Config:**
```yaml
scrape_configs:
  - job_name: 'llm-monitoring'
    static_configs:
      - targets: ['localhost:8000']
```

### Custom Metrics

For completely custom collectors:

```python
from llmops_monitoring.instrumentation.base import MetricCollector, CollectorRegistry

class MyCustomCollector(MetricCollector):
    def collect(self, result, args, kwargs, context):
        # Your custom logic
        return {"custom_attributes": {"my_metric": 123}}

    @property
    def metric_type(self) -> str:
        return "custom"

CollectorRegistry.register("my_custom", MyCustomCollector)

@monitor_llm(collectors=["my_custom"])
async def my_function():
    ...
```

## Visualization with Grafana

Start the monitoring stack:

```bash
docker-compose up -d
```

Access Grafana at `http://localhost:3000` (admin/admin)

The dashboard includes:
- Total events and volume metrics
- Time-series charts by operation
- Session analysis
- Error tracking
- Hierarchical trace viewer

## Storage Backends

### Parquet (Local Development)

```python
config = MonitorConfig(
    storage=StorageConfig(
        backend="parquet",
        output_dir="./monitoring_data",
        partition_by="date"  # or "session_id"
    )
)
```

Files are written as `./monitoring_data/YYYY-MM-DD/events_*.parquet`

### PostgreSQL (Production)

```python
config = MonitorConfig(
    storage=StorageConfig(
        backend="postgres",
        connection_string="postgresql://user:pass@host:5432/db",
        table_name="metric_events",
        pool_size=20
    )
)
```

Tables are created automatically with proper indexes.

### MySQL (Production)

```python
config = MonitorConfig(
    storage=StorageConfig(
        backend="mysql",
        connection_string="mysql://user:pass@host:3306/monitoring",
        table_name="metric_events",
        pool_size=20
    )
)
```

Tables are created automatically with InnoDB engine and proper indexes.

### ClickHouse (Analytics)

```python
config = MonitorConfig(
    storage=StorageConfig(
        backend="clickhouse",
        connection_string="clickhouse://default:@localhost:9000/monitoring",
        table_name="metric_events",
        schema_name="monitoring",  # database name
        batch_size=100
    )
)
```

**Optimized for:**
- High-volume event ingestion (millions of events/sec)
- Fast time-series analytics with native DATE partitioning
- Percentile calculations using `quantile()` functions (P50, P95, P99)
- Complex aggregations across large datasets
- JSON field querying with `JSONExtractFloat()`, `JSONExtractString()`

Tables use MergeTree engine with compression and are partitioned by month automatically.

## Extension Points

### 1. Custom Metric Collectors

Implement `MetricCollector` to add new metric types:

```python
class MyCollector(MetricCollector):
    def collect(self, result, args, kwargs, context):
        # Extract metrics
        return {"custom_attributes": {...}}

    @property
    def metric_type(self) -> str:
        return "my_metric"
```

### 2. Custom Storage Backends

Implement `StorageBackend` for new storage systems:

```python
class RedisBackend(StorageBackend):
    async def initialize(self): ...
    async def write_event(self, event): ...
    async def write_batch(self, events): ...
    async def close(self): ...
```

### 3. Custom Transport Mechanisms

Replace the async queue with Kafka, Redis, etc. by modifying `MonitoringWriter`.

## Performance

- **Overhead**: < 1% for typical workloads
- **Async writes**: No blocking of application code
- **Batching**: Configurable batch sizes for efficiency
- **Buffering**: Handles bursts without data loss
- **Graceful shutdown**: Flushes all pending events

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/llmops-monitoring
cd llmops-monitoring

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run examples
python llmops_monitoring/examples/01_simple_example.py
python llmops_monitoring/examples/02_agentic_workflow.py
python llmops_monitoring/examples/03_custom_collector.py
python llmops_monitoring/examples/04_mysql_backend.py
python llmops_monitoring/examples/05_cost_calculation.py
python llmops_monitoring/examples/06_prometheus_exporter.py
python llmops_monitoring/examples/07_aggregation_api.py
python llmops_monitoring/examples/08_websocket_streaming.py
python llmops_monitoring/examples/09_clickhouse_backend.py
python llmops_monitoring/examples/10_graphql_queries.py
python llmops_monitoring/examples/11_anomaly_detection.py
python llmops_monitoring/examples/12_datadog_integration.py
python llmops_monitoring/examples/13_measurement_strategies.py

# Start monitoring stack
docker-compose up -d
```

## REST API for Querying Data ✨ NEW!

Query and aggregate stored monitoring data via REST API:

```python
from llmops_monitoring import MonitorConfig
from llmops_monitoring.api import run_api_server

# Start API server
config = MonitorConfig.for_local_dev()
run_api_server(config, port=8080)

# API available at http://localhost:8080
# Interactive docs at http://localhost:8080/docs
```

**Available Endpoints:**
- `GET /api/health` - Health check
- `GET /api/v1/events` - Query events with filters
- `GET /api/v1/sessions` - List sessions
- `GET /api/v1/sessions/{session_id}` - Session details
- `GET /api/v1/sessions/{session_id}/traces` - Get traces
- `GET /api/v1/metrics/summary` - Summary statistics
- `GET /api/v1/metrics/operations` - Metrics by operation
- `GET /api/v1/metrics/models` - Metrics by model
- `GET /api/v1/metrics/costs` - Cost analytics

**Query Examples:**
```bash
# Get summary statistics
curl http://localhost:8080/api/v1/metrics/summary

# List recent sessions
curl http://localhost:8080/api/v1/sessions?limit=10

# Get metrics by operation
curl http://localhost:8080/api/v1/metrics/operations

# Get cost analytics grouped by model
curl 'http://localhost:8080/api/v1/metrics/costs?group_by=model'
```

## Real-time WebSocket Streaming ✨ NEW!

Stream monitoring events in real-time via WebSockets:

```python
from llmops_monitoring import MonitorConfig, initialize_monitoring
from llmops_monitoring.schema.config import WebSocketConfig

# Enable WebSocket streaming
config = MonitorConfig.for_local_dev()
config.extensions["websocket"] = WebSocketConfig(
    enabled=True
).model_dump()

await initialize_monitoring(config)
```

**WebSocket Endpoints:**
- `WS /api/v1/stream` - All events in real-time
- `WS /api/v1/stream/sessions/{session_id}` - Session-specific events
- `WS /api/v1/stream/operations/{operation_name}` - Operation-specific events

**Python Client Example:**
```python
import asyncio
import websockets
import json

async def listen_to_events():
    uri = 'ws://localhost:8080/api/v1/stream'
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            event = json.loads(message)
            print(f"Received event: {event['data']['operation_name']}")

asyncio.run(listen_to_events())
```

**JavaScript Client Example:**
```javascript
const ws = new WebSocket('ws://localhost:8080/api/v1/stream');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received event:', data);
};
```

## GraphQL API ✨ NEW!

Flexible, strongly-typed querying with GraphQL:

```python
from llmops_monitoring import MonitorConfig
from llmops_monitoring.api import create_api_server

# GraphQL endpoint automatically enabled
config = MonitorConfig.for_local_dev()
app = create_api_server(config)

# GraphQL Playground: http://localhost:8080/graphql
```

**Example Queries:**

```graphql
# Get summary with nested data
query {
  summary {
    totalEvents
    totalCost_usd
    errorRate
  }
  operations {
    operationName
    count
    p95DurationMs
    estimatedCostUsd
  }
}

# Query events with filters
query {
  events(filter: { limit: 10, errorOnly: false }) {
    eventId
    operationName
    timestamp
    durationMs
    textMetrics {
      charCount
      wordCount
    }
  }
}

# Real-time subscription
subscription {
  eventStream(operationName: "my_operation") {
    eventId
    operationName
    timestamp
  }
}
```

**Benefits:**
- Request exactly the fields you need
- Combine multiple queries in single request
- Strongly typed schema with auto-completion
- Real-time subscriptions via WebSocket
- Interactive GraphQL Playground
- Self-documenting API

**Installation:**
```bash
pip install 'llamonitor-async[graphql]'
```

## ML-based Anomaly Detection ✨ NEW!

Automatically detect anomalous behavior in your LLM operations:

```python
from llmops_monitoring.anomaly import AnomalyDetectionService, AnomalyResult

# Define alert handler
def alert_handler(anomaly: AnomalyResult):
    print(f"🚨 Anomaly detected in {anomaly.operation_name}")
    print(f"   Severity: {anomaly.severity.value}")
    print(f"   Metric: {anomaly.metric_name}")
    print(f"   Score: {anomaly.anomaly_score:.3f}")

# Initialize service
service = AnomalyDetectionService(alert_callback=alert_handler)

# Train on historical data
await service.train(historical_events)

# Detect anomalies in new data
anomalies = await service.detect(new_events)
```

**Detection Methods:**

1. **Z-Score Detection**: Detects deviations from mean (assumes normal distribution)
   - Best for: Latency, character counts
   - Threshold: 3 standard deviations (configurable)

2. **IQR Detection**: Interquartile range method (robust to outliers)
   - Best for: All metrics, especially with outliers
   - Threshold: 1.5 × IQR (configurable)

3. **Error Rate Detection**: Detects unusual error patterns
   - Best for: Catching sudden error spikes
   - Threshold: 10% error rate (configurable)

**Severity Levels:**
- LOW (score < 0.5)
- MEDIUM (score 0.5-0.7)
- HIGH (score 0.7-0.9) → Alerts sent
- CRITICAL (score ≥ 0.9) → Alerts sent

**Use Cases:**
- Detect latency spikes
- Identify unusual output sizes
- Catch error rate increases
- Monitor model behavior changes
- Alert on performance degradation

**Example Output:**
```
🚨 ANOMALY DETECTED!
Operation: generate_summary
Metric: duration_ms
Severity: HIGH
Anomaly Score: 0.847
Method: z_score
Value: 2500ms (expected: 120ms ± 50ms)
Z-score: 5.2
```

## Datadog Integration ✨ NEW!

Export metrics to Datadog for centralized monitoring and alerting:

```python
from llmops_monitoring import initialize_monitoring, MonitorConfig
from llmops_monitoring.schema.config import DatadogConfig

# Configure with Datadog exporter
config = MonitorConfig.for_local_dev()
config.extensions["datadog"] = DatadogConfig(
    enabled=True,
    api_key="your-datadog-api-key",
    app_key="your-datadog-app-key",
    site="datadoghq.com",  # or datadoghq.eu
    namespace="llmops"
).model_dump()

await initialize_monitoring(config)
```

**Available Metrics:**
- `llmops.operations.count` - Operation counts by operation_name, model, type
- `llmops.errors.count` - Error counts by operation_name, error_type
- `llmops.operation.duration.ms` - Latency distribution (for P95/P99)
- `llmops.characters.total` - Character counts for token estimation
- `llmops.cost.usd` - Cost per operation

**DogStatsD Support:**
```python
# Use DogStatsD instead of HTTP API (faster, lower latency)
config.extensions["datadog"] = DatadogConfig(
    enabled=True,
    use_statsd=True,
    statsd_host="localhost",
    statsd_port=8125,
    namespace="llmops"
).model_dump()
```

**Datadog Query Examples:**
```
# Average latency by operation
avg:llmops.operation.duration.ms{*} by {operation_name}

# Error rate
sum:llmops.errors.count{*}.as_rate()

# Total cost by model
sum:llmops.cost.usd{*} by {model}

# P95 latency
p95:llmops.operation.duration.ms{*} by {operation_name}
```

**Benefits:**
- Centralized monitoring with infrastructure metrics
- Powerful alerting and anomaly detection
- Correlation with logs, traces, and APM
- Pre-built dashboards and widgets
- SLO tracking and reporting

## Multi-Agent Workflow Intelligence 🔥 INNOVATIVE!

**The FIRST monitoring system designed specifically for multi-agent LLM workflows.**

As LLM applications evolve from single-model calls to complex multi-agent systems (LangGraph, CrewAI, AutoGen), traditional monitoring tools fall short. llamonitor-async provides deep visibility into agent interactions, handoffs, and coordination patterns.

### Why This Matters

Multi-agent systems are becoming the standard:
- **Customer Support**: Classifier → Specialist → Response Generator
- **RAG Pipelines**: Router → Multiple Retrievers → Synthesizer
- **Agent Orchestration**: Coordinator → Task Specialists → Fallback Handlers
- **Autonomous Systems**: Planning → Execution → Validation → Reflection

But existing tools treat agents as black boxes with no visibility into:
- Which handoffs are failing
- Where information is being lost
- Which agents are bottlenecks
- How agent teams perform together

### Key Capabilities

#### 1. **Automatic Agent Detection**
Identifies agents from operation traces automatically:
```python
from llmops_monitoring.agent import AgentDetector

detector = AgentDetector()
agents = await detector.detect_agents(events)

# Automatically discovers:
# - Agent names and roles (classifier, retriever, generator, etc.)
# - Performance metrics (latency, success rate, invocations)
# - Collaboration patterns
```

#### 2. **Handoff Quality Scoring**
Evaluates agent-to-agent transitions with multi-component scoring:
```python
from llmops_monitoring.agent import HandoffAnalyzer

analyzer = HandoffAnalyzer()
handoffs = await analyzer.analyze_handoffs(operations, events)

# Quality Score Components:
# - Correctness (40%): Did the target agent succeed?
# - Efficiency (30%): Was the handoff fast?
# - Success Chain (30%): Overall workflow success
```

**Quality Levels:**
- **EXCELLENT** (>0.9): Perfect handoff
- **GOOD** (0.7-0.9): Working well
- **ACCEPTABLE** (0.5-0.7): Could be improved
- **POOR** (0.3-0.5): Needs attention
- **FAILED** (<0.3): Critical issue

#### 3. **Context Drift Detection** 🔥 **GENUINELY INNOVATIVE**

Detects when information is lost as tasks pass through agent chains:

```python
from llmops_monitoring.agent import ContextDriftDetector

drift_detector = ContextDriftDetector()
drift_analyses = await drift_detector.detect_drift(operations, events)

# Detects lost entities:
# - Customer names, emails, phone numbers
# - Order numbers, IDs, quantities
# - Quoted text and proper nouns

for analysis in drift_analyses:
    if analysis.has_high_drift:  # >40% information loss
        print(f"⚠️ High drift: {analysis.from_agent} → {analysis.to_agent}")
        print(f"   Lost: {len(analysis.lost_entities)} entities")
        for entity in analysis.lost_entities:
            print(f"   - {entity.value} ({entity.entity_type})")
```

**Real Example:**
```
Input: "John Smith at john@example.com needs help with order #12345"
         ↓ (Classifier → Knowledge Base)
Output: "order help"  ❌ Lost: name, email, order number!
```

This catches critical bugs that would otherwise go unnoticed in production.

#### 4. **Coordination Graph Building**

Visualizes agent interactions as a directed graph:
```python
from llmops_monitoring.agent import CoordinationGraphBuilder

graph_builder = CoordinationGraphBuilder()
graph = await graph_builder.build_graph(agents, operations, handoffs, session_id)

# Graph includes:
# - Nodes: Agents with performance metrics
# - Edges: Handoffs with quality scores
# - Metrics: Max depth, critical path, bottlenecks
# - Export: NetworkX, Cytoscape, D3.js formats

# Export for visualization
networkx_data = graph_builder.export_for_visualization(graph, format="networkx")
d3_data = graph_builder.export_for_visualization(graph, format="d3")
```

#### 5. **Bottleneck Detection**

Identifies performance bottlenecks in agent systems:
```python
from llmops_monitoring.agent import BottleneckDetector

bottleneck_detector = BottleneckDetector()
bottlenecks = await bottleneck_detector.detect_bottlenecks(
    agents, operations, handoffs, nodes, edges
)

# Bottleneck Score Components:
# - Utilization (30%): Traffic volume
# - Latency (30%): Response time
# - Failure Rate (20%): Error frequency
# - Downstream Impact (20%): Delays to other agents

for bottleneck in bottlenecks:
    print(f"{bottleneck.severity}: {bottleneck.agent_name}")
    print(f"  Score: {bottleneck.bottleneck_score:.2f}")
    print(f"  P95 Latency: {bottleneck.p95_latency_ms:.0f}ms")
    print(f"  Recommendations:")
    for rec in bottleneck.recommendations:
        print(f"    • {rec}")
```

#### 6. **Coalition Analytics**

Identifies and evaluates agent teams:
```python
from llmops_monitoring.agent import CoalitionAnalyzer

coalition_analyzer = CoalitionAnalyzer()
coalitions = await coalition_analyzer.discover_coalitions(
    agents, operations, handoffs, session_id
)

# Discovers teams that work together:
# - Customer Support Team: Classifier + Knowledge Base + Response Generator
# - Escalation Team: Classifier + Escalation Handler
# - RAG Pipeline: Router + Retrievers + Synthesizer

for coalition in coalitions:
    analysis = coalition_analyzer.analyze_coalition_performance(coalition)
    print(f"{coalition.coalition_name}")
    print(f"  Performance: {analysis['performance_rating']}")
    print(f"  Success Rate: {coalition.success_rate * 100:.1f}%")
    print(f"  Avg Latency: {coalition.avg_total_latency_ms:.1f}ms")
```

### Quick Start

```python
from llmops_monitoring.agent import (
    AgentDetector,
    HandoffAnalyzer,
    ContextDriftDetector,
    CoordinationGraphBuilder,
    BottleneckDetector,
    CoalitionAnalyzer
)

# 1. Detect agents from your traces
detector = AgentDetector()
agents = await detector.detect_agents(events, auto_register=True)
operations = await detector.detect_agent_operations(events)

# 2. Analyze handoff quality
handoff_analyzer = HandoffAnalyzer()
handoffs = await handoff_analyzer.analyze_handoffs(operations, events)
stats = handoff_analyzer.calculate_handoff_statistics()
print(f"Average handoff quality: {stats['avg_quality_score']:.2f}")

# 3. Detect context drift
drift_detector = ContextDriftDetector()
drift_analyses = await drift_detector.detect_drift(operations, events)
high_drift = drift_detector.get_high_drift_handoffs(threshold=0.4)
if high_drift:
    print(f"⚠️ Found {len(high_drift)} handoffs with high drift!")

# 4. Build coordination graph
graph_builder = CoordinationGraphBuilder()
graph = await graph_builder.build_graph(agents, operations, handoffs, session_id)
print(f"Max depth: {graph.max_depth}, Critical path: {graph.critical_path_ms}ms")

# 5. Detect bottlenecks
bottleneck_detector = BottleneckDetector()
bottlenecks = await bottleneck_detector.detect_bottlenecks(
    agents, operations, handoffs, nodes, edges
)

# 6. Analyze coalitions
coalition_analyzer = CoalitionAnalyzer()
coalitions = await coalition_analyzer.discover_coalitions(
    agents, operations, handoffs, session_id
)
```

### Example Notebook

See **[notebooks/03_multi_agent_intelligence.ipynb](notebooks/03_multi_agent_intelligence.ipynb)** for a complete walkthrough with:
- Simulated multi-agent customer support workflow
- Step-by-step analysis of all 6 capabilities
- Real examples of context drift detection
- Visualization export formats

### Use Cases

**1. Customer Support Optimization**
- Identify low-quality handoffs between agents
- Reduce context loss in agent chains
- Optimize routing decisions

**2. RAG Pipeline Debugging**
- Track information flow through retrieval stages
- Detect when critical context is dropped
- Measure retriever → generator quality

**3. Agent Orchestration Performance**
- Find bottleneck agents slowing down workflows
- Compare different agent team compositions
- Optimize task allocation

**4. A/B Testing Agent Architectures**
- Compare coalition performance
- Measure handoff quality across variants
- Data-driven agent selection

### Why This is Innovative

**No other monitoring tool provides:**
1. ✅ Agent-specific detection and profiling
2. ✅ Handoff quality scoring with multi-component analysis
3. ✅ Context drift detection (information loss tracking)
4. ✅ Coordination graph with bottleneck identification
5. ✅ Coalition discovery and performance analysis

This is **genuinely novel** capability designed for the next generation of LLM applications.

---

## Roadmap

- [x] **MySQL backend implementation** ✅ (v0.1.1)
- [x] **Built-in cost calculation with pricing data** ✅ (v0.1.1)
- [x] **Prometheus exporter** ✅ (v0.2.0)
- [x] **Aggregation server with REST API** ✅ (v0.2.0)
- [x] **Real-time streaming with WebSockets** ✅ (v0.2.0)
- [x] **ClickHouse backend for analytics** ✅ (v0.3.0)
- [x] **GraphQL backend support** ✅ (v0.3.0)
- [x] **ML-based anomaly detection** ✅ (v0.3.0)
- [x] **Datadog integration** ✅ (v0.3.0)
- [x] **Pluggable measurement strategies** ✅ (v0.4.0)
- [x] **Multi-Agent Workflow Intelligence** ✅ (v0.5.0)

## Contributing

Contributions are welcome! Areas of focus:

1. **Storage Backends**: MySQL, ClickHouse, MongoDB, S3, etc.
2. **Collectors**: Cost tracking, latency patterns, cache hit rates
3. **Visualization**: New Grafana dashboards, custom analytics
4. **Documentation**: Tutorials, use cases, best practices

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project synthesizes ideas from:
- OpenTelemetry distributed tracing standards
- Langfuse and LangSmith observability platforms
- Academic research on LLM agent monitoring (AgentOps, LumiMAS)
- Production lessons from the LLM community

## Citation

If you use this in research, please cite:

```bibtex
@software{llamonitor_async,
  title = {llamonitor-async: Lightweight Async Monitoring for LLM Applications},
  author = {Guy Bass},
  year = {2025},
  url = {https://github.com/guybass/LLMOps_monitoring_async-}
}
```

---

**Built with the principle of "leaving space for air conditioning" - designed for the features you'll need tomorrow.**
