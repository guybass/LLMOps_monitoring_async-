# LLMOps Monitoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

An **extensible, async-first monitoring framework** for LLM applications. Measures text and image usage with pluggable storage backends and automatic hierarchical tracking.

## Design Philosophy: "Leave Space for Air Conditioning"

Every component has clear extension points for future enhancements. Whether you need custom metric collectors, new storage backends, or specialized aggregation strategies, the architecture supports growth without breaking existing code.

## Features

- **Async-First**: Non-blocking metric collection with buffered batch writes
- **Hierarchical Tracking**: Automatic parent-child relationships across nested operations
- **Flexible Metrics**: Measure text (characters, words, bytes) and images (count, pixels, file size)
- **Pluggable Storage**: Local Parquet, PostgreSQL, MySQL (easily add more)
- **Simple API**: Single decorator for most use cases
- **Production-Ready**: Error handling, retries, graceful shutdown
- **Extensible**: Custom collectors, backends, and aggregation strategies

## Quick Start

### Installation

```bash
# Basic installation
pip install llmops-monitoring

# With storage backends
pip install llmops-monitoring[parquet]    # For local Parquet files
pip install llmops-monitoring[postgres]   # For PostgreSQL
pip install llmops-monitoring[all]        # Everything
```

### Basic Usage

```python
import asyncio
from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig

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
│  • MetricCollectors (text, image, custom)                   │
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
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              Storage Backend                                │
│  • Parquet (local files)                                    │
│  • PostgreSQL (production)                                  │
│  • Custom backends                                          │
└─────────────────────────────────────────────────────────────┘
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

### Custom Metrics

```python
from llmops_monitoring.instrumentation.base import MetricCollector, CollectorRegistry

class CostCollector(MetricCollector):
    def collect(self, result, args, kwargs, context):
        # Your cost calculation logic
        return {"custom_attributes": {"cost_usd": 0.002}}

    @property
    def metric_type(self) -> str:
        return "cost"

# Register
CollectorRegistry.register("cost", CostCollector)

# Use
@monitor_llm(collectors=["cost"])
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

# Start monitoring stack
docker-compose up -d
```

## Roadmap

- [ ] MySQL backend implementation
- [ ] ClickHouse backend for analytics
- [ ] GraphQL backend support
- [ ] Real-time streaming with WebSockets
- [ ] Built-in cost calculation with pricing data
- [ ] ML-based anomaly detection
- [ ] Aggregation server with REST API
- [ ] Prometheus exporter
- [ ] Datadog integration

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
@software{llmops_monitoring,
  title = {LLMOps Monitoring: Extensible Async Framework for LLM Observability},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/guybass/LLMOps_monitoring_async-}
}
```

---

**Built with the principle of "leaving space for air conditioning" - designed for the features you'll need tomorrow.**
