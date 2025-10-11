# Architecture Documentation

## Overview

LLMOps Monitoring is built with three core layers:

1. **Instrumentation Layer**: Decorators and metric collectors
2. **Transport Layer**: Async queue and batching
3. **Storage Layer**: Pluggable backends

## Design Principles

### 1. "Leave Space for Air Conditioning"

Every component is designed for future extension:

- **Metric Collectors**: Abstract `MetricCollector` interface
- **Storage Backends**: Abstract `StorageBackend` interface
- **Transport Mechanisms**: Swappable queue implementations
- **Context Management**: Extensible span attributes

### 2. Separation of Concerns

- **Measurement** happens in application code (instrumentation)
- **Delivery** happens asynchronously (transport)
- **Analysis** happens separately (storage + visualization)

### 3. Never Break User Code

- Decorator always returns original result
- Errors in monitoring never crash application
- `fail_silently=True` by default

## Layer Details

### Instrumentation Layer

```
@monitor_llm decorator
    │
    ├─→ SpanContext (contextvars)
    │   ├─ session_id
    │   ├─ trace_id
    │   ├─ span_id
    │   └─ parent_span_id
    │
    ├─→ MetricCollectors
    │   ├─ TextCollector
    │   ├─ ImageCollector
    │   └─ CustomCollectors...
    │
    └─→ MetricEvent (Pydantic model)
```

**Key Files**:
- `instrumentation/decorators.py`: Main decorator logic
- `instrumentation/context.py`: Hierarchical tracking
- `instrumentation/base.py`: Collector interface
- `instrumentation/collectors/`: Built-in collectors

### Transport Layer

```
MetricEvent
    │
    ↓
AsyncQueue (maxsize=10000)
    │
    ↓
Background Worker
    ├─ Buffering (batch_size)
    ├─ Flush Interval
    └─ Retry Logic
    │
    ↓
StorageBackend
```

**Key Files**:
- `transport/writer.py`: MonitoringWriter singleton
- `transport/backends/base.py`: Backend interface
- `transport/backends/parquet.py`: Parquet implementation
- `transport/backends/postgres.py`: PostgreSQL implementation

### Storage Layer

#### Parquet Backend

- **Structure**: `output_dir/YYYY-MM-DD/events_*.parquet`
- **Partitioning**: By date or session_id
- **Format**: Columnar, compressed (Snappy)
- **Use Case**: Local development, data science

#### PostgreSQL Backend

- **Table**: `metric_events`
- **Indexes**:
  - `(session_id, timestamp DESC)`
  - `(trace_id, timestamp DESC)`
  - `(parent_span_id)`
  - `(operation_name, timestamp DESC)`
- **Use Case**: Production deployments

## Data Flow

1. **Function Call**
   ```python
   @monitor_llm(...)
   async def my_func():
       return result
   ```

2. **Span Creation**
   - Create SpanContext
   - Inherit session/trace from parent
   - Generate unique span_id

3. **Function Execution**
   - Execute original function
   - Measure duration
   - Capture result

4. **Metric Collection**
   - Run all collectors
   - Extract metrics from result
   - Build MetricEvent

5. **Async Write**
   - Push event to queue (non-blocking)
   - Return control to application
   - Background worker handles delivery

6. **Storage**
   - Worker batches events
   - Flushes on batch size or interval
   - Retries on failure (optional)

## Hierarchical Tracking

Uses Python's `contextvars` for automatic propagation:

```python
# Parent operation
with SpanContext(session_id="s1", trace_id="t1") as parent:
    parent_span_id = None  # Root

    # Child operation (automatic inheritance)
    with SpanContext() as child:
        child.session_id == "s1"  # Inherited
        child.trace_id == "t1"     # Inherited
        child.parent_span_id == parent.span_id  # Linked
```

Works seamlessly across async/await:

```python
@monitor_llm("parent")
async def parent_op():
    await child_op()  # Context propagates

@monitor_llm("child")
async def child_op():
    # Automatically linked to parent
    pass
```

## Extension Points

### 1. Custom Metric Collectors

```python
class MyCollector(MetricCollector):
    def collect(self, result, args, kwargs, context):
        # Extract your metrics
        return {"custom_attributes": {...}}

    @property
    def metric_type(self) -> str:
        return "my_metric"

# Register
CollectorRegistry.register("my_metric", MyCollector)

# Use
@monitor_llm(collectors=["my_metric"])
async def my_func():
    pass
```

### 2. Custom Storage Backends

```python
class MyBackend(StorageBackend):
    async def initialize(self):
        # Setup connections
        pass

    async def write_event(self, event: MetricEvent):
        # Single write
        pass

    async def write_batch(self, events: List[MetricEvent]):
        # Batch write (more efficient)
        pass

    async def close(self):
        # Cleanup
        pass
```

Add to `transport/writer.py`:

```python
elif backend_type == "my_backend":
    from .backends.my_backend import MyBackend
    return MyBackend(self.config.storage)
```

### 3. Custom Transport Mechanisms

Replace `asyncio.Queue` with Kafka, Redis, etc.:

```python
class KafkaTransport:
    async def put(self, event: MetricEvent):
        # Publish to Kafka
        pass

# Modify MonitoringWriter.__init__ to use KafkaTransport
```

### 4. Schema Evolution

Add new metric types:

```python
# 1. Add Pydantic model
class AudioMetrics(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int

# 2. Update MetricEvent
class MetricEvent(BaseModel):
    ...
    audio_metrics: Optional[AudioMetrics] = None

# 3. Create collector
class AudioCollector(MetricCollector):
    ...

# 4. Register
CollectorRegistry.register("audio", AudioCollector)
```

## Performance Characteristics

### Overhead

- **Decorator**: ~0.1ms per call
- **Context management**: ~0.05ms
- **Metric collection**: ~0.5ms (depends on collectors)
- **Queue write**: ~0.01ms (non-blocking)
- **Total**: ~0.7ms per monitored operation

### Throughput

- **Queue capacity**: Configurable (default 10,000)
- **Batch size**: Configurable (default 100)
- **Flush interval**: Configurable (default 5s)
- **Max throughput**: ~20,000 events/second

### Scaling

- **Horizontal**: Run multiple instances with shared storage
- **Vertical**: Increase queue size and batch size
- **Storage**: PostgreSQL handles millions of events

## Security Considerations

- **PII**: Custom collectors can filter sensitive data
- **Connection strings**: Use environment variables
- **Network**: TLS for database connections
- **Access**: Database user should have minimal privileges

## Monitoring the Monitor

```python
writer = MonitoringWriter.get_instance_sync()
health = await writer.health_check()

# Returns:
{
    "running": True,
    "queue_size": 42,
    "buffer_size": 15,
    "backend_healthy": True
}
```

## Future Enhancements

1. **Aggregation Server**
   - REST API for querying
   - Pre-computed aggregations
   - Real-time dashboards

2. **ML-based Analysis**
   - Anomaly detection
   - Cost optimization suggestions
   - Performance profiling

3. **Multi-tenant Support**
   - Org/team isolation
   - Usage quotas
   - Access control

4. **Streaming Integration**
   - Kafka connector
   - WebSocket streaming
   - Real-time alerts
