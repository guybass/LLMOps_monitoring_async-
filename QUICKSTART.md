# Quick Start Guide

Get started with LLMOps Monitoring in 5 minutes!

## Step 1: Installation

```bash
# Option 1: Local development with Parquet
pip install pydantic aiofiles python-dotenv pandas pyarrow

# Option 2: Production with PostgreSQL
pip install pydantic aiofiles python-dotenv asyncpg sqlalchemy
```

## Step 2: Basic Usage

Create a file `test_monitoring.py`:

```python
import asyncio
from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig

# Simulate an LLM response
class LLMResponse:
    def __init__(self, text: str):
        self.text = text

# Add monitoring decorator
@monitor_llm(
    operation_name="generate_text",
    measure_text=True,
    custom_attributes={"model": "gpt-4"}
)
async def my_llm_call(prompt: str) -> LLMResponse:
    await asyncio.sleep(0.1)  # Simulate API call
    return LLMResponse(text=f"Response to: {prompt} " * 20)

async def main():
    # Initialize monitoring
    config = MonitorConfig.for_local_dev()
    writer = await initialize_monitoring(config)

    # Make some calls
    for i in range(5):
        response = await my_llm_call(f"Query {i}")
        print(f"Call {i}: {len(response.text)} characters")

    # Wait for events to flush
    await asyncio.sleep(3)
    await writer.stop()

    print("\nDone! Check ./dev_monitoring_data for Parquet files.")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python test_monitoring.py
```

You'll see Parquet files created in `./dev_monitoring_data/YYYY-MM-DD/`

## Step 3: View the Data

### Option A: Query with Pandas

```python
import pandas as pd
from glob import glob

# Read all parquet files
files = glob("./dev_monitoring_data/**/*.parquet", recursive=True)
df = pd.concat([pd.read_parquet(f) for f in files])

print(df[['operation_name', 'text_char_count', 'text_word_count', 'duration_ms']])
```

### Option B: Use PostgreSQL + Grafana

Start the stack:

```bash
docker-compose up -d
```

Update your code to use PostgreSQL:

```python
config = MonitorConfig(
    storage=StorageConfig(
        backend="postgres",
        connection_string="postgresql://monitoring:monitoring_password@localhost:5432/llm_monitoring"
    )
)
```

Access Grafana at `http://localhost:3000` (admin/admin)

## Step 4: Hierarchical Tracking

Track nested operations automatically:

```python
from llmops_monitoring.instrumentation.context import monitoring_session

@monitor_llm("step_1")
async def step_1():
    return "result 1"

@monitor_llm("step_2")
async def step_2():
    return "result 2"

@monitor_llm("workflow")
async def run_workflow():
    # These are automatically linked as children
    r1 = await step_1()
    r2 = await step_2()
    return f"{r1} + {r2}"

# Group operations by session
with monitoring_session("user-123"):
    result = await run_workflow()
```

Query hierarchical data:

```sql
SELECT
    session_id,
    trace_id,
    span_id,
    parent_span_id,
    operation_name,
    duration_ms
FROM metric_events
WHERE session_id = 'user-123'
ORDER BY timestamp;
```

## Step 5: Custom Metrics

Add your own collectors:

```python
from llmops_monitoring.instrumentation.base import MetricCollector, CollectorRegistry

class CostCollector(MetricCollector):
    def collect(self, result, args, kwargs, context):
        # Calculate cost based on your logic
        text_length = len(result.text) if hasattr(result, 'text') else 0
        cost = (text_length / 1000) * 0.002  # $0.002 per 1k chars

        return {
            "custom_attributes": {
                "estimated_cost_usd": round(cost, 6)
            }
        }

    @property
    def metric_type(self) -> str:
        return "cost"

# Register
CollectorRegistry.register("cost", CostCollector)

# Use it
@monitor_llm(collectors=["cost"])
async def my_function():
    return LLMResponse(text="response...")
```

## Next Steps

1. **Try the examples**:
   ```bash
   python llmops_monitoring/examples/01_simple_example.py
   python llmops_monitoring/examples/02_agentic_workflow.py
   python llmops_monitoring/examples/03_custom_collector.py
   ```

2. **Read the architecture docs**: `llmops_monitoring/docs/ARCHITECTURE.md`

3. **Explore Grafana dashboards**: `http://localhost:3000`

4. **Create your own collectors**: See `CONTRIBUTING.md`

## Troubleshooting

### "Module not found"

Make sure you installed the dependencies:
```bash
pip install -r requirements.txt
```

### "Cannot connect to PostgreSQL"

Make sure Docker is running:
```bash
docker-compose ps
docker-compose up -d
```

### "Queue is full"

Increase queue size in config:
```python
config = MonitorConfig(max_queue_size=50000)
```

### "Events not appearing"

Make sure you wait for flush:
```python
await asyncio.sleep(config.storage.flush_interval_seconds + 1)
await writer.stop()  # Graceful shutdown with flush
```

## Common Patterns

### Pattern 1: Session Management

```python
with monitoring_session(f"user-{user_id}"):
    # All operations grouped by user session
    result = await process_user_request(request)
```

### Pattern 2: Multiple Traces per Session

```python
with monitoring_session(f"user-{user_id}"):
    with monitoring_trace(f"conversation-{conv_id}"):
        # Nested grouping
        response = await handle_message(message)
```

### Pattern 3: Custom Attributes

```python
@monitor_llm(
    custom_attributes={
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000
    }
)
async def my_call():
    ...
```

### Pattern 4: Selective Measurement

```python
@monitor_llm(
    measure_text=["char_count", "word_count"],  # Only these
    measure_images=False  # Skip image metrics
)
async def text_only():
    ...
```

Happy monitoring! 🚀
