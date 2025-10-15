# Quick Start Guide

Get started with LLMOps Monitoring in 5 minutes!

## Step 1: Installation

```bash
# Option 1: Local development with Parquet
pip install 'llamonitor-async[parquet]'

# Option 2: Production with PostgreSQL
pip install 'llamonitor-async[postgres]'

# Option 3: Production with MySQL
pip install 'llamonitor-async[mysql]'

# Option 4: Everything
pip install 'llamonitor-async[all]'
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

### Option B: Use PostgreSQL

Update your code to use PostgreSQL:

```python
config = MonitorConfig(
    storage=StorageConfig(
        backend="postgres",
        connection_string="postgresql://user:pass@localhost:5432/monitoring"
    )
)
```

### Option C: Use MySQL

Update your code to use MySQL:

```python
config = MonitorConfig(
    storage=StorageConfig(
        backend="mysql",
        connection_string="mysql://user:pass@localhost:3306/monitoring"
    )
)
```

### Option D: Use Grafana for Visualization

Start the monitoring stack:

```bash
docker-compose up -d
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

## Step 5: Built-in Cost Tracking

Track LLM costs automatically with the built-in cost collector:

```python
# Enable cost calculation by adding "cost" collector and model name
@monitor_llm(
    operation_name="my_llm_call",
    measure_text=True,
    collectors=["cost"],  # Enable cost tracking
    custom_attributes={
        "model": "gpt-4o-mini"  # Required for cost calculation
    }
)
async def my_llm_call(prompt: str):
    return LLMResponse(text="response...")
```

**Supported Models (18 total):**
- **OpenAI**: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo
- **Anthropic**: claude-3-opus, claude-3-sonnet, claude-3-5-sonnet, claude-3-haiku
- **Google**: gemini-1.5-pro, gemini-1.5-flash, gemini-1.0-pro
- **Meta**: llama-3-8b, llama-3-70b
- **Mistral**: mixtral-8x7b, mistral-small, mistral-medium, mistral-large

**For exact costs**, provide token counts in custom_attributes:

```python
@monitor_llm(
    collectors=["cost"],
    custom_attributes={
        "model": "gpt-4",
        "input_tokens": 150,
        "output_tokens": 500
    }
)
async def precise_cost_tracking():
    ...
```

Query costs:

```python
import pandas as pd
df = pd.read_parquet("./dev_monitoring_data/**/*.parquet")

# Extract cost from custom_attributes
df['cost'] = df['custom_attributes'].apply(lambda x: x.get('estimated_cost_usd'))
print(f"Total cost: ${df['cost'].sum():.6f}")
```

## Step 6: Custom Metrics

For completely custom collectors, extend MetricCollector:

```python
from llmops_monitoring.instrumentation.base import MetricCollector, CollectorRegistry

class MyCustomCollector(MetricCollector):
    def collect(self, result, args, kwargs, context):
        # Your custom logic here
        return {"custom_attributes": {"my_metric": 123}}

    @property
    def metric_type(self) -> str:
        return "custom"

# Register and use
CollectorRegistry.register("my_custom", MyCustomCollector)

@monitor_llm(collectors=["my_custom"])
async def my_function():
    ...
```

## Next Steps

1. **Try the examples**:
   ```bash
   python llmops_monitoring/examples/01_simple_example.py
   python llmops_monitoring/examples/02_agentic_workflow.py
   python llmops_monitoring/examples/03_custom_collector.py
   python llmops_monitoring/examples/04_mysql_backend.py
   python llmops_monitoring/examples/05_cost_calculation.py
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
