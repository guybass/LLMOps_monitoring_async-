# Testing Guide

Complete guide to testing your LLMOps Monitoring system.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# For Option C (real LLM tests)
pip install openai python-dotenv
```

## Option A: Basic Test (No API Needed)

Tests core functionality with simulated responses.

```bash
python test_basic_monitoring.py
```

**What it tests:**
- ✓ Text metrics collection (char_count, word_count, byte_size)
- ✓ Image metrics collection (count, file_size_bytes)
- ✓ Multimodal monitoring
- ✓ Hierarchical tracking (parent-child operations)
- ✓ Parquet file writing

**Output:**
- Creates `./test_monitoring_data/` with Parquet files
- Shows test results in console

---

## Option B: Run Examples

Test with the pre-built examples:

```bash
# Simple example
python llmops_monitoring/examples/01_simple_example.py

# Agentic workflow with hierarchy
python llmops_monitoring/examples/02_agentic_workflow.py

# Custom collector example
python llmops_monitoring/examples/03_custom_collector.py
```

---

## Option C: Real Agent Graph Test

Tests with actual OpenAI API calls in a multi-node agent workflow.

### Setup

1. **Add your API key to `.env`:**

```bash
# Edit .env file
OPENAI_API_KEY=sk-your-key-here
```

2. **Run the test:**

```bash
python test_agent_graph_real.py
```

### What it tests

**Agent Graph Topology:**
```
User Query
    ↓
[Router Agent] ──→ Intent classification
    ↓
[Researcher Agent] ──→ Information gathering
    ↓
[Analyzer Agent] ──→ Data analysis
    ↓
[Synthesizer Agent] ──→ Final response
```

**Measurements per node:**
- INPUT text capacity (characters, words, bytes)
- OUTPUT text capacity
- Duration
- Model used (metadata)

**Hierarchical Tracking:**
- Orchestrator → Individual agents (parent-child)
- Session grouping
- Trace grouping

---

## Analyze Results

After running any test, analyze the collected data:

```bash
python analyze_results.py
```

**What you'll see:**

1. **Summary Statistics**
   - Total events, sessions, traces
   - Text capacity totals (chars, words, bytes)
   - Performance metrics

2. **Metrics by Operation/Node**
   - Breakdown per function/agent
   - Shows which nodes consume most capacity

3. **Graph Topology**
   - Visual tree of parent-child relationships
   - Shows hierarchical execution

4. **Model Usage**
   - Which models were called
   - Frequency of each model

5. **Timeline**
   - Recent events in chronological order

6. **Exported Report**
   - Detailed report saved to `monitoring_report.txt`

---

## Example Output

### test_basic_monitoring.py
```
✓ Monitoring initialized (Parquet backend)
  Output: ./test_monitoring_data

Test 1: Text Monitoring
  ✓ Generated 2,500 characters
  ✓ ~417 words

Test 2: Image Monitoring
  ✓ Processed 2 images
  ✓ Total size: 45,500 bytes

...

✓ Created 1 Parquet file(s)
```

### test_agent_graph_real.py
```
Processing: 'What are the key benefits of async programming?'

  [Router] Processing query...
  [Router] Output: 185 chars, 29 words

  [Researcher] Gathering information...
  [Researcher] Output: 1,247 chars, 198 words

  [Analyzer] Analyzing data...
  [Analyzer] Output: 843 chars, 127 words

  [Synthesizer] Creating final response...
  [Synthesizer] Output: 512 chars, 79 words

Workflow Complete!
```

### analyze_results.py
```
SUMMARY STATISTICS
Total Events:          16
Total Text Characters: 12,450
Total Text Words:      1,987
Avg Duration:          1,234.5 ms

METRICS BY OPERATION / NODE
                      Count  Total Chars  Total Words  Avg Duration
orchestrator              3        2,500          398      5,123.4
router_agent              3          555           87        342.1
researcher_agent          3        3,741          594      1,876.5
analyzer_agent            3        2,529          381      1,543.2
synthesizer_agent         3        1,536          237      1,234.8

GRAPH TOPOLOGY
Session: test-session-1
  orchestrator (2500 chars, 5123.4ms)
    └─ router_agent (185 chars, 342.1ms)
    └─ researcher_agent (1247 chars, 1876.5ms)
    └─ analyzer_agent (843 chars, 1543.2ms)
    └─ synthesizer_agent (512 chars, 1234.8ms)
```

---

## What to Look For

### ✅ Correct Measurements
- Character counts match actual output
- Word counts are reasonable
- Byte sizes ≥ character counts (UTF-8)

### ✅ Hierarchy Tracking
- Parent operations show child operations
- Session/trace IDs group related operations
- Span parent-child links are correct

### ✅ Model Metadata
- Model names stored in `custom_attributes`
- NOT used for measurement (that's based on text/image capacity)

### ✅ Performance
- Events written asynchronously (no blocking)
- Batch writes to Parquet
- Overhead < 1% of total execution time

---

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install pydantic aiofiles pandas pyarrow
```

### "OPENAI_API_KEY not found"
Add your key to `.env`:
```
OPENAI_API_KEY=sk-your-key-here
```

### "No Parquet files found"
Run a test first to generate data:
```bash
python test_basic_monitoring.py
```

### Events not appearing
Wait for flush interval (2 seconds by default), or increase wait time before checking files.

---

## Next Steps

1. ✅ Run basic test: `python test_basic_monitoring.py`
2. ✅ Analyze results: `python analyze_results.py`
3. ✅ Add API key to `.env`
4. ✅ Run graph test: `python test_agent_graph_real.py`
5. ✅ Analyze again: `python analyze_results.py`
6. 🚀 Integrate into your own application!

---

## Integration Example

```python
from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig

# Initialize once at startup
await initialize_monitoring(MonitorConfig.for_local_dev())

# Decorate your LLM functions
@monitor_llm(
    operation_name="my_llm_call",
    measure_text=True,
    custom_attributes={"model": "gpt-4"}
)
async def my_llm_function(prompt: str):
    # Your LLM API call
    return response
```

That's it! Monitoring happens automatically in the background.
