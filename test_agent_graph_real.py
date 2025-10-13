"""
Agent Graph Test with Real LLM Calls - Option C

Tests realistic multi-node agent workflow with actual OpenAI API calls.
Focuses on measuring INPUT/OUTPUT CAPACITY (text amounts, not tokens).

Prerequisites:
1. Add OPENAI_API_KEY to .env file
2. Install: pip install openai python-dotenv

Run with: python test_agent_graph_real.py
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig
from llmops_monitoring.instrumentation.context import monitoring_session, monitoring_trace
from llmops_monitoring.schema.config import StorageConfig


# Check for OpenAI
try:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    print("ERROR: OpenAI not installed. Run: pip install openai")
    sys.exit(1)

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in .env file")
    print("Please add your OpenAI API key to .env")
    sys.exit(1)


# ============================================================================
# AGENT GRAPH NODES
# Each node represents a function/method in your agentic system
# ============================================================================

@monitor_llm(
    operation_name="router_agent",
    operation_type="agent_node",
    measure_text=True,
    custom_attributes={"model": "gpt-4o-mini", "node_type": "router", "graph_position": "entry"}
)
async def router_agent(user_query: str) -> dict:
    """
    Node 1: Routes user query to appropriate workflow.
    INPUT: User query (text)
    OUTPUT: Routing decision + extracted intent (text)
    """
    print("  [Router] Processing query...")

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a routing agent. Classify the query intent and suggest next steps. Be concise."},
            {"role": "user", "content": user_query}
        ],
        max_tokens=200
    )

    result_text = response.choices[0].message.content
    print(f"  [Router] Output: {len(result_text)} chars, {len(result_text.split())} words")

    return {"text": result_text, "intent": "research"}


@monitor_llm(
    operation_name="researcher_agent",
    operation_type="agent_node",
    measure_text=True,
    custom_attributes={"model": "gpt-4o-mini", "node_type": "researcher", "graph_position": "middle"}
)
async def researcher_agent(intent: str, query: str) -> dict:
    """
    Node 2: Researches and gathers information.
    INPUT: Intent + original query (text)
    OUTPUT: Research findings (text)
    """
    print("  [Researcher] Gathering information...")

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a research agent. Provide detailed information on the topic. Be thorough."},
            {"role": "user", "content": f"Research this query: {query}\nIntent: {intent}"}
        ],
        max_tokens=500
    )

    result_text = response.choices[0].message.content
    print(f"  [Researcher] Output: {len(result_text)} chars, {len(result_text.split())} words")

    return {"text": result_text}


@monitor_llm(
    operation_name="analyzer_agent",
    operation_type="agent_node",
    measure_text=True,
    custom_attributes={"model": "gpt-4o-mini", "node_type": "analyzer", "graph_position": "middle"}
)
async def analyzer_agent(research_data: str) -> dict:
    """
    Node 3: Analyzes research findings.
    INPUT: Research data (text)
    OUTPUT: Analysis and key points (text)
    """
    print("  [Analyzer] Analyzing data...")

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an analysis agent. Extract key insights and patterns. Be analytical."},
            {"role": "user", "content": f"Analyze this information:\n\n{research_data[:1000]}"}  # Limit input
        ],
        max_tokens=400
    )

    result_text = response.choices[0].message.content
    print(f"  [Analyzer] Output: {len(result_text)} chars, {len(result_text.split())} words")

    return {"text": result_text}


@monitor_llm(
    operation_name="synthesizer_agent",
    operation_type="agent_node",
    measure_text=True,
    custom_attributes={"model": "gpt-4o-mini", "node_type": "synthesizer", "graph_position": "exit"}
)
async def synthesizer_agent(analysis: str, original_query: str) -> dict:
    """
    Node 4: Synthesizes final response.
    INPUT: Analysis + original query (text)
    OUTPUT: Final user-facing response (text)
    """
    print("  [Synthesizer] Creating final response...")

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a synthesis agent. Create a clear, helpful response for the user."},
            {"role": "user", "content": f"Original query: {original_query}\n\nAnalysis:\n{analysis}\n\nProvide a helpful answer."}
        ],
        max_tokens=300
    )

    result_text = response.choices[0].message.content
    print(f"  [Synthesizer] Output: {len(result_text)} chars, {len(result_text.split())} words")

    return {"text": result_text}


# ============================================================================
# ORCHESTRATOR - Coordinates the agent graph
# ============================================================================

@monitor_llm(
    operation_name="orchestrator",
    operation_type="workflow",
    measure_text=True,
    custom_attributes={"workflow": "agent_graph"}
)
async def orchestrator(user_query: str) -> str:
    """
    Orchestrates the entire agent graph workflow.
    All child nodes are automatically tracked hierarchically.

    Graph topology:
    User Query → Router → Researcher → Analyzer → Synthesizer → Response
    """
    print(f"\n{'='*60}")
    print(f"Processing: '{user_query}'")
    print(f"{'='*60}\n")

    # Node 1: Route
    route_result = await router_agent(user_query)

    # Node 2: Research
    research_result = await researcher_agent(route_result["intent"], user_query)

    # Node 3: Analyze
    analysis_result = await analyzer_agent(research_result["text"])

    # Node 4: Synthesize
    final_result = await synthesizer_agent(analysis_result["text"], user_query)

    print(f"\n{'='*60}")
    print("Workflow Complete!")
    print(f"{'='*60}\n")

    return final_result["text"]


# ============================================================================
# TEST EXECUTION
# ============================================================================

async def run_graph_tests():
    """Run the agent graph tests."""
    print("=" * 70)
    print("LLMOps Monitoring - Agent Graph Test (Real LLM Calls)")
    print("=" * 70)
    print()
    print("This test demonstrates:")
    print("  ✓ Multi-node agent workflow")
    print("  ✓ Hierarchical tracking (orchestrator → nodes)")
    print("  ✓ Text capacity measurement (chars, words, bytes)")
    print("  ✓ Model metadata tracking")
    print("  ✓ Graph topology preservation")
    print()

    # Initialize monitoring
    config = MonitorConfig(
        storage=StorageConfig(
            backend="parquet",
            output_dir="./test_monitoring_data",
            batch_size=20,
            flush_interval_seconds=2.0
        )
    )
    writer = await initialize_monitoring(config)
    print("✓ Monitoring initialized")
    print()

    # Test queries
    test_queries = [
        "What are the key benefits of async programming in Python?",
        "How does OpenTelemetry help with distributed tracing?",
        "Explain the concept of 'leave space for air conditioning' in software design"
    ]

    # Run tests with session/trace context
    for i, query in enumerate(test_queries):
        session_id = f"test-session-{i+1}"
        trace_id = f"trace-query-{i+1}"

        with monitoring_session(session_id):
            with monitoring_trace(trace_id):
                try:
                    result = await orchestrator(query)
                    print(f"Final Response ({len(result)} chars):")
                    print(f"  {result[:200]}...")
                    print()
                except Exception as e:
                    print(f"ERROR: {e}")
                    print()

    # Flush events
    print("Flushing events to storage...")
    await asyncio.sleep(3)
    await writer.stop()
    print("✓ All events written")
    print()

    # Summary
    output_path = Path(config.storage.output_dir)
    if output_path.exists():
        parquet_files = list(output_path.rglob("*.parquet"))
        print(f"✓ Created {len(parquet_files)} Parquet file(s) in {output_path}")
        print()

    print("=" * 70)
    print("Graph Test Complete!")
    print("=" * 70)
    print()
    print("Next: Analyze the results")
    print("  python analyze_results.py")
    print()
    print("What to look for:")
    print("  • Each node (router, researcher, analyzer, synthesizer) as separate spans")
    print("  • Parent-child relationships (orchestrator → nodes)")
    print("  • Text capacity metrics (char_count, word_count, byte_size)")
    print("  • Model names in custom_attributes")
    print("  • Session/trace grouping")


if __name__ == "__main__":
    try:
        asyncio.run(run_graph_tests())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
