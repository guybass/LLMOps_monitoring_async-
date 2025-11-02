"""
Example 19: Token Consumption by Code Component

Demonstrates the complete integration of topology scanning with runtime monitoring
to identify which functions/modules consume the most tokens.

This enables data-driven optimization of LLM applications:
- Scan repository to understand code structure
- Track token consumption during runtime
- Correlate metrics with topology
- Identify optimization opportunities
- Generate actionable insights

Flow:
1. Scan repository topology
2. Simulate LLM operations with monitoring
3. Correlate token consumption with code components
4. Analyze hotspots and optimization candidates
5. Generate reports and visualizations
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4

# Topology components
from llmops_monitoring.topology import (
    CodeParser,
    DependencyGraphBuilder,
    TokenConsumptionCorrelator,
    TopologyVisualizer
)

# Monitoring components
from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.schema.config import MonitoringConfig, StorageConfig
from llmops_monitoring.instrumentation.decorators import monitor_llm
from llmops_monitoring import LLMOpsMonitor


# ========== Simulated Application Code ==========
# These functions represent your actual LLM application


async def generate_text_summary(text: str) -> str:
    """
    Simulate a text summarization function.
    In real code, this would call an LLM.
    """
    # Simulate tokens (in reality, captured by @monitor_llm)
    tokens = len(text.split()) * 1.5
    return f"Summary of {len(text)} chars (simulated {int(tokens)} tokens)"


async def analyze_sentiment(text: str) -> str:
    """Simulate sentiment analysis."""
    tokens = len(text.split()) * 1.2
    return f"Sentiment: positive (simulated {int(tokens)} tokens)"


async def extract_entities(text: str) -> list:
    """Simulate entity extraction."""
    tokens = len(text.split()) * 2.0
    return [f"Entity{i}" for i in range(3)]


async def complex_workflow(input_data: str) -> dict:
    """
    Complex workflow that calls multiple LLM functions.
    This represents a realistic application pattern.
    """
    results = {}

    # Step 1: Summarize
    results["summary"] = await generate_text_summary(input_data)

    # Step 2: Analyze sentiment
    results["sentiment"] = await analyze_sentiment(input_data)

    # Step 3: Extract entities
    results["entities"] = await extract_entities(input_data)

    # Step 4: Generate final report (high token consumer)
    report_prompt = f"Based on: {results}"
    report_tokens = len(report_prompt.split()) * 3.5  # Simulate expensive operation
    results["report"] = f"Final report (simulated {int(report_tokens)} tokens)"

    return results


# ========== Main Example ==========


async def run_topology_token_analysis():
    """
    Complete example: Scan topology + Track tokens + Analyze correlation.
    """
    print("=" * 80)
    print("Example 19: Token Consumption by Code Component")
    print("=" * 80)
    print()

    # ========== Step 1: Scan Repository Topology ==========
    print("🔍 Step 1: Scanning Repository Topology")
    print("-" * 80)

    repo_path = Path(__file__).parent.parent.parent
    print(f"Repository: {repo_path}")
    print()

    parser = CodeParser()
    modules = parser.parse_repository(
        str(repo_path),
        include_patterns=["**/*.py"],
        exclude_patterns=[
            "**/test_*.py",
            "**/tests/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/venv/**",
            "**/dist/**",
            "**/build/**"
        ]
    )

    builder = DependencyGraphBuilder()
    topology = builder.build_repository_topology(modules)

    print(f"✅ Topology scanned:")
    print(f"   - {topology.metrics.total_modules} modules")
    print(f"   - {topology.metrics.total_functions} functions")
    print(f"   - {topology.metrics.total_classes} classes")
    print(f"   - {topology.metrics.total_lines:,} lines of code")
    print()

    # ========== Step 2: Initialize Token Correlator ==========
    print("🔗 Step 2: Initializing Token Consumption Correlator")
    print("-" * 80)

    correlator = TokenConsumptionCorrelator()
    correlator.load_topology(topology)

    print(f"✅ Correlator ready:")
    print(f"   - Tracking {len(correlator.tracked_functions)} functions")
    print(f"   - Tracking {len(correlator.tracked_modules)} modules")
    print()

    # ========== Step 3: Simulate Runtime with Monitoring ==========
    print("🚀 Step 3: Simulating Runtime Operations with Monitoring")
    print("-" * 80)

    # Simulate various operations with different token consumption patterns
    simulated_events = []

    # High-frequency, low-token function
    for i in range(50):
        event = MetricEvent(
            event_id=uuid4(),
            session_id=uuid4(),
            parent_event_id=None,
            timestamp=datetime.now(),
            event_type="completion",
            model_name="gpt-4",
            prompt="Short prompt",
            completion="Short response",
            prompt_tokens=10,
            completion_tokens=15,
            total_tokens=25,
            latency_ms=100.0,
            cost_usd=0.0005,
            metadata={"function_name": "generate_text_summary"}
        )
        simulated_events.append(event)
        correlator.track_event(event)

    # Medium-frequency, medium-token function
    for i in range(30):
        event = MetricEvent(
            event_id=uuid4(),
            session_id=uuid4(),
            parent_event_id=None,
            timestamp=datetime.now(),
            event_type="completion",
            model_name="gpt-4",
            prompt="Medium length prompt for sentiment analysis",
            completion="Detailed sentiment response",
            prompt_tokens=25,
            completion_tokens=40,
            total_tokens=65,
            latency_ms=150.0,
            cost_usd=0.0013,
            metadata={"function_name": "analyze_sentiment"}
        )
        simulated_events.append(event)
        correlator.track_event(event)

    # Low-frequency, high-token function (optimization candidate!)
    for i in range(10):
        event = MetricEvent(
            event_id=uuid4(),
            session_id=uuid4(),
            parent_event_id=None,
            timestamp=datetime.now(),
            event_type="completion",
            model_name="gpt-4",
            prompt="Very long prompt for entity extraction with lots of context " * 20,
            completion="Extensive entity extraction results " * 30,
            prompt_tokens=400,
            completion_tokens=600,
            total_tokens=1000,
            latency_ms=800.0,
            cost_usd=0.02,
            metadata={"function_name": "extract_entities"}
        )
        simulated_events.append(event)
        correlator.track_event(event)

    # Complex workflow (high-token, high-complexity)
    for i in range(15):
        event = MetricEvent(
            event_id=uuid4(),
            session_id=uuid4(),
            parent_event_id=None,
            timestamp=datetime.now(),
            event_type="completion",
            model_name="gpt-4",
            prompt="Complex workflow with multiple steps " * 25,
            completion="Comprehensive workflow results " * 35,
            prompt_tokens=500,
            completion_tokens=700,
            total_tokens=1200,
            latency_ms=1000.0,
            cost_usd=0.024,
            metadata={"function_name": "complex_workflow"}
        )
        simulated_events.append(event)
        correlator.track_event(event)

    print(f"✅ Simulated {len(simulated_events)} LLM operations")
    print(f"   - Total tokens tracked: {sum(e.total_tokens for e in simulated_events):,}")
    print(f"   - Total cost: ${sum(e.cost_usd for e in simulated_events):.4f}")
    print()

    # ========== Step 4: Analyze Top Token Consumers ==========
    print("📊 Step 4: Analyzing Top Token Consumers")
    print("-" * 80)

    top_functions = correlator.get_top_token_consumers(limit=10, component_type="function")

    print(f"\n🔥 Top 10 Functions by Token Consumption:")
    print()
    for i, usage in enumerate(top_functions, 1):
        func = usage.function_info
        tokens = usage.token_consumption

        print(f"  {i}. {func.name}")
        print(f"     📍 {func.qualified_name}")
        print(f"     💰 {tokens.total_tokens:,} tokens (${tokens.total_cost_usd:.4f})")
        print(f"     📞 {tokens.total_calls} calls")
        print(f"     📈 {tokens.avg_tokens_per_call:.1f} avg tokens/call")
        print(f"     📏 {usage.tokens_per_line:.1f} tokens/line")
        print(f"     🔧 Complexity: {func.complexity:.1f}")
        print()

    # Module-level analysis
    print("\n📦 Top 5 Modules by Token Consumption:")
    print()
    top_modules = correlator.get_top_token_consumers(limit=5, component_type="module")

    for i, usage in enumerate(top_modules, 1):
        print(f"  {i}. {usage.function_info.name}")
        print(f"     Total: {usage.token_consumption.total_tokens:,} tokens")
        print(f"     Calls: {usage.token_consumption.total_calls}")
        print()

    # ========== Step 5: Find Optimization Candidates ==========
    print("🎯 Step 5: Identifying Optimization Candidates")
    print("-" * 80)

    candidates = correlator.find_optimization_candidates(
        min_tokens=500,
        min_calls=5
    )

    print(f"\n⚡ Found {len(candidates)} optimization opportunities:")
    print()

    for i, usage in enumerate(candidates[:5], 1):
        func = usage.function_info
        tokens = usage.token_consumption

        print(f"  {i}. {func.name}")
        print(f"     Optimization Score: {usage.optimization_potential:.2f} / 1.00")
        print(f"     Total Tokens: {tokens.total_tokens:,}")
        print(f"     Total Cost: ${tokens.total_cost_usd:.4f}")
        print(f"     Complexity: {func.complexity:.1f}")
        print(f"     Tokens/Line: {usage.tokens_per_line:.1f}")
        print(f"     💡 Potential Impact: High" if usage.optimization_potential > 0.7 else "     💡 Potential Impact: Medium")
        print()

    # ========== Step 6: Comprehensive Hotspot Analysis ==========
    print("🔍 Step 6: Comprehensive Hotspot Analysis")
    print("-" * 80)

    hotspots = correlator.analyze_hotspots(top_n=20)

    print(f"\n📈 Analysis Summary:")
    print(f"   - Total tokens analyzed: {hotspots.total_tokens_analyzed:,}")
    print(f"   - High complexity + high tokens: {len(hotspots.high_complexity_high_tokens)} functions")
    print(f"   - Optimization candidates: {len(hotspots.optimization_candidates)}")
    print(f"   - Potential savings: {hotspots.potential_savings:.1f}%")
    print()

    print("💡 Recommendations:")
    print()
    for i, recommendation in enumerate(hotspots.recommendations, 1):
        print(f"   {i}. {recommendation}")
    print()

    # ========== Step 7: Generate Detailed Reports ==========
    print("📄 Step 7: Generating Detailed Reports")
    print("-" * 80)

    # Export correlation data
    correlation_data = correlator.export_correlation_data()

    output_dir = repo_path / "reports"
    output_dir.mkdir(exist_ok=True)

    correlation_file = output_dir / "token_consumption_correlation.json"
    with open(correlation_file, 'w') as f:
        json.dump(correlation_data, f, indent=2)

    print(f"✅ Correlation data exported to: {correlation_file}")

    # Generate function report for top consumer
    if top_functions:
        top_func = top_functions[0]
        report = correlator.get_function_report(top_func.function_info.qualified_name)

        if report:
            report_file = output_dir / "top_consumer_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"✅ Top consumer report: {report_file}")

    print()

    # ========== Step 8: Generate Visualization ==========
    print("📊 Step 8: Generating Visualization")
    print("-" * 80)

    visualizer = TopologyVisualizer()

    # Export interactive HTML with token data overlay
    html_file = output_dir / "token_consumption_graph.html"
    visualizer.export_html_viewer(
        topology,
        str(html_file),
        visualization_type="d3-force"
    )

    print(f"✅ Interactive visualization: {html_file}")
    print(f"   Open in browser to explore token consumption by component")
    print()

    # ========== Step 9: Persist to Database (Optional) ==========
    print("💾 Step 9: Persisting to Database (Optional)")
    print("-" * 80)
    print("To persist this data to PostgreSQL:")
    print()
    print("  1. Configure PostgreSQL backend in monitoring config")
    print("  2. Use backend.topology_storage.save_topology_snapshot(topology)")
    print("  3. Use backend.topology_storage.save_token_consumption() for each component")
    print()
    print("Example code:")
    print("""
    # Save topology snapshot
    snapshot_id = await backend.topology_storage.save_topology_snapshot(topology)

    # Save token consumption for each function
    for usage in top_functions:
        await backend.topology_storage.save_token_consumption(
            snapshot_id=snapshot_id,
            component_type="function",
            component_name=usage.function_info.qualified_name,
            total_tokens=usage.token_consumption.total_tokens,
            total_calls=usage.token_consumption.total_calls,
            avg_tokens_per_call=usage.token_consumption.avg_tokens_per_call,
            total_cost_usd=usage.token_consumption.total_cost_usd
        )
    """)
    print()

    # ========== Summary ==========
    print("=" * 80)
    print("📋 Summary")
    print("=" * 80)
    print(f"""
✅ Successfully demonstrated topology-based token consumption analysis!

Key Insights:
🔍 Repository Analysis:
   - Scanned {topology.metrics.total_modules} modules with {topology.metrics.total_functions} functions
   - Total codebase: {topology.metrics.total_lines:,} lines

💰 Token Consumption:
   - Tracked {len(simulated_events)} LLM operations
   - Total tokens: {sum(e.total_tokens for e in simulated_events):,}
   - Total cost: ${sum(e.cost_usd for e in simulated_events):.4f}

🎯 Optimization Opportunities:
   - Found {len(candidates)} functions worth optimizing
   - Potential savings: {hotspots.potential_savings:.1f}%
   - High-impact candidates: {sum(1 for c in candidates if c.optimization_potential > 0.7)}

📊 Output Files:
   - Correlation data: {correlation_file}
   - Interactive visualization: {html_file}
   - Function reports: {output_dir}

💡 Next Steps:
   1. Review optimization candidates and refactor high-token functions
   2. Set up continuous monitoring with PostgreSQL backend
   3. Create alerts for functions exceeding token budgets
   4. Integrate with CI/CD to catch regressions
   5. Use topology diff to track changes over time

🚀 This analysis enables data-driven optimization of your LLM application!
    """)

    print("\n✨ Example complete!")
    print("=" * 80)


# ========== Advanced: Real-time Integration ==========


async def demonstrate_realtime_integration():
    """
    Demonstrate real-time integration with actual monitoring.

    In production, you would:
    1. Run topology scan once per deployment
    2. Initialize correlator with topology
    3. Track events in real-time via decorator integration
    4. Periodically analyze and alert on hotspots
    """
    print("\n")
    print("=" * 80)
    print("Advanced: Real-time Integration Pattern")
    print("=" * 80)
    print()

    print("Production Integration Pattern:")
    print("""
# 1. Initialize monitoring with topology support
config = MonitoringConfig(
    storage=StorageConfig(
        backend="postgres",
        connection_string="postgresql://..."
    )
)

monitor = LLMOpsMonitor(config)
await monitor.initialize()

# 2. Scan topology at startup
parser = CodeParser()
modules = parser.parse_repository("/path/to/repo")
builder = DependencyGraphBuilder()
topology = builder.build_repository_topology(modules)

# 3. Save to database
snapshot_id = await monitor.backend.topology_storage.save_topology_snapshot(topology)

# 4. Initialize correlator
correlator = TokenConsumptionCorrelator()
correlator.load_topology(topology)

# 5. Enhance decorator to track function context
@monitor_llm(monitor=monitor)
async def my_llm_function(prompt: str):
    # Decorator automatically tracks function name
    # and correlates with topology
    response = await llm_call(prompt)
    return response

# 6. Periodic analysis (e.g., every hour)
async def analyze_and_alert():
    while True:
        await asyncio.sleep(3600)  # 1 hour

        # Query recent events
        events = await get_recent_events()

        # Track in correlator
        for event in events:
            correlator.track_event(event)

        # Analyze
        hotspots = correlator.analyze_hotspots()

        # Alert if needed
        if hotspots.potential_savings > 20:
            send_alert(f"High optimization potential: {hotspots.potential_savings:.1f}%")

        # Update database
        for usage in hotspots.top_functions_by_tokens:
            await monitor.backend.topology_storage.save_token_consumption(
                snapshot_id=snapshot_id,
                component_type="function",
                component_name=usage.function_info.qualified_name,
                total_tokens=usage.token_consumption.total_tokens,
                total_calls=usage.token_consumption.total_calls,
                avg_tokens_per_call=usage.token_consumption.avg_tokens_per_call,
                total_cost_usd=usage.token_consumption.total_cost_usd,
                first_seen=usage.token_consumption.first_seen,
                last_seen=usage.token_consumption.last_seen
            )
    """)


async def main():
    """Run all examples."""
    # Main topology-token analysis
    await run_topology_token_analysis()

    # Show real-time integration pattern
    await demonstrate_realtime_integration()


if __name__ == "__main__":
    print("\n🚀 Starting Token Consumption by Code Component Example...")
    print()

    asyncio.run(main())
