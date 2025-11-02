"""
Example: Datadog Integration for Monitoring and Alerting

This example demonstrates how to export LLM monitoring metrics to Datadog
for visualization, alerting, and correlation with other infrastructure metrics.

Features:
- Operation counts and error rates
- Latency distributions (P50, P95, P99)
- Character counts for token usage estimation
- Cost tracking per operation/model
- Custom tags for filtering and grouping
- Support for both HTTP API and DogStatsD

Requirements:
    pip install 'llamonitor-async[datadog]'

    Datadog account with:
    - API key
    - App key (for HTTP API) or DogStatsD agent

Usage:
    # Set environment variables
    export DATADOG_API_KEY="your-api-key"
    export DATADOG_APP_KEY="your-app-key"

    python llmops_monitoring/examples/12_datadog_integration.py
"""

import asyncio
import os
import random
from datetime import datetime

from llamonitor import monitor_llm, initialize_monitoring, MonitorConfig
from llmops_monitoring.schema.config import StorageConfig, DatadogConfig
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


@monitor_llm(
    operation_name="datadog_llm_call",
    measure_text=True,
    collectors=["cost"],
    custom_attributes={"model": "gpt-4o-mini", "environment": "production"}
)
async def llm_call_with_datadog(prompt: str) -> dict:
    """Simulates LLM operation with Datadog metrics."""
    # Simulate variable latency
    await asyncio.sleep(random.uniform(0.05, 0.15))

    response = f"Response to: {prompt}"
    return {"text": response}


@monitor_llm(
    operation_name="datadog_data_processing",
    measure_text=True,
    custom_attributes={"model": "claude-3-sonnet", "environment": "production"}
)
async def data_processing_with_metrics(data: str) -> dict:
    """Simulates data processing with metrics."""
    await asyncio.sleep(random.uniform(0.03, 0.08))

    processed = f"Processed: {data}"
    return {"text": processed}


async def generate_sample_traffic():
    """Generate sample LLM traffic for Datadog metrics."""
    logger.info("Generating sample traffic...")

    prompts = [
        "What is machine learning?",
        "Explain quantum computing",
        "How do neural networks work?",
        "What is reinforcement learning?",
        "Describe transformer architecture"
    ]

    # Generate diverse traffic
    for i in range(50):
        prompt = prompts[i % len(prompts)]

        # LLM calls
        await llm_call_with_datadog(prompt)

        # Data processing
        if i % 3 == 0:
            await data_processing_with_metrics(f"dataset_{i}")

    logger.info("✓ Generated 50+ operations")


async def main():
    """Main example demonstrating Datadog integration."""
    logger.info("="*70)
    logger.info("Datadog Integration Example")
    logger.info("="*70)

    # Get Datadog credentials from environment
    api_key = os.getenv("DATADOG_API_KEY", "")
    app_key = os.getenv("DATADOG_APP_KEY", "")
    use_statsd = os.getenv("DATADOG_USE_STATSD", "false").lower() == "true"

    if not api_key and not use_statsd:
        logger.error("\n⚠️  ERROR: Datadog API key not set!")
        logger.error("Please set environment variables:")
        logger.error("  export DATADOG_API_KEY='your-api-key'")
        logger.error("  export DATADOG_APP_KEY='your-app-key'")
        logger.error("\nOr use DogStatsD:")
        logger.error("  export DATADOG_USE_STATSD=true")
        logger.error("\nRunning in demo mode (metrics won't be sent)...")
        # Continue with demo mode
        api_key = "demo-key"
        app_key = "demo-app-key"

    # Configure monitoring with Datadog exporter
    config = MonitorConfig(
        storage=StorageConfig(
            backend="parquet",
            output_dir="./datadog_demo_data",
            batch_size=50,
            flush_interval_seconds=5.0
        )
    )

    # Add Datadog configuration
    config.extensions["datadog"] = DatadogConfig(
        enabled=True,
        api_key=api_key,
        app_key=app_key,
        site="datadoghq.com",  # or datadoghq.eu, us3.datadoghq.com, etc.
        use_statsd=use_statsd,
        statsd_host="localhost",
        statsd_port=8125,
        namespace="llmops",  # Metric prefix
        submission_interval=10.0
    ).model_dump()

    # Initialize monitoring
    await initialize_monitoring(config)
    logger.info("✓ Monitoring initialized with Datadog exporter")

    if use_statsd:
        logger.info(f"Using DogStatsD (localhost:8125)")
    else:
        logger.info(f"Using Datadog HTTP API (site: datadoghq.com)")

    # Generate sample traffic
    logger.info("\n" + "="*70)
    logger.info("Generating Sample LLM Traffic")
    logger.info("="*70)
    await generate_sample_traffic()

    # Wait for flush
    logger.info("\nWaiting for metrics to be flushed to Datadog...")
    await asyncio.sleep(6)

    logger.info("\n" + "="*70)
    logger.info("✓ Datadog Integration Complete!")
    logger.info("="*70)

    logger.info("\nMetrics Sent to Datadog:")
    logger.info("  • llmops.operations.count - Operation counts by operation_name, model")
    logger.info("  • llmops.errors.count - Error counts")
    logger.info("  • llmops.operation.duration.ms - Latency distribution")
    logger.info("  • llmops.characters.total - Character counts")
    logger.info("  • llmops.cost.usd - Cost per operation")

    logger.info("\nView Metrics in Datadog:")
    logger.info("  1. Go to https://app.datadoghq.com/metric/explorer")
    logger.info("  2. Search for 'llmops.*' metrics")
    logger.info("  3. Filter by tags:")
    logger.info("     - operation_name:datadog_llm_call")
    logger.info("     - model:gpt-4o-mini")
    logger.info("     - environment:production")

    logger.info("\nCreate Dashboards:")
    logger.info("  • Operation counts over time")
    logger.info("  • Error rate trends")
    logger.info("  • P95/P99 latency by operation")
    logger.info("  • Cost breakdown by model")
    logger.info("  • Compare performance across models")

    logger.info("\nSet Up Monitors:")
    logger.info("  • Alert on high error rates")
    logger.info("  • Alert on latency spikes (P95 > threshold)")
    logger.info("  • Alert on cost anomalies")
    logger.info("  • Composite monitors for SLOs")

    logger.info("\nDatadog Query Examples:")
    logger.info("  # Average latency by operation")
    logger.info("  avg:llmops.operation.duration.ms{*} by {operation_name}")
    logger.info("")
    logger.info("  # Error rate")
    logger.info("  sum:llmops.errors.count{*}.as_rate()")
    logger.info("")
    logger.info("  # Total cost by model")
    logger.info("  sum:llmops.cost.usd{*} by {model}")
    logger.info("")
    logger.info("  # P95 latency")
    logger.info("  p95:llmops.operation.duration.ms{*} by {operation_name}")

    logger.info("\n" + "="*70)
    logger.info("Key Benefits:")
    logger.info("  ✓ Centralized monitoring with infrastructure metrics")
    logger.info("  ✓ Powerful alerting and anomaly detection")
    logger.info("  ✓ Correlation with logs, traces, and APM")
    logger.info("  ✓ Pre-built dashboards and widgets")
    logger.info("  ✓ Advanced analytics and forecasting")
    logger.info("  ✓ SLO tracking and reporting")
    logger.info("="*70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error running example: {e}", exc_info=True)
