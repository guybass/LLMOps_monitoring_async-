"""
Example: Using ClickHouse Backend for Analytics Workloads

This example demonstrates how to use the ClickHouse backend for storing
monitoring events optimized for fast analytical queries and aggregations.

ClickHouse is ideal for:
- High-volume event ingestion
- Fast time-series analytics
- Percentile calculations
- Complex aggregations across large datasets

Requirements:
    pip install 'llamonitor-async[clickhouse]'

    ClickHouse server running and accessible
    Create database: CREATE DATABASE monitoring;

Usage:
    python llmops_monitoring/examples/09_clickhouse_backend.py
"""

import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig
from llmops_monitoring.schema.config import StorageConfig
from llmops_monitoring.aggregation import create_aggregation_service
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)

# Load environment variables
load_dotenv()


@monitor_llm(
    operation_name="clickhouse_llm_call",
    measure_text=True,
    collectors=["cost"],
    custom_attributes={
        "model": "gpt-4o-mini",
        "environment": "analytics"
    }
)
async def llm_call_with_analytics(prompt: str, model: str = "gpt-4o-mini") -> dict:
    """Simulate LLM function for ClickHouse analytics testing."""
    await asyncio.sleep(0.1)  # Simulate processing
    response_text = f"ClickHouse analytics response for: {prompt}"
    return {"text": response_text}


@monitor_llm(
    operation_name="clickhouse_data_processing",
    measure_text=True,
    custom_attributes={
        "backend": "clickhouse",
        "data_type": "time_series"
    }
)
async def data_processing_operation(data: str) -> dict:
    """Simulate data processing operation."""
    await asyncio.sleep(0.05)
    return {"text": f"Processed: {data}"}


async def generate_sample_events():
    """Generate sample events for analytics demonstration."""
    logger.info("Generating sample events for analytics...")

    models = ["gpt-4o-mini", "gpt-4", "claude-3-sonnet", "gemini-1.5-flash"]
    prompts = [
        "Analyze customer sentiment",
        "Generate product description",
        "Summarize document",
        "Extract key insights",
        "Translate text",
        "Answer question",
        "Code review",
        "Data analysis"
    ]

    # Generate diverse events for analytics
    for i in range(50):
        model = models[i % len(models)]
        prompt = prompts[i % len(prompts)]

        # Update custom attributes with model
        result = await llm_call_with_analytics(prompt, model=model)

        # Also generate some data processing events
        if i % 3 == 0:
            await data_processing_operation(f"dataset_{i}")

    logger.info("✓ Generated 50+ sample events")


async def run_analytics_queries(config: MonitorConfig):
    """Demonstrate ClickHouse's analytical query capabilities."""
    logger.info("\n" + "="*60)
    logger.info("Running analytical queries on ClickHouse...")
    logger.info("="*60)

    # Create aggregation service
    service = await create_aggregation_service(config)

    try:
        # Wait for flush
        await asyncio.sleep(6)

        # 1. Summary Statistics
        logger.info("\n1. Overall Summary Statistics:")
        logger.info("-" * 40)
        stats = await service.get_summary_stats()

        logger.info(f"  Total Events: {stats.get('total_events', 0)}")
        logger.info(f"  Total Sessions: {stats.get('total_sessions', 0)}")
        logger.info(f"  Total Operations: {stats.get('total_operations', 0)}")
        logger.info(f"  Total Errors: {stats.get('total_errors', 0)}")
        logger.info(f"  Error Rate: {stats.get('error_rate', 0):.2%}")
        logger.info(f"  Avg Duration: {stats.get('avg_duration_ms', 0):.2f}ms")
        logger.info(f"  Total Characters: {stats.get('total_characters', 0):,}")
        logger.info(f"  Total Cost: ${stats.get('total_cost_usd', 0):.6f}")

        # 2. Aggregation by Operation (with percentiles)
        logger.info("\n2. Metrics by Operation (with percentiles):")
        logger.info("-" * 40)
        operations = await service.aggregate_by_operation()

        for op in operations[:5]:  # Top 5
            logger.info(f"  Operation: {op['operation_name']}")
            logger.info(f"    Count: {op['count']}")
            logger.info(f"    Avg Duration: {op.get('avg_duration_ms', 0):.2f}ms")
            logger.info(f"    P50: {op.get('p50_duration_ms', 0):.2f}ms")
            logger.info(f"    P95: {op.get('p95_duration_ms', 0):.2f}ms")
            logger.info(f"    P99: {op.get('p99_duration_ms', 0):.2f}ms")
            logger.info(f"    Errors: {op.get('error_count', 0)}")
            logger.info(f"    Total Chars: {op.get('total_characters', 0):,}")
            logger.info("")

        # 3. Aggregation by Model
        logger.info("3. Metrics by Model:")
        logger.info("-" * 40)
        models = await service.aggregate_by_model()

        for model_data in models:
            logger.info(f"  Model: {model_data['model']}")
            logger.info(f"    Count: {model_data['count']}")
            logger.info(f"    Avg Duration: {model_data.get('avg_duration_ms', 0):.2f}ms")
            logger.info(f"    Total Chars: {model_data.get('total_characters', 0):,}")
            logger.info(f"    Estimated Cost: ${model_data.get('estimated_cost_usd', 0):.6f}")
            logger.info("")

        # 4. Cost Analytics (grouped by model)
        logger.info("4. Cost Analytics by Model:")
        logger.info("-" * 40)
        costs = await service.aggregate_costs(group_by="model")

        for cost_data in costs:
            logger.info(f"  Model: {cost_data['model']}")
            logger.info(f"    Total Cost: ${cost_data.get('total_cost_usd', 0):.6f}")
            logger.info(f"    Operation Count: {cost_data.get('operation_count', 0)}")
            logger.info(f"    Avg Cost/Op: ${cost_data.get('avg_cost_per_operation_usd', 0):.6f}")
            logger.info("")

        # 5. Recent Sessions
        logger.info("5. Recent Sessions:")
        logger.info("-" * 40)
        sessions = await service.get_sessions(limit=5)

        for session in sessions:
            logger.info(f"  Session: {session['session_id']}")
            logger.info(f"    Start: {session.get('start_time')}")
            logger.info(f"    Events: {session.get('event_count', 0)}")
            logger.info(f"    Operations: {session.get('operation_count', 0)}")
            logger.info(f"    Errors: {session.get('error_count', 0)}")
            logger.info(f"    Total Duration: {session.get('total_duration_ms', 0):.2f}ms")
            logger.info("")

        logger.info("="*60)
        logger.info("✓ Analytics queries completed successfully!")
        logger.info("="*60)

    finally:
        await service.close()


async def main():
    """Main example demonstrating ClickHouse backend usage."""
    logger.info("Running ClickHouse backend example...")
    logger.info("This demonstrates ClickHouse's analytical capabilities")
    logger.info("")

    # Get ClickHouse connection string from environment or use default
    clickhouse_conn = os.getenv(
        "CLICKHOUSE_CONNECTION_STRING",
        "clickhouse://default:@localhost:9000/monitoring"
    )

    # Configure monitoring with ClickHouse backend
    config = MonitorConfig(
        storage=StorageConfig(
            backend="clickhouse",
            connection_string=clickhouse_conn,
            table_name="metric_events",
            schema_name="monitoring",  # database name
            batch_size=100,
            flush_interval_seconds=5.0
        ),
        max_queue_size=10000,
        fail_silently=False
    )

    # Initialize monitoring
    try:
        await initialize_monitoring(config)
        logger.info("✓ ClickHouse backend initialized successfully")
    except RuntimeError as e:
        if "requires clickhouse-driver" in str(e):
            logger.error(
                "ClickHouse backend requires clickhouse-driver. "
                "Install with: pip install 'llamonitor-async[clickhouse]'"
            )
            return
        raise

    # Generate sample events
    await generate_sample_events()

    # Run analytics queries
    await run_analytics_queries(config)

    logger.info("")
    logger.info("To query the data directly in ClickHouse:")
    logger.info("  clickhouse-client --database monitoring")
    logger.info("  SELECT * FROM metric_events ORDER BY timestamp DESC LIMIT 10;")
    logger.info("")
    logger.info("To calculate percentiles in ClickHouse:")
    logger.info("  SELECT")
    logger.info("    operation_name,")
    logger.info("    quantile(0.5)(duration_ms) as p50,")
    logger.info("    quantile(0.95)(duration_ms) as p95,")
    logger.info("    quantile(0.99)(duration_ms) as p99")
    logger.info("  FROM metric_events")
    logger.info("  GROUP BY operation_name;")
    logger.info("")
    logger.info("To view cost by model:")
    logger.info("  SELECT")
    logger.info("    JSONExtractString(custom_attributes, 'model') as model,")
    logger.info("    sum(JSONExtractFloat(custom_attributes, 'estimated_cost_usd')) as total_cost")
    logger.info("  FROM metric_events")
    logger.info("  GROUP BY model")
    logger.info("  ORDER BY total_cost DESC;")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(f"Error running example: {e}", exc_info=True)
