"""
Example: GraphQL API for Flexible Monitoring Queries

This example demonstrates how to use the GraphQL API for querying
monitoring data with flexible, nested queries and real-time subscriptions.

GraphQL provides:
- Flexible query structure (request exactly what you need)
- Strongly typed schema
- Nested queries in single request
- Real-time subscriptions
- GraphQL Playground for interactive exploration

Requirements:
    pip install 'llamonitor-async[graphql,api]'

Usage:
    python llmops_monitoring/examples/10_graphql_queries.py

    Then visit:
    - http://localhost:8080/graphql - GraphQL Playground
    - Run queries interactively in the browser
"""

import asyncio
from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig
from llmops_monitoring.schema.config import StorageConfig
from llmops_monitoring.api import create_api_server
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


@monitor_llm(
    operation_name="graphql_example_operation",
    measure_text=True,
    collectors=["cost"],
    custom_attributes={"model": "gpt-4o-mini", "environment": "demo"}
)
async def example_operation(prompt: str) -> dict:
    """Example monitored operation."""
    await asyncio.sleep(0.1)
    return {"text": f"Response to: {prompt}"}


async def generate_sample_data():
    """Generate sample events for GraphQL queries."""
    logger.info("Generating sample monitoring data...")

    prompts = [
        "What is AI?",
        "Explain machine learning",
        "How does deep learning work?",
        "What is natural language processing?",
        "Tell me about neural networks"
    ]

    for prompt in prompts:
        await example_operation(prompt)

    logger.info("✓ Generated sample data")


def print_graphql_examples():
    """Print example GraphQL queries."""
    logger.info("\n" + "="*70)
    logger.info("GraphQL API is running!")
    logger.info("="*70)
    logger.info("")
    logger.info("GraphQL Playground: http://localhost:8080/graphql")
    logger.info("")
    logger.info("Example Queries:")
    logger.info("-" * 70)

    logger.info("\n1. Get Summary Statistics:")
    logger.info("""
query {
  summary {
    totalEvents
    totalSessions
    totalOperations
    totalErrors
    errorRate
    avgDurationMs
    totalCost usd
  }
}
""")

    logger.info("\n2. Query Events with Filters:")
    logger.info("""
query {
  events(filter: { limit: 5, errorOnly: false }) {
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
""")

    logger.info("\n3. Get Sessions with Nested Traces:")
    logger.info("""
query {
  sessions(limit: 5) {
    sessionId
    startTime
    eventCount
    errorCount
  }
}
""")

    logger.info("\n4. Aggregate by Operation with Percentiles:")
    logger.info("""
query {
  operations {
    operationName
    count
    errorCount
    avgDurationMs
    p50DurationMs
    p95DurationMs
    p99DurationMs
    totalCharacters
    estimatedCostUsd
  }
}
""")

    logger.info("\n5. Aggregate by Model:")
    logger.info("""
query {
  models {
    model
    count
    errorCount
    avgDurationMs
    totalCharacters
    estimatedCostUsd
  }
}
""")

    logger.info("\n6. Cost Analytics:")
    logger.info("""
query {
  costs(filter: { groupBy: "model" }) {
    groupKey
    totalCostUsd
    operationCount
    avgCostPerOperationUsd
  }
}
""")

    logger.info("\n7. Complex Nested Query (combine multiple queries):")
    logger.info("""
query {
  summary {
    totalEvents
    totalCost usd
  }
  operations {
    operationName
    count
    avgDurationMs
  }
  models {
    model
    estimatedCostUsd
  }
}
""")

    logger.info("\n8. Real-time Subscription (WebSocket):")
    logger.info("""
subscription {
  eventStream {
    eventId
    operationName
    timestamp
    durationMs
  }
}
""")

    logger.info("\n9. Filtered Subscription:")
    logger.info("""
subscription {
  eventStream(operationName: "graphql_example_operation", errorOnly: false) {
    eventId
    operationName
    timestamp
    textMetrics {
      charCount
    }
  }
}
""")

    logger.info("\n" + "="*70)
    logger.info("Benefits of GraphQL:")
    logger.info("-" * 70)
    logger.info("  ✓ Request exactly the fields you need")
    logger.info("  ✓ Combine multiple queries in single request")
    logger.info("  ✓ Strongly typed schema with auto-completion")
    logger.info("  ✓ Real-time subscriptions for live updates")
    logger.info("  ✓ Interactive playground for exploration")
    logger.info("  ✓ Self-documenting API")
    logger.info("="*70)


def run_with_graphql():
    """Run API server with GraphQL support."""
    import uvicorn

    # Configure monitoring
    config = MonitorConfig(
        storage=StorageConfig(
            backend="parquet",
            output_dir="./graphql_demo_data",
            batch_size=50,
            flush_interval_seconds=2.0
        )
    )

    # Create FastAPI app with GraphQL
    app = create_api_server(config)

    # Generate sample data in background
    async def startup_task():
        await asyncio.sleep(1)  # Wait for server to start
        await initialize_monitoring(config)
        await generate_sample_data()
        await asyncio.sleep(3)  # Wait for flush
        print_graphql_examples()

    # Start background task
    asyncio.create_task(startup_task())

    # Run server
    logger.info("Starting API server with GraphQL support...")
    logger.info("Server will be available at http://localhost:8080")
    logger.info("")

    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    try:
        run_with_graphql()
    except KeyboardInterrupt:
        logger.warning("\nServer stopped by user")
    except Exception as e:
        logger.error(f"Error running server: {e}", exc_info=True)
