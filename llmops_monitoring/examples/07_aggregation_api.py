"""
Example: Aggregation REST API Server

This example demonstrates how to run the REST API server for querying
and aggregating monitoring data.

Features demonstrated:
- Start REST API server
- Query endpoints for events, sessions, traces
- Aggregate metrics by operation and model
- Cost analytics
- Summary statistics

Requirements:
    pip install 'llamon

itor-async[api,parquet]'

Usage:
    # Start the server
    python llmops_monitoring/examples/07_aggregation_api.py

    # In another terminal, query the API
    curl http://localhost:8080/api/v1/metrics/summary
    curl http://localhost:8080/api/v1/sessions
    curl http://localhost:8080/api/v1/metrics/operations

    # Or visit the interactive docs
    http://localhost:8080/docs
"""

import asyncio
import random
from datetime import datetime
from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig
from llmops_monitoring.api import run_api_server
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class LLMResponse:
    """Simulated LLM response object."""
    def __init__(self, text: str):
        self.text = text


# Example monitored functions to generate sample data
@monitor_llm(
    operation_name="summarize_document",
    measure_text=True,
    collectors=["cost"],
    custom_attributes={
        "model": "gpt-4o-mini",
        "temperature": 0.7
    }
)
async def summarize_document(text: str) -> LLMResponse:
    """Simulate document summarization."""
    await asyncio.sleep(random.uniform(0.1, 0.3))
    return LLMResponse(text=f"Summary: {text[:50]}...")


@monitor_llm(
    operation_name="translate_text",
    measure_text=True,
    collectors=["cost"],
    custom_attributes={
        "model": "claude-3-5-sonnet",
        "source_lang": "en",
        "target_lang": "es"
    }
)
async def translate_text(text: str) -> LLMResponse:
    """Simulate text translation."""
    await asyncio.sleep(random.uniform(0.2, 0.5))
    return LLMResponse(text=f"Translated: {text}")


@monitor_llm(
    operation_name="generate_code",
    measure_text=True,
    collectors=["cost"],
    custom_attributes={
        "model": "gpt-4",
        "language": "python"
    }
)
async def generate_code(prompt: str) -> LLMResponse:
    """Simulate code generation."""
    await asyncio.sleep(random.uniform(0.3, 0.6))
    return LLMResponse(text=f"def example():\n    # Generated code\n    pass")


async def generate_sample_data():
    """Generate sample monitoring data for the API."""
    logger.info("=" * 80)
    logger.info("Generating Sample Data for API")
    logger.info("=" * 80)
    logger.info("")

    # Initialize monitoring with Parquet backend
    config = MonitorConfig.for_local_dev()
    writer = await initialize_monitoring(config)

    logger.info("✓ Monitoring initialized")
    logger.info("")
    logger.info("🔄 Generating sample events...")
    logger.info("")

    # Generate diverse operations
    operations = [
        (summarize_document, "This is a long document about machine learning..."),
        (translate_text, "Hello, how are you today?"),
        (generate_code, "Write a function to calculate fibonacci numbers"),
    ]

    # Run 30 operations
    for i in range(30):
        op_func, text = random.choice(operations)

        try:
            result = await op_func(text)
            logger.info(f"  ✓ Operation {i+1}/30: {op_func.__name__} completed")
        except Exception as e:
            logger.info(f"  ✗ Operation {i+1}/30: {op_func.__name__} failed ({type(e).__name__})")

        await asyncio.sleep(0.1)

    logger.info("")
    logger.info("✓ Sample data generated")
    logger.info("")

    # Flush and close
    logger.info("⏳ Flushing events...")
    await asyncio.sleep(3)
    await writer.stop()

    logger.info("✓ Events flushed to storage")
    logger.info("")


def main():
    """Main function to run the API server."""
    logger.info("=" * 80)
    logger.info("Aggregation REST API Server Example")
    logger.info("=" * 80)
    logger.info("")

    # Ask user if they want to generate sample data
    logger.info("This example demonstrates the REST API for querying monitoring data.")
    logger.info("")
    logger.info("Would you like to generate sample data first? (y/n)")
    response = input().strip().lower()

    if response == 'y':
        logger.info("")
        asyncio.run(generate_sample_data())

    logger.info("=" * 80)
    logger.info("Starting REST API Server")
    logger.info("=" * 80)
    logger.info("")

    # Configure the API server
    config = MonitorConfig.for_local_dev()

    logger.info("📊 API Server Configuration:")
    logger.info(f"   Backend: {config.storage.backend}")
    logger.info(f"   Data location: {config.storage.output_dir}")
    logger.info("")

    logger.info("🚀 Starting API server on http://0.0.0.0:8080")
    logger.info("")
    logger.info("📚 API Documentation:")
    logger.info("   Interactive docs: http://localhost:8080/docs")
    logger.info("   OpenAPI spec: http://localhost:8080/openapi.json")
    logger.info("")
    logger.info("🔧 Example API Requests:")
    logger.info("")
    logger.info("   # Health check")
    logger.info("   curl http://localhost:8080/api/health")
    logger.info("")
    logger.info("   # Get summary statistics")
    logger.info("   curl http://localhost:8080/api/v1/metrics/summary")
    logger.info("")
    logger.info("   # List sessions")
    logger.info("   curl http://localhost:8080/api/v1/sessions")
    logger.info("")
    logger.info("   # Get metrics by operation")
    logger.info("   curl http://localhost:8080/api/v1/metrics/operations")
    logger.info("")
    logger.info("   # Get metrics by model")
    logger.info("   curl http://localhost:8080/api/v1/metrics/models")
    logger.info("")
    logger.info("   # Get cost analytics")
    logger.info("   curl 'http://localhost:8080/api/v1/metrics/costs?group_by=model'")
    logger.info("")
    logger.info("   # Query events")
    logger.info("   curl 'http://localhost:8080/api/v1/events?limit=10'")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("")

    # Run the API server
    try:
        run_api_server(config, host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        logger.info("")
        logger.info("✅ Server stopped")
        logger.info("")
    except Exception as e:
        logger.error(f"Error running API server: {e}", exc_info=True)
        logger.info("")
        logger.info("💡 Make sure you have the API dependencies installed:")
        logger.info("   pip install 'llamonitor-async[api,parquet]'")
        logger.info("")


if __name__ == "__main__":
    main()
