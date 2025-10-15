"""
Example: Using MySQL Backend for Production Monitoring

This example demonstrates how to use the MySQL backend for storing
monitoring events in production environments.

Requirements:
    pip install 'llamonitor-async[mysql]'

    MySQL server running and accessible
    Create database: CREATE DATABASE monitoring;

Usage:
    python llmops_monitoring/examples/04_mysql_backend.py
"""

import asyncio
import os
from dotenv import load_dotenv

from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig
from llmops_monitoring.schema.config import StorageConfig
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)

# Load environment variables
load_dotenv()


@monitor_llm(
    operation_name="mysql_test_function",
    measure_text=True,
    custom_attributes={"backend": "mysql", "environment": "production"}
)
async def test_mysql_function(prompt: str) -> dict:
    """Simulate LLM function for MySQL backend testing."""
    await asyncio.sleep(0.1)  # Simulate processing
    response_text = f"MySQL backend response for: {prompt}"
    return {"text": response_text}


async def main():
    """Main example demonstrating MySQL backend usage."""
    logger.info("Running MySQL backend example...")

    # Get MySQL connection string from environment or use default
    mysql_conn = os.getenv(
        "MYSQL_CONNECTION_STRING",
        "mysql://root:password@localhost:3306/monitoring"
    )

    # Configure monitoring with MySQL backend
    config = MonitorConfig(
        storage=StorageConfig(
            backend="mysql",
            connection_string=mysql_conn,
            table_name="metric_events",
            pool_size=10,
            batch_size=50,
            flush_interval_seconds=5.0
        ),
        max_queue_size=10000,
        fail_silently=False
    )

    # Initialize monitoring
    try:
        await initialize_monitoring(config)
        logger.info("✓ MySQL backend initialized successfully")
    except RuntimeError as e:
        if "requires aiomysql" in str(e):
            logger.error("MySQL backend requires aiomysql. Install with: pip install 'llamonitor-async[mysql]'")
            return
        raise

    # Make monitored function calls
    logger.info("Making monitored function calls...")

    test_prompts = [
        "What is machine learning?",
        "Explain quantum computing",
        "How do neural networks work?",
        "What is reinforcement learning?",
        "Describe deep learning architectures"
    ]

    for prompt in test_prompts:
        result = await test_mysql_function(prompt)
        logger.debug(f"Result: {result['text'][:50]}...")

    logger.info("✓ All function calls completed")

    # Wait for all events to be written
    logger.info("Flushing events to MySQL...")
    await asyncio.sleep(6)  # Wait for flush interval

    logger.info("✓ MySQL backend example completed successfully!")
    logger.info("")
    logger.info("To query the data in MySQL:")
    logger.info("  mysql -u root -p monitoring")
    logger.info("  SELECT * FROM metric_events ORDER BY timestamp DESC LIMIT 10;")
    logger.info("")
    logger.info("To view text metrics:")
    logger.info("  SELECT operation_name, text_char_count, text_word_count")
    logger.info("  FROM metric_events WHERE text_char_count IS NOT NULL;")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(f"Error running example: {e}", exc_info=True)
