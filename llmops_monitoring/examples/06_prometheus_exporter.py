"""
Example: Prometheus Metrics Exporter

This example demonstrates how to use the Prometheus exporter to expose
LLM monitoring metrics for scraping by Prometheus.

Features demonstrated:
- Enable Prometheus metrics endpoint
- Expose operation counts, latencies, costs
- Track errors and queue health
- Configure custom port and host

Requirements:
    pip install 'llamonitor-async[prometheus,parquet]'

Usage:
    python llmops_monitoring/examples/06_prometheus_exporter.py

    Then in another terminal, scrape metrics:
    curl http://localhost:8000/metrics
"""

import asyncio
import random
from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig
from llmops_monitoring.schema.config import PrometheusConfig
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class LLMResponse:
    """Simulated LLM response object."""
    def __init__(self, text: str):
        self.text = text


# Example 1: Basic function with cost tracking
@monitor_llm(
    operation_name="summarize_text",
    measure_text=True,
    collectors=["cost"],
    custom_attributes={
        "model": "gpt-4o-mini",
        "temperature": 0.7
    }
)
async def summarize_text(text: str) -> LLMResponse:
    """Simulate text summarization."""
    await asyncio.sleep(random.uniform(0.1, 0.3))
    summary = f"Summary of {len(text)} characters: " + text[:50]
    return LLMResponse(text=summary * 5)


# Example 2: Translation function
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
    translated = f"Translated: {text}" * 3
    return LLMResponse(text=translated)


# Example 3: Function that sometimes fails (for error tracking)
@monitor_llm(
    operation_name="classify_sentiment",
    measure_text=True,
    collectors=["cost"],
    custom_attributes={
        "model": "gemini-1.5-flash",
        "task": "classification"
    }
)
async def classify_sentiment(text: str) -> LLMResponse:
    """Simulate sentiment classification with occasional errors."""
    await asyncio.sleep(random.uniform(0.05, 0.15))

    # 10% chance of error
    if random.random() < 0.1:
        raise ValueError("Simulated classification error")

    return LLMResponse(text=f"Sentiment: positive (confidence: {random.random():.2f})")


async def main():
    """Main example demonstrating Prometheus exporter."""
    logger.info("=" * 80)
    logger.info("Prometheus Exporter Example")
    logger.info("=" * 80)
    logger.info("")

    # Configure monitoring with Prometheus enabled
    config = MonitorConfig.for_local_dev()
    config.extensions["prometheus"] = PrometheusConfig(
        enabled=True,
        port=8000,
        host="0.0.0.0"
    ).model_dump()

    # Initialize monitoring
    writer = await initialize_monitoring(config)

    logger.info("✓ Monitoring initialized with Prometheus exporter")
    logger.info("")
    logger.info("📊 Prometheus metrics available at:")
    logger.info("   http://localhost:8000/metrics")
    logger.info("")
    logger.info("💡 Try these commands in another terminal:")
    logger.info("   curl http://localhost:8000/metrics")
    logger.info("   curl http://localhost:8000/metrics | grep llm_")
    logger.info("   curl http://localhost:8000/metrics | grep llm_operations_total")
    logger.info("")
    logger.info("🔧 Or configure Prometheus to scrape this endpoint:")
    logger.info("   scrape_configs:")
    logger.info("     - job_name: 'llm-monitoring'")
    logger.info("       static_configs:")
    logger.info("         - targets: ['localhost:8000']")
    logger.info("")

    # Wait a bit for server to fully start
    await asyncio.sleep(2)

    logger.info("🚀 Starting LLM operations (generating metrics)...")
    logger.info("")

    # Generate diverse operations to populate metrics
    operations = [
        ("summarize_text", summarize_text, "This is a long document that needs summarization"),
        ("translate_text", translate_text, "Hello, how are you doing today?"),
        ("classify_sentiment", classify_sentiment, "I absolutely love this product!"),
    ]

    # Run 20 operations with random distribution
    for i in range(20):
        op_name, op_func, text = random.choice(operations)

        try:
            result = await op_func(text)
            logger.info(f"  ✓ Operation {i+1}/20: {op_name} completed")
        except Exception as e:
            logger.info(f"  ✗ Operation {i+1}/20: {op_name} failed ({type(e).__name__})")

        # Small delay between operations
        await asyncio.sleep(0.1)

    logger.info("")
    logger.info("✓ All operations completed")
    logger.info("")

    # Wait for metrics to be exported
    logger.info("⏳ Flushing events and updating metrics...")
    await asyncio.sleep(6)

    # Check health including Prometheus status
    health = await writer.health_check()
    logger.info("")
    logger.info("📊 System Health:")
    logger.info(f"   Writer running: {health['running']}")
    logger.info(f"   Queue size: {health['queue_size']}")
    logger.info(f"   Buffer size: {health['buffer_size']}")
    if "prometheus" in health:
        prom_health = health["prometheus"]
        logger.info(f"   Prometheus healthy: {prom_health['healthy']}")
        logger.info(f"   Metrics exported: {prom_health['metrics_exported']}")
        logger.info(f"   Endpoint: {prom_health['endpoint']}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("📈 Prometheus Metrics Available")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Counters (always increasing):")
    logger.info("  • llm_operations_total{operation_name, model, operation_type}")
    logger.info("  • llm_errors_total{operation_name, error_type}")
    logger.info("  • llm_text_characters_total{operation_name}")
    logger.info("")
    logger.info("Histograms (distributions with buckets):")
    logger.info("  • llm_operation_duration_seconds{operation_name, model}")
    logger.info("  • llm_cost_usd{operation_name, model}")
    logger.info("")
    logger.info("Gauges (current values):")
    logger.info("  • llm_queue_size")
    logger.info("  • llm_buffer_size")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    # Keep server running for a bit so user can scrape
    logger.info("⏰ Keeping server running for 30 seconds...")
    logger.info("   Check http://localhost:8000/metrics in your browser")
    logger.info("   Press Ctrl+C to stop early")
    logger.info("")

    try:
        await asyncio.sleep(30)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    # Cleanup
    await writer.stop()
    logger.info("")
    logger.info("✅ Example completed!")
    logger.info("")
    logger.info("💡 Next Steps:")
    logger.info("   1. Configure Prometheus to scrape http://localhost:8000/metrics")
    logger.info("   2. Set up Grafana dashboards to visualize metrics")
    logger.info("   3. Create alerts based on error rates or latencies")
    logger.info("   4. Monitor costs and resource usage")
    logger.info("")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error running example: {e}", exc_info=True)
