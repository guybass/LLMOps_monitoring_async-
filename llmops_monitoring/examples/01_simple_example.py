"""
Simple Example: Basic Usage

This example demonstrates the most basic usage of the monitoring system.
"""

import asyncio
from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig
from llmops_monitoring.schema.config import StorageConfig


# Simulated LLM response
class LLMResponse:
    def __init__(self, text: str):
        self.text = text


# Monitored function
@monitor_llm(
    operation_name="generate_text",
    measure_text=True,  # Collect all text metrics
    custom_attributes={"model": "gpt-4"}
)
async def generate_text(prompt: str) -> LLMResponse:
    """Simulate an LLM call."""
    await asyncio.sleep(0.1)  # Simulate API latency
    response_text = f"Generated response for: {prompt}. " * 10
    return LLMResponse(text=response_text)


async def main():
    # Initialize monitoring with local Parquet storage
    config = MonitorConfig(
        storage=StorageConfig(
            backend="parquet",
            output_dir="./simple_monitoring_data",
            batch_size=10,
            flush_interval_seconds=2.0
        )
    )

    writer = await initialize_monitoring(config)

    print("Running simple example...")

    # Make some LLM calls
    for i in range(5):
        prompt = f"Tell me about topic {i}"
        response = await generate_text(prompt)
        print(f"Call {i+1}: Generated {len(response.text)} characters")

    # Give time for events to flush
    await asyncio.sleep(3)

    # Stop monitoring
    await writer.stop()

    print("\nDone! Check ./simple_monitoring_data for Parquet files.")
    print("Events contain char_count, word_count, byte_size, and line_count metrics.")


if __name__ == "__main__":
    asyncio.run(main())
