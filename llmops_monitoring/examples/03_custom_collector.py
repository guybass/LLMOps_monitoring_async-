"""
Custom Collector Example: Extensibility

This example demonstrates how to create and register custom metric collectors.
"""

import asyncio
from typing import Any, Dict, Optional

from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig
from llmops_monitoring.instrumentation.base import MetricCollector, CollectorRegistry
from llmops_monitoring.schema.config import StorageConfig


class CostCollector(MetricCollector):
    """
    Custom collector that calculates cost based on text length.

    This is a simplified example - in production, you'd use actual token counts
    and model-specific pricing.
    """

    def __init__(self, price_per_1k_chars: float = 0.002):
        self.price_per_1k_chars = price_per_1k_chars

    def collect(
        self,
        result: Any,
        args: tuple,
        kwargs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate cost based on character count."""
        text = self._extract_text(result)
        if not text:
            return {}

        char_count = len(text)
        cost = (char_count / 1000) * self.price_per_1k_chars

        # Return in custom_attributes (since we don't have a CostMetrics schema yet)
        return {
            "custom_attributes": {
                "estimated_cost_usd": round(cost, 6),
                "cost_basis": "char_count"
            }
        }

    @property
    def metric_type(self) -> str:
        return "cost"

    def _extract_text(self, result: Any) -> Optional[str]:
        """Extract text from result."""
        if isinstance(result, str):
            return result
        if hasattr(result, 'text'):
            return result.text
        return None


# Register custom collector
CollectorRegistry.register("cost", CostCollector)


class LLMResponse:
    def __init__(self, text: str):
        self.text = text


@monitor_llm(
    operation_name="generate_with_cost",
    measure_text=["char_count", "word_count"],
    collectors=["cost"],  # Use our custom collector
    custom_attributes={"model": "gpt-4"}
)
async def generate_with_cost_tracking(prompt: str) -> LLMResponse:
    """LLM call with cost tracking."""
    await asyncio.sleep(0.1)
    response_text = f"Response to '{prompt}': " + "detailed answer " * 30
    return LLMResponse(text=response_text)


async def main():
    # Initialize monitoring
    config = MonitorConfig(
        storage=StorageConfig(
            backend="parquet",
            output_dir="./custom_collector_data",
            batch_size=5
        )
    )

    writer = await initialize_monitoring(config)

    print("Running custom collector example...")
    print("Tracking both text metrics AND estimated cost.\n")

    # Make some calls
    prompts = [
        "Short prompt",
        "Medium length prompt with more details",
        "Very long prompt with lots of context and information that will generate a longer response"
    ]

    for i, prompt in enumerate(prompts):
        response = await generate_with_cost_tracking(prompt)
        char_count = len(response.text)
        estimated_cost = (char_count / 1000) * 0.002
        print(f"Call {i+1}: {char_count} chars, ~${estimated_cost:.6f}")

    # Flush
    await asyncio.sleep(3)
    await writer.stop()

    print("\nDone! Check ./custom_collector_data")
    print("Events include 'custom_attributes' with 'estimated_cost_usd'")
    print("\nTo add your own collector:")
    print("1. Extend MetricCollector")
    print("2. Implement collect() method")
    print("3. Register with CollectorRegistry")
    print("4. Use in @monitor_llm(collectors=['your_collector'])")


if __name__ == "__main__":
    asyncio.run(main())
