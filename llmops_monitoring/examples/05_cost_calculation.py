"""
Example: Built-in Cost Calculation

This example demonstrates the built-in cost calculation collector that
estimates costs based on model pricing and measured metrics.

Features:
- Automatic cost calculation based on model name
- Support for 18+ major LLM models (OpenAI, Anthropic, Google, Meta, Mistral)
- Pricing database included in the package
- Custom pricing overrides supported

Requirements:
    pip install 'llamonitor-async[parquet]'

Usage:
    python llmops_monitoring/examples/05_cost_calculation.py
"""

import asyncio
from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class LLMResponse:
    """Simulated LLM response object."""
    def __init__(self, text: str):
        self.text = text


# Example 1: Basic cost calculation with GPT-4o-mini
@monitor_llm(
    operation_name="gpt4o_mini_call",
    measure_text=True,
    collectors=["cost"],
    custom_attributes={
        "model": "gpt-4o-mini",
        "temperature": 0.7
    }
)
async def call_gpt4o_mini(prompt: str) -> LLMResponse:
    """Simulate GPT-4o-mini call."""
    await asyncio.sleep(0.1)
    # Simulate a 500-word response
    response_text = "This is a simulated response. " * 50
    return LLMResponse(text=response_text)


# Example 2: Claude 3.5 Sonnet
@monitor_llm(
    operation_name="claude_sonnet_call",
    measure_text=True,
    collectors=["cost"],
    custom_attributes={
        "model": "claude-3-5-sonnet",
        "max_tokens": 2000
    }
)
async def call_claude_sonnet(prompt: str) -> LLMResponse:
    """Simulate Claude 3.5 Sonnet call."""
    await asyncio.sleep(0.15)
    # Simulate a longer response
    response_text = "This is a detailed response from Claude. " * 100
    return LLMResponse(text=response_text)


# Example 3: Gemini 1.5 Flash (most cost-effective)
@monitor_llm(
    operation_name="gemini_flash_call",
    measure_text=True,
    collectors=["cost"],
    custom_attributes={
        "model": "gemini-1.5-flash",
        "use_case": "batch_processing"
    }
)
async def call_gemini_flash(prompt: str) -> LLMResponse:
    """Simulate Gemini 1.5 Flash call."""
    await asyncio.sleep(0.05)
    response_text = "Quick response from Gemini. " * 30
    return LLMResponse(text=response_text)


# Example 4: With exact token counts (more accurate)
@monitor_llm(
    operation_name="gpt4_with_tokens",
    measure_text=True,
    collectors=["cost"],
    custom_attributes={
        "model": "gpt-4",
        "input_tokens": 150,
        "output_tokens": 500
    }
)
async def call_gpt4_with_tokens(prompt: str) -> LLMResponse:
    """
    Simulate GPT-4 call with exact token counts.

    When you provide input_tokens and output_tokens in custom_attributes,
    the cost collector uses exact pricing instead of estimating from char_count.
    """
    await asyncio.sleep(0.2)
    response_text = "Precise cost calculation with token counts. " * 50
    return LLMResponse(text=response_text)


async def main():
    """Main example demonstrating cost calculation."""
    logger.info("=" * 70)
    logger.info("Cost Calculation Example")
    logger.info("=" * 70)

    # Initialize monitoring with local parquet storage
    config = MonitorConfig.for_local_dev()
    writer = await initialize_monitoring(config)

    logger.info("\n📊 Making monitored LLM calls with cost tracking...\n")

    # Call different models
    prompts = [
        "What is machine learning?",
        "Explain quantum computing",
        "How do neural networks work?"
    ]

    # Test 1: GPT-4o-mini (very cost-effective)
    logger.info("1️⃣  Testing GPT-4o-mini (input: $0.00015/1K, output: $0.0006/1K)")
    for prompt in prompts[:2]:
        await call_gpt4o_mini(prompt)
    logger.info("   ✓ 2 calls completed\n")

    # Test 2: Claude 3.5 Sonnet (balanced)
    logger.info("2️⃣  Testing Claude 3.5 Sonnet (input: $0.003/1K, output: $0.015/1K)")
    await call_claude_sonnet(prompts[0])
    logger.info("   ✓ 1 call completed\n")

    # Test 3: Gemini 1.5 Flash (fastest & cheapest)
    logger.info("3️⃣  Testing Gemini 1.5 Flash (input: $0.000075/1K, output: $0.0003/1K)")
    await call_gemini_flash(prompts[1])
    logger.info("   ✓ 1 call completed\n")

    # Test 4: GPT-4 with exact token counts
    logger.info("4️⃣  Testing GPT-4 with exact token counts (input: $0.03/1K, output: $0.06/1K)")
    await call_gpt4_with_tokens(prompts[2])
    logger.info("   ✓ 1 call completed\n")

    # Wait for events to flush
    logger.info("💾 Flushing events to storage...")
    await asyncio.sleep(3)
    await writer.stop()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Cost calculation example completed!")
    logger.info("=" * 70)

    # Show how to query the data
    logger.info("\n📖 To view cost data with pandas:\n")
    logger.info("import pandas as pd")
    logger.info("from glob import glob")
    logger.info("")
    logger.info("files = glob('./dev_monitoring_data/**/*.parquet', recursive=True)")
    logger.info("df = pd.concat([pd.read_parquet(f) for f in files])")
    logger.info("")
    logger.info("# View cost breakdown")
    logger.info("cost_cols = ['operation_name', 'custom_attributes']")
    logger.info("df_cost = df[cost_cols].copy()")
    logger.info("")
    logger.info("# Extract cost from custom_attributes JSON")
    logger.info("df_cost['model'] = df['custom_attributes'].apply(lambda x: x.get('model'))")
    logger.info("df_cost['cost'] = df['custom_attributes'].apply(")
    logger.info("    lambda x: x.get('estimated_cost_usd'))")
    logger.info("")
    logger.info("print(df_cost[['operation_name', 'model', 'cost']])")
    logger.info("print(f'\\nTotal estimated cost: ${df_cost[\"cost\"].sum():.6f}')")

    logger.info("\n📊 Supported Models (18 total):\n")
    logger.info("OpenAI:")
    logger.info("  - gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo")
    logger.info("\nAnthropic:")
    logger.info("  - claude-3-opus, claude-3-sonnet, claude-3-5-sonnet, claude-3-haiku")
    logger.info("\nGoogle:")
    logger.info("  - gemini-1.5-pro, gemini-1.5-flash, gemini-1.0-pro")
    logger.info("\nMeta:")
    logger.info("  - llama-3-8b, llama-3-70b")
    logger.info("\nMistral:")
    logger.info("  - mixtral-8x7b, mistral-small, mistral-medium, mistral-large")

    logger.info("\n💡 Tips:")
    logger.info("  - Provide 'model' in custom_attributes for automatic pricing")
    logger.info("  - Add 'input_tokens' and 'output_tokens' for exact costs")
    logger.info("  - Custom pricing can be configured via CostCollector")
    logger.info("  - Unknown models fall back to default pricing")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error running example: {e}", exc_info=True)
