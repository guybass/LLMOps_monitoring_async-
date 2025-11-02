"""
Example: Pluggable Measurement Strategies

Demonstrates the new measurement system that supports:
- Capacity-based measurement (chars, words, bytes, lines)
- Token-based measurement (industry standard)
- Hybrid measurement (both capacity AND tokens)
- Auto-selection based on model

This provides flexibility for different use cases:
- Cost tracking: Use token measurement
- Performance monitoring: Use capacity measurement
- Migration: Use hybrid to compare both
- Auto-detection: Let the system choose
"""

import asyncio
from llmops_monitoring.measurement import (
    MeasurementStrategyResolver,
    measure_text,
    StrategyRegistry
)
from llmops_monitoring.decorators import monitor_llm
from llmops_monitoring.transport.writer import MonitoringWriter


async def demo_capacity_strategy():
    """Demo 1: Capacity-based measurement (chars, words, bytes, lines)."""
    print("\n" + "="*70)
    print("Demo 1: Capacity Strategy")
    print("="*70)

    resolver = MeasurementStrategyResolver(mode="capacity")

    text = """Hello, world! This is a test of the capacity measurement strategy.
It counts characters, words, bytes, and lines."""

    result = await resolver.measure(text)

    print(f"\nText: {text[:50]}...")
    print(f"\nCapacity Metrics:")
    print(f"  Characters: {result.char_count}")
    print(f"  Words: {result.word_count}")
    print(f"  Bytes: {result.byte_size}")
    print(f"  Lines: {result.line_count}")
    print(f"\nMetadata:")
    print(f"  Strategy: {result.metadata.strategy_name}")
    print(f"  Type: {result.metadata.measurement_type.value}")
    print(f"  Reliability: {result.metadata.reliability.value}")
    print(f"  Time: {result.metadata.measurement_time_ms:.2f}ms")
    print(f"  Confidence: {result.metadata.confidence_score}")


async def demo_token_strategy():
    """Demo 2: Token-based measurement (industry standard)."""
    print("\n" + "="*70)
    print("Demo 2: Token Strategy")
    print("="*70)

    resolver = MeasurementStrategyResolver(mode="token")

    text = "Hello, world! This is a test of the token measurement strategy."

    # With GPT-4 model (uses tiktoken)
    context = {"model": "gpt-4"}
    result = await resolver.measure(text, context)

    print(f"\nText: {text}")
    print(f"Model: {context['model']}")
    print(f"\nToken Metrics:")
    print(f"  Total Tokens: {result.total_tokens}")
    print(f"\nMetadata:")
    print(f"  Strategy: {result.metadata.strategy_name}")
    print(f"  Tokenizer: {result.metadata.tokenizer_name}")
    print(f"  Reliability: {result.metadata.reliability.value}")
    print(f"  Time: {result.metadata.measurement_time_ms:.2f}ms")

    # Without model (falls back to estimation)
    result_estimated = await resolver.measure(text)
    print(f"\nWithout model (estimated):")
    print(f"  Total Tokens: {result_estimated.total_tokens}")
    print(f"  Reliability: {result_estimated.metadata.reliability.value}")
    print(f"  Fallback Reason: {result_estimated.metadata.fallback_reason}")


async def demo_hybrid_strategy():
    """Demo 3: Hybrid measurement (both capacity AND tokens)."""
    print("\n" + "="*70)
    print("Demo 3: Hybrid Strategy")
    print("="*70)

    resolver = MeasurementStrategyResolver(mode="hybrid")

    text = "The hybrid strategy measures both capacity AND tokens in parallel!"

    context = {"model": "gpt-4"}
    result = await resolver.measure(text, context)

    print(f"\nText: {text}")
    print(f"\nCapacity Metrics:")
    print(f"  Characters: {result.char_count}")
    print(f"  Words: {result.word_count}")
    print(f"  Bytes: {result.byte_size}")
    print(f"\nToken Metrics:")
    print(f"  Total Tokens: {result.total_tokens}")
    print(f"\nMetadata:")
    print(f"  Strategy: {result.metadata.strategy_name}")
    print(f"  Type: {result.metadata.measurement_type.value}")
    print(f"  Reliability: {result.metadata.reliability.value}")
    print(f"  Time: {result.metadata.measurement_time_ms:.2f}ms")
    print(f"  Capacity Available: {result.metadata.extra['capacity_available']}")
    print(f"  Tokens Available: {result.metadata.extra['tokens_available']}")
    print(f"  Parallel Execution: {result.metadata.extra['parallel_execution']}")


async def demo_auto_strategy():
    """Demo 4: Auto-selection based on model."""
    print("\n" + "="*70)
    print("Demo 4: Auto Strategy")
    print("="*70)

    resolver = MeasurementStrategyResolver(mode="auto")

    text = "Auto strategy automatically selects the best measurement approach!"

    # With GPT-4 model → will use TokenStrategy
    print("\nWith GPT-4 model:")
    result_gpt4 = await resolver.measure(text, {"model": "gpt-4"})
    print(f"  Selected Strategy: {result_gpt4.metadata.strategy_name}")
    print(f"  Total Tokens: {result_gpt4.total_tokens}")

    # With Claude model → will use TokenStrategy (estimation)
    print("\nWith Claude model:")
    result_claude = await resolver.measure(text, {"model": "claude-3-opus"})
    print(f"  Selected Strategy: {result_claude.metadata.strategy_name}")
    print(f"  Total Tokens: {result_claude.total_tokens}")
    print(f"  Reliability: {result_claude.metadata.reliability.value}")

    # Without model → will use CapacityStrategy
    print("\nWithout model:")
    result_no_model = await resolver.measure(text)
    print(f"  Selected Strategy: {result_no_model.metadata.strategy_name}")
    print(f"  Characters: {result_no_model.char_count}")


async def demo_convenience_function():
    """Demo 5: Using the convenience function."""
    print("\n" + "="*70)
    print("Demo 5: Convenience Function")
    print("="*70)

    text = "Quick measurement using the convenience function!"

    # Simple usage
    result = await measure_text(text, mode="token", context={"model": "gpt-4"})

    print(f"\nText: {text}")
    print(f"Total Tokens: {result.total_tokens}")
    print(f"Strategy: {result.metadata.strategy_name}")


@monitor_llm(
    operation_name="mock_llm_call_capacity",
    measurement="capacity",
    measure_text=True
)
async def mock_llm_with_capacity(prompt: str) -> str:
    """Mock LLM call with capacity measurement."""
    await asyncio.sleep(0.1)  # Simulate API call
    return f"Response to: {prompt}"


@monitor_llm(
    operation_name="mock_llm_call_tokens",
    measurement="token",
    measure_text=True,
    custom_attributes={"model": "gpt-4"}
)
async def mock_llm_with_tokens(prompt: str, model: str = "gpt-4") -> str:
    """Mock LLM call with token measurement."""
    await asyncio.sleep(0.1)  # Simulate API call
    return f"Response to: {prompt}"


@monitor_llm(
    operation_name="mock_llm_call_hybrid",
    measurement="hybrid",
    measure_text=True,
    custom_attributes={"model": "gpt-4"}
)
async def mock_llm_with_hybrid(prompt: str) -> str:
    """Mock LLM call with hybrid measurement."""
    await asyncio.sleep(0.1)  # Simulate API call
    return f"Response to: {prompt}"


@monitor_llm(
    operation_name="mock_llm_call_auto",
    measurement="auto",
    measure_text=True,
    custom_attributes={"model": "gpt-4"}
)
async def mock_llm_with_auto(prompt: str) -> str:
    """Mock LLM call with auto measurement."""
    await asyncio.sleep(0.1)  # Simulate API call
    return f"Response to: {prompt}"


async def demo_decorator_integration():
    """Demo 6: Using measurement with @monitor_llm decorator."""
    print("\n" + "="*70)
    print("Demo 6: Decorator Integration")
    print("="*70)

    # Initialize monitoring writer
    writer = MonitoringWriter.get_instance()
    await writer.start()

    try:
        # Capacity measurement
        print("\n1. Capacity measurement:")
        result = await mock_llm_with_capacity("What is the capital of France?")
        print(f"   Result: {result}")

        # Token measurement
        print("\n2. Token measurement:")
        result = await mock_llm_with_tokens("Explain quantum computing")
        print(f"   Result: {result}")

        # Hybrid measurement
        print("\n3. Hybrid measurement:")
        result = await mock_llm_with_hybrid("Tell me a joke")
        print(f"   Result: {result}")

        # Auto measurement
        print("\n4. Auto measurement:")
        result = await mock_llm_with_auto("What is 2+2?")
        print(f"   Result: {result}")

        # Wait for events to be processed
        await asyncio.sleep(1)

        print("\n✓ All measurements collected successfully!")
        print("  Events have been sent to the monitoring backend.")

    finally:
        await writer.stop()


async def demo_strategy_info():
    """Demo 7: Strategy information and configuration."""
    print("\n" + "="*70)
    print("Demo 7: Strategy Information")
    print("="*70)

    # List available strategies
    strategies = StrategyRegistry.list_strategies()
    print(f"\nAvailable Strategies: {strategies}")

    # Get resolver info
    resolver = MeasurementStrategyResolver(
        mode="hybrid",
        config={
            "prefer_tokens": True,
            "fallback_enabled": True,
            "parallel_hybrid": True
        }
    )

    info = resolver.get_strategy_info()
    print(f"\nResolver Configuration:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Get strategy priorities
    print(f"\nStrategy Priorities:")
    for name in strategies:
        strategy = StrategyRegistry.get_strategy(name)
        if strategy:
            print(f"  {name}: priority={strategy.get_priority()}, "
                  f"cost={strategy.estimate_cost_impact()}")


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("PLUGGABLE MEASUREMENT STRATEGIES - COMPREHENSIVE DEMO")
    print("="*70)

    # Run demos
    await demo_capacity_strategy()
    await demo_token_strategy()
    await demo_hybrid_strategy()
    await demo_auto_strategy()
    await demo_convenience_function()
    await demo_decorator_integration()
    await demo_strategy_info()

    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
1. **Capacity Strategy**: Fast, reliable, language-agnostic
   - Use for: Performance monitoring, cross-language comparison
   - Metrics: Characters, words, bytes, lines
   - Latency: <1ms

2. **Token Strategy**: Industry standard, accurate cost tracking
   - Use for: Cost calculation, billing accuracy
   - Metrics: Input tokens, output tokens, total tokens
   - Latency: 5-20ms (with caching)

3. **Hybrid Strategy**: Both capacity AND tokens
   - Use for: Migration, cross-validation, complete visibility
   - Metrics: All capacity + token metrics
   - Latency: max(capacity, tokens) in parallel mode

4. **Auto Strategy**: Intelligent selection
   - Detects model from context
   - Selects best strategy automatically
   - Falls back gracefully when needed

5. **Decorator Integration**: Seamless monitoring
   - Add `measurement` parameter to @monitor_llm
   - Works with existing measure_text parameter
   - Automatic context propagation

RECOMMENDATION:
- Start with "auto" mode for most use cases
- Use "token" mode when model is known and cost tracking is critical
- Use "hybrid" mode during migration or for cross-validation
- Use "capacity" mode for maximum performance
    """)

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Check for tiktoken availability
    try:
        import tiktoken
        print("✓ tiktoken is available (token measurement will work for OpenAI models)")
    except ImportError:
        print("⚠ tiktoken not installed (token measurement will use estimation)")
        print("  Install with: pip install tiktoken")

    # Run demos
    asyncio.run(main())
