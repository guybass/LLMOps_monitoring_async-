"""
Basic Monitoring Test - Option A

Tests basic functionality without needing API keys.
Run with: python test_basic_monitoring.py
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig
from llmops_monitoring.schema.config import StorageConfig


# Simulated responses (no API needed)
class SimulatedTextResponse:
    def __init__(self, text: str):
        self.text = text


class SimulatedImageResponse:
    def __init__(self, images: list):
        self.images = images


# Test 1: Basic text monitoring
@monitor_llm(
    operation_name="test_text_function",
    measure_text=True,
    custom_attributes={"model": "test-model", "test": "basic"}
)
async def test_text_function(prompt: str) -> SimulatedTextResponse:
    """Simulate LLM text generation."""
    await asyncio.sleep(0.05)
    response_text = f"Generated response for: {prompt}. " * 50
    return SimulatedTextResponse(text=response_text)


# Test 2: Image monitoring
@monitor_llm(
    operation_name="test_image_function",
    measure_images=True,
    custom_attributes={"model": "vision-model"}
)
async def test_image_function() -> SimulatedImageResponse:
    """Simulate image processing."""
    await asyncio.sleep(0.03)
    # Simulate image data (bytes)
    fake_images = [b"fake_image_data_1" * 1000, b"fake_image_data_2" * 1500]
    return SimulatedImageResponse(images=fake_images)


# Test 3: Both text and images
@monitor_llm(
    operation_name="test_multimodal",
    measure_text=["char_count", "word_count"],
    measure_images=["count", "file_size_bytes"],
    custom_attributes={"model": "multimodal-model"}
)
async def test_multimodal() -> SimulatedTextResponse:
    """Simulate multimodal response."""
    await asyncio.sleep(0.04)
    return SimulatedTextResponse(text="Multimodal response with both text and images. " * 30)


# Test 4: Nested calls (hierarchical tracking)
@monitor_llm("parent_operation")
async def parent_operation() -> str:
    """Parent operation that calls children."""
    child1_result = await child_operation_1()
    child2_result = await child_operation_2()
    return f"{child1_result} + {child2_result}"


@monitor_llm("child_operation_1", measure_text=True)
async def child_operation_1() -> SimulatedTextResponse:
    """First child operation."""
    await asyncio.sleep(0.02)
    return SimulatedTextResponse(text="Child 1 result. " * 20)


@monitor_llm("child_operation_2", measure_text=True)
async def child_operation_2() -> SimulatedTextResponse:
    """Second child operation."""
    await asyncio.sleep(0.02)
    return SimulatedTextResponse(text="Child 2 result. " * 25)


async def run_tests():
    """Run all basic tests."""
    print("=" * 60)
    print("LLMOps Monitoring - Basic Test Suite")
    print("=" * 60)
    print()

    # Initialize monitoring
    config = MonitorConfig(
        storage=StorageConfig(
            backend="parquet",
            output_dir="./test_monitoring_data",
            batch_size=10,
            flush_interval_seconds=1.0
        )
    )
    writer = await initialize_monitoring(config)
    print("✓ Monitoring initialized (Parquet backend)")
    print(f"  Output: {config.storage.output_dir}")
    print()

    # Test 1: Text monitoring
    print("Test 1: Text Monitoring")
    print("-" * 40)
    result1 = await test_text_function("Hello world!")
    print(f"  ✓ Generated {len(result1.text)} characters")
    print(f"  ✓ ~{len(result1.text.split())} words")
    print()

    # Test 2: Image monitoring
    print("Test 2: Image Monitoring")
    print("-" * 40)
    result2 = await test_image_function()
    total_size = sum(len(img) for img in result2.images)
    print(f"  ✓ Processed {len(result2.images)} images")
    print(f"  ✓ Total size: {total_size:,} bytes")
    print()

    # Test 3: Multimodal
    print("Test 3: Multimodal (Text + Images)")
    print("-" * 40)
    result3 = await test_multimodal()
    print(f"  ✓ Generated {len(result3.text)} characters")
    print()

    # Test 4: Hierarchical tracking
    print("Test 4: Hierarchical Tracking (Nested Calls)")
    print("-" * 40)
    result4 = await parent_operation()
    print(f"  ✓ Parent operation completed")
    print(f"  ✓ Called 2 child operations")
    print(f"  ✓ Result: {result4[:50]}...")
    print()

    # Wait for events to flush
    print("Flushing events to storage...")
    await asyncio.sleep(2)
    await writer.stop()
    print("✓ All events written to Parquet files")
    print()

    # Verify files created
    output_path = Path(config.storage.output_dir)
    if output_path.exists():
        parquet_files = list(output_path.rglob("*.parquet"))
        print(f"✓ Created {len(parquet_files)} Parquet file(s)")
        for pf in parquet_files:
            print(f"  - {pf.relative_to(output_path.parent)}")
    print()

    print("=" * 60)
    print("Basic Tests Complete!")
    print("=" * 60)
    print()
    print("Next Steps:")
    print("1. Check the data:")
    print("   python analyze_results.py")
    print()
    print("2. Run example:")
    print("   python llmops_monitoring/examples/01_simple_example.py")
    print()
    print("3. Test with real LLM calls:")
    print("   - Add API keys to .env")
    print("   - python test_agent_graph_real.py")


if __name__ == "__main__":
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()
