"""
Example: Real-time WebSocket Event Streaming

This example demonstrates how to stream monitoring events in real-time
via WebSockets.

Features demonstrated:
- Enable WebSocket streaming
- Connect WebSocket clients
- Receive events in real-time
- Filter events by session/operation
- Heartbeat/ping-pong

Requirements:
    pip install 'llamonitor-async[api,parquet]' websockets

Usage:
    # Terminal 1: Run this script to start server + generate events
    python llmops_monitoring/examples/08_websocket_streaming.py

    # Terminal 2: Connect with WebSocket client
    python -c "
    import asyncio
    import websockets
    import json

    async def listen():
        uri = 'ws://localhost:8080/api/v1/stream'
        async with websockets.connect(uri) as ws:
            async for message in ws:
                data = json.loads(message)
                print(json.dumps(data, indent=2))

    asyncio.run(listen())
    "
"""

import asyncio
import random
from datetime import datetime
from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig
from llmops_monitoring.schema.config import WebSocketConfig
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class LLMResponse:
    """Simulated LLM response object."""
    def __init__(self, text: str):
        self.text = text


# Example monitored functions
@monitor_llm(
    operation_name="summarize_text",
    measure_text=True,
    collectors=["cost"],
    custom_attributes={
        "model": "gpt-4o-mini"
    }
)
async def summarize_text(text: str) -> LLMResponse:
    """Simulate text summarization."""
    await asyncio.sleep(random.uniform(0.1, 0.2))
    return LLMResponse(text=f"Summary: {text[:30]}...")


@monitor_llm(
    operation_name="translate_text",
    measure_text=True,
    collectors=["cost"],
    custom_attributes={
        "model": "claude-3-5-sonnet"
    }
)
async def translate_text(text: str) -> LLMResponse:
    """Simulate text translation."""
    await asyncio.sleep(random.uniform(0.1, 0.2))
    return LLMResponse(text=f"Translated: {text}")


async def generate_events():
    """Generate events continuously."""
    operations = [
        (summarize_text, "This is a document about machine learning"),
        (translate_text, "Hello world"),
    ]

    while True:
        op_func, text = random.choice(operations)

        try:
            result = await op_func(text)
            logger.info(f"  Generated event: {op_func.__name__}")
        except Exception as e:
            logger.error(f"  Error: {e}")

        await asyncio.sleep(2)  # Generate event every 2 seconds


async def run_streaming_demo():
    """Run the WebSocket streaming demonstration."""
    logger.info("=" * 80)
    logger.info("WebSocket Real-time Streaming Example")
    logger.info("=" * 80)
    logger.info("")

    # Configure monitoring with WebSocket streaming enabled
    config = MonitorConfig.for_local_dev()
    config.extensions["websocket"] = WebSocketConfig(
        enabled=True,
        path="/api/v1/stream"
    ).model_dump()

    # Initialize monitoring
    writer = await initialize_monitoring(config)

    logger.info("✓ Monitoring initialized with WebSocket streaming")
    logger.info("")
    logger.info("📡 WebSocket Endpoints:")
    logger.info("   General stream: ws://localhost:8080/api/v1/stream")
    logger.info("   Session stream: ws://localhost:8080/api/v1/stream/sessions/{session_id}")
    logger.info("   Operation stream: ws://localhost:8080/api/v1/stream/operations/{operation_name}")
    logger.info("")
    logger.info("💡 Connect with WebSocket client:")
    logger.info("")
    logger.info("   Python example:")
    logger.info("   ---------------")
    logger.info("   import asyncio")
    logger.info("   import websockets")
    logger.info("   import json")
    logger.info("")
    logger.info("   async def listen():")
    logger.info("       uri = 'ws://localhost:8080/api/v1/stream'")
    logger.info("       async with websockets.connect(uri) as ws:")
    logger.info("           async for message in ws:")
    logger.info("               data = json.loads(message)")
    logger.info("               print(json.dumps(data, indent=2))")
    logger.info("")
    logger.info("   asyncio.run(listen())")
    logger.info("")
    logger.info("   JavaScript example:")
    logger.info("   ------------------")
    logger.info("   const ws = new WebSocket('ws://localhost:8080/api/v1/stream');")
    logger.info("   ws.onmessage = (event) => {")
    logger.info("       const data = JSON.parse(event.data);")
    logger.info("       console.log(data);")
    logger.info("   };")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    # Start generating events
    logger.info("🚀 Starting event generation...")
    logger.info("   Events will be generated every 2 seconds")
    logger.info("   Connect a WebSocket client to see real-time updates")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    logger.info("")

    try:
        await generate_events()
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Stopping event generation...")

    # Cleanup
    await writer.stop()
    logger.info("✅ Example completed!")


def main():
    """Main function."""
    # Note: This example focuses on the streaming part
    # You would typically run the API server separately
    logger.info("")
    logger.info("This example demonstrates WebSocket streaming.")
    logger.info("For full functionality, run the API server in another terminal:")
    logger.info("")
    logger.info("  python llmops_monitoring/examples/07_aggregation_api.py")
    logger.info("")
    logger.info("Then connect WebSocket clients to receive events in real-time.")
    logger.info("")

    # Run the demo
    try:
        asyncio.run(run_streaming_demo())
    except KeyboardInterrupt:
        logger.info("")
        logger.info("✅ Stopped")


if __name__ == "__main__":
    main()
