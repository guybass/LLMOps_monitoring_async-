"""
Agentic Workflow Example: Hierarchical Tracking

This example demonstrates how the monitoring system automatically tracks
parent-child relationships in complex agent workflows.
"""

import asyncio
from llmops_monitoring import monitor_llm, initialize_monitoring, MonitorConfig
from llmops_monitoring.instrumentation.context import monitoring_session, monitoring_trace
from llmops_monitoring.schema.config import StorageConfig


class LLMResponse:
    def __init__(self, text: str):
        self.text = text


@monitor_llm(
    operation_name="classify_intent",
    operation_type="llm_call",
    measure_text=["char_count", "word_count"],
    custom_attributes={"agent": "classifier"}
)
async def classify_intent(query: str) -> LLMResponse:
    """First agent: classifies user intent."""
    await asyncio.sleep(0.05)
    return LLMResponse(text=f"Intent: research_question")


@monitor_llm(
    operation_name="search_knowledge_base",
    operation_type="tool_call",
    measure_text=True,
    custom_attributes={"agent": "retriever"}
)
async def search_knowledge_base(intent: str) -> LLMResponse:
    """Second agent: retrieves relevant information."""
    await asyncio.sleep(0.1)
    return LLMResponse(text="Retrieved knowledge: " + "Some relevant information. " * 20)


@monitor_llm(
    operation_name="generate_response",
    operation_type="llm_call",
    measure_text=True,
    custom_attributes={"agent": "generator"}
)
async def generate_response(knowledge: str) -> LLMResponse:
    """Third agent: generates final response."""
    await asyncio.sleep(0.15)
    return LLMResponse(text=f"Based on {knowledge[:50]}..., here is the answer: " + "detailed response " * 15)


@monitor_llm(
    operation_name="orchestrator",
    operation_type="agent_workflow",
    measure_text=True
)
async def run_agent_workflow(user_query: str, session_id: str) -> LLMResponse:
    """
    Orchestrator that coordinates multiple agents.
    All nested calls will be tracked hierarchically.
    """
    print(f"\nProcessing query: {user_query}")

    # Step 1: Classify intent
    intent_result = await classify_intent(user_query)
    print(f"  - Intent classified: {intent_result.text[:30]}...")

    # Step 2: Search knowledge base
    knowledge = await search_knowledge_base(intent_result.text)
    print(f"  - Knowledge retrieved: {len(knowledge.text)} chars")

    # Step 3: Generate response
    final_response = await generate_response(knowledge.text)
    print(f"  - Response generated: {len(final_response.text)} chars")

    return final_response


async def main():
    # Initialize monitoring
    config = MonitorConfig.for_local_dev()
    config.storage.output_dir = "./agentic_monitoring_data"

    writer = await initialize_monitoring(config)

    print("Running agentic workflow example...")
    print("This demonstrates hierarchical span tracking.\n")

    # Simulate multiple user sessions
    for session_num in range(3):
        session_id = f"session-{session_num}"

        # Each session can have multiple traces (conversations)
        with monitoring_session(session_id):
            for trace_num in range(2):
                with monitoring_trace(f"conversation-{trace_num}"):
                    user_query = f"What is the answer to question {session_num}-{trace_num}?"

                    # Run the workflow (all nested calls tracked automatically)
                    response = await run_agent_workflow(user_query, session_id)

    # Flush events
    await asyncio.sleep(3)
    await writer.stop()

    print("\nDone! Check ./agentic_monitoring_data for results.")
    print("\nThe Parquet files contain hierarchical data:")
    print("  - session_id groups all operations in a session")
    print("  - trace_id groups operations in a conversation")
    print("  - parent_span_id links child operations to parents")
    print("\nYou can reconstruct the full call hierarchy from this data!")


if __name__ == "__main__":
    asyncio.run(main())
