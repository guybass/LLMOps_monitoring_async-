"""
Example 14: Basic Agent Intelligence

Demonstrates the Multi-Agent Workflow Intelligence Layer:
- Automatic agent detection from monitoring events
- Handoff quality analysis
- Context drift detection
- Coordination graph visualization
- Bottleneck identification
- Coalition analytics

This example simulates a multi-agent customer support system with:
- Routing Agent: Routes customer inquiries
- Research Agent: Researches product information
- Support Agent: Provides customer support
- Escalation Agent: Handles complex issues
"""

import asyncio
from uuid import uuid4, UUID
from datetime import datetime, timedelta
import json
from pathlib import Path

from llmops_monitoring.agent import (
    AgentIntelligenceService,
    AgentDetector,
    HandoffAnalyzer,
    ContextDriftDetector,
    CoordinationGraphBuilder,
    BottleneckDetector,
    CoalitionAnalyzer
)
from llmops_monitoring.schema.events import MetricEvent


# ========== Simulated Multi-Agent Customer Support System ==========


def create_agent_event(
    session_id: UUID,
    agent_name: str,
    agent_type: str,
    operation: str,
    prompt: str,
    completion: str,
    tokens: int,
    latency: float,
    parent_event_id: UUID = None,
    metadata: dict = None
) -> MetricEvent:
    """Create a simulated agent event."""
    base_metadata = {
        "agent_name": agent_name,
        "agent_type": agent_type,
        "operation": operation,
        "role": agent_type
    }

    if metadata:
        base_metadata.update(metadata)

    return MetricEvent(
        event_id=uuid4(),
        session_id=session_id,
        parent_event_id=parent_event_id,
        timestamp=datetime.now(),
        event_type="completion",
        model_name="gpt-4",
        prompt=prompt,
        completion=completion,
        prompt_tokens=tokens // 2,
        completion_tokens=tokens // 2,
        total_tokens=tokens,
        latency_ms=latency,
        cost_usd=tokens * 0.00002,
        metadata=base_metadata
    )


async def simulate_customer_support_workflow(session_id: UUID) -> list:
    """
    Simulate a multi-agent customer support workflow.

    Flow:
    1. Routing Agent receives inquiry
    2. Routes to Research Agent for product info
    3. Research Agent hands off to Support Agent
    4. Support Agent resolves issue OR escalates
    5. If escalated, Escalation Agent takes over
    """
    events = []

    print(f"\n📞 Simulating customer support session: {session_id}")
    print("-" * 80)

    # Step 1: Routing Agent receives customer inquiry
    routing_event = create_agent_event(
        session_id=session_id,
        agent_name="RoutingAgent",
        agent_type="router",
        operation="route_inquiry",
        prompt="Customer inquiry: 'My subscription is not working after the recent update'",
        completion="Routing to research agent for product information about recent updates",
        tokens=150,
        latency=200.0,
        metadata={"intent": "technical_support", "priority": "medium"}
    )
    events.append(routing_event)
    print(f"  ✓ Routing Agent: Analyzed inquiry and determined route")
    await asyncio.sleep(0.1)

    # Step 2: Research Agent gathers product information
    research_event = create_agent_event(
        session_id=session_id,
        agent_name="ResearchAgent",
        agent_type="researcher",
        operation="gather_information",
        prompt="Find information about recent subscription system updates and known issues",
        completion="Found: Recent update v2.3.1 introduced OAuth changes. Known issue with legacy accounts.",
        tokens=450,
        latency=800.0,
        parent_event_id=routing_event.event_id,
        metadata={"sources": ["knowledge_base", "release_notes"], "confidence": 0.85}
    )
    events.append(research_event)
    print(f"  ✓ Research Agent: Gathered relevant product information")
    await asyncio.sleep(0.1)

    # Step 3: Handoff to Support Agent (with context)
    support_event = create_agent_event(
        session_id=session_id,
        agent_name="SupportAgent",
        agent_type="support",
        operation="provide_support",
        prompt="Customer has subscription issue. Context: OAuth changes in v2.3.1, legacy account issue",
        completion="Provided solution: Update account credentials in new OAuth flow. Sent detailed guide.",
        tokens=320,
        latency=600.0,
        parent_event_id=research_event.event_id,
        metadata={
            "handoff_from": "ResearchAgent",
            "context_received": True,
            "solution_provided": True,
            "satisfaction_score": 0.9
        }
    )
    events.append(support_event)
    print(f"  ✓ Support Agent: Provided solution based on research")
    await asyncio.sleep(0.1)

    # Step 4: Follow-up check
    followup_event = create_agent_event(
        session_id=session_id,
        agent_name="SupportAgent",
        agent_type="support",
        operation="follow_up",
        prompt="Check if customer successfully resolved the issue",
        completion="Customer confirmed: Issue resolved. Subscription is now working correctly.",
        tokens=100,
        latency=150.0,
        parent_event_id=support_event.event_id,
        metadata={"resolved": True, "follow_up_required": False}
    )
    events.append(followup_event)
    print(f"  ✓ Support Agent: Confirmed resolution")

    print(f"\n✅ Session complete: {len(events)} agent interactions")

    return events


async def simulate_escalation_workflow(session_id: UUID) -> list:
    """Simulate a workflow that requires escalation."""
    events = []

    print(f"\n⚠️  Simulating escalation workflow: {session_id}")
    print("-" * 80)

    # Routing
    routing_event = create_agent_event(
        session_id=session_id,
        agent_name="RoutingAgent",
        agent_type="router",
        operation="route_inquiry",
        prompt="Customer inquiry: 'Data privacy concern about recent security breach'",
        completion="High priority issue detected. Routing to research for latest information.",
        tokens=120,
        latency=180.0,
        metadata={"intent": "security_concern", "priority": "high"}
    )
    events.append(routing_event)
    print(f"  ✓ Routing Agent: Detected high priority security concern")
    await asyncio.sleep(0.1)

    # Research (takes longer due to complexity)
    research_event = create_agent_event(
        session_id=session_id,
        agent_name="ResearchAgent",
        agent_type="researcher",
        operation="gather_information",
        prompt="Research recent security incidents and company's public response",
        completion="Found: Security advisory published 2 days ago. Affected users being notified.",
        tokens=600,
        latency=1500.0,  # High latency = potential bottleneck
        parent_event_id=routing_event.event_id,
        metadata={"sources": ["security_advisories", "pr_statements"], "sensitive": True}
    )
    events.append(research_event)
    print(f"  ✓ Research Agent: Gathered security information (slow response)")
    await asyncio.sleep(0.1)

    # Support attempts to handle but needs escalation
    support_attempt_event = create_agent_event(
        session_id=session_id,
        agent_name="SupportAgent",
        agent_type="support",
        operation="assess_issue",
        prompt="Customer concerned about data privacy. Context: Recent security advisory.",
        completion="Issue complexity beyond standard support. Escalating to specialized team.",
        tokens=200,
        latency=300.0,
        parent_event_id=research_event.event_id,
        metadata={
            "handoff_from": "ResearchAgent",
            "escalation_required": True,
            "escalation_reason": "security_specialist_needed"
        }
    )
    events.append(support_attempt_event)
    print(f"  ✓ Support Agent: Determined escalation needed")
    await asyncio.sleep(0.1)

    # Escalation to specialist
    escalation_event = create_agent_event(
        session_id=session_id,
        agent_name="EscalationAgent",
        agent_type="escalation",
        operation="handle_escalation",
        prompt="Security concern escalation. Context: Data privacy inquiry, security advisory.",
        completion="Provided detailed response about security measures, user data protection, and next steps.",
        tokens=550,
        latency=900.0,
        parent_event_id=support_attempt_event.event_id,
        metadata={
            "handoff_from": "SupportAgent",
            "context_received": True,
            "specialist_type": "security",
            "resolution_time_ms": 900
        }
    )
    events.append(escalation_event)
    print(f"  ✓ Escalation Agent: Handled security concern")

    print(f"\n✅ Escalation complete: {len(events)} agent interactions")

    return events


async def simulate_poor_handoff_workflow(session_id: UUID) -> list:
    """Simulate a workflow with poor handoffs (context loss)."""
    events = []

    print(f"\n⚠️  Simulating workflow with poor handoffs: {session_id}")
    print("-" * 80)

    # Initial routing
    routing_event = create_agent_event(
        session_id=session_id,
        agent_name="RoutingAgent",
        agent_type="router",
        operation="route_inquiry",
        prompt="Customer inquiry about billing discrepancy",
        completion="Routing to support agent",
        tokens=100,
        latency=150.0,
        metadata={"intent": "billing", "priority": "medium"}
    )
    events.append(routing_event)
    print(f"  ✓ Routing Agent: Routed billing inquiry")
    await asyncio.sleep(0.1)

    # Support agent WITHOUT proper context
    support_event = create_agent_event(
        session_id=session_id,
        agent_name="SupportAgent",
        agent_type="support",
        operation="handle_inquiry",
        prompt="Handle billing issue",  # Vague prompt - context loss!
        completion="Asked customer to provide full account details and description of issue",
        tokens=120,
        latency=200.0,
        parent_event_id=routing_event.event_id,
        metadata={
            "handoff_from": "RoutingAgent",
            "context_received": False,  # Context loss!
            "additional_info_requested": True
        }
    )
    events.append(support_event)
    print(f"  ⚠️  Support Agent: Had to ask customer for details (context loss)")
    await asyncio.sleep(0.1)

    # Customer re-explains (poor experience)
    re_explanation_event = create_agent_event(
        session_id=session_id,
        agent_name="SupportAgent",
        agent_type="support",
        operation="handle_inquiry",
        prompt="Customer re-explained: 'I was charged twice for my monthly subscription'",
        completion="Investigated and found duplicate charge. Processed refund.",
        tokens=280,
        latency=500.0,
        parent_event_id=support_event.event_id,
        metadata={
            "context_drift_score": 0.75,  # High drift
            "customer_had_to_repeat": True,
            "resolution_delayed": True
        }
    )
    events.append(re_explanation_event)
    print(f"  ✓ Support Agent: Resolved after customer re-explained")

    print(f"\n⚠️  Workflow had handoff issues: {len(events)} interactions")

    return events


# ========== Main Example ==========


async def run_agent_intelligence_example():
    """
    Complete demonstration of agent intelligence capabilities.
    """
    print("=" * 80)
    print("Example 14: Multi-Agent Workflow Intelligence")
    print("=" * 80)
    print()

    # ========== Step 1: Initialize Service ==========
    print("🚀 Step 1: Initializing Agent Intelligence Service")
    print("-" * 80)

    service = AgentIntelligenceService(storage_backend=None)
    await service.initialize()

    print("✅ Service initialized with all analyzers:")
    print("   - AgentDetector: Identifies agents from traces")
    print("   - HandoffAnalyzer: Analyzes handoff quality")
    print("   - ContextDriftDetector: Detects context loss")
    print("   - CoordinationGraphBuilder: Visualizes workflow")
    print("   - BottleneckDetector: Identifies performance issues")
    print("   - CoalitionAnalyzer: Finds agent collaboration patterns")
    print()

    # ========== Step 2: Simulate Workflows ==========
    print("🎭 Step 2: Simulating Multi-Agent Workflows")
    print("-" * 80)

    # Good workflow
    session1 = uuid4()
    events1 = await simulate_customer_support_workflow(session1)
    for event in events1:
        await service.process_event(event)

    # Escalation workflow
    session2 = uuid4()
    events2 = await simulate_escalation_workflow(session2)
    for event in events2:
        await service.process_event(event)

    # Poor handoff workflow
    session3 = uuid4()
    events3 = await simulate_poor_handoff_workflow(session3)
    for event in events3:
        await service.process_event(event)

    print()

    # ========== Step 3: Analyze Agents ==========
    print("🤖 Step 3: Analyzing Detected Agents")
    print("-" * 80)

    all_agents = await service.get_active_agents()

    print(f"\n📊 Detected {len(all_agents)} unique agents:")
    print()

    for agent in all_agents:
        print(f"  • {agent.name} ({agent.agent_type})")
        print(f"    Role: {agent.role}")
        print(f"    Operations: {agent.total_operations}")
        print(f"    Total Tokens: {agent.total_tokens:,}")
        print(f"    Total Cost: ${agent.total_cost:.4f}")
        print(f"    Status: {agent.status}")
        print()

    # ========== Step 4: Analyze Each Session ==========
    print("🔍 Step 4: Comprehensive Session Analysis")
    print("-" * 80)

    for i, session_id in enumerate([session1, session2, session3], 1):
        print(f"\n{'='*80}")
        print(f"Session {i} Analysis: {session_id}")
        print(f"{'='*80}")

        analysis = await service.analyze_session(session_id)

        print(f"\n📈 Session Overview:")
        print(f"   Events: {analysis['event_count']}")
        print(f"   Agents: {len(analysis['agents'])} involved")

        if analysis['agents']:
            print(f"\n🤖 Participating Agents:")
            for agent_data in analysis['agents']:
                print(f"      - {agent_data['name']} ({agent_data['agent_type']})")

        if analysis['coordination_graph']:
            graph = analysis['coordination_graph']
            print(f"\n🕸️  Coordination Graph:")
            print(f"      Nodes: {graph['nodes']}")
            print(f"      Edges: {graph['edges']}")
            print(f"      Execution Paths: {graph['execution_paths']}")
            print(f"      Parallelism Degree: {graph['parallelism_degree']:.2f}")
            print(f"      Longest Path: {graph['longest_path_ms']:.1f}ms")

        if analysis['bottlenecks']:
            print(f"\n🚦 Bottlenecks Detected: {len(analysis['bottlenecks'])}")
            for bn in analysis['bottlenecks']:
                print(f"      ⚠️  Agent: {bn['agent_id']}")
                print(f"         Severity: {bn['severity']}")
                print(f"         Avg Duration: {bn['avg_duration_ms']:.1f}ms")
                print(f"         P95 Duration: {bn['p95_duration_ms']:.1f}ms")
                print(f"         💡 {bn['recommendation']}")

        if analysis['coalitions']:
            print(f"\n🤝 Coalitions Detected: {len(analysis['coalitions'])}")
            for coalition in analysis['coalitions']:
                print(f"      Type: {coalition['coalition_type']}")
                print(f"      Agents: {len(coalition['agent_ids'])}")
                print(f"      Cohesion: {coalition['cohesion_score']:.2f}")
                print(f"      Interactions: {coalition['total_interactions']}")

        if analysis['context_drift']:
            print(f"\n⚠️  Context Drift Events: {len(analysis['context_drift'])}")
            for drift in analysis['context_drift'][:3]:  # Show first 3
                print(f"      Event {drift['event_index']}: Score {drift['drift_score']:.2f}")
                print(f"         Reason: {drift['reason']}")

    print()

    # ========== Step 5: Get Session Summaries ==========
    print("📊 Step 5: Session Summaries")
    print("-" * 80)
    print()

    for i, session_id in enumerate([session1, session2, session3], 1):
        summary = await service.get_session_summary(session_id)

        workflow_type = [
            "Standard Support Workflow",
            "Escalation Workflow",
            "Poor Handoff Workflow"
        ][i-1]

        print(f"Session {i} - {workflow_type}:")
        print(f"   Agents: {summary['agent_count']}")
        print(f"   Events: {summary['event_count']}")
        print(f"   Total Tokens: {summary['total_tokens']:,}")
        print(f"   Total Cost: ${summary['total_cost_usd']:.4f}")
        print(f"   Avg Latency: {summary['avg_latency_ms']:.1f}ms")
        print(f"   Bottlenecks: {summary['bottleneck_count']} ({summary['high_severity_bottlenecks']} high)")
        print(f"   Coalitions: {summary['coalition_count']}")
        print(f"   Parallelism: {summary['parallelism_degree']:.2f}")
        print()

    # ========== Step 6: Export Data ==========
    print("💾 Step 6: Exporting Analysis Data")
    print("-" * 80)

    output_dir = Path(__file__).parent.parent.parent / "reports"
    output_dir.mkdir(exist_ok=True)

    # Export full analysis
    all_analysis = {
        "sessions": {}
    }

    for session_id in [session1, session2, session3]:
        analysis = await service.analyze_session(session_id)
        all_analysis["sessions"][str(session_id)] = analysis

    output_file = output_dir / "agent_intelligence_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(all_analysis, f, indent=2, default=str)

    print(f"✅ Analysis exported to: {output_file}")
    print()

    # ========== Step 7: Recommendations ==========
    print("💡 Step 7: Key Insights & Recommendations")
    print("-" * 80)
    print()

    print("✅ Good Practices Observed:")
    print("   1. Session 1 showed clean handoffs with proper context preservation")
    print("   2. Agents correctly specialized in their roles (routing, research, support)")
    print("   3. Coalition formation enabled efficient collaboration")
    print()

    print("⚠️  Issues Identified:")
    print("   1. Session 2: ResearchAgent bottleneck (1500ms latency)")
    print("   2. Session 3: Context drift during handoffs")
    print("   3. Some agents show high token consumption relative to value")
    print()

    print("🎯 Recommended Actions:")
    print("   1. Optimize ResearchAgent performance or implement caching")
    print("   2. Improve handoff protocols to preserve context")
    print("   3. Consider adding specialized security agent for faster escalations")
    print("   4. Monitor agent coalition patterns for workflow optimization")
    print()

    # ========== Cleanup ==========
    await service.shutdown()

    print("=" * 80)
    print("✨ Agent Intelligence Example Complete!")
    print("=" * 80)
    print()
    print("This example demonstrated:")
    print("  ✓ Automatic agent detection from monitoring events")
    print("  ✓ Real-time handoff quality analysis")
    print("  ✓ Context drift detection")
    print("  ✓ Coordination graph building")
    print("  ✓ Bottleneck identification")
    print("  ✓ Coalition pattern recognition")
    print()
    print("Next steps:")
    print("  1. Integrate with your monitoring system")
    print("  2. Configure PostgreSQL storage for persistence")
    print("  3. Set up dashboards to visualize agent interactions")
    print("  4. Create alerts for bottlenecks and poor handoffs")
    print()


if __name__ == "__main__":
    print("\n🚀 Starting Agent Intelligence Example...")
    print()

    asyncio.run(run_agent_intelligence_example())
