"""
Example: ML-based Anomaly Detection for Monitoring Data

This example demonstrates how to detect anomalous behavior in LLM operations
using statistical and ML-based methods.

Features:
- Z-score based detection (for normally distributed metrics)
- IQR (Interquartile Range) detection (robust to outliers)
- Error rate anomaly detection
- Automatic baseline training
- Real-time anomaly alerts
- Configurable severity thresholds

Requirements:
    pip install 'llamonitor-async'

Usage:
    python llmops_monitoring/examples/11_anomaly_detection.py
"""

import asyncio
import random
from datetime import datetime

from llamonitor import monitor_llm, initialize_monitoring, MonitorConfig
from llmops_monitoring.schema.config import StorageConfig
from llmops_monitoring.anomaly import AnomalyDetectionService, AnomalyResult
from llmops_monitoring.aggregation import create_aggregation_service
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


@monitor_llm(
    operation_name="normal_operation",
    measure_text=True,
    custom_attributes={"model": "gpt-4o-mini"}
)
async def normal_operation(prompt: str) -> dict:
    """Simulates normal LLM operation with consistent behavior."""
    await asyncio.sleep(random.uniform(0.08, 0.12))  # Normal latency
    response = f"Normal response to: {prompt}"
    return {"text": response}


@monitor_llm(
    operation_name="anomalous_operation",
    measure_text=True,
    custom_attributes={"model": "gpt-4"}
)
async def anomalous_operation(prompt: str, inject_anomaly: bool = False) -> dict:
    """Simulates operation with potential anomalies."""
    if inject_anomaly:
        # Inject anomalous behavior
        await asyncio.sleep(random.uniform(0.5, 1.0))  # Much higher latency
        response = f"Anomalous long response: {prompt * 10}"  # Much more text
    else:
        await asyncio.sleep(random.uniform(0.08, 0.12))  # Normal
        response = f"Normal response to: {prompt}"

    return {"text": response}


def alert_handler(anomaly: AnomalyResult):
    """Handle anomaly alerts."""
    logger.warning(f"\n{'='*70}")
    logger.warning(f"🚨 ANOMALY DETECTED!")
    logger.warning(f"{'='*70}")
    logger.warning(f"Operation: {anomaly.operation_name}")
    logger.warning(f"Metric: {anomaly.metric_name}")
    logger.warning(f"Severity: {anomaly.severity.value if anomaly.severity else 'N/A'}")
    logger.warning(f"Anomaly Score: {anomaly.anomaly_score:.3f}")
    logger.warning(f"Detection Method: {anomaly.method}")
    logger.warning(f"Timestamp: {anomaly.timestamp}")
    logger.warning(f"Details: {anomaly.details}")
    logger.warning(f"{'='*70}\n")


async def generate_training_data():
    """Generate normal baseline data for training."""
    logger.info("Generating training data (normal behavior)...")

    # Generate 100 normal events
    for i in range(100):
        await normal_operation(f"Training prompt {i}")

        # Also some normal anomalous_operation events
        await anomalous_operation(f"Training prompt {i}", inject_anomaly=False)

    logger.info("✓ Generated 200 training events")


async def generate_test_data_with_anomalies():
    """Generate test data with some anomalies injected."""
    logger.info("\nGenerating test data with injected anomalies...")

    # Generate mostly normal events with some anomalies
    for i in range(50):
        await normal_operation(f"Test prompt {i}")

        # Inject anomalies in some events
        inject_anomaly = (i % 10 == 0)  # Every 10th event is anomalous
        await anomalous_operation(f"Test prompt {i}", inject_anomaly=inject_anomaly)

    logger.info("✓ Generated 100 test events (10% with injected anomalies)")


async def main():
    """Main example demonstrating anomaly detection."""
    logger.info("="*70)
    logger.info("ML-based Anomaly Detection Example")
    logger.info("="*70)

    # Configure monitoring
    config = MonitorConfig(
        storage=StorageConfig(
            backend="parquet",
            output_dir="./anomaly_demo_data",
            batch_size=50,
            flush_interval_seconds=2.0
        )
    )

    # Initialize monitoring
    await initialize_monitoring(config)
    logger.info("✓ Monitoring initialized")

    # Phase 1: Generate training data
    logger.info("\n" + "="*70)
    logger.info("Phase 1: Generating Baseline Training Data")
    logger.info("="*70)
    await generate_training_data()

    # Wait for flush
    await asyncio.sleep(3)

    # Phase 2: Load training data and train detectors
    logger.info("\n" + "="*70)
    logger.info("Phase 2: Training Anomaly Detectors")
    logger.info("="*70)

    # Create aggregation service to load historical data
    agg_service = await create_aggregation_service(config)

    # Query all training events
    training_events = await agg_service.query_events(limit=1000)
    logger.info(f"Loaded {len(training_events)} training events")

    # Initialize anomaly detection service with alert callback
    anomaly_service = AnomalyDetectionService(alert_callback=alert_handler)

    # Train detectors
    await anomaly_service.train(training_events)

    logger.info("\nTraining Statistics:")
    stats = anomaly_service.get_stats()
    logger.info(f"  Training Size: {stats['training_size']} events")
    logger.info(f"  Detectors: {', '.join(stats['detectors'])}")
    logger.info("✓ Detectors trained successfully")

    # Phase 3: Generate test data with anomalies
    logger.info("\n" + "="*70)
    logger.info("Phase 3: Testing with Anomalous Data")
    logger.info("="*70)
    await generate_test_data_with_anomalies()

    # Wait for flush
    await asyncio.sleep(3)

    # Phase 4: Detect anomalies
    logger.info("\n" + "="*70)
    logger.info("Phase 4: Running Anomaly Detection")
    logger.info("="*70)

    # Query test events
    test_events = await agg_service.query_events(limit=1000)
    logger.info(f"Loaded {len(test_events)} total events for analysis")

    # Detect anomalies
    anomalies = await anomaly_service.detect(test_events)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("Anomaly Detection Summary")
    logger.info("="*70)

    stats = anomaly_service.get_stats()
    logger.info(f"Total Events Checked: {stats['total_checked']}")
    logger.info(f"Anomalies Found: {stats['anomalies_found']}")
    logger.info(f"Alerts Sent: {stats['alerts_sent']}")

    # Group anomalies by severity
    by_severity = {}
    for anomaly in anomalies:
        severity = anomaly.severity.value if anomaly.severity else "none"
        if severity not in by_severity:
            by_severity[severity] = 0
        by_severity[severity] += 1

    logger.info("\nAnomalies by Severity:")
    for severity, count in sorted(by_severity.items()):
        logger.info(f"  {severity.upper()}: {count}")

    # Group anomalies by detection method
    by_method = {}
    for anomaly in anomalies:
        method = anomaly.method
        if method not in by_method:
            by_method[method] = 0
        by_method[method] += 1

    logger.info("\nAnomalies by Detection Method:")
    for method, count in sorted(by_method.items()):
        logger.info(f"  {method}: {count}")

    # Group anomalies by operation
    by_operation = {}
    for anomaly in anomalies:
        op_name = anomaly.operation_name
        if op_name not in by_operation:
            by_operation[op_name] = 0
        by_operation[op_name] += 1

    logger.info("\nAnomalies by Operation:")
    for op_name, count in sorted(by_operation.items()):
        logger.info(f"  {op_name}: {count}")

    # Show some example anomalies
    logger.info("\nExample Anomalies Detected:")
    logger.info("-" * 70)
    for anomaly in anomalies[:5]:  # Show first 5
        logger.info(f"\n  Operation: {anomaly.operation_name}")
        logger.info(f"  Metric: {anomaly.metric_name}")
        logger.info(f"  Method: {anomaly.method}")
        logger.info(f"  Score: {anomaly.anomaly_score:.3f}")
        logger.info(f"  Severity: {anomaly.severity.value if anomaly.severity else 'N/A'}")
        if "value" in anomaly.details:
            logger.info(f"  Value: {anomaly.details['value']:.2f}")
        if "mean" in anomaly.details:
            logger.info(f"  Expected (mean): {anomaly.details['mean']:.2f}")
        if "z_score" in anomaly.details:
            logger.info(f"  Z-score: {anomaly.details['z_score']:.2f}")

    logger.info("\n" + "="*70)
    logger.info("✓ Anomaly detection complete!")
    logger.info("="*70)

    logger.info("\nKey Insights:")
    logger.info("  • Z-score detector works well for normally distributed metrics")
    logger.info("  • IQR detector is more robust to outliers")
    logger.info("  • Error rate detector catches unusual error patterns")
    logger.info("  • Multiple detectors provide comprehensive coverage")
    logger.info("  • Alerts sent only for HIGH/CRITICAL severity")

    # Cleanup
    await agg_service.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error running example: {e}", exc_info=True)
