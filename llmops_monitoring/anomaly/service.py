"""
Anomaly detection service for monitoring system.

Manages multiple anomaly detectors and integrates with the monitoring pipeline.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta

from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.anomaly.detector import (
    AnomalyDetector,
    AnomalyResult,
    AnomalySeverity,
    ZScoreDetector,
    IQRDetector,
    ErrorRateDetector
)
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class AnomalyDetectionService:
    """
    Service for detecting anomalies in monitoring data.

    Features:
    - Multiple detection methods (Z-score, IQR, error rate)
    - Automatic training on historical data
    - Real-time anomaly detection
    - Alerting and callbacks
    - Periodic retraining
    """

    def __init__(
        self,
        detectors: Optional[List[AnomalyDetector]] = None,
        alert_callback: Optional[Callable] = None
    ):
        """
        Initialize anomaly detection service.

        Args:
            detectors: List of anomaly detectors to use (defaults to all built-in detectors)
            alert_callback: Optional callback function for alerts: f(anomaly_result)
        """
        if detectors is None:
            # Default detectors
            self.detectors = [
                ZScoreDetector(threshold=3.0),
                IQRDetector(k=1.5),
                ErrorRateDetector(threshold=0.1)
            ]
        else:
            self.detectors = detectors

        self.alert_callback = alert_callback
        self.is_trained = False
        self.training_history: List[MetricEvent] = []
        self.detection_stats = {
            "total_checked": 0,
            "anomalies_found": 0,
            "alerts_sent": 0
        }

    async def train(self, events: List[MetricEvent]) -> None:
        """
        Train all detectors on historical data.

        Args:
            events: Historical events for training baselines
        """
        logger.info(f"Training anomaly detectors on {len(events)} events...")

        # Train each detector
        for detector in self.detectors:
            try:
                detector.fit(events)
            except Exception as e:
                logger.error(f"Error training {detector.__class__.__name__}: {e}")

        self.is_trained = True
        self.training_history = events
        logger.info("Anomaly detection training complete")

    async def detect(self, events: List[MetricEvent]) -> List[AnomalyResult]:
        """
        Detect anomalies in new events.

        Args:
            events: Events to check for anomalies

        Returns:
            List of detected anomalies
        """
        if not self.is_trained:
            logger.warning("Anomaly detectors not trained. Call train() first.")
            return []

        all_anomalies = []
        self.detection_stats["total_checked"] += len(events)

        # Run all detectors
        for detector in self.detectors:
            try:
                anomalies = detector.detect(events)
                all_anomalies.extend(anomalies)
            except Exception as e:
                logger.error(f"Error in {detector.__class__.__name__}.detect(): {e}")

        # Deduplicate and sort by severity
        unique_anomalies = self._deduplicate_anomalies(all_anomalies)

        self.detection_stats["anomalies_found"] += len(unique_anomalies)

        # Send alerts for high/critical anomalies
        await self._send_alerts(unique_anomalies)

        return unique_anomalies

    def _deduplicate_anomalies(self, anomalies: List[AnomalyResult]) -> List[AnomalyResult]:
        """
        Deduplicate anomalies (same operation + metric + timestamp).

        When multiple detectors flag the same issue, keep the highest severity.
        """
        # Group by (operation_name, metric_name, timestamp)
        grouped: Dict[tuple, List[AnomalyResult]] = {}

        for anomaly in anomalies:
            key = (anomaly.operation_name, anomaly.metric_name, anomaly.timestamp)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(anomaly)

        # Keep highest severity for each group
        deduplicated = []
        for group in grouped.values():
            # Sort by anomaly_score (descending)
            sorted_group = sorted(group, key=lambda a: a.anomaly_score, reverse=True)
            deduplicated.append(sorted_group[0])

        return deduplicated

    async def _send_alerts(self, anomalies: List[AnomalyResult]) -> None:
        """Send alerts for high-severity anomalies."""
        if not self.alert_callback:
            return

        for anomaly in anomalies:
            # Only alert on high/critical severity
            if anomaly.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]:
                try:
                    if asyncio.iscoroutinefunction(self.alert_callback):
                        await self.alert_callback(anomaly)
                    else:
                        self.alert_callback(anomaly)

                    self.detection_stats["alerts_sent"] += 1

                except Exception as e:
                    logger.error(f"Error sending alert: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            **self.detection_stats,
            "is_trained": self.is_trained,
            "detectors": [d.__class__.__name__ for d in self.detectors],
            "training_size": len(self.training_history)
        }

    async def retrain(self, events: List[MetricEvent]) -> None:
        """
        Retrain detectors with new data.

        Useful for adapting to changing baselines over time.

        Args:
            events: New events to retrain on
        """
        logger.info(f"Retraining anomaly detectors with {len(events)} new events")
        await self.train(events)


# Singleton instance for global access
_anomaly_service: Optional[AnomalyDetectionService] = None


def get_anomaly_service() -> Optional[AnomalyDetectionService]:
    """Get global anomaly detection service instance."""
    return _anomaly_service


def initialize_anomaly_service(
    detectors: Optional[List[AnomalyDetector]] = None,
    alert_callback: Optional[Callable] = None
) -> AnomalyDetectionService:
    """
    Initialize global anomaly detection service.

    Args:
        detectors: List of detectors to use
        alert_callback: Alert callback function

    Returns:
        Initialized service instance

    Example:
        ```python
        def alert_handler(anomaly: AnomalyResult):
            print(f"ALERT: {anomaly.severity} anomaly in {anomaly.operation_name}")

        service = initialize_anomaly_service(alert_callback=alert_handler)
        await service.train(historical_events)
        ```
    """
    global _anomaly_service

    _anomaly_service = AnomalyDetectionService(
        detectors=detectors,
        alert_callback=alert_callback
    )

    logger.info("Anomaly detection service initialized")
    return _anomaly_service
