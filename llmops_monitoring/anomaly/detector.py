"""
Anomaly detection algorithms for monitoring metrics.

Provides statistical and ML-based methods for detecting anomalous behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import statistics
import logging

from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomaly: bool
    anomaly_score: float  # 0.0 to 1.0
    severity: Optional[AnomalySeverity] = None
    method: str = ""
    details: Dict[str, Any] = None
    timestamp: datetime = None
    operation_name: str = ""
    metric_name: str = ""

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class AnomalyDetector(ABC):
    """Base class for anomaly detectors."""

    @abstractmethod
    def detect(self, events: List[MetricEvent]) -> List[AnomalyResult]:
        """
        Detect anomalies in a list of events.

        Args:
            events: List of metric events to analyze

        Returns:
            List of anomaly results
        """
        pass

    @abstractmethod
    def fit(self, events: List[MetricEvent]) -> None:
        """
        Train/fit the detector on historical data.

        Args:
            events: Historical events for training
        """
        pass

    def _determine_severity(self, anomaly_score: float) -> AnomalySeverity:
        """Determine severity based on anomaly score."""
        if anomaly_score >= 0.9:
            return AnomalySeverity.CRITICAL
        elif anomaly_score >= 0.7:
            return AnomalySeverity.HIGH
        elif anomaly_score >= 0.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW


class ZScoreDetector(AnomalyDetector):
    """
    Z-score based anomaly detection.

    Detects anomalies based on standard deviations from the mean.
    Works well for normally distributed metrics.
    """

    def __init__(self, threshold: float = 3.0):
        """
        Initialize Z-score detector.

        Args:
            threshold: Number of standard deviations for anomaly threshold
        """
        self.threshold = threshold
        self.baselines: Dict[str, Dict[str, float]] = {}  # operation_name -> metric -> {mean, stdev}

    def fit(self, events: List[MetricEvent]) -> None:
        """Calculate baseline statistics from historical data."""
        # Group metrics by operation
        metrics_by_operation: Dict[str, Dict[str, List[float]]] = {}

        for event in events:
            op_name = event.operation_name
            if op_name not in metrics_by_operation:
                metrics_by_operation[op_name] = {
                    "duration_ms": [],
                    "char_count": [],
                    "word_count": []
                }

            if event.duration_ms is not None:
                metrics_by_operation[op_name]["duration_ms"].append(event.duration_ms)

            if event.text_metrics:
                if event.text_metrics.char_count:
                    metrics_by_operation[op_name]["char_count"].append(event.text_metrics.char_count)
                if event.text_metrics.word_count:
                    metrics_by_operation[op_name]["word_count"].append(event.text_metrics.word_count)

        # Calculate baselines
        for op_name, metrics in metrics_by_operation.items():
            self.baselines[op_name] = {}

            for metric_name, values in metrics.items():
                if len(values) >= 2:  # Need at least 2 values for stdev
                    mean = statistics.mean(values)
                    stdev = statistics.stdev(values)
                    self.baselines[op_name][metric_name] = {
                        "mean": mean,
                        "stdev": stdev
                    }

        logger.info(f"Z-score detector fitted on {len(events)} events for {len(self.baselines)} operations")

    def detect(self, events: List[MetricEvent]) -> List[AnomalyResult]:
        """Detect anomalies using Z-score method."""
        results = []

        for event in events:
            op_name = event.operation_name

            if op_name not in self.baselines:
                continue  # No baseline for this operation

            # Check duration_ms
            if event.duration_ms is not None and "duration_ms" in self.baselines[op_name]:
                result = self._check_metric(
                    event.duration_ms,
                    self.baselines[op_name]["duration_ms"],
                    operation_name=op_name,
                    metric_name="duration_ms",
                    timestamp=event.timestamp
                )
                if result.is_anomaly:
                    results.append(result)

            # Check text metrics
            if event.text_metrics:
                if event.text_metrics.char_count and "char_count" in self.baselines[op_name]:
                    result = self._check_metric(
                        event.text_metrics.char_count,
                        self.baselines[op_name]["char_count"],
                        operation_name=op_name,
                        metric_name="char_count",
                        timestamp=event.timestamp
                    )
                    if result.is_anomaly:
                        results.append(result)

                if event.text_metrics.word_count and "word_count" in self.baselines[op_name]:
                    result = self._check_metric(
                        event.text_metrics.word_count,
                        self.baselines[op_name]["word_count"],
                        operation_name=op_name,
                        metric_name="word_count",
                        timestamp=event.timestamp
                    )
                    if result.is_anomaly:
                        results.append(result)

        return results

    def _check_metric(
        self,
        value: float,
        baseline: Dict[str, float],
        operation_name: str,
        metric_name: str,
        timestamp: datetime
    ) -> AnomalyResult:
        """Check if a metric value is anomalous."""
        mean = baseline["mean"]
        stdev = baseline["stdev"]

        if stdev == 0:
            # Can't detect anomalies if no variation
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                method="z_score",
                operation_name=operation_name,
                metric_name=metric_name,
                timestamp=timestamp
            )

        z_score = abs((value - mean) / stdev)
        is_anomaly = z_score > self.threshold

        # Normalize anomaly score to 0-1 range
        # threshold=3 maps to 0.5, threshold=6 maps to 1.0
        anomaly_score = min(z_score / (self.threshold * 2), 1.0)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            severity=self._determine_severity(anomaly_score) if is_anomaly else None,
            method="z_score",
            operation_name=operation_name,
            metric_name=metric_name,
            timestamp=timestamp,
            details={
                "value": value,
                "mean": mean,
                "stdev": stdev,
                "z_score": z_score,
                "threshold": self.threshold
            }
        )


class IQRDetector(AnomalyDetector):
    """
    Interquartile Range (IQR) based anomaly detection.

    More robust to outliers than Z-score method.
    Detects anomalies outside the (Q1 - k*IQR, Q3 + k*IQR) range.
    """

    def __init__(self, k: float = 1.5):
        """
        Initialize IQR detector.

        Args:
            k: Multiplier for IQR range (1.5 = outliers, 3.0 = extreme outliers)
        """
        self.k = k
        self.baselines: Dict[str, Dict[str, Dict[str, float]]] = {}  # operation -> metric -> {q1, q3, iqr}

    def fit(self, events: List[MetricEvent]) -> None:
        """Calculate IQR baselines from historical data."""
        # Group metrics by operation
        metrics_by_operation: Dict[str, Dict[str, List[float]]] = {}

        for event in events:
            op_name = event.operation_name
            if op_name not in metrics_by_operation:
                metrics_by_operation[op_name] = {
                    "duration_ms": [],
                    "char_count": [],
                    "word_count": []
                }

            if event.duration_ms is not None:
                metrics_by_operation[op_name]["duration_ms"].append(event.duration_ms)

            if event.text_metrics:
                if event.text_metrics.char_count:
                    metrics_by_operation[op_name]["char_count"].append(event.text_metrics.char_count)
                if event.text_metrics.word_count:
                    metrics_by_operation[op_name]["word_count"].append(event.text_metrics.word_count)

        # Calculate IQR baselines
        for op_name, metrics in metrics_by_operation.items():
            self.baselines[op_name] = {}

            for metric_name, values in metrics.items():
                if len(values) >= 4:  # Need at least 4 values for quartiles
                    values_sorted = sorted(values)
                    q1 = statistics.median(values_sorted[:len(values_sorted)//2])
                    q3 = statistics.median(values_sorted[len(values_sorted)//2:])
                    iqr = q3 - q1

                    self.baselines[op_name][metric_name] = {
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "lower_bound": q1 - self.k * iqr,
                        "upper_bound": q3 + self.k * iqr
                    }

        logger.info(f"IQR detector fitted on {len(events)} events for {len(self.baselines)} operations")

    def detect(self, events: List[MetricEvent]) -> List[AnomalyResult]:
        """Detect anomalies using IQR method."""
        results = []

        for event in events:
            op_name = event.operation_name

            if op_name not in self.baselines:
                continue

            # Check duration_ms
            if event.duration_ms is not None and "duration_ms" in self.baselines[op_name]:
                result = self._check_metric(
                    event.duration_ms,
                    self.baselines[op_name]["duration_ms"],
                    operation_name=op_name,
                    metric_name="duration_ms",
                    timestamp=event.timestamp
                )
                if result.is_anomaly:
                    results.append(result)

            # Check text metrics
            if event.text_metrics:
                if event.text_metrics.char_count and "char_count" in self.baselines[op_name]:
                    result = self._check_metric(
                        event.text_metrics.char_count,
                        self.baselines[op_name]["char_count"],
                        operation_name=op_name,
                        metric_name="char_count",
                        timestamp=event.timestamp
                    )
                    if result.is_anomaly:
                        results.append(result)

                if event.text_metrics.word_count and "word_count" in self.baselines[op_name]:
                    result = self._check_metric(
                        event.text_metrics.word_count,
                        self.baselines[op_name]["word_count"],
                        operation_name=op_name,
                        metric_name="word_count",
                        timestamp=event.timestamp
                    )
                    if result.is_anomaly:
                        results.append(result)

        return results

    def _check_metric(
        self,
        value: float,
        baseline: Dict[str, float],
        operation_name: str,
        metric_name: str,
        timestamp: datetime
    ) -> AnomalyResult:
        """Check if a metric value is outside IQR bounds."""
        lower_bound = baseline["lower_bound"]
        upper_bound = baseline["upper_bound"]
        iqr = baseline["iqr"]

        is_anomaly = value < lower_bound or value > upper_bound

        if is_anomaly:
            if value < lower_bound:
                distance = lower_bound - value
            else:
                distance = value - upper_bound

            # Normalize anomaly score based on distance from bounds
            anomaly_score = min(distance / (iqr * 2), 1.0)
        else:
            anomaly_score = 0.0

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            severity=self._determine_severity(anomaly_score) if is_anomaly else None,
            method="iqr",
            operation_name=operation_name,
            metric_name=metric_name,
            timestamp=timestamp,
            details={
                "value": value,
                "q1": baseline["q1"],
                "q3": baseline["q3"],
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "k": self.k
            }
        )


class ErrorRateDetector(AnomalyDetector):
    """
    Error rate anomaly detection.

    Detects unusual spikes in error rates for operations.
    """

    def __init__(self, threshold: float = 0.1, window_size: int = 100):
        """
        Initialize error rate detector.

        Args:
            threshold: Error rate threshold (0.1 = 10%)
            window_size: Number of recent events to consider
        """
        self.threshold = threshold
        self.window_size = window_size
        self.baseline_error_rates: Dict[str, float] = {}

    def fit(self, events: List[MetricEvent]) -> None:
        """Calculate baseline error rates from historical data."""
        error_counts: Dict[str, int] = {}
        total_counts: Dict[str, int] = {}

        for event in events:
            op_name = event.operation_name

            if op_name not in error_counts:
                error_counts[op_name] = 0
                total_counts[op_name] = 0

            total_counts[op_name] += 1
            if event.error:
                error_counts[op_name] += 1

        # Calculate baseline error rates
        for op_name in error_counts:
            if total_counts[op_name] > 0:
                self.baseline_error_rates[op_name] = error_counts[op_name] / total_counts[op_name]

        logger.info(f"Error rate detector fitted on {len(events)} events")

    def detect(self, events: List[MetricEvent]) -> List[AnomalyResult]:
        """Detect anomalous error rates."""
        results = []

        # Group by operation
        events_by_operation: Dict[str, List[MetricEvent]] = {}
        for event in events:
            op_name = event.operation_name
            if op_name not in events_by_operation:
                events_by_operation[op_name] = []
            events_by_operation[op_name].append(event)

        # Check error rates for each operation
        for op_name, op_events in events_by_operation.items():
            # Take last window_size events
            recent_events = op_events[-self.window_size:]

            error_count = sum(1 for e in recent_events if e.error)
            error_rate = error_count / len(recent_events)

            baseline_rate = self.baseline_error_rates.get(op_name, 0.0)

            # Anomaly if error rate significantly exceeds baseline
            is_anomaly = error_rate > max(baseline_rate * 2, self.threshold)

            if is_anomaly:
                anomaly_score = min(error_rate / self.threshold, 1.0)

                results.append(AnomalyResult(
                    is_anomaly=True,
                    anomaly_score=anomaly_score,
                    severity=self._determine_severity(anomaly_score),
                    method="error_rate",
                    operation_name=op_name,
                    metric_name="error_rate",
                    timestamp=datetime.utcnow(),
                    details={
                        "current_error_rate": error_rate,
                        "baseline_error_rate": baseline_rate,
                        "threshold": self.threshold,
                        "window_size": self.window_size,
                        "error_count": error_count,
                        "total_count": len(recent_events)
                    }
                ))

        return results
