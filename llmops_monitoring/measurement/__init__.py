"""
Pluggable Measurement Strategy System.

Allows seamless switching between capacity-based and token-based metrics.
"""

from llmops_monitoring.measurement.base import (
    MeasurementStrategy,
    MeasurementResult,
    MeasurementMetadata,
    MeasurementType,
    MeasurementReliability,
    StrategyRegistry
)
from llmops_monitoring.measurement.resolver import (
    MeasurementStrategyResolver,
    StrategyMode,
    measure_text
)

# Auto-register built-in strategies
from llmops_monitoring.measurement.strategies import capacity, token, hybrid


__all__ = [
    # Base classes
    "MeasurementStrategy",
    "MeasurementResult",
    "MeasurementMetadata",
    "MeasurementType",
    "MeasurementReliability",
    "StrategyRegistry",

    # Resolver
    "MeasurementStrategyResolver",
    "StrategyMode",
    "measure_text",
]
