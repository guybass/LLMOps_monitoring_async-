"""
Capacity-based measurement strategy.

Measures text by characters, words, bytes, and lines.
Language-agnostic and always reliable.
"""

import time
from typing import Dict, Any, Optional

from llmops_monitoring.measurement.base import (
    MeasurementStrategy,
    MeasurementResult,
    MeasurementMetadata,
    MeasurementType,
    MeasurementReliability,
    StrategyRegistry
)


class CapacityStrategy(MeasurementStrategy):
    """
    Capacity-based measurement strategy.

    Measures:
    - Character count
    - Word count
    - Byte size
    - Line count

    Benefits:
    - 100% reliable
    - Language-agnostic
    - No external dependencies
    - Negligible performance impact (<1ms)

    Best for:
    - Performance monitoring
    - Cross-language comparisons
    - Debugging
    - When tokenizer unavailable
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.strategy_name = "CapacityStrategy"

    async def measure(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MeasurementResult:
        """
        Measure text capacity metrics.

        Args:
            text: Text to measure
            context: Optional context (ignored for capacity)

        Returns:
            MeasurementResult with capacity metrics
        """
        start_time = time.time()

        # Count characters
        char_count = len(text)

        # Count words (split on whitespace)
        word_count = len(text.split())

        # Count bytes (UTF-8 encoding)
        byte_size = len(text.encode('utf-8'))

        # Count lines
        line_count = text.count('\n') + 1 if text else 0

        measurement_time_ms = (time.time() - start_time) * 1000

        # Create metadata
        metadata = MeasurementMetadata(
            strategy_name=self.strategy_name,
            measurement_type=MeasurementType.CAPACITY,
            reliability=MeasurementReliability.EXACT,
            measurement_time_ms=measurement_time_ms,
            confidence_score=1.0,
            extra={
                "encoding": "utf-8",
                "word_split_method": "whitespace"
            }
        )

        return MeasurementResult(
            char_count=char_count,
            word_count=word_count,
            byte_size=byte_size,
            line_count=line_count,
            metadata=metadata
        )

    def supports_model(self, model: Optional[str]) -> bool:
        """
        Capacity strategy supports all models.

        Returns:
            Always True (universal support)
        """
        return True

    def get_priority(self) -> int:
        """
        Get strategy priority.

        Returns:
            30 (low-medium priority, used as fallback)
        """
        return 30  # Lower priority than token strategies

    def estimate_cost_impact(self) -> float:
        """
        Estimate computational cost.

        Returns:
            0.01 (negligible cost, <1ms)
        """
        return 0.01


# Auto-register strategy
StrategyRegistry.register("capacity", CapacityStrategy)
