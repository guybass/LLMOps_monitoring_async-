"""
Hybrid measurement strategy.

Combines capacity and token measurements for comprehensive monitoring.
"""

import asyncio
from typing import Dict, Any, Optional

from llmops_monitoring.measurement.base import (
    MeasurementStrategy,
    MeasurementResult,
    MeasurementMetadata,
    MeasurementType,
    MeasurementReliability,
    StrategyRegistry
)
from llmops_monitoring.measurement.strategies.capacity import CapacityStrategy
from llmops_monitoring.measurement.strategies.token import TokenStrategy
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class HybridStrategy(MeasurementStrategy):
    """
    Hybrid measurement strategy.

    Executes both capacity and token measurements in parallel.

    Benefits:
    - Complete visibility (both capacity AND tokens)
    - Cross-validation of measurements
    - Migration safety (compare old vs new metrics)
    - Flexibility (use either metric as needed)

    Use cases:
    - Migration from capacity to tokens
    - Cross-model performance comparison
    - Debugging token counting issues
    - Comprehensive cost + performance analysis

    Performance:
    - Runs both strategies in parallel
    - Total latency ≈ max(capacity_time, token_time)
    - Capacity always succeeds, tokens best-effort
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.strategy_name = "HybridStrategy"

        # Initialize sub-strategies
        self.capacity_strategy = CapacityStrategy(config)
        self.token_strategy = TokenStrategy(config)

        # Configuration
        self.parallel_execution = config.get("parallel_execution", True) if config else True

    async def measure(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MeasurementResult:
        """
        Measure text using both capacity and token strategies.

        Args:
            text: Text to measure
            context: Context with model information

        Returns:
            MeasurementResult with both capacity and token metrics
        """
        if self.parallel_execution:
            # Run both strategies in parallel
            capacity_task = asyncio.create_task(
                self.capacity_strategy.measure(text, context)
            )
            token_task = asyncio.create_task(
                self.token_strategy.measure(text, context)
            )

            # Wait for both (with error handling)
            results = await asyncio.gather(
                capacity_task,
                token_task,
                return_exceptions=True
            )

            capacity_result = results[0] if not isinstance(results[0], Exception) else None
            token_result = results[1] if not isinstance(results[1], Exception) else None

        else:
            # Run sequentially (capacity first, always succeeds)
            try:
                capacity_result = await self.capacity_strategy.measure(text, context)
            except Exception as e:
                logger.error(f"Capacity measurement failed: {e}")
                capacity_result = None

            try:
                token_result = await self.token_strategy.measure(text, context)
            except Exception as e:
                logger.warning(f"Token measurement failed: {e}")
                token_result = None

        # Merge results
        merged_result = self._merge_results(capacity_result, token_result)

        return merged_result

    def _merge_results(
        self,
        capacity_result: Optional[MeasurementResult],
        token_result: Optional[MeasurementResult]
    ) -> MeasurementResult:
        """
        Merge capacity and token measurement results.

        Args:
            capacity_result: Result from capacity strategy
            token_result: Result from token strategy

        Returns:
            Merged MeasurementResult
        """
        # Start with empty result
        merged = MeasurementResult()

        # Add capacity metrics
        if capacity_result:
            merged.char_count = capacity_result.char_count
            merged.word_count = capacity_result.word_count
            merged.byte_size = capacity_result.byte_size
            merged.line_count = capacity_result.line_count

        # Add token metrics
        if token_result:
            merged.input_tokens = token_result.input_tokens
            merged.output_tokens = token_result.output_tokens
            merged.total_tokens = token_result.total_tokens

        # Determine overall reliability
        if capacity_result and token_result:
            # Both succeeded
            if token_result.metadata.reliability == MeasurementReliability.EXACT:
                overall_reliability = MeasurementReliability.EXACT
            else:
                overall_reliability = MeasurementReliability.ESTIMATED
        elif capacity_result:
            # Only capacity succeeded (tokens unavailable)
            overall_reliability = MeasurementReliability.FALLBACK
        else:
            # Neither succeeded (shouldn't happen)
            overall_reliability = MeasurementReliability.UNAVAILABLE

        # Calculate combined measurement time
        measurement_time_ms = 0.0
        if capacity_result and capacity_result.metadata:
            measurement_time_ms = capacity_result.metadata.measurement_time_ms or 0.0
        if token_result and token_result.metadata:
            # If parallel, use max; if sequential, use sum
            token_time = token_result.metadata.measurement_time_ms or 0.0
            if self.parallel_execution:
                measurement_time_ms = max(measurement_time_ms, token_time)
            else:
                measurement_time_ms += token_time

        # Create combined metadata
        merged.metadata = MeasurementMetadata(
            strategy_name=self.strategy_name,
            measurement_type=MeasurementType.HYBRID,
            reliability=overall_reliability,
            measurement_time_ms=measurement_time_ms,
            tokenizer_name=token_result.metadata.tokenizer_name if token_result and token_result.metadata else None,
            confidence_score=self._calculate_confidence(capacity_result, token_result),
            extra={
                "capacity_available": capacity_result is not None,
                "tokens_available": token_result is not None,
                "parallel_execution": self.parallel_execution,
                "capacity_metadata": capacity_result.metadata.__dict__ if capacity_result and capacity_result.metadata else None,
                "token_metadata": token_result.metadata.__dict__ if token_result and token_result.metadata else None
            }
        )

        return merged

    def _calculate_confidence(
        self,
        capacity_result: Optional[MeasurementResult],
        token_result: Optional[MeasurementResult]
    ) -> float:
        """
        Calculate overall confidence score.

        Args:
            capacity_result: Capacity measurement result
            token_result: Token measurement result

        Returns:
            Combined confidence score (0.0 to 1.0)
        """
        scores = []

        if capacity_result and capacity_result.metadata:
            scores.append(capacity_result.metadata.confidence_score)

        if token_result and token_result.metadata:
            scores.append(token_result.metadata.confidence_score)

        if not scores:
            return 0.0

        # Average of available confidence scores
        return sum(scores) / len(scores)

    def supports_model(self, model: Optional[str]) -> bool:
        """
        Check if strategy supports the model.

        Hybrid supports all models (capacity always works).

        Returns:
            Always True
        """
        return True

    def get_priority(self) -> int:
        """
        Get strategy priority.

        Returns:
            80 (highest priority when explicitly requested)
        """
        return 80

    def estimate_cost_impact(self) -> float:
        """
        Estimate computational cost.

        Returns:
            0.31 (sum of both strategies, but parallelized)
        """
        if self.parallel_execution:
            # Parallel: cost is max of both
            return max(
                self.capacity_strategy.estimate_cost_impact(),
                self.token_strategy.estimate_cost_impact()
            )
        else:
            # Sequential: cost is sum of both
            return (
                self.capacity_strategy.estimate_cost_impact() +
                self.token_strategy.estimate_cost_impact()
            )


# Auto-register strategy
StrategyRegistry.register("hybrid", HybridStrategy)
