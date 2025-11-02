"""
Measurement Strategy Resolver.

Automatically selects and executes the best measurement strategy.
"""

from typing import Dict, Any, Optional, List
from enum import Enum

from llmops_monitoring.measurement.base import (
    MeasurementStrategy,
    MeasurementResult,
    StrategyRegistry
)
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class StrategyMode(Enum):
    """Strategy selection mode."""
    AUTO = "auto"  # Auto-select based on context
    CAPACITY = "capacity"  # Use capacity strategy
    TOKEN = "token"  # Use token strategy
    HYBRID = "hybrid"  # Use hybrid strategy


class MeasurementStrategyResolver:
    """
    Resolves and executes measurement strategies.

    Handles:
    - Auto-detection of model/provider from context
    - Strategy selection based on mode and configuration
    - Fallback chains when strategies fail
    - Efficient strategy execution

    Example:
        resolver = MeasurementStrategyResolver(mode="auto")
        result = await resolver.measure("Hello world", context={"model": "gpt-4"})
    """

    def __init__(
        self,
        mode: str = "auto",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize resolver.

        Args:
            mode: Strategy mode ("auto", "capacity", "token", "hybrid")
            config: Configuration for strategies
        """
        self.mode = StrategyMode(mode.lower())
        self.config = config or {}

        # Strategy preferences
        self.prefer_tokens = self.config.get("prefer_tokens", True)
        self.fallback_enabled = self.config.get("fallback_enabled", True)
        self.parallel_hybrid = self.config.get("parallel_hybrid", True)

        # Initialize strategies (lazy loading)
        self._strategies: Dict[str, MeasurementStrategy] = {}

        logger.debug(f"MeasurementStrategyResolver initialized with mode={mode}")

    def _get_strategy(self, name: str) -> Optional[MeasurementStrategy]:
        """
        Get or create strategy instance.

        Args:
            name: Strategy name

        Returns:
            Strategy instance or None
        """
        if name not in self._strategies:
            strategy = StrategyRegistry.get_strategy(name, self.config)
            if strategy:
                self._strategies[name] = strategy

        return self._strategies.get(name)

    def _extract_model_from_context(
        self,
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Extract model name from context.

        Args:
            context: Measurement context

        Returns:
            Model name or None
        """
        if not context:
            return None

        # Direct model field
        model = context.get("model")
        if model:
            return model

        # Check custom_attributes
        custom_attrs = context.get("custom_attributes", {})
        if isinstance(custom_attrs, dict):
            model = custom_attrs.get("model")
            if model:
                return model

        # Check function_args (if available)
        func_args = context.get("function_args", {})
        if isinstance(func_args, dict):
            model = func_args.get("model")
            if model:
                return model

        return None

    def _detect_provider_from_model(self, model: Optional[str]) -> Optional[str]:
        """
        Detect provider from model name.

        Args:
            model: Model name

        Returns:
            Provider name or None
        """
        if not model:
            return None

        model_lower = model.lower()

        if "gpt" in model_lower or "davinci" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower or "palm" in model_lower:
            return "google"
        elif "llama" in model_lower:
            return "meta"
        elif "mistral" in model_lower:
            return "mistral"

        return None

    def _select_strategy_auto(
        self,
        context: Optional[Dict[str, Any]]
    ) -> MeasurementStrategy:
        """
        Auto-select best strategy based on context.

        Priority:
        1. HybridStrategy if both capacity and tokens needed
        2. TokenStrategy if model recognized and tokenizer available
        3. CapacityStrategy as fallback

        Args:
            context: Measurement context

        Returns:
            Selected strategy
        """
        # Extract model
        model = self._extract_model_from_context(context)

        # Check if user explicitly wants hybrid
        if context and context.get("use_hybrid"):
            strategy = self._get_strategy("hybrid")
            if strategy:
                logger.debug("Auto-selected HybridStrategy (explicit request)")
                return strategy

        # Check if we should use tokens
        if self.prefer_tokens and model:
            token_strategy = self._get_strategy("token")
            if token_strategy and token_strategy.supports_model(model):
                logger.debug(f"Auto-selected TokenStrategy for model={model}")
                return token_strategy

        # Fallback to capacity
        capacity_strategy = self._get_strategy("capacity")
        if capacity_strategy:
            logger.debug("Auto-selected CapacityStrategy (fallback)")
            return capacity_strategy

        # Ultimate fallback - create capacity strategy
        logger.warning("No registered strategies found, creating default CapacityStrategy")
        from llmops_monitoring.measurement.strategies.capacity import CapacityStrategy
        return CapacityStrategy(self.config)

    def _select_strategy(
        self,
        context: Optional[Dict[str, Any]]
    ) -> MeasurementStrategy:
        """
        Select strategy based on mode.

        Args:
            context: Measurement context

        Returns:
            Selected strategy
        """
        if self.mode == StrategyMode.AUTO:
            return self._select_strategy_auto(context)

        elif self.mode == StrategyMode.CAPACITY:
            strategy = self._get_strategy("capacity")
            if not strategy:
                from llmops_monitoring.measurement.strategies.capacity import CapacityStrategy
                strategy = CapacityStrategy(self.config)
            return strategy

        elif self.mode == StrategyMode.TOKEN:
            strategy = self._get_strategy("token")
            if not strategy:
                from llmops_monitoring.measurement.strategies.token import TokenStrategy
                strategy = TokenStrategy(self.config)
            return strategy

        elif self.mode == StrategyMode.HYBRID:
            strategy = self._get_strategy("hybrid")
            if not strategy:
                from llmops_monitoring.measurement.strategies.hybrid import HybridStrategy
                strategy = HybridStrategy(self.config)
            return strategy

        # Fallback
        logger.warning(f"Unknown mode {self.mode}, using capacity")
        from llmops_monitoring.measurement.strategies.capacity import CapacityStrategy
        return CapacityStrategy(self.config)

    async def measure(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MeasurementResult:
        """
        Measure text using selected strategy.

        Args:
            text: Text to measure
            context: Measurement context

        Returns:
            MeasurementResult
        """
        # Select strategy
        strategy = self._select_strategy(context)

        # Execute measurement
        try:
            result = await strategy.measure(text, context)
            return result

        except Exception as e:
            logger.error(f"Strategy {strategy.strategy_name} failed: {e}")

            # Fallback to capacity if enabled
            if self.fallback_enabled and strategy.strategy_name != "CapacityStrategy":
                logger.info("Falling back to CapacityStrategy")
                fallback_strategy = self._get_strategy("capacity")
                if not fallback_strategy:
                    from llmops_monitoring.measurement.strategies.capacity import CapacityStrategy
                    fallback_strategy = CapacityStrategy(self.config)

                try:
                    result = await fallback_strategy.measure(text, context)
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback strategy also failed: {fallback_error}")

            # Re-raise if no fallback or fallback failed
            raise

    async def measure_multiple(
        self,
        texts: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[MeasurementResult]:
        """
        Measure multiple texts using selected strategy.

        Args:
            texts: List of texts to measure
            context: Measurement context

        Returns:
            List of MeasurementResults
        """
        results = []
        for text in texts:
            result = await self.measure(text, context)
            results.append(result)
        return results

    def get_available_strategies(self) -> List[str]:
        """
        Get list of available strategies.

        Returns:
            List of strategy names
        """
        return StrategyRegistry.list_strategies()

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about current resolver configuration.

        Returns:
            Configuration dictionary
        """
        return {
            "mode": self.mode.value,
            "prefer_tokens": self.prefer_tokens,
            "fallback_enabled": self.fallback_enabled,
            "parallel_hybrid": self.parallel_hybrid,
            "available_strategies": self.get_available_strategies(),
            "loaded_strategies": list(self._strategies.keys())
        }


# Convenience function for simple usage
async def measure_text(
    text: str,
    mode: str = "auto",
    context: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> MeasurementResult:
    """
    Convenience function to measure text.

    Args:
        text: Text to measure
        mode: Strategy mode ("auto", "capacity", "token", "hybrid")
        context: Measurement context
        config: Strategy configuration

    Returns:
        MeasurementResult

    Example:
        result = await measure_text("Hello world", mode="token", context={"model": "gpt-4"})
        print(f"Tokens: {result.total_tokens}")
    """
    resolver = MeasurementStrategyResolver(mode=mode, config=config)
    return await resolver.measure(text, context)
