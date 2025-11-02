"""
Base classes for measurement strategies.

Defines the strategy pattern for pluggable measurement systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Type
from enum import Enum

from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class MeasurementType(Enum):
    """Type of measurement performed."""
    CAPACITY = "capacity"  # Character/word/byte counts
    TOKEN = "token"  # Token counts using tokenizers
    HYBRID = "hybrid"  # Both capacity and tokens
    CUSTOM = "custom"  # User-defined measurements


class MeasurementReliability(Enum):
    """Reliability level of measurement."""
    EXACT = "exact"  # 100% accurate measurement
    ESTIMATED = "estimated"  # Estimation with confidence score
    FALLBACK = "fallback"  # Fallback method used
    UNAVAILABLE = "unavailable"  # Measurement not available


@dataclass
class MeasurementMetadata:
    """
    Metadata about how measurement was performed.

    Provides observability into the measurement process.
    """
    strategy_name: str
    measurement_type: MeasurementType
    reliability: MeasurementReliability

    # Timing information
    measurement_time_ms: Optional[float] = None

    # Tokenizer information (for token measurements)
    tokenizer_name: Optional[str] = None
    tokenizer_version: Optional[str] = None

    # Confidence and quality
    confidence_score: float = 1.0  # 0.0 to 1.0

    # Fallback information
    fallback_reason: Optional[str] = None
    primary_strategy_failed: bool = False

    # Additional context
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasurementResult:
    """
    Unified measurement result containing all metric types.

    Supports capacity metrics, token metrics, or both.
    """
    # Capacity metrics (always available)
    char_count: Optional[int] = None
    word_count: Optional[int] = None
    byte_size: Optional[int] = None
    line_count: Optional[int] = None

    # Token metrics (best-effort)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Metadata
    metadata: MeasurementMetadata = None

    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def has_capacity_metrics(self) -> bool:
        """Check if capacity metrics are available."""
        return self.char_count is not None

    def has_token_metrics(self) -> bool:
        """Check if token metrics are available."""
        return self.total_tokens is not None

    def is_complete(self) -> bool:
        """Check if measurement is complete (has at least one metric type)."""
        return self.has_capacity_metrics() or self.has_token_metrics()


class MeasurementStrategy(ABC):
    """
    Base class for measurement strategies.

    All measurement strategies must implement this interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy with configuration.

        Args:
            config: Strategy-specific configuration
        """
        self.config = config or {}
        self.strategy_name = self.__class__.__name__

    @abstractmethod
    async def measure(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MeasurementResult:
        """
        Measure the given text.

        Args:
            text: Text to measure
            context: Optional context (model name, provider, etc.)

        Returns:
            MeasurementResult with metrics and metadata
        """
        pass

    @abstractmethod
    def supports_model(self, model: Optional[str]) -> bool:
        """
        Check if this strategy supports the given model.

        Args:
            model: Model name (e.g., "gpt-4", "claude-3-opus")

        Returns:
            True if strategy can handle this model
        """
        pass

    def get_priority(self) -> int:
        """
        Get strategy priority for auto-selection.

        Higher priority strategies are preferred.
        Returns:
            Priority value (0-100)
        """
        return 50  # Default medium priority

    def estimate_cost_impact(self) -> float:
        """
        Estimate computational cost of this measurement.

        Returns:
            Relative cost (0.0 = free, 1.0 = expensive)
        """
        return 0.1  # Default low cost


class StrategyRegistry:
    """
    Registry for measurement strategies.

    Manages registration and retrieval of strategies.
    """

    _strategies: Dict[str, Type[MeasurementStrategy]] = {}
    _instances: Dict[str, MeasurementStrategy] = {}

    @classmethod
    def register(
        cls,
        name: str,
        strategy_class: Type[MeasurementStrategy]
    ) -> None:
        """
        Register a measurement strategy.

        Args:
            name: Strategy name
            strategy_class: Strategy class
        """
        cls._strategies[name] = strategy_class
        logger.debug(f"Registered measurement strategy: {name}")

    @classmethod
    def get_strategy(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[MeasurementStrategy]:
        """
        Get a strategy instance by name.

        Args:
            name: Strategy name
            config: Strategy configuration

        Returns:
            Strategy instance or None
        """
        # Return cached instance if no config change
        if name in cls._instances and config is None:
            return cls._instances[name]

        # Create new instance
        strategy_class = cls._strategies.get(name)
        if strategy_class:
            instance = strategy_class(config)
            if config is None:
                cls._instances[name] = instance
            return instance

        logger.warning(f"Strategy not found: {name}")
        return None

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all registered strategies."""
        return list(cls._strategies.keys())

    @classmethod
    def get_all_strategies(
        cls,
        config: Optional[Dict[str, Any]] = None
    ) -> List[MeasurementStrategy]:
        """Get all registered strategy instances."""
        return [
            cls.get_strategy(name, config)
            for name in cls._strategies.keys()
        ]
