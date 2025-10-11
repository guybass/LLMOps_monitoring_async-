"""
Base abstractions for metric collection.

Extension Point: Implement MetricCollector to add new metric types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type


class MetricCollector(ABC):
    """
    Abstract base class for metric collectors.

    Collectors extract specific metrics from operation results.
    Users can implement custom collectors for new metric types.

    Example:
        class TokenCollector(MetricCollector):
            def collect(self, result, args, kwargs):
                return {"token_count": result.usage.total_tokens}
    """

    @abstractmethod
    def collect(
        self,
        result: Any,
        args: tuple,
        kwargs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract metrics from operation result.

        Args:
            result: The return value of the monitored function
            args: Positional arguments passed to the function
            kwargs: Keyword arguments passed to the function
            context: Additional context (e.g., timing, span info)

        Returns:
            Dictionary of metrics extracted from the result
        """
        pass

    @property
    @abstractmethod
    def metric_type(self) -> str:
        """
        Return the metric type identifier.

        This determines where metrics are stored in the event model.
        Common values: "text", "image", "token", "cost"
        """
        pass

    def should_collect(
        self,
        result: Any,
        args: tuple,
        kwargs: Dict[str, Any]
    ) -> bool:
        """
        Determine if this collector should run for this operation.

        Override to add conditional collection logic.

        Returns:
            True if collector should run, False otherwise
        """
        return True


class CollectorRegistry:
    """
    Registry for managing metric collectors.

    Extension Point: Register custom collectors here.
    """

    _collectors: Dict[str, Type[MetricCollector]] = {}
    _instances: Dict[str, MetricCollector] = {}

    @classmethod
    def register(cls, name: str, collector_class: Type[MetricCollector]) -> None:
        """
        Register a new collector type.

        Args:
            name: Unique identifier for the collector
            collector_class: The collector class (not instance)

        Example:
            CollectorRegistry.register("custom", MyCustomCollector)
        """
        cls._collectors[name] = collector_class

    @classmethod
    def get(cls, name: str) -> Optional[MetricCollector]:
        """Get a collector instance by name."""
        if name in cls._instances:
            return cls._instances[name]

        if name in cls._collectors:
            cls._instances[name] = cls._collectors[name]()
            return cls._instances[name]

        return None

    @classmethod
    def get_all(cls, names: Optional[List[str]] = None) -> List[MetricCollector]:
        """
        Get multiple collector instances.

        Args:
            names: List of collector names. If None, returns all registered collectors.

        Returns:
            List of collector instances
        """
        if names is None:
            names = list(cls._collectors.keys())

        return [collector for name in names if (collector := cls.get(name)) is not None]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered collector names."""
        return list(cls._collectors.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered collectors (mainly for testing)."""
        cls._collectors.clear()
        cls._instances.clear()
