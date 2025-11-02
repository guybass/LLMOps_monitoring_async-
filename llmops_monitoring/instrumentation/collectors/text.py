"""
Text metric collector.

Measures various aspects of text content with flexible configuration.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

from llmops_monitoring.instrumentation.base import MetricCollector
from llmops_monitoring.schema.events import TextMetrics


class TextCollector(MetricCollector):
    """
    Collector for text-based metrics.

    Supports flexible measurement options:
    - char_count: Total character count (including spaces)
    - word_count: Total word count
    - byte_size: Size in bytes (UTF-8 encoding)
    - line_count: Number of lines
    - input_tokens: Token count for input (when using measurement strategies)
    - output_tokens: Token count for output (when using measurement strategies)
    - total_tokens: Total token count (when using measurement strategies)

    Users can configure which metrics to collect.
    """

    def __init__(
        self,
        measure: Optional[List[str]] = None,
        text_extractor: Optional[callable] = None,
        measurement: Optional[Union[str, Dict[str, Any]]] = None
    ):
        """
        Initialize text collector.

        Args:
            measure: List of metrics to collect. Options:
                     ['char_count', 'word_count', 'byte_size', 'line_count']
                     If None, collects all metrics.
            text_extractor: Custom function to extract text from result.
                           Signature: (result, args, kwargs) -> str
                           If None, uses default extraction logic.
            measurement: Measurement strategy configuration. Options:
                        - None: Use legacy capacity-based measurement
                        - "auto": Auto-select strategy based on model
                        - "capacity": Use capacity-based measurement
                        - "token": Use token-based measurement
                        - "hybrid": Use both capacity and token measurements
                        - Dict: Full configuration
        """
        self.measure = measure or ["char_count", "word_count", "byte_size", "line_count"]
        self.text_extractor = text_extractor or self._default_extract_text
        self.measurement = measurement
        self._resolver = None

    def collect(
        self,
        result: Any,
        args: tuple,
        kwargs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract text metrics from result."""
        text = self.text_extractor(result, args, kwargs)

        if text is None:
            return {}

        # Check if we should use new measurement strategies
        measurement_config = self.measurement
        if context and "measurement" in context:
            # Override with context measurement if available
            measurement_config = context["measurement"]

        if measurement_config:
            # Use new measurement strategy system
            return asyncio.run(self._collect_with_strategy(text, context or {}))
        else:
            # Legacy mode: simple capacity-based measurement
            return self._collect_legacy(text)

    def _collect_legacy(self, text: str) -> Dict[str, Any]:
        """Legacy collection method for backward compatibility."""
        metrics = {}

        if "char_count" in self.measure:
            metrics["char_count"] = len(text)

        if "word_count" in self.measure:
            metrics["word_count"] = len(text.split())

        if "byte_size" in self.measure:
            metrics["byte_size"] = len(text.encode('utf-8'))

        if "line_count" in self.measure:
            metrics["line_count"] = text.count('\n') + 1

        return {"text_metrics": TextMetrics(**metrics)}

    async def _collect_with_strategy(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect metrics using measurement strategies."""
        try:
            # Import here to avoid circular dependencies
            from llmops_monitoring.measurement import MeasurementStrategyResolver

            # Get or create resolver
            if not self._resolver:
                measurement_config = context.get("measurement", self.measurement)

                # Parse measurement config
                if isinstance(measurement_config, str):
                    mode = measurement_config
                    config = {}
                elif isinstance(measurement_config, dict):
                    mode = measurement_config.get("mode", "auto")
                    config = measurement_config
                else:
                    mode = "auto"
                    config = {}

                self._resolver = MeasurementStrategyResolver(mode=mode, config=config)

            # Prepare measurement context
            measurement_context = {
                "model": context.get("model"),
                "custom_attributes": context.get("custom_attributes", {}),
                "function_args": context.get("function_args", {})
            }

            # Measure text
            result = await self._resolver.measure(text, measurement_context)

            # Convert MeasurementResult to TextMetrics
            metrics = {}

            # Capacity metrics
            if result.char_count is not None:
                metrics["char_count"] = result.char_count
            if result.word_count is not None:
                metrics["word_count"] = result.word_count
            if result.byte_size is not None:
                metrics["byte_size"] = result.byte_size
            if result.line_count is not None:
                metrics["line_count"] = result.line_count

            # Token metrics
            if result.input_tokens is not None:
                metrics["input_tokens"] = result.input_tokens
            if result.output_tokens is not None:
                metrics["output_tokens"] = result.output_tokens
            if result.total_tokens is not None:
                metrics["total_tokens"] = result.total_tokens

            return {"text_metrics": TextMetrics(**metrics)}

        except Exception as e:
            # Fall back to legacy mode if strategy fails
            import logging
            logging.warning(f"Measurement strategy failed: {e}, falling back to legacy mode")
            return self._collect_legacy(text)

    @property
    def metric_type(self) -> str:
        return "text"

    def should_collect(self, result: Any, args: tuple, kwargs: Dict[str, Any]) -> bool:
        """Only collect if we can extract text."""
        text = self.text_extractor(result, args, kwargs)
        return text is not None and len(text) > 0

    @staticmethod
    def _default_extract_text(result: Any, args: tuple, kwargs: Dict[str, Any]) -> Optional[str]:
        """
        Default text extraction logic.

        Tries common patterns:
        1. result is a string
        2. result.text (common in LLM responses)
        3. result["text"] or result.get("text")
        4. result.content (some APIs)
        5. First string argument (prompt)
        """
        # Direct string
        if isinstance(result, str):
            return result

        # Common attribute patterns
        if hasattr(result, 'text') and isinstance(result.text, str):
            return result.text

        if hasattr(result, 'content') and isinstance(result.content, str):
            return result.content

        # Dict-like access
        if isinstance(result, dict):
            if "text" in result:
                return result["text"]
            if "content" in result:
                return result["content"]
            if "output" in result:
                return result["output"]

        # Try to extract from nested structures (e.g., OpenAI response)
        if hasattr(result, 'choices') and len(result.choices) > 0:
            choice = result.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content
            if hasattr(choice, 'text'):
                return choice.text

        return None


class MultiTextCollector(TextCollector):
    """
    Collector for operations that process multiple text inputs/outputs.

    Example: Batch processing, multi-turn conversations, etc.
    """

    def __init__(
        self,
        measure: Optional[List[str]] = None,
        aggregate: bool = True
    ):
        """
        Initialize multi-text collector.

        Args:
            measure: List of metrics to collect
            aggregate: If True, sum all texts. If False, collect per-item metrics.
        """
        super().__init__(measure=measure)
        self.aggregate = aggregate

    def collect(
        self,
        result: Any,
        args: tuple,
        kwargs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract metrics from multiple texts."""
        texts = self._extract_texts(result, args, kwargs)

        if not texts:
            return {}

        if self.aggregate:
            # Combine all texts and measure once
            combined_text = "\n".join(texts)
            return super().collect(combined_text, args, kwargs, context)
        else:
            # Measure each text separately (store in custom_metrics)
            metrics_list = []
            for text in texts:
                text_result = super().collect(text, args, kwargs, context)
                if text_result:
                    metrics_list.append(text_result.get("text_metrics"))

            # Aggregate the metrics
            if metrics_list:
                aggregated = self._aggregate_metrics(metrics_list)
                return {"text_metrics": TextMetrics(**aggregated)}

            return {}

    def _extract_texts(self, result: Any, args: tuple, kwargs: Dict[str, Any]) -> List[str]:
        """Extract list of texts from result."""
        texts = []

        # Result is a list of strings
        if isinstance(result, list):
            for item in result:
                if isinstance(item, str):
                    texts.append(item)
                elif hasattr(item, 'text'):
                    texts.append(item.text)

        # Result has a 'texts' or 'outputs' attribute
        elif hasattr(result, 'texts'):
            texts = result.texts
        elif hasattr(result, 'outputs'):
            texts = result.outputs

        return [t for t in texts if t]  # Filter out None/empty

    def _aggregate_metrics(self, metrics_list: List[TextMetrics]) -> Dict[str, Any]:
        """Aggregate multiple text metrics."""
        aggregated = {
            "char_count": sum(m.char_count or 0 for m in metrics_list),
            "word_count": sum(m.word_count or 0 for m in metrics_list),
            "byte_size": sum(m.byte_size or 0 for m in metrics_list),
            "line_count": sum(m.line_count or 0 for m in metrics_list),
        }
        return aggregated
