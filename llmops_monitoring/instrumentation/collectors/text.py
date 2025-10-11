"""
Text metric collector.

Measures various aspects of text content with flexible configuration.
"""

from typing import Any, Dict, List, Optional

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

    Users can configure which metrics to collect.
    """

    def __init__(
        self,
        measure: Optional[List[str]] = None,
        text_extractor: Optional[callable] = None
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
        """
        self.measure = measure or ["char_count", "word_count", "byte_size", "line_count"]
        self.text_extractor = text_extractor or self._default_extract_text

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
