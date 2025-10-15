"""
Cost calculation collector.

Calculates estimated costs for LLM operations based on model pricing
and measured metrics (text, images).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from llmops_monitoring.instrumentation.base import MetricCollector
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class CostCollector(MetricCollector):
    """
    Collector for cost estimation based on model pricing.

    Calculates costs using:
    - Text metrics (converted to tokens)
    - Image metrics
    - Model-specific pricing from pricing database

    Pricing data is loaded from llmops_monitoring/data/pricing.json

    Example:
        @monitor_llm(
            collectors=["cost"],
            custom_attributes={"model": "gpt-4o-mini"}
        )
        async def my_llm_call():
            return {"text": "response..."}
    """

    def __init__(
        self,
        pricing_file: Optional[str] = None,
        custom_pricing: Optional[Dict[str, Dict[str, float]]] = None,
        model_attribute: str = "model"
    ):
        """
        Initialize cost collector.

        Args:
            pricing_file: Path to custom pricing JSON file. If None, uses built-in pricing.
            custom_pricing: Custom pricing overrides. Format:
                {
                    "model-name": {
                        "input_cost_per_1k_tokens": 0.001,
                        "output_cost_per_1k_tokens": 0.002
                    }
                }
            model_attribute: Name of the custom attribute that contains the model name.
                            Default: "model"
        """
        self.pricing_data = self._load_pricing_data(pricing_file)
        self.custom_pricing = custom_pricing or {}
        self.model_attribute = model_attribute

        # Cache defaults for faster access
        self.defaults = self.pricing_data.get("defaults", {})
        self.chars_to_tokens_ratio = self.defaults.get("chars_to_tokens_ratio", 4)
        self.fallback_cost_per_1k_chars = self.defaults.get("fallback_cost_per_1k_chars", 0.001)
        self.image_cost_per_image = self.defaults.get("image_cost_per_image", 0.00085)

        logger.debug(f"CostCollector initialized with {len(self.pricing_data.get('models', {}))} models")

    def collect(
        self,
        result: Any,
        args: tuple,
        kwargs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate cost based on metrics and model pricing.

        Extracts model name from custom_attributes and calculates cost
        based on text and image metrics from the context.
        """
        if context is None:
            return {}

        # Extract model name from context's custom_attributes
        custom_attrs = context.get("custom_attributes", {})
        model_name = custom_attrs.get(self.model_attribute)

        if not model_name:
            logger.debug("No model name found in custom attributes, skipping cost calculation")
            return {}

        # Get pricing for the model
        pricing = self._get_model_pricing(model_name)

        # Calculate text cost
        text_cost = self._calculate_text_cost(context, pricing)

        # Calculate image cost
        image_cost = self._calculate_image_cost(context, pricing)

        # Total cost
        total_cost = text_cost + image_cost

        if total_cost == 0:
            return {}

        # Return cost in custom_attributes
        cost_data = {
            "estimated_cost_usd": round(total_cost, 6),
            "model": model_name,
            "cost_breakdown": {
                "text_cost_usd": round(text_cost, 6),
                "image_cost_usd": round(image_cost, 6)
            }
        }

        logger.debug(f"Cost calculated for {model_name}: ${total_cost:.6f}")

        return {"custom_attributes": cost_data}

    @property
    def metric_type(self) -> str:
        return "cost"

    def should_collect(self, result: Any, args: tuple, kwargs: Dict[str, Any]) -> bool:
        """Only collect if model name is present in custom attributes."""
        custom_attrs = kwargs.get("custom_attributes", {})
        return self.model_attribute in custom_attrs

    def _load_pricing_data(self, pricing_file: Optional[str] = None) -> Dict[str, Any]:
        """Load pricing data from JSON file."""
        if pricing_file is None:
            # Use built-in pricing file
            package_dir = Path(__file__).parent.parent.parent
            pricing_file = package_dir / "data" / "pricing.json"
        else:
            pricing_file = Path(pricing_file)

        if not pricing_file.exists():
            logger.warning(f"Pricing file not found: {pricing_file}, using fallback costs")
            return {"models": {}, "defaults": {}}

        try:
            with open(pricing_file, 'r') as f:
                data = json.load(f)
            logger.debug(f"Loaded pricing data from {pricing_file}")
            return data
        except Exception as e:
            logger.error(f"Error loading pricing data: {e}")
            return {"models": {}, "defaults": {}}

    def _get_model_pricing(self, model_name: str) -> Dict[str, float]:
        """
        Get pricing for a specific model.

        Returns pricing dict with input_cost_per_1k_tokens and output_cost_per_1k_tokens.
        Falls back to defaults if model not found.
        """
        # Check custom pricing first
        if model_name in self.custom_pricing:
            return self.custom_pricing[model_name]

        # Check pricing database
        models = self.pricing_data.get("models", {})
        if model_name in models:
            return models[model_name]

        # Check for partial matches (e.g., "gpt-4o-2024-05-13" matches "gpt-4o")
        for db_model, pricing in models.items():
            if model_name.startswith(db_model):
                logger.debug(f"Using pricing for {db_model} (matched from {model_name})")
                return pricing

        # Fallback: use default costs
        logger.warning(f"Model '{model_name}' not found in pricing database, using fallback costs")
        return {
            "input_cost_per_1k_tokens": self.fallback_cost_per_1k_chars,
            "output_cost_per_1k_tokens": self.fallback_cost_per_1k_chars
        }

    def _calculate_text_cost(self, context: Dict[str, Any], pricing: Dict[str, float]) -> float:
        """
        Calculate cost for text metrics.

        Uses char_count from text_metrics and converts to tokens using the
        chars_to_tokens_ratio (default: 4 chars = 1 token).

        Note: This assumes the entire text is output. For more accurate cost
        calculation, users should provide input_tokens and output_tokens
        separately in custom_attributes.
        """
        # Check if user provided token counts directly
        custom_attrs = context.get("custom_attributes", {})
        input_tokens = custom_attrs.get("input_tokens")
        output_tokens = custom_attrs.get("output_tokens")

        if input_tokens is not None and output_tokens is not None:
            # Use exact token counts if provided
            input_cost = (input_tokens / 1000) * pricing.get("input_cost_per_1k_tokens", 0)
            output_cost = (output_tokens / 1000) * pricing.get("output_cost_per_1k_tokens", 0)
            return input_cost + output_cost

        # Otherwise, estimate from char_count
        text_metrics = context.get("text_metrics")
        if not text_metrics:
            return 0.0

        char_count = getattr(text_metrics, "char_count", None)
        if char_count is None or char_count == 0:
            return 0.0

        # Convert chars to tokens
        estimated_tokens = char_count / self.chars_to_tokens_ratio

        # For simplicity, assume output tokens (typically more expensive)
        # Users can provide exact token counts for more accuracy
        cost_per_1k = pricing.get("output_cost_per_1k_tokens", self.fallback_cost_per_1k_chars)
        return (estimated_tokens / 1000) * cost_per_1k

    def _calculate_image_cost(self, context: Dict[str, Any], pricing: Dict[str, float]) -> float:
        """
        Calculate cost for image metrics.

        Uses image_count from context. Pricing can specify custom image costs
        per model, otherwise uses default.
        """
        image_metrics = context.get("image_metrics")
        if not image_metrics:
            return 0.0

        image_count = getattr(image_metrics, "count", None)
        if image_count is None or image_count == 0:
            return 0.0

        # Check if model has custom image pricing
        cost_per_image = pricing.get("image_cost_per_image", self.image_cost_per_image)

        return image_count * cost_per_image
