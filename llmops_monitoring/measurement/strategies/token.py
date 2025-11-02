"""
Token-based measurement strategy.

Measures text using provider-specific tokenizers for accurate cost calculation.
"""

import time
import asyncio
from typing import Dict, Any, Optional
from functools import lru_cache

from llmops_monitoring.measurement.base import (
    MeasurementStrategy,
    MeasurementResult,
    MeasurementMetadata,
    MeasurementType,
    MeasurementReliability,
    StrategyRegistry
)
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


# Model to tokenizer mapping
MODEL_TOKENIZER_MAP = {
    # OpenAI GPT-4 family
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4o": "cl100k_base",
    "gpt-4o-mini": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",

    # OpenAI GPT-3 family
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "davinci": "p50k_base",

    # Anthropic Claude (uses their API)
    "claude-3-opus": "claude",
    "claude-3-sonnet": "claude",
    "claude-3-5-sonnet": "claude",
    "claude-3-haiku": "claude",

    # Google Gemini
    "gemini-1.5-pro": "gemini",
    "gemini-1.5-flash": "gemini",
    "gemini-1.0-pro": "gemini",

    # Meta Llama
    "llama-3-8b": "llama",
    "llama-3-70b": "llama",
    "llama-2-7b": "llama",
    "llama-2-13b": "llama",
    "llama-2-70b": "llama",
}


class TokenStrategy(MeasurementStrategy):
    """
    Token-based measurement strategy.

    Uses provider-specific tokenizers for accurate token counts.

    Benefits:
    - Accurate cost calculation
    - Industry-standard metric
    - Provider-specific token counting

    Limitations:
    - Requires tokenizer libraries
    - Adds 5-20ms latency
    - May fallback to estimation

    Best for:
    - Cost tracking
    - Billing accuracy
    - Provider API alignment
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.strategy_name = "TokenStrategy"

        # Configuration
        self.async_tokenization = config.get("async_tokenization", True) if config else True
        self.tokenization_timeout_ms = config.get("tokenization_timeout_ms", 100) if config else 100
        self.cache_size = config.get("cache_size", 1000) if config else 1000
        self.fallback_to_estimation = config.get("fallback_to_estimation", True) if config else True

        # Check available tokenizers
        self.tiktoken_available = self._check_tiktoken()
        self.sentencepiece_available = self._check_sentencepiece()

    def _check_tiktoken(self) -> bool:
        """Check if tiktoken is available (OpenAI tokenizer)."""
        try:
            import tiktoken
            return True
        except ImportError:
            return False

    def _check_sentencepiece(self) -> bool:
        """Check if sentencepiece is available (Llama tokenizer)."""
        try:
            import sentencepiece
            return True
        except ImportError:
            return False

    def _get_tokenizer_for_model(self, model: str) -> Optional[str]:
        """
        Get tokenizer name for a model.

        Args:
            model: Model name

        Returns:
            Tokenizer name or None
        """
        # Exact match
        if model in MODEL_TOKENIZER_MAP:
            return MODEL_TOKENIZER_MAP[model]

        # Prefix match (e.g., "gpt-4-0613" matches "gpt-4")
        for model_prefix, tokenizer in MODEL_TOKENIZER_MAP.items():
            if model.startswith(model_prefix):
                return tokenizer

        return None

    @lru_cache(maxsize=1000)
    def _count_tokens_tiktoken(self, text: str, encoding_name: str) -> int:
        """
        Count tokens using tiktoken (cached).

        Args:
            text: Text to tokenize
            encoding_name: Tiktoken encoding name

        Returns:
            Token count
        """
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)

    def _count_tokens_estimation(self, text: str) -> int:
        """
        Estimate token count using character ratio.

        Uses industry-standard 4:1 character-to-token ratio.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        char_count = len(text)
        # Average ratio: 4 characters ≈ 1 token
        return max(1, char_count // 4)

    async def measure(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MeasurementResult:
        """
        Measure text using token counting.

        Args:
            text: Text to measure
            context: Context with model information

        Returns:
            MeasurementResult with token metrics
        """
        start_time = time.time()

        # Extract model from context
        model = context.get("model") if context else None
        if not model:
            # Try to infer from custom_attributes
            custom_attrs = context.get("custom_attributes", {}) if context else {}
            model = custom_attrs.get("model")

        # Get tokenizer for model
        tokenizer_name = self._get_tokenizer_for_model(model) if model else None

        # Attempt token counting
        tokens = None
        reliability = MeasurementReliability.UNAVAILABLE
        fallback_reason = None

        if tokenizer_name:
            try:
                if tokenizer_name in ["cl100k_base", "p50k_base"] and self.tiktoken_available:
                    # Use tiktoken for OpenAI models
                    if self.async_tokenization:
                        tokens = await asyncio.wait_for(
                            asyncio.to_thread(self._count_tokens_tiktoken, text, tokenizer_name),
                            timeout=self.tokenization_timeout_ms / 1000
                        )
                    else:
                        tokens = self._count_tokens_tiktoken(text, tokenizer_name)

                    reliability = MeasurementReliability.EXACT

                elif tokenizer_name == "llama" and self.sentencepiece_available:
                    # TODO: Implement sentencepiece tokenization
                    fallback_reason = "sentencepiece not yet implemented"

                elif tokenizer_name in ["claude", "gemini"]:
                    # These require API calls - fall back to estimation
                    fallback_reason = f"{tokenizer_name} requires API call"

                else:
                    fallback_reason = f"tokenizer {tokenizer_name} not available"

            except asyncio.TimeoutError:
                fallback_reason = "tokenization timeout"
                logger.warning(f"Token counting timeout for model {model}")

            except Exception as e:
                fallback_reason = f"tokenization error: {str(e)}"
                logger.error(f"Error counting tokens: {e}")

        else:
            fallback_reason = "model not recognized"

        # Fallback to estimation if needed
        if tokens is None and self.fallback_to_estimation:
            tokens = self._count_tokens_estimation(text)
            reliability = MeasurementReliability.ESTIMATED

        measurement_time_ms = (time.time() - start_time) * 1000

        # Determine confidence score
        confidence_score = {
            MeasurementReliability.EXACT: 1.0,
            MeasurementReliability.ESTIMATED: 0.75,
            MeasurementReliability.UNAVAILABLE: 0.0
        }.get(reliability, 0.5)

        # Create metadata
        metadata = MeasurementMetadata(
            strategy_name=self.strategy_name,
            measurement_type=MeasurementType.TOKEN,
            reliability=reliability,
            measurement_time_ms=measurement_time_ms,
            tokenizer_name=tokenizer_name,
            confidence_score=confidence_score,
            fallback_reason=fallback_reason,
            primary_strategy_failed=(reliability != MeasurementReliability.EXACT),
            extra={
                "model": model,
                "tiktoken_available": self.tiktoken_available,
                "sentencepiece_available": self.sentencepiece_available,
                "estimation_ratio": 4.0 if reliability == MeasurementReliability.ESTIMATED else None
            }
        )

        return MeasurementResult(
            total_tokens=tokens,
            metadata=metadata
        )

    def supports_model(self, model: Optional[str]) -> bool:
        """
        Check if strategy supports the model.

        Returns:
            True if we have a tokenizer for this model
        """
        if not model:
            return False

        tokenizer = self._get_tokenizer_for_model(model)
        return tokenizer is not None

    def get_priority(self) -> int:
        """
        Get strategy priority.

        Returns:
            70 (high priority for known models)
        """
        return 70

    def estimate_cost_impact(self) -> float:
        """
        Estimate computational cost.

        Returns:
            0.3 (moderate cost, 5-20ms)
        """
        return 0.3


# Auto-register strategy
StrategyRegistry.register("token", TokenStrategy)
