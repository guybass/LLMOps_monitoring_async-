"""
Core decorator for monitoring LLM operations.

The @monitor_llm decorator is the main entry point for instrumentation.
"""

import asyncio
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Union

from llmops_monitoring.instrumentation.base import CollectorRegistry, MetricCollector
from llmops_monitoring.instrumentation.collectors.text import TextCollector
from llmops_monitoring.instrumentation.collectors.image import ImageCollector
from llmops_monitoring.instrumentation.collectors.cost import CostCollector
from llmops_monitoring.instrumentation.context import SpanContext
from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.transport.writer import MonitoringWriter


def monitor_llm(
    operation_name: Optional[str] = None,
    operation_type: str = "llm_call",
    measure_text: Optional[Union[bool, List[str]]] = None,
    measure_images: Optional[Union[bool, List[str]]] = None,
    collectors: Optional[List[Union[str, MetricCollector]]] = None,
    measurement: Optional[Union[str, Dict[str, Any]]] = None,
    session_id_from: Optional[str] = None,
    trace_id_from: Optional[str] = None,
    custom_attributes: Optional[Dict[str, Any]] = None,
    enabled: bool = True,
) -> Callable:
    """
    Decorator for monitoring LLM operations.

    This is the main entry point for instrumentation. It wraps functions/methods
    that call LLMs and automatically collects metrics, maintaining hierarchical
    relationships across nested calls.

    Args:
        operation_name: Name for this operation. Defaults to function name.
        operation_type: Type of operation (llm_call, embedding, agent_step, etc.)
        measure_text: Text metrics to collect. Options:
                     - True: collect all text metrics
                     - List[str]: specific metrics like ["char_count", "word_count"]
                     - False/None: don't collect text metrics
        measure_images: Image metrics to collect (same options as measure_text)
        collectors: Additional custom collectors to use
        measurement: Measurement strategy configuration. Options:
                    - None: Use default (auto mode)
                    - "auto": Auto-select strategy based on model
                    - "capacity": Use capacity-based measurement (chars/words/bytes)
                    - "token": Use token-based measurement (industry standard)
                    - "hybrid": Use both capacity and token measurements
                    - Dict: Full configuration (e.g., {"mode": "token", "prefer_tokens": True})
        session_id_from: Kwarg name to extract session_id from
        trace_id_from: Kwarg name to extract trace_id from
        custom_attributes: Static attributes to add to all events
        enabled: If False, decorator does nothing (useful for testing)

    Returns:
        Decorated function that collects metrics

    Examples:
        Basic usage:
            @monitor_llm(operation_name="generate_response")
            async def my_llm_call(prompt: str):
                return await llm.generate(prompt)

        With token measurement:
            @monitor_llm(
                operation_name="chat_completion",
                measurement="token",
                custom_attributes={"model": "gpt-4"}
            )
            async def chat(messages: list):
                return await openai.chat(messages)

        With hybrid measurement:
            @monitor_llm(
                operation_name="generate",
                measurement="hybrid",
                measure_text=True
            )
            async def generate(prompt: str):
                return await model.generate(prompt)
    """

    def decorator(func: Callable) -> Callable:
        # If disabled, return original function
        if not enabled:
            return func

        # Set operation name
        op_name = operation_name or func.__name__

        # Build collector list
        collector_instances = _build_collectors(
            measure_text=measure_text,
            measure_images=measure_images,
            custom_collectors=collectors,
            measurement=measurement
        )

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await _monitor_execution(
                func=func,
                args=args,
                kwargs=kwargs,
                operation_name=op_name,
                operation_type=operation_type,
                collectors=collector_instances,
                session_id_from=session_id_from,
                trace_id_from=trace_id_from,
                custom_attributes=custom_attributes or {},
                measurement=measurement,
                is_async=True
            )

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return asyncio.run(_monitor_execution(
                func=func,
                args=args,
                kwargs=kwargs,
                operation_name=op_name,
                operation_type=operation_type,
                collectors=collector_instances,
                session_id_from=session_id_from,
                trace_id_from=trace_id_from,
                custom_attributes=custom_attributes or {},
                measurement=measurement,
                is_async=False
            ))

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


async def _monitor_execution(
    func: Callable,
    args: tuple,
    kwargs: Dict[str, Any],
    operation_name: str,
    operation_type: str,
    collectors: List[MetricCollector],
    session_id_from: Optional[str],
    trace_id_from: Optional[str],
    custom_attributes: Dict[str, Any],
    measurement: Optional[Union[str, Dict[str, Any]]],
    is_async: bool
) -> Any:
    """Execute function with monitoring."""
    # Extract session/trace IDs from kwargs if specified
    session_id = kwargs.get(session_id_from) if session_id_from else None
    trace_id = kwargs.get(trace_id_from) if trace_id_from else None

    # Create span context (inherits from parent if exists)
    span_ctx = SpanContext(
        session_id=session_id,
        trace_id=trace_id,
        operation_name=operation_name
    )

    start_time = time.time()
    error = None
    error_type = None
    result = None

    with span_ctx:
        try:
            # Execute the function
            if is_async:
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

        except Exception as e:
            error = str(e)
            error_type = type(e).__name__
            raise  # Re-raise the exception

        finally:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Extract model from kwargs or custom_attributes for measurement context
            model = kwargs.get("model") or custom_attributes.get("model")

            # Collect metrics from all collectors
            collected_metrics = {}
            context = {
                "duration_ms": duration_ms,
                "span_ctx": span_ctx,
                "custom_attributes": custom_attributes,
                "measurement": measurement,
                "model": model,
                "function_args": kwargs
            }

            for collector in collectors:
                try:
                    if collector.should_collect(result, args, kwargs):
                        metrics = collector.collect(result, args, kwargs, context)
                        # Update context with newly collected metrics so later collectors can use them
                        if "text_metrics" in metrics:
                            context["text_metrics"] = metrics["text_metrics"]
                        if "image_metrics" in metrics:
                            context["image_metrics"] = metrics["image_metrics"]
                        collected_metrics.update(metrics)
                except Exception:
                    # Fail silently - never break user code
                    pass

            # Create event
            event = MetricEvent(
                session_id=span_ctx.session_id,
                trace_id=span_ctx.trace_id,
                span_id=span_ctx.span_id,
                parent_span_id=span_ctx.parent_span_id,
                operation_name=operation_name,
                operation_type=operation_type,
                duration_ms=duration_ms,
                error=error,
                error_type=error_type,
                custom_attributes=custom_attributes,
                **collected_metrics
            )

            # Send to writer (async, non-blocking)
            try:
                writer = MonitoringWriter.get_instance_sync()
                if writer:
                    await writer.write_event(event)
            except Exception:
                # Fail silently
                pass

    return result


def _build_collectors(
    measure_text: Optional[Union[bool, List[str]]],
    measure_images: Optional[Union[bool, List[str]]],
    custom_collectors: Optional[List[Union[str, MetricCollector]]],
    measurement: Optional[Union[str, Dict[str, Any]]]
) -> List[MetricCollector]:
    """Build list of collector instances from configuration."""
    collectors = []

    # Add text collector
    if measure_text:
        if measure_text is True:
            collectors.append(TextCollector(measurement=measurement))
        elif isinstance(measure_text, list):
            collectors.append(TextCollector(measure=measure_text, measurement=measurement))

    # Add image collector
    if measure_images:
        if measure_images is True:
            collectors.append(ImageCollector())
        elif isinstance(measure_images, list):
            collectors.append(ImageCollector(measure=measure_images))

    # Add custom collectors
    if custom_collectors:
        for collector in custom_collectors:
            if isinstance(collector, str):
                # Get from registry
                instance = CollectorRegistry.get(collector)
                if instance:
                    collectors.append(instance)
            elif isinstance(collector, MetricCollector):
                collectors.append(collector)

    return collectors


# Register built-in collectors
CollectorRegistry.register("text", TextCollector)
CollectorRegistry.register("image", ImageCollector)
CollectorRegistry.register("cost", CostCollector)
