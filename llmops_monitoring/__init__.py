"""
LLMOps Async Monitoring

An extensible, async-first monitoring framework for LLM applications.
Measures text and image usage with pluggable storage backends.

Design Principle: "Leave space for air conditioning" - every component
has clear extension points for future enhancements.
"""

__version__ = "0.1.0"
__author__ = "LLMOps Monitoring Contributors"
__license__ = "Apache-2.0"

from llmops_monitoring.instrumentation.decorators import monitor_llm
from llmops_monitoring.schema.config import MonitorConfig
from llmops_monitoring.transport.writer import MonitoringWriter

__all__ = [
    "monitor_llm",
    "MonitorConfig",
    "MonitoringWriter",
]
