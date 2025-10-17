"""
REST API for querying and aggregating monitoring data.

This module provides a FastAPI-based HTTP API for accessing
monitoring events and metrics.
"""

from llmops_monitoring.api.server import create_api_server

__all__ = ["create_api_server"]
