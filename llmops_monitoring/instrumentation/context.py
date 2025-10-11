"""
Context management for hierarchical span tracking.

Uses Python's contextvars to maintain parent-child relationships
across async boundaries automatically.
"""

import contextvars
from contextlib import contextmanager
from typing import Optional
from uuid import uuid4


# Context variables for hierarchical tracking
_session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "session_id", default=None
)
_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trace_id", default=None
)
_span_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "span_id", default=None
)
_parent_span_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "parent_span_id", default=None
)


class SpanContext:
    """
    Context manager for span lifecycle.

    Automatically manages parent-child relationships in hierarchical traces.
    Works seamlessly across async/await boundaries.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        operation_name: Optional[str] = None
    ):
        self.session_id = session_id
        self.trace_id = trace_id
        self.span_id = span_id or str(uuid4())
        self.operation_name = operation_name

        # Will be set in __enter__
        self.parent_span_id: Optional[str] = None
        self._tokens: list = []

    def __enter__(self) -> "SpanContext":
        """Enter span context and set up parent-child relationship."""
        # Inherit from parent context
        if self.session_id is None:
            self.session_id = _session_id_var.get()
        if self.trace_id is None:
            self.trace_id = _trace_id_var.get()

        # Generate IDs if not provided
        if self.session_id is None:
            self.session_id = str(uuid4())
        if self.trace_id is None:
            self.trace_id = str(uuid4())

        # Set parent from current span (if exists)
        self.parent_span_id = _span_id_var.get()

        # Update context variables
        self._tokens.append(_session_id_var.set(self.session_id))
        self._tokens.append(_trace_id_var.set(self.trace_id))
        self._tokens.append(_span_id_var.set(self.span_id))
        self._tokens.append(_parent_span_id_var.set(self.parent_span_id))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit span context and restore parent."""
        # Restore previous context
        for token in reversed(self._tokens):
            token.var.reset(token)

        return False

    @classmethod
    def current(cls) -> "SpanContext":
        """Get the current span context."""
        return cls(
            session_id=_session_id_var.get(),
            trace_id=_trace_id_var.get(),
            span_id=_span_id_var.get()
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for event creation."""
        return {
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id
        }


@contextmanager
def monitoring_session(session_id: Optional[str] = None):
    """
    Create a new monitoring session.

    All operations within this context will be grouped under the same session.

    Args:
        session_id: Optional session identifier. Generated if not provided.

    Example:
        with monitoring_session("user-123"):
            result = my_llm_function()
    """
    span_ctx = SpanContext(session_id=session_id or str(uuid4()))
    with span_ctx:
        yield span_ctx


@contextmanager
def monitoring_trace(trace_id: Optional[str] = None):
    """
    Create a new monitoring trace within current session.

    Useful for grouping related operations within a session.

    Args:
        trace_id: Optional trace identifier. Generated if not provided.

    Example:
        with monitoring_session("user-123"):
            with monitoring_trace("workflow-456"):
                step1()
                step2()
    """
    span_ctx = SpanContext(trace_id=trace_id or str(uuid4()))
    with span_ctx:
        yield span_ctx


def get_current_session_id() -> Optional[str]:
    """Get the current session ID from context."""
    return _session_id_var.get()


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID from context."""
    return _trace_id_var.get()


def get_current_span_id() -> Optional[str]:
    """Get the current span ID from context."""
    return _span_id_var.get()


def get_current_parent_span_id() -> Optional[str]:
    """Get the current parent span ID from context."""
    return _parent_span_id_var.get()


def set_session_id(session_id: str) -> None:
    """Manually set session ID in context."""
    _session_id_var.set(session_id)


def set_trace_id(trace_id: str) -> None:
    """Manually set trace ID in context."""
    _trace_id_var.set(trace_id)


def clear_context() -> None:
    """Clear all context variables (mainly for testing)."""
    _session_id_var.set(None)
    _trace_id_var.set(None)
    _span_id_var.set(None)
    _parent_span_id_var.set(None)
