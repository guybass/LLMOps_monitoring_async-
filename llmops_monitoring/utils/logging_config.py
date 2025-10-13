"""
Centralized logging configuration for llamonitor-async.

Provides consistent logging setup across all modules with configurable
levels, formats, and handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with consistent configuration.

    Args:
        name: Logger name (usually __name__ of the module)
        level: Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
               If None, uses environment variable LLMOPS_LOG_LEVEL or defaults to INFO

    Returns:
        Configured logger instance

    Example:
        >>> from llmops_monitoring.utils.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    logger = logging.getLogger(name)

    # Don't add handlers if already configured (prevents duplicates)
    if logger.handlers:
        return logger

    # Determine log level
    if level is None:
        import os
        level = os.getenv("LLMOPS_LOG_LEVEL", "INFO").upper()

    logger.setLevel(getattr(logging, level, logging.INFO))

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # Set format
    formatter = logging.Formatter(DEFAULT_FORMAT)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger


def configure_logging(
    level: str = "INFO",
    format_style: str = "default",
    log_file: Optional[Path] = None,
    quiet: bool = False
) -> None:
    """
    Configure global logging settings for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_style: Log format ("default", "simple", "detailed")
        log_file: Optional file path for file logging
        quiet: If True, suppress console output

    Example:
        >>> from llmops_monitoring.utils.logging_config import configure_logging
        >>> configure_logging(level="DEBUG", format_style="detailed")
    """
    # Select format
    format_map = {
        "default": DEFAULT_FORMAT,
        "simple": SIMPLE_FORMAT,
        "detailed": DETAILED_FORMAT,
    }
    log_format = format_map.get(format_style, DEFAULT_FORMAT)

    # Get root logger
    root_logger = logging.getLogger("llmops_monitoring")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler (unless quiet)
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(DETAILED_FORMAT))
        root_logger.addHandler(file_handler)

    root_logger.propagate = False


def disable_external_loggers(level: str = "WARNING") -> None:
    """
    Disable or reduce verbosity of external library loggers.

    Args:
        level: Minimum level for external loggers (default: WARNING)

    Common noisy loggers:
    - urllib3
    - asyncio
    - aiohttp
    - httpx
    """
    external_loggers = [
        "urllib3",
        "asyncio",
        "aiohttp",
        "httpx",
        "httpcore",
    ]

    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(getattr(logging, level.upper(), logging.WARNING))


# Module-level logger for this configuration module
logger = get_logger(__name__)
