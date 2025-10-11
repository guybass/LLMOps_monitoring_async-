"""
MySQL storage backend (similar to PostgreSQL).

Note: This is a stub implementation. For production use, implement
similar to PostgresBackend with aiomysql.
"""

import logging
from typing import List

from llmops_monitoring.schema.config import StorageConfig
from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.transport.backends.base import StorageBackend


logger = logging.getLogger(__name__)


class MySQLBackend(StorageBackend):
    """
    MySQL storage backend.

    TODO: Implement using aiomysql (similar to PostgresBackend)

    Extension Point: Implement this backend for MySQL support
    """

    def __init__(self, config: StorageConfig):
        self.config = config
        raise NotImplementedError(
            "MySQL backend not yet implemented. "
            "Contributions welcome! See PostgresBackend for reference implementation."
        )

    async def initialize(self) -> None:
        pass

    async def write_event(self, event: MetricEvent) -> None:
        pass

    async def write_batch(self, events: List[MetricEvent]) -> None:
        pass

    async def close(self) -> None:
        pass
