"""
Async writer for metric events with buffering and batching.

Handles delivery of events to storage backends without blocking application code.
"""

import asyncio
import logging
from typing import List, Optional
from datetime import datetime

from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.schema.config import MonitorConfig
from llmops_monitoring.transport.backends.base import StorageBackend


logger = logging.getLogger(__name__)


class MonitoringWriter:
    """
    Singleton writer that manages async delivery of metric events.

    Features:
    - Non-blocking writes via async queue
    - Automatic batching for efficiency
    - Configurable flush intervals
    - Graceful shutdown with flush
    - Error handling with optional retries

    Extension Point: Replace queue implementation (e.g., with Kafka, Redis)
    """

    _instance: Optional["MonitoringWriter"] = None
    _lock = asyncio.Lock()

    def __init__(self, config: Optional[MonitorConfig] = None):
        """
        Initialize the writer.

        Args:
            config: Monitoring configuration
        """
        if MonitoringWriter._instance is not None:
            raise RuntimeError("MonitoringWriter is a singleton. Use get_instance()")

        self.config = config or MonitorConfig.for_local_dev()
        self.backend: Optional[StorageBackend] = None
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.worker_task: Optional[asyncio.Task] = None
        self.running = False
        self.buffer: List[MetricEvent] = []
        self.last_flush = datetime.utcnow()

        MonitoringWriter._instance = self

    @classmethod
    async def get_instance(
        cls,
        config: Optional[MonitorConfig] = None,
        auto_start: bool = True
    ) -> "MonitoringWriter":
        """
        Get or create the singleton instance.

        Args:
            config: Configuration (only used on first call)
            auto_start: If True, start the worker automatically

        Returns:
            MonitoringWriter instance
        """
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    instance = cls(config)
                    if auto_start:
                        await instance.start()

        return cls._instance

    @classmethod
    def get_instance_sync(cls) -> Optional["MonitoringWriter"]:
        """Get existing instance synchronously (for use in decorators)."""
        return cls._instance

    async def start(self) -> None:
        """Start the background worker."""
        if self.running:
            return

        # Initialize backend
        self.backend = await self._create_backend()
        await self.backend.initialize()

        # Start worker
        self.running = True
        self.worker_task = asyncio.create_task(self._worker())

        logger.info("MonitoringWriter started")

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the writer and flush remaining events.

        Args:
            timeout: Maximum time to wait for flush (seconds)
        """
        if not self.running:
            return

        self.running = False

        # Wait for queue to drain
        try:
            await asyncio.wait_for(self.queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for queue to drain. {self.queue.qsize()} events lost")

        # Flush buffer
        if self.buffer:
            await self._flush_buffer()

        # Cancel worker
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

        # Close backend
        if self.backend:
            await self.backend.close()

        logger.info("MonitoringWriter stopped")

    async def write_event(self, event: MetricEvent) -> None:
        """
        Write an event to the queue (non-blocking).

        Args:
            event: Metric event to write

        Raises:
            QueueFull: If queue is at max capacity (only if fail_silently=False)
        """
        try:
            await self.queue.put(event)
        except asyncio.QueueFull:
            if not self.config.fail_silently:
                raise
            logger.warning("Queue full, dropping event")

    async def _worker(self) -> None:
        """Background worker that processes events from queue."""
        while self.running:
            try:
                # Wait for event or timeout
                try:
                    event = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=self.config.storage.flush_interval_seconds
                    )
                    self.buffer.append(event)
                    self.queue.task_done()

                except asyncio.TimeoutError:
                    # Flush on timeout
                    if self.buffer:
                        await self._flush_buffer()
                    continue

                # Flush if buffer is full
                if len(self.buffer) >= self.config.storage.batch_size:
                    await self._flush_buffer()

                # Flush if interval elapsed
                elapsed = (datetime.utcnow() - self.last_flush).total_seconds()
                if elapsed >= self.config.storage.flush_interval_seconds:
                    await self._flush_buffer()

            except Exception as e:
                if self.config.fail_silently:
                    logger.error(f"Error in writer worker: {e}")
                else:
                    raise

    async def _flush_buffer(self) -> None:
        """Flush buffered events to storage."""
        if not self.buffer:
            return

        try:
            if self.backend.supports_batch_writes() and len(self.buffer) > 1:
                await self.backend.write_batch(self.buffer)
            else:
                for event in self.buffer:
                    await self.backend.write_event(event)

            logger.debug(f"Flushed {len(self.buffer)} events to storage")
            self.buffer.clear()
            self.last_flush = datetime.utcnow()

        except Exception as e:
            if self.config.retry_failed_writes:
                logger.warning(f"Flush failed, will retry: {e}")
                # Keep events in buffer for retry
            else:
                logger.error(f"Flush failed, dropping {len(self.buffer)} events: {e}")
                self.buffer.clear()

    async def _create_backend(self) -> StorageBackend:
        """Create storage backend from configuration."""
        backend_type = self.config.storage.backend

        if backend_type == "parquet":
            from llmops_monitoring.transport.backends.parquet import ParquetBackend
            return ParquetBackend(self.config.storage)

        elif backend_type == "postgres":
            from llmops_monitoring.transport.backends.postgres import PostgresBackend
            return PostgresBackend(self.config.storage)

        elif backend_type == "mysql":
            from llmops_monitoring.transport.backends.mysql import MySQLBackend
            return MySQLBackend(self.config.storage)

        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    async def health_check(self) -> dict:
        """
        Check health of the writer.

        Returns:
            Health status dictionary
        """
        return {
            "running": self.running,
            "queue_size": self.queue.qsize(),
            "buffer_size": len(self.buffer),
            "backend_healthy": await self.backend.health_check() if self.backend else False
        }


# Convenience function for quick setup
async def initialize_monitoring(config: Optional[MonitorConfig] = None) -> MonitoringWriter:
    """
    Initialize and start the monitoring system.

    Args:
        config: Configuration (uses defaults if None)

    Returns:
        Started MonitoringWriter instance

    Example:
        await initialize_monitoring(MonitorConfig.for_production("postgres://..."))
    """
    return await MonitoringWriter.get_instance(config=config, auto_start=True)
