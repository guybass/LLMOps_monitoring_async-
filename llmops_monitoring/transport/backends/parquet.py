"""
Parquet storage backend for local file-based storage.

Stores events in partitioned Parquet files for efficient querying.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List
import json

from llmops_monitoring.schema.config import StorageConfig
from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.transport.backends.base import StorageBackend


logger = logging.getLogger(__name__)


class ParquetBackend(StorageBackend):
    """
    Local Parquet file storage backend.

    Features:
    - Partitioned by date for efficient time-range queries
    - Async writes using aiofiles
    - Automatic directory creation
    - Efficient columnar storage

    Extension Point: Add custom partitioning strategies
    """

    def __init__(self, config: StorageConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.partition_by = config.partition_by
        self._pandas_available = self._check_pandas()
        self._pyarrow_available = self._check_pyarrow()

    async def initialize(self) -> None:
        """Create output directory if it doesn't exist."""
        if not self._pandas_available or not self._pyarrow_available:
            raise RuntimeError(
                "Parquet backend requires pandas and pyarrow. "
                "Install with: pip install 'llmops-monitoring[parquet]'"
            )

        await asyncio.get_event_loop().run_in_executor(
            None,
            self.output_dir.mkdir,
            True,  # parents
            True   # exist_ok
        )
        logger.info(f"Initialized Parquet backend at {self.output_dir}")

    async def write_event(self, event: MetricEvent) -> None:
        """Write a single event to Parquet file."""
        await self.write_batch([event])

    async def write_batch(self, events: List[MetricEvent]) -> None:
        """Write multiple events in a batch."""
        if not events:
            return

        import pandas as pd

        # Convert events to records
        records = [self._event_to_record(event) for event in events]

        # Create DataFrame
        df = pd.DataFrame(records)

        # Partition and write
        await self._write_dataframe(df, events[0].timestamp)

    async def close(self) -> None:
        """No cleanup needed for Parquet backend."""
        logger.info("Closed Parquet backend")

    def _event_to_record(self, event: MetricEvent) -> dict:
        """Convert event to flat record for DataFrame."""
        record = {
            "event_id": str(event.event_id),
            "schema_version": event.schema_version,
            "session_id": event.session_id,
            "trace_id": event.trace_id,
            "span_id": event.span_id,
            "parent_span_id": event.parent_span_id,
            "operation_name": event.operation_name,
            "operation_type": event.operation_type,
            "timestamp": event.timestamp,
            "duration_ms": event.duration_ms,
            "error": event.error,
            "error_type": event.error_type,
        }

        # Flatten text metrics
        if event.text_metrics:
            record["text_char_count"] = event.text_metrics.char_count
            record["text_word_count"] = event.text_metrics.word_count
            record["text_byte_size"] = event.text_metrics.byte_size
            record["text_line_count"] = event.text_metrics.line_count
            record["text_custom_metrics"] = json.dumps(event.text_metrics.custom_metrics)
        else:
            record["text_char_count"] = None
            record["text_word_count"] = None
            record["text_byte_size"] = None
            record["text_line_count"] = None
            record["text_custom_metrics"] = None

        # Flatten image metrics
        if event.image_metrics:
            record["image_count"] = event.image_metrics.count
            record["image_total_pixels"] = event.image_metrics.total_pixels
            record["image_file_size_bytes"] = event.image_metrics.file_size_bytes
            record["image_width"] = event.image_metrics.width
            record["image_height"] = event.image_metrics.height
            record["image_format"] = event.image_metrics.format
            record["image_custom_metrics"] = json.dumps(event.image_metrics.custom_metrics)
        else:
            record["image_count"] = None
            record["image_total_pixels"] = None
            record["image_file_size_bytes"] = None
            record["image_width"] = None
            record["image_height"] = None
            record["image_format"] = None
            record["image_custom_metrics"] = None

        # Custom attributes as JSON
        record["custom_attributes"] = json.dumps(event.custom_attributes)

        return record

    async def _write_dataframe(self, df, timestamp: datetime) -> None:
        """Write DataFrame to partitioned Parquet file."""
        import pandas as pd

        # Determine partition path
        if self.partition_by == "date":
            partition_path = self.output_dir / timestamp.strftime("%Y-%m-%d")
        elif self.partition_by == "session_id":
            session_id = df["session_id"].iloc[0]
            partition_path = self.output_dir / session_id
        else:
            partition_path = self.output_dir

        # Create partition directory
        await asyncio.get_event_loop().run_in_executor(
            None,
            partition_path.mkdir,
            True,  # parents
            True   # exist_ok
        )

        # Generate filename
        filename = f"events_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.parquet"
        file_path = partition_path / filename

        # Write to Parquet (in executor to avoid blocking)
        await asyncio.get_event_loop().run_in_executor(
            None,
            df.to_parquet,
            str(file_path),
            "pyarrow",
            None,  # compression
            "snappy"  # compression type
        )

        logger.debug(f"Wrote {len(df)} events to {file_path}")

    def _check_pandas(self) -> bool:
        """Check if pandas is available."""
        try:
            import pandas
            return True
        except ImportError:
            return False

    def _check_pyarrow(self) -> bool:
        """Check if pyarrow is available."""
        try:
            import pyarrow
            return True
        except ImportError:
            return False

    def supports_batch_writes(self) -> bool:
        return True
