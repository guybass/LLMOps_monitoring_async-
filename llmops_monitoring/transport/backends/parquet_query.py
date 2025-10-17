"""
Parquet query backend for reading and aggregating stored events.

Provides efficient querying of Parquet files using pandas.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import UUID

from llmops_monitoring.schema.config import StorageConfig
from llmops_monitoring.schema.events import MetricEvent, TextMetrics, ImageMetrics
from llmops_monitoring.transport.backends.query_base import QueryBackend, QueryFilter


logger = logging.getLogger(__name__)


class ParquetQueryBackend(QueryBackend):
    """
    Query backend for Parquet files.

    Features:
    - Read from partitioned Parquet files
    - Filter by various criteria
    - Aggregate metrics by operation, model, time period
    - Calculate costs and latencies
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize Parquet query backend.

        Args:
            config: Storage configuration with output_dir
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self._pandas_available = self._check_pandas()
        self._pyarrow_available = self._check_pyarrow()

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

    async def _load_dataframe(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        """
        Load DataFrame from Parquet files.

        Args:
            start_time: Only load files from this time forward
            end_time: Only load files up to this time

        Returns:
            pandas DataFrame with all matching events
        """
        if not self._pandas_available or not self._pyarrow_available:
            raise RuntimeError(
                "Parquet query backend requires pandas and pyarrow. "
                "Install with: pip install 'llamonitor-async[parquet]'"
            )

        import pandas as pd

        # Find all Parquet files
        parquet_files = list(self.output_dir.rglob("*.parquet"))

        if not parquet_files:
            return pd.DataFrame()

        # Load all files in executor (non-blocking)
        dfs = []
        for file_path in parquet_files:
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                pd.read_parquet,
                str(file_path)
            )
            dfs.append(df)

        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        # Filter by time range if specified
        if not combined_df.empty:
            if start_time:
                combined_df = combined_df[combined_df["timestamp"] >= start_time]
            if end_time:
                combined_df = combined_df[combined_df["timestamp"] <= end_time]

        return combined_df

    def _df_row_to_event(self, row: Dict) -> MetricEvent:
        """
        Convert DataFrame row to MetricEvent.

        Args:
            row: Dictionary representing a DataFrame row

        Returns:
            MetricEvent instance
        """
        # Parse text metrics
        text_metrics = None
        if row.get("text_char_count") is not None:
            text_metrics = TextMetrics(
                char_count=row.get("text_char_count"),
                word_count=row.get("text_word_count"),
                byte_size=row.get("text_byte_size"),
                line_count=row.get("text_line_count"),
                custom_metrics=json.loads(row.get("text_custom_metrics", "{}") or "{}")
            )

        # Parse image metrics
        image_metrics = None
        if row.get("image_count") is not None:
            image_metrics = ImageMetrics(
                count=row.get("image_count"),
                total_pixels=row.get("image_total_pixels"),
                file_size_bytes=row.get("image_file_size_bytes"),
                width=row.get("image_width"),
                height=row.get("image_height"),
                format=row.get("image_format"),
                custom_metrics=json.loads(row.get("image_custom_metrics", "{}") or "{}")
            )

        # Parse custom attributes
        custom_attributes = json.loads(row.get("custom_attributes", "{}") or "{}")

        return MetricEvent(
            event_id=UUID(row["event_id"]),
            schema_version=row["schema_version"],
            session_id=row["session_id"],
            trace_id=row["trace_id"],
            span_id=row["span_id"],
            parent_span_id=row.get("parent_span_id"),
            operation_name=row["operation_name"],
            operation_type=row["operation_type"],
            timestamp=row["timestamp"],
            duration_ms=row.get("duration_ms"),
            error=row.get("error", False),
            error_type=row.get("error_type"),
            error_message=row.get("error_message"),
            text_metrics=text_metrics,
            image_metrics=image_metrics,
            custom_attributes=custom_attributes
        )

    async def query_events(self, filter: QueryFilter) -> List[MetricEvent]:
        """Query events with filtering."""
        df = await self._load_dataframe(filter.start_time, filter.end_time)

        if df.empty:
            return []

        # Apply filters
        if filter.session_id:
            df = df[df["session_id"] == filter.session_id]
        if filter.trace_id:
            df = df[df["trace_id"] == filter.trace_id]
        if filter.operation_name:
            df = df[df["operation_name"] == filter.operation_name]
        if filter.operation_type:
            df = df[df["operation_type"] == filter.operation_type]
        if filter.error_only:
            df = df[df["error"] == True]

        # Apply pagination
        df = df.iloc[filter.offset:filter.offset + filter.limit]

        # Convert to events
        events = [self._df_row_to_event(row) for row in df.to_dict("records")]
        return events

    async def get_sessions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get list of sessions with summary stats."""
        df = await self._load_dataframe(start_time, end_time)

        if df.empty:
            return []

        # Group by session_id
        grouped = df.groupby("session_id").agg({
            "timestamp": ["min", "max", "count"],
            "operation_name": "nunique",
            "error": "sum",
            "duration_ms": "sum"
        }).reset_index()

        # Flatten column names
        grouped.columns = [
            "session_id",
            "start_time",
            "end_time",
            "event_count",
            "operation_count",
            "error_count",
            "total_duration_ms"
        ]

        # Sort by start time descending
        grouped = grouped.sort_values("start_time", ascending=False)

        # Apply limit
        grouped = grouped.head(limit)

        return grouped.to_dict("records")

    async def get_traces(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get list of traces for a session."""
        df = await self._load_dataframe()

        if df.empty:
            return []

        # Filter by session
        df = df[df["session_id"] == session_id]

        if df.empty:
            return []

        # Group by trace_id
        grouped = df.groupby("trace_id").agg({
            "timestamp": ["min", "max", "count"],
            "span_id": "count",
            "error": "sum",
            "duration_ms": "sum"
        }).reset_index()

        # Flatten column names
        grouped.columns = [
            "trace_id",
            "start_time",
            "end_time",
            "event_count",
            "span_count",
            "error_count",
            "total_duration_ms"
        ]

        # Sort by start time descending
        grouped = grouped.sort_values("start_time", ascending=False)

        # Apply limit
        grouped = grouped.head(limit)

        return grouped.to_dict("records")

    async def aggregate_by_operation(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics by operation name."""
        import pandas as pd
        import numpy as np

        df = await self._load_dataframe(start_time, end_time)

        if df.empty:
            return []

        # Group by operation_name
        grouped = df.groupby("operation_name").agg({
            "event_id": "count",
            "error": "sum",
            "duration_ms": ["mean", lambda x: x.quantile(0.5), lambda x: x.quantile(0.95), lambda x: x.quantile(0.99)],
            "text_char_count": "sum",
        }).reset_index()

        # Flatten column names
        grouped.columns = [
            "operation_name",
            "count",
            "error_count",
            "avg_duration_ms",
            "p50_duration_ms",
            "p95_duration_ms",
            "p99_duration_ms",
            "total_characters"
        ]

        # Extract costs from custom_attributes
        costs = []
        for op in grouped["operation_name"]:
            op_df = df[df["operation_name"] == op]
            total_cost = 0
            for _, row in op_df.iterrows():
                attrs = json.loads(row.get("custom_attributes", "{}") or "{}")
                cost = attrs.get("estimated_cost_usd", 0) or attrs.get("cost_usd", 0) or 0
                total_cost += cost
            costs.append(total_cost)

        grouped["estimated_cost_usd"] = costs

        # Fill NaN values
        grouped = grouped.fillna(0)

        return grouped.to_dict("records")

    async def aggregate_by_model(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics by model."""
        df = await self._load_dataframe(start_time, end_time)

        if df.empty:
            return []

        # Extract model from custom_attributes
        models = []
        for _, row in df.iterrows():
            attrs = json.loads(row.get("custom_attributes", "{}") or "{}")
            model = attrs.get("model", "unknown")
            models.append(model)

        df["model"] = models

        # Group by model
        grouped = df.groupby("model").agg({
            "event_id": "count",
            "error": "sum",
            "duration_ms": "mean",
            "text_char_count": "sum",
        }).reset_index()

        # Flatten column names
        grouped.columns = [
            "model",
            "count",
            "error_count",
            "avg_duration_ms",
            "total_characters"
        ]

        # Extract costs
        costs = []
        for model in grouped["model"]:
            model_df = df[df["model"] == model]
            total_cost = 0
            for _, row in model_df.iterrows():
                attrs = json.loads(row.get("custom_attributes", "{}") or "{}")
                cost = attrs.get("estimated_cost_usd", 0) or attrs.get("cost_usd", 0) or 0
                total_cost += cost
            costs.append(total_cost)

        grouped["estimated_cost_usd"] = costs

        # Fill NaN values
        grouped = grouped.fillna(0)

        return grouped.to_dict("records")

    async def aggregate_costs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        group_by: str = "model"
    ) -> List[Dict[str, Any]]:
        """Aggregate cost metrics."""
        df = await self._load_dataframe(start_time, end_time)

        if df.empty:
            return []

        # Extract grouping field from custom_attributes
        if group_by == "model":
            group_values = []
            for _, row in df.iterrows():
                attrs = json.loads(row.get("custom_attributes", "{}") or "{}")
                group_values.append(attrs.get("model", "unknown"))
            df["group_field"] = group_values
        elif group_by == "operation":
            df["group_field"] = df["operation_name"]
        elif group_by == "session":
            df["group_field"] = df["session_id"]
        elif group_by == "day":
            df["group_field"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        else:
            raise ValueError(f"Invalid group_by: {group_by}")

        # Extract costs
        costs = []
        for _, row in df.iterrows():
            attrs = json.loads(row.get("custom_attributes", "{}") or "{}")
            cost = attrs.get("estimated_cost_usd", 0) or attrs.get("cost_usd", 0) or 0
            costs.append(cost)

        df["cost"] = costs

        # Group and aggregate
        grouped = df.groupby("group_field").agg({
            "cost": "sum",
            "event_id": "count",
        }).reset_index()

        grouped.columns = [group_by, "total_cost_usd", "operation_count"]
        grouped["avg_cost_per_operation_usd"] = grouped["total_cost_usd"] / grouped["operation_count"]

        # Fill NaN values
        grouped = grouped.fillna(0)

        return grouped.to_dict("records")

    async def get_summary_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get overall summary statistics."""
        df = await self._load_dataframe(start_time, end_time)

        if df.empty:
            return {
                "total_events": 0,
                "total_sessions": 0,
                "total_operations": 0,
                "total_errors": 0,
                "error_rate": 0.0,
                "avg_duration_ms": 0.0,
                "total_characters": 0,
                "total_cost_usd": 0.0
            }

        # Calculate basic stats
        total_events = len(df)
        total_sessions = df["session_id"].nunique()
        total_operations = df["operation_name"].nunique()
        total_errors = df["error"].sum()
        error_rate = total_errors / total_events if total_events > 0 else 0.0
        avg_duration_ms = df["duration_ms"].mean() if "duration_ms" in df.columns else 0.0
        total_characters = df["text_char_count"].sum() if "text_char_count" in df.columns else 0

        # Calculate total cost
        total_cost = 0
        for _, row in df.iterrows():
            attrs = json.loads(row.get("custom_attributes", "{}") or "{}")
            cost = attrs.get("estimated_cost_usd", 0) or attrs.get("cost_usd", 0) or 0
            total_cost += cost

        return {
            "total_events": int(total_events),
            "total_sessions": int(total_sessions),
            "total_operations": int(total_operations),
            "total_errors": int(total_errors),
            "error_rate": float(error_rate),
            "avg_duration_ms": float(avg_duration_ms) if avg_duration_ms else 0.0,
            "total_characters": int(total_characters),
            "total_cost_usd": float(total_cost)
        }

    async def health_check(self) -> bool:
        """Check if the query backend is healthy."""
        try:
            # Check if output directory exists
            if not self.output_dir.exists():
                return False

            # Check if pandas and pyarrow are available
            if not self._pandas_available or not self._pyarrow_available:
                return False

            return True
        except Exception:
            return False
