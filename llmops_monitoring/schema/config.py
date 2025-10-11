"""Configuration models for the monitoring system."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class StorageConfig(BaseModel):
    """Configuration for storage backends.

    Extension Point: Add new backend types here.
    """
    backend: Literal["parquet", "postgres", "mysql"] = "parquet"
    connection_string: Optional[str] = None

    # Parquet-specific
    output_dir: str = "./monitoring_data"
    partition_by: str = "date"  # date, session_id, etc.

    # SQL-specific
    table_name: str = "metric_events"
    schema_name: Optional[str] = None
    pool_size: int = 10

    # Common options
    batch_size: int = 100
    flush_interval_seconds: float = 5.0

    class Config:
        json_schema_extra = {
            "example": {
                "backend": "postgres",
                "connection_string": "postgresql://user:pass@localhost:5432/monitoring",
                "table_name": "metric_events",
                "batch_size": 100,
                "flush_interval_seconds": 5.0
            }
        }


class MonitorConfig(BaseSettings):
    """
    Global configuration for the monitoring system.

    Can be configured via:
    1. Direct instantiation: MonitorConfig(backend="postgres", ...)
    2. Environment variables: LLMOPS_BACKEND=postgres
    3. .env file
    """
    # Storage
    storage: StorageConfig = Field(default_factory=StorageConfig)

    # Instrumentation
    enabled: bool = True
    auto_start: bool = True

    # Performance
    max_queue_size: int = 10000
    worker_threads: int = 1

    # Error handling
    fail_silently: bool = True  # Never crash user code
    retry_failed_writes: bool = True
    max_retries: int = 3

    # Extension point for future features
    extensions: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        env_prefix = "LLMOPS_"
        env_nested_delimiter = "__"

        json_schema_extra = {
            "example": {
                "storage": {
                    "backend": "postgres",
                    "connection_string": "postgresql://localhost/monitoring"
                },
                "enabled": True,
                "max_queue_size": 10000,
                "fail_silently": True
            }
        }

    @classmethod
    def from_env(cls) -> "MonitorConfig":
        """Load configuration from environment variables."""
        return cls()

    @classmethod
    def for_local_dev(cls) -> "MonitorConfig":
        """Preset configuration for local development."""
        return cls(
            storage=StorageConfig(
                backend="parquet",
                output_dir="./dev_monitoring_data",
                batch_size=50,
                flush_interval_seconds=2.0
            )
        )

    @classmethod
    def for_production(cls, connection_string: str) -> "MonitorConfig":
        """Preset configuration for production deployment."""
        return cls(
            storage=StorageConfig(
                backend="postgres",
                connection_string=connection_string,
                batch_size=500,
                flush_interval_seconds=10.0,
                pool_size=20
            ),
            max_queue_size=50000,
            worker_threads=2
        )
