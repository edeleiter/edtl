"""Backend connection management for Snowflake and DuckDB."""

from backends.base import Backend
from backends.duckdb_backend import DuckDBBackend
from backends.snowflake_backend import SnowflakeBackend

__all__ = ["Backend", "DuckDBBackend", "SnowflakeBackend"]
