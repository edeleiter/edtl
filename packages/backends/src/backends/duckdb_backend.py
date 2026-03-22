"""DuckDB backend for local/real-time inference."""

import ibis
import ibis.expr.types as ir

from backends.base import Backend
from backends.config import DuckDBConfig


class DuckDBBackend(Backend):
    """DuckDB backend for local/real-time inference."""

    def __init__(self, config: DuckDBConfig | None = None):
        self.config = config or DuckDBConfig()

    def connect(self) -> ibis.BaseBackend:
        return ibis.duckdb.connect(
            database=self.config.database,
            threads=self.config.threads,
        )

    def disconnect(self, conn: ibis.BaseBackend) -> None:
        conn.disconnect()

    def load_reference_table(
        self,
        conn: ibis.BaseBackend,
        table_name: str,
        parquet_path: str,
    ) -> ir.Table:
        conn.read_parquet(parquet_path, table_name=table_name)
        return conn.table(table_name)
