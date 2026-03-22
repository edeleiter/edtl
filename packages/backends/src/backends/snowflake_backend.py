"""Snowflake backend for training-time ETL and batch inference."""

import ibis
import ibis.expr.types as ir

from backends.base import Backend
from backends.config import SnowflakeConfig


class SnowflakeBackend(Backend):
    """Snowflake backend for training-time ETL and batch inference."""

    def __init__(self, config: SnowflakeConfig | None = None):
        self.config = config or SnowflakeConfig()

    def connect(self) -> ibis.BaseBackend:
        if not self.config.is_configured:
            raise ConnectionError(
                "Snowflake credentials are not configured "
                "(SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, and SNOWFLAKE_PASSWORD or "
                "SNOWFLAKE_AUTHENTICATOR=externalbrowser). "
                "For local development, use the DuckDB backend instead: "
                "set USE_BACKEND=duckdb in your environment."
            )
        connect_kwargs = {
            "account": self.config.account,
            "user": self.config.user,
            "database": self.config.database,
            "schema": self.config.schema_name,
            "warehouse": self.config.warehouse,
            "role": self.config.role if self.config.role else None,
        }
        if self.config.authenticator == "externalbrowser":
            connect_kwargs["authenticator"] = "externalbrowser"
        else:
            connect_kwargs["password"] = self.config.password
        return ibis.snowflake.connect(**connect_kwargs)

    def disconnect(self, conn: ibis.BaseBackend) -> None:
        conn.disconnect()

    def load_reference_table(
        self,
        conn: ibis.BaseBackend,
        table_name: str,
        parquet_path: str,
    ) -> ir.Table:
        return conn.table(table_name)
