"""Backend configuration from environment variables."""

from pydantic_settings import BaseSettings


class SnowflakeConfig(BaseSettings):
    """Snowflake connection settings from environment variables."""

    model_config = {"env_prefix": "SNOWFLAKE_"}

    account: str = ""
    user: str = ""
    password: str = ""
    database: str = ""
    schema_name: str = "PUBLIC"
    warehouse: str = ""
    role: str = ""
    authenticator: str = ""  # "externalbrowser", "snowflake" (default), or SSO URL

    @property
    def is_configured(self) -> bool:
        """Return True if the minimum required credentials are set."""
        if self.authenticator == "externalbrowser":
            return bool(self.account and self.user)
        return bool(self.account and self.user and self.password)


class DuckDBConfig(BaseSettings):
    """DuckDB connection settings."""

    model_config = {"env_prefix": "DUCKDB_"}

    database: str = ":memory:"
    threads: int = 4
