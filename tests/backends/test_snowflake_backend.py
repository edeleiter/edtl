import os

import pytest

from backends.base import Backend
from backends.config import SnowflakeConfig
from backends.snowflake_backend import SnowflakeBackend


def test_snowflake_backend_is_backend():
    backend = SnowflakeBackend()
    assert isinstance(backend, Backend)


def test_snowflake_raises_without_credentials():
    """Explicitly empty config — ignores .env."""
    empty_config = SnowflakeConfig(
        account="", user="", password="", authenticator=""
    )
    backend = SnowflakeBackend(config=empty_config)
    with pytest.raises(ConnectionError, match="Snowflake credentials are not configured"):
        backend.connect()


# Live Snowflake tests — skip if not configured
pytestmark_live = pytest.mark.skipif(
    not os.environ.get("SNOWFLAKE_ACCOUNT"),
    reason="SNOWFLAKE_ACCOUNT not set",
)


@pytestmark_live
def test_snowflake_backend_connects():
    backend = SnowflakeBackend()
    conn = backend.connect()
    result = conn.raw_sql("SELECT 1 AS x").fetchone()
    assert result[0] == 1
    backend.disconnect(conn)
