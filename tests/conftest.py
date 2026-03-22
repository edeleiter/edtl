"""Shared test fixtures for the unified-etl monorepo."""

import datetime

import ibis
import pytest
from dotenv import load_dotenv

# Load .env at the repo root before any tests run.
# This makes SNOWFLAKE_* vars available without manual export.
load_dotenv()


@pytest.fixture
def duckdb_conn():
    """Fresh in-memory DuckDB connection for each test."""
    conn = ibis.duckdb.connect()
    yield conn
    conn.disconnect()


@pytest.fixture
def sample_data():
    """Standard sample dataset for integration tests."""
    return {
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        "category": ["A", "B", "A", "C", "B"],
        "event_ts": [
            datetime.datetime(2025, 1, 6, 10, 0),
            datetime.datetime(2025, 1, 7, 14, 30),
            datetime.datetime(2025, 1, 8, 9, 15),
            datetime.datetime(2025, 1, 9, 18, 0),
            datetime.datetime(2025, 1, 10, 7, 45),
        ],
        "numerator": [10.0, 20.0, 30.0, 40.0, 50.0],
        "denominator": [2.0, 4.0, 0.0, 8.0, 10.0],
    }
