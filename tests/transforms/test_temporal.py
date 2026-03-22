import datetime

import ibis
import pytest

from transforms.features.temporal import (
    extract_dow,
    extract_hour,
    days_since,
)


@pytest.fixture
def con():
    return ibis.duckdb.connect()


@pytest.fixture
def sample_table():
    return ibis.memtable(
        {
            "event_ts": [
                datetime.datetime(2025, 1, 6, 14, 30),   # Monday
                datetime.datetime(2025, 1, 7, 9, 0),     # Tuesday
                datetime.datetime(2025, 1, 11, 22, 15),   # Saturday
            ],
        }
    )


def test_extract_dow(con, sample_table):
    expr = extract_dow(sample_table, "event_ts")
    result = con.execute(expr)
    assert list(result["event_ts_dow"]) == [0, 1, 5]


def test_extract_hour(con, sample_table):
    expr = extract_hour(sample_table, "event_ts")
    result = con.execute(expr)
    assert list(result["event_ts_hour"]) == [14, 9, 22]


def test_days_since(con, sample_table):
    ref_date = datetime.date(2025, 1, 1)
    expr = days_since(sample_table, "event_ts", ref_date)
    result = con.execute(expr)
    assert list(result["event_ts_days_since"]) == [5, 6, 10]
