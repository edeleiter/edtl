"""Tests for target variable engineering."""

import ibis
import pandas as pd
import pytest

from nfl.target import add_target_label, TARGET_COLUMN, LABEL_MAP, INVERSE_LABEL_MAP


@pytest.fixture
def classified_table():
    """Table with decision column (output of classify_decision)."""
    return ibis.memtable({
        "decision": ["go_for_it", "punt", "field_goal", "go_for_it"],
        "ydstogo": [3, 8, 12, 1],
    })


def test_add_target_label(duckdb_conn, classified_table):
    result = duckdb_conn.execute(add_target_label(classified_table))
    assert TARGET_COLUMN in result.columns
    labels = list(result[TARGET_COLUMN])
    assert labels == [0, 1, 2, 0]


def test_label_map_completeness():
    assert set(LABEL_MAP.keys()) == {"go_for_it", "punt", "field_goal"}
    assert set(LABEL_MAP.values()) == {0, 1, 2}


def test_inverse_label_map():
    for label, code in LABEL_MAP.items():
        assert INVERSE_LABEL_MAP[code] == label


def test_unknown_decision_gets_null(duckdb_conn):
    table = ibis.memtable({"decision": ["unknown_play"], "ydstogo": [5]})
    result = duckdb_conn.execute(add_target_label(table))
    assert result[TARGET_COLUMN].iloc[0] is None or pd.isna(result[TARGET_COLUMN].iloc[0])
