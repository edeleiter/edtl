"""Tests for fourth-down filtering and decision classification."""

import ibis
import pytest

from nfl.fourth_down_filter import filter_fourth_downs, classify_decision


@pytest.fixture
def pbp_table():
    """Synthetic PBP data with various downs and play types."""
    return ibis.memtable({
        "game_id": ["g1"] * 8,
        "down": [1, 2, 3, 4, 4, 4, 4, 4],
        "play_type": ["pass", "run", "pass", "pass", "run", "punt", "field_goal", "penalty"],
        "ydstogo": [10, 5, 3, 2, 1, 8, 12, 5],
        "yardline_100": [75, 60, 45, 30, 1, 80, 35, 50],
    })


def test_filter_fourth_downs(duckdb_conn, pbp_table):
    result = duckdb_conn.execute(filter_fourth_downs(pbp_table))
    # Should keep only down==4 with valid play types (not penalty)
    assert len(result) == 4
    assert all(result["down"] == 4)


def test_classify_decision(duckdb_conn, pbp_table):
    filtered = filter_fourth_downs(pbp_table)
    classified = classify_decision(filtered)
    result = duckdb_conn.execute(classified)
    decisions = list(result["decision"])
    assert decisions.count("go_for_it") == 2  # pass + run
    assert decisions.count("punt") == 1
    assert decisions.count("field_goal") == 1
