"""Cross-backend parity test — verifies transforms produce identical results."""

import ibis
import numpy as np
import pandas as pd
import pytest

from nfl.features import build_fourth_down_features, MODEL_FEATURE_COLUMNS


@pytest.fixture
def golden_data():
    """Deterministic game states for parity testing."""
    return pd.DataFrame({
        "ydstogo": [3, 10, 1, 5, 7],
        "yardline_100": [35, 60, 5, 50, 80],
        "score_differential": [-7, 0, 14, -3, 21],
        "half_seconds_remaining": [600, 900, 100, 1200, 50],
        "game_seconds_remaining": [2400, 1800, 100, 3000, 50],
        "quarter_seconds_remaining": [600, 900, 100, 300, 50],
        "qtr": [2, 1, 4, 3, 4],
        "goal_to_go": [0, 0, 1, 0, 0],
        "wp": [0.35, 0.50, 0.85, 0.40, 0.95],
    })


def test_parity_duckdb_vs_duckdb(golden_data):
    """Two independent DuckDB connections produce identical feature results."""
    con1 = ibis.duckdb.connect()
    con2 = ibis.duckdb.connect()

    table1 = ibis.memtable(golden_data)
    table2 = ibis.memtable(golden_data)

    result1 = con1.execute(build_fourth_down_features(table1))
    result2 = con2.execute(build_fourth_down_features(table2))

    # Compare all model feature columns
    for col in MODEL_FEATURE_COLUMNS:
        vals1 = result1[col].values
        vals2 = result2[col].values
        np.testing.assert_allclose(
            vals1, vals2, atol=1e-10,
            err_msg=f"Parity failure on column: {col}",
        )


def test_feature_pipeline_deterministic(golden_data):
    """Same input always produces same output."""
    con = ibis.duckdb.connect()

    result_a = con.execute(build_fourth_down_features(ibis.memtable(golden_data)))
    result_b = con.execute(build_fourth_down_features(ibis.memtable(golden_data)))

    pd.testing.assert_frame_equal(
        result_a[MODEL_FEATURE_COLUMNS],
        result_b[MODEL_FEATURE_COLUMNS],
    )


def test_feature_values_are_correct(golden_data):
    """Spot-check specific feature values for known inputs."""
    con = ibis.duckdb.connect()
    result = con.execute(build_fourth_down_features(ibis.memtable(golden_data)))

    # Row 0: yardline_100=35, ydstogo=3, score_diff=-7, qtr=2
    assert result["is_opponent_territory"].iloc[0] == 1  # 35 <= 50
    assert result["is_fg_range"].iloc[0] == 1  # 35 <= 40
    assert result["is_short_yardage"].iloc[0] == 0  # 3 > 2
    assert result["is_trailing"].iloc[0] == 1  # -7 < 0
    assert result["is_second_half"].iloc[0] == 0  # qtr 2 < 3
    assert result["abs_score_diff"].iloc[0] == 7

    # Row 2: yardline_100=5, goal_to_go=1, qtr=4, game_seconds=100, score_diff=14
    assert result["is_fg_range"].iloc[2] == 1  # 5 <= 40
    assert result["is_second_half"].iloc[2] == 1  # qtr 4 >= 3
    assert result["is_late_and_trailing"].iloc[2] == 0  # leading (14 > 0)
    assert result["is_two_score_game"].iloc[2] == 1  # abs(14) >= 9
