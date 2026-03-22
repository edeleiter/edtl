"""Tests for NFL feature engineering."""

import ibis
import pytest

from nfl.features import (
    MODEL_FEATURE_COLUMNS,
    build_fourth_down_features,
    add_field_position_features,
    add_game_state_features,
    add_time_features,
)


@pytest.fixture
def con():
    return ibis.duckdb.connect()


@pytest.fixture
def game_state_table():
    """Synthetic game states for testing feature transforms."""
    return ibis.memtable({
        "ydstogo": [3, 10, 1],
        "yardline_100": [35, 60, 5],
        "score_differential": [-7, 0, 14],
        "half_seconds_remaining": [600, 900, 100],
        "game_seconds_remaining": [2400, 1800, 100],
        "quarter_seconds_remaining": [600, 900, 100],
        "qtr": [2, 1, 4],
        "goal_to_go": [0, 0, 1],
        "wp": [0.35, 0.50, 0.85],
    })


def test_model_feature_columns_count():
    assert len(MODEL_FEATURE_COLUMNS) == 17


def test_build_fourth_down_features_adds_all_columns(con, game_state_table):
    result = con.execute(build_fourth_down_features(game_state_table))
    for col in MODEL_FEATURE_COLUMNS:
        assert col in result.columns, f"Missing column: {col}"


def test_field_position_features(con, game_state_table):
    result = con.execute(add_field_position_features(game_state_table))
    # yardline_100=35 → is_opponent_territory=1 (<=50), is_fg_range=1 (<=40)
    assert result["is_opponent_territory"].iloc[0] == 1
    assert result["is_fg_range"].iloc[0] == 1
    # yardline_100=60 → is_opponent_territory=0, is_fg_range=0
    assert result["is_opponent_territory"].iloc[1] == 0
    assert result["is_fg_range"].iloc[1] == 0
    # ydstogo=1 → is_short_yardage=1
    assert result["is_short_yardage"].iloc[2] == 1
    # ydstogo=10 → is_short_yardage=0
    assert result["is_short_yardage"].iloc[1] == 0


def test_game_state_features(con, game_state_table):
    result = con.execute(add_game_state_features(game_state_table))
    # score_diff=-7 → is_trailing=1, is_two_score_game=0 (abs(7)<9)
    assert result["is_trailing"].iloc[0] == 1
    assert result["is_two_score_game"].iloc[0] == 0
    # score_diff=14 → is_trailing=0, is_two_score_game=1
    assert result["is_trailing"].iloc[2] == 0
    assert result["is_two_score_game"].iloc[2] == 1
    # abs_score_diff
    assert result["abs_score_diff"].iloc[0] == 7


def test_time_features(con, game_state_table):
    result = con.execute(add_time_features(game_state_table))
    # qtr=2 → is_second_half=0
    assert result["is_second_half"].iloc[0] == 0
    # qtr=4 → is_second_half=1
    assert result["is_second_half"].iloc[2] == 1
    # qtr=4, game_seconds=100, score_diff=14 (leading) → is_late_and_trailing=0
    assert result["is_late_and_trailing"].iloc[2] == 0
