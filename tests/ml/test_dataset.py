"""Tests for the dataset builder."""

import ibis
import pandas as pd
import pytest

from ml.dataset import build_training_dataset, split_features_target, train_test_split_by_season
from nfl.features import MODEL_FEATURE_COLUMNS
from nfl.target import TARGET_COLUMN


@pytest.fixture
def synthetic_pbp():
    """Synthetic PBP data with 4th-down plays."""
    return ibis.memtable({
        "game_id": ["g1"] * 6,
        "down": [4, 4, 4, 4, 1, 3],
        "play_type": ["pass", "punt", "field_goal", "run", "pass", "run"],
        "ydstogo": [3, 8, 12, 1, 10, 5],
        "yardline_100": [35, 80, 35, 5, 75, 60],
        "score_differential": [-7, 0, 3, 14, -3, 7],
        "half_seconds_remaining": [600, 900, 100, 50, 800, 700],
        "game_seconds_remaining": [2400, 1800, 100, 50, 3000, 2500],
        "quarter_seconds_remaining": [600, 900, 100, 50, 800, 700],
        "qtr": [2, 1, 4, 4, 1, 2],
        "goal_to_go": [0, 0, 0, 1, 0, 0],
        "wp": [0.35, 0.50, 0.65, 0.85, 0.45, 0.55],
        "epa": [1.5, -0.5, 0.2, 3.0, 0.8, -0.2],
        "posteam": ["KC"] * 6,
        "defteam": ["DET"] * 6,
        "season": [2022] * 4 + [2023] * 2,
        "week": [1] * 6,
    })


def test_build_training_dataset_produces_features(duckdb_conn, synthetic_pbp):
    dataset = duckdb_conn.execute(build_training_dataset(synthetic_pbp))
    # Should have all model feature columns + target
    for col in MODEL_FEATURE_COLUMNS:
        assert col in dataset.columns, f"Missing: {col}"
    assert TARGET_COLUMN in dataset.columns
    # Should only have 4th-down plays (4 of 6 rows)
    assert len(dataset) == 4


def test_split_features_target(duckdb_conn, synthetic_pbp):
    dataset = duckdb_conn.execute(build_training_dataset(synthetic_pbp))
    X, y = split_features_target(dataset)
    assert list(X.columns) == MODEL_FEATURE_COLUMNS
    assert y.name == TARGET_COLUMN
    assert len(X) == len(y)


def test_train_test_split_by_season(duckdb_conn, synthetic_pbp):
    dataset = duckdb_conn.execute(build_training_dataset(synthetic_pbp))
    # All synthetic data is season 2022, so test_seasons=[2023] gives empty test
    train, test = train_test_split_by_season(dataset, test_seasons=[2023])
    assert len(train) == 4
    assert len(test) == 0
