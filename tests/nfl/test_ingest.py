"""Tests for NFL data ingestion — uses synthetic data, no network calls."""

import pandas as pd
import pytest

from nfl.ingest import PBP_MINIMUM_COLUMNS, pbp_to_parquet, load_pbp_from_parquet


def test_pbp_minimum_columns_has_required_fields():
    required = {"game_id", "play_type", "down", "ydstogo", "yardline_100", "epa", "wp", "qtr"}
    assert required.issubset(set(PBP_MINIMUM_COLUMNS))


def test_parquet_round_trip(tmp_path):
    df = pd.DataFrame({
        "game_id": ["2023_01_KC_DET"],
        "play_type": ["pass"],
        "down": [4],
        "ydstogo": [3],
        "yardline_100": [35],
    })
    path = pbp_to_parquet(df, tmp_path / "test.parquet")
    loaded = load_pbp_from_parquet(path)
    assert len(loaded) == 1
    assert loaded["game_id"].iloc[0] == "2023_01_KC_DET"
