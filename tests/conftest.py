"""Shared test fixtures for the unified-etl monorepo."""

import datetime

import ibis
import numpy as np
import pandas as pd
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


@pytest.fixture
def synthetic_training_data():
    """Small synthetic dataset for fast model tests."""
    from nfl.features import MODEL_FEATURE_COLUMNS
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({col: np.random.randn(n) for col in MODEL_FEATURE_COLUMNS})
    y = pd.Series(np.random.choice([0, 1, 2], size=n))
    return X, y


@pytest.fixture
def trained_model(synthetic_training_data):
    """A small fitted FourthDownModel for testing."""
    from ml.model import FourthDownModel
    X, y = synthetic_training_data
    model = FourthDownModel(hyperparams={"n_estimators": 10})
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def saved_model_dir(trained_model, tmp_path):
    """A trained model saved to a temp directory."""
    from ml.serialize import save_model
    model, X, y = trained_model
    save_model(model, tmp_path / "model")
    return tmp_path / "model"


@pytest.fixture
def sample_game_state():
    """Standard game state dict for prediction tests."""
    return {
        "ydstogo": 3,
        "yardline_100": 35,
        "score_differential": -7,
        "half_seconds_remaining": 600,
        "game_seconds_remaining": 2400,
        "quarter_seconds_remaining": 600,
        "qtr": 3,
        "goal_to_go": 0,
        "wp": 0.35,
    }
