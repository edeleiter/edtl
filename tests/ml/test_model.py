"""Tests for XGBoost model wrapper."""

import numpy as np
import pandas as pd
import pytest

from ml.model import FourthDownModel, DEFAULT_HYPERPARAMS
from nfl.features import MODEL_FEATURE_COLUMNS


@pytest.fixture
def synthetic_training_data():
    """Small synthetic dataset for fast model tests."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({col: np.random.randn(n) for col in MODEL_FEATURE_COLUMNS})
    y = pd.Series(np.random.choice([0, 1, 2], size=n))
    return X, y


def test_model_initializes():
    model = FourthDownModel()
    assert not model.is_fitted


def test_model_fits_and_predicts(synthetic_training_data):
    X, y = synthetic_training_data
    model = FourthDownModel(hyperparams={"n_estimators": 10})
    model.fit(X, y)
    assert model.is_fitted

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert all(p in [0, 1, 2] for p in preds)


def test_model_predict_proba(synthetic_training_data):
    X, y = synthetic_training_data
    model = FourthDownModel(hyperparams={"n_estimators": 10})
    model.fit(X, y)

    probas = model.predict_proba(X)
    assert probas.shape == (len(X), 3)
    # Each row should sum to ~1.0
    row_sums = probas.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_model_feature_importances(synthetic_training_data):
    X, y = synthetic_training_data
    model = FourthDownModel(hyperparams={"n_estimators": 10})
    model.fit(X, y)

    importances = model.feature_importances()
    assert len(importances) == len(MODEL_FEATURE_COLUMNS)
    assert all(isinstance(v, float) for v in importances.values())


def test_default_hyperparams():
    assert "n_estimators" in DEFAULT_HYPERPARAMS
    assert "max_depth" in DEFAULT_HYPERPARAMS
    assert "learning_rate" in DEFAULT_HYPERPARAMS
