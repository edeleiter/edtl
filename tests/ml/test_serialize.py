"""Tests for model serialization — save/load round-trip."""

import json

import numpy as np
import pytest

from nfl.features import MODEL_FEATURE_COLUMNS


def test_save_creates_files(saved_model_dir):
    assert (saved_model_dir / "model.joblib").exists()
    assert (saved_model_dir / "metadata.json").exists()


def test_metadata_has_required_keys(saved_model_dir):
    meta = json.loads((saved_model_dir / "metadata.json").read_text())
    assert "version" in meta
    assert "trained_at" in meta
    assert "feature_columns" in meta
    assert meta["feature_columns"] == MODEL_FEATURE_COLUMNS
    assert meta["n_classes"] == 3


def test_load_round_trip(saved_model_dir, synthetic_training_data):
    from ml.serialize import load_model
    model = load_model(saved_model_dir)
    assert model.is_fitted
    X, y = synthetic_training_data
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert all(p in [0, 1, 2] for p in preds)
