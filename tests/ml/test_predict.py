"""Tests for the prediction module."""

import pytest

from ml.predict import FourthDownPredictor


def test_predictor_single(saved_model_dir, sample_game_state):
    predictor = FourthDownPredictor(saved_model_dir)
    result = predictor.predict_from_game_state(sample_game_state)
    assert "recommendation" in result
    assert result["recommendation"] in ("go_for_it", "punt", "field_goal")
    assert "probabilities" in result
    assert len(result["probabilities"]) == 3
    # Probabilities should sum to ~1
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 0.01


def test_predictor_batch(saved_model_dir, sample_game_state):
    predictor = FourthDownPredictor(saved_model_dir)
    states = [sample_game_state, {**sample_game_state, "ydstogo": 10, "yardline_100": 80}]
    results = predictor.predict_batch(states)
    assert len(results) == 2
    for r in results:
        assert r["recommendation"] in ("go_for_it", "punt", "field_goal")
