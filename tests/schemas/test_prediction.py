import polars as pl
import pytest
from pydantic import ValidationError

from schemas._base import Decision
from schemas.game import GameState
from schemas.prediction import (
    PredictionOutput,
    PredictionLogEntry,
    PredictionOutcome,
)


def _make_game_state():
    return GameState(
        ydstogo=3, yardline_100=35, score_differential=-7,
        half_seconds_remaining=600, game_seconds_remaining=2400,
        quarter_seconds_remaining=600, qtr=3, goal_to_go=0, wp=0.35,
    )


def _make_output():
    return PredictionOutput(
        recommendation=Decision.GO_FOR_IT,
        probabilities={
            Decision.GO_FOR_IT: 0.65,
            Decision.PUNT: 0.20,
            Decision.FIELD_GOAL: 0.15,
        },
        confidence=0.65,
    )


def test_prediction_output_valid():
    output = _make_output()
    assert output.recommendation == Decision.GO_FOR_IT
    assert output.confidence == 0.65


def test_prediction_output_probabilities_must_sum_to_one():
    with pytest.raises(ValidationError):
        PredictionOutput(
            recommendation=Decision.GO_FOR_IT,
            probabilities={
                Decision.GO_FOR_IT: 0.5,
                Decision.PUNT: 0.2,
                Decision.FIELD_GOAL: 0.1,
            },
            confidence=0.5,
        )


def test_prediction_log_entry_creates_with_timestamp():
    entry = PredictionLogEntry(
        request_id="req_abc123",
        model_version="v1",
        input=_make_game_state(),
        output=_make_output(),
        latency_ms=12.5,
    )
    assert entry.request_id == "req_abc123"
    assert entry.created_at is not None


def test_prediction_log_entry_to_flat_polars():
    entries = [
        PredictionLogEntry(
            request_id=f"req_{i}",
            model_version="v1",
            input=_make_game_state(),
            output=_make_output(),
            latency_ms=10.0 + i,
        )
        for i in range(5)
    ]
    df = PredictionLogEntry.to_flat_polars(entries)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 5
    assert "request_id" in df.columns
    assert "input_ydstogo" in df.columns
    assert "output_recommendation" in df.columns


def test_prediction_outcome_valid():
    outcome = PredictionOutcome(
        request_id="req_abc123",
        actual_decision=Decision.GO_FOR_IT,
        was_correct=True,
        epa_result=2.5,
    )
    assert outcome.was_correct is True
