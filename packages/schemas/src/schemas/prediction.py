"""Prediction input, output, logging, and outcome models."""

import datetime

import polars as pl
from pydantic import Field, model_validator

from schemas._base import (
    StrictModel,
    TimestampMixin,
    PolarsConvertible,
    Probability,
    LatencyMs,
    Decision,
)
from schemas.game import GameState


class PredictionInput(StrictModel):
    """Validated wrapper around GameState for inference requests."""

    game_state: GameState


class PredictionOutput(StrictModel):
    """Model prediction result with probabilities."""

    recommendation: Decision
    probabilities: dict[Decision, Probability]
    confidence: Probability = Field(description="Max probability across classes")

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> "PredictionOutput":
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Probabilities must sum to 1.0, got {total:.4f}: {self.probabilities}"
            )
        return self

    @model_validator(mode="after")
    def confidence_matches_max_prob(self) -> "PredictionOutput":
        max_prob = max(self.probabilities.values())
        if abs(self.confidence - max_prob) > 0.001:
            raise ValueError(
                f"Confidence ({self.confidence}) must match max probability ({max_prob})"
            )
        return self


class PredictionLogEntry(TimestampMixin, StrictModel):
    """A complete prediction request + response, for logging."""

    request_id: str = Field(description="Unique request identifier")
    model_version: str = Field(description="Model version that produced this prediction")
    input: GameState
    output: PredictionOutput
    latency_ms: LatencyMs
    outcome: "PredictionOutcome | None" = None

    @classmethod
    def to_flat_polars(cls, entries: list["PredictionLogEntry"]) -> pl.DataFrame:
        """Flatten nested models into a flat Polars DataFrame for storage."""
        rows = []
        for entry in entries:
            row = {
                "request_id": entry.request_id,
                "created_at": entry.created_at,
                "model_version": entry.model_version,
                "latency_ms": entry.latency_ms,
                **{f"input_{k}": v for k, v in entry.input.model_dump().items()},
                "output_recommendation": entry.output.recommendation.value,
                "output_confidence": entry.output.confidence,
                "output_prob_go": entry.output.probabilities.get(Decision.GO_FOR_IT, 0.0),
                "output_prob_punt": entry.output.probabilities.get(Decision.PUNT, 0.0),
                "output_prob_fg": entry.output.probabilities.get(Decision.FIELD_GOAL, 0.0),
            }
            if entry.outcome:
                row["actual_decision"] = entry.outcome.actual_decision.value
                row["was_correct"] = entry.outcome.was_correct
                row["epa_result"] = entry.outcome.epa_result
            else:
                row["actual_decision"] = None
                row["was_correct"] = None
                row["epa_result"] = None
            rows.append(row)
        return pl.DataFrame(rows)


class PredictionOutcome(TimestampMixin, StrictModel):
    """Ground-truth outcome, attached after the fact."""

    request_id: str
    actual_decision: Decision
    was_correct: bool
    epa_result: float | None = None
