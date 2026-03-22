# Central Pydantic Schemas Package — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Plan 9 of 10**

**Goal:** Define every data contract in the unified-etl platform as a Pydantic v2 model in a central `schemas` package, providing runtime validation, consistent serialization (JSON, Arrow IPC, Parquet), auto-generated documentation, and a single source of truth for what every component expects and produces.

**Architecture:** A `schemas` package at `packages/schemas/` contains submodules grouped by domain: `game` (NFL game state), `prediction` (model input/output), `transforms` (pipeline config), `features` (feature metadata and analysis results), `quality` (data quality/drift/rules), `monitoring` (performance tracking, alerts, decay), `interpretation` (plain-English interpretations), and `cache` (cache entry metadata). Every other package imports from `schemas` — no package defines its own data models. All models use Pydantic v2 with strict validation, custom validators where needed, and `.model_dump()` / `.model_validate()` for serialization. Models that represent DataFrame-shaped data include a `.to_polars()` / `.from_polars()` bridge for Arrow IPC compatibility with the cache layer.

**Tech Stack:**
- **Core:** Pydantic v2 (BaseModel, Field, validators, computed fields)
- **Serialization:** JSON (metadata), Arrow IPC via Polars (DataFrames)
- **Settings:** pydantic-settings (environment-based configuration)
- **Testing:** pytest, hypothesis (property-based testing for validators)

**Prerequisite:** None — this plan should be implemented FIRST (before or alongside Plan 1), as all other plans depend on these models. Plans 1-8 should reference `schemas` models instead of defining their own dataclasses.

---

## Design Principles

**1. Validate at the boundary, trust internally.** Pydantic validation runs when data enters the system (API requests, file loads, cache reads) and when data leaves (API responses, cache writes). Internal function calls between trusted modules pass validated model instances without re-validation.

**2. Models are the documentation.** If you want to know what a prediction log entry looks like, read `schemas.prediction.PredictionLogEntry`. Field names, types, descriptions, constraints, and examples are all in one place. No need to trace through code.

**3. Strict by default.** All models use `model_config = ConfigDict(strict=True)` — no silent type coercion. If a field expects `int` and gets `"3"`, it fails. This catches data quality issues early rather than silently converting and producing wrong results downstream.

**4. Polars-friendly.** Models that represent tabular data include class methods for round-tripping to/from Polars DataFrames. This bridges the gap between Pydantic's row-oriented validation and Polars's columnar efficiency.

---

## Package Structure

```
packages/schemas/
├── pyproject.toml
└── src/
    └── schemas/
        ├── __init__.py              # Re-exports commonly used models
        ├── _base.py                 # Base classes, shared validators, common types
        │
        ├── game.py                  # NFL game state models
        │   └── GameState, GameContext
        │
        ├── prediction.py            # Prediction input/output/logging
        │   └── PredictionInput, PredictionOutput, PredictionLogEntry,
        │       PredictionOutcome, PredictionBatch
        │
        ├── transforms.py            # Transform pipeline configuration
        │   └── TransformConfig, TransformRequest, TransformResponse,
        │       PipelineDefinition, ReferenceDataManifest
        │
        ├── features.py              # Feature analysis results
        │   └── ImportanceResult, CorrelationResult, VIFResult,
        │       DistributionStats, SubsetEvaluation, FeatureMetadata
        │
        ├── quality.py               # Data quality models
        │   └── ColumnSchema, DataSchema, ValidationError, ValidationResult,
        │       ColumnProfile, DataProfile, ReferenceProfile,
        │       DriftResult, ColumnDrift, RuleDefinition, RuleResult,
        │       QualityReport
        │
        ├── monitoring.py            # Model monitoring models
        │   └── PredictionRecord, PerformanceSnapshot, PerformanceTimeline,
        │       PredictionDriftResult, PSIResult, PSIReport,
        │       SegmentedReport, DecaySignal, RetrainRecommendation
        │
        ├── alerts.py                # Alerting models
        │   └── AlertThresholds, AlertRule, Alert, AlertHistory
        │
        ├── interpretation.py        # Interpretation results
        │   └── Interpretation, InterpretationLevel
        │
        ├── cache.py                 # Cache metadata models
        │   └── CacheEntry, CacheStats, RegistryEntry
        │
        ├── config.py                # Application configuration
        │   └── SnowflakeConfig, DuckDBConfig, MonitoringConfig,
        │       CacheConfig, AppConfig
        │
        └── api.py                   # API request/response models
            └── HealthResponse, TransformAPIRequest, TransformAPIResponse,
                FourthDownRequest, FourthDownResponse, BatchRequest,
                BatchResponse

```

---

## Task 1: Package Scaffolding + Base Classes

**Files:**
- Create: `packages/schemas/pyproject.toml`
- Create: `packages/schemas/src/schemas/__init__.py`
- Create: `packages/schemas/src/schemas/_base.py`
- Modify: `pyproject.toml` (add to workspace)
- Test: `tests/schemas/test_base.py`

**Step 1: Create schemas package pyproject.toml**

```toml
# packages/schemas/pyproject.toml
[project]
name = "schemas"
version = "0.1.0"
description = "Central Pydantic v2 schemas for the unified-etl platform"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6.0",
    "pydantic-settings>=2.0.0",
    "polars>=1.0.0",
    "pyarrow>=14.0.0",
]
```

Add `"packages/schemas"` to `[tool.uv.workspace] members` and `"packages/schemas/src"` to `[tool.pytest.ini_options] pythonpath` in root `pyproject.toml`.

Also add `schemas` as a dependency in every other package's `pyproject.toml`:

```toml
# In each of: transforms, backends, api, dashboard, nfl, ml,
# data_quality, monitoring, cache
dependencies = [
    "schemas",
    # ... existing deps
]

[tool.uv.sources]
schemas = { workspace = true }
```

**Step 2: Write the failing test**

```python
# tests/schemas/test_base.py
import datetime

import polars as pl
import pytest
from pydantic import ValidationError as PydanticValidationError

from schemas._base import (
    StrictModel,
    TimestampMixin,
    PolarsConvertible,
    Probability,
    PositiveInt,
    YardLine,
    Quarter,
    Severity,
)


def test_strict_model_rejects_wrong_type():
    class Example(StrictModel):
        value: int

    with pytest.raises(PydanticValidationError):
        Example(value="not_an_int")


def test_strict_model_accepts_correct_type():
    class Example(StrictModel):
        value: int

    obj = Example(value=42)
    assert obj.value == 42


def test_timestamp_mixin_auto_populates():
    class Example(TimestampMixin, StrictModel):
        name: str

    obj = Example(name="test")
    assert obj.created_at is not None
    assert isinstance(obj.created_at, datetime.datetime)
    assert obj.created_at.tzinfo is not None  # Must be timezone-aware


def test_probability_type_validates_range():
    class Example(StrictModel):
        p: Probability

    obj = Example(p=0.5)
    assert obj.p == 0.5

    with pytest.raises(PydanticValidationError):
        Example(p=1.5)

    with pytest.raises(PydanticValidationError):
        Example(p=-0.1)


def test_yard_line_validates_range():
    class Example(StrictModel):
        yl: YardLine

    obj = Example(yl=50)
    assert obj.yl == 50

    with pytest.raises(PydanticValidationError):
        Example(yl=0)

    with pytest.raises(PydanticValidationError):
        Example(yl=100)


def test_quarter_validates_range():
    class Example(StrictModel):
        q: Quarter

    Example(q=1)
    Example(q=4)
    Example(q=5)  # Overtime

    with pytest.raises(PydanticValidationError):
        Example(q=0)

    with pytest.raises(PydanticValidationError):
        Example(q=6)


def test_severity_enum():
    assert Severity.ERROR.value == "error"
    assert Severity.WARNING.value == "warning"
    assert Severity.INFO.value == "info"
    assert Severity.OK.value == "ok"


def test_polars_convertible_round_trip():
    class Row(PolarsConvertible, StrictModel):
        name: str
        value: float

    rows = [Row(name="a", value=1.0), Row(name="b", value=2.0)]
    df = Row.to_polars(rows)

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2
    assert df.columns == ["name", "value"]

    restored = Row.from_polars(df)
    assert len(restored) == 2
    assert restored[0].name == "a"
    assert restored[1].value == 2.0
```

**Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/schemas/test_base.py -v
```

Expected: FAIL.

**Step 4: Implement base classes**

```python
# packages/schemas/src/schemas/_base.py
"""Base classes, shared types, and mixins for all schema models.

Every model in the schemas package inherits from StrictModel,
which enforces strict type checking — no silent coercion.

Common constrained types (Probability, YardLine, Quarter, etc.)
are defined here so validators are written once and reused
across all domain models.
"""

import datetime
from enum import Enum
from typing import Annotated, ClassVar, Self

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base for all schema models. Strict validation, no coercion."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,              # Immutable by default
        extra="forbid",           # No unknown fields
        ser_json_timedelta="float",
        ser_json_bytes="base64",
        validate_default=True,
    )


class MutableModel(BaseModel):
    """Base for models that need mutation (e.g., builders, accumulators)."""

    model_config = ConfigDict(
        strict=True,
        frozen=False,
        extra="forbid",
        validate_default=True,
    )


class TimestampMixin(BaseModel):
    """Mixin that auto-populates a timezone-aware created_at field."""

    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="UTC timestamp when this record was created",
    )


class PolarsConvertible(BaseModel):
    """Mixin for models that represent tabular rows.

    Provides class methods to convert a list of model instances to/from
    a Polars DataFrame. This bridges Pydantic's row-oriented validation
    with Polars's columnar efficiency.
    """

    @classmethod
    def to_polars(cls, rows: list[Self]) -> pl.DataFrame:
        """Convert a list of model instances to a Polars DataFrame."""
        if not rows:
            # Return empty DataFrame with correct schema
            return pl.DataFrame(
                schema={
                    name: _pydantic_type_to_polars(field.annotation)
                    for name, field in cls.model_fields.items()
                }
            )
        return pl.DataFrame([row.model_dump() for row in rows])

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> list[Self]:
        """Convert a Polars DataFrame to a list of model instances."""
        return [cls.model_validate(row) for row in df.to_dicts()]


# --- Constrained Types ---

Probability = Annotated[float, Field(ge=0.0, le=1.0, description="Probability value [0, 1]")]

PositiveInt = Annotated[int, Field(gt=0)]

NonNegativeInt = Annotated[int, Field(ge=0)]

NonNegativeFloat = Annotated[float, Field(ge=0.0)]

YardLine = Annotated[int, Field(ge=1, le=99, description="Yards from opponent end zone (1-99)")]

YardsToGo = Annotated[int, Field(ge=1, le=99, description="Yards needed for first down (1-99)")]

Quarter = Annotated[int, Field(ge=1, le=5, description="Game quarter (1-4, 5=OT)")]

GameSeconds = Annotated[int, Field(ge=0, le=3600, description="Game seconds remaining (0-3600)")]

HalfSeconds = Annotated[int, Field(ge=0, le=1800, description="Half seconds remaining (0-1800)")]

QuarterSeconds = Annotated[int, Field(ge=0, le=900, description="Quarter seconds remaining (0-900)")]

LatencyMs = Annotated[float, Field(ge=0.0, description="Latency in milliseconds")]


# --- Enums ---

class Severity(str, Enum):
    """Severity levels used across quality, alerts, and interpretation."""

    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Decision(str, Enum):
    """4th-down decision options."""

    GO_FOR_IT = "go_for_it"
    PUNT = "punt"
    FIELD_GOAL = "field_goal"


class DriftSeverity(str, Enum):
    """Drift detection severity."""

    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


# --- Helpers ---

def _pydantic_type_to_polars(annotation) -> pl.DataType:
    """Best-effort mapping from Python/Pydantic types to Polars dtypes."""
    import typing

    origin = getattr(annotation, "__origin__", None)

    # Unwrap Annotated types
    if origin is Annotated:
        args = getattr(annotation, "__args__", ())
        if args:
            annotation = args[0]
            origin = getattr(annotation, "__origin__", None)

    if annotation is int:
        return pl.Int64
    elif annotation is float:
        return pl.Float64
    elif annotation is str:
        return pl.Utf8
    elif annotation is bool:
        return pl.Boolean
    elif annotation is datetime.datetime:
        return pl.Datetime("us", "UTC")
    elif annotation is datetime.date:
        return pl.Date
    else:
        return pl.Utf8  # Fallback: serialize as string
```

**Step 5: Create __init__.py with re-exports**

```python
# packages/schemas/src/schemas/__init__.py
"""Central schema package — import commonly used models from here.

Usage:
    from schemas import GameState, PredictionOutput, Severity
    from schemas.monitoring import PerformanceSnapshot
    from schemas.quality import QualityReport
"""

from schemas._base import (
    StrictModel,
    MutableModel,
    TimestampMixin,
    PolarsConvertible,
    Probability,
    PositiveInt,
    NonNegativeInt,
    NonNegativeFloat,
    YardLine,
    YardsToGo,
    Quarter,
    GameSeconds,
    HalfSeconds,
    QuarterSeconds,
    LatencyMs,
    Severity,
    Decision,
    DriftSeverity,
)

__all__ = [
    "StrictModel",
    "MutableModel",
    "TimestampMixin",
    "PolarsConvertible",
    "Probability",
    "PositiveInt",
    "NonNegativeInt",
    "NonNegativeFloat",
    "YardLine",
    "YardsToGo",
    "Quarter",
    "GameSeconds",
    "HalfSeconds",
    "QuarterSeconds",
    "LatencyMs",
    "Severity",
    "Decision",
    "DriftSeverity",
]
```

**Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/schemas/test_base.py -v
```

Expected: All PASS.

**Step 7: Commit**

```bash
git add -A
git commit -m "feat(schemas): add base classes, constrained types, and enums"
```

---

## Task 2: Game State + Prediction Models

These are the most cross-cutting models — used by NFL, ML, API, cache, and monitoring.

**Files:**
- Create: `packages/schemas/src/schemas/game.py`
- Create: `packages/schemas/src/schemas/prediction.py`
- Test: `tests/schemas/test_game.py`
- Test: `tests/schemas/test_prediction.py`

**Step 1: Write the failing tests**

```python
# tests/schemas/test_game.py
import pytest
from pydantic import ValidationError

from schemas.game import GameState, GameContext


def test_game_state_valid():
    gs = GameState(
        ydstogo=3, yardline_100=35, score_differential=-7,
        half_seconds_remaining=600, game_seconds_remaining=2400,
        quarter_seconds_remaining=600, qtr=3, goal_to_go=0, wp=0.35,
    )
    assert gs.ydstogo == 3
    assert gs.wp == 0.35


def test_game_state_validates_ranges():
    with pytest.raises(ValidationError):
        GameState(
            ydstogo=0, yardline_100=35, score_differential=-7,  # ydstogo must be >= 1
            half_seconds_remaining=600, game_seconds_remaining=2400,
            quarter_seconds_remaining=600, qtr=3, goal_to_go=0, wp=0.35,
        )

    with pytest.raises(ValidationError):
        GameState(
            ydstogo=3, yardline_100=35, score_differential=-7,
            half_seconds_remaining=600, game_seconds_remaining=2400,
            quarter_seconds_remaining=600, qtr=6, goal_to_go=0, wp=0.35,  # qtr must be 1-5
        )


def test_game_state_validates_wp():
    with pytest.raises(ValidationError):
        GameState(
            ydstogo=3, yardline_100=35, score_differential=-7,
            half_seconds_remaining=600, game_seconds_remaining=2400,
            quarter_seconds_remaining=600, qtr=3, goal_to_go=0, wp=1.5,  # wp must be 0-1
        )


def test_game_state_to_dict():
    gs = GameState(
        ydstogo=3, yardline_100=35, score_differential=-7,
        half_seconds_remaining=600, game_seconds_remaining=2400,
        quarter_seconds_remaining=600, qtr=3, goal_to_go=0, wp=0.35,
    )
    d = gs.model_dump()
    assert d["ydstogo"] == 3
    assert len(d) == 9


def test_game_context_extends_game_state():
    ctx = GameContext(
        ydstogo=3, yardline_100=35, score_differential=-7,
        half_seconds_remaining=600, game_seconds_remaining=2400,
        quarter_seconds_remaining=600, qtr=3, goal_to_go=0, wp=0.35,
        game_id="2023_01_KC_DET", season=2023, week=1,
        posteam="KC", defteam="DET",
    )
    assert ctx.game_id == "2023_01_KC_DET"
    assert ctx.posteam == "KC"
```

```python
# tests/schemas/test_prediction.py
import datetime

import polars as pl
import pytest
from pydantic import ValidationError

from schemas.prediction import (
    PredictionInput,
    PredictionOutput,
    PredictionLogEntry,
    PredictionOutcome,
)
from schemas._base import Decision


def test_prediction_output_valid():
    output = PredictionOutput(
        recommendation=Decision.GO_FOR_IT,
        probabilities={
            Decision.GO_FOR_IT: 0.65,
            Decision.PUNT: 0.20,
            Decision.FIELD_GOAL: 0.15,
        },
        confidence=0.65,
    )
    assert output.recommendation == Decision.GO_FOR_IT
    assert output.confidence == 0.65


def test_prediction_output_probabilities_must_sum_to_one():
    with pytest.raises(ValidationError):
        PredictionOutput(
            recommendation=Decision.GO_FOR_IT,
            probabilities={
                Decision.GO_FOR_IT: 0.5,
                Decision.PUNT: 0.2,
                Decision.FIELD_GOAL: 0.1,  # Sums to 0.8, not 1.0
            },
            confidence=0.5,
        )


def test_prediction_log_entry_creates_with_timestamp():
    from schemas.game import GameState

    entry = PredictionLogEntry(
        request_id="req_abc123",
        model_version="v1",
        input=GameState(
            ydstogo=3, yardline_100=35, score_differential=-7,
            half_seconds_remaining=600, game_seconds_remaining=2400,
            quarter_seconds_remaining=600, qtr=3, goal_to_go=0, wp=0.35,
        ),
        output=PredictionOutput(
            recommendation=Decision.GO_FOR_IT,
            probabilities={
                Decision.GO_FOR_IT: 0.65,
                Decision.PUNT: 0.20,
                Decision.FIELD_GOAL: 0.15,
            },
            confidence=0.65,
        ),
        latency_ms=12.5,
    )
    assert entry.request_id == "req_abc123"
    assert entry.created_at is not None


def test_prediction_log_entry_to_polars():
    from schemas.game import GameState

    entries = [
        PredictionLogEntry(
            request_id=f"req_{i}",
            model_version="v1",
            input=GameState(
                ydstogo=3, yardline_100=35, score_differential=-7,
                half_seconds_remaining=600, game_seconds_remaining=2400,
                quarter_seconds_remaining=600, qtr=3, goal_to_go=0, wp=0.35,
            ),
            output=PredictionOutput(
                recommendation=Decision.GO_FOR_IT,
                probabilities={
                    Decision.GO_FOR_IT: 0.65,
                    Decision.PUNT: 0.20,
                    Decision.FIELD_GOAL: 0.15,
                },
                confidence=0.65,
            ),
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
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/schemas/test_game.py tests/schemas/test_prediction.py -v
```

Expected: FAIL.

**Step 3: Implement game models**

```python
# packages/schemas/src/schemas/game.py
"""NFL game state models.

GameState is the core input to the 4th-down model — the 9 fields
that define a game situation. GameContext extends it with metadata
(game_id, season, teams) for logging and analysis.
"""

from pydantic import Field

from schemas._base import (
    StrictModel,
    YardLine,
    YardsToGo,
    Quarter,
    GameSeconds,
    HalfSeconds,
    QuarterSeconds,
    Probability,
)


class GameState(StrictModel):
    """The 9 features that define a 4th-down game situation.

    This is the model's input contract — every inference request
    must provide exactly these fields with valid ranges.
    """

    ydstogo: YardsToGo = Field(description="Yards needed for first down")
    yardline_100: YardLine = Field(description="Yards from opponent end zone")
    score_differential: int = Field(description="Positive = leading, negative = trailing")
    half_seconds_remaining: HalfSeconds
    game_seconds_remaining: GameSeconds
    quarter_seconds_remaining: QuarterSeconds
    qtr: Quarter
    goal_to_go: int = Field(ge=0, le=1, description="1 if goal-to-go, 0 otherwise")
    wp: Probability = Field(description="Pre-play win probability")


class GameContext(GameState):
    """GameState extended with metadata for logging and analysis.

    Used in prediction logs and training data where you need to
    know which game, season, and teams are involved.
    """

    game_id: str = Field(description="Unique game identifier (e.g., 2023_01_KC_DET)")
    season: int = Field(ge=2000, le=2100, description="NFL season year")
    week: int = Field(ge=1, le=22, description="Season week (1-18 regular, 19-22 playoffs)")
    posteam: str = Field(min_length=2, max_length=3, description="Possessing team abbreviation")
    defteam: str = Field(min_length=2, max_length=3, description="Defending team abbreviation")
```

**Step 4: Implement prediction models**

```python
# packages/schemas/src/schemas/prediction.py
"""Prediction input, output, logging, and outcome models.

These models define the contract between the API, the model,
the cache, and the monitoring system. A prediction flows through:
  API request → PredictionInput → model → PredictionOutput → PredictionLogEntry → monitoring
"""

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
    """Validated wrapper around GameState for inference requests.

    Currently identical to GameState, but separated so the API
    contract can evolve independently of the game state definition.
    """

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
    """A complete prediction request + response, for logging.

    This is what gets written to the prediction log (cache/Parquet)
    and consumed by the monitoring dashboard.
    """

    request_id: str = Field(description="Unique request identifier")
    model_version: str = Field(description="Model version that produced this prediction")
    input: GameState
    output: PredictionOutput
    latency_ms: LatencyMs
    outcome: "PredictionOutcome | None" = None

    @classmethod
    def to_flat_polars(cls, entries: list["PredictionLogEntry"]) -> pl.DataFrame:
        """Flatten nested models into a flat Polars DataFrame for storage.

        Prefixes nested field names: input.ydstogo → input_ydstogo,
        output.recommendation → output_recommendation.
        """
        rows = []
        for entry in entries:
            row = {
                "request_id": entry.request_id,
                "created_at": entry.created_at,
                "model_version": entry.model_version,
                "latency_ms": entry.latency_ms,
                # Flatten input
                **{f"input_{k}": v for k, v in entry.input.model_dump().items()},
                # Flatten output
                "output_recommendation": entry.output.recommendation.value,
                "output_confidence": entry.output.confidence,
                "output_prob_go": entry.output.probabilities.get(Decision.GO_FOR_IT, 0.0),
                "output_prob_punt": entry.output.probabilities.get(Decision.PUNT, 0.0),
                "output_prob_fg": entry.output.probabilities.get(Decision.FIELD_GOAL, 0.0),
            }
            # Flatten outcome if present
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
```

**Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/schemas/test_game.py tests/schemas/test_prediction.py -v
```

Expected: All PASS.

**Step 6: Commit**

```bash
git add -A
git commit -m "feat(schemas): add game state + prediction models with validators"
```

---

## Task 3: Transform + Feature Analysis Models

**Files:**
- Create: `packages/schemas/src/schemas/transforms.py`
- Create: `packages/schemas/src/schemas/features.py`
- Test: `tests/schemas/test_transforms.py`
- Test: `tests/schemas/test_features.py`

**Step 1: Write the failing tests**

```python
# tests/schemas/test_transforms.py
import pytest
from pydantic import ValidationError

from schemas.transforms import (
    TransformStep,
    PipelineDefinition,
    TransformRequest,
    TransformResponse,
    ReferenceDataManifest,
)


def test_transform_step_valid():
    step = TransformStep(
        name="clip_outliers",
        params={"column": "value", "lower": 0.0, "upper": 100.0},
    )
    assert step.name == "clip_outliers"


def test_pipeline_definition():
    pipeline = PipelineDefinition(
        name="fourth_down_features",
        steps=[
            TransformStep(name="clip_outliers", params={"column": "value", "lower": 0.0, "upper": 100.0}),
            TransformStep(name="zscore_normalize", params={"column": "value"}),
        ],
        version="1.0.0",
    )
    assert len(pipeline.steps) == 2
    assert pipeline.version == "1.0.0"


def test_transform_request():
    req = TransformRequest(
        data=[{"value": 5.0, "numerator": 10.0, "denominator": 2.0}],
        transforms=["clip_outliers"],
        transform_params={"clip_outliers": {"column": "value", "lower": 0.0, "upper": 10.0}},
    )
    assert len(req.data) == 1


def test_reference_data_manifest():
    manifest = ReferenceDataManifest(
        tables={"category_map": "/data/ref/category_map.parquet"},
    )
    assert "category_map" in manifest.tables
```

```python
# tests/schemas/test_features.py
import numpy as np
import polars as pl
import pytest

from schemas.features import (
    ImportanceResult,
    ImportanceMethod,
    CorrelationPair,
    VIFEntry,
    DistributionStats,
    SubsetEvaluation,
    FeatureMetadata,
)


def test_importance_result():
    result = ImportanceResult(
        method=ImportanceMethod.GINI,
        entries=[
            {"feature": "ydstogo", "importance": 0.35, "std": None},
            {"feature": "yardline_100", "importance": 0.25, "std": None},
        ],
    )
    assert result.method == ImportanceMethod.GINI
    assert len(result.entries) == 2


def test_correlation_pair():
    pair = CorrelationPair(
        feature_a="score_differential",
        feature_b="is_trailing",
        correlation=0.92,
    )
    assert abs(pair.correlation) <= 1.0


def test_correlation_pair_validates_range():
    with pytest.raises(Exception):
        CorrelationPair(feature_a="a", feature_b="b", correlation=1.5)


def test_vif_entry():
    entry = VIFEntry(feature="score_differential", vif=7.3)
    assert entry.vif > 0


def test_distribution_stats():
    stats = DistributionStats(
        column="ydstogo",
        count=500,
        null_count=0,
        null_pct=0.0,
        mean=5.3,
        std=3.2,
        min=1.0,
        max=15.0,
        median=4.0,
        p25=3.0,
        p75=7.0,
    )
    assert stats.column == "ydstogo"
    assert stats.null_pct == 0.0


def test_subset_evaluation():
    ev = SubsetEvaluation(
        name="top_5",
        features=["ydstogo", "yardline_100", "wp", "qtr", "score_differential"],
        n_features=5,
        accuracy=0.83,
        cv_mean=0.81,
        cv_std=0.02,
        cv_scores=[0.79, 0.81, 0.83, 0.81, 0.81],
    )
    assert ev.n_features == 5
    assert len(ev.cv_scores) == 5


def test_feature_metadata():
    meta = FeatureMetadata(
        name="ydstogo",
        dtype="int64",
        description="Yards needed for first down",
        source="raw_pbp",
        is_engineered=False,
    )
    assert not meta.is_engineered
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/schemas/test_transforms.py tests/schemas/test_features.py -v
```

Expected: FAIL.

**Step 3: Implement transforms models**

```python
# packages/schemas/src/schemas/transforms.py
"""Transform pipeline configuration models."""

from typing import Any

from pydantic import Field

from schemas._base import StrictModel


class TransformStep(StrictModel):
    """A single transform step in a pipeline."""

    name: str = Field(description="Registered transform function name")
    params: dict[str, Any] = Field(default_factory=dict, description="Transform parameters")


class PipelineDefinition(StrictModel):
    """An ordered sequence of transforms that defines a feature pipeline."""

    name: str
    steps: list[TransformStep]
    version: str = Field(default="0.1.0")
    description: str = ""


class TransformRequest(StrictModel, frozen=False):
    """API request to apply transforms to input data."""

    data: list[dict[str, Any]]
    transforms: list[str]
    transform_params: dict[str, dict[str, Any]] = Field(default_factory=dict)


class TransformResponse(StrictModel):
    """API response with transformed data."""

    data: list[dict[str, Any]]
    transforms_applied: list[str]


class ReferenceDataManifest(StrictModel):
    """Manifest mapping table names to Parquet file paths."""

    tables: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of table_name → file path",
    )
```

**Step 4: Implement features models**

```python
# packages/schemas/src/schemas/features.py
"""Feature analysis result models — importance, correlation, VIF, distributions."""

from enum import Enum

from pydantic import Field

from schemas._base import StrictModel, Probability


class ImportanceMethod(str, Enum):
    GINI = "gini"
    PERMUTATION = "permutation"
    SHAP = "shap"


class ImportanceEntry(StrictModel):
    """A single feature's importance score."""

    feature: str
    importance: float
    std: float | None = None


class ImportanceResult(StrictModel):
    """Feature importance scores from a single method."""

    method: ImportanceMethod
    entries: list[ImportanceEntry]


class CorrelationPair(StrictModel):
    """A pair of correlated features."""

    feature_a: str
    feature_b: str
    correlation: float = Field(ge=-1.0, le=1.0)


class VIFEntry(StrictModel):
    """Variance Inflation Factor for a single feature."""

    feature: str
    vif: float = Field(ge=1.0, description="VIF ≥ 1.0 (1.0 = no collinearity)")


class DistributionStats(StrictModel):
    """Statistical profile of a single feature column."""

    column: str
    count: int
    null_count: int
    null_pct: float = Field(ge=0.0, le=100.0)
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    p25: float | None = None
    p75: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    unique_count: int | None = None
    histogram_counts: list[int] | None = None
    histogram_bin_edges: list[float] | None = None


class SubsetEvaluation(StrictModel):
    """Evaluation results for a feature subset."""

    name: str
    features: list[str]
    n_features: int
    accuracy: Probability
    cv_mean: float
    cv_std: float = Field(ge=0.0)
    cv_scores: list[float]


class FeatureMetadata(StrictModel):
    """Metadata about a single feature in the model."""

    name: str
    dtype: str
    description: str = ""
    source: str = ""
    is_engineered: bool = False
    transform_name: str | None = None
```

**Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/schemas/test_transforms.py tests/schemas/test_features.py -v
```

Expected: All PASS.

**Step 6: Commit**

```bash
git add -A
git commit -m "feat(schemas): add transform + feature analysis models"
```

---

## Task 4: Data Quality Models

**Files:**
- Create: `packages/schemas/src/schemas/quality.py`
- Test: `tests/schemas/test_quality.py`

**Step 1: Write the failing test**

```python
# tests/schemas/test_quality.py
import pytest
from pydantic import ValidationError

from schemas.quality import (
    ColumnSchema,
    DataSchema,
    ValidationIssue,
    SchemaValidationResult,
    ColumnDrift,
    DriftResult,
    RuleDefinition,
    RuleResult,
    ColumnProfile,
    DataProfile,
    ReferenceProfile,
    QualityReport,
    QualityStatus,
)
from schemas._base import Severity, DriftSeverity


def test_column_schema():
    cs = ColumnSchema(
        name="down",
        dtype="int64",
        nullable=False,
        min_value=1,
        max_value=4,
        is_numeric=True,
    )
    assert cs.name == "down"
    assert not cs.nullable


def test_data_schema():
    ds = DataSchema(
        columns=[
            ColumnSchema(name="down", dtype="int64", nullable=False, is_numeric=True, min_value=1, max_value=4),
            ColumnSchema(name="posteam", dtype="object", nullable=False, allowed_values=["KC", "DET"]),
        ]
    )
    assert len(ds.columns) == 2


def test_validation_issue():
    issue = ValidationIssue(
        column="down",
        message="Value out of range: max=99, expected <= 4",
        severity=Severity.ERROR,
    )
    assert issue.severity == Severity.ERROR


def test_schema_validation_result_pass():
    result = SchemaValidationResult(issues=[])
    assert result.is_valid


def test_schema_validation_result_fail():
    result = SchemaValidationResult(
        issues=[
            ValidationIssue(column="x", message="bad", severity=Severity.ERROR),
        ]
    )
    assert not result.is_valid


def test_column_drift():
    drift = ColumnDrift(
        column="value",
        severity=DriftSeverity.CRITICAL,
        ks_statistic=0.35,
        ks_pvalue=0.0001,
        mean_shift=4.5,
    )
    assert drift.severity == DriftSeverity.CRITICAL


def test_rule_result():
    result = RuleResult(
        rule_name="valid_down",
        column="down",
        severity=Severity.ERROR,
        total_rows=10000,
        failing_rows=3,
        failing_pct=0.03,
        message="3 rows failed",
    )
    assert result.passing_rows == 9997


def test_quality_report():
    report = QualityReport(
        status=QualityStatus.PASS,
        row_count=1000,
        schema_result=SchemaValidationResult(issues=[]),
        rule_results=[],
    )
    assert report.status == QualityStatus.PASS
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/schemas/test_quality.py -v
```

Expected: FAIL.

**Step 3: Implement quality models**

```python
# packages/schemas/src/schemas/quality.py
"""Data quality, drift, and validation models."""

from typing import Any

from pydantic import Field, computed_field

from schemas._base import (
    StrictModel,
    MutableModel,
    TimestampMixin,
    Severity,
    DriftSeverity,
    NonNegativeInt,
    NonNegativeFloat,
)


# --- Schema ---

class ColumnSchema(StrictModel):
    """Expected schema for a single column."""

    name: str
    dtype: str
    nullable: bool = True
    is_numeric: bool = False
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None


class DataSchema(StrictModel):
    """Expected schema for a DataFrame."""

    columns: list[ColumnSchema]
    row_count_min: int | None = None
    row_count_max: int | None = None


# --- Validation ---

class ValidationIssue(StrictModel):
    """A single validation issue found during schema checking."""

    column: str
    message: str
    severity: Severity
    details: dict[str, Any] = Field(default_factory=dict)


class SchemaValidationResult(StrictModel):
    """Result of validating a DataFrame against a schema."""

    issues: list[ValidationIssue] = Field(default_factory=list)

    @computed_field
    @property
    def is_valid(self) -> bool:
        return not any(i.severity in (Severity.ERROR, Severity.CRITICAL) for i in self.issues)

    @computed_field
    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity in (Severity.ERROR, Severity.CRITICAL))

    @computed_field
    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)


# --- Drift ---

class ColumnDrift(StrictModel):
    """Drift analysis for a single column."""

    column: str
    severity: DriftSeverity
    ks_statistic: float | None = None
    ks_pvalue: float | None = None
    mean_shift: float | None = None
    std_ratio: float | None = None
    null_rate_drift: float | None = None
    new_categories: list[str] | None = None
    missing_categories: list[str] | None = None
    category_proportion_drift: float | None = None


class DriftResult(StrictModel):
    """Drift analysis across all columns."""

    columns: dict[str, ColumnDrift] = Field(default_factory=dict)

    @computed_field
    @property
    def has_critical(self) -> bool:
        return any(d.severity == DriftSeverity.CRITICAL for d in self.columns.values())

    @computed_field
    @property
    def has_warnings(self) -> bool:
        return any(d.severity == DriftSeverity.WARNING for d in self.columns.values())


# --- Rules ---

class RuleDefinition(StrictModel):
    """Definition of a business rule (without the callable check)."""

    name: str
    description: str
    column: str
    severity: Severity = Severity.ERROR


class RuleResult(StrictModel):
    """Result of applying a single business rule."""

    rule_name: str
    column: str
    severity: Severity
    total_rows: NonNegativeInt
    failing_rows: NonNegativeInt
    failing_pct: NonNegativeFloat
    message: str
    sample_failing_indices: list[int] = Field(default_factory=list)

    @computed_field
    @property
    def passing_rows(self) -> int:
        return self.total_rows - self.failing_rows


# --- Profiling ---

class ColumnProfile(StrictModel):
    """Statistical profile for a single column."""

    name: str
    dtype: str
    count: NonNegativeInt
    null_count: NonNegativeInt
    null_pct: NonNegativeFloat
    unique_count: NonNegativeInt | None = None
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    p01: float | None = None
    p05: float | None = None
    p25: float | None = None
    p75: float | None = None
    p95: float | None = None
    p99: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    histogram_counts: list[int] | None = None
    histogram_bin_edges: list[float] | None = None
    value_counts: dict[str, int] | None = None


class DataProfile(StrictModel):
    """Statistical profile for an entire DataFrame."""

    row_count: NonNegativeInt
    column_count: NonNegativeInt
    columns: dict[str, ColumnProfile] = Field(default_factory=dict)


class ReferenceProfile(TimestampMixin, StrictModel):
    """Named reference profile snapshot from training data."""

    name: str
    profile: DataProfile


# --- Report ---

class QualityStatus(str, __import__("enum").Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class QualityReport(TimestampMixin, StrictModel):
    """Unified quality report across all validation layers."""

    status: QualityStatus
    row_count: NonNegativeInt
    schema_result: SchemaValidationResult | None = None
    drift_result: DriftResult | None = None
    rule_results: list[RuleResult] = Field(default_factory=list)
    profile: DataProfile | None = None
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/schemas/test_quality.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(schemas): add data quality, drift, profiling, and validation models"
```

---

## Task 5: Monitoring + Alert Models

**Files:**
- Create: `packages/schemas/src/schemas/monitoring.py`
- Create: `packages/schemas/src/schemas/alerts.py`
- Test: `tests/schemas/test_monitoring.py`
- Test: `tests/schemas/test_alerts.py`

> **Note for the engineer:** These tests follow the same pattern as Tasks 2-4. Implement the models based on the dataclass definitions in Plans 7 (model-monitoring-dashboard). Convert every `@dataclass` to a `StrictModel` subclass. Key models: `PredictionRecord`, `PerformanceSnapshot`, `PerformanceTimeline`, `PredictionDriftResult`, `PSIResult`, `PSIReport`, `SegmentedReport`, `DecaySignal`, `RetrainRecommendation`, `AlertThresholds`, `Alert`, `AlertHistory`.

> Tests should validate field constraints, serialization round-trips, and computed properties.

**Step 1:** Write failing tests for all monitoring and alert models.
**Step 2:** Run tests, verify fail.
**Step 3:** Implement `schemas/monitoring.py` with all monitoring models from Plan 7.
**Step 4:** Implement `schemas/alerts.py` with alert models from Plan 7.
**Step 5:** Run tests, verify pass.
**Step 6:** Commit: `feat(schemas): add monitoring + alert models`

---

## Task 6: Interpretation + Cache + Config Models

**Files:**
- Create: `packages/schemas/src/schemas/interpretation.py`
- Create: `packages/schemas/src/schemas/cache.py`
- Create: `packages/schemas/src/schemas/config.py`
- Create: `packages/schemas/src/schemas/api.py`
- Test: `tests/schemas/test_interpretation.py`
- Test: `tests/schemas/test_cache.py`
- Test: `tests/schemas/test_config.py`

**Step 1:** Implement `schemas/interpretation.py`:

```python
# packages/schemas/src/schemas/interpretation.py
"""Interpretation models — shared across feature analysis and data quality."""

from schemas._base import StrictModel, Severity


# Reuse Severity enum levels for interpretation
# OK, INFO, WARNING, ERROR, CRITICAL — single definition, used everywhere
InterpretationLevel = Severity


class Interpretation(StrictModel):
    """A structured plain-English interpretation of a statistical result."""

    level: InterpretationLevel
    headline: str
    explanation: str
    guidance: str
    metric_name: str = ""
    metric_value: float | str | None = None
    tooltip: str | None = None
```

> This fixes the duplicate `InterpretationLevel` issue from the consistency review — one definition, imported everywhere.

**Step 2:** Implement `schemas/cache.py`:

```python
# packages/schemas/src/schemas/cache.py
"""Cache metadata models for the Dragonfly persistence layer."""

import datetime

from pydantic import Field

from schemas._base import StrictModel, TimestampMixin, NonNegativeInt


class CacheEntry(TimestampMixin, StrictModel):
    """Metadata about a cached value in Dragonfly."""

    key: str
    size_bytes: NonNegativeInt
    ttl_seconds: int | None = None
    content_hash: str | None = None
    data_type: str = Field(description="'dataframe', 'value', or 'raw'")
    source: str = Field(default="", description="How this was computed")


class CacheStats(StrictModel):
    """Overall cache statistics."""

    total_entries: NonNegativeInt
    total_size_bytes: NonNegativeInt
    hit_count: NonNegativeInt = 0
    miss_count: NonNegativeInt = 0
    eviction_count: NonNegativeInt = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


class RegistryEntry(StrictModel):
    """A registered data source in the cache registry."""

    pattern: str = Field(description="Key pattern with {param} placeholders")
    ttl_seconds: int
    description: str = ""
```

**Step 3:** Implement `schemas/config.py`:

```python
# packages/schemas/src/schemas/config.py
"""Application configuration models using pydantic-settings."""

from pydantic_settings import BaseSettings


class SnowflakeConfig(BaseSettings):
    model_config = {"env_prefix": "SNOWFLAKE_"}

    account: str = ""
    user: str = ""
    password: str = ""
    database: str = ""
    schema_name: str = "PUBLIC"
    warehouse: str = ""
    role: str = ""


class DuckDBConfig(BaseSettings):
    model_config = {"env_prefix": "DUCKDB_"}

    database: str = ":memory:"
    threads: int = 4


class CacheConfig(BaseSettings):
    model_config = {"env_prefix": "CACHE_"}

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    max_memory_mb: int = 4096
    default_ttl_seconds: int = 3600


class MonitoringConfig(BaseSettings):
    model_config = {"env_prefix": "MONITORING_"}

    data_dir: str = "monitoring_data/predictions"
    alert_accuracy_warning: float = 0.70
    alert_accuracy_critical: float = 0.60
    alert_latency_p95_warning: float = 100.0
    alert_latency_p95_critical: float = 500.0


class AppConfig(BaseSettings):
    """Top-level application config composing all sub-configs."""

    model_config = {"env_prefix": "APP_"}

    model_dir: str = "models/latest"
    reference_data_dir: str = "reference_data"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 2
```

**Step 4:** Implement `schemas/api.py`:

```python
# packages/schemas/src/schemas/api.py
"""FastAPI request/response models.

These are the external-facing API contracts. They wrap internal
schema models for HTTP serialization.
"""

from pydantic import Field

from schemas._base import StrictModel, Decision
from schemas.game import GameState
from schemas.prediction import PredictionOutput


class HealthResponse(StrictModel):
    status: str
    backend: str
    model_version: str | None = None


class FourthDownRequest(GameState):
    """API request body for 4th-down prediction (inherits GameState fields)."""

    pass


class FourthDownResponse(StrictModel):
    """API response for 4th-down prediction."""

    recommendation: str
    probabilities: dict[str, float]
    request_id: str | None = None


class FourthDownBatchRequest(StrictModel):
    game_states: list[GameState]


class FourthDownBatchResponse(StrictModel):
    predictions: list[FourthDownResponse]
```

**Step 5:** Write tests, run, verify pass.
**Step 6:** Commit: `feat(schemas): add interpretation, cache, config, and API models`

---

## Task 7: Update __init__.py with Full Re-Exports + Cross-Plan Reference Guide

**Files:**
- Modify: `packages/schemas/src/schemas/__init__.py`
- Create: `packages/schemas/MIGRATION.md`

**Step 1: Update __init__.py**

Add imports for all commonly used models so consumers can do `from schemas import GameState, PredictionOutput, Severity`.

**Step 2: Create migration guide**

```markdown
# Schemas Migration Guide

This document maps old patterns (from Plans 1-8) to the new schemas package.

## Where to Import From

| Old import | New import |
|-----------|-----------|
| `from api.models import GameState` | `from schemas.game import GameState` |
| `from api.models import TransformRequest` | `from schemas.transforms import TransformRequest` |
| `from monitoring.schemas import PredictionLog` | `from schemas.prediction import PredictionLogEntry` |
| `from data_quality.schema import ValidationError` | `from schemas.quality import ValidationIssue` |
| `from data_quality.schema import Severity` | `from schemas import Severity` |
| `from data_quality.drift import DriftSeverity` | `from schemas import DriftSeverity` |
| `from ml.feature_analysis.interpret import Interpretation` | `from schemas.interpretation import Interpretation` |
| `from ml.feature_analysis.interpret import InterpretationLevel` | `from schemas.interpretation import InterpretationLevel` |
| `from ml.evaluate import EvaluationReport` | `from schemas.monitoring import EvaluationReport` |
| `from backends.config import SnowflakeConfig` | `from schemas.config import SnowflakeConfig` |

## Dataclass → Pydantic Conversion

Every `@dataclass` in Plans 1-8 becomes a `StrictModel` subclass:

```python
# Old (Plans 1-8)
@dataclass
class ImportanceResult:
    method: str
    feature_names: list[str]
    scores: np.ndarray

# New (schemas package)
class ImportanceResult(StrictModel):
    method: ImportanceMethod
    entries: list[ImportanceEntry]  # Structured, validated
```

## Key Changes

1. **Raw dicts → Pydantic models:** `{"ydstogo": 3, ...}` → `GameState(ydstogo=3, ...)`
2. **Dataclasses → StrictModel:** All `@dataclass` → `class X(StrictModel)`
3. **Pandas → Polars:** `pd.DataFrame` → `pl.DataFrame` (see Plan 10)
4. **Loose types → constrained types:** `wp: float` → `wp: Probability` (validates 0-1)
5. **Duplicate definitions → single source:** `InterpretationLevel` defined once in `schemas.interpretation`
```

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(schemas): finalize re-exports and migration guide"
```

---

## Summary: Run Order

| Task | What it builds | Key files |
|------|---------------|-----------|
| 1 | Base classes + constrained types + enums | `_base.py` |
| 2 | Game state + prediction models | `game.py`, `prediction.py` |
| 3 | Transform + feature analysis models | `transforms.py`, `features.py` |
| 4 | Data quality models | `quality.py` |
| 5 | Monitoring + alert models | `monitoring.py`, `alerts.py` |
| 6 | Interpretation + cache + config + API models | `interpretation.py`, `cache.py`, `config.py`, `api.py` |
| 7 | Re-exports + migration guide | `__init__.py`, `MIGRATION.md` |

## Model Inventory (42 models)

| Domain | Models | Module |
|--------|--------|--------|
| **Base** (5) | `StrictModel`, `MutableModel`, `TimestampMixin`, `PolarsConvertible`, enums | `_base.py` |
| **Game** (2) | `GameState`, `GameContext` | `game.py` |
| **Prediction** (4) | `PredictionInput`, `PredictionOutput`, `PredictionLogEntry`, `PredictionOutcome` | `prediction.py` |
| **Transforms** (5) | `TransformStep`, `PipelineDefinition`, `TransformRequest`, `TransformResponse`, `ReferenceDataManifest` | `transforms.py` |
| **Features** (7) | `ImportanceResult`, `ImportanceEntry`, `CorrelationPair`, `VIFEntry`, `DistributionStats`, `SubsetEvaluation`, `FeatureMetadata` | `features.py` |
| **Quality** (12) | `ColumnSchema`, `DataSchema`, `ValidationIssue`, `SchemaValidationResult`, `ColumnDrift`, `DriftResult`, `RuleDefinition`, `RuleResult`, `ColumnProfile`, `DataProfile`, `ReferenceProfile`, `QualityReport` | `quality.py` |
| **Monitoring** (5+) | `PerformanceSnapshot`, `PerformanceTimeline`, `PSIResult`, `DecaySignal`, `RetrainRecommendation` | `monitoring.py` |
| **Alerts** (3) | `AlertThresholds`, `Alert`, `AlertHistory` | `alerts.py` |
| **Interpretation** (1) | `Interpretation` (+ reuses `Severity` as `InterpretationLevel`) | `interpretation.py` |
| **Cache** (3) | `CacheEntry`, `CacheStats`, `RegistryEntry` | `cache.py` |
| **Config** (5) | `SnowflakeConfig`, `DuckDBConfig`, `CacheConfig`, `MonitoringConfig`, `AppConfig` | `config.py` |
| **API** (5) | `HealthResponse`, `FourthDownRequest`, `FourthDownResponse`, `FourthDownBatchRequest`, `FourthDownBatchResponse` | `api.py` |

All Plans 1-8 should reference `schemas` models instead of defining their own. The `MIGRATION.md` guide maps every old import to its new location.
