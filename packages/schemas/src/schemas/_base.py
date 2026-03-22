"""Base classes, shared types, and mixins for all schema models."""

import datetime
from enum import Enum
from typing import Annotated, Self

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base for all schema models. Strict validation, no coercion."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
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
    a Polars DataFrame.

    Limitation: to_polars() uses model_dump() which converts nested
    Pydantic models to dicts — Polars represents these as Struct columns.
    For models with nested models, override to_polars() to flatten fields.
    """

    @classmethod
    def to_polars(cls, rows: list[Self]) -> pl.DataFrame:
        if not rows:
            return pl.DataFrame(
                schema={
                    name: _pydantic_type_to_polars(field.annotation)
                    for name, field in cls.model_fields.items()
                }
            )
        return pl.DataFrame([row.model_dump() for row in rows])

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> list[Self]:
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
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Decision(str, Enum):
    GO_FOR_IT = "go_for_it"
    PUNT = "punt"
    FIELD_GOAL = "field_goal"


class DriftSeverity(str, Enum):
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


# --- Helpers ---

def _pydantic_type_to_polars(annotation) -> pl.DataType:
    """Best-effort mapping from Python/Pydantic types to Polars dtypes."""
    from typing import get_args, get_origin
    origin = get_origin(annotation)

    # Unwrap Annotated types
    if origin is Annotated:
        args = get_args(annotation)
        if args:
            annotation = args[0]
            origin = get_origin(annotation)

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
        return pl.Utf8
