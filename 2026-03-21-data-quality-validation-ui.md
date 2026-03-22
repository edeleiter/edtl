# Data Quality & Validation UI — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a data quality library and Streamlit UI that validates incoming data against learned schemas, detects anomalies (drift, outliers, missing data spikes), enforces business rules, and produces actionable quality reports — so that bad data is caught before it enters the training or inference pipelines.

**Architecture:** A `data_quality` package provides three layers: schema validation (column types, nullability, value ranges learned from training data), statistical profiling (distribution drift detection via KL divergence and KS tests, outlier flagging, missing data analysis), and business rules (NFL-specific constraints like valid yard lines, valid quarters, etc.). Each layer returns structured `ValidationResult` objects that the Streamlit pages render as red/yellow/green dashboards. The profiling layer uses Ibis for pushdown-compatible statistics. A `ReferenceProfile` captures the "expected" distribution from training data and is snapshotted alongside the model.

**Tech Stack:**
- **Validation:** pandera (schema validation), scipy (statistical tests)
- **Profiling:** Ibis + DuckDB (statistics), numpy (drift computation)
- **Visualization:** Plotly, Streamlit
- **Inherits:** Everything from unified-etl + NFL ML pipeline + feature selection plans

**Prerequisite:** The unified-etl framework (Plan 1) and NFL ML pipeline (Plan 2) must be implemented first. Feature selection (Plan 3) is optional.

---

## Context: Why Data Quality Matters for ML Pipelines

Data quality issues are the #1 cause of ML model degradation in production. Common failure modes:

- **Schema violations:** A new season's data has a renamed column, a changed type, or drops a field entirely. The pipeline silently produces nulls or crashes.
- **Distribution drift:** Offensive strategies change year over year. If 4th-down attempt rates shift, the model's training distribution no longer matches inference inputs.
- **Outliers:** A data error produces `yardline_100 = 200` or `score_differential = 999`. Without validation, these flow through and distort predictions.
- **Missing data spikes:** A data provider drops a field mid-season. Null rates jump from 0.1% to 40%. The model fills nulls with zeros and starts hallucinating.
- **Business rule violations:** `down = 5`, negative game seconds, or a punt from the opponent's 5-yard line — physically impossible but data entry errors happen.

This plan builds a "data firewall" that catches all of these before data enters the pipeline.

---

## Extended Project Structure

```
unified-etl/
├── packages/
│   ├── data_quality/                       # Data quality library
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── data_quality/
│   │           ├── __init__.py
│   │           ├── schema.py               # Schema validation + learning
│   │           ├── profiler.py             # Statistical profiling
│   │           ├── drift.py                # Distribution drift detection
│   │           ├── rules.py                # Business rule engine
│   │           ├── report.py               # Validation report builder
│   │           └── reference_profile.py    # Training data profile snapshot
│   │
│   └── dashboard/
│       └── src/
│           └── dashboard/
│               └── pages/
│                   ├── ... (existing)
│                   ├── 08_data_quality_overview.py
│                   ├── 09_drift_monitor.py
│                   └── 10_validation_rules.py
│
└── tests/
    └── data_quality/
        ├── test_schema.py
        ├── test_profiler.py
        ├── test_drift.py
        ├── test_rules.py
        └── test_report.py
```

---

## Task 1: Package Scaffolding + Schema Validation

**Files:**
- Create: `packages/data_quality/pyproject.toml`
- Create: `packages/data_quality/src/data_quality/__init__.py`
- Create: `packages/data_quality/src/data_quality/schema.py`
- Modify: `pyproject.toml` (add to workspace members + pythonpath)
- Test: `tests/data_quality/test_schema.py`

**Step 1: Add data_quality to workspace**

Modify root `pyproject.toml` — add `"packages/data_quality"` to `[tool.uv.workspace] members` and `"packages/data_quality/src"` to `[tool.pytest.ini_options] pythonpath`.

**Step 2: Create data_quality package pyproject.toml**

```toml
# packages/data_quality/pyproject.toml
[project]
name = "data_quality"
version = "0.1.0"
description = "Data quality validation, profiling, and drift detection"
requires-python = ">=3.11"
dependencies = [
    "transforms",
    "backends",
    "ibis-framework[duckdb]>=9.0.0",
    "pandas>=2.0.0",
    "numpy>=1.26.0",
    "scipy>=1.12.0",
    "pyarrow>=14.0.0",
]

[tool.uv.sources]
transforms = { workspace = true }
backends = { workspace = true }
```

**Step 3: Write the failing test**

```python
# tests/data_quality/test_schema.py
import numpy as np
import pandas as pd
import pytest

from data_quality.schema import (
    learn_schema,
    validate_schema,
    DataSchema,
    ColumnSchema,
    SchemaValidationResult,
    Severity,
)


@pytest.fixture
def training_data():
    return pd.DataFrame(
        {
            "down": [1, 2, 3, 4, 4],
            "ydstogo": [10, 7, 3, 1, 8],
            "yardline_100": [75, 68, 30, 45, 80],
            "epa": [0.5, -0.3, 1.2, -0.8, 0.0],
            "posteam": ["KC", "DET", "BUF", "KC", "SF"],
            "qtr": [1, 1, 2, 3, 4],
            "wp": [0.55, 0.52, 0.60, 0.45, 0.30],
        }
    )


def test_learn_schema(training_data):
    schema = learn_schema(training_data)
    assert isinstance(schema, DataSchema)
    assert len(schema.columns) == 7
    assert "down" in schema.columns
    assert schema.columns["down"].dtype == "int64"
    assert schema.columns["down"].nullable is False


def test_schema_captures_ranges(training_data):
    schema = learn_schema(training_data)
    down_schema = schema.columns["down"]
    assert down_schema.min_value == 1
    assert down_schema.max_value == 4
    wp_schema = schema.columns["wp"]
    assert wp_schema.min_value >= 0.0
    assert wp_schema.max_value <= 1.0


def test_schema_captures_categories(training_data):
    schema = learn_schema(training_data)
    posteam_schema = schema.columns["posteam"]
    assert posteam_schema.allowed_values is not None
    assert "KC" in posteam_schema.allowed_values


def test_validate_schema_pass(training_data):
    schema = learn_schema(training_data)
    result = validate_schema(training_data, schema)
    assert isinstance(result, SchemaValidationResult)
    assert result.is_valid
    assert len(result.errors) == 0


def test_validate_schema_missing_column(training_data):
    schema = learn_schema(training_data)
    bad_data = training_data.drop(columns=["epa"])
    result = validate_schema(bad_data, schema)
    assert not result.is_valid
    assert any("epa" in e.message for e in result.errors)


def test_validate_schema_wrong_type(training_data):
    schema = learn_schema(training_data)
    bad_data = training_data.copy()
    bad_data["down"] = bad_data["down"].astype(str)
    result = validate_schema(bad_data, schema)
    assert not result.is_valid


def test_validate_schema_out_of_range(training_data):
    schema = learn_schema(training_data)
    bad_data = training_data.copy()
    bad_data.loc[0, "down"] = 99
    result = validate_schema(bad_data, schema)
    assert not result.is_valid
    assert any(e.severity == Severity.ERROR for e in result.errors)


def test_validate_schema_unexpected_nulls(training_data):
    schema = learn_schema(training_data)
    bad_data = training_data.copy()
    bad_data.loc[0, "down"] = None
    result = validate_schema(bad_data, schema)
    assert not result.is_valid


def test_validate_schema_unknown_category(training_data):
    schema = learn_schema(training_data)
    bad_data = training_data.copy()
    bad_data.loc[0, "posteam"] = "FAKE_TEAM"
    result = validate_schema(bad_data, schema)
    # Unknown category should be a warning, not error
    warnings = [e for e in result.errors if e.severity == Severity.WARNING]
    assert len(warnings) > 0


def test_schema_serialization(training_data):
    schema = learn_schema(training_data)
    d = schema.to_dict()
    restored = DataSchema.from_dict(d)
    assert len(restored.columns) == len(schema.columns)
    assert restored.columns["down"].max_value == schema.columns["down"].max_value
```

**Step 4: Run tests to verify they fail**

```bash
uv run pytest tests/data_quality/test_schema.py -v
```

Expected: FAIL.

**Step 5: Implement schema validation**

```python
# packages/data_quality/src/data_quality/schema.py
"""Schema validation — learn expected structure from training data,
then validate new data against it.

The schema captures:
- Expected columns and their dtypes
- Nullability (was the column ever null in training data?)
- Numeric ranges (observed min/max with configurable margin)
- Categorical allowed values (observed unique values)

This is the first line of defense: catch structural breaks before
any feature engineering or model inference happens.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class Severity(Enum):
    ERROR = "error"      # Hard failure — data cannot proceed
    WARNING = "warning"  # Suspicious — flag for review
    INFO = "info"        # Informational — no action needed


@dataclass
class ValidationError:
    """A single validation issue."""

    column: str
    message: str
    severity: Severity
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaValidationResult:
    """Result of validating a DataFrame against a schema."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == Severity.WARNING)

    def summary(self) -> str:
        if self.is_valid:
            return "Schema validation PASSED"
        lines = [
            f"Schema validation FAILED: {self.error_count} errors, {self.warning_count} warnings"
        ]
        for e in self.errors:
            lines.append(f"  [{e.severity.value.upper()}] {e.column}: {e.message}")
        return "\n".join(lines)


@dataclass
class ColumnSchema:
    """Expected schema for a single column."""

    name: str
    dtype: str  # pandas dtype name
    nullable: bool = True
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None  # For categoricals
    is_numeric: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "nullable": self.nullable,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "allowed_values": self.allowed_values,
            "is_numeric": self.is_numeric,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnSchema":
        return cls(**d)


@dataclass
class DataSchema:
    """Expected schema for a DataFrame."""

    columns: dict[str, ColumnSchema] = field(default_factory=dict)
    row_count_range: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return {
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "row_count_range": self.row_count_range,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DataSchema":
        columns = {k: ColumnSchema.from_dict(v) for k, v in d["columns"].items()}
        return cls(
            columns=columns,
            row_count_range=tuple(d["row_count_range"]) if d.get("row_count_range") else None,
        )


def learn_schema(
    df: pd.DataFrame,
    range_margin: float = 0.1,
    max_categories: int = 100,
) -> DataSchema:
    """Learn a schema from training data.

    Examines each column to determine type, nullability, value ranges,
    and allowed categories. The learned schema represents "what the
    model was trained on" and is the baseline for validation.

    Args:
        df: Training DataFrame.
        range_margin: Fractional margin to add to observed min/max
            for numeric columns (0.1 = ±10%).
        max_categories: Max unique values before treating a string
            column as free-text rather than categorical.
    """
    columns = {}

    for col_name in df.columns:
        col = df[col_name]
        dtype_name = str(col.dtype)
        nullable = bool(col.isna().any())

        col_schema = ColumnSchema(
            name=col_name,
            dtype=dtype_name,
            nullable=nullable,
        )

        if pd.api.types.is_numeric_dtype(col):
            col_schema.is_numeric = True
            col_min = float(col.min())
            col_max = float(col.max())
            margin = abs(col_max - col_min) * range_margin
            col_schema.min_value = col_min - margin
            col_schema.max_value = col_max + margin

        elif pd.api.types.is_string_dtype(col) or pd.api.types.is_object_dtype(col):
            unique_vals = col.dropna().unique().tolist()
            if len(unique_vals) <= max_categories:
                col_schema.allowed_values = sorted(str(v) for v in unique_vals)

        columns[col_name] = col_schema

    return DataSchema(columns=columns)


def validate_schema(
    df: pd.DataFrame,
    schema: DataSchema,
) -> SchemaValidationResult:
    """Validate a DataFrame against a learned schema.

    Checks:
    1. All expected columns are present
    2. No unexpected null values
    3. Numeric values within expected range
    4. Categorical values in allowed set
    5. Data types are compatible
    """
    errors: list[ValidationError] = []

    # Check for missing columns
    for col_name, col_schema in schema.columns.items():
        if col_name not in df.columns:
            errors.append(
                ValidationError(
                    column=col_name,
                    message=f"Missing column '{col_name}' (expected dtype: {col_schema.dtype})",
                    severity=Severity.ERROR,
                )
            )
            continue

        col = df[col_name]

        # Type check (flexible: int vs float is OK)
        if col_schema.is_numeric and not pd.api.types.is_numeric_dtype(col):
            errors.append(
                ValidationError(
                    column=col_name,
                    message=f"Expected numeric type, got {col.dtype}",
                    severity=Severity.ERROR,
                )
            )
            continue

        # Nullability check
        if not col_schema.nullable and col.isna().any():
            null_count = int(col.isna().sum())
            errors.append(
                ValidationError(
                    column=col_name,
                    message=f"Unexpected nulls: {null_count} null values in non-nullable column",
                    severity=Severity.ERROR,
                    details={"null_count": null_count},
                )
            )

        # Range check for numerics
        if col_schema.is_numeric and col_schema.min_value is not None:
            non_null = col.dropna()
            if len(non_null) > 0:
                actual_min = float(non_null.min())
                actual_max = float(non_null.max())
                if actual_min < col_schema.min_value:
                    errors.append(
                        ValidationError(
                            column=col_name,
                            message=f"Value below range: min={actual_min:.4f}, expected >= {col_schema.min_value:.4f}",
                            severity=Severity.ERROR,
                            details={"actual_min": actual_min, "schema_min": col_schema.min_value},
                        )
                    )
                if actual_max > col_schema.max_value:
                    errors.append(
                        ValidationError(
                            column=col_name,
                            message=f"Value above range: max={actual_max:.4f}, expected <= {col_schema.max_value:.4f}",
                            severity=Severity.ERROR,
                            details={"actual_max": actual_max, "schema_max": col_schema.max_value},
                        )
                    )

        # Category check
        if col_schema.allowed_values is not None:
            actual_vals = set(str(v) for v in col.dropna().unique())
            unknown = actual_vals - set(col_schema.allowed_values)
            if unknown:
                errors.append(
                    ValidationError(
                        column=col_name,
                        message=f"Unknown categories: {unknown}",
                        severity=Severity.WARNING,
                        details={"unknown_values": sorted(unknown)},
                    )
                )

    # Check for extra columns (info only)
    expected_cols = set(schema.columns.keys())
    actual_cols = set(df.columns)
    extra = actual_cols - expected_cols
    if extra:
        for col_name in sorted(extra):
            errors.append(
                ValidationError(
                    column=col_name,
                    message=f"Unexpected extra column '{col_name}'",
                    severity=Severity.INFO,
                )
            )

    has_errors = any(e.severity == Severity.ERROR for e in errors)
    return SchemaValidationResult(is_valid=not has_errors, errors=errors)
```

```python
# packages/data_quality/src/data_quality/__init__.py
from data_quality.schema import (
    learn_schema,
    validate_schema,
    DataSchema,
    SchemaValidationResult,
    Severity,
)

__all__ = [
    "learn_schema",
    "validate_schema",
    "DataSchema",
    "SchemaValidationResult",
    "Severity",
]
```

**Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/data_quality/test_schema.py -v
```

Expected: All PASS.

**Step 7: Commit**

```bash
git add -A
git commit -m "feat(data-quality): add schema learning + validation"
```

---

## Task 2: Statistical Profiler + Reference Profile

**Files:**
- Create: `packages/data_quality/src/data_quality/profiler.py`
- Create: `packages/data_quality/src/data_quality/reference_profile.py`
- Test: `tests/data_quality/test_profiler.py`

**Step 1: Write the failing test**

```python
# tests/data_quality/test_profiler.py
import numpy as np
import pandas as pd
import pytest

from data_quality.profiler import (
    profile_dataframe,
    ColumnProfile,
    DataProfile,
)
from data_quality.reference_profile import (
    create_reference_profile,
    save_reference_profile,
    load_reference_profile,
    ReferenceProfile,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "value_a": np.random.normal(10, 2, 500).tolist(),
            "value_b": np.random.exponential(3, 500).tolist(),
            "category": np.random.choice(["X", "Y", "Z"], 500).tolist(),
            "with_nulls": [None if i % 10 == 0 else float(i) for i in range(500)],
        }
    )


def test_profile_dataframe(sample_data):
    profile = profile_dataframe(sample_data)
    assert isinstance(profile, DataProfile)
    assert profile.row_count == 500
    assert len(profile.columns) == 4


def test_column_profile_numeric(sample_data):
    profile = profile_dataframe(sample_data)
    p = profile.columns["value_a"]
    assert isinstance(p, ColumnProfile)
    assert p.dtype == "float64"
    assert p.null_count == 0
    assert p.null_pct == 0.0
    assert abs(p.mean - 10.0) < 1.0  # approximately normal(10, 2)
    assert p.histogram is not None


def test_column_profile_categorical(sample_data):
    profile = profile_dataframe(sample_data)
    p = profile.columns["category"]
    assert p.unique_count == 3
    assert p.value_counts is not None
    assert "X" in p.value_counts


def test_column_profile_with_nulls(sample_data):
    profile = profile_dataframe(sample_data)
    p = profile.columns["with_nulls"]
    assert p.null_count == 50
    assert abs(p.null_pct - 10.0) < 0.5


def test_reference_profile_round_trip(sample_data, tmp_path):
    profile = profile_dataframe(sample_data)
    ref = create_reference_profile(profile, name="training_v1")
    assert isinstance(ref, ReferenceProfile)

    path = tmp_path / "ref_profile.json"
    save_reference_profile(ref, path)
    loaded = load_reference_profile(path)

    assert loaded.name == "training_v1"
    assert loaded.profile.row_count == 500
    assert len(loaded.profile.columns) == 4
    assert loaded.profile.columns["value_a"].mean == profile.columns["value_a"].mean
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/data_quality/test_profiler.py -v
```

Expected: FAIL.

**Step 3: Implement the profiler**

```python
# packages/data_quality/src/data_quality/profiler.py
"""Statistical profiling for DataFrames.

Produces a comprehensive statistical profile of each column:
- Counts: total, null, unique
- Numerics: mean, std, min, max, percentiles, skew, kurtosis, histogram
- Categoricals: value counts, top values
- Missing data patterns

The profile is a snapshot of "what the data looks like" — used as
the baseline for drift detection and the data quality dashboard.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    """Statistical profile for a single column."""

    name: str
    dtype: str
    count: int
    null_count: int
    null_pct: float
    unique_count: int

    # Numeric stats (None for non-numeric columns)
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
    histogram: dict[str, list] | None = None  # {counts, bin_edges}

    # Categorical stats
    value_counts: dict[str, int] | None = None
    top_values: list[str] | None = None

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DataProfile:
    """Statistical profile for an entire DataFrame."""

    row_count: int
    column_count: int
    columns: dict[str, ColumnProfile] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DataProfile":
        columns = {k: ColumnProfile.from_dict(v) for k, v in d["columns"].items()}
        return cls(
            row_count=d["row_count"],
            column_count=d["column_count"],
            columns=columns,
        )


def profile_dataframe(
    df: pd.DataFrame,
    n_histogram_bins: int = 50,
    max_categories: int = 50,
) -> DataProfile:
    """Compute a full statistical profile of a DataFrame.

    This runs locally on a Pandas DataFrame (not pushed down via Ibis)
    because it needs access to the full distribution for percentiles,
    skew, kurtosis, and histogram computation. For Snowflake-scale
    data, profile a sample rather than the full dataset.
    """
    columns = {}

    for col_name in df.columns:
        col = df[col_name]
        total = len(col)
        nulls = int(col.isna().sum())

        profile = ColumnProfile(
            name=col_name,
            dtype=str(col.dtype),
            count=total,
            null_count=nulls,
            null_pct=round(100.0 * nulls / total, 2) if total > 0 else 0.0,
            unique_count=int(col.nunique()),
        )

        non_null = col.dropna()

        if pd.api.types.is_numeric_dtype(col) and len(non_null) > 0:
            arr = non_null.to_numpy(dtype=float)
            profile.mean = float(np.mean(arr))
            profile.std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            profile.min = float(np.min(arr))
            profile.max = float(np.max(arr))
            profile.median = float(np.median(arr))
            profile.p01 = float(np.percentile(arr, 1))
            profile.p05 = float(np.percentile(arr, 5))
            profile.p25 = float(np.percentile(arr, 25))
            profile.p75 = float(np.percentile(arr, 75))
            profile.p95 = float(np.percentile(arr, 95))
            profile.p99 = float(np.percentile(arr, 99))
            profile.skew = float(pd.Series(arr).skew())
            profile.kurtosis = float(pd.Series(arr).kurtosis())

            counts, bin_edges = np.histogram(arr, bins=n_histogram_bins)
            profile.histogram = {
                "counts": counts.tolist(),
                "bin_edges": bin_edges.tolist(),
            }

        elif len(non_null) > 0:
            vc = non_null.value_counts()
            if len(vc) <= max_categories:
                profile.value_counts = {str(k): int(v) for k, v in vc.items()}
            profile.top_values = [str(v) for v in vc.head(10).index.tolist()]

        columns[col_name] = profile

    return DataProfile(
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
    )
```

```python
# packages/data_quality/src/data_quality/reference_profile.py
"""Reference profile — snapshot of training data statistics.

Saved alongside the model, the reference profile is the "expected"
distribution. New data is compared against it to detect drift.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from data_quality.profiler import DataProfile


@dataclass
class ReferenceProfile:
    """A named, timestamped reference profile."""

    name: str
    created_at: str
    profile: DataProfile

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "profile": self.profile.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReferenceProfile":
        return cls(
            name=d["name"],
            created_at=d["created_at"],
            profile=DataProfile.from_dict(d["profile"]),
        )


def create_reference_profile(
    profile: DataProfile,
    name: str = "default",
) -> ReferenceProfile:
    """Wrap a DataProfile as a named reference."""
    return ReferenceProfile(
        name=name,
        created_at=datetime.now(timezone.utc).isoformat(),
        profile=profile,
    )


def save_reference_profile(ref: ReferenceProfile, path: str | Path) -> Path:
    """Save reference profile to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ref.to_dict(), f, indent=2, default=str)
    return path


def load_reference_profile(path: str | Path) -> ReferenceProfile:
    """Load reference profile from JSON."""
    with open(path) as f:
        return ReferenceProfile.from_dict(json.load(f))
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/data_quality/test_profiler.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(data-quality): add statistical profiler + reference profile"
```

---

## Task 3: Distribution Drift Detection

**Files:**
- Create: `packages/data_quality/src/data_quality/drift.py`
- Test: `tests/data_quality/test_drift.py`

**Step 1: Write the failing test**

```python
# tests/data_quality/test_drift.py
import numpy as np
import pandas as pd
import pytest

from data_quality.profiler import profile_dataframe
from data_quality.drift import (
    detect_drift,
    DriftResult,
    ColumnDrift,
    DriftSeverity,
)


@pytest.fixture
def reference_data():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "value": np.random.normal(10, 2, 1000).tolist(),
            "category": np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2]).tolist(),
        }
    )


@pytest.fixture
def no_drift_data():
    """Data from the same distribution."""
    np.random.seed(99)
    return pd.DataFrame(
        {
            "value": np.random.normal(10, 2, 500).tolist(),
            "category": np.random.choice(["A", "B", "C"], 500, p=[0.5, 0.3, 0.2]).tolist(),
        }
    )


@pytest.fixture
def drifted_data():
    """Data with obvious drift — shifted mean and changed category ratios."""
    np.random.seed(99)
    return pd.DataFrame(
        {
            "value": np.random.normal(20, 5, 500).tolist(),  # mean shifted 10→20
            "category": np.random.choice(["A", "B", "C", "D"], 500, p=[0.1, 0.1, 0.3, 0.5]).tolist(),
        }
    )


def test_detect_no_drift(reference_data, no_drift_data):
    ref_profile = profile_dataframe(reference_data)
    new_profile = profile_dataframe(no_drift_data)
    result = detect_drift(ref_profile, new_profile)

    assert isinstance(result, DriftResult)
    assert result.columns["value"].severity == DriftSeverity.NONE


def test_detect_numeric_drift(reference_data, drifted_data):
    ref_profile = profile_dataframe(reference_data)
    new_profile = profile_dataframe(drifted_data)
    result = detect_drift(ref_profile, new_profile)

    value_drift = result.columns["value"]
    assert isinstance(value_drift, ColumnDrift)
    assert value_drift.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)
    assert value_drift.ks_statistic is not None
    assert value_drift.ks_pvalue is not None
    assert value_drift.ks_pvalue < 0.05  # Should reject H0 of same distribution
    assert value_drift.mean_shift is not None


def test_detect_categorical_drift(reference_data, drifted_data):
    ref_profile = profile_dataframe(reference_data)
    new_profile = profile_dataframe(drifted_data)
    result = detect_drift(ref_profile, new_profile)

    cat_drift = result.columns["category"]
    assert cat_drift.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)
    assert cat_drift.new_categories is not None
    assert "D" in cat_drift.new_categories


def test_detect_null_rate_drift(reference_data):
    """Null rate spiking should be flagged."""
    ref_profile = profile_dataframe(reference_data)
    bad_data = reference_data.copy()
    bad_data.loc[bad_data.index[:300], "value"] = None  # 30% null
    new_profile = profile_dataframe(bad_data)
    result = detect_drift(ref_profile, new_profile)

    assert result.columns["value"].null_rate_drift is not None
    assert result.columns["value"].null_rate_drift > 0.25


def test_drift_result_summary(reference_data, drifted_data):
    ref_profile = profile_dataframe(reference_data)
    new_profile = profile_dataframe(drifted_data)
    result = detect_drift(ref_profile, new_profile)
    summary = result.summary()
    assert isinstance(summary, str)
    assert "value" in summary
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/data_quality/test_drift.py -v
```

Expected: FAIL.

**Step 3: Implement drift detection**

```python
# packages/data_quality/src/data_quality/drift.py
"""Distribution drift detection.

Compares a new dataset's profile against a reference (training) profile
to detect when the data distribution has shifted. Uses:

- KS test (numeric): non-parametric test for distribution difference.
  p-value < 0.05 = distributions are statistically different.
- Mean/std shift (numeric): simple but interpretable — how many
  reference-stds has the mean shifted?
- Category proportion drift (categorical): chi-squared-style comparison
  of category frequencies.
- Null rate drift: absolute change in null percentage.

Severity thresholds are configurable. Defaults:
  NONE: no significant drift detected
  WARNING: drift detected but within tolerance
  CRITICAL: severe drift — model predictions may be unreliable
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import stats as scipy_stats

from data_quality.profiler import DataProfile, ColumnProfile


class DriftSeverity(Enum):
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ColumnDrift:
    """Drift analysis for a single column."""

    column: str
    severity: DriftSeverity

    # Numeric drift metrics
    ks_statistic: float | None = None
    ks_pvalue: float | None = None
    mean_shift: float | None = None       # In units of reference std
    std_ratio: float | None = None        # new_std / ref_std

    # Categorical drift metrics
    new_categories: list[str] | None = None
    missing_categories: list[str] | None = None
    category_proportion_drift: float | None = None  # Jensen-Shannon divergence

    # Null drift
    null_rate_drift: float | None = None  # Absolute change in null %

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class DriftResult:
    """Drift analysis for all columns."""

    columns: dict[str, ColumnDrift] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["Drift Analysis Summary:"]
        for col, drift in self.columns.items():
            icon = {"none": "✅", "warning": "⚠️", "critical": "🚨"}[drift.severity.value]
            detail = ""
            if drift.ks_pvalue is not None:
                detail += f" KS p={drift.ks_pvalue:.4f}"
            if drift.mean_shift is not None:
                detail += f" mean_shift={drift.mean_shift:.2f}σ"
            if drift.null_rate_drift is not None and drift.null_rate_drift > 0.01:
                detail += f" null_drift={drift.null_rate_drift:+.1%}"
            if drift.new_categories:
                detail += f" new_cats={drift.new_categories}"
            lines.append(f"  {icon} {col}: {drift.severity.value}{detail}")
        return "\n".join(lines)

    @property
    def has_critical(self) -> bool:
        return any(d.severity == DriftSeverity.CRITICAL for d in self.columns.values())

    @property
    def has_warnings(self) -> bool:
        return any(d.severity == DriftSeverity.WARNING for d in self.columns.values())


def detect_drift(
    ref_profile: DataProfile,
    new_profile: DataProfile,
    ks_warning_threshold: float = 0.05,
    ks_critical_threshold: float = 0.001,
    mean_shift_warning: float = 1.0,     # 1 std
    mean_shift_critical: float = 3.0,     # 3 stds
    null_rate_warning: float = 0.05,      # 5% absolute increase
    null_rate_critical: float = 0.20,     # 20% absolute increase
) -> DriftResult:
    """Compare new data profile against reference profile for drift.

    Analyzes each column that exists in both profiles.
    """
    result = DriftResult()

    common_cols = set(ref_profile.columns.keys()) & set(new_profile.columns.keys())

    for col_name in sorted(common_cols):
        ref_col = ref_profile.columns[col_name]
        new_col = new_profile.columns[col_name]

        drift = ColumnDrift(column=col_name, severity=DriftSeverity.NONE)

        # Null rate drift
        ref_null_pct = ref_col.null_pct / 100.0
        new_null_pct = new_col.null_pct / 100.0
        drift.null_rate_drift = new_null_pct - ref_null_pct

        severity = DriftSeverity.NONE

        if drift.null_rate_drift >= null_rate_critical:
            severity = DriftSeverity.CRITICAL
        elif drift.null_rate_drift >= null_rate_warning:
            severity = _max_severity(severity, DriftSeverity.WARNING)

        # Numeric drift
        if ref_col.mean is not None and new_col.mean is not None:
            # Mean shift in units of reference std
            ref_std = ref_col.std if ref_col.std and ref_col.std > 0 else 1.0
            drift.mean_shift = abs(new_col.mean - ref_col.mean) / ref_std
            drift.std_ratio = (new_col.std / ref_std) if new_col.std else None

            if drift.mean_shift >= mean_shift_critical:
                severity = _max_severity(severity, DriftSeverity.CRITICAL)
            elif drift.mean_shift >= mean_shift_warning:
                severity = _max_severity(severity, DriftSeverity.WARNING)

            # KS test using histograms as proxy
            # (we don't have raw data, only profiles — approximate with histogram samples)
            if ref_col.histogram and new_col.histogram:
                ref_samples = _histogram_to_samples(ref_col.histogram, n=1000)
                new_samples = _histogram_to_samples(new_col.histogram, n=1000)
                ks_stat, ks_pval = scipy_stats.ks_2samp(ref_samples, new_samples)
                drift.ks_statistic = float(ks_stat)
                drift.ks_pvalue = float(ks_pval)

                if ks_pval < ks_critical_threshold:
                    severity = _max_severity(severity, DriftSeverity.CRITICAL)
                elif ks_pval < ks_warning_threshold:
                    severity = _max_severity(severity, DriftSeverity.WARNING)

        # Categorical drift
        if ref_col.value_counts and new_col.value_counts:
            ref_cats = set(ref_col.value_counts.keys())
            new_cats = set(new_col.value_counts.keys())
            drift.new_categories = sorted(new_cats - ref_cats) or None
            drift.missing_categories = sorted(ref_cats - new_cats) or None

            if drift.new_categories:
                severity = _max_severity(severity, DriftSeverity.WARNING)

            # Jensen-Shannon divergence for proportion drift
            all_cats = sorted(ref_cats | new_cats)
            ref_total = sum(ref_col.value_counts.values())
            new_total = sum(new_col.value_counts.values())
            ref_probs = np.array([ref_col.value_counts.get(c, 0) / ref_total for c in all_cats])
            new_probs = np.array([new_col.value_counts.get(c, 0) / new_total for c in all_cats])

            # Add small epsilon to avoid log(0)
            eps = 1e-10
            ref_probs = ref_probs + eps
            new_probs = new_probs + eps
            ref_probs /= ref_probs.sum()
            new_probs /= new_probs.sum()

            m = (ref_probs + new_probs) / 2
            jsd = 0.5 * (
                np.sum(ref_probs * np.log(ref_probs / m))
                + np.sum(new_probs * np.log(new_probs / m))
            )
            drift.category_proportion_drift = float(jsd)

            if jsd > 0.1:
                severity = _max_severity(severity, DriftSeverity.CRITICAL)
            elif jsd > 0.02:
                severity = _max_severity(severity, DriftSeverity.WARNING)

        drift.severity = severity
        result.columns[col_name] = drift

    return result


def _histogram_to_samples(histogram: dict, n: int = 1000) -> np.ndarray:
    """Reconstruct approximate samples from a histogram for KS testing."""
    counts = np.array(histogram["counts"])
    edges = np.array(histogram["bin_edges"])
    total = counts.sum()
    if total == 0:
        return np.zeros(n)
    probs = counts / total
    bin_indices = np.random.choice(len(counts), size=n, p=probs)
    # Sample uniformly within each bin
    samples = np.array([
        np.random.uniform(edges[i], edges[i + 1])
        for i in bin_indices
    ])
    return samples


def _max_severity(a: DriftSeverity, b: DriftSeverity) -> DriftSeverity:
    """Return the more severe of two severities."""
    order = {DriftSeverity.NONE: 0, DriftSeverity.WARNING: 1, DriftSeverity.CRITICAL: 2}
    return a if order[a] >= order[b] else b
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/data_quality/test_drift.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(data-quality): add distribution drift detection (KS, JSD, null rate)"
```

---

## Task 4: Business Rule Engine

**Files:**
- Create: `packages/data_quality/src/data_quality/rules.py`
- Test: `tests/data_quality/test_rules.py`

**Step 1: Write the failing test**

```python
# tests/data_quality/test_rules.py
import pandas as pd
import pytest

from data_quality.rules import (
    RuleEngine,
    Rule,
    RuleResult,
    nfl_default_rules,
)
from data_quality.schema import Severity


@pytest.fixture
def clean_data():
    return pd.DataFrame(
        {
            "down": [1, 2, 3, 4],
            "ydstogo": [10, 7, 3, 1],
            "yardline_100": [75, 50, 30, 5],
            "qtr": [1, 2, 3, 4],
            "game_seconds_remaining": [3600, 2700, 1800, 300],
            "score_differential": [0, 7, -14, -3],
            "wp": [0.50, 0.65, 0.30, 0.45],
        }
    )


@pytest.fixture
def dirty_data():
    return pd.DataFrame(
        {
            "down": [0, 5, 4, 4],             # 0 and 5 are invalid
            "ydstogo": [10, 7, -1, 100],       # negative and > 99
            "yardline_100": [75, 0, 101, 5],   # 0 and 101 are invalid
            "qtr": [1, 6, 3, 4],               # 6 is invalid
            "game_seconds_remaining": [3600, -10, 1800, 5000],  # negative and > 3600
            "score_differential": [0, 7, -14, -3],
            "wp": [0.50, 1.5, -0.1, 0.45],    # out of [0, 1]
        }
    )


def test_rule_engine_clean_data(clean_data):
    engine = RuleEngine(rules=nfl_default_rules())
    results = engine.validate(clean_data)
    errors = [r for r in results if r.severity == Severity.ERROR]
    assert len(errors) == 0


def test_rule_engine_dirty_data(dirty_data):
    engine = RuleEngine(rules=nfl_default_rules())
    results = engine.validate(dirty_data)
    errors = [r for r in results if r.severity == Severity.ERROR]
    assert len(errors) > 0
    # Should catch down=0, down=5, ydstogo=-1, yardline_100=0, etc.
    error_cols = {r.column for r in errors}
    assert "down" in error_cols
    assert "yardline_100" in error_cols


def test_custom_rule():
    rule = Rule(
        name="punt_from_own_territory",
        description="Punts should not happen inside opponent's 20",
        column="yardline_100",
        check=lambda df: ~((df.get("play_type") == "punt") & (df["yardline_100"] <= 20)),
        severity=Severity.WARNING,
    )
    data = pd.DataFrame(
        {
            "play_type": ["punt", "punt", "pass"],
            "yardline_100": [15, 80, 10],
        }
    )
    engine = RuleEngine(rules=[rule])
    results = engine.validate(data)
    warnings = [r for r in results if r.severity == Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].failing_rows == 1


def test_rule_result_structure():
    result = RuleResult(
        rule_name="test",
        column="x",
        severity=Severity.ERROR,
        total_rows=100,
        failing_rows=5,
        failing_pct=5.0,
        message="5 rows failed",
    )
    assert result.passing_rows == 95
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/data_quality/test_rules.py -v
```

Expected: FAIL.

**Step 3: Implement the rule engine**

```python
# packages/data_quality/src/data_quality/rules.py
"""Business rule engine for domain-specific data validation.

Rules are beyond what schema validation catches — they encode
domain knowledge about what's *possible* in NFL data, not just
what was *observed* in training data.

Example: a schema learned from data might allow ydstogo=99 because
it appeared once. But a business rule knows ydstogo should be 1-99
and down should be 1-4.

Rules are defined as callables that take a DataFrame and return
a boolean mask (True = passes, False = fails).
"""

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from data_quality.schema import Severity


@dataclass
class Rule:
    """A single validation rule."""

    name: str
    description: str
    column: str
    check: Callable[[pd.DataFrame], pd.Series]  # Returns bool mask
    severity: Severity = Severity.ERROR


@dataclass
class RuleResult:
    """Result of applying a single rule."""

    rule_name: str
    column: str
    severity: Severity
    total_rows: int
    failing_rows: int
    failing_pct: float
    message: str
    sample_failing_indices: list[int] = field(default_factory=list)

    @property
    def passing_rows(self) -> int:
        return self.total_rows - self.failing_rows


class RuleEngine:
    """Applies a list of rules to a DataFrame."""

    def __init__(self, rules: list[Rule]):
        self.rules = rules

    def validate(self, df: pd.DataFrame) -> list[RuleResult]:
        """Apply all rules and return results."""
        results = []
        for rule in self.rules:
            # Skip rules for columns not in the data
            if rule.column not in df.columns and rule.column != "__cross_column__":
                continue

            try:
                mask = rule.check(df)
                failing = ~mask
                n_failing = int(failing.sum())
                n_total = len(df)

                if n_failing > 0:
                    sample_indices = list(
                        df.index[failing][:10]  # First 10 failing rows
                    )
                    results.append(
                        RuleResult(
                            rule_name=rule.name,
                            column=rule.column,
                            severity=rule.severity,
                            total_rows=n_total,
                            failing_rows=n_failing,
                            failing_pct=round(100.0 * n_failing / n_total, 2),
                            message=f"{rule.description}: {n_failing} rows failed ({100.0 * n_failing / n_total:.1f}%)",
                            sample_failing_indices=sample_indices,
                        )
                    )
            except Exception as e:
                results.append(
                    RuleResult(
                        rule_name=rule.name,
                        column=rule.column,
                        severity=Severity.ERROR,
                        total_rows=len(df),
                        failing_rows=0,
                        failing_pct=0.0,
                        message=f"Rule execution error: {e}",
                    )
                )

        return results


def nfl_default_rules() -> list[Rule]:
    """Default business rules for NFL play-by-play data.

    These encode physical and logical constraints about football:
    - Down is 1-4 (5 would mean overtime edge case, handled separately)
    - Yards to go is 1-99
    - Yard line from end zone is 1-99
    - Quarter is 1-5 (5 = overtime)
    - Game clock values are non-negative and within bounds
    - Win probability is between 0 and 1
    """
    return [
        Rule(
            name="valid_down",
            description="Down must be 1-4",
            column="down",
            check=lambda df: df["down"].between(1, 4),
            severity=Severity.ERROR,
        ),
        Rule(
            name="valid_ydstogo",
            description="Yards to go must be 1-99",
            column="ydstogo",
            check=lambda df: df["ydstogo"].between(1, 99),
            severity=Severity.ERROR,
        ),
        Rule(
            name="valid_yardline",
            description="Yard line must be 1-99",
            column="yardline_100",
            check=lambda df: df["yardline_100"].between(1, 99),
            severity=Severity.ERROR,
        ),
        Rule(
            name="valid_quarter",
            description="Quarter must be 1-5",
            column="qtr",
            check=lambda df: df["qtr"].between(1, 5),
            severity=Severity.ERROR,
        ),
        Rule(
            name="valid_game_seconds",
            description="Game seconds remaining must be 0-3600",
            column="game_seconds_remaining",
            check=lambda df: df["game_seconds_remaining"].between(0, 3600),
            severity=Severity.ERROR,
        ),
        Rule(
            name="valid_wp",
            description="Win probability must be 0-1",
            column="wp",
            check=lambda df: df["wp"].between(0.0, 1.0),
            severity=Severity.ERROR,
        ),
        Rule(
            name="ydstogo_not_greater_than_yardline",
            description="Yards to go should not exceed yards from end zone",
            column="ydstogo",
            check=lambda df: df["ydstogo"] <= df["yardline_100"] if "yardline_100" in df.columns else pd.Series(True, index=df.index),
            severity=Severity.WARNING,
        ),
    ]
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/data_quality/test_rules.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(data-quality): add business rule engine + NFL default rules"
```

---

## Task 5: Validation Report Builder

**Files:**
- Create: `packages/data_quality/src/data_quality/report.py`
- Test: `tests/data_quality/test_report.py`

**Step 1: Write the failing test**

```python
# tests/data_quality/test_report.py
import numpy as np
import pandas as pd
import pytest

from data_quality.report import (
    build_validation_report,
    ValidationReport,
    OverallStatus,
)
from data_quality.schema import learn_schema
from data_quality.profiler import profile_dataframe
from data_quality.rules import nfl_default_rules


@pytest.fixture
def training_data():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "down": np.random.choice([1, 2, 3, 4], 200).tolist(),
            "ydstogo": np.random.randint(1, 15, 200).tolist(),
            "yardline_100": np.random.randint(1, 99, 200).tolist(),
            "qtr": np.random.choice([1, 2, 3, 4], 200).tolist(),
            "game_seconds_remaining": np.random.randint(0, 3600, 200).tolist(),
            "score_differential": np.random.randint(-21, 21, 200).tolist(),
            "wp": np.random.uniform(0.05, 0.95, 200).tolist(),
        }
    )


def test_report_clean_data(training_data):
    schema = learn_schema(training_data)
    ref_profile = profile_dataframe(training_data)

    report = build_validation_report(
        data=training_data,
        schema=schema,
        reference_profile=ref_profile,
        rules=nfl_default_rules(),
    )
    assert isinstance(report, ValidationReport)
    assert report.status == OverallStatus.PASS


def test_report_dirty_data(training_data):
    schema = learn_schema(training_data)
    ref_profile = profile_dataframe(training_data)

    dirty = training_data.copy()
    dirty.loc[0, "down"] = 99
    dirty.loc[1, "wp"] = 5.0
    dirty = dirty.drop(columns=["score_differential"])

    report = build_validation_report(
        data=dirty,
        schema=schema,
        reference_profile=ref_profile,
        rules=nfl_default_rules(),
    )
    assert report.status == OverallStatus.FAIL
    assert report.schema_result is not None
    assert not report.schema_result.is_valid


def test_report_to_dict(training_data):
    schema = learn_schema(training_data)
    ref_profile = profile_dataframe(training_data)

    report = build_validation_report(
        data=training_data,
        schema=schema,
        reference_profile=ref_profile,
        rules=nfl_default_rules(),
    )
    d = report.to_dict()
    assert "status" in d
    assert "schema" in d
    assert "drift" in d
    assert "rules" in d
    assert "profile" in d
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/data_quality/test_report.py -v
```

Expected: FAIL.

**Step 3: Implement the report builder**

```python
# packages/data_quality/src/data_quality/report.py
"""Validation report builder.

Orchestrates all three validation layers (schema, drift, rules)
into a single unified report with an overall pass/warn/fail status.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import pandas as pd

from data_quality.schema import (
    DataSchema,
    SchemaValidationResult,
    Severity,
    validate_schema,
)
from data_quality.profiler import DataProfile, profile_dataframe
from data_quality.drift import DriftResult, detect_drift
from data_quality.rules import Rule, RuleEngine, RuleResult


class OverallStatus(Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class ValidationReport:
    """Unified validation report across all layers."""

    status: OverallStatus
    timestamp: str
    row_count: int

    # Layer results
    schema_result: SchemaValidationResult | None = None
    drift_result: DriftResult | None = None
    rule_results: list[RuleResult] = field(default_factory=list)
    profile: DataProfile | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "row_count": self.row_count,
            "schema": {
                "is_valid": self.schema_result.is_valid if self.schema_result else None,
                "error_count": self.schema_result.error_count if self.schema_result else 0,
                "warning_count": self.schema_result.warning_count if self.schema_result else 0,
            },
            "drift": {
                "has_critical": self.drift_result.has_critical if self.drift_result else False,
                "has_warnings": self.drift_result.has_warnings if self.drift_result else False,
                "columns": {
                    k: v.to_dict()
                    for k, v in (self.drift_result.columns.items() if self.drift_result else {})
                },
            },
            "rules": {
                "total": len(self.rule_results),
                "failing": sum(1 for r in self.rule_results),
                "results": [
                    {
                        "name": r.rule_name,
                        "column": r.column,
                        "severity": r.severity.value,
                        "failing_rows": r.failing_rows,
                        "message": r.message,
                    }
                    for r in self.rule_results
                ],
            },
            "profile": self.profile.to_dict() if self.profile else None,
        }

    def summary(self) -> str:
        icon = {"pass": "✅", "warning": "⚠️", "fail": "🚨"}[self.status.value]
        lines = [
            f"{icon} Overall Status: {self.status.value.upper()}",
            f"   Rows: {self.row_count:,}",
            f"   Timestamp: {self.timestamp}",
        ]
        if self.schema_result:
            lines.append(
                f"   Schema: {'PASS' if self.schema_result.is_valid else 'FAIL'} "
                f"({self.schema_result.error_count} errors, {self.schema_result.warning_count} warnings)"
            )
        if self.drift_result:
            drift_status = "CRITICAL" if self.drift_result.has_critical else (
                "WARNING" if self.drift_result.has_warnings else "OK"
            )
            lines.append(f"   Drift: {drift_status}")
        if self.rule_results:
            lines.append(f"   Rules: {len(self.rule_results)} violations")
        return "\n".join(lines)


def build_validation_report(
    data: pd.DataFrame,
    schema: DataSchema,
    reference_profile: DataProfile | None = None,
    rules: list[Rule] | None = None,
) -> ValidationReport:
    """Run all validation layers and produce a unified report.

    Args:
        data: The DataFrame to validate.
        schema: Learned schema from training data.
        reference_profile: Statistical profile from training data (for drift).
        rules: Business rules to check.

    Returns:
        ValidationReport with overall status and per-layer results.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Layer 1: Schema validation
    schema_result = validate_schema(data, schema)

    # Layer 2: Statistical profiling + drift detection
    current_profile = profile_dataframe(data)
    drift_result = None
    if reference_profile:
        drift_result = detect_drift(reference_profile, current_profile)

    # Layer 3: Business rules
    rule_results = []
    if rules:
        engine = RuleEngine(rules=rules)
        rule_results = engine.validate(data)

    # Determine overall status
    status = OverallStatus.PASS

    if schema_result and not schema_result.is_valid:
        status = OverallStatus.FAIL

    if drift_result and drift_result.has_critical:
        status = OverallStatus.FAIL
    elif drift_result and drift_result.has_warnings and status == OverallStatus.PASS:
        status = OverallStatus.WARNING

    rule_errors = [r for r in rule_results if r.severity == Severity.ERROR]
    rule_warnings = [r for r in rule_results if r.severity == Severity.WARNING]
    if rule_errors:
        status = OverallStatus.FAIL
    elif rule_warnings and status == OverallStatus.PASS:
        status = OverallStatus.WARNING

    return ValidationReport(
        status=status,
        timestamp=timestamp,
        row_count=len(data),
        schema_result=schema_result,
        drift_result=drift_result,
        rule_results=rule_results,
        profile=current_profile,
    )
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/data_quality/test_report.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(data-quality): add unified validation report builder"
```

---

## Task 6: Streamlit — Data Quality Overview Page

**Files:**
- Create: `packages/dashboard/src/dashboard/pages/08_data_quality_overview.py`

**Step 1: Implement the page**

```python
# packages/dashboard/src/dashboard/pages/08_data_quality_overview.py
"""Data Quality Overview — upload data, run all validations, see results."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_quality.schema import learn_schema, DataSchema, Severity
from data_quality.profiler import profile_dataframe
from data_quality.reference_profile import (
    load_reference_profile,
    ReferenceProfile,
)
from data_quality.rules import nfl_default_rules
from data_quality.report import build_validation_report, OverallStatus

st.header("🛡️ Data Quality Overview")

# --- Reference Data ---
st.sidebar.subheader("Reference (Training) Data")
ref_source = st.sidebar.radio(
    "Reference source",
    ["Upload training CSV", "Load saved profile", "Use sample data"],
)

ref_profile = None
schema = None

if ref_source == "Upload training CSV":
    ref_file = st.sidebar.file_uploader("Training data CSV", type=["csv"], key="ref_upload")
    if ref_file:
        ref_df = pd.read_csv(ref_file)
        schema = learn_schema(ref_df)
        ref_profile = profile_dataframe(ref_df)
        st.sidebar.success(f"Reference: {len(ref_df):,} rows, {len(ref_df.columns)} columns")

elif ref_source == "Load saved profile":
    profile_path = st.sidebar.text_input("Profile JSON path", value="models/latest/reference_profile.json")
    if Path(profile_path).exists():
        ref = load_reference_profile(profile_path)
        ref_profile = ref.profile
        st.sidebar.success(f"Loaded profile: {ref.name}")
    else:
        st.sidebar.warning("Profile file not found")

elif ref_source == "Use sample data":
    import numpy as np
    np.random.seed(42)
    ref_df = pd.DataFrame(
        {
            "down": np.random.choice([1, 2, 3, 4], 1000).tolist(),
            "ydstogo": np.random.randint(1, 15, 1000).tolist(),
            "yardline_100": np.random.randint(1, 99, 1000).tolist(),
            "qtr": np.random.choice([1, 2, 3, 4], 1000).tolist(),
            "game_seconds_remaining": np.random.randint(0, 3600, 1000).tolist(),
            "score_differential": np.random.randint(-21, 21, 1000).tolist(),
            "wp": np.random.uniform(0.05, 0.95, 1000).tolist(),
        }
    )
    schema = learn_schema(ref_df)
    ref_profile = profile_dataframe(ref_df)
    st.sidebar.success("Using sample reference data")

# --- Incoming Data ---
st.subheader("Upload Data to Validate")

data_file = st.file_uploader("Upload CSV to validate", type=["csv"], key="data_upload")

if data_file is None:
    st.info("Upload a CSV file to run data quality validation.")
    st.stop()

data = pd.read_csv(data_file)
st.caption(f"Uploaded: {len(data):,} rows × {len(data.columns)} columns")

if schema is None:
    schema = learn_schema(data)
    st.warning("No reference data provided — schema learned from uploaded data (self-validation only)")

# --- Run Validation ---
if st.button("Run Validation", type="primary", use_container_width=True):
    with st.spinner("Running schema validation, drift detection, and business rules..."):
        report = build_validation_report(
            data=data,
            schema=schema,
            reference_profile=ref_profile,
            rules=nfl_default_rules(),
        )

    # --- Overall Status Banner ---
    status_colors = {
        OverallStatus.PASS: ("✅", "success"),
        OverallStatus.WARNING: ("⚠️", "warning"),
        OverallStatus.FAIL: ("🚨", "error"),
    }
    icon, method = status_colors[report.status]
    getattr(st, method)(f"{icon} Overall Status: **{report.status.value.upper()}**")

    # --- Three-Column Summary ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Schema")
        if report.schema_result:
            if report.schema_result.is_valid:
                st.success("PASS")
            else:
                st.error(f"{report.schema_result.error_count} errors, {report.schema_result.warning_count} warnings")
                for e in report.schema_result.errors[:10]:
                    severity_icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}
                    st.markdown(
                        f"{severity_icon.get(e.severity.value, '⚪')} **{e.column}**: {e.message}"
                    )

    with col2:
        st.markdown("### Drift")
        if report.drift_result:
            if report.drift_result.has_critical:
                st.error("CRITICAL drift detected")
            elif report.drift_result.has_warnings:
                st.warning("Drift warnings")
            else:
                st.success("No significant drift")

            # Drift heatmap
            drift_data = []
            for col_name, drift in report.drift_result.columns.items():
                drift_data.append({
                    "column": col_name,
                    "severity": drift.severity.value,
                    "score": {"none": 0, "warning": 1, "critical": 2}[drift.severity.value],
                })
            if drift_data:
                drift_df = pd.DataFrame(drift_data)
                fig = px.bar(
                    drift_df,
                    x="score",
                    y="column",
                    orientation="h",
                    color="severity",
                    color_discrete_map={"none": "#2ecc71", "warning": "#f39c12", "critical": "#e74c3c"},
                    title="Drift by Column",
                )
                fig.update_layout(
                    showlegend=True,
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(visible=False),
                    height=max(200, len(drift_data) * 25),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No reference profile — drift detection skipped")

    with col3:
        st.markdown("### Business Rules")
        if report.rule_results:
            n_errors = sum(1 for r in report.rule_results if r.severity == Severity.ERROR)
            n_warnings = sum(1 for r in report.rule_results if r.severity == Severity.WARNING)
            if n_errors > 0:
                st.error(f"{n_errors} rule violations")
            elif n_warnings > 0:
                st.warning(f"{n_warnings} rule warnings")

            for r in report.rule_results:
                severity_icon = {"error": "🔴", "warning": "🟡"}
                st.markdown(
                    f"{severity_icon.get(r.severity.value, '⚪')} **{r.rule_name}**: "
                    f"{r.failing_rows} rows ({r.failing_pct:.1f}%)"
                )
        else:
            st.success("All rules pass")

    # --- Column-Level Detail ---
    st.markdown("---")
    with st.expander("Column-level profiling"):
        if report.profile:
            profile_rows = []
            for col_name, cp in report.profile.columns.items():
                profile_rows.append({
                    "column": col_name,
                    "dtype": cp.dtype,
                    "null_%": cp.null_pct,
                    "unique": cp.unique_count,
                    "mean": round(cp.mean, 3) if cp.mean is not None else None,
                    "std": round(cp.std, 3) if cp.std is not None else None,
                    "min": round(cp.min, 3) if cp.min is not None else None,
                    "max": round(cp.max, 3) if cp.max is not None else None,
                })
            st.dataframe(pd.DataFrame(profile_rows), use_container_width=True)

    # --- Raw Report JSON ---
    with st.expander("Raw report JSON"):
        st.json(report.to_dict())
```

**Step 2: Manual verification**

```bash
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Navigate to Data Quality Overview. Upload CSVs. Verify all three validation layers render.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add data quality overview page"
```

---

## Task 7: Streamlit — Drift Monitor Page

**Files:**
- Create: `packages/dashboard/src/dashboard/pages/09_drift_monitor.py`

**Step 1: Implement the page**

```python
# packages/dashboard/src/dashboard/pages/09_drift_monitor.py
"""Drift Monitor — detailed drift analysis with distribution overlays."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_quality.profiler import profile_dataframe, ColumnProfile
from data_quality.drift import detect_drift, DriftSeverity

st.header("📉 Distribution Drift Monitor")
st.markdown(
    "Compare incoming data distributions against the training reference. "
    "Overlay histograms to visually inspect where drift is happening."
)

# --- Upload Both Datasets ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference Data")
    ref_file = st.file_uploader("Upload reference CSV", type=["csv"], key="drift_ref")
    ref_df = None
    if ref_file:
        ref_df = pd.read_csv(ref_file)
        st.caption(f"{len(ref_df):,} rows")

with col2:
    st.subheader("New Data")
    new_file = st.file_uploader("Upload new CSV", type=["csv"], key="drift_new")
    new_df = None
    if new_file:
        new_df = pd.read_csv(new_file)
        st.caption(f"{len(new_df):,} rows")

if ref_df is None or new_df is None:
    st.info("Upload both reference and new data to run drift analysis.")
    st.stop()

# --- Run Drift Analysis ---
with st.spinner("Profiling both datasets and detecting drift..."):
    ref_profile = profile_dataframe(ref_df)
    new_profile = profile_dataframe(new_df)
    drift_result = detect_drift(ref_profile, new_profile)

# --- Summary ---
st.markdown("---")
if drift_result.has_critical:
    st.error("🚨 Critical drift detected in one or more columns")
elif drift_result.has_warnings:
    st.warning("⚠️ Drift warnings detected")
else:
    st.success("✅ No significant drift detected")

# --- Per-Column Analysis ---
st.subheader("Column-by-Column Analysis")

common_cols = sorted(
    set(ref_profile.columns.keys()) & set(new_profile.columns.keys())
)

# Sort by severity (critical first)
severity_order = {DriftSeverity.CRITICAL: 0, DriftSeverity.WARNING: 1, DriftSeverity.NONE: 2}
common_cols = sorted(
    common_cols,
    key=lambda c: severity_order.get(
        drift_result.columns.get(c, type("", (), {"severity": DriftSeverity.NONE})).severity,
        2,
    ),
)

for col_name in common_cols:
    drift = drift_result.columns.get(col_name)
    if drift is None:
        continue

    severity_icon = {
        DriftSeverity.NONE: "✅",
        DriftSeverity.WARNING: "⚠️",
        DriftSeverity.CRITICAL: "🚨",
    }

    with st.expander(
        f"{severity_icon[drift.severity]} {col_name} — {drift.severity.value.upper()}",
        expanded=drift.severity != DriftSeverity.NONE,
    ):
        ref_col = ref_profile.columns[col_name]
        new_col = new_profile.columns[col_name]

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        if ref_col.mean is not None:
            m1.metric("Ref Mean", f"{ref_col.mean:.3f}")
            m2.metric(
                "New Mean",
                f"{new_col.mean:.3f}" if new_col.mean else "N/A",
                delta=f"{(new_col.mean or 0) - ref_col.mean:.3f}" if new_col.mean else None,
            )
        if drift.ks_pvalue is not None:
            m3.metric("KS p-value", f"{drift.ks_pvalue:.4f}")
        if drift.mean_shift is not None:
            m4.metric("Mean Shift", f"{drift.mean_shift:.2f}σ")

        # Null rate comparison
        if drift.null_rate_drift and abs(drift.null_rate_drift) > 0.001:
            st.markdown(
                f"**Null rate:** ref={ref_col.null_pct:.1f}% → new={new_col.null_pct:.1f}% "
                f"(Δ={drift.null_rate_drift:+.1%})"
            )

        # Histogram overlay for numeric columns
        if ref_col.histogram and new_col.histogram:
            fig = go.Figure()

            ref_centers = [
                (ref_col.histogram["bin_edges"][i] + ref_col.histogram["bin_edges"][i + 1]) / 2
                for i in range(len(ref_col.histogram["counts"]))
            ]
            new_centers = [
                (new_col.histogram["bin_edges"][i] + new_col.histogram["bin_edges"][i + 1]) / 2
                for i in range(len(new_col.histogram["counts"]))
            ]

            # Normalize to density for fair comparison
            ref_total = sum(ref_col.histogram["counts"]) or 1
            new_total = sum(new_col.histogram["counts"]) or 1

            fig.add_trace(
                go.Bar(
                    x=ref_centers,
                    y=[c / ref_total for c in ref_col.histogram["counts"]],
                    name="Reference",
                    marker_color="rgba(52, 152, 219, 0.5)",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=new_centers,
                    y=[c / new_total for c in new_col.histogram["counts"]],
                    name="New Data",
                    marker_color="rgba(231, 76, 60, 0.5)",
                )
            )
            fig.update_layout(
                barmode="overlay",
                title=f"{col_name} Distribution Overlay",
                xaxis_title=col_name,
                yaxis_title="Density",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Category comparison for categorical columns
        if ref_col.value_counts and new_col.value_counts:
            all_cats = sorted(
                set(ref_col.value_counts.keys()) | set(new_col.value_counts.keys())
            )
            cat_df = pd.DataFrame(
                {
                    "category": all_cats,
                    "reference": [ref_col.value_counts.get(c, 0) for c in all_cats],
                    "new": [new_col.value_counts.get(c, 0) for c in all_cats],
                }
            )
            fig_cat = go.Figure()
            fig_cat.add_trace(go.Bar(x=cat_df["category"], y=cat_df["reference"], name="Reference"))
            fig_cat.add_trace(go.Bar(x=cat_df["category"], y=cat_df["new"], name="New Data"))
            fig_cat.update_layout(barmode="group", title=f"{col_name} Category Comparison", height=300)
            st.plotly_chart(fig_cat, use_container_width=True)

            if drift.new_categories:
                st.warning(f"New categories: {drift.new_categories}")
            if drift.missing_categories:
                st.info(f"Missing categories: {drift.missing_categories}")
```

**Step 2: Manual verification**

```bash
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Navigate to Drift Monitor. Upload two CSVs. Verify histogram overlays and drift metrics render.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add drift monitor page with distribution overlays"
```

---

## Task 8: Streamlit — Validation Rules Page

**Files:**
- Create: `packages/dashboard/src/dashboard/pages/10_validation_rules.py`

**Step 1: Implement the page**

```python
# packages/dashboard/src/dashboard/pages/10_validation_rules.py
"""Validation Rules Manager — view, test, and manage business rules."""

import pandas as pd
import streamlit as st

from data_quality.rules import nfl_default_rules, RuleEngine, Rule
from data_quality.schema import Severity

st.header("📋 Validation Rules")
st.markdown(
    "View all active business rules, test them against uploaded data, "
    "and inspect which rows fail each rule."
)

# --- Rule Catalog ---
st.subheader("Active Rules")

rules = nfl_default_rules()

rule_catalog = []
for r in rules:
    rule_catalog.append(
        {
            "Name": r.name,
            "Column": r.column,
            "Description": r.description,
            "Severity": r.severity.value.upper(),
        }
    )

st.dataframe(
    pd.DataFrame(rule_catalog),
    use_container_width=True,
    column_config={
        "Severity": st.column_config.TextColumn(
            width="small",
        ),
    },
)

# --- Test Against Data ---
st.markdown("---")
st.subheader("Test Rules Against Data")

data_file = st.file_uploader("Upload CSV to validate", type=["csv"])
if data_file is None:
    st.info("Upload a CSV to test rules against it.")
    st.stop()

data = pd.read_csv(data_file)
st.caption(f"Uploaded: {len(data):,} rows × {len(data.columns)} columns")

if st.button("Run Rules", type="primary"):
    engine = RuleEngine(rules=rules)
    results = engine.validate(data)

    if not results:
        st.success("All rules pass — no violations found.")
    else:
        # Summary metrics
        n_errors = sum(1 for r in results if r.severity == Severity.ERROR)
        n_warnings = sum(1 for r in results if r.severity == Severity.WARNING)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Violations", len(results))
        col2.metric("Errors", n_errors)
        col3.metric("Warnings", n_warnings)

        # Per-rule detail
        st.markdown("---")
        for result in sorted(results, key=lambda r: r.severity.value):
            severity_color = {
                "error": "🔴",
                "warning": "🟡",
            }
            icon = severity_color.get(result.severity.value, "⚪")

            with st.expander(
                f"{icon} {result.rule_name} — {result.failing_rows} failing rows ({result.failing_pct:.1f}%)",
                expanded=result.severity == Severity.ERROR,
            ):
                st.markdown(f"**Column:** `{result.column}`")
                st.markdown(f"**Message:** {result.message}")

                # Show progress bar for pass rate
                pass_pct = (result.total_rows - result.failing_rows) / result.total_rows
                st.progress(pass_pct, text=f"Pass rate: {pass_pct:.1%}")

                # Show failing rows
                if result.sample_failing_indices:
                    st.markdown("**Sample failing rows:**")
                    failing_df = data.iloc[result.sample_failing_indices]
                    # Highlight the problematic column
                    st.dataframe(
                        failing_df,
                        use_container_width=True,
                        column_config={
                            result.column: st.column_config.NumberColumn(
                                help="This column triggered the rule violation",
                            ),
                        },
                    )
```

**Step 2: Manual verification**

```bash
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Navigate to Validation Rules. Upload data. Verify rule results and failing rows display.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add validation rules manager page"
```

---

## Summary: Run Order

| Task | What it builds | Key files |
|------|---------------|-----------|
| 1 | Schema learning + validation | `data_quality/schema.py` |
| 2 | Statistical profiler + reference profile | `data_quality/profiler.py`, `reference_profile.py` |
| 3 | Distribution drift detection | `data_quality/drift.py` |
| 4 | Business rule engine + NFL rules | `data_quality/rules.py` |
| 5 | Unified validation report builder | `data_quality/report.py` |
| 6 | Streamlit: Data Quality Overview page | `pages/08_data_quality_overview.py` |
| 7 | Streamlit: Drift Monitor page | `pages/09_drift_monitor.py` |
| 8 | Streamlit: Validation Rules page | `pages/10_validation_rules.py` |

Tasks 1–5 are pure backend (testable, no UI). Tasks 6–8 are Streamlit pages.

## Integration with Training Pipeline

To use the data quality system in production, add to `scripts/train_fourth_down_model.py`:

```python
# After Step 1 (load data), before Step 2 (build features):
from data_quality.schema import learn_schema
from data_quality.profiler import profile_dataframe
from data_quality.reference_profile import create_reference_profile, save_reference_profile
from data_quality.rules import nfl_default_rules
from data_quality.report import build_validation_report

# Validate incoming data
schema = learn_schema(raw_df)
ref_profile = profile_dataframe(raw_df)
report = build_validation_report(raw_df, schema, rules=nfl_default_rules())

if report.status.value == "fail":
    print(report.summary())
    print("DATA QUALITY CHECK FAILED — aborting training.")
    sys.exit(1)

# Save reference profile alongside model for inference-time drift detection
save_reference_profile(
    create_reference_profile(ref_profile, name=f"training_{args.output_dir}"),
    Path(args.output_dir) / "reference_profile.json",
)
```

This ensures that every training run validates its input and snapshots the "expected" profile for future drift detection.
