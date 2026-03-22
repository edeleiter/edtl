# Unified ETL Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python monorepo where feature transforms are defined once in Ibis expressions, executed on Snowflake (training/batch) or DuckDB (real-time inference), exposed via FastAPI, and surfaced through a Streamlit dashboard — with a cross-backend validation harness to guarantee calculation parity.

**Architecture:** A shared `transforms` library defines all feature engineering as Ibis expression graphs. A `backends` module handles Snowflake and DuckDB connection/session management. FastAPI serves inference (real-time via DuckDB, batch via Snowflake). Streamlit provides forms and dashboards. A validation harness runs golden datasets through both backends and asserts parity within configurable tolerance. Reference data (lookup tables, normalization params) is snapshotted from Snowflake to Parquet for DuckDB at inference time.

**Tech Stack:**
- **Core:** Python 3.11+, Ibis 9.x, DuckDB, Snowflake (via ibis-snowflake)
- **API:** FastAPI, uvicorn, pydantic
- **Frontend:** Streamlit
- **Testing:** pytest, pytest-snapshot
- **Build:** uv (package manager), monorepo with namespace packages

---

## Project Structure

```
unified-etl/
├── pyproject.toml                          # Monorepo root, uv workspace
├── README.md
├── docs/
│   └── plans/
│       └── 2026-03-21-unified-etl-framework.md
│
├── packages/
│   ├── transforms/                         # Shared transform definitions
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── transforms/
│   │           ├── __init__.py
│   │           ├── registry.py             # Transform registry + metadata
│   │           ├── expressions.py          # Ibis expression definitions
│   │           ├── reference_data.py       # Reference table management
│   │           └── features/
│   │               ├── __init__.py
│   │               ├── numeric.py          # Numeric feature transforms
│   │               ├── categorical.py      # Categorical encoding transforms
│   │               └── temporal.py         # Time-based feature transforms
│   │
│   ├── backends/                           # Backend connection management
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── backends/
│   │           ├── __init__.py
│   │           ├── base.py                 # Abstract backend interface
│   │           ├── snowflake_backend.py    # Snowflake/Ibis connection
│   │           ├── duckdb_backend.py       # DuckDB/Ibis connection
│   │           └── config.py              # Backend config (env vars, secrets)
│   │
│   ├── api/                                # FastAPI inference service
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── api/
│   │           ├── __init__.py
│   │           ├── main.py                 # FastAPI app + lifespan
│   │           ├── routes/
│   │           │   ├── __init__.py
│   │           │   ├── inference.py        # Real-time inference endpoint
│   │           │   ├── batch.py            # Batch inference endpoint
│   │           │   └── health.py           # Health check
│   │           ├── models.py               # Pydantic request/response models
│   │           └── dependencies.py         # FastAPI dependency injection
│   │
│   └── dashboard/                          # Streamlit frontend
│       ├── pyproject.toml
│       └── src/
│           └── dashboard/
│               ├── __init__.py
│               ├── app.py                  # Streamlit entrypoint
│               ├── pages/
│               │   ├── 01_data_explorer.py
│               │   ├── 02_feature_preview.py
│               │   └── 03_inference_form.py
│               └── components/
│                   ├── __init__.py
│                   └── feature_table.py
│
├── validation/                             # Cross-backend validation harness
│   ├── pyproject.toml
│   └── src/
│       └── validation/
│           ├── __init__.py
│           ├── harness.py                  # Run golden data through both backends
│           ├── comparator.py               # DataFrame diff with tolerance
│           └── golden_data/
│               └── .gitkeep
│
├── snowflake/                              # Snowflake deployment artifacts
│   ├── deploy.py                           # Upload UDFs/SPs to Snowflake
│   ├── tasks.sql                           # Snowflake Task definitions
│   └── stages.sql                          # Stage for reference data Parquet
│
├── reference_data/                         # Snapshotted reference tables (Parquet)
│   └── .gitkeep
│
└── tests/
    ├── conftest.py                         # Shared fixtures (DuckDB in-memory, mock Snowflake)
    ├── transforms/
    │   ├── test_numeric.py
    │   ├── test_categorical.py
    │   └── test_temporal.py
    ├── backends/
    │   ├── test_duckdb_backend.py
    │   └── test_snowflake_backend.py
    ├── api/
    │   ├── test_inference.py
    │   └── test_batch.py
    ├── dashboard/
    │   └── test_feature_preview.py
    └── validation/
        └── test_harness.py
```

---

## Task 1: Monorepo Scaffolding + uv Workspace

**Files:**
- Create: `pyproject.toml` (root)
- Create: `packages/transforms/pyproject.toml`
- Create: `packages/backends/pyproject.toml`
- Create: `packages/api/pyproject.toml`
- Create: `packages/dashboard/pyproject.toml`
- Create: `validation/pyproject.toml`
- Create: `README.md`

**Step 1: Create root pyproject.toml with uv workspace**

```toml
# pyproject.toml
[project]
name = "unified-etl"
version = "0.1.0"
description = "Unified ETL framework: Ibis transforms on Snowflake + DuckDB"
requires-python = ">=3.11"

[tool.uv.workspace]
members = [
    "packages/transforms",
    "packages/backends",
    "packages/api",
    "packages/dashboard",
    "validation",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = [
    "packages/transforms/src",
    "packages/backends/src",
    "packages/api/src",
    "packages/dashboard/src",
    "validation/src",
]
```

**Step 2: Create transforms package pyproject.toml**

```toml
# packages/transforms/pyproject.toml
[project]
name = "transforms"
version = "0.1.0"
description = "Shared Ibis transform definitions"
requires-python = ">=3.11"
dependencies = [
    "ibis-framework[duckdb,snowflake]>=9.0.0",
    "pyarrow>=14.0.0",
]
```

**Step 3: Create backends package pyproject.toml**

```toml
# packages/backends/pyproject.toml
[project]
name = "backends"
version = "0.1.0"
description = "Backend connection management for Snowflake and DuckDB"
requires-python = ">=3.11"
dependencies = [
    "transforms",
    "ibis-framework[duckdb,snowflake]>=9.0.0",
    "pydantic-settings>=2.0.0",
]

[tool.uv.sources]
transforms = { workspace = true }
```

**Step 4: Create api package pyproject.toml**

```toml
# packages/api/pyproject.toml
[project]
name = "api"
version = "0.1.0"
description = "FastAPI inference service"
requires-python = ">=3.11"
dependencies = [
    "transforms",
    "backends",
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.0.0",
]

[tool.uv.sources]
transforms = { workspace = true }
backends = { workspace = true }
```

**Step 5: Create dashboard package pyproject.toml**

```toml
# packages/dashboard/pyproject.toml
[project]
name = "dashboard"
version = "0.1.0"
description = "Streamlit dashboard and forms"
requires-python = ">=3.11"
dependencies = [
    "transforms",
    "backends",
    "streamlit>=1.35.0",
    "plotly>=5.18.0",
]

[tool.uv.sources]
transforms = { workspace = true }
backends = { workspace = true }
```

**Step 6: Create validation package pyproject.toml**

```toml
# validation/pyproject.toml
[project]
name = "validation"
version = "0.1.0"
description = "Cross-backend validation harness"
requires-python = ">=3.11"
dependencies = [
    "transforms",
    "backends",
    "pandas>=2.0.0",
    "pyarrow>=14.0.0",
]

[tool.uv.sources]
transforms = { workspace = true }
backends = { workspace = true }
```

**Step 7: Create all __init__.py files and .gitkeep files**

Create empty `__init__.py` in every package directory listed in the project structure. Create `.gitkeep` in `validation/src/validation/golden_data/` and `reference_data/`.

**Step 8: Create README.md**

```markdown
# Unified ETL Framework

Ibis-based feature transforms that run identically on Snowflake (training/batch)
and DuckDB (real-time inference), with FastAPI serving and Streamlit dashboard.

## Quick Start

```bash
uv sync
uv run pytest
uv run uvicorn api.main:app --reload        # API
uv run streamlit run packages/dashboard/src/dashboard/app.py  # Dashboard
```

## Architecture

- `packages/transforms/` — Ibis expression definitions (the source of truth)
- `packages/backends/` — Snowflake + DuckDB connection management
- `packages/api/` — FastAPI inference service
- `packages/dashboard/` — Streamlit frontend
- `validation/` — Cross-backend parity harness
- `snowflake/` — Deployment scripts for UDFs, SPs, and Tasks
```

**Step 9: Initialize git repo**

```bash
git init
git add -A
git commit -m "chore: scaffold monorepo with uv workspace"
```

---

## Task 2: Backend Interface + DuckDB Implementation

**Files:**
- Create: `packages/backends/src/backends/__init__.py`
- Create: `packages/backends/src/backends/base.py`
- Create: `packages/backends/src/backends/config.py`
- Create: `packages/backends/src/backends/duckdb_backend.py`
- Test: `tests/backends/test_duckdb_backend.py`

**Step 1: Write the failing test for DuckDB backend**

```python
# tests/backends/test_duckdb_backend.py
import ibis
import pytest

from backends.base import Backend
from backends.duckdb_backend import DuckDBBackend


def test_duckdb_backend_is_backend():
    backend = DuckDBBackend()
    assert isinstance(backend, Backend)


def test_duckdb_backend_returns_ibis_connection():
    backend = DuckDBBackend()
    conn = backend.connect()
    assert hasattr(conn, "table")
    assert hasattr(conn, "execute")


def test_duckdb_backend_execute_expression():
    backend = DuckDBBackend()
    conn = backend.connect()
    # Create an in-memory table and query it
    t = ibis.memtable({"x": [1, 2, 3], "y": [10, 20, 30]})
    result = conn.execute(t.mutate(z=t.x + t.y))
    assert list(result["z"]) == [11, 22, 33]


def test_duckdb_backend_load_parquet(tmp_path):
    """Backend can load reference data from Parquet files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Write a test parquet file
    table = pa.table({"id": [1, 2], "name": ["a", "b"]})
    path = tmp_path / "ref.parquet"
    pq.write_table(table, path)

    backend = DuckDBBackend()
    conn = backend.connect()
    ref_table = backend.load_reference_table(conn, "ref_test", str(path))
    result = conn.execute(ref_table)
    assert len(result) == 2
    assert list(result["name"]) == ["a", "b"]


def test_duckdb_backend_disconnect():
    backend = DuckDBBackend()
    conn = backend.connect()
    backend.disconnect(conn)
    # After disconnect, connection should not be usable
    # (DuckDB may or may not raise — just verify disconnect doesn't error)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/backends/test_duckdb_backend.py -v
```

Expected: FAIL — modules don't exist yet.

**Step 3: Implement base backend interface**

```python
# packages/backends/src/backends/base.py
from abc import ABC, abstractmethod
from typing import Any

import ibis
import ibis.expr.types as ir


class Backend(ABC):
    """Abstract backend interface for Ibis connections."""

    @abstractmethod
    def connect(self) -> ibis.BaseBackend:
        """Create and return an Ibis backend connection."""
        ...

    @abstractmethod
    def disconnect(self, conn: ibis.BaseBackend) -> None:
        """Close the connection."""
        ...

    @abstractmethod
    def load_reference_table(
        self,
        conn: ibis.BaseBackend,
        table_name: str,
        parquet_path: str,
    ) -> ir.Table:
        """Load a reference table from a Parquet file into the backend.

        Returns an Ibis table expression pointing at the loaded data.
        """
        ...
```

**Step 4: Implement config**

```python
# packages/backends/src/backends/config.py
from pydantic_settings import BaseSettings


class SnowflakeConfig(BaseSettings):
    """Snowflake connection settings from environment variables."""

    model_config = {"env_prefix": "SNOWFLAKE_"}

    account: str = ""
    user: str = ""
    password: str = ""
    database: str = ""
    schema_name: str = "PUBLIC"  # 'schema' conflicts with pydantic
    warehouse: str = ""
    role: str = ""


class DuckDBConfig(BaseSettings):
    """DuckDB connection settings."""

    model_config = {"env_prefix": "DUCKDB_"}

    database: str = ":memory:"
    threads: int = 4
```

**Step 5: Implement DuckDB backend**

```python
# packages/backends/src/backends/duckdb_backend.py
import ibis
import ibis.expr.types as ir

from backends.base import Backend
from backends.config import DuckDBConfig


class DuckDBBackend(Backend):
    """DuckDB backend for local/real-time inference."""

    def __init__(self, config: DuckDBConfig | None = None):
        self.config = config or DuckDBConfig()

    def connect(self) -> ibis.BaseBackend:
        conn = ibis.duckdb.connect(
            database=self.config.database,
            threads=self.config.threads,
        )
        return conn

    def disconnect(self, conn: ibis.BaseBackend) -> None:
        conn.disconnect()

    def load_reference_table(
        self,
        conn: ibis.BaseBackend,
        table_name: str,
        parquet_path: str,
    ) -> ir.Table:
        conn.read_parquet(parquet_path, table_name=table_name)
        return conn.table(table_name)
```

**Step 6: Update backends __init__.py**

```python
# packages/backends/src/backends/__init__.py
from backends.base import Backend
from backends.duckdb_backend import DuckDBBackend

__all__ = ["Backend", "DuckDBBackend"]
```

**Step 7: Run tests to verify they pass**

```bash
uv run pytest tests/backends/test_duckdb_backend.py -v
```

Expected: All PASS.

**Step 8: Commit**

```bash
git add -A
git commit -m "feat: add backend interface + DuckDB implementation"
```

---

## Task 3: Snowflake Backend Implementation

**Files:**
- Create: `packages/backends/src/backends/snowflake_backend.py`
- Modify: `packages/backends/src/backends/__init__.py`
- Test: `tests/backends/test_snowflake_backend.py`

> **Note:** Snowflake tests require live credentials. These tests should be marked with `@pytest.mark.snowflake` and skipped in CI unless credentials are available.

**Step 1: Write the failing test**

```python
# tests/backends/test_snowflake_backend.py
import os

import pytest

from backends.base import Backend
from backends.snowflake_backend import SnowflakeBackend


# Skip all tests in this module if SNOWFLAKE_ACCOUNT not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("SNOWFLAKE_ACCOUNT"),
    reason="SNOWFLAKE_ACCOUNT not set — skipping live Snowflake tests",
)


def test_snowflake_backend_is_backend():
    backend = SnowflakeBackend()
    assert isinstance(backend, Backend)


def test_snowflake_backend_connects():
    backend = SnowflakeBackend()
    conn = backend.connect()
    # Verify we can execute a trivial query
    result = conn.raw_sql("SELECT 1 AS x").fetchone()
    assert result[0] == 1
    backend.disconnect(conn)
```

**Step 2: Run tests to verify they fail (or skip)**

```bash
uv run pytest tests/backends/test_snowflake_backend.py -v
```

Expected: SKIP (no credentials) or FAIL (module missing).

**Step 3: Implement Snowflake backend**

```python
# packages/backends/src/backends/snowflake_backend.py
import ibis
import ibis.expr.types as ir

from backends.base import Backend
from backends.config import SnowflakeConfig


class SnowflakeBackend(Backend):
    """Snowflake backend for training-time ETL and batch inference."""

    def __init__(self, config: SnowflakeConfig | None = None):
        self.config = config or SnowflakeConfig()

    def connect(self) -> ibis.BaseBackend:
        conn = ibis.snowflake.connect(
            account=self.config.account,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            schema=self.config.schema_name,
            warehouse=self.config.warehouse,
            role=self.config.role if self.config.role else None,
        )
        return conn

    def disconnect(self, conn: ibis.BaseBackend) -> None:
        conn.disconnect()

    def load_reference_table(
        self,
        conn: ibis.BaseBackend,
        table_name: str,
        parquet_path: str,
    ) -> ir.Table:
        # For Snowflake, reference tables live in the warehouse already.
        # This method assumes the table exists; use deploy.py to upload
        # Parquet to a stage and COPY INTO the table.
        return conn.table(table_name)
```

**Step 4: Update __init__.py**

```python
# packages/backends/src/backends/__init__.py
from backends.base import Backend
from backends.duckdb_backend import DuckDBBackend
from backends.snowflake_backend import SnowflakeBackend

__all__ = ["Backend", "DuckDBBackend", "SnowflakeBackend"]
```

**Step 5: Run tests**

```bash
uv run pytest tests/backends/ -v
```

Expected: DuckDB tests PASS, Snowflake tests SKIP.

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: add Snowflake backend implementation"
```

---

## Task 4: Transform Registry + First Feature Module

**Files:**
- Create: `packages/transforms/src/transforms/__init__.py`
- Create: `packages/transforms/src/transforms/registry.py`
- Create: `packages/transforms/src/transforms/expressions.py`
- Create: `packages/transforms/src/transforms/features/__init__.py`
- Create: `packages/transforms/src/transforms/features/numeric.py`
- Test: `tests/transforms/test_numeric.py`

**Step 1: Write the failing test for numeric transforms**

```python
# tests/transforms/test_numeric.py
import ibis
import pytest

from transforms.features.numeric import (
    zscore_normalize,
    log_transform,
    clip_outliers,
    ratio,
)
from transforms.registry import TransformRegistry


@pytest.fixture
def con():
    return ibis.duckdb.connect()


@pytest.fixture
def sample_table(con):
    return ibis.memtable(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
            "numerator": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "denominator": [2.0, 4.0, 5.0, 8.0, 10.0, 0.0],
        }
    )


def test_zscore_normalize(con, sample_table):
    expr = zscore_normalize(sample_table, "value")
    result = con.execute(expr)
    # z-score of mean should be close to 0
    assert abs(result["value_zscore"].mean()) < 1e-10


def test_log_transform(con, sample_table):
    expr = log_transform(sample_table, "value")
    result = con.execute(expr)
    assert result["value_log"].iloc[0] == pytest.approx(0.0, abs=1e-10)  # ln(1) = 0


def test_clip_outliers(con, sample_table):
    expr = clip_outliers(sample_table, "value", lower=1.0, upper=10.0)
    result = con.execute(expr)
    assert result["value_clipped"].max() == 10.0
    assert result["value_clipped"].min() == 1.0


def test_ratio_with_zero_denominator(con, sample_table):
    expr = ratio(sample_table, "numerator", "denominator")
    result = con.execute(expr)
    # Zero denominator should produce null, not inf
    assert result["numerator_denominator_ratio"].iloc[-1] is None or (
        import_pandas_and_check_na(result["numerator_denominator_ratio"].iloc[-1])
    )


def import_pandas_and_check_na(val):
    import pandas as pd
    return pd.isna(val)


def test_transforms_are_registered():
    registry = TransformRegistry()
    registry.register("zscore", zscore_normalize)
    assert "zscore" in registry
    assert registry.get("zscore") is zscore_normalize


def test_registry_list_all():
    registry = TransformRegistry()
    registry.register("zscore", zscore_normalize)
    registry.register("log", log_transform)
    names = registry.list_transforms()
    assert "zscore" in names
    assert "log" in names
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/transforms/test_numeric.py -v
```

Expected: FAIL — modules don't exist.

**Step 3: Implement the transform registry**

```python
# packages/transforms/src/transforms/registry.py
from typing import Callable, Any


class TransformRegistry:
    """Registry for named transform functions.

    Each transform is a callable that takes an Ibis table expression
    and returns a new Ibis table expression with additional/modified columns.
    """

    def __init__(self):
        self._transforms: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._transforms[name] = fn

    def get(self, name: str) -> Callable:
        return self._transforms[name]

    def __contains__(self, name: str) -> bool:
        return name in self._transforms

    def list_transforms(self) -> list[str]:
        return list(self._transforms.keys())
```

**Step 4: Implement numeric feature transforms**

```python
# packages/transforms/src/transforms/features/numeric.py
"""Numeric feature transforms — all defined as Ibis expressions.

Each function takes an Ibis table and returns a new table with
the transformed column(s) added. The expressions are backend-agnostic:
they compile to SQL on Snowflake and execute natively on DuckDB.
"""

import ibis
import ibis.expr.types as ir


def zscore_normalize(table: ir.Table, column: str) -> ir.Table:
    """Add a z-score normalized column: (x - mean) / stddev."""
    col = table[column]
    mean = col.mean()
    std = col.std()
    return table.mutate(**{f"{column}_zscore": (col - mean) / std})


def log_transform(table: ir.Table, column: str) -> ir.Table:
    """Add a natural log transformed column. Assumes positive values."""
    col = table[column]
    return table.mutate(**{f"{column}_log": col.log()})


def clip_outliers(
    table: ir.Table, column: str, lower: float, upper: float
) -> ir.Table:
    """Add a clipped column bounded by [lower, upper]."""
    col = table[column]
    clipped = ibis.greatest(ibis.least(col, upper), lower)
    return table.mutate(**{f"{column}_clipped": clipped})


def ratio(
    table: ir.Table, numerator_col: str, denominator_col: str
) -> ir.Table:
    """Add a ratio column. Returns null when denominator is zero."""
    num = table[numerator_col]
    den = table[denominator_col]
    safe_ratio = ibis.case().when(den != 0, num / den).else_(ibis.null()).end()
    return table.mutate(
        **{f"{numerator_col}_{denominator_col}_ratio": safe_ratio}
    )
```

**Step 5: Create __init__.py files**

```python
# packages/transforms/src/transforms/__init__.py
from transforms.registry import TransformRegistry

__all__ = ["TransformRegistry"]
```

```python
# packages/transforms/src/transforms/features/__init__.py
```

**Step 6: Implement expressions.py (pipeline composer)**

```python
# packages/transforms/src/transforms/expressions.py
"""Compose multiple transforms into a pipeline."""

from typing import Callable

import ibis.expr.types as ir


def apply_pipeline(
    table: ir.Table,
    transforms: list[Callable[..., ir.Table]],
) -> ir.Table:
    """Apply a sequence of transforms to a table expression.

    Each transform is a callable (table, **kwargs) -> table.
    For transforms that need extra args, use functools.partial.
    """
    result = table
    for transform in transforms:
        result = transform(result)
    return result
```

**Step 7: Run tests to verify they pass**

```bash
uv run pytest tests/transforms/test_numeric.py -v
```

Expected: All PASS.

**Step 8: Commit**

```bash
git add -A
git commit -m "feat: add transform registry + numeric feature transforms"
```

---

## Task 5: Categorical and Temporal Transform Modules

**Files:**
- Create: `packages/transforms/src/transforms/features/categorical.py`
- Create: `packages/transforms/src/transforms/features/temporal.py`
- Test: `tests/transforms/test_categorical.py`
- Test: `tests/transforms/test_temporal.py`

**Step 1: Write failing tests for categorical transforms**

```python
# tests/transforms/test_categorical.py
import ibis
import pytest

from transforms.features.categorical import one_hot_flag, label_encode_from_map


@pytest.fixture
def con():
    return ibis.duckdb.connect()


@pytest.fixture
def sample_table():
    return ibis.memtable(
        {
            "color": ["red", "blue", "green", "red", "blue"],
            "size": ["S", "M", "L", "XL", "M"],
        }
    )


def test_one_hot_flag(con, sample_table):
    expr = one_hot_flag(sample_table, "color", "red")
    result = con.execute(expr)
    assert list(result["color_is_red"]) == [1, 0, 0, 1, 0]


def test_label_encode_from_map(con, sample_table):
    mapping = {"S": 0, "M": 1, "L": 2, "XL": 3}
    expr = label_encode_from_map(sample_table, "size", mapping)
    result = con.execute(expr)
    assert list(result["size_encoded"]) == [0, 1, 2, 3, 1]
```

**Step 2: Write failing tests for temporal transforms**

```python
# tests/transforms/test_temporal.py
import datetime

import ibis
import pytest

from transforms.features.temporal import (
    extract_dow,
    extract_hour,
    days_since,
)


@pytest.fixture
def con():
    return ibis.duckdb.connect()


@pytest.fixture
def sample_table():
    return ibis.memtable(
        {
            "event_ts": [
                datetime.datetime(2025, 1, 6, 14, 30),   # Monday
                datetime.datetime(2025, 1, 7, 9, 0),     # Tuesday
                datetime.datetime(2025, 1, 11, 22, 15),   # Saturday
            ],
        }
    )


def test_extract_dow(con, sample_table):
    expr = extract_dow(sample_table, "event_ts")
    result = con.execute(expr)
    # Ibis day_of_week.index(): Monday=0, Sunday=6
    assert list(result["event_ts_dow"]) == [0, 1, 5]


def test_extract_hour(con, sample_table):
    expr = extract_hour(sample_table, "event_ts")
    result = con.execute(expr)
    assert list(result["event_ts_hour"]) == [14, 9, 22]


def test_days_since(con, sample_table):
    ref_date = datetime.date(2025, 1, 1)
    expr = days_since(sample_table, "event_ts", ref_date)
    result = con.execute(expr)
    assert list(result["event_ts_days_since"]) == [5, 6, 10]
```

**Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/transforms/test_categorical.py tests/transforms/test_temporal.py -v
```

Expected: FAIL — modules don't exist.

**Step 4: Implement categorical transforms**

```python
# packages/transforms/src/transforms/features/categorical.py
"""Categorical feature transforms as Ibis expressions."""

import ibis
import ibis.expr.types as ir


def one_hot_flag(table: ir.Table, column: str, value: str) -> ir.Table:
    """Add a binary flag column: 1 if column == value, else 0."""
    flag = ibis.case().when(table[column] == value, 1).else_(0).end()
    return table.mutate(**{f"{column}_is_{value}": flag})


def label_encode_from_map(
    table: ir.Table, column: str, mapping: dict[str, int]
) -> ir.Table:
    """Encode a categorical column using an explicit mapping dict.

    Unmapped values become null. The mapping is typically derived from
    training data and snapshotted as reference data.
    """
    case_expr = ibis.case()
    for label, code in mapping.items():
        case_expr = case_expr.when(table[column] == label, code)
    encoded = case_expr.else_(ibis.null()).end()
    return table.mutate(**{f"{column}_encoded": encoded})
```

**Step 5: Implement temporal transforms**

```python
# packages/transforms/src/transforms/features/temporal.py
"""Temporal feature transforms as Ibis expressions."""

import datetime

import ibis
import ibis.expr.types as ir


def extract_dow(table: ir.Table, column: str) -> ir.Table:
    """Extract day of week (Monday=0, Sunday=6)."""
    dow = table[column].day_of_week.index()
    return table.mutate(**{f"{column}_dow": dow})


def extract_hour(table: ir.Table, column: str) -> ir.Table:
    """Extract hour of day (0-23)."""
    hour = table[column].hour()
    return table.mutate(**{f"{column}_hour": hour})


def days_since(
    table: ir.Table, column: str, reference_date: datetime.date
) -> ir.Table:
    """Compute days elapsed since a reference date."""
    delta = table[column].date() - reference_date
    # Ibis returns an interval; extract days as integer
    return table.mutate(
        **{f"{column}_days_since": delta.cast("int64")}
    )
```

> **Implementation note for the engineer:** The `days_since` function casts an interval to int64. This works on DuckDB. If Snowflake handles the interval → integer cast differently, this is exactly the kind of thing the validation harness (Task 7) will catch. The fix would be to use `ibis.greatest(...)` or explicit `.epoch_seconds() / 86400` instead. Test against both backends and adjust.

**Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/transforms/ -v
```

Expected: All PASS.

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: add categorical + temporal transform modules"
```

---

## Task 6: Reference Data Management

**Files:**
- Create: `packages/transforms/src/transforms/reference_data.py`
- Test: `tests/transforms/test_reference_data.py` (was listed in structure but not yet created)

**Step 1: Write the failing test**

```python
# tests/transforms/test_reference_data.py
import ibis
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from transforms.reference_data import (
    snapshot_table_to_parquet,
    load_reference_tables,
    ReferenceDataManifest,
)


@pytest.fixture
def con():
    return ibis.duckdb.connect()


@pytest.fixture
def ref_dir(tmp_path):
    # Write two reference parquet files
    pq.write_table(
        pa.table({"category": ["A", "B", "C"], "code": [0, 1, 2]}),
        tmp_path / "category_map.parquet",
    )
    pq.write_table(
        pa.table({"feature": ["x"], "mean": [5.0], "std": [2.0]}),
        tmp_path / "normalization_params.parquet",
    )
    return tmp_path


def test_manifest_from_directory(ref_dir):
    manifest = ReferenceDataManifest.from_directory(ref_dir)
    assert "category_map" in manifest.tables
    assert "normalization_params" in manifest.tables


def test_load_reference_tables_into_duckdb(con, ref_dir):
    manifest = ReferenceDataManifest.from_directory(ref_dir)
    loaded = load_reference_tables(con, manifest)
    assert "category_map" in loaded
    result = con.execute(loaded["category_map"])
    assert len(result) == 3
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/transforms/test_reference_data.py -v
```

Expected: FAIL.

**Step 3: Implement reference data management**

```python
# packages/transforms/src/transforms/reference_data.py
"""Reference data management.

Reference tables (lookup maps, normalization parameters, etc.) are
computed at training time in Snowflake, snapshotted to Parquet, and
loaded into DuckDB at inference API startup.
"""

from dataclasses import dataclass, field
from pathlib import Path

import ibis
import ibis.expr.types as ir


@dataclass
class ReferenceDataManifest:
    """Manifest of available reference data Parquet files.

    Table names are derived from filenames (stem without extension).
    """

    tables: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "ReferenceDataManifest":
        directory = Path(directory)
        tables = {}
        for path in sorted(directory.glob("*.parquet")):
            table_name = path.stem
            tables[table_name] = path
        return cls(tables=tables)


def load_reference_tables(
    conn: ibis.BaseBackend,
    manifest: ReferenceDataManifest,
) -> dict[str, ir.Table]:
    """Load all reference tables from manifest into the given backend.

    For DuckDB, this reads Parquet files directly.
    Returns a dict mapping table_name -> Ibis table expression.
    """
    loaded = {}
    for table_name, path in manifest.tables.items():
        conn.read_parquet(str(path), table_name=table_name)
        loaded[table_name] = conn.table(table_name)
    return loaded


def snapshot_table_to_parquet(
    conn: ibis.BaseBackend,
    table_name: str,
    output_path: str | Path,
) -> Path:
    """Snapshot a table from the backend to a local Parquet file.

    Typically called against a Snowflake connection at training time
    to create reference data files for inference.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = conn.table(table_name)
    # Execute and write via PyArrow
    result = conn.execute(table)
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow_table = pa.Table.from_pandas(result)
    pq.write_table(arrow_table, output_path)
    return output_path
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/transforms/test_reference_data.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add reference data manifest + Parquet loading"
```

---

## Task 7: Cross-Backend Validation Harness

This is the critical component that guarantees training/inference parity.

**Files:**
- Create: `validation/src/validation/__init__.py`
- Create: `validation/src/validation/comparator.py`
- Create: `validation/src/validation/harness.py`
- Test: `tests/validation/test_harness.py`

**Step 1: Write the failing test**

```python
# tests/validation/test_harness.py
import datetime

import ibis
import pytest

from transforms.features.numeric import zscore_normalize, clip_outliers, ratio
from transforms.features.categorical import one_hot_flag
from transforms.features.temporal import extract_hour
from validation.comparator import compare_dataframes
from validation.harness import validate_transform_parity


@pytest.fixture
def golden_data():
    """A small dataset that exercises edge cases."""
    return {
        "value": [0.0, 1.0, -1.0, 100.0, 0.001],
        "numerator": [10.0, 0.0, -5.0, 100.0, 1.0],
        "denominator": [2.0, 0.0, 3.0, 0.001, 1.0],
        "color": ["red", "blue", "red", "green", "blue"],
        "event_ts": [
            datetime.datetime(2025, 1, 6, 0, 0),
            datetime.datetime(2025, 6, 15, 12, 30),
            datetime.datetime(2025, 12, 31, 23, 59),
            datetime.datetime(2025, 3, 1, 6, 0),
            datetime.datetime(2025, 7, 4, 18, 45),
        ],
    }


def test_comparator_identical():
    import pandas as pd

    df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = compare_dataframes(df1, df2)
    assert result.is_equal


def test_comparator_within_tolerance():
    import pandas as pd

    df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    df2 = pd.DataFrame({"a": [1.0000001, 2.0000001, 3.0000001]})
    result = compare_dataframes(df1, df2, atol=1e-6)
    assert result.is_equal


def test_comparator_detects_difference():
    import pandas as pd

    df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    df2 = pd.DataFrame({"a": [1.0, 2.0, 999.0]})
    result = compare_dataframes(df1, df2)
    assert not result.is_equal
    assert "a" in result.differing_columns


def test_validate_parity_duckdb_only(golden_data):
    """Validate that the same transform produces identical results
    when run twice on DuckDB (sanity check of the harness itself)."""
    from functools import partial

    transform = partial(clip_outliers, column="value", lower=0.0, upper=50.0)
    result = validate_transform_parity(
        golden_data=golden_data,
        transform=transform,
        backend_a_factory=lambda: ibis.duckdb.connect(),
        backend_b_factory=lambda: ibis.duckdb.connect(),
    )
    assert result.is_equal, f"Parity check failed: {result.report()}"
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/validation/test_harness.py -v
```

Expected: FAIL.

**Step 3: Implement the comparator**

```python
# validation/src/validation/comparator.py
"""DataFrame comparison with configurable tolerance."""

from dataclasses import dataclass, field

import pandas as pd
import numpy as np


@dataclass
class ComparisonResult:
    """Result of comparing two DataFrames."""

    is_equal: bool
    differing_columns: list[str] = field(default_factory=list)
    max_abs_diff: dict[str, float] = field(default_factory=dict)
    row_count_match: bool = True
    column_match: bool = True
    missing_columns_a: list[str] = field(default_factory=list)
    missing_columns_b: list[str] = field(default_factory=list)

    def report(self) -> str:
        if self.is_equal:
            return "DataFrames are equal within tolerance."
        lines = ["PARITY CHECK FAILED:"]
        if not self.row_count_match:
            lines.append("  - Row counts differ")
        if not self.column_match:
            lines.append(f"  - Missing in A: {self.missing_columns_a}")
            lines.append(f"  - Missing in B: {self.missing_columns_b}")
        for col in self.differing_columns:
            diff = self.max_abs_diff.get(col, "N/A")
            lines.append(f"  - Column '{col}': max abs diff = {diff}")
        return "\n".join(lines)


def compare_dataframes(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> ComparisonResult:
    """Compare two DataFrames column-by-column with tolerance for numerics.

    Non-numeric columns are compared exactly.
    Null positions must match in both DataFrames.
    """
    # Check row counts
    if len(df_a) != len(df_b):
        return ComparisonResult(
            is_equal=False,
            row_count_match=False,
        )

    # Check column names
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    if cols_a != cols_b:
        return ComparisonResult(
            is_equal=False,
            column_match=False,
            missing_columns_a=sorted(cols_b - cols_a),
            missing_columns_b=sorted(cols_a - cols_b),
        )

    # Sort columns for consistent comparison
    columns = sorted(df_a.columns)
    differing = []
    max_diffs = {}

    for col in columns:
        a = df_a[col]
        b = df_b[col]

        # Check null positions match
        null_match = a.isna().equals(b.isna())
        if not null_match:
            differing.append(col)
            max_diffs[col] = float("inf")
            continue

        # Drop nulls for value comparison
        mask = a.notna()
        a_vals = a[mask].reset_index(drop=True)
        b_vals = b[mask].reset_index(drop=True)

        if len(a_vals) == 0:
            continue

        if pd.api.types.is_numeric_dtype(a_vals):
            abs_diff = np.abs(a_vals.to_numpy() - b_vals.to_numpy())
            max_diff = float(abs_diff.max())
            threshold = atol + rtol * np.abs(a_vals.to_numpy())
            if (abs_diff > threshold).any():
                differing.append(col)
                max_diffs[col] = max_diff
        else:
            if not a_vals.equals(b_vals):
                differing.append(col)
                max_diffs[col] = float("nan")

    return ComparisonResult(
        is_equal=len(differing) == 0,
        differing_columns=differing,
        max_abs_diff=max_diffs,
    )
```

**Step 4: Implement the harness**

```python
# validation/src/validation/harness.py
"""Cross-backend validation harness.

Runs the same Ibis transform on two backends and compares results.
Typical usage: backend_a = DuckDB, backend_b = Snowflake.
For CI without Snowflake creds, run DuckDB vs DuckDB as a sanity check.
"""

from typing import Callable

import ibis
import ibis.expr.types as ir
import pandas as pd

from validation.comparator import ComparisonResult, compare_dataframes


def validate_transform_parity(
    golden_data: dict,
    transform: Callable[[ir.Table], ir.Table],
    backend_a_factory: Callable[[], ibis.BaseBackend],
    backend_b_factory: Callable[[], ibis.BaseBackend],
    atol: float = 1e-8,
    rtol: float = 1e-5,
    sort_by: str | list[str] | None = None,
) -> ComparisonResult:
    """Run a transform on both backends and compare results.

    Args:
        golden_data: Dict of column_name -> list of values.
        transform: Callable (ibis.Table) -> ibis.Table.
        backend_a_factory: Callable that returns an Ibis connection.
        backend_b_factory: Callable that returns an Ibis connection.
        atol: Absolute tolerance for numeric comparison.
        rtol: Relative tolerance for numeric comparison.
        sort_by: Column(s) to sort by before comparison (eliminates
                 ordering differences between backends).

    Returns:
        ComparisonResult with parity details.
    """
    # Run on backend A
    conn_a = backend_a_factory()
    table_a = ibis.memtable(golden_data)
    result_a_expr = transform(table_a)
    df_a: pd.DataFrame = conn_a.execute(result_a_expr)

    # Run on backend B
    conn_b = backend_b_factory()
    table_b = ibis.memtable(golden_data)
    result_b_expr = transform(table_b)
    df_b: pd.DataFrame = conn_b.execute(result_b_expr)

    # Sort to eliminate ordering differences
    if sort_by:
        if isinstance(sort_by, str):
            sort_by = [sort_by]
        df_a = df_a.sort_values(sort_by).reset_index(drop=True)
        df_b = df_b.sort_values(sort_by).reset_index(drop=True)

    return compare_dataframes(df_a, df_b, atol=atol, rtol=rtol)
```

```python
# validation/src/validation/__init__.py
from validation.harness import validate_transform_parity
from validation.comparator import compare_dataframes, ComparisonResult

__all__ = ["validate_transform_parity", "compare_dataframes", "ComparisonResult"]
```

**Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/validation/ -v
```

Expected: All PASS.

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: add cross-backend validation harness"
```

---

## Task 8: FastAPI Inference Service

**Files:**
- Create: `packages/api/src/api/__init__.py`
- Create: `packages/api/src/api/main.py`
- Create: `packages/api/src/api/models.py`
- Create: `packages/api/src/api/dependencies.py`
- Create: `packages/api/src/api/routes/__init__.py`
- Create: `packages/api/src/api/routes/health.py`
- Create: `packages/api/src/api/routes/inference.py`
- Create: `packages/api/src/api/routes/batch.py`
- Test: `tests/api/test_inference.py`

**Step 1: Write the failing test**

```python
# tests/api/test_inference.py
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    """Create a test client with DuckDB backend and no reference data."""
    import os
    os.environ["DUCKDB_DATABASE"] = ":memory:"
    os.environ["REFERENCE_DATA_DIR"] = str(tmp_path)

    from api.main import create_app
    app = create_app()
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_inference_single_row(client):
    payload = {
        "data": [
            {"value": 5.0, "numerator": 10.0, "denominator": 2.0}
        ],
        "transforms": ["clip_outliers"],
        "transform_params": {
            "clip_outliers": {"column": "value", "lower": 0.0, "upper": 10.0}
        },
    }
    resp = client.post("/inference/transform", json=payload)
    assert resp.status_code == 200
    result = resp.json()
    assert len(result["data"]) == 1
    assert "value_clipped" in result["data"][0]
    assert result["data"][0]["value_clipped"] == 5.0


def test_inference_multiple_transforms(client):
    payload = {
        "data": [
            {"value": 100.0, "numerator": 10.0, "denominator": 0.0}
        ],
        "transforms": ["clip_outliers", "ratio"],
        "transform_params": {
            "clip_outliers": {"column": "value", "lower": 0.0, "upper": 50.0},
            "ratio": {"numerator_col": "numerator", "denominator_col": "denominator"},
        },
    }
    resp = client.post("/inference/transform", json=payload)
    assert resp.status_code == 200
    result = resp.json()
    assert result["data"][0]["value_clipped"] == 50.0
    assert result["data"][0]["numerator_denominator_ratio"] is None


def test_inference_unknown_transform(client):
    payload = {
        "data": [{"value": 1.0}],
        "transforms": ["nonexistent_transform"],
        "transform_params": {},
    }
    resp = client.post("/inference/transform", json=payload)
    assert resp.status_code == 400
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/api/test_inference.py -v
```

Expected: FAIL.

**Step 3: Implement Pydantic models**

```python
# packages/api/src/api/models.py
from typing import Any

from pydantic import BaseModel


class TransformRequest(BaseModel):
    """Request to apply transforms to input data."""

    data: list[dict[str, Any]]
    transforms: list[str]
    transform_params: dict[str, dict[str, Any]] = {}


class TransformResponse(BaseModel):
    """Response with transformed data."""

    data: list[dict[str, Any]]
    transforms_applied: list[str]


class HealthResponse(BaseModel):
    status: str
    backend: str
```

**Step 4: Implement dependencies (backend + registry setup)**

```python
# packages/api/src/api/dependencies.py
"""FastAPI dependency injection for backend and transform registry."""

from functools import partial, lru_cache
from pathlib import Path
import os

import ibis

from backends.duckdb_backend import DuckDBBackend
from transforms.registry import TransformRegistry
from transforms.features.numeric import (
    zscore_normalize,
    log_transform,
    clip_outliers,
    ratio,
)
from transforms.features.categorical import one_hot_flag, label_encode_from_map
from transforms.features.temporal import extract_dow, extract_hour, days_since
from transforms.reference_data import ReferenceDataManifest, load_reference_tables


@lru_cache
def get_registry() -> TransformRegistry:
    """Build and return the transform registry.

    All available transforms are registered here. This is the single
    source of truth for what transforms the API can apply.
    """
    registry = TransformRegistry()
    registry.register("zscore_normalize", zscore_normalize)
    registry.register("log_transform", log_transform)
    registry.register("clip_outliers", clip_outliers)
    registry.register("ratio", ratio)
    registry.register("one_hot_flag", one_hot_flag)
    registry.register("extract_dow", extract_dow)
    registry.register("extract_hour", extract_hour)
    return registry


class AppState:
    """Holds the DuckDB connection and reference tables for the app lifetime."""

    def __init__(self):
        self.backend = DuckDBBackend()
        self.conn: ibis.BaseBackend | None = None
        self.reference_tables: dict = {}

    def startup(self):
        self.conn = self.backend.connect()
        ref_dir = os.environ.get("REFERENCE_DATA_DIR", "reference_data")
        ref_path = Path(ref_dir)
        if ref_path.exists() and any(ref_path.glob("*.parquet")):
            manifest = ReferenceDataManifest.from_directory(ref_path)
            self.reference_tables = load_reference_tables(self.conn, manifest)

    def shutdown(self):
        if self.conn:
            self.backend.disconnect(self.conn)
```

**Step 5: Implement route handlers**

```python
# packages/api/src/api/routes/__init__.py
```

```python
# packages/api/src/api/routes/health.py
from fastapi import APIRouter

from api.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", backend="duckdb")
```

```python
# packages/api/src/api/routes/inference.py
from functools import partial

import ibis
from fastapi import APIRouter, HTTPException, Request

from api.models import TransformRequest, TransformResponse
from api.dependencies import get_registry

router = APIRouter(prefix="/inference")


@router.post("/transform", response_model=TransformResponse)
def transform(req: TransformRequest, request: Request):
    registry = get_registry()
    app_state = request.app.state.app_state

    # Validate all transforms exist before executing
    for name in req.transforms:
        if name not in registry:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown transform: '{name}'. "
                f"Available: {registry.list_transforms()}",
            )

    # Load input data into DuckDB via ibis.memtable
    table = ibis.memtable(req.data)

    # Apply transforms sequentially
    for name in req.transforms:
        fn = registry.get(name)
        params = req.transform_params.get(name, {})
        table = fn(table, **params)

    # Execute on DuckDB and return
    result_df = app_state.conn.execute(table)
    # Convert NaN/NaT to None for JSON serialization
    result_df = result_df.where(result_df.notna(), None)
    data = result_df.to_dict(orient="records")

    return TransformResponse(
        data=data,
        transforms_applied=req.transforms,
    )
```

```python
# packages/api/src/api/routes/batch.py
from fastapi import APIRouter

router = APIRouter(prefix="/batch")


@router.post("/submit")
def submit_batch():
    """Submit a batch inference job (runs on Snowflake).

    TODO: Implement in a later task. This will:
    1. Accept a Snowflake table name as input
    2. Run transforms via Snowpark on Snowflake
    3. Write results to an output table
    4. Return a job ID for polling
    """
    return {"status": "not_implemented", "message": "Batch inference coming soon"}
```

**Step 6: Implement the FastAPI app factory**

```python
# packages/api/src/api/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.dependencies import AppState
from api.routes import health, inference, batch


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage DuckDB connection + reference data lifecycle."""
    app_state = AppState()
    app_state.startup()
    app.state.app_state = app_state
    yield
    app_state.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Unified ETL Inference API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(inference.router)
    app.include_router(batch.router)
    return app


# Default app instance for `uvicorn api.main:app`
app = create_app()
```

```python
# packages/api/src/api/__init__.py
```

**Step 7: Run tests to verify they pass**

```bash
uv run pytest tests/api/test_inference.py -v
```

Expected: All PASS.

**Step 8: Commit**

```bash
git add -A
git commit -m "feat: add FastAPI inference service with DuckDB backend"
```

---

## Task 9: Streamlit Dashboard

**Files:**
- Create: `packages/dashboard/src/dashboard/__init__.py`
- Create: `packages/dashboard/src/dashboard/app.py`
- Create: `packages/dashboard/src/dashboard/pages/01_data_explorer.py`
- Create: `packages/dashboard/src/dashboard/pages/02_feature_preview.py`
- Create: `packages/dashboard/src/dashboard/pages/03_inference_form.py`
- Create: `packages/dashboard/src/dashboard/components/__init__.py`
- Create: `packages/dashboard/src/dashboard/components/feature_table.py`

> **Note:** Streamlit apps are difficult to unit test. Instead, this task focuses on correct implementation. Manual verification: `uv run streamlit run packages/dashboard/src/dashboard/app.py`

**Step 1: Implement the main app entry point**

```python
# packages/dashboard/src/dashboard/app.py
"""Streamlit dashboard entrypoint.

Run with: uv run streamlit run packages/dashboard/src/dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Unified ETL Dashboard",
    page_icon="🔄",
    layout="wide",
)

st.title("Unified ETL Dashboard")
st.markdown(
    "Explore data, preview feature transforms, and submit inference requests."
)

st.sidebar.success("Select a page above.")

st.markdown("### Quick Links")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Data Explorer**\n\nBrowse raw data from Snowflake or local files.")
with col2:
    st.info("**Feature Preview**\n\nApply transforms interactively and inspect results.")
with col3:
    st.info("**Inference Form**\n\nSubmit data for real-time or batch inference.")
```

**Step 2: Implement the data explorer page**

```python
# packages/dashboard/src/dashboard/pages/01_data_explorer.py
"""Data Explorer page — browse tables and uploaded data."""

import streamlit as st
import ibis
import pandas as pd

st.header("Data Explorer")

data_source = st.radio(
    "Data source",
    ["Upload CSV", "Sample data"],
    horizontal=True,
)

if data_source == "Upload CSV":
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df, use_container_width=True)
        st.caption(f"{len(df)} rows × {len(df.columns)} columns")
        # Store in session state for other pages
        st.session_state["explorer_data"] = df
else:
    # Provide sample data for quick exploration
    sample = pd.DataFrame(
        {
            "value": [1.0, 2.5, 3.7, 100.0, 0.5],
            "category": ["A", "B", "A", "C", "B"],
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="D"),
            "numerator": [10, 20, 30, 40, 50],
            "denominator": [2, 0, 5, 10, 25],
        }
    )
    st.dataframe(sample, use_container_width=True)
    st.session_state["explorer_data"] = sample

# Basic stats
if "explorer_data" in st.session_state:
    df = st.session_state["explorer_data"]
    with st.expander("Column statistics"):
        st.dataframe(df.describe(), use_container_width=True)
```

**Step 3: Implement the feature preview page**

```python
# packages/dashboard/src/dashboard/pages/02_feature_preview.py
"""Feature Preview page — apply transforms interactively."""

import streamlit as st
import ibis
import pandas as pd

from transforms.features.numeric import zscore_normalize, clip_outliers, ratio, log_transform
from transforms.features.categorical import one_hot_flag
from transforms.features.temporal import extract_dow, extract_hour

st.header("Feature Preview")

if "explorer_data" not in st.session_state:
    st.warning("Load data in the Data Explorer first.")
    st.stop()

df = st.session_state["explorer_data"]
con = ibis.duckdb.connect()

# Show available transforms
TRANSFORMS = {
    "Z-Score Normalize": {"fn": zscore_normalize, "params": ["column"]},
    "Log Transform": {"fn": log_transform, "params": ["column"]},
    "Clip Outliers": {"fn": clip_outliers, "params": ["column", "lower", "upper"]},
    "Ratio": {"fn": ratio, "params": ["numerator_col", "denominator_col"]},
    "One-Hot Flag": {"fn": one_hot_flag, "params": ["column", "value"]},
    "Extract Day of Week": {"fn": extract_dow, "params": ["column"]},
    "Extract Hour": {"fn": extract_hour, "params": ["column"]},
}

selected = st.selectbox("Select transform", list(TRANSFORMS.keys()))
transform_info = TRANSFORMS[selected]

# Dynamic parameter inputs based on transform
params = {}
numeric_cols = list(df.select_dtypes(include="number").columns)
all_cols = list(df.columns)

for param in transform_info["params"]:
    if param == "column":
        params[param] = st.selectbox(f"{param}", all_cols, key=f"param_{param}")
    elif param in ("numerator_col", "denominator_col"):
        params[param] = st.selectbox(f"{param}", numeric_cols, key=f"param_{param}")
    elif param in ("lower", "upper"):
        default = 0.0 if param == "lower" else 100.0
        params[param] = st.number_input(f"{param}", value=default, key=f"param_{param}")
    elif param == "value":
        col_for_values = params.get("column", all_cols[0])
        unique_vals = df[col_for_values].unique().tolist() if col_for_values in df.columns else []
        params[param] = st.selectbox(f"{param}", unique_vals, key=f"param_{param}")

if st.button("Apply Transform", type="primary"):
    table = ibis.memtable(df)
    try:
        result_expr = transform_info["fn"](table, **params)
        result_df = con.execute(result_expr)
        st.dataframe(result_df, use_container_width=True)

        # Highlight new columns
        new_cols = [c for c in result_df.columns if c not in df.columns]
        if new_cols:
            st.success(f"New columns added: {', '.join(new_cols)}")
    except Exception as e:
        st.error(f"Transform failed: {e}")
```

**Step 4: Implement the inference form page**

```python
# packages/dashboard/src/dashboard/pages/03_inference_form.py
"""Inference Form page — submit data to the API for preprocessing."""

import streamlit as st
import requests
import pandas as pd
import json

st.header("Inference Form")

api_url = st.text_input("API URL", value="http://localhost:8000")

st.subheader("Input Data")

input_mode = st.radio("Input mode", ["Manual entry", "Upload CSV"], horizontal=True)

if input_mode == "Manual entry":
    st.caption("Enter JSON array of objects:")
    default_json = json.dumps(
        [{"value": 5.0, "numerator": 10.0, "denominator": 2.0}],
        indent=2,
    )
    raw_input = st.text_area("Data (JSON)", value=default_json, height=150)
    try:
        input_data = json.loads(raw_input)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        input_data = df.to_dict(orient="records")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Upload a CSV file to continue.")
        st.stop()

st.subheader("Transforms")

available_transforms = [
    "zscore_normalize", "log_transform", "clip_outliers",
    "ratio", "one_hot_flag", "extract_dow", "extract_hour",
]

selected_transforms = st.multiselect("Select transforms to apply", available_transforms)

# Collect parameters for each selected transform
transform_params = {}
for t in selected_transforms:
    with st.expander(f"Parameters for {t}"):
        if t == "clip_outliers":
            col = st.text_input("column", key=f"{t}_col")
            lower = st.number_input("lower", value=0.0, key=f"{t}_lower")
            upper = st.number_input("upper", value=100.0, key=f"{t}_upper")
            transform_params[t] = {"column": col, "lower": lower, "upper": upper}
        elif t == "ratio":
            num = st.text_input("numerator_col", key=f"{t}_num")
            den = st.text_input("denominator_col", key=f"{t}_den")
            transform_params[t] = {"numerator_col": num, "denominator_col": den}
        elif t in ("zscore_normalize", "log_transform", "extract_dow", "extract_hour"):
            col = st.text_input("column", key=f"{t}_col")
            transform_params[t] = {"column": col}
        elif t == "one_hot_flag":
            col = st.text_input("column", key=f"{t}_col")
            val = st.text_input("value", key=f"{t}_val")
            transform_params[t] = {"column": col, "value": val}

if st.button("Submit Inference Request", type="primary"):
    payload = {
        "data": input_data,
        "transforms": selected_transforms,
        "transform_params": transform_params,
    }
    try:
        resp = requests.post(f"{api_url}/inference/transform", json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            st.success(f"Transforms applied: {result['transforms_applied']}")
            st.dataframe(pd.DataFrame(result["data"]), use_container_width=True)
        else:
            st.error(f"API error {resp.status_code}: {resp.json().get('detail', resp.text)}")
    except requests.ConnectionError:
        st.error(f"Cannot connect to API at {api_url}. Is the server running?")
```

**Step 5: Create component placeholder**

```python
# packages/dashboard/src/dashboard/components/__init__.py
```

```python
# packages/dashboard/src/dashboard/components/feature_table.py
"""Reusable Streamlit component for displaying feature transform results."""

import streamlit as st
import pandas as pd


def show_feature_table(
    original: pd.DataFrame,
    transformed: pd.DataFrame,
    highlight_new: bool = True,
):
    """Display a transformed DataFrame with new columns highlighted."""
    new_cols = [c for c in transformed.columns if c not in original.columns]

    if highlight_new and new_cols:
        st.caption(f"New columns: {', '.join(new_cols)}")

    st.dataframe(
        transformed,
        use_container_width=True,
        column_config={
            col: st.column_config.NumberColumn(format="%.4f")
            for col in new_cols
            if pd.api.types.is_numeric_dtype(transformed[col])
        },
    )
```

```python
# packages/dashboard/src/dashboard/__init__.py
```

**Step 6: Manual verification**

```bash
# Terminal 1: Start the API
uv run uvicorn api.main:app --reload --port 8000

# Terminal 2: Start Streamlit
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Verify: navigate pages, upload CSV, preview transforms, submit inference.

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: add Streamlit dashboard with data explorer, feature preview, and inference form"
```

---

## Task 10: Snowflake Deployment Artifacts

**Files:**
- Create: `snowflake/deploy.py`
- Create: `snowflake/tasks.sql`
- Create: `snowflake/stages.sql`

**Step 1: Create stage definitions**

```sql
-- snowflake/stages.sql
-- Run once to set up stages for reference data and Python packages.

CREATE STAGE IF NOT EXISTS reference_data_stage
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Parquet files for reference data (lookup tables, normalization params)';

CREATE STAGE IF NOT EXISTS python_packages_stage
    COMMENT = 'Python packages for UDFs and stored procedures';
```

**Step 2: Create deployment script**

```python
# snowflake/deploy.py
"""Deploy transforms as Snowflake UDFs and stored procedures.

Usage:
    uv run python snowflake/deploy.py --action upload-reference
    uv run python snowflake/deploy.py --action create-sproc

Requires SNOWFLAKE_* environment variables to be set.
"""

import argparse
from pathlib import Path

import ibis

from backends.snowflake_backend import SnowflakeBackend
from transforms.reference_data import ReferenceDataManifest


def upload_reference_data(conn, ref_dir: str = "reference_data"):
    """Upload local Parquet reference files to Snowflake stage."""
    manifest = ReferenceDataManifest.from_directory(ref_dir)
    for table_name, path in manifest.tables.items():
        stage_path = f"@reference_data_stage/{path.name}"
        print(f"Uploading {path} -> {stage_path}")
        conn.raw_sql(f"PUT 'file://{path.resolve()}' {stage_path} AUTO_COMPRESS=FALSE OVERWRITE=TRUE")
        print(f"  Loading into table {table_name}...")
        conn.raw_sql(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM @reference_data_stage/{path.name}
            (FILE_FORMAT => (TYPE = PARQUET))
        """)
        print(f"  Done: {table_name}")


def create_transform_sproc(conn):
    """Create a stored procedure that runs the transform pipeline.

    This is a template — customize for your specific pipeline.
    """
    sproc_sql = """
    CREATE OR REPLACE PROCEDURE run_feature_pipeline(
        source_table VARCHAR,
        output_table VARCHAR
    )
    RETURNS VARCHAR
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.11'
    PACKAGES = ('ibis-framework', 'pyarrow')
    HANDLER = 'main'
    AS
    $$
def main(session, source_table: str, output_table: str) -> str:
    import ibis

    # Connect Ibis to the current Snowflake session
    conn = ibis.snowflake.connect(session=session)
    table = conn.table(source_table)

    # Import and apply transforms
    # NOTE: In production, the transforms package would be uploaded
    # to a stage and imported. For now, inline the logic.
    # See: https://docs.snowflake.com/en/developer-guide/stored-procedure/stored-procedures-python

    # Example: clip + ratio
    clipped = table.mutate(
        value_clipped=ibis.greatest(ibis.least(table["value"], 100.0), 0.0)
    )

    # Write results
    conn.create_table(output_table, clipped, overwrite=True)
    return f"Pipeline complete: {output_table}"
    $$;
    """
    conn.raw_sql(sproc_sql)
    print("Created stored procedure: run_feature_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Deploy to Snowflake")
    parser.add_argument(
        "--action",
        choices=["upload-reference", "create-sproc", "all"],
        required=True,
    )
    args = parser.parse_args()

    backend = SnowflakeBackend()
    conn = backend.connect()

    try:
        if args.action in ("upload-reference", "all"):
            upload_reference_data(conn)
        if args.action in ("create-sproc", "all"):
            create_transform_sproc(conn)
    finally:
        backend.disconnect(conn)


if __name__ == "__main__":
    main()
```

**Step 3: Create task definitions**

```sql
-- snowflake/tasks.sql
-- Snowflake Tasks for scheduled pipeline execution.
-- Customize schedule and warehouse as needed.

CREATE OR REPLACE TASK feature_pipeline_daily
    WAREHOUSE = 'COMPUTE_WH'
    SCHEDULE = 'USING CRON 0 6 * * * America/Denver'
    COMMENT = 'Daily feature engineering pipeline'
AS
    CALL run_feature_pipeline('raw_events', 'features_latest');

-- To enable:
-- ALTER TASK feature_pipeline_daily RESUME;

-- Batch inference task (triggered manually or by upstream)
CREATE OR REPLACE TASK batch_inference
    WAREHOUSE = 'COMPUTE_WH'
    COMMENT = 'Batch inference preprocessing'
AS
    CALL run_feature_pipeline('inference_input', 'inference_features');
```

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: add Snowflake deployment scripts and task definitions"
```

---

## Task 11: Integration Test + conftest

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_integration.py`

**Step 1: Create shared test fixtures**

```python
# tests/conftest.py
import datetime

import ibis
import pytest


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
```

**Step 2: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration test: transforms → DuckDB → API → result."""

from functools import partial

import ibis
import pytest

from transforms.features.numeric import clip_outliers, ratio
from transforms.features.categorical import one_hot_flag
from transforms.features.temporal import extract_hour
from transforms.expressions import apply_pipeline


def test_full_pipeline_on_duckdb(duckdb_conn, sample_data):
    """Apply a realistic multi-step pipeline and verify output schema."""
    table = ibis.memtable(sample_data)

    pipeline = [
        partial(clip_outliers, column="value", lower=0.0, upper=10.0),
        partial(ratio, numerator_col="numerator", denominator_col="denominator"),
        partial(one_hot_flag, column="category", value="A"),
        partial(extract_hour, column="event_ts"),
    ]

    result_expr = apply_pipeline(table, pipeline)
    result = duckdb_conn.execute(result_expr)

    # Verify all original + new columns exist
    assert "value_clipped" in result.columns
    assert "numerator_denominator_ratio" in result.columns
    assert "category_is_A" in result.columns
    assert "event_ts_hour" in result.columns

    # Verify row count preserved
    assert len(result) == 5

    # Verify clipping worked
    assert result["value_clipped"].max() <= 10.0

    # Verify zero-denominator handling
    assert result["numerator_denominator_ratio"].isna().sum() == 1

    # Verify one-hot
    assert list(result["category_is_A"]) == [1, 0, 1, 0, 0]


def test_api_roundtrip(sample_data):
    """Full API roundtrip test."""
    import os
    import tempfile

    os.environ["DUCKDB_DATABASE"] = ":memory:"

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["REFERENCE_DATA_DIR"] = tmpdir

        from fastapi.testclient import TestClient
        from api.main import create_app

        app = create_app()
        client = TestClient(app)

        payload = {
            "data": [dict(zip(sample_data.keys(), vals)) for vals in zip(*sample_data.values())],
            "transforms": ["clip_outliers", "ratio"],
            "transform_params": {
                "clip_outliers": {"column": "value", "lower": 0.0, "upper": 10.0},
                "ratio": {"numerator_col": "numerator", "denominator_col": "denominator"},
            },
        }

        resp = client.post("/inference/transform", json=payload)
        assert resp.status_code == 200
        result = resp.json()
        assert len(result["data"]) == 5
        assert result["transforms_applied"] == ["clip_outliers", "ratio"]
```

**Step 3: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: All PASS (Snowflake tests SKIP).

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: add integration tests + shared conftest"
```

---

## Summary: Run Order

| Task | What it builds | Key files |
|------|---------------|-----------|
| 1 | Monorepo scaffolding | `pyproject.toml`, workspace config |
| 2 | Backend interface + DuckDB | `backends/base.py`, `duckdb_backend.py` |
| 3 | Snowflake backend | `snowflake_backend.py` |
| 4 | Transform registry + numeric | `registry.py`, `features/numeric.py` |
| 5 | Categorical + temporal transforms | `features/categorical.py`, `features/temporal.py` |
| 6 | Reference data management | `reference_data.py` |
| 7 | Cross-backend validation harness | `validation/harness.py`, `comparator.py` |
| 8 | FastAPI inference service | `api/main.py`, routes |
| 9 | Streamlit dashboard | `dashboard/app.py`, pages |
| 10 | Snowflake deployment | `deploy.py`, SQL scripts |
| 11 | Integration tests | `test_integration.py` |

Each task is independently committable and testable. Tasks 1-7 are the core — a Claude Code session could stop there and have a fully functional transform + validation layer. Tasks 8-11 build the serving and UI on top.
