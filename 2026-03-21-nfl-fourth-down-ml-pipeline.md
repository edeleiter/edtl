# NFL 4th Down Decision Model — ML Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an end-to-end ML pipeline on top of the unified-etl framework that predicts optimal 4th-down decisions (go for it, punt, or field goal) using NFL play-by-play data — demonstrating the full training-on-Snowflake, inference-on-DuckDB workflow with a real, interesting dataset.

**Architecture:** An `nfl` package extends the base `transforms` library with NFL-specific features. An `ml` package handles model training (XGBoost), evaluation, and serialization. The training pipeline ingests nflverse play-by-play data into Snowflake, applies Ibis transforms for feature engineering, trains the model, and snapshots reference data. The inference API accepts game-state inputs and returns go/punt/kick recommendations with probabilities. The Streamlit dashboard adds an interactive 4th-down calculator page.

**Tech Stack:**
- **Data:** nfl-data-py (nflverse play-by-play data, CC-BY-4.0 / CC-BY-SA-4.0 licensed)
- **ML:** scikit-learn, XGBoost
- **Serialization:** joblib (model), Parquet (reference data)
- **Everything else:** Inherits from unified-etl (Ibis, DuckDB, Snowflake, FastAPI, Streamlit)

**Prerequisite:** The unified-etl framework (Tasks 1–11 from the base plan) must be implemented first.

---

## Context: Why 4th Down?

4th down decisions are one of the most analyzed problems in football analytics. On 4th down, the offense must choose: go for it (attempt to gain a first down), punt (give the ball to the other team with better field position), or attempt a field goal. The "correct" decision depends on game state — score differential, field position, time remaining, distance to first down, and more.

This is a great ML exercise because:
- The target variable is well-defined (the decision + the outcome)
- The features are a mix of numeric (yards to go, score diff, time) and categorical (down, quarter, formation)
- The data is freely available and richly annotated
- It's a classification problem with 3 classes
- Real-time inference is meaningful (a coach's tablet scenario)
- The data volume is moderate (~50K plays/year × 20+ years = ~1M plays)

We'll train a model that predicts **expected points added (EPA) for each decision option**, then recommend the option with the highest EPA. This is essentially what Ben Baldwin's 4th down model does — we're rebuilding a simplified version as a framework exercise.

---

## Extended Project Structure

```
unified-etl/
├── ... (existing structure from base plan)
│
├── packages/
│   ├── ... (existing packages)
│   │
│   ├── nfl/                                # NFL-specific data + transforms
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── nfl/
│   │           ├── __init__.py
│   │           ├── ingest.py               # Download + load nflverse data
│   │           ├── fourth_down_filter.py    # Filter PBP to 4th down plays
│   │           ├── features.py             # NFL-specific Ibis transforms
│   │           └── target.py               # Target variable engineering
│   │
│   ├── ml/                                 # ML training + inference
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── ml/
│   │           ├── __init__.py
│   │           ├── dataset.py              # Training dataset builder
│   │           ├── model.py                # Model definition + training
│   │           ├── evaluate.py             # Evaluation metrics + reports
│   │           ├── serialize.py            # Save/load model + metadata
│   │           └── predict.py              # Inference-time prediction
│   │
│   ├── api/                                # (extend existing)
│   │   └── src/
│   │       └── api/
│   │           └── routes/
│   │               └── fourth_down.py      # 4th down recommendation endpoint
│   │
│   └── dashboard/                          # (extend existing)
│       └── src/
│           └── dashboard/
│               └── pages/
│                   └── 04_fourth_down_calculator.py
│
├── models/                                 # Serialized models + metadata
│   └── .gitkeep
│
├── notebooks/                              # Exploration (optional)
│   └── 01_data_exploration.py              # Quick look at the data
│
└── tests/
    ├── nfl/
    │   ├── test_ingest.py
    │   ├── test_fourth_down_filter.py
    │   ├── test_features.py
    │   └── test_target.py
    ├── ml/
    │   ├── test_dataset.py
    │   ├── test_model.py
    │   ├── test_evaluate.py
    │   └── test_predict.py
    └── api/
        └── test_fourth_down.py
```

---

## Task 1: NFL Package Scaffolding + Data Ingestion

**Files:**
- Create: `packages/nfl/pyproject.toml`
- Create: `packages/nfl/src/nfl/__init__.py`
- Create: `packages/nfl/src/nfl/ingest.py`
- Modify: `pyproject.toml` (add `nfl` to workspace members)
- Test: `tests/nfl/test_ingest.py`

**Step 1: Add nfl package to workspace**

Modify root `pyproject.toml`:

```toml
# Add to [tool.uv.workspace] members list:
[tool.uv.workspace]
members = [
    "packages/transforms",
    "packages/backends",
    "packages/api",
    "packages/dashboard",
    "packages/nfl",
    "packages/ml",
    "validation",
]
```

Add to `[tool.pytest.ini_options]` pythonpath:

```toml
pythonpath = [
    "packages/transforms/src",
    "packages/backends/src",
    "packages/api/src",
    "packages/dashboard/src",
    "packages/nfl/src",
    "packages/ml/src",
    "validation/src",
]
```

**Step 2: Create nfl package pyproject.toml**

```toml
# packages/nfl/pyproject.toml
[project]
name = "nfl"
version = "0.1.0"
description = "NFL-specific data ingestion and transforms"
requires-python = ">=3.11"
dependencies = [
    "transforms",
    "backends",
    "nfl-data-py>=0.3.0",
    "ibis-framework[duckdb]>=9.0.0",
    "pyarrow>=14.0.0",
    "pandas>=2.0.0",
]

[tool.uv.sources]
transforms = { workspace = true }
backends = { workspace = true }
```

> **Note for the engineer:** We're using `nfl-data-py` rather than `nflreadpy` because it's on PyPI and pip-installable. The library was archived in Sep 2025 but is still functional for downloading historical data. If installation fails, switch to downloading Parquet files directly from the nflverse-data GitHub releases — the ingest module is designed to support both paths.

**Step 3: Write the failing test**

```python
# tests/nfl/test_ingest.py
import pandas as pd
import pytest

from nfl.ingest import (
    load_pbp_data,
    pbp_to_parquet,
    load_pbp_from_parquet,
    PBP_MINIMUM_COLUMNS,
)


def test_pbp_minimum_columns_defined():
    """Sanity check: we know what columns we need."""
    assert "play_type" in PBP_MINIMUM_COLUMNS
    assert "down" in PBP_MINIMUM_COLUMNS
    assert "ydstogo" in PBP_MINIMUM_COLUMNS
    assert "yardline_100" in PBP_MINIMUM_COLUMNS
    assert "epa" in PBP_MINIMUM_COLUMNS
    assert "game_id" in PBP_MINIMUM_COLUMNS


def test_load_pbp_data_returns_dataframe():
    """Load a single recent year — this hits the network."""
    df = load_pbp_data(years=[2023])
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # Verify essential columns are present
    for col in PBP_MINIMUM_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_pbp_round_trip_parquet(tmp_path):
    """Save and reload PBP data via Parquet preserves shape."""
    # Create a small fake DataFrame with the right columns
    fake_pbp = pd.DataFrame(
        {
            "game_id": ["2023_01_KC_DET", "2023_01_KC_DET"],
            "play_type": ["pass", "run"],
            "down": [1, 2],
            "ydstogo": [10, 7],
            "yardline_100": [75, 68],
            "epa": [0.5, -0.3],
            "posteam": ["KC", "KC"],
            "defteam": ["DET", "DET"],
            "score_differential": [0, 0],
            "half_seconds_remaining": [1800, 1770],
            "game_seconds_remaining": [3600, 3570],
            "quarter_seconds_remaining": [900, 870],
            "qtr": [1, 1],
            "goal_to_go": [0, 0],
            "wp": [0.55, 0.54],
            "season": [2023, 2023],
            "week": [1, 1],
        }
    )

    path = tmp_path / "pbp_2023.parquet"
    pbp_to_parquet(fake_pbp, path)
    reloaded = load_pbp_from_parquet(path)

    assert len(reloaded) == len(fake_pbp)
    assert list(reloaded.columns) == list(fake_pbp.columns)
```

**Step 4: Run tests to verify they fail**

```bash
uv run pytest tests/nfl/test_ingest.py -v
```

Expected: FAIL — module doesn't exist.

**Step 5: Implement the ingest module**

```python
# packages/nfl/src/nfl/ingest.py
"""NFL play-by-play data ingestion via nflverse.

Downloads PBP data, selects relevant columns, and provides
Parquet I/O for caching and Snowflake loading.

Data attribution: nflverse (CC-BY-4.0), FTN Data (CC-BY-SA-4.0).
"""

from pathlib import Path

import pandas as pd

# Columns required for 4th-down modeling.
# This is the contract between ingestion and feature engineering.
PBP_MINIMUM_COLUMNS = [
    "game_id",
    "play_type",
    "down",
    "ydstogo",
    "yardline_100",
    "epa",
    "posteam",
    "defteam",
    "score_differential",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "quarter_seconds_remaining",
    "qtr",
    "goal_to_go",
    "wp",
    "season",
    "week",
]


def load_pbp_data(
    years: list[int],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Download play-by-play data from nflverse for given years.

    Args:
        years: NFL seasons to download (e.g., [2021, 2022, 2023]).
        columns: Columns to keep. Defaults to PBP_MINIMUM_COLUMNS.

    Returns:
        DataFrame with PBP data for the requested years.
    """
    import nfl_data_py as nfl

    columns = columns or PBP_MINIMUM_COLUMNS
    df = nfl.import_pbp_data(years=years, columns=columns, downcast=False)
    return df


def pbp_to_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a PBP DataFrame to Parquet.

    Use this to cache downloaded data locally or prepare for
    Snowflake stage upload.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_pbp_from_parquet(path: str | Path) -> pd.DataFrame:
    """Load PBP data from a local Parquet file."""
    return pd.read_parquet(path)
```

```python
# packages/nfl/src/nfl/__init__.py
from nfl.ingest import load_pbp_data, PBP_MINIMUM_COLUMNS

__all__ = ["load_pbp_data", "PBP_MINIMUM_COLUMNS"]
```

**Step 6: Run tests**

```bash
uv run pytest tests/nfl/test_ingest.py -v
```

Expected: `test_load_pbp_data_returns_dataframe` may PASS (requires network) or be marked slow. Others PASS.

> **Note:** If `nfl-data-py` fails to install or the network call fails, mark the network test with `@pytest.mark.network` and skip in CI. The Parquet round-trip test is the critical one.

**Step 7: Commit**

```bash
git add -A
git commit -m "feat(nfl): add PBP data ingestion via nflverse"
```

---

## Task 2: 4th Down Play Filtering

**Files:**
- Create: `packages/nfl/src/nfl/fourth_down_filter.py`
- Test: `tests/nfl/test_fourth_down_filter.py`

**Step 1: Write the failing test**

```python
# tests/nfl/test_fourth_down_filter.py
import ibis
import pytest

from nfl.fourth_down_filter import (
    filter_fourth_downs,
    classify_decision,
    DECISION_GO,
    DECISION_PUNT,
    DECISION_FG,
)


@pytest.fixture
def con():
    return ibis.duckdb.connect()


@pytest.fixture
def sample_pbp():
    """Minimal PBP data with 4th down and non-4th-down plays."""
    return ibis.memtable(
        {
            "game_id": [
                "2023_01_KC_DET",
                "2023_01_KC_DET",
                "2023_01_KC_DET",
                "2023_01_KC_DET",
                "2023_01_KC_DET",
                "2023_01_KC_DET",
            ],
            "down": [4, 4, 4, 1, 2, 3],
            "play_type": ["pass", "punt", "field_goal", "run", "pass", "pass"],
            "ydstogo": [2, 8, 3, 10, 7, 4],
            "yardline_100": [45, 75, 28, 80, 50, 30],
            "epa": [1.5, -0.3, 0.8, 0.2, -0.1, 0.4],
            "posteam": ["KC"] * 6,
            "defteam": ["DET"] * 6,
            "score_differential": [0, -7, 3, 0, 0, 0],
            "half_seconds_remaining": [900, 600, 300, 1800, 1750, 1700],
            "game_seconds_remaining": [2700, 2400, 2100, 3600, 3550, 3500],
            "quarter_seconds_remaining": [900, 600, 300, 900, 850, 800],
            "qtr": [2, 3, 4, 1, 1, 1],
            "goal_to_go": [0, 0, 1, 0, 0, 0],
            "wp": [0.55, 0.35, 0.60, 0.50, 0.51, 0.52],
            "season": [2023] * 6,
            "week": [1] * 6,
        }
    )


def test_filter_fourth_downs_only(con, sample_pbp):
    expr = filter_fourth_downs(sample_pbp)
    result = con.execute(expr)
    assert len(result) == 3
    assert all(result["down"] == 4)


def test_classify_decision(con, sample_pbp):
    filtered = filter_fourth_downs(sample_pbp)
    expr = classify_decision(filtered)
    result = con.execute(expr)
    assert "decision" in result.columns
    decisions = list(result["decision"])
    assert decisions[0] == DECISION_GO       # play_type="pass" on 4th down
    assert decisions[1] == DECISION_PUNT     # play_type="punt"
    assert decisions[2] == DECISION_FG       # play_type="field_goal"


def test_decision_constants():
    assert DECISION_GO == "go_for_it"
    assert DECISION_PUNT == "punt"
    assert DECISION_FG == "field_goal"
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/nfl/test_fourth_down_filter.py -v
```

Expected: FAIL.

**Step 3: Implement the filter module**

```python
# packages/nfl/src/nfl/fourth_down_filter.py
"""Filter play-by-play data to 4th-down situations and classify decisions.

A 4th-down play is classified into one of three decisions:
- go_for_it: the offense attempted to convert (run or pass)
- punt: the offense punted
- field_goal: the offense attempted a field goal

Trick plays, aborted snaps, penalties, and other oddities are
excluded to keep the training data clean.
"""

import ibis
import ibis.expr.types as ir

DECISION_GO = "go_for_it"
DECISION_PUNT = "punt"
DECISION_FG = "field_goal"

# Play types that count as "going for it"
_GO_PLAY_TYPES = ("pass", "run", "qb_kneel", "qb_spike")


def filter_fourth_downs(table: ir.Table) -> ir.Table:
    """Filter to only 4th-down plays with valid play types.

    Excludes penalties, no-plays, and special teams plays that
    aren't punts or field goals.
    """
    valid_types = (*_GO_PLAY_TYPES, "punt", "field_goal")
    return table.filter(
        (table["down"] == 4)
        & table["play_type"].isin(valid_types)
    )


def classify_decision(table: ir.Table) -> ir.Table:
    """Add a 'decision' column classifying the 4th-down choice.

    Assumes the table has already been filtered to 4th-down plays.
    """
    play_type = table["play_type"]
    decision = (
        ibis.case()
        .when(play_type == "punt", DECISION_PUNT)
        .when(play_type == "field_goal", DECISION_FG)
        .else_(DECISION_GO)  # pass, run, qb_kneel, qb_spike
        .end()
    )
    return table.mutate(decision=decision)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/nfl/test_fourth_down_filter.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(nfl): add 4th-down play filter + decision classifier"
```

---

## Task 3: NFL-Specific Feature Engineering (Ibis Transforms)

These are the features that feed the model. They're all defined as Ibis expressions, so they run on both Snowflake and DuckDB.

**Files:**
- Create: `packages/nfl/src/nfl/features.py`
- Test: `tests/nfl/test_features.py`

**Step 1: Write the failing test**

```python
# tests/nfl/test_features.py
import ibis
import pytest

from nfl.features import (
    add_field_position_features,
    add_game_state_features,
    add_time_features,
    build_fourth_down_features,
)


@pytest.fixture
def con():
    return ibis.duckdb.connect()


@pytest.fixture
def fourth_down_plays():
    return ibis.memtable(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "down": [4, 4, 4, 4],
            "ydstogo": [1, 3, 7, 15],
            "yardline_100": [50, 30, 85, 10],
            "score_differential": [0, -14, 7, -3],
            "half_seconds_remaining": [1800, 120, 900, 30],
            "game_seconds_remaining": [3600, 1920, 900, 30],
            "quarter_seconds_remaining": [900, 120, 900, 30],
            "qtr": [1, 2, 3, 4],
            "goal_to_go": [0, 0, 0, 1],
            "wp": [0.50, 0.25, 0.70, 0.40],
            "posteam": ["KC", "DET", "BUF", "SF"],
            "defteam": ["DET", "KC", "MIA", "PHI"],
            "play_type": ["pass", "run", "punt", "field_goal"],
            "epa": [2.0, -0.5, -1.0, 1.5],
            "decision": ["go_for_it", "go_for_it", "punt", "field_goal"],
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 2, 3, 4],
        }
    )


def test_field_position_features(con, fourth_down_plays):
    expr = add_field_position_features(fourth_down_plays)
    result = con.execute(expr)
    assert "is_opponent_territory" in result.columns
    assert "is_fg_range" in result.columns
    assert "is_short_yardage" in result.columns
    # yardline_100=50 -> opponent territory = False (midfield)
    # yardline_100=30 -> opponent territory = True
    assert result.iloc[1]["is_opponent_territory"] == 1
    # yardline_100=10, goal_to_go=1 -> fg_range = True
    assert result.iloc[3]["is_fg_range"] == 1
    # ydstogo=1 -> short yardage
    assert result.iloc[0]["is_short_yardage"] == 1


def test_game_state_features(con, fourth_down_plays):
    expr = add_game_state_features(fourth_down_plays)
    result = con.execute(expr)
    assert "is_trailing" in result.columns
    assert "is_two_score_game" in result.columns
    assert "abs_score_diff" in result.columns
    # score_diff=-14 -> trailing, two-score game
    assert result.iloc[1]["is_trailing"] == 1
    assert result.iloc[1]["is_two_score_game"] == 1
    assert result.iloc[1]["abs_score_diff"] == 14


def test_time_features(con, fourth_down_plays):
    expr = add_time_features(fourth_down_plays)
    result = con.execute(expr)
    assert "is_second_half" in result.columns
    assert "is_late_and_trailing" in result.columns
    # qtr=4, game_seconds_remaining=30, score_diff=-3 -> late and trailing
    assert result.iloc[3]["is_late_and_trailing"] == 1
    # qtr=1 -> not second half
    assert result.iloc[0]["is_second_half"] == 0


def test_full_feature_pipeline(con, fourth_down_plays):
    """End-to-end: build all features at once."""
    expr = build_fourth_down_features(fourth_down_plays)
    result = con.execute(expr)
    # Verify all feature columns exist
    expected_new = [
        "is_opponent_territory",
        "is_fg_range",
        "is_short_yardage",
        "is_trailing",
        "is_two_score_game",
        "abs_score_diff",
        "is_second_half",
        "is_late_and_trailing",
        "log_ydstogo",
    ]
    for col in expected_new:
        assert col in result.columns, f"Missing feature: {col}"
    # Row count preserved
    assert len(result) == 4
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/nfl/test_features.py -v
```

Expected: FAIL.

**Step 3: Implement NFL feature transforms**

```python
# packages/nfl/src/nfl/features.py
"""NFL 4th-down feature engineering as Ibis expressions.

All functions take and return Ibis table expressions, making them
executable on both Snowflake (training) and DuckDB (inference).

Feature groups:
- Field position: where on the field is the play?
- Game state: score, lead/trail, margin
- Time: quarter, seconds remaining, urgency
"""

import ibis
import ibis.expr.types as ir


def add_field_position_features(table: ir.Table) -> ir.Table:
    """Add features derived from field position.

    - is_opponent_territory: yardline_100 <= 50 (closer to opponent end zone)
    - is_fg_range: yardline_100 <= 40 (roughly makeable FG distance)
    - is_short_yardage: ydstogo <= 2
    - log_ydstogo: log(ydstogo) — compresses long distances
    """
    yl = table["yardline_100"]
    ytg = table["ydstogo"]

    return table.mutate(
        is_opponent_territory=ibis.case().when(yl <= 50, 1).else_(0).end(),
        is_fg_range=ibis.case().when(yl <= 40, 1).else_(0).end(),
        is_short_yardage=ibis.case().when(ytg <= 2, 1).else_(0).end(),
        log_ydstogo=ibis.greatest(ytg, 1).cast("float64").log(),
    )


def add_game_state_features(table: ir.Table) -> ir.Table:
    """Add features derived from score and game state.

    - is_trailing: team is behind
    - is_two_score_game: abs(score_diff) >= 9 (need 2+ scores)
    - abs_score_diff: absolute value of score differential
    """
    sd = table["score_differential"]

    return table.mutate(
        is_trailing=ibis.case().when(sd < 0, 1).else_(0).end(),
        is_two_score_game=ibis.case().when(sd.abs() >= 9, 1).else_(0).end(),
        abs_score_diff=sd.abs(),
    )


def add_time_features(table: ir.Table) -> ir.Table:
    """Add features derived from game clock.

    - is_second_half: 3rd or 4th quarter
    - is_late_and_trailing: 4th quarter, < 300 seconds left, trailing
    """
    qtr = table["qtr"]
    gsr = table["game_seconds_remaining"]
    sd = table["score_differential"]

    return table.mutate(
        is_second_half=ibis.case().when(qtr >= 3, 1).else_(0).end(),
        is_late_and_trailing=(
            ibis.case()
            .when((qtr == 4) & (gsr <= 300) & (sd < 0), 1)
            .else_(0)
            .end()
        ),
    )


def build_fourth_down_features(table: ir.Table) -> ir.Table:
    """Apply the full feature pipeline for 4th-down modeling.

    This is the canonical pipeline — used at both training and
    inference time. Composes all feature groups.
    """
    table = add_field_position_features(table)
    table = add_game_state_features(table)
    table = add_time_features(table)
    return table


# Ordered list of feature columns the model expects as input.
# This is the contract between feature engineering and the model.
MODEL_FEATURE_COLUMNS = [
    "ydstogo",
    "yardline_100",
    "score_differential",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "qtr",
    "goal_to_go",
    "wp",
    "is_opponent_territory",
    "is_fg_range",
    "is_short_yardage",
    "log_ydstogo",
    "is_trailing",
    "is_two_score_game",
    "abs_score_diff",
    "is_second_half",
    "is_late_and_trailing",
]
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/nfl/test_features.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(nfl): add 4th-down feature engineering transforms"
```

---

## Task 4: Target Variable Engineering

**Files:**
- Create: `packages/nfl/src/nfl/target.py`
- Test: `tests/nfl/test_target.py`

**Step 1: Write the failing test**

```python
# tests/nfl/test_target.py
import ibis
import pytest

from nfl.target import add_target_label, TARGET_COLUMN, LABEL_MAP


@pytest.fixture
def con():
    return ibis.duckdb.connect()


@pytest.fixture
def fourth_down_with_decisions():
    return ibis.memtable(
        {
            "decision": ["go_for_it", "punt", "field_goal", "go_for_it"],
            "epa": [2.0, -0.5, 0.8, -1.0],
            "ydstogo": [1, 8, 3, 5],
        }
    )


def test_add_target_label(con, fourth_down_with_decisions):
    expr = add_target_label(fourth_down_with_decisions)
    result = con.execute(expr)
    assert TARGET_COLUMN in result.columns
    labels = list(result[TARGET_COLUMN])
    # go_for_it=0, punt=1, field_goal=2
    assert labels == [0, 1, 2, 0]


def test_label_map_complete():
    assert "go_for_it" in LABEL_MAP
    assert "punt" in LABEL_MAP
    assert "field_goal" in LABEL_MAP
    assert len(LABEL_MAP) == 3


def test_label_map_values_are_sequential():
    values = sorted(LABEL_MAP.values())
    assert values == [0, 1, 2]
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/nfl/test_target.py -v
```

Expected: FAIL.

**Step 3: Implement target engineering**

```python
# packages/nfl/src/nfl/target.py
"""Target variable engineering for 4th-down decision model.

The target is a 3-class label:
  0 = go_for_it
  1 = punt
  2 = field_goal

We predict which decision the coach *should* make, evaluated by
the EPA (expected points added) outcome. At training time, we use
the actual decision as the label. The model learns which game states
correspond to which decisions being optimal.

A more sophisticated approach would label plays by which decision
*would have yielded* the highest EPA, but that requires counterfactual
estimation. For this educational exercise, we use the actual decision
as a proxy — coaches are generally rational, and the model will learn
the decision boundary even if some individual labels are "wrong".
"""

import ibis
import ibis.expr.types as ir

TARGET_COLUMN = "decision_label"

LABEL_MAP = {
    "go_for_it": 0,
    "punt": 1,
    "field_goal": 2,
}

INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def add_target_label(table: ir.Table) -> ir.Table:
    """Map the decision column to integer labels for classification."""
    case_expr = ibis.case()
    for label, code in LABEL_MAP.items():
        case_expr = case_expr.when(table["decision"] == label, code)
    return table.mutate(**{TARGET_COLUMN: case_expr.else_(ibis.null()).end()})
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/nfl/test_target.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(nfl): add target variable engineering for 4th-down model"
```

---

## Task 5: ML Package — Dataset Builder

**Files:**
- Create: `packages/ml/pyproject.toml`
- Create: `packages/ml/src/ml/__init__.py`
- Create: `packages/ml/src/ml/dataset.py`
- Test: `tests/ml/test_dataset.py`

**Step 1: Create ml package pyproject.toml**

```toml
# packages/ml/pyproject.toml
[project]
name = "ml"
version = "0.1.0"
description = "ML model training, evaluation, and inference"
requires-python = ">=3.11"
dependencies = [
    "transforms",
    "backends",
    "nfl",
    "scikit-learn>=1.4.0",
    "xgboost>=2.0.0",
    "pandas>=2.0.0",
    "numpy>=1.26.0",
    "joblib>=1.3.0",
    "ibis-framework[duckdb]>=9.0.0",
]

[tool.uv.sources]
transforms = { workspace = true }
backends = { workspace = true }
nfl = { workspace = true }
```

**Step 2: Write the failing test**

```python
# tests/ml/test_dataset.py
import ibis
import numpy as np
import pandas as pd
import pytest

from ml.dataset import (
    build_training_dataset,
    split_features_target,
    train_test_split_by_season,
)
from nfl.features import MODEL_FEATURE_COLUMNS
from nfl.target import TARGET_COLUMN


@pytest.fixture
def con():
    return ibis.duckdb.connect()


@pytest.fixture
def raw_fourth_down_plays():
    """Simulated 4th-down plays across multiple seasons."""
    np.random.seed(42)
    n = 200
    return ibis.memtable(
        {
            "game_id": [f"game_{i}" for i in range(n)],
            "down": [4] * n,
            "play_type": np.random.choice(
                ["pass", "run", "punt", "field_goal"], size=n, p=[0.3, 0.1, 0.4, 0.2]
            ).tolist(),
            "ydstogo": np.random.randint(1, 15, size=n).tolist(),
            "yardline_100": np.random.randint(1, 99, size=n).tolist(),
            "epa": np.random.normal(0, 1.5, size=n).tolist(),
            "posteam": ["KC"] * n,
            "defteam": ["DET"] * n,
            "score_differential": np.random.randint(-21, 21, size=n).tolist(),
            "half_seconds_remaining": np.random.randint(0, 1800, size=n).tolist(),
            "game_seconds_remaining": np.random.randint(0, 3600, size=n).tolist(),
            "quarter_seconds_remaining": np.random.randint(0, 900, size=n).tolist(),
            "qtr": np.random.choice([1, 2, 3, 4], size=n).tolist(),
            "goal_to_go": np.random.choice([0, 1], size=n, p=[0.8, 0.2]).tolist(),
            "wp": np.random.uniform(0.1, 0.9, size=n).tolist(),
            "season": np.random.choice([2021, 2022, 2023], size=n).tolist(),
            "week": np.random.randint(1, 18, size=n).tolist(),
        }
    )


def test_build_training_dataset(con, raw_fourth_down_plays):
    result_expr = build_training_dataset(raw_fourth_down_plays)
    result = con.execute(result_expr)

    # Should have all feature columns + target
    for col in MODEL_FEATURE_COLUMNS:
        assert col in result.columns, f"Missing feature: {col}"
    assert TARGET_COLUMN in result.columns

    # No nulls in target
    assert result[TARGET_COLUMN].notna().all()


def test_split_features_target(con, raw_fourth_down_plays):
    dataset = con.execute(build_training_dataset(raw_fourth_down_plays))
    X, y = split_features_target(dataset)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert list(X.columns) == MODEL_FEATURE_COLUMNS
    assert y.name == TARGET_COLUMN


def test_train_test_split_by_season(con, raw_fourth_down_plays):
    dataset = con.execute(build_training_dataset(raw_fourth_down_plays))
    train, test = train_test_split_by_season(
        dataset, test_seasons=[2023]
    )

    assert all(train["season"] != 2023)
    assert all(test["season"] == 2023)
    assert len(train) + len(test) == len(dataset)
```

**Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/ml/test_dataset.py -v
```

Expected: FAIL.

**Step 4: Implement the dataset builder**

```python
# packages/ml/src/ml/dataset.py
"""Training dataset construction.

Orchestrates the full pipeline: filter → classify → features → target.
Provides utilities for train/test splitting by season (temporal split).
"""

import pandas as pd
import ibis.expr.types as ir

from nfl.fourth_down_filter import filter_fourth_downs, classify_decision
from nfl.features import build_fourth_down_features, MODEL_FEATURE_COLUMNS
from nfl.target import add_target_label, TARGET_COLUMN


def build_training_dataset(pbp_table: ir.Table) -> ir.Table:
    """Full pipeline from raw PBP to model-ready features.

    Applies: filter 4th downs → classify decision → engineer features
    → add target label → drop rows with null target.

    This is an Ibis expression — it pushes down to whatever backend
    executes it (Snowflake at training time, DuckDB for testing).
    """
    table = filter_fourth_downs(pbp_table)
    table = classify_decision(table)
    table = build_fourth_down_features(table)
    table = add_target_label(table)
    # Drop rows where target is null (unclassifiable plays)
    table = table.filter(table[TARGET_COLUMN].notnull())
    return table


def split_features_target(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a materialized dataset into feature matrix X and target y.

    Args:
        dataset: DataFrame with MODEL_FEATURE_COLUMNS + TARGET_COLUMN.

    Returns:
        (X, y) where X is the feature matrix and y is the target series.
    """
    X = dataset[MODEL_FEATURE_COLUMNS].copy()
    y = dataset[TARGET_COLUMN].copy()
    return X, y


def train_test_split_by_season(
    dataset: pd.DataFrame,
    test_seasons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset by season for temporal train/test split.

    Temporal splitting prevents data leakage — we train on past
    seasons and evaluate on future seasons, which mirrors how
    the model would be used in practice.
    """
    test_mask = dataset["season"].isin(test_seasons)
    train = dataset[~test_mask].reset_index(drop=True)
    test = dataset[test_mask].reset_index(drop=True)
    return train, test
```

```python
# packages/ml/src/ml/__init__.py
```

**Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/ml/test_dataset.py -v
```

Expected: All PASS.

**Step 6: Commit**

```bash
git add -A
git commit -m "feat(ml): add training dataset builder with temporal splitting"
```

---

## Task 6: Model Training + Evaluation

**Files:**
- Create: `packages/ml/src/ml/model.py`
- Create: `packages/ml/src/ml/evaluate.py`
- Create: `packages/ml/src/ml/serialize.py`
- Test: `tests/ml/test_model.py`
- Test: `tests/ml/test_evaluate.py`

**Step 1: Write the failing test for model training**

```python
# tests/ml/test_model.py
import numpy as np
import pandas as pd
import pytest

from ml.model import FourthDownModel, DEFAULT_HYPERPARAMS
from nfl.features import MODEL_FEATURE_COLUMNS
from nfl.target import LABEL_MAP


@pytest.fixture
def training_data():
    """Synthetic training data."""
    np.random.seed(42)
    n = 500
    X = pd.DataFrame(
        {col: np.random.randn(n) for col in MODEL_FEATURE_COLUMNS}
    )
    # Make target somewhat correlated with features for a learnable problem
    # Short yardage + opponent territory → go; far from end zone → punt; mid-range → FG
    score = X["yardline_100"] + X["ydstogo"] * 2
    y = pd.Series(
        np.select(
            [score < -1, score > 1],
            [0, 1],  # go_for_it, punt
            default=2,  # field_goal
        ),
        name="decision_label",
    )
    return X, y


def test_model_instantiation():
    model = FourthDownModel()
    assert model is not None
    assert not model.is_fitted


def test_model_fit(training_data):
    X, y = training_data
    model = FourthDownModel()
    model.fit(X, y)
    assert model.is_fitted


def test_model_predict(training_data):
    X, y = training_data
    model = FourthDownModel()
    model.fit(X, y)
    predictions = model.predict(X.iloc[:5])
    assert len(predictions) == 5
    assert all(p in LABEL_MAP.values() for p in predictions)


def test_model_predict_proba(training_data):
    X, y = training_data
    model = FourthDownModel()
    model.fit(X, y)
    proba = model.predict_proba(X.iloc[:5])
    assert proba.shape == (5, 3)
    # Probabilities sum to 1
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_default_hyperparams():
    assert "n_estimators" in DEFAULT_HYPERPARAMS
    assert "max_depth" in DEFAULT_HYPERPARAMS
    assert "learning_rate" in DEFAULT_HYPERPARAMS
```

**Step 2: Write the failing test for evaluation**

```python
# tests/ml/test_evaluate.py
import numpy as np
import pandas as pd
import pytest

from ml.evaluate import evaluate_model, EvaluationReport
from ml.model import FourthDownModel
from nfl.features import MODEL_FEATURE_COLUMNS


@pytest.fixture
def fitted_model_and_data():
    np.random.seed(42)
    n = 300
    X = pd.DataFrame(
        {col: np.random.randn(n) for col in MODEL_FEATURE_COLUMNS}
    )
    score = X["yardline_100"] + X["ydstogo"] * 2
    y = pd.Series(
        np.select([score < -1, score > 1], [0, 1], default=2),
        name="decision_label",
    )
    model = FourthDownModel()
    model.fit(X, y)
    return model, X, y


def test_evaluate_model(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    report = evaluate_model(model, X, y)

    assert isinstance(report, EvaluationReport)
    assert 0.0 <= report.accuracy <= 1.0
    assert len(report.class_report) > 0
    assert report.confusion_matrix.shape == (3, 3)


def test_report_to_dict(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    report = evaluate_model(model, X, y)
    d = report.to_dict()

    assert "accuracy" in d
    assert "confusion_matrix" in d
    assert "class_report" in d
```

**Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/ml/test_model.py tests/ml/test_evaluate.py -v
```

Expected: FAIL.

**Step 4: Implement the model**

```python
# packages/ml/src/ml/model.py
"""XGBoost model for 4th-down decision classification.

Wraps XGBoost in a clean interface with fit/predict/predict_proba.
Hyperparameters are configurable, with sensible defaults for this
problem size (~50K-200K training samples, 17 features, 3 classes).
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from nfl.features import MODEL_FEATURE_COLUMNS

DEFAULT_HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}


class FourthDownModel:
    """4th-down decision classifier.

    Predicts one of: go_for_it (0), punt (1), field_goal (2).
    """

    def __init__(self, hyperparams: dict | None = None):
        params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self._model = XGBClassifier(**params)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
        verbose: bool = False,
    ) -> "FourthDownModel":
        """Train the model.

        Args:
            X: Feature matrix with columns matching MODEL_FEATURE_COLUMNS.
            y: Target labels (0, 1, 2).
            eval_set: Optional validation set for early stopping.
            verbose: Print training progress.
        """
        self._validate_features(X)
        fit_params = {"verbose": verbose}
        if eval_set:
            fit_params["eval_set"] = eval_set
            fit_params["early_stopping_rounds"] = 20
        self._model.fit(X, y, **fit_params)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict decision labels."""
        self._validate_features(X)
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities. Shape: (n_samples, 3)."""
        self._validate_features(X)
        return self._model.predict_proba(X)

    def feature_importances(self) -> dict[str, float]:
        """Return feature importance scores."""
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")
        return dict(
            zip(MODEL_FEATURE_COLUMNS, self._model.feature_importances_)
        )

    def _validate_features(self, X: pd.DataFrame) -> None:
        missing = set(MODEL_FEATURE_COLUMNS) - set(X.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
```

**Step 5: Implement evaluation**

```python
# packages/ml/src/ml/evaluate.py
"""Model evaluation metrics and reporting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from ml.model import FourthDownModel
from nfl.target import INVERSE_LABEL_MAP


@dataclass
class EvaluationReport:
    """Container for model evaluation results."""

    accuracy: float
    confusion_matrix: np.ndarray
    class_report: str
    feature_importances: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "class_report": self.class_report,
            "feature_importances": self.feature_importances,
        }

    def summary(self) -> str:
        lines = [
            f"Accuracy: {self.accuracy:.4f}",
            "",
            "Classification Report:",
            self.class_report,
            "",
            "Top 5 Features:",
        ]
        sorted_fi = sorted(
            self.feature_importances.items(), key=lambda x: x[1], reverse=True
        )
        for name, importance in sorted_fi[:5]:
            lines.append(f"  {name}: {importance:.4f}")
        return "\n".join(lines)


def evaluate_model(
    model: FourthDownModel,
    X: pd.DataFrame,
    y: pd.Series,
) -> EvaluationReport:
    """Evaluate a fitted model on a test set.

    Returns an EvaluationReport with accuracy, confusion matrix,
    per-class metrics, and feature importances.
    """
    predictions = model.predict(X)

    target_names = [
        INVERSE_LABEL_MAP[i] for i in sorted(INVERSE_LABEL_MAP.keys())
    ]

    return EvaluationReport(
        accuracy=accuracy_score(y, predictions),
        confusion_matrix=confusion_matrix(y, predictions, labels=[0, 1, 2]),
        class_report=classification_report(
            y, predictions, target_names=target_names, zero_division=0,
        ),
        feature_importances=model.feature_importances(),
    )
```

**Step 6: Implement serialization**

```python
# packages/ml/src/ml/serialize.py
"""Model serialization and loading.

Saves the XGBoost model + metadata (hyperparams, feature columns,
training date, evaluation metrics) as a versioned artifact.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from ml.model import FourthDownModel
from ml.evaluate import EvaluationReport
from nfl.features import MODEL_FEATURE_COLUMNS


def save_model(
    model: FourthDownModel,
    output_dir: str | Path,
    report: EvaluationReport | None = None,
    version: str | None = None,
) -> Path:
    """Save a trained model and metadata to disk.

    Creates:
      {output_dir}/model.joblib       — serialized model
      {output_dir}/metadata.json      — training metadata

    Args:
        model: Fitted FourthDownModel.
        output_dir: Directory to save artifacts.
        report: Optional evaluation report to include in metadata.
        version: Optional version string (defaults to timestamp).

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    version = version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "n_features": len(MODEL_FEATURE_COLUMNS),
        "n_classes": 3,
        "class_labels": ["go_for_it", "punt", "field_goal"],
    }
    if report:
        metadata["evaluation"] = report.to_dict()

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return output_dir


def load_model(model_dir: str | Path) -> FourthDownModel:
    """Load a trained model from disk.

    Args:
        model_dir: Directory containing model.joblib + metadata.json.

    Returns:
        Fitted FourthDownModel ready for prediction.
    """
    model_dir = Path(model_dir)
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")
    return joblib.load(model_path)
```

**Step 7: Run tests to verify they pass**

```bash
uv run pytest tests/ml/test_model.py tests/ml/test_evaluate.py -v
```

Expected: All PASS.

**Step 8: Commit**

```bash
git add -A
git commit -m "feat(ml): add XGBoost model, evaluation, and serialization"
```

---

## Task 7: Prediction Module (Inference-Time)

**Files:**
- Create: `packages/ml/src/ml/predict.py`
- Test: `tests/ml/test_predict.py`

**Step 1: Write the failing test**

```python
# tests/ml/test_predict.py
import numpy as np
import pandas as pd
import pytest

from ml.predict import FourthDownPredictor
from ml.model import FourthDownModel
from nfl.features import MODEL_FEATURE_COLUMNS


@pytest.fixture
def predictor(tmp_path):
    """Create a fitted model, save it, and return a predictor."""
    from ml.serialize import save_model

    np.random.seed(42)
    n = 300
    X = pd.DataFrame({col: np.random.randn(n) for col in MODEL_FEATURE_COLUMNS})
    score = X["yardline_100"] + X["ydstogo"] * 2
    y = pd.Series(np.select([score < -1, score > 1], [0, 1], default=2))

    model = FourthDownModel()
    model.fit(X, y)
    save_model(model, tmp_path / "model_v1")

    return FourthDownPredictor(model_dir=tmp_path / "model_v1")


def test_predictor_from_game_state(predictor):
    """Predict from a raw game-state dict (the API input format)."""
    game_state = {
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
    result = predictor.predict_from_game_state(game_state)

    assert "recommendation" in result
    assert result["recommendation"] in ("go_for_it", "punt", "field_goal")
    assert "probabilities" in result
    assert len(result["probabilities"]) == 3
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


def test_predictor_batch(predictor):
    """Predict for multiple game states at once."""
    game_states = [
        {
            "ydstogo": 1, "yardline_100": 45, "score_differential": 0,
            "half_seconds_remaining": 1800, "game_seconds_remaining": 3600,
            "quarter_seconds_remaining": 900, "qtr": 1, "goal_to_go": 0, "wp": 0.50,
        },
        {
            "ydstogo": 8, "yardline_100": 75, "score_differential": 14,
            "half_seconds_remaining": 1200, "game_seconds_remaining": 1200,
            "quarter_seconds_remaining": 300, "qtr": 4, "goal_to_go": 0, "wp": 0.85,
        },
    ]
    results = predictor.predict_batch(game_states)
    assert len(results) == 2
    assert all("recommendation" in r for r in results)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/ml/test_predict.py -v
```

Expected: FAIL.

**Step 3: Implement the predictor**

```python
# packages/ml/src/ml/predict.py
"""Inference-time prediction.

The FourthDownPredictor takes raw game-state dicts (as received
from the API), applies the same Ibis feature transforms used at
training time via DuckDB, and returns recommendations.

This is where the unified transform guarantee matters most:
the features computed here must be identical to those computed
on Snowflake during training.
"""

from pathlib import Path

import ibis
import pandas as pd

from ml.model import FourthDownModel
from ml.serialize import load_model
from nfl.features import build_fourth_down_features, MODEL_FEATURE_COLUMNS
from nfl.target import INVERSE_LABEL_MAP


class FourthDownPredictor:
    """Stateful predictor that holds a loaded model + DuckDB connection.

    Designed to be initialized once at API startup and reused for
    all inference requests.
    """

    def __init__(self, model_dir: str | Path):
        self.model: FourthDownModel = load_model(model_dir)
        self._con = ibis.duckdb.connect()

    def predict_from_game_state(self, game_state: dict) -> dict:
        """Predict the optimal 4th-down decision from a game-state dict.

        Args:
            game_state: Dict with keys matching raw PBP columns:
                ydstogo, yardline_100, score_differential,
                half_seconds_remaining, game_seconds_remaining,
                quarter_seconds_remaining, qtr, goal_to_go, wp.

        Returns:
            Dict with:
                recommendation: "go_for_it", "punt", or "field_goal"
                probabilities: {decision: probability} for each option
        """
        results = self.predict_batch([game_state])
        return results[0]

    def predict_batch(self, game_states: list[dict]) -> list[dict]:
        """Predict for multiple game states.

        Uses Ibis + DuckDB to apply the same feature transforms
        used at training time, then runs the model on the result.
        """
        # Build Ibis table from input dicts
        table = ibis.memtable(game_states)

        # Apply the canonical feature pipeline (same as training)
        features_expr = build_fourth_down_features(table)

        # Execute on DuckDB
        features_df = self._con.execute(features_expr)

        # Extract model input columns
        X = features_df[MODEL_FEATURE_COLUMNS]

        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        # Format results
        results = []
        for i in range(len(predictions)):
            label = int(predictions[i])
            proba = probabilities[i]
            results.append(
                {
                    "recommendation": INVERSE_LABEL_MAP[label],
                    "probabilities": {
                        INVERSE_LABEL_MAP[j]: round(float(proba[j]), 4)
                        for j in range(3)
                    },
                }
            )
        return results
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/ml/test_predict.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(ml): add inference-time predictor with DuckDB feature pipeline"
```

---

## Task 8: FastAPI 4th-Down Endpoint

**Files:**
- Create: `packages/api/src/api/routes/fourth_down.py`
- Modify: `packages/api/src/api/main.py` (add router)
- Modify: `packages/api/src/api/dependencies.py` (add predictor)
- Test: `tests/api/test_fourth_down.py`

**Step 1: Write the failing test**

```python
# tests/api/test_fourth_down.py
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ml.model import FourthDownModel
from ml.serialize import save_model
from nfl.features import MODEL_FEATURE_COLUMNS


@pytest.fixture
def client_with_model(tmp_path):
    """Create a test client with a trained model available."""
    # Train a quick model
    np.random.seed(42)
    n = 300
    X = pd.DataFrame({col: np.random.randn(n) for col in MODEL_FEATURE_COLUMNS})
    score = X["yardline_100"] + X["ydstogo"] * 2
    y = pd.Series(np.select([score < -1, score > 1], [0, 1], default=2))

    model = FourthDownModel()
    model.fit(X, y)

    model_dir = tmp_path / "model_v1"
    save_model(model, model_dir)

    os.environ["DUCKDB_DATABASE"] = ":memory:"
    os.environ["REFERENCE_DATA_DIR"] = str(tmp_path / "ref")
    os.environ["MODEL_DIR"] = str(model_dir)
    (tmp_path / "ref").mkdir()

    from api.main import create_app
    app = create_app()
    return TestClient(app)


def test_fourth_down_predict(client_with_model):
    payload = {
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
    resp = client_with_model.post("/fourth-down/predict", json=payload)
    assert resp.status_code == 200
    result = resp.json()
    assert "recommendation" in result
    assert result["recommendation"] in ("go_for_it", "punt", "field_goal")
    assert "probabilities" in result
    proba = result["probabilities"]
    assert abs(sum(proba.values()) - 1.0) < 0.01


def test_fourth_down_predict_batch(client_with_model):
    payload = {
        "game_states": [
            {
                "ydstogo": 1, "yardline_100": 2, "score_differential": -3,
                "half_seconds_remaining": 30, "game_seconds_remaining": 30,
                "quarter_seconds_remaining": 30, "qtr": 4, "goal_to_go": 1, "wp": 0.40,
            },
            {
                "ydstogo": 10, "yardline_100": 80, "score_differential": 14,
                "half_seconds_remaining": 1500, "game_seconds_remaining": 3300,
                "quarter_seconds_remaining": 600, "qtr": 1, "goal_to_go": 0, "wp": 0.75,
            },
        ]
    }
    resp = client_with_model.post("/fourth-down/predict-batch", json=payload)
    assert resp.status_code == 200
    results = resp.json()["predictions"]
    assert len(results) == 2
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/api/test_fourth_down.py -v
```

Expected: FAIL.

**Step 3: Implement the endpoint**

```python
# packages/api/src/api/routes/fourth_down.py
"""4th-down decision recommendation endpoint."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/fourth-down")


class GameState(BaseModel):
    """A single game-state snapshot for 4th-down prediction."""

    ydstogo: int = Field(..., ge=1, le=99, description="Yards to go for first down")
    yardline_100: int = Field(..., ge=1, le=99, description="Yards from opponent end zone")
    score_differential: int = Field(..., description="Positive = leading, negative = trailing")
    half_seconds_remaining: int = Field(..., ge=0, le=1800)
    game_seconds_remaining: int = Field(..., ge=0, le=3600)
    quarter_seconds_remaining: int = Field(..., ge=0, le=900)
    qtr: int = Field(..., ge=1, le=5, description="Quarter (5=OT)")
    goal_to_go: int = Field(..., ge=0, le=1)
    wp: float = Field(..., ge=0.0, le=1.0, description="Pre-play win probability")


class PredictionResponse(BaseModel):
    recommendation: str
    probabilities: dict[str, float]


class BatchRequest(BaseModel):
    game_states: list[GameState]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


@router.post("/predict", response_model=PredictionResponse)
def predict(game_state: GameState, request: Request):
    """Get a 4th-down recommendation for a single game state."""
    predictor = request.app.state.fourth_down_predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Set MODEL_DIR environment variable.",
        )
    result = predictor.predict_from_game_state(game_state.model_dump())
    return PredictionResponse(**result)


@router.post("/predict-batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    """Get 4th-down recommendations for multiple game states."""
    predictor = request.app.state.fourth_down_predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Set MODEL_DIR environment variable.",
        )
    states = [gs.model_dump() for gs in batch.game_states]
    results = predictor.predict_batch(states)
    return BatchResponse(
        predictions=[PredictionResponse(**r) for r in results]
    )
```

**Step 4: Modify dependencies.py — add predictor**

Add to `packages/api/src/api/dependencies.py`:

```python
# Add import at top:
from ml.predict import FourthDownPredictor

# Add to AppState class:
class AppState:
    def __init__(self):
        self.backend = DuckDBBackend()
        self.conn: ibis.BaseBackend | None = None
        self.reference_tables: dict = {}
        self.fourth_down_predictor: FourthDownPredictor | None = None

    def startup(self):
        self.conn = self.backend.connect()
        ref_dir = os.environ.get("REFERENCE_DATA_DIR", "reference_data")
        ref_path = Path(ref_dir)
        if ref_path.exists() and any(ref_path.glob("*.parquet")):
            manifest = ReferenceDataManifest.from_directory(ref_path)
            self.reference_tables = load_reference_tables(self.conn, manifest)

        # Load 4th-down model if available
        model_dir = os.environ.get("MODEL_DIR")
        if model_dir and Path(model_dir).exists():
            self.fourth_down_predictor = FourthDownPredictor(model_dir)
```

**Step 5: Modify main.py — add router**

Add to `packages/api/src/api/main.py`:

```python
from api.routes import health, inference, batch, fourth_down

# In create_app():
app.include_router(fourth_down.router)
```

**Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/api/test_fourth_down.py -v
```

Expected: All PASS.

**Step 7: Commit**

```bash
git add -A
git commit -m "feat(api): add 4th-down prediction endpoint"
```

---

## Task 9: Streamlit 4th-Down Calculator Page

**Files:**
- Create: `packages/dashboard/src/dashboard/pages/04_fourth_down_calculator.py`

**Step 1: Implement the calculator page**

```python
# packages/dashboard/src/dashboard/pages/04_fourth_down_calculator.py
"""Interactive 4th-down decision calculator.

Lets users input a game scenario and get a recommendation
from the trained model via the API, with probability visualization.
"""

import json

import requests
import streamlit as st
import pandas as pd

st.header("🏈 4th Down Decision Calculator")
st.markdown(
    "Enter a game scenario below and get an AI-powered recommendation "
    "on whether to **go for it**, **punt**, or **kick a field goal**."
)

api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")

# --- Game Scenario Input ---
st.subheader("Game Scenario")

col1, col2, col3 = st.columns(3)

with col1:
    qtr = st.selectbox("Quarter", [1, 2, 3, 4], index=3)
    ydstogo = st.slider("Yards to go", min_value=1, max_value=30, value=4)
    goal_to_go = st.checkbox("Goal to go?")

with col2:
    yardline_100 = st.slider(
        "Yards from opponent end zone",
        min_value=1, max_value=99, value=35,
        help="1 = opponent's 1 yard line, 50 = midfield, 99 = own 1 yard line",
    )
    score_diff = st.slider(
        "Score differential (your team)",
        min_value=-28, max_value=28, value=0,
        help="Positive = you're winning, negative = you're losing",
    )

with col3:
    mins_remaining = st.slider(
        "Minutes remaining in quarter", min_value=0, max_value=15, value=5,
    )
    secs_remaining = mins_remaining * 60
    wp = st.slider(
        "Pre-play win probability",
        min_value=0.01, max_value=0.99, value=0.50,
        help="Estimated chance your team wins before this play",
    )

# Compute derived time fields
quarter_seconds = secs_remaining
if qtr <= 2:
    half_seconds = (2 - qtr) * 900 + secs_remaining
    game_seconds = (4 - qtr) * 900 + secs_remaining
else:
    half_seconds = (4 - qtr) * 900 + secs_remaining
    game_seconds = half_seconds  # In 2nd half, game = half

# --- Get Recommendation ---
if st.button("Get Recommendation", type="primary", use_container_width=True):
    payload = {
        "ydstogo": ydstogo,
        "yardline_100": yardline_100,
        "score_differential": score_diff,
        "half_seconds_remaining": half_seconds,
        "game_seconds_remaining": game_seconds,
        "quarter_seconds_remaining": quarter_seconds,
        "qtr": qtr,
        "goal_to_go": int(goal_to_go),
        "wp": wp,
    }

    try:
        resp = requests.post(
            f"{api_url}/fourth-down/predict", json=payload, timeout=10
        )
        if resp.status_code == 200:
            result = resp.json()

            # Display recommendation
            rec = result["recommendation"]
            emoji = {"go_for_it": "🏃", "punt": "🦶", "field_goal": "🦵"}
            label = {"go_for_it": "GO FOR IT", "punt": "PUNT", "field_goal": "FIELD GOAL"}

            st.markdown("---")
            st.markdown(
                f"## {emoji.get(rec, '🏈')} Recommendation: **{label.get(rec, rec)}**"
            )

            # Display probabilities as a bar chart
            proba = result["probabilities"]
            proba_df = pd.DataFrame(
                {
                    "Decision": [label.get(k, k) for k in proba.keys()],
                    "Probability": list(proba.values()),
                }
            )
            st.bar_chart(proba_df.set_index("Decision"), horizontal=True)

            # Show confidence
            confidence = max(proba.values())
            if confidence > 0.7:
                st.success(f"High confidence: {confidence:.1%}")
            elif confidence > 0.5:
                st.info(f"Moderate confidence: {confidence:.1%}")
            else:
                st.warning(f"Low confidence: {confidence:.1%} — close call!")

            # Show raw probabilities
            with st.expander("Raw prediction details"):
                st.json(result)
                st.json({"input": payload})

        elif resp.status_code == 503:
            st.error("Model not loaded on the server. Train a model first.")
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")

    except requests.ConnectionError:
        st.error(
            f"Cannot connect to API at {api_url}. "
            "Start the server with: `uv run uvicorn api.main:app --reload`"
        )

# --- Educational Context ---
st.markdown("---")
with st.expander("About this model"):
    st.markdown("""
    This model is trained on NFL play-by-play data from nflverse.
    It learns which 4th-down decisions (go for it, punt, field goal)
    coaches historically make in each game situation, using features like:

    - **Field position:** yards to go, yard line, field goal range
    - **Game state:** score differential, win probability
    - **Time:** quarter, seconds remaining, late-game urgency
    - **Situation:** goal-to-go, short yardage

    The model uses XGBoost classification trained on multiple NFL seasons
    with temporal train/test splitting to prevent data leakage.

    **Data:** nflverse (CC-BY-4.0). This is an educational exercise
    and should not be used for actual coaching decisions.
    """)
```

**Step 2: Manual verification**

```bash
# Terminal 1: API with model
MODEL_DIR=models/latest uv run uvicorn api.main:app --reload

# Terminal 2: Streamlit
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Navigate to the 4th Down Calculator page. Enter scenarios. Verify recommendations display.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add 4th-down calculator page"
```

---

## Task 10: Training Script (Ties It All Together)

This is the script a data scientist runs to train the model end-to-end. It exercises the full framework: ingest → Ibis transforms → model training → evaluation → serialization.

**Files:**
- Create: `scripts/train_fourth_down_model.py`
- Create: `models/.gitkeep`

**Step 1: Implement the training script**

```python
# scripts/train_fourth_down_model.py
"""End-to-end training script for the 4th-down decision model.

Usage:
    # Full training (downloads data, trains, evaluates, saves)
    uv run python scripts/train_fourth_down_model.py

    # Use cached Parquet data
    uv run python scripts/train_fourth_down_model.py --data-dir data/pbp/

    # Specify training/test seasons
    uv run python scripts/train_fourth_down_model.py --train-seasons 2019 2020 2021 2022 --test-seasons 2023

    # Custom output directory
    uv run python scripts/train_fourth_down_model.py --output-dir models/v2
"""

import argparse
from pathlib import Path

import ibis
import pandas as pd

from nfl.ingest import load_pbp_data, pbp_to_parquet, load_pbp_from_parquet
from ml.dataset import build_training_dataset, split_features_target, train_test_split_by_season
from ml.model import FourthDownModel
from ml.evaluate import evaluate_model
from ml.serialize import save_model


def main():
    parser = argparse.ArgumentParser(description="Train 4th-down decision model")
    parser.add_argument(
        "--train-seasons",
        nargs="+",
        type=int,
        default=[2019, 2020, 2021, 2022],
        help="Seasons to use for training",
    )
    parser.add_argument(
        "--test-seasons",
        nargs="+",
        type=int,
        default=[2023],
        help="Seasons to hold out for testing",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with cached PBP Parquet files (skips download)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/latest",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--cache-data",
        action="store_true",
        help="Save downloaded data to data/pbp/ for future use",
    )
    args = parser.parse_args()

    all_seasons = sorted(set(args.train_seasons + args.test_seasons))

    # --- Step 1: Load Data ---
    print(f"Step 1: Loading PBP data for seasons {all_seasons}...")
    if args.data_dir:
        data_dir = Path(args.data_dir)
        dfs = []
        for season in all_seasons:
            path = data_dir / f"pbp_{season}.parquet"
            if path.exists():
                print(f"  Loading cached: {path}")
                dfs.append(load_pbp_from_parquet(path))
            else:
                print(f"  Downloading season {season}...")
                df = load_pbp_data(years=[season])
                dfs.append(df)
        raw_df = pd.concat(dfs, ignore_index=True)
    else:
        raw_df = load_pbp_data(years=all_seasons)

    print(f"  Loaded {len(raw_df):,} total plays")

    # Cache if requested
    if args.cache_data:
        cache_dir = Path("data/pbp")
        cache_dir.mkdir(parents=True, exist_ok=True)
        for season in all_seasons:
            season_df = raw_df[raw_df["season"] == season]
            path = cache_dir / f"pbp_{season}.parquet"
            pbp_to_parquet(season_df, path)
            print(f"  Cached: {path} ({len(season_df):,} plays)")

    # --- Step 2: Build Training Dataset (via Ibis + DuckDB) ---
    print("\nStep 2: Building training dataset via Ibis transforms...")
    con = ibis.duckdb.connect()
    table = ibis.memtable(raw_df)
    dataset_expr = build_training_dataset(table)
    dataset = con.execute(dataset_expr)
    print(f"  4th-down plays after filtering: {len(dataset):,}")

    # --- Step 3: Train/Test Split ---
    print(f"\nStep 3: Splitting — train: {args.train_seasons}, test: {args.test_seasons}")
    train_df, test_df = train_test_split_by_season(dataset, test_seasons=args.test_seasons)
    print(f"  Train: {len(train_df):,} plays")
    print(f"  Test:  {len(test_df):,} plays")

    # Decision distribution
    print("\n  Decision distribution (train):")
    for decision, count in train_df["decision"].value_counts().items():
        print(f"    {decision}: {count:,} ({count/len(train_df):.1%})")

    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    # --- Step 4: Train Model ---
    print("\nStep 4: Training XGBoost model...")
    model = FourthDownModel()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=True,
    )
    print("  Training complete.")

    # --- Step 5: Evaluate ---
    print("\nStep 5: Evaluating on test set...")
    report = evaluate_model(model, X_test, y_test)
    print(report.summary())

    # --- Step 6: Save Model ---
    output_dir = Path(args.output_dir)
    print(f"\nStep 6: Saving model to {output_dir}...")
    save_model(model, output_dir, report=report)
    print(f"  Saved: {output_dir}/model.joblib")
    print(f"  Saved: {output_dir}/metadata.json")

    print("\nDone! Start the API with:")
    print(f"  MODEL_DIR={output_dir} uv run uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
```

**Step 2: Test the script runs (manual)**

```bash
# Quick test with 1 season
uv run python scripts/train_fourth_down_model.py \
    --train-seasons 2022 \
    --test-seasons 2023 \
    --cache-data
```

Expected: Downloads data, builds features, trains model, prints evaluation, saves to `models/latest/`.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: add end-to-end training script"
```

---

## Task 11: Cross-Backend Parity Test for NFL Transforms

**Files:**
- Create: `tests/nfl/test_parity.py`

This validates that the NFL feature transforms produce identical results on two DuckDB instances (and could be extended to Snowflake when credentials are available).

**Step 1: Write the parity test**

```python
# tests/nfl/test_parity.py
"""Validate NFL feature transforms produce identical results across backends.

Runs in CI with DuckDB vs DuckDB. Extend to DuckDB vs Snowflake
by setting SNOWFLAKE_ACCOUNT and uncommenting the snowflake test.
"""

import ibis
import pytest

from nfl.features import build_fourth_down_features
from nfl.fourth_down_filter import filter_fourth_downs, classify_decision
from validation.harness import validate_transform_parity


@pytest.fixture
def golden_fourth_down_data():
    """Carefully chosen golden data that exercises edge cases."""
    return {
        "game_id": ["g1", "g2", "g3", "g4", "g5"],
        "down": [4, 4, 4, 4, 4],
        "play_type": ["pass", "run", "punt", "field_goal", "pass"],
        "ydstogo": [1, 15, 8, 3, 10],
        "yardline_100": [2, 99, 50, 25, 45],
        "epa": [5.0, -2.0, -0.5, 1.0, 0.0],
        "posteam": ["KC", "DET", "BUF", "SF", "PHI"],
        "defteam": ["DET", "KC", "MIA", "SEA", "DAL"],
        "score_differential": [-3, 0, 14, -21, 7],
        "half_seconds_remaining": [30, 1800, 900, 120, 600],
        "game_seconds_remaining": [30, 3600, 2700, 120, 2400],
        "quarter_seconds_remaining": [30, 900, 900, 120, 600],
        "qtr": [4, 1, 2, 4, 3],
        "goal_to_go": [1, 0, 0, 0, 0],
        "wp": [0.15, 0.50, 0.80, 0.10, 0.65],
        "season": [2023, 2023, 2023, 2023, 2023],
        "week": [1, 2, 3, 4, 5],
    }


def _full_pipeline(table):
    """The complete transform pipeline (filter + classify + features)."""
    t = filter_fourth_downs(table)
    t = classify_decision(t)
    t = build_fourth_down_features(t)
    return t


def test_parity_duckdb_vs_duckdb(golden_fourth_down_data):
    """Sanity check: same backend produces identical results."""
    result = validate_transform_parity(
        golden_data=golden_fourth_down_data,
        transform=_full_pipeline,
        backend_a_factory=lambda: ibis.duckdb.connect(),
        backend_b_factory=lambda: ibis.duckdb.connect(),
        sort_by="game_id",
    )
    assert result.is_equal, f"Parity failed:\n{result.report()}"


# Uncomment when Snowflake credentials are available in CI:
#
# @pytest.mark.snowflake
# def test_parity_duckdb_vs_snowflake(golden_fourth_down_data):
#     from backends.snowflake_backend import SnowflakeBackend
#
#     sf = SnowflakeBackend()
#     result = validate_transform_parity(
#         golden_data=golden_fourth_down_data,
#         transform=_full_pipeline,
#         backend_a_factory=lambda: ibis.duckdb.connect(),
#         backend_b_factory=lambda: sf.connect(),
#         sort_by="game_id",
#         atol=1e-6,  # Slightly looser tolerance for cross-engine
#     )
#     assert result.is_equal, f"Cross-backend parity failed:\n{result.report()}"
```

**Step 2: Run the parity test**

```bash
uv run pytest tests/nfl/test_parity.py -v
```

Expected: PASS.

**Step 3: Commit**

```bash
git add -A
git commit -m "test: add cross-backend parity test for NFL transforms"
```

---

## Summary: Run Order

| Task | What it builds | Key files |
|------|---------------|-----------|
| 1 | NFL package scaffolding + data ingestion | `nfl/ingest.py` |
| 2 | 4th-down play filtering | `nfl/fourth_down_filter.py` |
| 3 | NFL feature engineering (Ibis) | `nfl/features.py` |
| 4 | Target variable engineering | `nfl/target.py` |
| 5 | ML dataset builder | `ml/dataset.py` |
| 6 | XGBoost model + evaluation + serialization | `ml/model.py`, `evaluate.py`, `serialize.py` |
| 7 | Inference predictor (DuckDB features → model) | `ml/predict.py` |
| 8 | FastAPI 4th-down endpoint | `api/routes/fourth_down.py` |
| 9 | Streamlit 4th-down calculator | `dashboard/pages/04_fourth_down_calculator.py` |
| 10 | End-to-end training script | `scripts/train_fourth_down_model.py` |
| 11 | Cross-backend parity test for NFL transforms | `tests/nfl/test_parity.py` |

## End-to-End Workflow

After implementing both plans, the full workflow is:

```bash
# 1. Train the model (downloads NFL data, engineers features, trains XGBoost)
uv run python scripts/train_fourth_down_model.py --cache-data

# 2. Start the API (serves real-time inference via DuckDB)
MODEL_DIR=models/latest uv run uvicorn api.main:app --reload

# 3. Start the dashboard (interactive 4th-down calculator)
uv run streamlit run packages/dashboard/src/dashboard/app.py

# 4. Run all tests including parity validation
uv run pytest tests/ -v

# 5. (Optional) Deploy training pipeline to Snowflake
uv run python snowflake/deploy.py --action all
```

The critical thing this demonstrates: **the Ibis transforms in `nfl/features.py` are the single source of truth.** They execute on DuckDB locally during training (via `scripts/train_fourth_down_model.py`), on DuckDB at inference time (via `ml/predict.py` → FastAPI), and could execute on Snowflake for production-scale training — all from the same expression definitions, with the validation harness proving they produce identical results.
