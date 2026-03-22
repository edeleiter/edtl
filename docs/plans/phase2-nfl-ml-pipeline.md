# Phase 2 Implementation: NFL 4th-Down ML Pipeline (Plan 2, All 11 Tasks)

## Context

Phase 1 is complete — schemas, transforms, backends all working (57 tests passing). Phase 2 builds the NFL-specific ML pipeline on top: data ingestion, feature engineering, XGBoost model training, FastAPI inference endpoint, and Streamlit calculator page.

**Source plan:** `/mnt/f/edtl/2026-03-21-nfl-fourth-down-ml-pipeline.md`

**Existing code to reuse:**
- `schemas._base.Decision`, `schemas.game.GameState` — data contracts
- `transforms.features.numeric.log_transform` — log_ydstogo feature
- `backends.duckdb_backend.DuckDBBackend` — inference-time execution
- `transforms.registry.TransformRegistry` — register NFL transforms
- Ibis API: uses `ibis.cases()` not `ibis.case()` (Ibis 10+)

---

## Workspace Changes

Add to root `pyproject.toml`:
- `packages/nfl` and `packages/ml` as workspace members
- Their `src` dirs to pytest pythonpath
- `nfl-data-py`, `scikit-learn`, `xgboost`, `joblib` to dev dependencies

Create:
- `packages/nfl/pyproject.toml` — deps: transforms, backends, nfl-data-py, ibis, pandas, pyarrow
- `packages/ml/pyproject.toml` — deps: transforms, backends, nfl, scikit-learn, xgboost, joblib, ibis
- `models/.gitkeep`, `scripts/` directory

Update existing:
- `packages/api/pyproject.toml` — add ml, nfl as workspace deps
- `packages/dashboard/pyproject.toml` — add requests

---

## Build Order (7 steps)

### Step 1: Scaffold nfl + ml packages, update workspace
- Create `packages/nfl/pyproject.toml` + `src/nfl/__init__.py`
- Create `packages/ml/pyproject.toml` + `src/ml/__init__.py`
- Update root `pyproject.toml` workspace members + pythonpath
- Create `models/.gitkeep`, `scripts/` dir
- Verify: `uv sync` succeeds

### Step 2: NFL data package (Tasks 1-4)
**Files:**
- `packages/nfl/src/nfl/ingest.py` — `load_pbp_data()`, `pbp_to_parquet()`, `PBP_MINIMUM_COLUMNS`
- `packages/nfl/src/nfl/fourth_down_filter.py` — `filter_fourth_downs()`, `classify_decision()`
- `packages/nfl/src/nfl/features.py` — `MODEL_FEATURE_COLUMNS` (17 features), `build_fourth_down_features()`
- `packages/nfl/src/nfl/target.py` — `add_target_label()`, `LABEL_MAP`, `TARGET_COLUMN`
- `tests/nfl/test_ingest.py`, `test_fourth_down_filter.py`, `test_features.py`, `test_target.py`

**Key detail:** All feature transforms are Ibis expressions (backend-agnostic). Use `ibis.cases()` for case expressions. The 9 engineered features are:
- `is_opponent_territory`, `is_fg_range`, `is_short_yardage`, `log_ydstogo`
- `is_trailing`, `is_two_score_game`, `abs_score_diff`
- `is_second_half`, `is_late_and_trailing`

**Verify:** `uv run pytest tests/nfl/ -v`

### Step 3: ML dataset builder (Task 5)
**Files:**
- `packages/ml/src/ml/dataset.py` — `build_training_dataset()` (Ibis pipeline: filter → classify → features → target → drop nulls), `split_features_target()`, `train_test_split_by_season()`
- `tests/ml/test_dataset.py`

**Verify:** `uv run pytest tests/ml/test_dataset.py -v`

### Step 4: Model training + evaluation + serialization (Tasks 6-7)
**Files:**
- `packages/ml/src/ml/model.py` — `FourthDownModel` (XGBoost wrapper), `DEFAULT_HYPERPARAMS`
- `packages/ml/src/ml/evaluate.py` — `EvaluationReport`, `evaluate_model()`
- `packages/ml/src/ml/serialize.py` — `save_model()`, `load_model()` (joblib + metadata.json)
- `tests/ml/test_model.py`, `tests/ml/test_evaluate.py`

**Verify:** `uv run pytest tests/ml/ -v`

### Step 5: Prediction module (Task 8)
**Files:**
- `packages/ml/src/ml/predict.py` — `FourthDownPredictor` (loads model, applies feature pipeline via DuckDB, returns predictions)
- `tests/ml/test_predict.py`

**Verify:** `uv run pytest tests/ml/test_predict.py -v`

### Step 6: FastAPI endpoint + Streamlit page (Tasks 9-10)
**Files:**
- `packages/api/src/api/routes/fourth_down.py` — `/predict` and `/predict-batch` endpoints
- `packages/api/src/api/dependencies.py` — `AppState` with predictor init
- `packages/api/src/api/main.py` — FastAPI app with router
- `packages/dashboard/src/dashboard/pages/04_fourth_down_calculator.py`
- `tests/api/test_fourth_down.py`

**Verify:** `uv run pytest tests/api/ -v`

### Step 7: Training script + parity test (Tasks 10-11)
**Files:**
- `scripts/train_fourth_down_model.py` — CLI: `--train-seasons`, `--test-seasons`, `--output-dir`, `--cache-data`
- `tests/nfl/test_parity.py` — cross-backend transform parity validation

**Verify:** Full suite: `uv run pytest tests/ -v`

---

## Execution Strategy

Use subagent-driven development. No git commands from agents.

- **Step 1** is mechanical scaffolding (sonnet)
- **Steps 2-5** can be partially parallelized: Step 2 (NFL data) and Steps 3-5 (ML) are sequential within each track but the NFL tests don't depend on ML code
- **Steps 6-7** depend on Steps 2-5

## Final Verification Gate

```bash
uv run pytest tests/ -v                    # All tests pass
uv run python scripts/train_fourth_down_model.py \
    --train-seasons 2022 --test-seasons 2023 \
    --output-dir models/latest --cache-data  # Training works end-to-end
```
