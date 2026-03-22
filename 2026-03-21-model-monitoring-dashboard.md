# Model Monitoring Dashboard — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Plan 7 of 10**

**Goal:** Build a model monitoring library and multi-page Streamlit dashboard that tracks prediction performance over time, detects concept drift and prediction distribution shifts, provides segmented performance analysis, computes Population Stability Index (PSI) for features, logs all predictions for retrospective evaluation, generates automated retrain recommendations, and surfaces everything through an executive summary dashboard with traffic-light alerting.

**Architecture:** A `monitoring` package provides five modules: a prediction logger that records every inference with timestamp/input/output/ground-truth, a performance tracker that computes rolling accuracy/F1/confusion matrices over configurable time windows, a prediction drift detector that compares output distributions against training baselines, a PSI calculator for feature-level stability monitoring, and a retrain advisor that aggregates all signals into a scored recommendation. All data is stored as Parquet files (one per day/batch) and loaded via Ibis+DuckDB for fast aggregation in the dashboard. Streamlit pages provide an executive overview, time-series performance charts, prediction distribution analysis, segmented deep-dives (by quarter, field position, score context), and a retrain decision center.

**Tech Stack:**
- **Storage:** Parquet files (append-only prediction log, daily rollup)
- **Computation:** Ibis + DuckDB (aggregation), scikit-learn (metrics), scipy (statistical tests)
- **Visualization:** Plotly (time series, heatmaps, distributions), Streamlit
- **Alerting:** Traffic-light thresholds (configurable)
- **Inherits:** Everything from Plans 1-5

**Prerequisite:** Unified ETL (Plan 1), NFL ML Pipeline (Plan 2) must be implemented. Feature Selection (Plan 3), Data Quality (Plan 4), and Interpretation Engine (Plan 5) are recommended but not required.

---

## Context: Why Model Monitoring for the 4th-Down Model?

NFL football has natural concept drift built in. Rule changes (taunting penalties, kickoff rules), coaching philosophy shifts (the "analytics revolution" making teams more aggressive on 4th down), and team roster changes all mean that the relationship between game state and optimal 4th-down decisions evolves season over season.

Without monitoring, the model silently degrades. A model trained on 2019-2022 data might still predict "punt" in situations where 2024 coaches overwhelmingly go for it, because the league's aggressiveness baseline has shifted. The monitoring system detects this by comparing the model's prediction distribution against actual outcomes as they become available (ground truth arrives after each game, with full play-by-play data available within 48 hours via nflverse).

The monitoring dashboard answers five questions:
1. **Is the model still accurate?** (Performance over time)
2. **Are predictions changing?** (Prediction distribution drift)
3. **Are inputs changing?** (Feature drift via PSI)
4. **Where is it failing?** (Segmented performance analysis)
5. **Should we retrain?** (Automated recommendation)

---

## Extended Project Structure

```
unified-etl/
├── packages/
│   ├── monitoring/                          # Model monitoring library
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── monitoring/
│   │           ├── __init__.py
│   │           ├── prediction_log.py        # Log predictions + ground truth
│   │           ├── performance.py           # Rolling accuracy, F1, confusion matrix
│   │           ├── prediction_drift.py      # Output distribution monitoring
│   │           ├── psi.py                   # Population Stability Index
│   │           ├── segments.py              # Segmented performance analysis
│   │           ├── retrain_advisor.py       # Automated retrain recommendation
│   │           ├── alerts.py                # Threshold-based alerting
│   │           └── config.py                # Monitoring configuration
│   │
│   └── dashboard/
│       └── src/
│           └── dashboard/
│               └── pages/
│                   ├── ... (existing pages 01-11)
│                   ├── 12_monitoring_overview.py
│                   ├── 13_performance_timeline.py
│                   ├── 14_prediction_drift.py
│                   ├── 15_segment_analysis.py
│                   └── 16_retrain_center.py
│
├── monitoring_data/                         # Prediction logs + rollups
│   ├── predictions/                         # Raw prediction logs (Parquet per batch)
│   ├── ground_truth/                        # Ground truth labels (Parquet per week)
│   ├── rollups/                             # Pre-computed daily/weekly rollups
│   └── alerts/                              # Alert history
│
└── tests/
    └── monitoring/
        ├── test_prediction_log.py
        ├── test_performance.py
        ├── test_prediction_drift.py
        ├── test_psi.py
        ├── test_segments.py
        ├── test_retrain_advisor.py
        └── test_alerts.py
```

---

## Task 1: Monitoring Package Scaffolding + Configuration

**Files:**
- Create: `packages/monitoring/pyproject.toml`
- Create: `packages/monitoring/src/monitoring/__init__.py`
- Create: `packages/monitoring/src/monitoring/config.py`
- Modify: `pyproject.toml` (add to workspace)
- Create: `monitoring_data/predictions/.gitkeep`
- Create: `monitoring_data/ground_truth/.gitkeep`
- Create: `monitoring_data/rollups/.gitkeep`
- Create: `monitoring_data/alerts/.gitkeep`

**Step 1: Add monitoring to workspace**

Add `"packages/monitoring"` to `[tool.uv.workspace] members` and `"packages/monitoring/src"` to `[tool.pytest.ini_options] pythonpath` in root `pyproject.toml`.

**Step 2: Create monitoring package pyproject.toml**

```toml
# packages/monitoring/pyproject.toml
[project]
name = "monitoring"
version = "0.1.0"
description = "ML model monitoring: prediction logging, performance tracking, drift detection"
requires-python = ">=3.11"
dependencies = [
    "transforms",
    "backends",
    "nfl",
    "ml",
    "data-quality",
    "ibis-framework[duckdb]>=9.0.0",
    "pandas>=2.0.0",
    "numpy>=1.26.0",
    "scipy>=1.12.0",
    "scikit-learn>=1.4.0",
    "pyarrow>=14.0.0",
]

[tool.uv.sources]
transforms = { workspace = true }
backends = { workspace = true }
nfl = { workspace = true }
ml = { workspace = true }
data-quality = { workspace = true }
```

**Step 3: Implement configuration**

```python
# packages/monitoring/src/monitoring/config.py
"""Monitoring configuration with sensible defaults.

All thresholds are configurable. Defaults are tuned for the
4th-down decision model but work for most classification tasks.

Traffic-light system:
  GREEN  = healthy, no action needed
  YELLOW = warning, investigate within a week
  RED    = critical, investigate immediately / consider retraining
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AlertThresholds:
    """Thresholds for traffic-light alerting."""

    # Performance thresholds (relative to baseline)
    accuracy_yellow: float = 0.03   # 3% drop from baseline
    accuracy_red: float = 0.05      # 5% drop from baseline
    f1_yellow: float = 0.05
    f1_red: float = 0.10

    # Prediction drift thresholds (KL divergence)
    prediction_kl_yellow: float = 0.05
    prediction_kl_red: float = 0.15

    # PSI thresholds (per feature)
    psi_yellow: float = 0.10
    psi_red: float = 0.25

    # Null rate spike
    null_spike_yellow: float = 0.05  # 5% absolute increase
    null_spike_red: float = 0.15

    # Volume anomaly (relative to expected)
    volume_low_yellow: float = 0.50   # < 50% of expected volume
    volume_low_red: float = 0.20      # < 20% of expected volume


@dataclass
class MonitoringConfig:
    """Top-level monitoring configuration."""

    # Data paths
    predictions_dir: Path = Path("monitoring_data/predictions")
    ground_truth_dir: Path = Path("monitoring_data/ground_truth")
    rollups_dir: Path = Path("monitoring_data/rollups")
    alerts_dir: Path = Path("monitoring_data/alerts")

    # Baseline model info
    model_version: str = "latest"
    baseline_accuracy: float = 0.0    # Set after training
    baseline_f1: float = 0.0

    # Time windows for rolling metrics
    rolling_window_days: int = 7
    comparison_window_days: int = 30

    # Segments to track
    segments: dict[str, list] = field(default_factory=lambda: {
        "qtr": [1, 2, 3, 4],
        "is_short_yardage": [0, 1],
        "is_trailing": [0, 1],
        "is_fg_range": [0, 1],
    })

    # Alert thresholds
    thresholds: AlertThresholds = field(default_factory=AlertThresholds)

    def ensure_dirs(self) -> None:
        """Create monitoring data directories if they don't exist."""
        for d in [self.predictions_dir, self.ground_truth_dir,
                  self.rollups_dir, self.alerts_dir]:
            d.mkdir(parents=True, exist_ok=True)
```

```python
# packages/monitoring/src/monitoring/__init__.py
from monitoring.config import MonitoringConfig, AlertThresholds

__all__ = ["MonitoringConfig", "AlertThresholds"]
```

**Step 4: Create .gitkeep files**

```bash
mkdir -p monitoring_data/{predictions,ground_truth,rollups,alerts}
touch monitoring_data/{predictions,ground_truth,rollups,alerts}/.gitkeep
```

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(monitoring): scaffold monitoring package with configuration"
```

---

## Task 2: Prediction Logger

**Files:**
- Create: `packages/monitoring/src/monitoring/prediction_log.py`
- Test: `tests/monitoring/test_prediction_log.py`

**Step 1: Write the failing test**

```python
# tests/monitoring/test_prediction_log.py
import datetime

import pandas as pd
import pytest

from monitoring.prediction_log import (
    PredictionRecord,
    PredictionLog,
    log_prediction,
    log_batch,
    load_predictions,
    attach_ground_truth,
)
from monitoring.config import MonitoringConfig


@pytest.fixture
def config(tmp_path):
    return MonitoringConfig(
        predictions_dir=tmp_path / "predictions",
        ground_truth_dir=tmp_path / "ground_truth",
        rollups_dir=tmp_path / "rollups",
        alerts_dir=tmp_path / "alerts",
    )


@pytest.fixture
def sample_predictions():
    return [
        PredictionRecord(
            timestamp=datetime.datetime(2024, 1, 6, 14, 0),
            game_id="2024_01_KC_DET",
            prediction=0,  # go_for_it
            probabilities=[0.65, 0.20, 0.15],
            features={"ydstogo": 3, "yardline_100": 35, "qtr": 4},
        ),
        PredictionRecord(
            timestamp=datetime.datetime(2024, 1, 6, 14, 5),
            game_id="2024_01_KC_DET",
            prediction=1,  # punt
            probabilities=[0.10, 0.75, 0.15],
            features={"ydstogo": 8, "yardline_100": 75, "qtr": 2},
        ),
    ]


def test_log_single_prediction(config, sample_predictions):
    config.ensure_dirs()
    log_prediction(sample_predictions[0], config)
    df = load_predictions(config)
    assert len(df) == 1
    assert df.iloc[0]["prediction"] == 0


def test_log_batch(config, sample_predictions):
    config.ensure_dirs()
    log_batch(sample_predictions, config, batch_id="week_01")
    df = load_predictions(config)
    assert len(df) == 2


def test_load_predictions_empty(config):
    config.ensure_dirs()
    df = load_predictions(config)
    assert len(df) == 0


def test_attach_ground_truth(config, sample_predictions):
    config.ensure_dirs()
    log_batch(sample_predictions, config, batch_id="week_01")

    ground_truth = pd.DataFrame({
        "game_id": ["2024_01_KC_DET", "2024_01_KC_DET"],
        "play_index": [0, 1],
        "actual_decision": [0, 1],  # go_for_it, punt
        "actual_epa": [2.5, -0.3],
    })

    attach_ground_truth(ground_truth, config, batch_id="week_01_gt")
    gt_df = pd.read_parquet(config.ground_truth_dir / "week_01_gt.parquet")
    assert len(gt_df) == 2
    assert "actual_decision" in gt_df.columns


def test_prediction_record_to_dict(sample_predictions):
    d = sample_predictions[0].to_dict()
    assert "timestamp" in d
    assert "prediction" in d
    assert "probabilities" in d
    assert "features" in d
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/monitoring/test_prediction_log.py -v
```

Expected: FAIL.

**Step 3: Implement the prediction logger**

```python
# packages/monitoring/src/monitoring/prediction_log.py
"""Prediction logging — record every inference for retrospective analysis.

Every prediction the model makes is logged with:
- Timestamp
- Game/play identifier
- Predicted class + probability distribution
- Input features (for segment analysis and debugging)
- Ground truth (attached later when available)

Stored as Parquet files (one per batch/day) for efficient
time-range queries via Ibis+DuckDB.
"""

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from monitoring.config import MonitoringConfig


@dataclass
class PredictionRecord:
    """A single prediction to be logged."""

    timestamp: datetime.datetime
    prediction: int                     # Predicted class label
    probabilities: list[float]          # [p(go), p(punt), p(fg)]
    features: dict[str, Any]            # Input feature values
    game_id: str = ""                   # Game identifier for joining ground truth
    model_version: str = ""
    request_id: str = ""                # Unique request ID for tracing

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "prediction": self.prediction,
            "prob_go": self.probabilities[0],
            "prob_punt": self.probabilities[1],
            "prob_fg": self.probabilities[2],
            "confidence": max(self.probabilities),
            "game_id": self.game_id,
            "model_version": self.model_version,
            "request_id": self.request_id,
            "features_json": json.dumps(self.features),
            # Also flatten key features for segment queries
            **{f"feat_{k}": v for k, v in self.features.items()},
        }


def log_prediction(
    record: PredictionRecord,
    config: MonitoringConfig,
    batch_id: str | None = None,
) -> Path:
    """Log a single prediction to Parquet."""
    return log_batch([record], config, batch_id=batch_id)


def log_batch(
    records: list[PredictionRecord],
    config: MonitoringConfig,
    batch_id: str | None = None,
) -> Path:
    """Log a batch of predictions to a single Parquet file.

    Args:
        records: List of prediction records.
        config: Monitoring configuration.
        batch_id: Optional batch name. Defaults to timestamp.

    Returns:
        Path to the written Parquet file.
    """
    if not batch_id:
        batch_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = [r.to_dict() for r in records]
    df = pd.DataFrame(rows)

    path = config.predictions_dir / f"{batch_id}.parquet"
    config.predictions_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_predictions(
    config: MonitoringConfig,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
) -> pd.DataFrame:
    """Load prediction logs via DuckDB for memory efficiency.

    Uses DuckDB's Parquet glob reader to push down date filters
    without loading all files into memory. This avoids OOM errors
    when prediction logs grow beyond a few hundred MB.

    Previous implementation used pd.read_parquet() + pd.concat()
    which loaded ALL Parquet files into memory simultaneously.
    """
    import ibis

    parquet_pattern = str(config.predictions_dir / "*.parquet")
    con = ibis.duckdb.connect()

    try:
        t = con.read_parquet(parquet_pattern)
    except Exception:
        return pd.DataFrame()

    if "timestamp" in t.columns:
        t = t.cast({"timestamp": "timestamp"})
        if start_date:
            t = t.filter(t.timestamp >= start_date)
        if end_date:
            t = t.filter(t.timestamp <= end_date)

    return t.to_pandas()


def attach_ground_truth(
    ground_truth: pd.DataFrame,
    config: MonitoringConfig,
    batch_id: str | None = None,
) -> Path:
    """Save ground truth labels for later joining with predictions.

    Ground truth arrives delayed (after games are played and PBP
    data is published, typically 24-48 hours). Saved separately
    and joined at analysis time.

    Expected columns: game_id, play_index, actual_decision, actual_epa
    """
    if not batch_id:
        batch_id = datetime.datetime.now().strftime("gt_%Y%m%d_%H%M%S")

    path = config.ground_truth_dir / f"{batch_id}.parquet"
    config.ground_truth_dir.mkdir(parents=True, exist_ok=True)
    ground_truth.to_parquet(path, index=False)
    return path
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/monitoring/test_prediction_log.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(monitoring): add prediction logger with Parquet storage"
```

---

## Task 3: Performance Tracker (Rolling Metrics)

**Files:**
- Create: `packages/monitoring/src/monitoring/performance.py`
- Test: `tests/monitoring/test_performance.py`

**Step 1: Write the failing test**

```python
# tests/monitoring/test_performance.py
import datetime

import numpy as np
import pandas as pd
import pytest

from monitoring.performance import (
    compute_performance_metrics,
    compute_rolling_performance,
    PerformanceSnapshot,
    PerformanceTimeline,
)


@pytest.fixture
def predictions_with_gt():
    """Predictions joined with ground truth."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="6h")
    actuals = np.random.choice([0, 1, 2], n, p=[0.35, 0.45, 0.20])
    # Predictions are mostly correct but with some noise
    preds = actuals.copy()
    noise_idx = np.random.choice(n, size=40, replace=False)
    preds[noise_idx] = np.random.choice([0, 1, 2], size=40)

    return pd.DataFrame({
        "timestamp": dates,
        "prediction": preds,
        "actual_decision": actuals,
        "confidence": np.random.uniform(0.4, 0.95, n),
        "feat_qtr": np.random.choice([1, 2, 3, 4], n),
        "feat_ydstogo": np.random.randint(1, 15, n),
    })


def test_compute_performance_metrics(predictions_with_gt):
    snapshot = compute_performance_metrics(
        predictions_with_gt["prediction"],
        predictions_with_gt["actual_decision"],
    )
    assert isinstance(snapshot, PerformanceSnapshot)
    assert 0.0 <= snapshot.accuracy <= 1.0
    assert snapshot.accuracy > 0.5  # Should be mostly correct
    assert snapshot.confusion_matrix.shape == (3, 3)
    assert snapshot.per_class_f1 is not None
    assert len(snapshot.per_class_f1) == 3
    assert snapshot.n_samples == 200


def test_rolling_performance(predictions_with_gt):
    timeline = compute_rolling_performance(
        predictions_with_gt,
        window_days=7,
        step_days=1,
    )
    assert isinstance(timeline, PerformanceTimeline)
    assert len(timeline.snapshots) > 0
    # Each snapshot should have a date and metrics
    for ts, snap in timeline.snapshots:
        assert isinstance(ts, datetime.date)
        assert 0.0 <= snap.accuracy <= 1.0


def test_rolling_performance_to_dataframe(predictions_with_gt):
    timeline = compute_rolling_performance(
        predictions_with_gt, window_days=7, step_days=1,
    )
    df = timeline.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "date" in df.columns
    assert "accuracy" in df.columns
    assert "f1_macro" in df.columns
    assert "n_samples" in df.columns


def test_performance_snapshot_comparison():
    s1 = PerformanceSnapshot(
        accuracy=0.80, f1_macro=0.78, n_samples=100,
        confusion_matrix=np.eye(3), per_class_f1={"go": 0.80, "punt": 0.78, "fg": 0.76},
    )
    s2 = PerformanceSnapshot(
        accuracy=0.72, f1_macro=0.70, n_samples=100,
        confusion_matrix=np.eye(3), per_class_f1={"go": 0.72, "punt": 0.70, "fg": 0.68},
    )
    delta = s1.compare(s2)
    assert delta["accuracy_delta"] == pytest.approx(-0.08, abs=0.001)
    assert delta["f1_delta"] == pytest.approx(-0.08, abs=0.001)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/monitoring/test_performance.py -v
```

Expected: FAIL.

**Step 3: Implement performance tracking**

```python
# packages/monitoring/src/monitoring/performance.py
"""Performance tracking — rolling accuracy, F1, and confusion matrices.

Computes model performance over configurable time windows so you
can see trends: is the model getting worse over time? Did performance
drop after a specific date? Which classes are degrading?

Requires ground truth labels joined with predictions. Ground truth
arrives delayed (24-48 hours for NFL PBP data), so the most recent
window may have incomplete coverage.
"""

import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from nfl.target import INVERSE_LABEL_MAP


@dataclass
class PerformanceSnapshot:
    """Performance metrics for a single time window."""

    accuracy: float
    f1_macro: float
    n_samples: int
    confusion_matrix: np.ndarray
    per_class_f1: dict[str, float] | None = None
    per_class_precision: dict[str, float] | None = None
    per_class_recall: dict[str, float] | None = None

    def compare(self, other: "PerformanceSnapshot") -> dict:
        """Compare this snapshot against another (e.g., baseline)."""
        return {
            "accuracy_delta": other.accuracy - self.accuracy,
            "f1_delta": other.f1_macro - self.f1_macro,
            "n_samples_delta": other.n_samples - self.n_samples,
        }

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "f1_macro": round(self.f1_macro, 4),
            "n_samples": self.n_samples,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class_f1": self.per_class_f1,
        }


@dataclass
class PerformanceTimeline:
    """Performance metrics over a series of time windows."""

    snapshots: list[tuple[datetime.date, PerformanceSnapshot]] = field(
        default_factory=list
    )

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for date, snap in self.snapshots:
            row = {"date": date, "accuracy": snap.accuracy,
                   "f1_macro": snap.f1_macro, "n_samples": snap.n_samples}
            if snap.per_class_f1:
                for cls, f1 in snap.per_class_f1.items():
                    row[f"f1_{cls}"] = f1
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def latest(self) -> PerformanceSnapshot | None:
        return self.snapshots[-1][1] if self.snapshots else None

    @property
    def trend(self) -> str:
        """Simple trend detection: improving, stable, or degrading."""
        if len(self.snapshots) < 3:
            return "insufficient_data"
        recent = [s.accuracy for _, s in self.snapshots[-3:]]
        older = [s.accuracy for _, s in self.snapshots[:3]]
        avg_recent = np.mean(recent)
        avg_older = np.mean(older)
        delta = avg_recent - avg_older
        if delta > 0.02:
            return "improving"
        elif delta < -0.02:
            return "degrading"
        return "stable"


def compute_performance_metrics(
    y_pred: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    labels: list[int] | None = None,
) -> PerformanceSnapshot:
    """Compute classification metrics for a single batch/window.

    Args:
        y_pred: Predicted class labels.
        y_true: Ground truth class labels.
        labels: Class labels to include [0, 1, 2].
    """
    if labels is None:
        labels = [0, 1, 2]

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    # Filter to rows where both exist
    valid = ~(pd.isna(y_pred) | pd.isna(y_true))
    y_pred = y_pred[valid].astype(int)
    y_true = y_true[valid].astype(int)

    if len(y_pred) == 0:
        return PerformanceSnapshot(
            accuracy=0.0, f1_macro=0.0, n_samples=0,
            confusion_matrix=np.zeros((3, 3)),
        )

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )

    class_names = [INVERSE_LABEL_MAP.get(i, str(i)) for i in labels]

    return PerformanceSnapshot(
        accuracy=float(acc),
        f1_macro=float(f1),
        n_samples=len(y_pred),
        confusion_matrix=cm,
        per_class_f1=dict(zip(class_names, [round(float(x), 4) for x in f1_per_class])),
        per_class_precision=dict(zip(class_names, [round(float(x), 4) for x in precision])),
        per_class_recall=dict(zip(class_names, [round(float(x), 4) for x in recall])),
    )


def compute_rolling_performance(
    df: pd.DataFrame,
    window_days: int = 7,
    step_days: int = 1,
    pred_col: str = "prediction",
    actual_col: str = "actual_decision",
    time_col: str = "timestamp",
) -> PerformanceTimeline:
    """Compute rolling performance metrics over a time series.

    Slides a window of `window_days` across the data, stepping
    by `step_days`, and computes metrics for each window.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.dropna(subset=[pred_col, actual_col])

    if len(df) == 0:
        return PerformanceTimeline()

    min_date = df[time_col].dt.date.min()
    max_date = df[time_col].dt.date.max()

    snapshots = []
    window = datetime.timedelta(days=window_days)
    step = datetime.timedelta(days=step_days)

    current = min_date
    while current + window <= max_date + datetime.timedelta(days=1):
        end = current + window
        mask = (df[time_col].dt.date >= current) & (df[time_col].dt.date < end)
        window_df = df[mask]

        if len(window_df) >= 5:  # Minimum samples for meaningful metrics
            snap = compute_performance_metrics(
                window_df[pred_col], window_df[actual_col]
            )
            snapshots.append((current, snap))

        current += step

    return PerformanceTimeline(snapshots=snapshots)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/monitoring/test_performance.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(monitoring): add rolling performance tracker with timeline"
```

---

## Task 4: Prediction Distribution Drift

**Files:**
- Create: `packages/monitoring/src/monitoring/prediction_drift.py`
- Test: `tests/monitoring/test_prediction_drift.py`

**Step 1: Write the failing test**

```python
# tests/monitoring/test_prediction_drift.py
import numpy as np
import pandas as pd
import pytest

from monitoring.prediction_drift import (
    compute_prediction_distribution,
    detect_prediction_drift,
    PredictionDistribution,
    PredictionDriftResult,
)


@pytest.fixture
def baseline_predictions():
    """Training-time prediction distribution: 35% go, 45% punt, 20% FG."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "prediction": np.random.choice([0, 1, 2], n, p=[0.35, 0.45, 0.20]),
        "confidence": np.random.uniform(0.4, 0.95, n),
        "prob_go": np.random.dirichlet([3.5, 4.5, 2.0], n)[:, 0],
        "prob_punt": np.random.dirichlet([3.5, 4.5, 2.0], n)[:, 1],
        "prob_fg": np.random.dirichlet([3.5, 4.5, 2.0], n)[:, 2],
    })


@pytest.fixture
def similar_predictions():
    """Production predictions with similar distribution."""
    np.random.seed(99)
    n = 500
    return pd.DataFrame({
        "prediction": np.random.choice([0, 1, 2], n, p=[0.36, 0.44, 0.20]),
        "confidence": np.random.uniform(0.4, 0.95, n),
        "prob_go": np.random.dirichlet([3.5, 4.5, 2.0], n)[:, 0],
        "prob_punt": np.random.dirichlet([3.5, 4.5, 2.0], n)[:, 1],
        "prob_fg": np.random.dirichlet([3.5, 4.5, 2.0], n)[:, 2],
    })


@pytest.fixture
def drifted_predictions():
    """Predictions with shifted distribution: more aggressive (more go_for_it)."""
    np.random.seed(99)
    n = 500
    return pd.DataFrame({
        "prediction": np.random.choice([0, 1, 2], n, p=[0.55, 0.30, 0.15]),
        "confidence": np.random.uniform(0.3, 0.80, n),
        "prob_go": np.random.dirichlet([5.5, 3.0, 1.5], n)[:, 0],
        "prob_punt": np.random.dirichlet([5.5, 3.0, 1.5], n)[:, 1],
        "prob_fg": np.random.dirichlet([5.5, 3.0, 1.5], n)[:, 2],
    })


def test_compute_prediction_distribution(baseline_predictions):
    dist = compute_prediction_distribution(baseline_predictions)
    assert isinstance(dist, PredictionDistribution)
    assert len(dist.class_proportions) == 3
    assert abs(sum(dist.class_proportions.values()) - 1.0) < 0.01
    assert dist.mean_confidence > 0
    assert dist.confidence_histogram is not None


def test_no_prediction_drift(baseline_predictions, similar_predictions):
    baseline_dist = compute_prediction_distribution(baseline_predictions)
    current_dist = compute_prediction_distribution(similar_predictions)
    result = detect_prediction_drift(baseline_dist, current_dist)
    assert isinstance(result, PredictionDriftResult)
    assert result.severity in ("none", "warning")


def test_prediction_drift_detected(baseline_predictions, drifted_predictions):
    baseline_dist = compute_prediction_distribution(baseline_predictions)
    drifted_dist = compute_prediction_distribution(drifted_predictions)
    result = detect_prediction_drift(baseline_dist, drifted_dist)
    assert result.severity in ("warning", "critical")
    assert result.kl_divergence > 0.01
    assert result.class_shifts is not None
    # go_for_it proportion should show positive shift
    assert result.class_shifts.get("go_for_it", 0) > 0.10


def test_confidence_distribution_shift(baseline_predictions, drifted_predictions):
    baseline_dist = compute_prediction_distribution(baseline_predictions)
    drifted_dist = compute_prediction_distribution(drifted_predictions)
    result = detect_prediction_drift(baseline_dist, drifted_dist)
    assert result.confidence_shift is not None
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/monitoring/test_prediction_drift.py -v
```

Expected: FAIL.

**Step 3: Implement prediction drift detection**

```python
# packages/monitoring/src/monitoring/prediction_drift.py
"""Prediction distribution drift detection.

Monitors what the model is *predicting* (not just input features).
If the model starts predicting "go_for_it" much more often than
during training, something has changed — either the input distribution
shifted or the decision boundary itself is no longer appropriate.

Also monitors confidence calibration: if the model becomes less
confident over time, it's encountering data further from its
training distribution.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp

from nfl.target import INVERSE_LABEL_MAP


@dataclass
class PredictionDistribution:
    """Snapshot of prediction output distribution."""

    class_proportions: dict[str, float]
    mean_confidence: float
    confidence_std: float
    confidence_histogram: dict[str, list] | None = None
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "class_proportions": self.class_proportions,
            "mean_confidence": round(self.mean_confidence, 4),
            "confidence_std": round(self.confidence_std, 4),
            "n_samples": self.n_samples,
        }


@dataclass
class PredictionDriftResult:
    """Result of prediction distribution drift analysis."""

    severity: str  # "none", "warning", "critical"
    kl_divergence: float
    chi_squared_pvalue: float | None = None
    class_shifts: dict[str, float] | None = None
    confidence_shift: float | None = None
    confidence_ks_pvalue: float | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def compute_prediction_distribution(
    df: pd.DataFrame,
    pred_col: str = "prediction",
    confidence_col: str = "confidence",
    n_bins: int = 20,
) -> PredictionDistribution:
    """Compute the prediction output distribution."""
    preds = df[pred_col].to_numpy()
    n = len(preds)

    # Class proportions
    proportions = {}
    for label_int in range(3):
        label_name = INVERSE_LABEL_MAP.get(label_int, str(label_int))
        proportions[label_name] = float(np.mean(preds == label_int))

    # Confidence distribution
    conf = df[confidence_col].to_numpy() if confidence_col in df.columns else np.array([])
    mean_conf = float(np.mean(conf)) if len(conf) > 0 else 0.0
    std_conf = float(np.std(conf)) if len(conf) > 0 else 0.0

    hist = None
    if len(conf) > 0:
        counts, edges = np.histogram(conf, bins=n_bins, range=(0, 1))
        hist = {"counts": counts.tolist(), "bin_edges": edges.tolist()}

    return PredictionDistribution(
        class_proportions=proportions,
        mean_confidence=mean_conf,
        confidence_std=std_conf,
        confidence_histogram=hist,
        n_samples=n,
    )


def detect_prediction_drift(
    baseline: PredictionDistribution,
    current: PredictionDistribution,
    kl_warning: float = 0.05,
    kl_critical: float = 0.15,
) -> PredictionDriftResult:
    """Compare current prediction distribution against baseline.

    Uses:
    - KL divergence on class proportions (how different is the prediction mix?)
    - Per-class proportion shift (which classes changed?)
    - KS test on confidence distribution (is the model less confident?)
    """
    # KL divergence on class proportions
    class_names = sorted(baseline.class_proportions.keys())
    eps = 1e-10
    p = np.array([baseline.class_proportions[c] + eps for c in class_names])
    q = np.array([current.class_proportions.get(c, eps) + eps for c in class_names])
    p /= p.sum()
    q /= q.sum()
    kl_div = float(entropy(q, p))  # KL(current || baseline)

    # Per-class shifts
    class_shifts = {}
    for c in class_names:
        baseline_prop = baseline.class_proportions.get(c, 0)
        current_prop = current.class_proportions.get(c, 0)
        class_shifts[c] = round(current_prop - baseline_prop, 4)

    # Confidence shift
    conf_shift = current.mean_confidence - baseline.mean_confidence

    # KS test on confidence histograms (reconstruct samples)
    ks_pvalue = None
    if baseline.confidence_histogram and current.confidence_histogram:
        baseline_samples = _histogram_to_samples(baseline.confidence_histogram, 1000)
        current_samples = _histogram_to_samples(current.confidence_histogram, 1000)
        _, ks_pvalue = ks_2samp(baseline_samples, current_samples)

    # Determine severity
    if kl_div >= kl_critical:
        severity = "critical"
    elif kl_div >= kl_warning:
        severity = "warning"
    else:
        severity = "none"

    return PredictionDriftResult(
        severity=severity,
        kl_divergence=round(kl_div, 6),
        class_shifts=class_shifts,
        confidence_shift=round(conf_shift, 4),
        confidence_ks_pvalue=round(float(ks_pvalue), 6) if ks_pvalue is not None else None,
    )


def _histogram_to_samples(histogram: dict, n: int = 1000) -> np.ndarray:
    counts = np.array(histogram["counts"])
    edges = np.array(histogram["bin_edges"])
    total = counts.sum()
    if total == 0:
        return np.zeros(n)
    probs = counts / total
    bin_indices = np.random.choice(len(counts), size=n, p=probs)
    return np.array([np.random.uniform(edges[i], edges[i+1]) for i in bin_indices])
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/monitoring/test_prediction_drift.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(monitoring): add prediction distribution drift detection"
```

---

## Task 5: Population Stability Index (PSI)

**Files:**
- Create: `packages/monitoring/src/monitoring/psi.py`
- Test: `tests/monitoring/test_psi.py`

**Step 1: Write the failing test**

```python
# tests/monitoring/test_psi.py
import numpy as np
import pandas as pd
import pytest

from monitoring.psi import (
    compute_psi,
    compute_feature_psi_report,
    PSIResult,
    PSIReport,
)


@pytest.fixture
def baseline_features():
    np.random.seed(42)
    return pd.DataFrame({
        "ydstogo": np.random.randint(1, 15, 1000),
        "yardline_100": np.random.randint(1, 99, 1000),
        "score_diff": np.random.randint(-21, 21, 1000),
    })


@pytest.fixture
def stable_features():
    np.random.seed(99)
    return pd.DataFrame({
        "ydstogo": np.random.randint(1, 15, 500),
        "yardline_100": np.random.randint(1, 99, 500),
        "score_diff": np.random.randint(-21, 21, 500),
    })


@pytest.fixture
def drifted_features():
    np.random.seed(99)
    return pd.DataFrame({
        "ydstogo": np.random.randint(1, 5, 500),  # Shorter yardage
        "yardline_100": np.random.randint(1, 40, 500),  # Closer to end zone
        "score_diff": np.random.randint(-7, 7, 500),  # Tighter games
    })


def test_psi_stable(baseline_features, stable_features):
    result = compute_psi(baseline_features["ydstogo"], stable_features["ydstogo"])
    assert isinstance(result, PSIResult)
    assert result.psi < 0.10  # Should be stable
    assert result.severity == "none"


def test_psi_drifted(baseline_features, drifted_features):
    result = compute_psi(baseline_features["ydstogo"], drifted_features["ydstogo"])
    assert result.psi > 0.10  # Should detect drift
    assert result.severity in ("warning", "critical")


def test_feature_psi_report(baseline_features, drifted_features):
    report = compute_feature_psi_report(baseline_features, drifted_features)
    assert isinstance(report, PSIReport)
    assert len(report.features) == 3
    assert "ydstogo" in report.features
    # At least one feature should show drift
    assert any(r.severity != "none" for r in report.features.values())


def test_psi_report_to_dataframe(baseline_features, drifted_features):
    report = compute_feature_psi_report(baseline_features, drifted_features)
    df = report.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "feature" in df.columns
    assert "psi" in df.columns
    assert "severity" in df.columns


def test_psi_buckets(baseline_features, stable_features):
    result = compute_psi(
        baseline_features["ydstogo"], stable_features["ydstogo"], n_bins=10
    )
    assert result.bucket_details is not None
    assert len(result.bucket_details) == 10
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/monitoring/test_psi.py -v
```

Expected: FAIL.

**Step 3: Implement PSI**

```python
# packages/monitoring/src/monitoring/psi.py
"""Population Stability Index (PSI) for feature-level monitoring.

PSI measures how much a feature's distribution has shifted from
training (baseline) to production. Unlike drift detection which
uses KS tests, PSI provides a single scalar score per feature
that's easy to track over time and set alerts on.

PSI interpretation:
  < 0.10: No significant shift
  0.10 - 0.25: Moderate shift, investigate
  > 0.25: Significant shift, likely action needed

PSI = Σ (actual% - expected%) × ln(actual% / expected%)
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PSIBucket:
    """Details for a single PSI histogram bucket."""

    bin_low: float
    bin_high: float
    expected_pct: float
    actual_pct: float
    psi_contribution: float


@dataclass
class PSIResult:
    """PSI computation result for a single feature."""

    feature: str
    psi: float
    severity: str  # "none", "warning", "critical"
    bucket_details: list[PSIBucket] | None = None

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 4),
            "severity": self.severity,
        }


@dataclass
class PSIReport:
    """PSI results for all monitored features."""

    features: dict[str, PSIResult] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [r.to_dict() for r in self.features.values()]
        return (
            pd.DataFrame(rows)
            .sort_values("psi", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def max_psi(self) -> float:
        if not self.features:
            return 0.0
        return max(r.psi for r in self.features.values())

    @property
    def has_critical(self) -> bool:
        return any(r.severity == "critical" for r in self.features.values())


def compute_psi(
    baseline: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    n_bins: int = 10,
    psi_warning: float = 0.10,
    psi_critical: float = 0.25,
    feature_name: str = "",
) -> PSIResult:
    """Compute PSI between baseline and current distributions.

    Uses quantile-based binning from the baseline distribution to
    ensure each bin has roughly equal baseline representation.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    # Remove NaN
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]

    if len(baseline) == 0 or len(current) == 0:
        return PSIResult(feature=feature_name, psi=0.0, severity="none")

    # Quantile-based bins from baseline
    quantiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(baseline, quantiles)
    bins = np.unique(bins)  # Remove duplicates

    if len(bins) < 2:
        return PSIResult(feature=feature_name, psi=0.0, severity="none")

    # Count observations in each bin
    baseline_counts = np.histogram(baseline, bins=bins)[0]
    current_counts = np.histogram(current, bins=bins)[0]

    # Convert to percentages
    eps = 1e-4  # Avoid division by zero
    baseline_pct = (baseline_counts / len(baseline)) + eps
    current_pct = (current_counts / len(current)) + eps

    # Normalize
    baseline_pct /= baseline_pct.sum()
    current_pct /= current_pct.sum()

    # PSI per bucket
    psi_per_bucket = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
    total_psi = float(np.sum(psi_per_bucket))

    # Bucket details
    buckets = []
    for i in range(len(psi_per_bucket)):
        buckets.append(PSIBucket(
            bin_low=float(bins[i]),
            bin_high=float(bins[i + 1]) if i + 1 < len(bins) else float("inf"),
            expected_pct=float(baseline_pct[i]),
            actual_pct=float(current_pct[i]),
            psi_contribution=float(psi_per_bucket[i]),
        ))

    # Severity
    if total_psi >= psi_critical:
        severity = "critical"
    elif total_psi >= psi_warning:
        severity = "warning"
    else:
        severity = "none"

    return PSIResult(
        feature=feature_name,
        psi=total_psi,
        severity=severity,
        bucket_details=buckets,
    )


def compute_feature_psi_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str] | None = None,
    n_bins: int = 10,
) -> PSIReport:
    """Compute PSI for all numeric features in the DataFrames."""
    if columns is None:
        columns = [
            c for c in baseline_df.columns
            if c in current_df.columns and pd.api.types.is_numeric_dtype(baseline_df[c])
        ]

    results = {}
    for col in columns:
        results[col] = compute_psi(
            baseline_df[col], current_df[col],
            n_bins=n_bins, feature_name=col,
        )

    return PSIReport(features=results)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/monitoring/test_psi.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(monitoring): add Population Stability Index computation"
```

---

## Task 6: Segmented Performance Analysis

**Files:**
- Create: `packages/monitoring/src/monitoring/segments.py`
- Test: `tests/monitoring/test_segments.py`

**Step 1: Write the failing test**

```python
# tests/monitoring/test_segments.py
import numpy as np
import pandas as pd
import pytest

from monitoring.segments import (
    compute_segmented_performance,
    find_degraded_segments,
    SegmentedReport,
)


@pytest.fixture
def predictions_with_segments():
    np.random.seed(42)
    n = 400
    qtr = np.random.choice([1, 2, 3, 4], n)
    actuals = np.random.choice([0, 1, 2], n, p=[0.35, 0.45, 0.20])
    preds = actuals.copy()

    # Deliberately degrade Q4 predictions
    q4_mask = qtr == 4
    q4_indices = np.where(q4_mask)[0]
    noise_q4 = np.random.choice(q4_indices, size=len(q4_indices) // 2, replace=False)
    preds[noise_q4] = np.random.choice([0, 1, 2], size=len(noise_q4))

    return pd.DataFrame({
        "prediction": preds,
        "actual_decision": actuals,
        "feat_qtr": qtr,
        "feat_is_trailing": np.random.choice([0, 1], n),
        "feat_is_fg_range": np.random.choice([0, 1], n),
    })


def test_segmented_performance(predictions_with_segments):
    report = compute_segmented_performance(
        predictions_with_segments,
        segment_columns=["feat_qtr"],
    )
    assert isinstance(report, SegmentedReport)
    assert "feat_qtr" in report.segments
    # Should have 4 segments (quarters 1-4)
    assert len(report.segments["feat_qtr"]) == 4


def test_segmented_performance_multi(predictions_with_segments):
    report = compute_segmented_performance(
        predictions_with_segments,
        segment_columns=["feat_qtr", "feat_is_trailing"],
    )
    assert "feat_qtr" in report.segments
    assert "feat_is_trailing" in report.segments


def test_find_degraded_segments(predictions_with_segments):
    report = compute_segmented_performance(
        predictions_with_segments,
        segment_columns=["feat_qtr"],
    )
    degraded = find_degraded_segments(report, accuracy_threshold=0.70)
    # Q4 should be degraded since we deliberately added noise
    q4_degraded = [
        d for d in degraded if d["segment_col"] == "feat_qtr" and d["segment_value"] == 4
    ]
    assert len(q4_degraded) >= 1 or len(degraded) > 0


def test_segmented_report_to_dataframe(predictions_with_segments):
    report = compute_segmented_performance(
        predictions_with_segments,
        segment_columns=["feat_qtr"],
    )
    df = report.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "segment_col" in df.columns
    assert "segment_value" in df.columns
    assert "accuracy" in df.columns
    assert "n_samples" in df.columns
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/monitoring/test_segments.py -v
```

Expected: FAIL.

**Step 3: Implement segmented analysis**

```python
# packages/monitoring/src/monitoring/segments.py
"""Segmented performance analysis — find WHERE the model is failing.

Overall accuracy might look fine while specific segments degrade.
For the 4th-down model, important segments include:
- By quarter (model often degrades in late-game situations)
- By field position (FG range vs deep in own territory)
- By score context (trailing vs leading, close vs blowout)
- By yards to go (short yardage vs long yardage)

This helps answer: "The model is 78% accurate overall, but only
55% accurate on 4th-and-short in the 4th quarter when trailing."
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from monitoring.performance import compute_performance_metrics, PerformanceSnapshot


@dataclass
class SegmentedReport:
    """Performance broken down by segment."""

    segments: dict[str, dict[str, PerformanceSnapshot]] = field(default_factory=dict)
    # Structure: {segment_column: {segment_value: PerformanceSnapshot}}

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for seg_col, seg_values in self.segments.items():
            for seg_val, snap in seg_values.items():
                rows.append({
                    "segment_col": seg_col,
                    "segment_value": seg_val,
                    "accuracy": snap.accuracy,
                    "f1_macro": snap.f1_macro,
                    "n_samples": snap.n_samples,
                    **(snap.per_class_f1 or {}),
                })
        return pd.DataFrame(rows)


def compute_segmented_performance(
    df: pd.DataFrame,
    segment_columns: list[str],
    pred_col: str = "prediction",
    actual_col: str = "actual_decision",
    min_samples: int = 10,
) -> SegmentedReport:
    """Compute performance metrics for each segment.

    Args:
        df: Predictions with ground truth and feature columns.
        segment_columns: Columns to segment by (e.g., ["feat_qtr"]).
        pred_col: Prediction column name.
        actual_col: Ground truth column name.
        min_samples: Minimum samples for a segment to be included.
    """
    report = SegmentedReport()

    for col in segment_columns:
        if col not in df.columns:
            continue

        segments = {}
        for value in sorted(df[col].dropna().unique()):
            mask = df[col] == value
            seg_df = df[mask]

            if len(seg_df) < min_samples:
                continue

            snap = compute_performance_metrics(
                seg_df[pred_col], seg_df[actual_col]
            )
            segments[value] = snap

        report.segments[col] = segments

    return report


def find_degraded_segments(
    report: SegmentedReport,
    accuracy_threshold: float = 0.65,
    f1_threshold: float = 0.60,
) -> list[dict]:
    """Find segments where performance is below threshold.

    Returns a list of degraded segments sorted by severity
    (lowest accuracy first).
    """
    degraded = []

    for seg_col, seg_values in report.segments.items():
        for seg_val, snap in seg_values.items():
            if snap.accuracy < accuracy_threshold or snap.f1_macro < f1_threshold:
                degraded.append({
                    "segment_col": seg_col,
                    "segment_value": seg_val,
                    "accuracy": snap.accuracy,
                    "f1_macro": snap.f1_macro,
                    "n_samples": snap.n_samples,
                    "per_class_f1": snap.per_class_f1,
                })

    degraded.sort(key=lambda x: x["accuracy"])
    return degraded
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/monitoring/test_segments.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(monitoring): add segmented performance analysis"
```

---

## Task 7: Retrain Advisor

**Files:**
- Create: `packages/monitoring/src/monitoring/retrain_advisor.py`
- Test: `tests/monitoring/test_retrain_advisor.py`

**Step 1: Write the failing test**

```python
# tests/monitoring/test_retrain_advisor.py
import pytest

from monitoring.retrain_advisor import (
    RetrainRecommendation,
    RetrainUrgency,
    compute_retrain_recommendation,
)
from monitoring.config import AlertThresholds


def test_no_retrain_needed():
    rec = compute_retrain_recommendation(
        accuracy_delta=-0.01,
        f1_delta=-0.01,
        max_psi=0.05,
        prediction_kl=0.02,
        n_degraded_segments=0,
        days_since_training=30,
    )
    assert isinstance(rec, RetrainRecommendation)
    assert rec.urgency == RetrainUrgency.NONE
    assert rec.score < 30


def test_retrain_recommended():
    rec = compute_retrain_recommendation(
        accuracy_delta=-0.06,
        f1_delta=-0.08,
        max_psi=0.30,
        prediction_kl=0.20,
        n_degraded_segments=3,
        days_since_training=120,
    )
    assert rec.urgency in (RetrainUrgency.RECOMMENDED, RetrainUrgency.URGENT)
    assert rec.score > 50


def test_urgent_retrain():
    rec = compute_retrain_recommendation(
        accuracy_delta=-0.12,
        f1_delta=-0.15,
        max_psi=0.50,
        prediction_kl=0.30,
        n_degraded_segments=5,
        days_since_training=200,
    )
    assert rec.urgency == RetrainUrgency.URGENT
    assert rec.score > 80


def test_recommendation_has_reasons():
    rec = compute_retrain_recommendation(
        accuracy_delta=-0.06,
        f1_delta=-0.08,
        max_psi=0.30,
        prediction_kl=0.20,
        n_degraded_segments=3,
        days_since_training=120,
    )
    assert len(rec.reasons) > 0
    assert all(isinstance(r, str) for r in rec.reasons)


def test_recommendation_to_dict():
    rec = compute_retrain_recommendation(
        accuracy_delta=-0.03, f1_delta=-0.02, max_psi=0.08,
        prediction_kl=0.03, n_degraded_segments=0, days_since_training=45,
    )
    d = rec.to_dict()
    assert "urgency" in d
    assert "score" in d
    assert "reasons" in d
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/monitoring/test_retrain_advisor.py -v
```

Expected: FAIL.

**Step 3: Implement the retrain advisor**

```python
# packages/monitoring/src/monitoring/retrain_advisor.py
"""Automated retrain recommendation engine.

Aggregates all monitoring signals into a single scored recommendation:
should we retrain the model? How urgently?

The scoring system weights different signals:
- Performance degradation (highest weight — direct impact)
- Feature drift via PSI (leading indicator)
- Prediction distribution drift (output-level change)
- Segment degradation count (breadth of impact)
- Time since last training (staleness)

Each signal contributes 0-20 points to a 0-100 score:
  0-30: No retrain needed
  30-60: Retrain recommended (plan for next cycle)
  60-100: Urgent retrain (schedule immediately)
"""

from dataclasses import dataclass, field
from enum import Enum

from monitoring.config import AlertThresholds


class RetrainUrgency(Enum):
    NONE = "none"
    RECOMMENDED = "recommended"
    URGENT = "urgent"


@dataclass
class RetrainRecommendation:
    """Scored retrain recommendation with supporting reasons."""

    urgency: RetrainUrgency
    score: int  # 0-100
    reasons: list[str] = field(default_factory=list)
    signal_scores: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "urgency": self.urgency.value,
            "score": self.score,
            "reasons": self.reasons,
            "signal_scores": self.signal_scores,
        }


def compute_retrain_recommendation(
    accuracy_delta: float,
    f1_delta: float,
    max_psi: float,
    prediction_kl: float,
    n_degraded_segments: int,
    days_since_training: int,
    thresholds: AlertThresholds | None = None,
) -> RetrainRecommendation:
    """Compute a scored retrain recommendation from monitoring signals.

    All delta values should be negative when performance degrades
    (current - baseline). PSI and KL are always positive.
    """
    if thresholds is None:
        thresholds = AlertThresholds()

    reasons = []
    signal_scores = {}
    total = 0

    # Signal 1: Accuracy degradation (0-25 points)
    acc_drop = abs(accuracy_delta) if accuracy_delta < 0 else 0
    if acc_drop >= thresholds.accuracy_red:
        acc_score = 25
        reasons.append(f"Accuracy dropped {acc_drop:.1%} (critical threshold: {thresholds.accuracy_red:.1%})")
    elif acc_drop >= thresholds.accuracy_yellow:
        acc_score = 15
        reasons.append(f"Accuracy dropped {acc_drop:.1%} (warning threshold: {thresholds.accuracy_yellow:.1%})")
    elif acc_drop > 0.01:
        acc_score = 5
    else:
        acc_score = 0
    signal_scores["accuracy"] = acc_score
    total += acc_score

    # Signal 2: F1 degradation (0-20 points)
    f1_drop = abs(f1_delta) if f1_delta < 0 else 0
    if f1_drop >= thresholds.f1_red:
        f1_score = 20
        reasons.append(f"F1 dropped {f1_drop:.1%} — class-level performance degrading")
    elif f1_drop >= thresholds.f1_yellow:
        f1_score = 12
        reasons.append(f"F1 dropped {f1_drop:.1%}")
    else:
        f1_score = 0
    signal_scores["f1"] = f1_score
    total += f1_score

    # Signal 3: Feature drift via PSI (0-20 points)
    if max_psi >= thresholds.psi_red:
        psi_score = 20
        reasons.append(f"Feature PSI reached {max_psi:.3f} — significant input distribution shift")
    elif max_psi >= thresholds.psi_yellow:
        psi_score = 10
        reasons.append(f"Feature PSI at {max_psi:.3f} — moderate input drift")
    else:
        psi_score = 0
    signal_scores["psi"] = psi_score
    total += psi_score

    # Signal 4: Prediction distribution drift (0-15 points)
    if prediction_kl >= 0.15:
        pred_score = 15
        reasons.append(f"Prediction distribution KL divergence: {prediction_kl:.4f} — model output shifted significantly")
    elif prediction_kl >= 0.05:
        pred_score = 8
        reasons.append(f"Prediction distribution showing drift (KL: {prediction_kl:.4f})")
    else:
        pred_score = 0
    signal_scores["prediction_drift"] = pred_score
    total += pred_score

    # Signal 5: Degraded segments (0-10 points)
    if n_degraded_segments >= 3:
        seg_score = 10
        reasons.append(f"{n_degraded_segments} segments below performance threshold — widespread degradation")
    elif n_degraded_segments >= 1:
        seg_score = 5
        reasons.append(f"{n_degraded_segments} segment(s) underperforming")
    else:
        seg_score = 0
    signal_scores["segments"] = seg_score
    total += seg_score

    # Signal 6: Staleness (0-10 points)
    if days_since_training >= 180:
        stale_score = 10
        reasons.append(f"Model is {days_since_training} days old — consider scheduled retrain")
    elif days_since_training >= 90:
        stale_score = 5
        reasons.append(f"Model is {days_since_training} days old")
    else:
        stale_score = 0
    signal_scores["staleness"] = stale_score
    total += stale_score

    # Determine urgency
    if total >= 60:
        urgency = RetrainUrgency.URGENT
    elif total >= 30:
        urgency = RetrainUrgency.RECOMMENDED
    else:
        urgency = RetrainUrgency.NONE

    if not reasons:
        reasons.append("All monitoring signals are within acceptable ranges")

    return RetrainRecommendation(
        urgency=urgency,
        score=min(total, 100),
        reasons=reasons,
        signal_scores=signal_scores,
    )
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/monitoring/test_retrain_advisor.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(monitoring): add automated retrain recommendation engine"
```

---

## Task 8: Alert System

**Files:**
- Create: `packages/monitoring/src/monitoring/alerts.py`
- Test: `tests/monitoring/test_alerts.py`

**Step 1: Write the failing test**

```python
# tests/monitoring/test_alerts.py
import datetime

import pytest

from monitoring.alerts import (
    Alert,
    AlertLevel,
    AlertHistory,
    create_alert,
    evaluate_alerts,
)
from monitoring.performance import PerformanceSnapshot
from monitoring.prediction_drift import PredictionDriftResult
from monitoring.psi import PSIReport, PSIResult
from monitoring.config import AlertThresholds
import numpy as np


def test_create_alert():
    alert = create_alert(
        level=AlertLevel.RED,
        category="performance",
        message="Accuracy dropped below threshold",
        metric_name="accuracy",
        metric_value=0.68,
        threshold=0.75,
    )
    assert isinstance(alert, Alert)
    assert alert.level == AlertLevel.RED
    assert alert.timestamp is not None


def test_evaluate_alerts_green():
    thresholds = AlertThresholds()
    baseline_acc = 0.80

    current = PerformanceSnapshot(
        accuracy=0.79, f1_macro=0.77, n_samples=100,
        confusion_matrix=np.eye(3),
    )
    drift = PredictionDriftResult(severity="none", kl_divergence=0.01)
    psi = PSIReport(features={"a": PSIResult(feature="a", psi=0.03, severity="none")})

    alerts = evaluate_alerts(
        current_performance=current,
        baseline_accuracy=baseline_acc,
        baseline_f1=0.78,
        prediction_drift=drift,
        psi_report=psi,
        thresholds=thresholds,
    )
    red_alerts = [a for a in alerts if a.level == AlertLevel.RED]
    assert len(red_alerts) == 0


def test_evaluate_alerts_red():
    thresholds = AlertThresholds()
    current = PerformanceSnapshot(
        accuracy=0.65, f1_macro=0.60, n_samples=100,
        confusion_matrix=np.eye(3),
    )
    drift = PredictionDriftResult(severity="critical", kl_divergence=0.25)
    psi = PSIReport(features={"a": PSIResult(feature="a", psi=0.40, severity="critical")})

    alerts = evaluate_alerts(
        current_performance=current,
        baseline_accuracy=0.80,
        baseline_f1=0.78,
        prediction_drift=drift,
        psi_report=psi,
        thresholds=thresholds,
    )
    red_alerts = [a for a in alerts if a.level == AlertLevel.RED]
    assert len(red_alerts) > 0


def test_alert_history():
    history = AlertHistory()
    history.add(create_alert(AlertLevel.GREEN, "test", "All good", "acc", 0.80, 0.75))
    history.add(create_alert(AlertLevel.RED, "test", "Bad", "acc", 0.65, 0.75))
    assert len(history.alerts) == 2
    assert history.latest_level == AlertLevel.RED
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/monitoring/test_alerts.py -v
```

Expected: FAIL.

**Step 3: Implement the alert system**

```python
# packages/monitoring/src/monitoring/alerts.py
"""Traffic-light alerting system.

Evaluates all monitoring signals against configured thresholds
and produces color-coded alerts (GREEN/YELLOW/RED).

GREEN: Everything is healthy, no action needed.
YELLOW: Warning — investigate within a week.
RED: Critical — investigate immediately, consider retraining.
"""

import datetime
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from monitoring.config import AlertThresholds
from monitoring.performance import PerformanceSnapshot
from monitoring.prediction_drift import PredictionDriftResult
from monitoring.psi import PSIReport


class AlertLevel(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class Alert:
    """A single monitoring alert."""

    level: AlertLevel
    category: str          # "performance", "prediction_drift", "feature_drift"
    message: str
    metric_name: str
    metric_value: float | str
    threshold: float | str
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now())

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "category": self.category,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AlertHistory:
    """Collection of alerts over time."""

    alerts: list[Alert] = field(default_factory=list)

    def add(self, alert: Alert) -> None:
        self.alerts.append(alert)

    @property
    def latest_level(self) -> AlertLevel:
        if not self.alerts:
            return AlertLevel.GREEN
        return max(self.alerts, key=lambda a: _level_order(a.level)).level

    def filter_by_level(self, level: AlertLevel) -> list[Alert]:
        return [a for a in self.alerts if a.level == level]

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([a.to_dict() for a in self.alerts])


def _level_order(level: AlertLevel) -> int:
    return {"green": 0, "yellow": 1, "red": 2}[level.value]


def create_alert(
    level: AlertLevel,
    category: str,
    message: str,
    metric_name: str,
    metric_value: float | str,
    threshold: float | str,
) -> Alert:
    return Alert(
        level=level, category=category, message=message,
        metric_name=metric_name, metric_value=metric_value,
        threshold=threshold,
    )


def evaluate_alerts(
    current_performance: PerformanceSnapshot,
    baseline_accuracy: float,
    baseline_f1: float,
    prediction_drift: PredictionDriftResult | None = None,
    psi_report: PSIReport | None = None,
    thresholds: AlertThresholds | None = None,
) -> list[Alert]:
    """Evaluate all signals and generate alerts."""
    if thresholds is None:
        thresholds = AlertThresholds()

    alerts = []

    # Performance alerts
    acc_drop = baseline_accuracy - current_performance.accuracy
    if acc_drop >= thresholds.accuracy_red:
        alerts.append(create_alert(
            AlertLevel.RED, "performance",
            f"Accuracy dropped {acc_drop:.1%} from baseline ({baseline_accuracy:.1%} → {current_performance.accuracy:.1%})",
            "accuracy_drop", round(acc_drop, 4), thresholds.accuracy_red,
        ))
    elif acc_drop >= thresholds.accuracy_yellow:
        alerts.append(create_alert(
            AlertLevel.YELLOW, "performance",
            f"Accuracy dropped {acc_drop:.1%} from baseline",
            "accuracy_drop", round(acc_drop, 4), thresholds.accuracy_yellow,
        ))

    f1_drop = baseline_f1 - current_performance.f1_macro
    if f1_drop >= thresholds.f1_red:
        alerts.append(create_alert(
            AlertLevel.RED, "performance",
            f"F1 dropped {f1_drop:.1%} from baseline",
            "f1_drop", round(f1_drop, 4), thresholds.f1_red,
        ))
    elif f1_drop >= thresholds.f1_yellow:
        alerts.append(create_alert(
            AlertLevel.YELLOW, "performance",
            f"F1 dropped {f1_drop:.1%} from baseline",
            "f1_drop", round(f1_drop, 4), thresholds.f1_yellow,
        ))

    # Prediction drift alerts
    if prediction_drift:
        if prediction_drift.kl_divergence >= thresholds.prediction_kl_red:
            alerts.append(create_alert(
                AlertLevel.RED, "prediction_drift",
                f"Prediction distribution KL divergence: {prediction_drift.kl_divergence:.4f}",
                "prediction_kl", prediction_drift.kl_divergence, thresholds.prediction_kl_red,
            ))
        elif prediction_drift.kl_divergence >= thresholds.prediction_kl_yellow:
            alerts.append(create_alert(
                AlertLevel.YELLOW, "prediction_drift",
                f"Prediction distribution showing drift (KL: {prediction_drift.kl_divergence:.4f})",
                "prediction_kl", prediction_drift.kl_divergence, thresholds.prediction_kl_yellow,
            ))

    # PSI alerts
    if psi_report:
        for feat_name, psi_result in psi_report.features.items():
            if psi_result.psi >= thresholds.psi_red:
                alerts.append(create_alert(
                    AlertLevel.RED, "feature_drift",
                    f"Feature '{feat_name}' PSI = {psi_result.psi:.3f} — significant shift",
                    f"psi_{feat_name}", psi_result.psi, thresholds.psi_red,
                ))
            elif psi_result.psi >= thresholds.psi_yellow:
                alerts.append(create_alert(
                    AlertLevel.YELLOW, "feature_drift",
                    f"Feature '{feat_name}' PSI = {psi_result.psi:.3f} — moderate shift",
                    f"psi_{feat_name}", psi_result.psi, thresholds.psi_yellow,
                ))

    # If no alerts, add a green one
    if not alerts:
        alerts.append(create_alert(
            AlertLevel.GREEN, "overall",
            "All monitoring signals within acceptable ranges",
            "overall_status", "healthy", "all thresholds",
        ))

    return alerts
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/monitoring/test_alerts.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(monitoring): add traffic-light alert system"
```

---

## Task 9: Streamlit — Monitoring Overview Page

**Files:**
- Create: `packages/dashboard/src/dashboard/pages/12_monitoring_overview.py`

**Step 1: Implement the executive dashboard**

```python
# packages/dashboard/src/dashboard/pages/12_monitoring_overview.py
"""Model Monitoring Overview — executive dashboard with traffic lights."""

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from monitoring.config import MonitoringConfig
from monitoring.prediction_log import load_predictions
from monitoring.performance import (
    compute_performance_metrics,
    compute_rolling_performance,
)
from monitoring.prediction_drift import (
    compute_prediction_distribution,
    detect_prediction_drift,
)
from monitoring.psi import compute_feature_psi_report
from monitoring.segments import compute_segmented_performance, find_degraded_segments
from monitoring.retrain_advisor import compute_retrain_recommendation
from monitoring.alerts import evaluate_alerts, AlertLevel

st.set_page_config(layout="wide")
st.header("📡 Model Monitoring Overview")

# --- Configuration ---
config = MonitoringConfig()
model_dir = st.sidebar.text_input("Model directory", "models/latest")

# Load model metadata for baseline
baseline_accuracy = 0.78
baseline_f1 = 0.76
model_version = "unknown"
days_since_training = 30

metadata_path = Path(model_dir) / "metadata.json"
if metadata_path.exists():
    with open(metadata_path) as f:
        meta = json.load(f)
    eval_data = meta.get("evaluation", {})
    baseline_accuracy = eval_data.get("accuracy", baseline_accuracy)
    model_version = meta.get("version", model_version)
    trained_at = meta.get("trained_at", "")
    if trained_at:
        try:
            trained_date = datetime.datetime.fromisoformat(trained_at).date()
            days_since_training = (datetime.date.today() - trained_date).days
        except ValueError:
            pass

st.sidebar.markdown(f"**Model:** {model_version}")
st.sidebar.markdown(f"**Baseline accuracy:** {baseline_accuracy:.1%}")
st.sidebar.markdown(f"**Days since training:** {days_since_training}")

# --- Load Data ---
# For the demo, allow CSV upload as a stand-in for the prediction log
st.sidebar.subheader("Data Source")
data_source = st.sidebar.radio("Source", ["Upload predictions CSV", "Load from log directory"])

df = None
if data_source == "Upload predictions CSV":
    uploaded = st.sidebar.file_uploader("Predictions with ground truth", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
elif data_source == "Load from log directory":
    config.ensure_dirs()
    df = load_predictions(config)

if df is None or len(df) == 0:
    st.info(
        "Upload a predictions CSV with columns: timestamp, prediction, "
        "actual_decision, confidence, prob_go, prob_punt, prob_fg, "
        "and feat_* columns for segment analysis."
    )
    st.stop()

st.caption(f"Loaded {len(df):,} predictions")

# --- Compute All Signals ---
with st.spinner("Computing monitoring signals..."):
    # Performance
    has_gt = "actual_decision" in df.columns
    performance = None
    timeline = None
    if has_gt:
        performance = compute_performance_metrics(df["prediction"], df["actual_decision"])
        if "timestamp" in df.columns:
            timeline = compute_rolling_performance(df, window_days=7)

    # Prediction drift (compare to uniform baseline or provided baseline)
    current_dist = compute_prediction_distribution(df)
    # Use training-time distribution as baseline (approximate from first 30%)
    n_baseline = max(int(len(df) * 0.3), 50)
    baseline_dist = compute_prediction_distribution(df.iloc[:n_baseline])
    current_recent = compute_prediction_distribution(df.iloc[-n_baseline:])
    pred_drift = detect_prediction_drift(baseline_dist, current_recent)

    # PSI (compare early vs recent)
    feat_cols = [c for c in df.columns if c.startswith("feat_") and pd.api.types.is_numeric_dtype(df[c])]
    psi_report = None
    if feat_cols:
        psi_report = compute_feature_psi_report(
            df.iloc[:n_baseline][feat_cols],
            df.iloc[-n_baseline:][feat_cols],
        )

    # Segments
    seg_report = None
    degraded = []
    if has_gt and feat_cols:
        seg_report = compute_segmented_performance(df, segment_columns=feat_cols[:4])
        degraded = find_degraded_segments(seg_report)

    # Alerts
    alerts = []
    if performance:
        alerts = evaluate_alerts(
            current_performance=performance,
            baseline_accuracy=baseline_accuracy,
            baseline_f1=baseline_f1,
            prediction_drift=pred_drift,
            psi_report=psi_report,
        )

    # Retrain recommendation
    retrain_rec = compute_retrain_recommendation(
        accuracy_delta=(performance.accuracy - baseline_accuracy) if performance else 0,
        f1_delta=(performance.f1_macro - baseline_f1) if performance else 0,
        max_psi=psi_report.max_psi if psi_report else 0,
        prediction_kl=pred_drift.kl_divergence,
        n_degraded_segments=len(degraded),
        days_since_training=days_since_training,
    )

# --- Traffic Light Banner ---
worst_alert = AlertLevel.GREEN
for a in alerts:
    if a.level == AlertLevel.RED:
        worst_alert = AlertLevel.RED
        break
    elif a.level == AlertLevel.YELLOW:
        worst_alert = AlertLevel.YELLOW

banner_config = {
    AlertLevel.GREEN: ("✅", "success", "All systems healthy"),
    AlertLevel.YELLOW: ("⚠️", "warning", "Warnings detected — review recommended"),
    AlertLevel.RED: ("🚨", "error", "Critical issues — immediate action needed"),
}
icon, method, text = banner_config[worst_alert]
getattr(st, method)(f"{icon} **{text}**")

# --- KPI Row ---
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    if performance:
        delta = performance.accuracy - baseline_accuracy
        st.metric("Accuracy", f"{performance.accuracy:.1%}",
                  delta=f"{delta:+.1%}", delta_color="normal")
    else:
        st.metric("Accuracy", "N/A (no ground truth)")

with kpi2:
    if performance:
        delta = performance.f1_macro - baseline_f1
        st.metric("F1 (macro)", f"{performance.f1_macro:.1%}",
                  delta=f"{delta:+.1%}", delta_color="normal")

with kpi3:
    st.metric("Prediction KL", f"{pred_drift.kl_divergence:.4f}",
              delta=f"{'⚠️' if pred_drift.severity != 'none' else '✅'}")

with kpi4:
    if psi_report:
        st.metric("Max PSI", f"{psi_report.max_psi:.3f}",
                  delta=f"{'🚨' if psi_report.has_critical else '✅'}")

with kpi5:
    urgency_emoji = {"none": "✅", "recommended": "⚠️", "urgent": "🚨"}
    st.metric("Retrain Score", f"{retrain_rec.score}/100",
              delta=urgency_emoji.get(retrain_rec.urgency.value, ""))

# --- Alerts ---
st.subheader("Active Alerts")
red_alerts = [a for a in alerts if a.level == AlertLevel.RED]
yellow_alerts = [a for a in alerts if a.level == AlertLevel.YELLOW]
green_alerts = [a for a in alerts if a.level == AlertLevel.GREEN]

for a in red_alerts:
    st.error(f"🔴 **{a.category}**: {a.message}")
for a in yellow_alerts:
    st.warning(f"🟡 **{a.category}**: {a.message}")
if not red_alerts and not yellow_alerts:
    for a in green_alerts:
        st.success(f"🟢 {a.message}")

# --- Performance Timeline ---
if timeline and len(timeline.snapshots) > 0:
    st.subheader("Performance Over Time")
    tl_df = timeline.to_dataframe()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tl_df["date"], y=tl_df["accuracy"], name="Accuracy", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=tl_df["date"], y=tl_df["f1_macro"], name="F1 (macro)", mode="lines+markers"))
    fig.add_hline(y=baseline_accuracy, line_dash="dash", line_color="green",
                  annotation_text=f"Baseline ({baseline_accuracy:.1%})")
    fig.add_hline(y=baseline_accuracy - 0.05, line_dash="dash", line_color="red",
                  annotation_text="Red threshold")
    fig.update_layout(title="Rolling 7-Day Performance", yaxis_title="Score",
                      yaxis_range=[0, 1], height=400)
    st.plotly_chart(fig, use_container_width=True)

    trend_emoji = {"improving": "📈", "stable": "➡️", "degrading": "📉", "insufficient_data": "❓"}
    st.markdown(f"**Trend:** {trend_emoji.get(timeline.trend, '')} {timeline.trend}")

# --- Retrain Recommendation ---
st.subheader("Retrain Recommendation")
rec_config = {
    "none": ("✅", "success"),
    "recommended": ("⚠️", "warning"),
    "urgent": ("🚨", "error"),
}
rec_icon, rec_method = rec_config.get(retrain_rec.urgency.value, ("❓", "info"))
getattr(st, rec_method)(
    f"{rec_icon} **{retrain_rec.urgency.value.upper()}** (Score: {retrain_rec.score}/100)"
)

for reason in retrain_rec.reasons:
    st.markdown(f"- {reason}")

# Signal breakdown
if retrain_rec.signal_scores:
    fig_signals = px.bar(
        x=list(retrain_rec.signal_scores.values()),
        y=list(retrain_rec.signal_scores.keys()),
        orientation="h",
        labels={"x": "Score (0-25)", "y": "Signal"},
        title="Retrain Score Breakdown",
        color=list(retrain_rec.signal_scores.values()),
        color_continuous_scale=["green", "yellow", "red"],
        range_color=[0, 25],
    )
    fig_signals.update_layout(height=250, showlegend=False)
    st.plotly_chart(fig_signals, use_container_width=True)

# --- Quick Links ---
st.markdown("---")
st.markdown("### Detailed Analysis")
col1, col2, col3, col4 = st.columns(4)
col1.page_link("pages/13_performance_timeline.py", label="📈 Performance Timeline", icon="📈")
col2.page_link("pages/14_prediction_drift.py", label="📊 Prediction Drift", icon="📊")
col3.page_link("pages/15_segment_analysis.py", label="🎯 Segment Analysis", icon="🎯")
col4.page_link("pages/16_retrain_center.py", label="🔄 Retrain Center", icon="🔄")
```

**Step 2: Manual verification**

```bash
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Navigate to Monitoring Overview. Upload a predictions CSV. Verify traffic lights, KPIs, alerts, timeline, and retrain recommendation all render.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add model monitoring overview page"
```

---

## Task 10: Streamlit — Performance Timeline Page

**Files:**
- Create: `packages/dashboard/src/dashboard/pages/13_performance_timeline.py`

**Step 1: Implement the page**

```python
# packages/dashboard/src/dashboard/pages/13_performance_timeline.py
"""Performance Timeline — deep-dive into accuracy and F1 over time."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from monitoring.performance import (
    compute_performance_metrics,
    compute_rolling_performance,
)
from nfl.target import INVERSE_LABEL_MAP

st.header("📈 Performance Timeline")

uploaded = st.file_uploader("Upload predictions CSV", type=["csv"])
if not uploaded:
    st.info("Upload predictions with ground truth to analyze performance over time.")
    st.stop()

df = pd.read_csv(uploaded)
if "actual_decision" not in df.columns:
    st.error("CSV must contain 'actual_decision' column (ground truth).")
    st.stop()

st.caption(f"{len(df):,} predictions loaded")

# --- Controls ---
col1, col2 = st.columns(2)
window = col1.slider("Rolling window (days)", 3, 30, 7)
baseline_acc = col2.number_input("Baseline accuracy", 0.0, 1.0, 0.78)

# --- Overall Metrics ---
overall = compute_performance_metrics(df["prediction"], df["actual_decision"])
st.subheader("Current Period Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{overall.accuracy:.1%}")
m2.metric("F1 (macro)", f"{overall.f1_macro:.1%}")
m3.metric("Samples", f"{overall.n_samples:,}")
if overall.per_class_f1:
    for i, (cls, f1) in enumerate(overall.per_class_f1.items()):
        m4.metric(f"F1 ({cls})", f"{f1:.1%}")

# --- Confusion Matrix ---
st.subheader("Confusion Matrix")
labels = [INVERSE_LABEL_MAP.get(i, str(i)) for i in range(3)]
fig_cm = px.imshow(
    overall.confusion_matrix,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=labels, y=labels,
    text_auto=True,
    color_continuous_scale="Blues",
    title="Confusion Matrix (Current Period)",
)
fig_cm.update_layout(height=400)
st.plotly_chart(fig_cm, use_container_width=True)

# --- Rolling Timeline ---
if "timestamp" in df.columns:
    st.subheader(f"Rolling {window}-Day Performance")
    timeline = compute_rolling_performance(df, window_days=window)

    if len(timeline.snapshots) > 0:
        tl_df = timeline.to_dataframe()

        # Main accuracy + F1 chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["Accuracy & F1 (Macro)", "Sample Count"],
                            row_heights=[0.7, 0.3])

        fig.add_trace(go.Scatter(x=tl_df["date"], y=tl_df["accuracy"],
                                 name="Accuracy", line=dict(color="#3498db", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=tl_df["date"], y=tl_df["f1_macro"],
                                 name="F1 Macro", line=dict(color="#e67e22", width=2)), row=1, col=1)

        fig.add_hline(y=baseline_acc, line_dash="dash", line_color="green",
                      annotation_text="Baseline", row=1, col=1)
        fig.add_hline(y=baseline_acc - 0.05, line_dash="dash", line_color="red",
                      annotation_text="Alert", row=1, col=1)

        fig.add_trace(go.Bar(x=tl_df["date"], y=tl_df["n_samples"],
                             name="Samples", marker_color="#95a5a6"), row=2, col=1)

        fig.update_layout(height=500, showlegend=True)
        fig.update_yaxes(range=[0, 1], row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Per-class F1 timeline
        f1_cols = [c for c in tl_df.columns if c.startswith("f1_")]
        if f1_cols:
            st.subheader("Per-Class F1 Over Time")
            fig_class = go.Figure()
            colors = {"f1_go_for_it": "#2ecc71", "f1_punt": "#3498db", "f1_field_goal": "#e74c3c"}
            for col in f1_cols:
                fig_class.add_trace(go.Scatter(
                    x=tl_df["date"], y=tl_df[col],
                    name=col.replace("f1_", ""),
                    line=dict(color=colors.get(col, "#95a5a6")),
                ))
            fig_class.update_layout(height=350, yaxis_range=[0, 1])
            st.plotly_chart(fig_class, use_container_width=True)
    else:
        st.warning("Not enough data for rolling timeline.")
else:
    st.warning("No 'timestamp' column — cannot compute rolling performance.")
```

**Step 2: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add performance timeline page"
```

---

## Task 11: Streamlit — Prediction Drift Page

**Files:**
- Create: `packages/dashboard/src/dashboard/pages/14_prediction_drift.py`

**Step 1: Implement the page**

```python
# packages/dashboard/src/dashboard/pages/14_prediction_drift.py
"""Prediction Drift — analyze how model outputs are shifting."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from monitoring.prediction_drift import (
    compute_prediction_distribution,
    detect_prediction_drift,
)
from monitoring.psi import compute_feature_psi_report
from nfl.target import INVERSE_LABEL_MAP

st.header("📊 Prediction & Feature Drift")

uploaded = st.file_uploader("Upload predictions CSV", type=["csv"])
if not uploaded:
    st.info("Upload predictions to analyze drift.")
    st.stop()

df = pd.read_csv(uploaded)
st.caption(f"{len(df):,} predictions loaded")

# Split into baseline (first half) and recent (second half)
split_pct = st.slider("Baseline split (%)", 10, 50, 30)
n_baseline = max(int(len(df) * split_pct / 100), 50)

baseline_df = df.iloc[:n_baseline]
recent_df = df.iloc[-n_baseline:]

st.caption(f"Baseline: first {n_baseline} predictions | Recent: last {n_baseline} predictions")

# --- Prediction Distribution ---
st.subheader("Prediction Class Distribution")

baseline_dist = compute_prediction_distribution(baseline_df)
recent_dist = compute_prediction_distribution(recent_df)
drift_result = detect_prediction_drift(baseline_dist, recent_dist)

# Side-by-side class proportions
severity_emoji = {"none": "✅", "warning": "⚠️", "critical": "🚨"}
st.markdown(f"**Drift severity:** {severity_emoji.get(drift_result.severity, '')} {drift_result.severity.upper()} (KL: {drift_result.kl_divergence:.4f})")

class_names = list(baseline_dist.class_proportions.keys())
fig_classes = go.Figure()
fig_classes.add_trace(go.Bar(
    name="Baseline", x=class_names,
    y=[baseline_dist.class_proportions[c] for c in class_names],
    marker_color="rgba(52, 152, 219, 0.7)",
))
fig_classes.add_trace(go.Bar(
    name="Recent", x=class_names,
    y=[recent_dist.class_proportions[c] for c in class_names],
    marker_color="rgba(231, 76, 60, 0.7)",
))
fig_classes.update_layout(barmode="group", title="Class Proportion: Baseline vs Recent",
                          yaxis_title="Proportion", height=350)
st.plotly_chart(fig_classes, use_container_width=True)

# Class shift table
if drift_result.class_shifts:
    shift_df = pd.DataFrame([
        {"Decision": k, "Baseline %": f"{baseline_dist.class_proportions[k]:.1%}",
         "Recent %": f"{recent_dist.class_proportions[k]:.1%}",
         "Shift": f"{v:+.1%}"}
        for k, v in drift_result.class_shifts.items()
    ])
    st.dataframe(shift_df, use_container_width=True)

# --- Confidence Distribution ---
st.subheader("Confidence Distribution")

col1, col2 = st.columns(2)
col1.metric("Baseline Mean Confidence", f"{baseline_dist.mean_confidence:.3f}")
col2.metric("Recent Mean Confidence", f"{recent_dist.mean_confidence:.3f}",
            delta=f"{drift_result.confidence_shift:+.3f}" if drift_result.confidence_shift else None)

if baseline_dist.confidence_histogram and recent_dist.confidence_histogram:
    fig_conf = go.Figure()
    b_centers = [(baseline_dist.confidence_histogram["bin_edges"][i] +
                  baseline_dist.confidence_histogram["bin_edges"][i+1]) / 2
                 for i in range(len(baseline_dist.confidence_histogram["counts"]))]
    r_centers = [(recent_dist.confidence_histogram["bin_edges"][i] +
                  recent_dist.confidence_histogram["bin_edges"][i+1]) / 2
                 for i in range(len(recent_dist.confidence_histogram["counts"]))]

    b_total = sum(baseline_dist.confidence_histogram["counts"]) or 1
    r_total = sum(recent_dist.confidence_histogram["counts"]) or 1

    fig_conf.add_trace(go.Bar(x=b_centers,
                              y=[c/b_total for c in baseline_dist.confidence_histogram["counts"]],
                              name="Baseline", marker_color="rgba(52,152,219,0.5)"))
    fig_conf.add_trace(go.Bar(x=r_centers,
                              y=[c/r_total for c in recent_dist.confidence_histogram["counts"]],
                              name="Recent", marker_color="rgba(231,76,60,0.5)"))
    fig_conf.update_layout(barmode="overlay", title="Confidence Distribution Overlay",
                           xaxis_title="Confidence", yaxis_title="Density", height=300)
    st.plotly_chart(fig_conf, use_container_width=True)

# --- Feature PSI ---
feat_cols = [c for c in df.columns if c.startswith("feat_") and pd.api.types.is_numeric_dtype(df[c])]
if feat_cols:
    st.subheader("Feature Stability (PSI)")
    psi_report = compute_feature_psi_report(baseline_df[feat_cols], recent_df[feat_cols])
    psi_df = psi_report.to_dataframe()

    fig_psi = px.bar(
        psi_df, x="psi", y="feature", orientation="h",
        color="severity",
        color_discrete_map={"none": "#2ecc71", "warning": "#f39c12", "critical": "#e74c3c"},
        title="Population Stability Index by Feature",
    )
    fig_psi.add_vline(x=0.10, line_dash="dash", line_color="orange", annotation_text="Warning")
    fig_psi.add_vline(x=0.25, line_dash="dash", line_color="red", annotation_text="Critical")
    fig_psi.update_layout(yaxis=dict(autorange="reversed"), height=max(250, len(feat_cols) * 30))
    st.plotly_chart(fig_psi, use_container_width=True)

    st.dataframe(psi_df, use_container_width=True)
```

**Step 2: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add prediction drift + PSI page"
```

---

## Task 12: Streamlit — Segment Analysis Page

**Files:**
- Create: `packages/dashboard/src/dashboard/pages/15_segment_analysis.py`

**Step 1: Implement the page**

```python
# packages/dashboard/src/dashboard/pages/15_segment_analysis.py
"""Segment Analysis — find WHERE the model is underperforming."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from monitoring.segments import (
    compute_segmented_performance,
    find_degraded_segments,
)

st.header("🎯 Segment Performance Analysis")
st.markdown(
    "Discover which game situations the model handles well and where it struggles. "
    "Overall accuracy can mask segment-level problems."
)

uploaded = st.file_uploader("Upload predictions CSV", type=["csv"])
if not uploaded:
    st.info("Upload predictions with ground truth and feat_* columns.")
    st.stop()

df = pd.read_csv(uploaded)
if "actual_decision" not in df.columns:
    st.error("CSV must contain 'actual_decision' column.")
    st.stop()

feat_cols = [c for c in df.columns if c.startswith("feat_")]
if not feat_cols:
    st.error("No feat_* columns found for segmentation.")
    st.stop()

# --- Select Segments ---
selected_segments = st.multiselect(
    "Segment by", feat_cols, default=feat_cols[:4]
)

if not selected_segments:
    st.stop()

# --- Compute ---
with st.spinner("Computing segmented performance..."):
    report = compute_segmented_performance(df, segment_columns=selected_segments)
    degraded = find_degraded_segments(report)

# --- Overview ---
seg_df = report.to_dataframe()

if degraded:
    st.error(f"🚨 {len(degraded)} underperforming segment(s) detected")
    deg_df = pd.DataFrame(degraded)
    st.dataframe(deg_df, use_container_width=True)
else:
    st.success("✅ All segments performing above threshold")

# --- Per-Segment Charts ---
for seg_col in selected_segments:
    if seg_col not in report.segments:
        continue

    st.subheader(f"Performance by {seg_col}")

    seg_data = report.segments[seg_col]
    chart_data = []
    for val, snap in sorted(seg_data.items(), key=lambda x: x[0]):
        chart_data.append({
            "Segment": str(val),
            "Accuracy": snap.accuracy,
            "F1 (macro)": snap.f1_macro,
            "Samples": snap.n_samples,
        })

    chart_df = pd.DataFrame(chart_data)

    # Grouped bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart_df["Segment"], y=chart_df["Accuracy"],
                         name="Accuracy", marker_color="#3498db"))
    fig.add_trace(go.Bar(x=chart_df["Segment"], y=chart_df["F1 (macro)"],
                         name="F1", marker_color="#e67e22"))
    fig.add_hline(y=0.65, line_dash="dash", line_color="red",
                  annotation_text="Degraded threshold")
    fig.update_layout(barmode="group", yaxis_range=[0, 1],
                      title=f"Accuracy & F1 by {seg_col}", height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Sample size annotation
    fig_samples = px.bar(
        chart_df, x="Segment", y="Samples",
        title=f"Sample Count by {seg_col}",
        color_discrete_sequence=["#95a5a6"],
    )
    fig_samples.update_layout(height=200)
    st.plotly_chart(fig_samples, use_container_width=True)

# --- Heatmap: Cross-Segment ---
if len(selected_segments) >= 2:
    st.subheader("Cross-Segment Heatmap")
    seg_a, seg_b = selected_segments[0], selected_segments[1]
    st.caption(f"Accuracy heatmap: {seg_a} × {seg_b}")

    # Compute cross-segment accuracy
    cross_data = []
    for a_val in sorted(df[seg_a].dropna().unique()):
        for b_val in sorted(df[seg_b].dropna().unique()):
            mask = (df[seg_a] == a_val) & (df[seg_b] == b_val)
            subset = df[mask]
            if len(subset) >= 10:
                acc = (subset["prediction"] == subset["actual_decision"]).mean()
                cross_data.append({seg_a: str(a_val), seg_b: str(b_val), "accuracy": acc})

    if cross_data:
        cross_df = pd.DataFrame(cross_data)
        pivot = cross_df.pivot(index=seg_a, columns=seg_b, values="accuracy")
        fig_heat = px.imshow(
            pivot, text_auto=".0%",
            color_continuous_scale="RdYlGn",
            zmin=0.4, zmax=1.0,
            title=f"Accuracy: {seg_a} × {seg_b}",
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
```

**Step 2: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add segment analysis page with cross-segment heatmap"
```

---

## Task 13: Streamlit — Retrain Decision Center

**Files:**
- Create: `packages/dashboard/src/dashboard/pages/16_retrain_center.py`

**Step 1: Implement the page**

```python
# packages/dashboard/src/dashboard/pages/16_retrain_center.py
"""Retrain Decision Center — all signals in one place to decide when to retrain."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from monitoring.retrain_advisor import (
    compute_retrain_recommendation,
    RetrainUrgency,
)

st.header("🔄 Retrain Decision Center")
st.markdown(
    "This page aggregates all monitoring signals into a single retrain recommendation. "
    "Adjust the inputs below to see how different scenarios affect the recommendation."
)

# --- Manual Signal Input ---
# In production, these would come from the monitoring pipeline.
# Here we allow manual input for exploration and "what-if" analysis.

st.subheader("Current Monitoring Signals")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Performance**")
    accuracy_delta = st.slider(
        "Accuracy change from baseline",
        -0.20, 0.05, -0.03, 0.01,
        help="Negative = model got worse",
    )
    f1_delta = st.slider(
        "F1 change from baseline",
        -0.25, 0.05, -0.04, 0.01,
    )

with col2:
    st.markdown("**Drift**")
    max_psi = st.slider(
        "Max feature PSI",
        0.0, 0.60, 0.08, 0.01,
        help="0.10 = warning, 0.25 = critical",
    )
    prediction_kl = st.slider(
        "Prediction KL divergence",
        0.0, 0.40, 0.03, 0.01,
        help="0.05 = warning, 0.15 = critical",
    )

with col3:
    st.markdown("**Context**")
    n_degraded = st.number_input("Degraded segments", 0, 20, 0)
    days_since = st.number_input("Days since training", 0, 365, 45)

# --- Compute Recommendation ---
rec = compute_retrain_recommendation(
    accuracy_delta=accuracy_delta,
    f1_delta=f1_delta,
    max_psi=max_psi,
    prediction_kl=prediction_kl,
    n_degraded_segments=n_degraded,
    days_since_training=days_since,
)

# --- Display ---
st.markdown("---")

urgency_config = {
    RetrainUrgency.NONE: ("✅", "success", "No retrain needed"),
    RetrainUrgency.RECOMMENDED: ("⚠️", "warning", "Retrain recommended"),
    RetrainUrgency.URGENT: ("🚨", "error", "Urgent retrain needed"),
}
icon, method, label = urgency_config[rec.urgency]

getattr(st, method)(f"## {icon} {label}")

# Score gauge
st.subheader(f"Retrain Score: {rec.score}/100")
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=rec.score,
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "#333"},
        "steps": [
            {"range": [0, 30], "color": "#2ecc71"},
            {"range": [30, 60], "color": "#f39c12"},
            {"range": [60, 100], "color": "#e74c3c"},
        ],
        "threshold": {
            "line": {"color": "black", "width": 3},
            "thickness": 0.8,
            "value": rec.score,
        },
    },
    title={"text": "Retrain Urgency Score"},
))
fig_gauge.update_layout(height=250)
st.plotly_chart(fig_gauge, use_container_width=True)

# Signal breakdown
st.subheader("Signal Breakdown")
fig_signals = go.Figure()
signal_names = list(rec.signal_scores.keys())
signal_vals = list(rec.signal_scores.values())
max_per_signal = [25, 20, 20, 15, 10, 10]  # Max possible per signal

colors = ["#e74c3c" if v > m * 0.6 else "#f39c12" if v > m * 0.3 else "#2ecc71"
          for v, m in zip(signal_vals, max_per_signal)]

fig_signals.add_trace(go.Bar(
    y=signal_names, x=signal_vals, orientation="h",
    marker_color=colors, text=[str(v) for v in signal_vals], textposition="outside",
))
fig_signals.add_trace(go.Bar(
    y=signal_names, x=[m - v for v, m in zip(signal_vals, max_per_signal)],
    orientation="h", marker_color="rgba(200,200,200,0.3)",
    showlegend=False,
))
fig_signals.update_layout(
    barmode="stack", height=250, xaxis_title="Points",
    title="Points by Signal (higher = more urgent)",
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig_signals, use_container_width=True)

# Reasons
st.subheader("Reasons")
for reason in rec.reasons:
    st.markdown(f"- {reason}")

# --- What-If Scenarios ---
st.markdown("---")
with st.expander("📋 Retrain Checklist"):
    st.markdown("""
    If retraining is recommended, follow this checklist:

    1. **Verify data availability** — Is there enough new labeled data since the last training?
    2. **Check data quality** — Run the Data Quality dashboard on the new data.
    3. **Choose training window** — Include recent data that reflects the current distribution.
    4. **Run training script:**
       ```bash
       uv run python scripts/train_fourth_down_model.py \\
           --train-seasons 2020 2021 2022 2023 2024 \\
           --test-seasons 2025 \\
           --output-dir models/v2 \\
           --cache-data
       ```
    5. **Compare models** — Evaluate new model against current model on the same test set.
    6. **Deploy if improved** — Update MODEL_DIR and restart the API.
    7. **Update baseline** — New model becomes the monitoring baseline.
    """)

with st.expander("📊 Historical recommendation (coming soon)"):
    st.info(
        "In a full deployment, this page would show retrain score over time, "
        "tracking how urgency has changed and when retrains were performed."
    )
```

**Step 2: Manual verification**

```bash
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Navigate to Retrain Center. Adjust sliders. Verify gauge and signal breakdown update.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add retrain decision center page"
```

---

## Summary: Run Order

| Task | What it builds | Key files |
|------|---------------|-----------|
| 1 | Monitoring scaffolding + config | `monitoring/config.py` |
| 2 | Prediction logger | `monitoring/prediction_log.py` |
| 3 | Rolling performance tracker | `monitoring/performance.py` |
| 4 | Prediction distribution drift | `monitoring/prediction_drift.py` |
| 5 | Population Stability Index | `monitoring/psi.py` |
| 6 | Segmented performance analysis | `monitoring/segments.py` |
| 7 | Retrain recommendation engine | `monitoring/retrain_advisor.py` |
| 8 | Traffic-light alert system | `monitoring/alerts.py` |
| 9 | Streamlit: Monitoring Overview | `pages/12_monitoring_overview.py` |
| 10 | Streamlit: Performance Timeline | `pages/13_performance_timeline.py` |
| 11 | Streamlit: Prediction Drift + PSI | `pages/14_prediction_drift.py` |
| 12 | Streamlit: Segment Analysis | `pages/15_segment_analysis.py` |
| 13 | Streamlit: Retrain Decision Center | `pages/16_retrain_center.py` |

Tasks 1-8 are pure backend (fully testable). Tasks 9-13 are Streamlit pages.

## Full Dashboard Map (All Plans)

After implementing all 6 plans, the Streamlit dashboard has 16 pages:

| # | Page | Plan |
|---|------|------|
| 01 | Data Explorer | Plan 1 (ETL Framework) |
| 02 | Feature Preview | Plan 1 |
| 03 | Inference Form | Plan 1 |
| 04 | 4th Down Calculator | Plan 2 (NFL ML) |
| 05 | Feature Importance | Plan 3 (Feature Selection) |
| 06 | Feature Distributions | Plan 3 |
| 07 | Feature Selection Lab | Plan 3 |
| 08 | Data Quality Overview | Plan 4 (Data Quality) |
| 09 | Drift Monitor | Plan 4 |
| 10 | Validation Rules | Plan 4 |
| 11 | Settings (WebGPU) | Plan 6 (WebGPU LLM) |
| 12 | Monitoring Overview | Plan 7 (Model Monitoring) |
| 13 | Performance Timeline | Plan 7 |
| 14 | Prediction Drift + PSI | Plan 7 |
| 15 | Segment Analysis | Plan 7 |
| 16 | Retrain Center | Plan 7 |

Total tasks across all plans: **72 tasks** (Plans 1-8).

> **Note:** Plan 8 (Docker Containerization) does not add dashboard pages but containerizes all services above into Docker images with docker-compose orchestration.
