# Feature Selection & Importance Explorer — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a feature analysis library and Streamlit UI that lets users explore feature importance (Gini, permutation, SHAP), visualize feature distributions, detect multicollinearity, and interactively select/deselect features — then retrain the model with the selected subset and compare evaluation metrics.

**Architecture:** A `feature_analysis` module in the `ml` package computes importance scores, correlation matrices, and distribution statistics — all as pure functions returning DataFrames so they're testable and UI-agnostic. Streamlit pages consume these functions and render interactive Plotly charts. A feature selection "session" lets users toggle features on/off, retrain in-browser, and compare before/after metrics side-by-side. All analysis runs on DuckDB via Ibis for distribution stats, and on materialized Pandas DataFrames for sklearn/SHAP computations.

**Tech Stack:**
- **Analysis:** scikit-learn (permutation importance), XGBoost (built-in Gini importance), SHAP
- **Visualization:** Plotly, Streamlit
- **Data:** Ibis + DuckDB (distribution stats), Pandas (importance computation)
- **Inherits:** Everything from unified-etl + NFL ML pipeline plans

**Prerequisite:** The unified-etl framework (Plan 1) and NFL ML pipeline (Plan 2) must be implemented first.

---

## Extended Project Structure

```
unified-etl/
├── packages/
│   ├── ml/
│   │   └── src/
│   │       └── ml/
│   │           ├── ... (existing)
│   │           └── feature_analysis/
│   │               ├── __init__.py
│   │               ├── importance.py       # Gini, permutation, SHAP importance
│   │               ├── correlation.py      # Correlation matrix + VIF
│   │               ├── distributions.py    # Feature distribution stats via Ibis
│   │               └── selection.py        # Feature subset evaluation
│   │
│   └── dashboard/
│       └── src/
│           └── dashboard/
│               └── pages/
│                   ├── ... (existing)
│                   ├── 05_feature_importance.py
│                   ├── 06_feature_distributions.py
│                   └── 07_feature_selection_lab.py
│
└── tests/
    └── ml/
        ├── ... (existing)
        ├── test_importance.py
        ├── test_correlation.py
        ├── test_distributions.py
        └── test_selection.py
```

---

## Task 1: Feature Importance Computation (Gini + Permutation)

**Files:**
- Create: `packages/ml/src/ml/feature_analysis/__init__.py`
- Create: `packages/ml/src/ml/feature_analysis/importance.py`
- Test: `tests/ml/test_importance.py`

**Step 1: Write the failing test**

```python
# tests/ml/test_importance.py
import numpy as np
import pandas as pd
import pytest

from ml.feature_analysis.importance import (
    compute_gini_importance,
    compute_permutation_importance,
    compute_shap_importance,
    ImportanceResult,
)
from ml.model import FourthDownModel
from nfl.features import MODEL_FEATURE_COLUMNS


@pytest.fixture
def fitted_model_and_data():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({col: np.random.randn(n) for col in MODEL_FEATURE_COLUMNS})
    # Make ydstogo and yardline_100 the most informative features
    signal = X["ydstogo"] * 3 + X["yardline_100"] * 2
    y = pd.Series(np.select([signal < -2, signal > 2], [0, 1], default=2))
    model = FourthDownModel()
    model.fit(X, y)
    return model, X, y


def test_importance_result_structure():
    result = ImportanceResult(
        method="gini",
        feature_names=["a", "b"],
        scores=np.array([0.6, 0.4]),
    )
    assert result.method == "gini"
    assert len(result.feature_names) == 2
    assert len(result.scores) == 2


def test_importance_result_to_dataframe():
    result = ImportanceResult(
        method="gini",
        feature_names=["a", "b"],
        scores=np.array([0.6, 0.4]),
    )
    df = result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["feature", "importance"]
    assert df.iloc[0]["feature"] == "a"  # sorted descending
    assert df.iloc[0]["importance"] == 0.6


def test_gini_importance(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    result = compute_gini_importance(model, X)

    assert isinstance(result, ImportanceResult)
    assert result.method == "gini"
    assert len(result.scores) == len(MODEL_FEATURE_COLUMNS)
    # Scores should sum to ~1.0 for tree-based models
    assert abs(result.scores.sum() - 1.0) < 0.01
    # The signal features should rank highly
    df = result.to_dataframe()
    top_5 = list(df.head(5)["feature"])
    assert "ydstogo" in top_5 or "yardline_100" in top_5


def test_permutation_importance(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    result = compute_permutation_importance(model, X, y, n_repeats=5)

    assert isinstance(result, ImportanceResult)
    assert result.method == "permutation"
    assert len(result.scores) == len(MODEL_FEATURE_COLUMNS)
    assert result.std is not None
    assert len(result.std) == len(MODEL_FEATURE_COLUMNS)


def test_shap_importance(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    result = compute_shap_importance(model, X.iloc[:100])

    assert isinstance(result, ImportanceResult)
    assert result.method == "shap"
    assert len(result.scores) == len(MODEL_FEATURE_COLUMNS)
    # SHAP values should have the raw matrix available
    assert result.shap_values is not None
    assert result.shap_values.shape[1] == len(MODEL_FEATURE_COLUMNS)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/ml/test_importance.py -v
```

Expected: FAIL.

**Step 3: Implement importance computations**

```python
# packages/ml/src/ml/feature_analysis/__init__.py
from ml.feature_analysis.importance import (
    compute_gini_importance,
    compute_permutation_importance,
    compute_shap_importance,
    ImportanceResult,
)
from ml.feature_analysis.correlation import (
    compute_correlation_matrix,
    compute_vif,
    CorrelationResult,
)
from ml.feature_analysis.distributions import compute_feature_distributions
from ml.feature_analysis.selection import evaluate_feature_subset

__all__ = [
    "compute_gini_importance",
    "compute_permutation_importance",
    "compute_shap_importance",
    "ImportanceResult",
    "compute_correlation_matrix",
    "compute_vif",
    "CorrelationResult",
    "compute_feature_distributions",
    "evaluate_feature_subset",
]
```

```python
# packages/ml/src/ml/feature_analysis/importance.py
"""Feature importance computation via multiple methods.

Three complementary approaches:
- Gini importance: fast, built into tree models, but biased toward
  high-cardinality and correlated features.
- Permutation importance: model-agnostic, slower, more reliable.
  Measures actual accuracy drop when a feature is shuffled.
- SHAP: gold standard, explains individual predictions. Slowest
  but most informative — produces per-sample contribution values.

Each method returns an ImportanceResult for consistent downstream use.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance as sklearn_perm_importance

from ml.model import FourthDownModel


@dataclass
class ImportanceResult:
    """Container for feature importance scores from any method."""

    method: str
    feature_names: list[str]
    scores: np.ndarray
    std: np.ndarray | None = None
    shap_values: np.ndarray | None = None  # Only populated by SHAP

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame sorted by importance descending."""
        df = pd.DataFrame(
            {"feature": self.feature_names, "importance": self.scores}
        )
        if self.std is not None:
            df["std"] = self.std
        return df.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_gini_importance(
    model: FourthDownModel, X: pd.DataFrame
) -> ImportanceResult:
    """Extract Gini (split-based) importance from the trained XGBoost model.

    Fast — no data reprocessing needed. But beware: Gini importance
    is biased toward features with more unique values and doesn't
    account for feature correlations.
    """
    importances = model.feature_importances()
    features = list(importances.keys())
    scores = np.array(list(importances.values()))

    return ImportanceResult(
        method="gini",
        feature_names=features,
        scores=scores,
    )


def compute_permutation_importance(
    model: FourthDownModel,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> ImportanceResult:
    """Compute permutation importance by shuffling each feature.

    For each feature, shuffles its values n_repeats times and measures
    how much the model's accuracy drops. More reliable than Gini but
    slower (requires n_features × n_repeats predictions).
    """
    result = sklearn_perm_importance(
        model._model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="accuracy",
        n_jobs=-1,
    )

    return ImportanceResult(
        method="permutation",
        feature_names=list(X.columns),
        scores=result.importances_mean,
        std=result.importances_std,
    )


def compute_shap_importance(
    model: FourthDownModel,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> ImportanceResult:
    """Compute SHAP values for feature importance.

    Uses TreeExplainer (exact and fast for tree models).
    Returns mean absolute SHAP value per feature as the importance
    score, plus the raw SHAP value matrix for beeswarm/waterfall plots.

    For multiclass, SHAP returns shape (n_samples, n_features, n_classes).
    We take the mean absolute value across samples and classes.
    """
    import shap

    # Subsample if needed (SHAP on large datasets is slow)
    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=42)

    explainer = shap.TreeExplainer(model._model)
    shap_values = explainer.shap_values(X)

    # shap_values is a list of arrays for multiclass, one per class
    # Each element: (n_samples, n_features)
    if isinstance(shap_values, list):
        # Stack to (n_samples, n_features, n_classes) and take mean abs
        stacked = np.stack(shap_values, axis=-1)
        mean_abs = np.mean(np.abs(stacked), axis=(0, 2))
        # For the raw matrix, use the "all classes" view
        raw_matrix = np.mean(np.abs(stacked), axis=2)  # (n_samples, n_features)
    else:
        # Binary or single output
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        raw_matrix = shap_values

    return ImportanceResult(
        method="shap",
        feature_names=list(X.columns),
        scores=mean_abs,
        shap_values=raw_matrix,
    )
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/ml/test_importance.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(ml): add Gini, permutation, and SHAP importance computation"
```

---

## Task 2: Correlation Matrix + Variance Inflation Factor

**Files:**
- Create: `packages/ml/src/ml/feature_analysis/correlation.py`
- Test: `tests/ml/test_correlation.py`

**Step 1: Write the failing test**

```python
# tests/ml/test_correlation.py
import numpy as np
import pandas as pd
import pytest

from ml.feature_analysis.correlation import (
    compute_correlation_matrix,
    compute_vif,
    find_highly_correlated_pairs,
    CorrelationResult,
)


@pytest.fixture
def feature_data():
    """Features with known correlation structure."""
    np.random.seed(42)
    n = 200
    a = np.random.randn(n)
    b = a * 0.95 + np.random.randn(n) * 0.1  # Highly correlated with a
    c = np.random.randn(n)  # Independent
    d = c + np.random.randn(n) * 0.5  # Moderately correlated with c
    e = np.random.randn(n)  # Independent
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})


def test_correlation_matrix_shape(feature_data):
    result = compute_correlation_matrix(feature_data)
    assert isinstance(result, CorrelationResult)
    assert result.matrix.shape == (5, 5)
    assert list(result.feature_names) == ["a", "b", "c", "d", "e"]


def test_correlation_matrix_diagonal_is_one(feature_data):
    result = compute_correlation_matrix(feature_data)
    np.testing.assert_allclose(np.diag(result.matrix), 1.0, atol=1e-10)


def test_correlation_matrix_symmetric(feature_data):
    result = compute_correlation_matrix(feature_data)
    np.testing.assert_allclose(result.matrix, result.matrix.T, atol=1e-10)


def test_highly_correlated_pairs(feature_data):
    result = compute_correlation_matrix(feature_data)
    pairs = find_highly_correlated_pairs(result, threshold=0.8)
    # a and b should be highly correlated
    pair_sets = [frozenset(p[:2]) for p in pairs]
    assert frozenset(("a", "b")) in pair_sets
    # c and d should NOT be above 0.8
    assert frozenset(("c", "d")) not in pair_sets


def test_vif_computation(feature_data):
    vif = compute_vif(feature_data)
    assert isinstance(vif, pd.DataFrame)
    assert list(vif.columns) == ["feature", "vif"]
    assert len(vif) == 5
    # a and b are collinear, so their VIFs should be high
    a_vif = vif[vif["feature"] == "a"]["vif"].iloc[0]
    e_vif = vif[vif["feature"] == "e"]["vif"].iloc[0]
    assert a_vif > e_vif  # Collinear feature has higher VIF


def test_correlation_to_dataframe(feature_data):
    result = compute_correlation_matrix(feature_data)
    df = result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5, 5)
    assert list(df.index) == list(df.columns)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/ml/test_correlation.py -v
```

Expected: FAIL.

**Step 3: Implement correlation analysis**

```python
# packages/ml/src/ml/feature_analysis/correlation.py
"""Correlation analysis and multicollinearity detection.

Multicollinearity is a common problem in feature engineering:
if two features are highly correlated, the model learns redundant
information and feature importance scores become unreliable
(importance gets split between the correlated pair).

Two tools:
- Correlation matrix: pairwise Pearson correlation. Quick visual.
- VIF (Variance Inflation Factor): detects multicollinearity that
  involves 3+ features (which pairwise correlation misses).
  VIF > 5 is a warning; VIF > 10 means serious collinearity.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CorrelationResult:
    """Container for a correlation matrix and feature names."""

    feature_names: list[str]
    matrix: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Return as a labeled DataFrame for display/plotting."""
        return pd.DataFrame(
            self.matrix,
            index=self.feature_names,
            columns=self.feature_names,
        )


def compute_correlation_matrix(
    X: pd.DataFrame, method: str = "pearson"
) -> CorrelationResult:
    """Compute pairwise correlation matrix.

    Args:
        X: Feature DataFrame.
        method: 'pearson', 'spearman', or 'kendall'.

    Returns:
        CorrelationResult with the matrix and feature names.
    """
    corr = X.corr(method=method)
    return CorrelationResult(
        feature_names=list(corr.columns),
        matrix=corr.to_numpy(),
    )


def find_highly_correlated_pairs(
    result: CorrelationResult,
    threshold: float = 0.8,
) -> list[tuple[str, str, float]]:
    """Find all feature pairs with |correlation| >= threshold.

    Returns:
        List of (feature_a, feature_b, correlation) tuples,
        sorted by absolute correlation descending.
    """
    pairs = []
    n = len(result.feature_names)
    for i in range(n):
        for j in range(i + 1, n):
            corr_val = result.matrix[i, j]
            if abs(corr_val) >= threshold:
                pairs.append(
                    (result.feature_names[i], result.feature_names[j], corr_val)
                )
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each feature.

    VIF measures how much the variance of a regression coefficient
    increases due to collinearity. Computed as 1 / (1 - R²) where
    R² is from regressing each feature on all others.

    Interpretation:
        VIF = 1: no collinearity
        VIF 1-5: low collinearity
        VIF 5-10: moderate (investigate)
        VIF > 10: high (consider removing)
    """
    from numpy.linalg import LinAlgError

    vif_data = []
    X_arr = X.to_numpy(dtype=float)
    n_features = X_arr.shape[1]

    for i in range(n_features):
        y_col = X_arr[:, i]
        X_others = np.delete(X_arr, i, axis=1)

        # Add intercept
        X_design = np.column_stack([np.ones(len(y_col)), X_others])

        try:
            # OLS: beta = (X'X)^-1 X'y
            beta = np.linalg.lstsq(X_design, y_col, rcond=None)[0]
            y_pred = X_design @ beta
            ss_res = np.sum((y_col - y_pred) ** 2)
            ss_tot = np.sum((y_col - y_col.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            vif = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else float("inf")
        except LinAlgError:
            vif = float("inf")

        vif_data.append({"feature": X.columns[i], "vif": round(vif, 2)})

    return pd.DataFrame(vif_data).sort_values("vif", ascending=False).reset_index(drop=True)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/ml/test_correlation.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(ml): add correlation matrix + VIF computation"
```

---

## Task 3: Feature Distribution Statistics via Ibis

**Files:**
- Create: `packages/ml/src/ml/feature_analysis/distributions.py`
- Test: `tests/ml/test_distributions.py`

**Step 1: Write the failing test**

```python
# tests/ml/test_distributions.py
import ibis
import numpy as np
import pandas as pd
import pytest

from ml.feature_analysis.distributions import (
    compute_feature_distributions,
    compute_target_split_distributions,
    DistributionStats,
)


@pytest.fixture
def con():
    return ibis.duckdb.connect()


@pytest.fixture
def feature_table():
    np.random.seed(42)
    return ibis.memtable(
        {
            "value_a": np.random.normal(10, 3, 200).tolist(),
            "value_b": np.random.exponential(2, 200).tolist(),
            "category": np.random.choice(["X", "Y", "Z"], 200).tolist(),
            "target": np.random.choice([0, 1, 2], 200).tolist(),
        }
    )


def test_distribution_stats_numeric(con, feature_table):
    stats = compute_feature_distributions(
        con, feature_table, columns=["value_a", "value_b"]
    )
    assert len(stats) == 2
    assert "value_a" in stats
    assert "value_b" in stats

    s = stats["value_a"]
    assert isinstance(s, DistributionStats)
    assert s.count > 0
    assert s.mean is not None
    assert s.std is not None
    assert s.min is not None
    assert s.max is not None
    assert s.median is not None
    assert s.p25 is not None
    assert s.p75 is not None
    assert s.null_count == 0
    assert s.null_pct == 0.0


def test_distribution_stats_with_nulls(con):
    table = ibis.memtable(
        {"x": [1.0, 2.0, None, 4.0, None]}
    )
    stats = compute_feature_distributions(con, table, columns=["x"])
    assert stats["x"].null_count == 2
    assert abs(stats["x"].null_pct - 40.0) < 0.1


def test_distribution_histogram_bins(con, feature_table):
    stats = compute_feature_distributions(
        con, feature_table, columns=["value_a"], n_bins=10
    )
    s = stats["value_a"]
    assert s.histogram is not None
    assert len(s.histogram["bin_edges"]) > 0
    assert len(s.histogram["counts"]) > 0


def test_target_split_distributions(con, feature_table):
    splits = compute_target_split_distributions(
        con, feature_table, feature_col="value_a", target_col="target"
    )
    # Should have one entry per target class
    assert len(splits) == 3
    assert 0 in splits
    assert 1 in splits
    assert 2 in splits
    for label, s in splits.items():
        assert s.count > 0
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/ml/test_distributions.py -v
```

Expected: FAIL.

**Step 3: Implement distribution statistics**

```python
# packages/ml/src/ml/feature_analysis/distributions.py
"""Feature distribution statistics computed via Ibis.

Uses Ibis expressions so that distribution stats push down to
Snowflake at training time and run on DuckDB locally. This avoids
pulling entire columns into memory for basic statistics.

For histograms, we materialize the column (since binning is hard
to express in pure Ibis SQL). This is fine for the dashboard use
case where you're looking at one feature at a time.
"""

from dataclasses import dataclass, field
from typing import Any

import ibis
import ibis.expr.types as ir
import numpy as np
import pandas as pd


@dataclass
class DistributionStats:
    """Statistics for a single feature column."""

    column: str
    count: int
    null_count: int
    null_pct: float
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    p25: float | None = None
    p75: float | None = None
    skew: float | None = None
    unique_count: int | None = None
    histogram: dict[str, list] | None = None


def compute_feature_distributions(
    con: ibis.BaseBackend,
    table: ir.Table,
    columns: list[str],
    n_bins: int = 30,
) -> dict[str, DistributionStats]:
    """Compute distribution statistics for specified columns.

    Pushes aggregation down to the backend via Ibis, then computes
    histograms locally from materialized data.
    """
    results = {}

    for col_name in columns:
        col = table[col_name]

        # Compute aggregates via Ibis (pushes down to SQL)
        agg_expr = table.aggregate(
            count=table.count(),
            null_count=col.isnull().sum(),
            mean=col.mean(),
            std=col.std(),
            min_val=col.min(),
            max_val=col.max(),
            median=col.approx_median(),
        )
        agg = con.execute(agg_expr).iloc[0]

        total = int(agg["count"])
        nulls = int(agg["null_count"])

        stats = DistributionStats(
            column=col_name,
            count=total,
            null_count=nulls,
            null_pct=round(100.0 * nulls / total, 2) if total > 0 else 0.0,
            mean=_safe_float(agg["mean"]),
            std=_safe_float(agg["std"]),
            min=_safe_float(agg["min_val"]),
            max=_safe_float(agg["max_val"]),
            median=_safe_float(agg["median"]),
        )

        # Percentiles + histogram require materialized data
        values = con.execute(table.select(col_name).filter(col.notnull()))
        arr = values[col_name].to_numpy()

        if len(arr) > 0 and np.issubdtype(arr.dtype, np.number):
            stats.p25 = float(np.percentile(arr, 25))
            stats.p75 = float(np.percentile(arr, 75))
            stats.skew = float(pd.Series(arr).skew())

            # Histogram
            counts, bin_edges = np.histogram(arr, bins=n_bins)
            stats.histogram = {
                "counts": counts.tolist(),
                "bin_edges": bin_edges.tolist(),
            }
        else:
            stats.unique_count = int(values[col_name].nunique())

        results[col_name] = stats

    return results


def compute_target_split_distributions(
    con: ibis.BaseBackend,
    table: ir.Table,
    feature_col: str,
    target_col: str,
) -> dict[int, DistributionStats]:
    """Compute feature distributions split by target class.

    Useful for visualizing how a feature's distribution differs
    across classes — a feature with high class separation is
    likely to be informative.
    """
    # Get unique target values
    targets = con.execute(
        table.select(target_col).distinct().order_by(target_col)
    )[target_col].tolist()

    results = {}
    for target_val in targets:
        filtered = table.filter(table[target_col] == target_val)
        stats = compute_feature_distributions(con, filtered, columns=[feature_col])
        results[target_val] = stats[feature_col]

    return results


def _safe_float(val: Any) -> float | None:
    """Safely convert a possibly-null aggregate to float."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return float(val)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/ml/test_distributions.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(ml): add Ibis-based feature distribution statistics"
```

---

## Task 4: Feature Subset Evaluation

**Files:**
- Create: `packages/ml/src/ml/feature_analysis/selection.py`
- Test: `tests/ml/test_selection.py`

**Step 1: Write the failing test**

```python
# tests/ml/test_selection.py
import numpy as np
import pandas as pd
import pytest

from ml.feature_analysis.selection import (
    evaluate_feature_subset,
    compare_feature_subsets,
    SubsetEvaluation,
)
from nfl.features import MODEL_FEATURE_COLUMNS


@pytest.fixture
def training_data():
    np.random.seed(42)
    n = 400
    X = pd.DataFrame({col: np.random.randn(n) for col in MODEL_FEATURE_COLUMNS})
    signal = X["ydstogo"] * 3 + X["yardline_100"] * 2
    y = pd.Series(np.select([signal < -2, signal > 2], [0, 1], default=2))
    return X, y


def test_evaluate_full_feature_set(training_data):
    X, y = training_data
    result = evaluate_feature_subset(X, y, features=MODEL_FEATURE_COLUMNS)

    assert isinstance(result, SubsetEvaluation)
    assert result.n_features == len(MODEL_FEATURE_COLUMNS)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.cv_mean <= 1.0
    assert result.cv_std >= 0.0


def test_evaluate_subset(training_data):
    X, y = training_data
    subset = ["ydstogo", "yardline_100", "score_differential"]
    result = evaluate_feature_subset(X, y, features=subset)

    assert result.n_features == 3
    assert result.features == subset


def test_compare_subsets(training_data):
    X, y = training_data
    subsets = {
        "all": MODEL_FEATURE_COLUMNS,
        "top_3": ["ydstogo", "yardline_100", "score_differential"],
        "minimal": ["ydstogo"],
    }
    comparison = compare_feature_subsets(X, y, subsets)

    assert len(comparison) == 3
    assert all(isinstance(r, SubsetEvaluation) for r in comparison.values())
    # All features should generally beat minimal
    assert comparison["all"].cv_mean >= comparison["minimal"].cv_mean - 0.05


def test_compare_subsets_to_dataframe(training_data):
    X, y = training_data
    subsets = {
        "all": MODEL_FEATURE_COLUMNS,
        "top_3": ["ydstogo", "yardline_100", "score_differential"],
    }
    comparison = compare_feature_subsets(X, y, subsets)
    df = SubsetEvaluation.comparison_dataframe(comparison)

    assert isinstance(df, pd.DataFrame)
    assert "subset_name" in df.columns
    assert "accuracy" in df.columns
    assert "cv_mean" in df.columns
    assert len(df) == 2
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/ml/test_selection.py -v
```

Expected: FAIL.

**Step 3: Implement feature subset evaluation**

```python
# packages/ml/src/ml/feature_analysis/selection.py
"""Feature subset evaluation — train model with different feature sets
and compare cross-validated performance.

This is the backend for the "Feature Selection Lab" Streamlit page,
where users toggle features on/off and see how it affects the model.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from ml.model import FourthDownModel


@dataclass
class SubsetEvaluation:
    """Evaluation results for a specific feature subset."""

    features: list[str]
    n_features: int
    accuracy: float
    cv_mean: float
    cv_std: float
    cv_scores: list[float]

    @staticmethod
    def comparison_dataframe(
        evaluations: dict[str, "SubsetEvaluation"],
    ) -> pd.DataFrame:
        """Convert a dict of evaluations to a comparison DataFrame."""
        rows = []
        for name, ev in evaluations.items():
            rows.append(
                {
                    "subset_name": name,
                    "n_features": ev.n_features,
                    "accuracy": round(ev.accuracy, 4),
                    "cv_mean": round(ev.cv_mean, 4),
                    "cv_std": round(ev.cv_std, 4),
                    "features": ", ".join(ev.features),
                }
            )
        return pd.DataFrame(rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)


def evaluate_feature_subset(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    n_cv_folds: int = 5,
    random_state: int = 42,
) -> SubsetEvaluation:
    """Train and cross-validate a model using only the specified features.

    Uses a fresh XGBoost model with default hyperparams to keep
    comparison fair across subsets.
    """
    X_subset = X[features]

    model = FourthDownModel(hyperparams={"n_estimators": 100, "random_state": random_state})

    # Cross-validation
    cv_scores = cross_val_score(
        model._model,
        X_subset,
        y,
        cv=n_cv_folds,
        scoring="accuracy",
        n_jobs=-1,
    )

    # Full-data accuracy for display
    model.fit(X_subset, y)
    full_acc = float((model.predict(X_subset) == y).mean())

    return SubsetEvaluation(
        features=features,
        n_features=len(features),
        accuracy=full_acc,
        cv_mean=float(cv_scores.mean()),
        cv_std=float(cv_scores.std()),
        cv_scores=cv_scores.tolist(),
    )


def compare_feature_subsets(
    X: pd.DataFrame,
    y: pd.Series,
    subsets: dict[str, list[str]],
    n_cv_folds: int = 5,
) -> dict[str, SubsetEvaluation]:
    """Evaluate multiple feature subsets and return results for comparison."""
    return {
        name: evaluate_feature_subset(X, y, features, n_cv_folds)
        for name, features in subsets.items()
    }
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/ml/test_selection.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(ml): add feature subset evaluation + comparison"
```

---

## Task 5: Streamlit — Feature Importance Page

**Files:**
- Create: `packages/dashboard/src/dashboard/pages/05_feature_importance.py`

**Step 1: Implement the page**

```python
# packages/dashboard/src/dashboard/pages/05_feature_importance.py
"""Feature Importance Explorer — visualize why each feature matters."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml.model import FourthDownModel
from ml.serialize import load_model
from ml.dataset import build_training_dataset, split_features_target, train_test_split_by_season
from ml.feature_analysis.importance import (
    compute_gini_importance,
    compute_permutation_importance,
    compute_shap_importance,
)
from nfl.features import MODEL_FEATURE_COLUMNS

st.header("📊 Feature Importance Explorer")

# --- Load Model + Data ---
model_dir = st.sidebar.text_input("Model directory", value="models/latest")
data_path = st.sidebar.text_input("Training data (Parquet)", value="data/pbp/")

if not Path(model_dir).exists():
    st.warning(
        f"Model not found at `{model_dir}`. "
        "Train a model first with `scripts/train_fourth_down_model.py`."
    )
    st.stop()

model = load_model(model_dir)

# Load metadata for context
metadata_path = Path(model_dir) / "metadata.json"
if metadata_path.exists():
    with open(metadata_path) as f:
        metadata = json.load(f)
    st.sidebar.json(metadata.get("evaluation", {}).get("feature_importances", {}))

st.info(
    "This page requires training data to compute permutation and SHAP importance. "
    "If you only have the model, Gini importance is still available."
)

# Try to load training data
has_data = False
X_sample = None
y_sample = None

if Path(data_path).exists():
    try:
        import ibis

        parquet_files = sorted(Path(data_path).glob("*.parquet"))
        if parquet_files:
            dfs = [pd.read_parquet(p) for p in parquet_files]
            raw = pd.concat(dfs, ignore_index=True)
            con = ibis.duckdb.connect()
            dataset = con.execute(build_training_dataset(ibis.memtable(raw)))
            X_full, y_full = split_features_target(dataset)
            # Subsample for speed
            n_sample = min(2000, len(X_full))
            idx = np.random.RandomState(42).choice(len(X_full), n_sample, replace=False)
            X_sample = X_full.iloc[idx].reset_index(drop=True)
            y_sample = y_full.iloc[idx].reset_index(drop=True)
            has_data = True
            st.success(f"Loaded {len(X_full):,} plays, using {n_sample:,} sample")
    except Exception as e:
        st.warning(f"Could not load training data: {e}")

# --- Importance Methods ---
st.subheader("Importance Method")

methods = ["Gini (built-in)"]
if has_data:
    methods.extend(["Permutation", "SHAP"])

method = st.radio("Select method", methods, horizontal=True)

with st.spinner("Computing importance..."):
    if method == "Gini (built-in)":
        result = compute_gini_importance(model, pd.DataFrame(columns=MODEL_FEATURE_COLUMNS))
    elif method == "Permutation" and has_data:
        n_repeats = st.slider("Permutation repeats", 3, 20, 10)
        result = compute_permutation_importance(model, X_sample, y_sample, n_repeats=n_repeats)
    elif method == "SHAP" and has_data:
        max_shap = st.slider("SHAP sample size", 50, 500, 200)
        result = compute_shap_importance(model, X_sample, max_samples=max_shap)
    else:
        st.stop()

# --- Horizontal Bar Chart ---
st.subheader(f"{method} Importance")

df_imp = result.to_dataframe()

fig = px.bar(
    df_imp,
    x="importance",
    y="feature",
    orientation="h",
    title=f"Feature Importance ({result.method})",
    labels={"importance": "Importance Score", "feature": ""},
    color="importance",
    color_continuous_scale="Viridis",
)
fig.update_layout(
    yaxis=dict(autorange="reversed"),
    showlegend=False,
    height=max(400, len(df_imp) * 28),
)

# Add error bars for permutation importance
if result.std is not None:
    fig.update_traces(
        error_x=dict(type="data", array=result.std, visible=True)
    )

st.plotly_chart(fig, use_container_width=True)

# --- SHAP Beeswarm (if available) ---
if method == "SHAP" and result.shap_values is not None:
    st.subheader("SHAP Value Distribution")
    st.caption(
        "Each dot is one play. Color = feature value (red=high, blue=low). "
        "Position = SHAP value (how much the feature pushed the prediction)."
    )

    # Build a manual beeswarm-style scatter using Plotly
    shap_df = pd.DataFrame(result.shap_values, columns=MODEL_FEATURE_COLUMNS)

    # Show top N features
    top_n = st.slider("Features to show", 5, len(MODEL_FEATURE_COLUMNS), 10)
    top_features = df_imp.head(top_n)["feature"].tolist()

    for feat in reversed(top_features):
        fig_strip = px.strip(
            x=shap_df[feat],
            labels={"x": f"SHAP value for {feat}"},
            title=feat,
        )
        fig_strip.update_layout(height=100, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_strip, use_container_width=True)

# --- Raw Data Table ---
with st.expander("Raw importance scores"):
    st.dataframe(df_imp, use_container_width=True)
```

**Step 2: Manual verification**

```bash
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Navigate to Feature Importance page. Verify Gini chart renders. If training data available, test Permutation and SHAP.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add feature importance explorer page"
```

---

## Task 6: Streamlit — Feature Distributions Page

**Files:**
- Create: `packages/dashboard/src/dashboard/pages/06_feature_distributions.py`

**Step 1: Implement the page**

```python
# packages/dashboard/src/dashboard/pages/06_feature_distributions.py
"""Feature Distributions — histograms, stats, and class-split views."""

from pathlib import Path

import ibis
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ml.dataset import build_training_dataset, split_features_target
from ml.feature_analysis.distributions import (
    compute_feature_distributions,
    compute_target_split_distributions,
)
from ml.feature_analysis.correlation import (
    compute_correlation_matrix,
    compute_vif,
    find_highly_correlated_pairs,
)
from nfl.features import MODEL_FEATURE_COLUMNS
from nfl.target import INVERSE_LABEL_MAP

st.header("📈 Feature Distributions & Correlations")

# --- Load Data ---
data_path = st.sidebar.text_input("Training data (Parquet)", value="data/pbp/")

if not Path(data_path).exists() or not list(Path(data_path).glob("*.parquet")):
    st.warning(
        f"No Parquet files found at `{data_path}`. "
        "Run the training script with `--cache-data` first."
    )
    st.stop()

@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    parquet_files = sorted(Path(path).glob("*.parquet"))
    dfs = [pd.read_parquet(p) for p in parquet_files]
    raw = pd.concat(dfs, ignore_index=True)
    con = ibis.duckdb.connect()
    return con.execute(build_training_dataset(ibis.memtable(raw)))

dataset = load_dataset(data_path)
X, y = split_features_target(dataset)
st.success(f"Loaded {len(dataset):,} 4th-down plays")

# --- Tab Layout ---
tab_hist, tab_class, tab_corr, tab_vif = st.tabs(
    ["Histograms", "Class Split", "Correlation Matrix", "VIF Analysis"]
)

# --- Tab 1: Histograms ---
with tab_hist:
    st.subheader("Feature Histograms")
    selected_feature = st.selectbox("Feature", MODEL_FEATURE_COLUMNS, key="hist_feat")

    con = ibis.duckdb.connect()
    table = ibis.memtable(dataset)
    stats = compute_feature_distributions(con, table, columns=[selected_feature])
    s = stats[selected_feature]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{s.mean:.2f}" if s.mean else "N/A")
    col2.metric("Std Dev", f"{s.std:.2f}" if s.std else "N/A")
    col3.metric("Null %", f"{s.null_pct:.1f}%")
    col4.metric("Skew", f"{s.skew:.2f}" if s.skew else "N/A")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Min", f"{s.min:.2f}" if s.min is not None else "N/A")
    col6.metric("P25", f"{s.p25:.2f}" if s.p25 is not None else "N/A")
    col7.metric("P75", f"{s.p75:.2f}" if s.p75 is not None else "N/A")
    col8.metric("Max", f"{s.max:.2f}" if s.max is not None else "N/A")

    if s.histogram:
        bin_centers = [
            (s.histogram["bin_edges"][i] + s.histogram["bin_edges"][i + 1]) / 2
            for i in range(len(s.histogram["counts"]))
        ]
        fig = px.bar(
            x=bin_centers,
            y=s.histogram["counts"],
            labels={"x": selected_feature, "y": "Count"},
            title=f"Distribution of {selected_feature}",
        )
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Class Split ---
with tab_class:
    st.subheader("Distribution by Decision Class")
    st.caption("See how feature distributions differ across go/punt/FG decisions.")

    class_feature = st.selectbox("Feature", MODEL_FEATURE_COLUMNS, key="class_feat")

    con2 = ibis.duckdb.connect()
    table2 = ibis.memtable(dataset)
    splits = compute_target_split_distributions(
        con2, table2, feature_col=class_feature, target_col="decision_label"
    )

    fig_class = go.Figure()
    colors = {"go_for_it": "#2ecc71", "punt": "#3498db", "field_goal": "#e74c3c"}

    for label_int, dist_stats in sorted(splits.items()):
        label_name = INVERSE_LABEL_MAP.get(label_int, str(label_int))
        if dist_stats.histogram:
            bin_centers = [
                (dist_stats.histogram["bin_edges"][i] + dist_stats.histogram["bin_edges"][i + 1]) / 2
                for i in range(len(dist_stats.histogram["counts"]))
            ]
            fig_class.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=dist_stats.histogram["counts"],
                    name=label_name,
                    marker_color=colors.get(label_name, "#95a5a6"),
                    opacity=0.7,
                )
            )

    fig_class.update_layout(
        barmode="overlay",
        title=f"{class_feature} by Decision Class",
        xaxis_title=class_feature,
        yaxis_title="Count",
    )
    st.plotly_chart(fig_class, use_container_width=True)

    # Show per-class stats table
    rows = []
    for label_int, dist_stats in sorted(splits.items()):
        label_name = INVERSE_LABEL_MAP.get(label_int, str(label_int))
        rows.append({
            "class": label_name,
            "count": dist_stats.count,
            "mean": round(dist_stats.mean, 3) if dist_stats.mean else None,
            "std": round(dist_stats.std, 3) if dist_stats.std else None,
            "median": round(dist_stats.median, 3) if dist_stats.median else None,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# --- Tab 3: Correlation Matrix ---
with tab_corr:
    st.subheader("Feature Correlation Matrix")

    corr_result = compute_correlation_matrix(X)
    corr_df = corr_result.to_dataframe()

    fig_corr = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Matrix",
        aspect="auto",
    )
    fig_corr.update_layout(height=700)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Highlight problematic pairs
    threshold = st.slider("High correlation threshold", 0.5, 0.99, 0.8, 0.05)
    pairs = find_highly_correlated_pairs(corr_result, threshold=threshold)

    if pairs:
        st.warning(f"Found {len(pairs)} highly correlated pair(s) (|r| ≥ {threshold}):")
        pair_df = pd.DataFrame(pairs, columns=["Feature A", "Feature B", "Correlation"])
        pair_df["Correlation"] = pair_df["Correlation"].round(4)
        st.dataframe(pair_df, use_container_width=True)
    else:
        st.success(f"No feature pairs with |correlation| ≥ {threshold}")

# --- Tab 4: VIF ---
with tab_vif:
    st.subheader("Variance Inflation Factor (VIF)")
    st.caption(
        "VIF measures multicollinearity. VIF > 5 = investigate, VIF > 10 = serious issue. "
        "Unlike correlation, VIF detects collinearity involving 3+ features."
    )

    with st.spinner("Computing VIF (this fits a regression per feature)..."):
        vif_df = compute_vif(X)

    fig_vif = px.bar(
        vif_df,
        x="vif",
        y="feature",
        orientation="h",
        title="Variance Inflation Factors",
        color="vif",
        color_continuous_scale=["green", "yellow", "red"],
        range_color=[1, max(10, vif_df["vif"].max())],
    )
    fig_vif.update_layout(yaxis=dict(autorange="reversed"), height=500)
    fig_vif.add_vline(x=5, line_dash="dash", line_color="orange", annotation_text="Warning")
    fig_vif.add_vline(x=10, line_dash="dash", line_color="red", annotation_text="Critical")
    st.plotly_chart(fig_vif, use_container_width=True)

    st.dataframe(vif_df, use_container_width=True)
```

**Step 2: Manual verification**

```bash
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Navigate to Feature Distributions page. Test all four tabs.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add feature distributions + correlation page"
```

---

## Task 7: Streamlit — Feature Selection Lab

**Files:**
- Create: `packages/dashboard/src/dashboard/pages/07_feature_selection_lab.py`

**Step 1: Implement the page**

```python
# packages/dashboard/src/dashboard/pages/07_feature_selection_lab.py
"""Feature Selection Lab — toggle features, retrain, compare models."""

from pathlib import Path

import ibis
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml.dataset import build_training_dataset, split_features_target, train_test_split_by_season
from ml.feature_analysis.importance import compute_gini_importance
from ml.feature_analysis.selection import (
    evaluate_feature_subset,
    compare_feature_subsets,
    SubsetEvaluation,
)
from ml.model import FourthDownModel
from ml.serialize import load_model
from nfl.features import MODEL_FEATURE_COLUMNS

st.header("🧪 Feature Selection Lab")
st.markdown(
    "Toggle features on/off, retrain the model, and compare performance. "
    "Use the importance scores and correlation analysis from other pages "
    "to guide your decisions."
)

# --- Load Data ---
data_path = st.sidebar.text_input("Training data (Parquet)", value="data/pbp/")

if not Path(data_path).exists() or not list(Path(data_path).glob("*.parquet")):
    st.warning("No training data found. Run the training script with `--cache-data`.")
    st.stop()

@st.cache_data
def load_and_split(path: str):
    parquet_files = sorted(Path(path).glob("*.parquet"))
    dfs = [pd.read_parquet(p) for p in parquet_files]
    raw = pd.concat(dfs, ignore_index=True)
    con = ibis.duckdb.connect()
    dataset = con.execute(build_training_dataset(ibis.memtable(raw)))
    train, test = train_test_split_by_season(dataset, test_seasons=[2023])
    X_train, y_train = split_features_target(train)
    X_test, y_test = split_features_target(test)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_and_split(data_path)
st.success(f"Train: {len(X_train):,} plays | Test: {len(X_test):,} plays")

# --- Feature Toggles ---
st.subheader("Select Features")

# Initialize session state for feature selection
if "selected_features" not in st.session_state:
    st.session_state["selected_features"] = list(MODEL_FEATURE_COLUMNS)

col_toggle, col_info = st.columns([2, 1])

with col_toggle:
    # Quick-select buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    if btn_col1.button("Select All"):
        st.session_state["selected_features"] = list(MODEL_FEATURE_COLUMNS)
    if btn_col2.button("Clear All"):
        st.session_state["selected_features"] = []
    if btn_col3.button("Top 10 by Gini"):
        # Quick Gini importance to auto-select top features
        quick_model = FourthDownModel(hyperparams={"n_estimators": 50})
        quick_model.fit(X_train, y_train)
        imp = compute_gini_importance(quick_model, X_train)
        top_10 = imp.to_dataframe().head(10)["feature"].tolist()
        st.session_state["selected_features"] = top_10

    # Feature checkboxes in a grid
    st.markdown("---")
    n_cols = 3
    feature_rows = [
        MODEL_FEATURE_COLUMNS[i : i + n_cols]
        for i in range(0, len(MODEL_FEATURE_COLUMNS), n_cols)
    ]

    selected = []
    for row in feature_rows:
        cols = st.columns(n_cols)
        for i, feat in enumerate(row):
            checked = cols[i].checkbox(
                feat,
                value=feat in st.session_state["selected_features"],
                key=f"feat_{feat}",
            )
            if checked:
                selected.append(feat)
    st.session_state["selected_features"] = selected

with col_info:
    st.metric("Features Selected", f"{len(selected)} / {len(MODEL_FEATURE_COLUMNS)}")
    if len(selected) == 0:
        st.error("Select at least one feature!")
        st.stop()

# --- Evaluate ---
st.subheader("Evaluation")

if st.button("Train & Evaluate", type="primary", use_container_width=True):
    with st.spinner("Training models and cross-validating..."):
        # Evaluate both full and selected subsets
        subsets = {
            "All Features": list(MODEL_FEATURE_COLUMNS),
            "Selected Features": selected,
        }
        results = compare_feature_subsets(X_train, y_train, subsets)

        # Also evaluate on held-out test set
        for name, feature_list in subsets.items():
            model = FourthDownModel(hyperparams={"n_estimators": 100})
            model.fit(X_train[feature_list], y_train)
            test_acc = float((model.predict(X_test[feature_list]) == y_test).mean())
            results[name].test_accuracy = test_acc  # type: ignore

    # Store results in session state
    st.session_state["eval_results"] = results
    st.session_state["eval_subsets"] = subsets

# --- Display Results ---
if "eval_results" in st.session_state:
    results = st.session_state["eval_results"]
    subsets = st.session_state["eval_subsets"]

    # Comparison metrics
    comp_col1, comp_col2 = st.columns(2)

    all_res = results["All Features"]
    sel_res = results["Selected Features"]

    with comp_col1:
        st.markdown("### All Features")
        st.metric("CV Accuracy", f"{all_res.cv_mean:.4f}", help=f"±{all_res.cv_std:.4f}")
        if hasattr(all_res, "test_accuracy"):
            st.metric("Test Accuracy", f"{all_res.test_accuracy:.4f}")
        st.metric("Feature Count", all_res.n_features)

    with comp_col2:
        st.markdown("### Selected Features")
        delta_cv = sel_res.cv_mean - all_res.cv_mean
        st.metric(
            "CV Accuracy",
            f"{sel_res.cv_mean:.4f}",
            delta=f"{delta_cv:+.4f}",
            delta_color="normal",
            help=f"±{sel_res.cv_std:.4f}",
        )
        if hasattr(sel_res, "test_accuracy"):
            delta_test = sel_res.test_accuracy - all_res.test_accuracy
            st.metric(
                "Test Accuracy",
                f"{sel_res.test_accuracy:.4f}",
                delta=f"{delta_test:+.4f}",
                delta_color="normal",
            )
        st.metric("Feature Count", sel_res.n_features)

    # CV score distribution
    st.subheader("Cross-Validation Score Distribution")

    cv_data = []
    for name, res in results.items():
        for score in res.cv_scores:
            cv_data.append({"Subset": name, "CV Score": score})

    fig_cv = px.box(
        pd.DataFrame(cv_data),
        x="Subset",
        y="CV Score",
        color="Subset",
        title="Cross-Validation Score Distribution",
        points="all",
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    # Interpretation
    if delta_cv >= -0.005:
        st.success(
            f"Selected features ({sel_res.n_features}) perform comparably to "
            f"all features ({all_res.n_features}). "
            "You could use the smaller set for a simpler, faster model."
        )
    elif delta_cv >= -0.02:
        st.info(
            f"Small accuracy drop ({delta_cv:+.4f}). "
            "The removed features carry some signal — consider if the simplicity tradeoff is worth it."
        )
    else:
        st.warning(
            f"Significant accuracy drop ({delta_cv:+.4f}). "
            "Important features may have been removed. Check the importance page."
        )

# --- Save Configuration ---
st.markdown("---")
with st.expander("Export selected feature list"):
    st.code(
        f"SELECTED_FEATURES = {json.dumps(selected, indent=2)}",
        language="python",
    )
    st.caption("Copy this into your code to use the selected feature subset.")

import json
```

**Step 2: Manual verification**

```bash
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Navigate to Feature Selection Lab. Toggle features, click Train & Evaluate, verify comparison renders.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add feature selection lab page"
```

---

## Summary: Run Order

| Task | What it builds | Key files |
|------|---------------|-----------|
| 1 | Gini + Permutation + SHAP importance | `ml/feature_analysis/importance.py` |
| 2 | Correlation matrix + VIF | `ml/feature_analysis/correlation.py` |
| 3 | Feature distribution stats via Ibis | `ml/feature_analysis/distributions.py` |
| 4 | Feature subset evaluation | `ml/feature_analysis/selection.py` |
| 5 | Streamlit: Feature Importance page | `pages/05_feature_importance.py` |
| 6 | Streamlit: Distributions + Correlations page | `pages/06_feature_distributions.py` |
| 7 | Streamlit: Feature Selection Lab page | `pages/07_feature_selection_lab.py` |

Tasks 1–4 are pure backend (testable, no UI). Tasks 5–7 are Streamlit pages consuming those backends.
