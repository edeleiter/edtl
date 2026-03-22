# Statistical Interpretation Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Plan 5 of 10**

**Goal:** Add plain-English interpretation to every statistical output in the feature selection and data quality UIs — so that any user, regardless of statistics background, can understand what the numbers mean, why they matter, and what to do about them.

**Architecture:** Two interpretation modules (`ml.feature_analysis.interpret` and `data_quality.interpret`) take raw statistical results as input and produce structured `Interpretation` objects containing: a plain-English summary, a severity/confidence level, contextual guidance on what to do next, and optional educational tooltips explaining the methodology. Streamlit pages are updated to render interpretations inline next to every chart and metric. An LLM-powered "Ask About This" feature (optional, using Anthropic API) lets users ask follow-up questions about any specific stat in context.

**Tech Stack:**
- **Core:** Pure Python (no new dependencies for interpretation logic)
- **Optional:** Anthropic API (Claude Sonnet) for contextual Q&A in artifacts
- **Inherits:** Everything from Plans 1–4

**Prerequisite:** Feature Selection UI (Plan 3) and Data Quality UI (Plan 4) must be implemented first.

> **Note:** Task 6 of this plan (Anthropic API-powered Q&A) has been superseded by Plan 6 (WebGPU LLM Advisor), which runs Qwen 2.5 1.5B directly in the browser with zero API dependencies. Implement Plan 6 instead of Task 6 below. Task 6 is retained as a fallback for environments where WebGPU is unavailable.

---

## Design Principles

**1. Every number gets a sentence.** No metric should appear on screen without at least one sentence explaining what it means in context. "KS p-value: 0.003" becomes "The distributions are statistically different (KS p=0.003). There's only a 0.3% chance this difference is due to random sampling."

**2. Interpretations are tiered.** Each interpretation has three levels of detail:
- **Headline:** One sentence, color-coded (green/yellow/red). Always visible.
- **Explanation:** 2-3 sentences explaining why this matters. Visible on click/expand.
- **Guidance:** What to do about it. Specific, actionable steps.

**3. Context matters.** The same VIF of 5.2 means different things depending on whether the feature is your most important predictor (keep it, accept the collinearity) or your least important (safe to remove). Interpretations should cross-reference other results when possible.

**4. No false precision.** "Accuracy dropped by 0.3%" is better than "accuracy dropped by 0.002847." Round aggressively in prose.

---

## Extended Project Structure

```
unified-etl/
├── packages/
│   ├── ml/
│   │   └── src/
│   │       └── ml/
│   │           └── feature_analysis/
│   │               ├── ... (existing)
│   │               └── interpret.py            # Feature selection interpretation
│   │
│   ├── data_quality/
│   │   └── src/
│   │       └── data_quality/
│   │           ├── ... (existing)
│   │           └── interpret.py                # Data quality interpretation
│   │
│   └── dashboard/
│       └── src/
│           └── dashboard/
│               ├── components/
│               │   ├── ... (existing)
│               │   ├── interpretation_card.py  # Reusable interpretation renderer
│               │   └── ask_about.py            # Optional LLM Q&A component
│               └── pages/
│                   ├── 05_feature_importance.py     # (modify)
│                   ├── 06_feature_distributions.py  # (modify)
│                   ├── 07_feature_selection_lab.py   # (modify)
│                   ├── 08_data_quality_overview.py   # (modify)
│                   ├── 09_drift_monitor.py           # (modify)
│                   └── 10_validation_rules.py        # (modify)
│
└── tests/
    ├── ml/
    │   └── test_feature_interpret.py
    └── data_quality/
        └── test_dq_interpret.py
```

---

## Task 1: Interpretation Data Model + Feature Importance Interpreter

**Files:**
- Create: `packages/ml/src/ml/feature_analysis/interpret.py`
- Test: `tests/ml/test_feature_interpret.py`

**Step 1: Write the failing test**

```python
# tests/ml/test_feature_interpret.py
import numpy as np
import pandas as pd
import pytest

from ml.feature_analysis.interpret import (
    Interpretation,
    InterpretationLevel,
    interpret_gini_importance,
    interpret_permutation_importance,
    interpret_shap_importance,
    interpret_correlation_pair,
    interpret_vif,
    interpret_feature_subset_comparison,
    interpret_feature_importance_ranking,
)
from ml.feature_analysis.importance import ImportanceResult
from ml.feature_analysis.correlation import CorrelationResult
from ml.feature_analysis.selection import SubsetEvaluation


def test_interpretation_structure():
    interp = Interpretation(
        level=InterpretationLevel.OK,
        headline="Everything looks fine",
        explanation="No issues detected.",
        guidance="No action needed.",
        metric_name="test_metric",
        metric_value=0.5,
    )
    assert interp.level == InterpretationLevel.OK
    assert len(interp.headline) > 0
    assert interp.tooltip is None  # Optional


def test_interpretation_levels():
    assert InterpretationLevel.OK.value == "ok"
    assert InterpretationLevel.INFO.value == "info"
    assert InterpretationLevel.WARNING.value == "warning"
    assert InterpretationLevel.CRITICAL.value == "critical"


# --- Gini Importance ---

def test_interpret_gini_dominant_feature():
    """A single feature with >40% importance should trigger a warning."""
    result = ImportanceResult(
        method="gini",
        feature_names=["a", "b", "c", "d"],
        scores=np.array([0.65, 0.15, 0.10, 0.10]),
    )
    interps = interpret_gini_importance(result)
    assert len(interps) >= 1
    # The dominant feature interpretation
    dominant = [i for i in interps if "a" in i.headline]
    assert len(dominant) >= 1
    assert dominant[0].level in (InterpretationLevel.WARNING, InterpretationLevel.INFO)


def test_interpret_gini_balanced():
    """Evenly distributed importance should be OK."""
    result = ImportanceResult(
        method="gini",
        feature_names=["a", "b", "c", "d"],
        scores=np.array([0.28, 0.26, 0.24, 0.22]),
    )
    interps = interpret_gini_importance(result)
    # Should have a summary interpretation
    summaries = [i for i in interps if i.metric_name == "gini_summary"]
    assert len(summaries) >= 1
    assert summaries[0].level == InterpretationLevel.OK


def test_interpret_gini_near_zero_features():
    """Features with ~0 importance should be flagged as removal candidates."""
    result = ImportanceResult(
        method="gini",
        feature_names=["a", "b", "c", "d"],
        scores=np.array([0.50, 0.45, 0.04, 0.01]),
    )
    interps = interpret_gini_importance(result)
    low_imp = [i for i in interps if i.metric_name == "gini_low_importance"]
    assert len(low_imp) >= 1
    assert "d" in low_imp[0].headline or "removal" in low_imp[0].guidance.lower()


# --- Permutation Importance ---

def test_interpret_permutation_negative():
    """Negative permutation importance means the feature hurts the model."""
    result = ImportanceResult(
        method="permutation",
        feature_names=["a", "b", "c"],
        scores=np.array([0.05, -0.02, 0.10]),
        std=np.array([0.01, 0.01, 0.02]),
    )
    interps = interpret_permutation_importance(result)
    negative = [i for i in interps if "b" in i.headline]
    assert len(negative) >= 1
    assert negative[0].level == InterpretationLevel.WARNING


def test_interpret_permutation_high_variance():
    """High std relative to mean indicates unstable importance."""
    result = ImportanceResult(
        method="permutation",
        feature_names=["a", "b"],
        scores=np.array([0.05, 0.04]),
        std=np.array([0.08, 0.01]),  # a has std > mean
    )
    interps = interpret_permutation_importance(result)
    unstable = [i for i in interps if i.metric_name == "permutation_unstable"]
    assert len(unstable) >= 1


# --- VIF ---

def test_interpret_vif_healthy():
    vif_df = pd.DataFrame({"feature": ["a", "b", "c"], "vif": [1.2, 2.1, 1.5]})
    interps = interpret_vif(vif_df)
    summary = [i for i in interps if i.metric_name == "vif_summary"]
    assert len(summary) >= 1
    assert summary[0].level == InterpretationLevel.OK


def test_interpret_vif_moderate():
    vif_df = pd.DataFrame({"feature": ["a", "b", "c"], "vif": [7.5, 1.2, 1.5]})
    interps = interpret_vif(vif_df)
    flagged = [i for i in interps if "a" in i.headline]
    assert len(flagged) >= 1
    assert flagged[0].level == InterpretationLevel.WARNING


def test_interpret_vif_severe():
    vif_df = pd.DataFrame({"feature": ["a", "b"], "vif": [15.0, 12.0]})
    interps = interpret_vif(vif_df)
    critical = [i for i in interps if i.level == InterpretationLevel.CRITICAL]
    assert len(critical) >= 1


# --- Correlation ---

def test_interpret_correlation_pair_high():
    interp = interpret_correlation_pair("score_differential", "is_trailing", 0.92)
    assert interp.level == InterpretationLevel.WARNING
    assert "redundant" in interp.explanation.lower() or "collinear" in interp.explanation.lower()


def test_interpret_correlation_pair_moderate():
    interp = interpret_correlation_pair("a", "b", 0.55)
    assert interp.level == InterpretationLevel.INFO


# --- Subset Comparison ---

def test_interpret_subset_comparable():
    all_eval = SubsetEvaluation(
        features=["a", "b", "c"], n_features=3,
        accuracy=0.85, cv_mean=0.83, cv_std=0.02, cv_scores=[0.81, 0.83, 0.85, 0.83, 0.83],
    )
    sel_eval = SubsetEvaluation(
        features=["a", "b"], n_features=2,
        accuracy=0.84, cv_mean=0.82, cv_std=0.02, cv_scores=[0.80, 0.82, 0.84, 0.82, 0.82],
    )
    interp = interpret_feature_subset_comparison(all_eval, sel_eval)
    assert interp.level == InterpretationLevel.OK
    assert "simpler" in interp.guidance.lower() or "comparable" in interp.headline.lower()


def test_interpret_subset_significant_drop():
    all_eval = SubsetEvaluation(
        features=["a", "b", "c"], n_features=3,
        accuracy=0.85, cv_mean=0.83, cv_std=0.02, cv_scores=[0.81, 0.83, 0.85, 0.83, 0.83],
    )
    sel_eval = SubsetEvaluation(
        features=["a"], n_features=1,
        accuracy=0.70, cv_mean=0.68, cv_std=0.05, cv_scores=[0.63, 0.68, 0.73, 0.68, 0.68],
    )
    interp = interpret_feature_subset_comparison(all_eval, sel_eval)
    assert interp.level == InterpretationLevel.CRITICAL


# --- Overall Ranking ---

def test_interpret_feature_ranking():
    result = ImportanceResult(
        method="gini",
        feature_names=["ydstogo", "yardline_100", "wp", "qtr", "goal_to_go"],
        scores=np.array([0.35, 0.25, 0.20, 0.12, 0.08]),
    )
    interps = interpret_feature_importance_ranking(result, top_n=3)
    assert len(interps) >= 1
    # Should mention the top features by name
    all_text = " ".join(i.headline + i.explanation for i in interps)
    assert "ydstogo" in all_text
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/ml/test_feature_interpret.py -v
```

Expected: FAIL.

**Step 3: Implement the feature interpretation engine**

```python
# packages/ml/src/ml/feature_analysis/interpret.py
"""Plain-English interpretation of feature analysis statistics.

Every function takes raw statistical results and returns structured
Interpretation objects with headlines, explanations, and guidance.

Design: Interpretations are tiered (OK/INFO/WARNING/CRITICAL) and
context-aware — the same number means different things depending on
the surrounding statistics.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from ml.feature_analysis.importance import ImportanceResult
from ml.feature_analysis.selection import SubsetEvaluation


# NOTE: When Plan 9 (schemas) is implemented, replace this local definition
# with: from schemas.interpretation import InterpretationLevel, Interpretation
# Plan 9 Task 6 centralizes these to avoid the duplicate definition in
# data_quality.interpret (see consistency review Issue #3).
class InterpretationLevel(Enum):
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Interpretation:
    """A structured interpretation of a statistical result."""

    level: InterpretationLevel
    headline: str            # One sentence, always visible
    explanation: str         # 2-3 sentences, shown on expand
    guidance: str            # What to do about it
    metric_name: str = ""    # Which metric this interprets
    metric_value: float | str | None = None
    tooltip: str | None = None  # Optional methodology explanation

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "headline": self.headline,
            "explanation": self.explanation,
            "guidance": self.guidance,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
        }


# ---------------------------------------------------------------------------
# Gini Importance
# ---------------------------------------------------------------------------

def interpret_gini_importance(result: ImportanceResult) -> list[Interpretation]:
    """Interpret Gini (split-based) feature importance scores."""
    interps = []
    df = result.to_dataframe()
    scores = df["importance"].to_numpy()
    features = df["feature"].tolist()

    # Overall distribution assessment
    max_score = float(scores.max())
    entropy = -np.sum(scores * np.log(scores + 1e-10))
    max_entropy = -np.log(1.0 / len(scores)) * len(scores) if len(scores) > 0 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

    if normalized_entropy > 0.85:
        interps.append(Interpretation(
            level=InterpretationLevel.OK,
            headline="Feature importance is well-distributed across features.",
            explanation=(
                f"No single feature dominates the model's decisions. "
                f"The most important feature ({features[0]}) accounts for "
                f"{max_score:.0%} of importance, and the distribution is fairly even."
            ),
            guidance="This is healthy — the model isn't over-reliant on any one signal.",
            metric_name="gini_summary",
            metric_value=f"entropy={normalized_entropy:.2f}",
            tooltip=(
                "Gini importance measures how much each feature reduces impurity "
                "across all tree splits. Higher = the feature is used more often "
                "in important splits. Caveat: Gini importance is biased toward "
                "features with more unique values (like continuous features over binary ones)."
            ),
        ))
    else:
        interps.append(Interpretation(
            level=InterpretationLevel.INFO,
            headline=f"Importance is concentrated — {features[0]} dominates at {max_score:.0%}.",
            explanation=(
                f"The top feature ({features[0]}) accounts for {max_score:.0%} of total "
                f"importance. This isn't necessarily bad — it may genuinely be the strongest "
                f"predictor. But it means the model is heavily reliant on this one signal."
            ),
            guidance=(
                f"Verify that {features[0]} will always be available at inference time. "
                f"Consider checking if this feature is a proxy for the target (data leakage)."
            ),
            metric_name="gini_summary",
            metric_value=f"entropy={normalized_entropy:.2f}",
        ))

    # Dominant feature check
    if max_score > 0.40:
        interps.append(Interpretation(
            level=InterpretationLevel.WARNING,
            headline=f"{features[0]} has unusually high importance ({max_score:.0%}).",
            explanation=(
                f"When a single feature accounts for over 40% of importance, it's worth "
                f"investigating. This could indicate data leakage (the feature is derived "
                f"from or highly correlated with the target), or it could mean this feature "
                f"genuinely carries most of the predictive signal."
            ),
            guidance=(
                f"Check if {features[0]} could be leaking information about the target. "
                f"Try training without it — if accuracy barely drops, it may be redundant "
                f"with other features. If accuracy drops a lot, it's a genuine strong signal."
            ),
            metric_name="gini_dominant",
            metric_value=max_score,
        ))

    # Near-zero importance features
    low_threshold = 0.02
    low_features = [f for f, s in zip(features, scores) if s < low_threshold]
    if low_features:
        interps.append(Interpretation(
            level=InterpretationLevel.INFO,
            headline=f"{len(low_features)} feature(s) contribute almost nothing: {', '.join(low_features[:5])}.",
            explanation=(
                f"These features have importance below {low_threshold:.0%}, meaning they "
                f"rarely influence the model's decisions. They may be noise, or they may "
                f"carry signal that's already captured by other features."
            ),
            guidance=(
                f"These are candidates for removal. Try dropping them in the Feature Selection "
                f"Lab and check if accuracy changes. Fewer features = simpler model, faster "
                f"inference, and less risk of overfitting."
            ),
            metric_name="gini_low_importance",
            metric_value=len(low_features),
        ))

    return interps


# ---------------------------------------------------------------------------
# Permutation Importance
# ---------------------------------------------------------------------------

def interpret_permutation_importance(result: ImportanceResult) -> list[Interpretation]:
    """Interpret permutation importance scores with error bars."""
    interps = []
    df = result.to_dataframe()
    features = df["feature"].tolist()
    scores = df["importance"].to_numpy()
    stds = df["std"].to_numpy() if "std" in df.columns else np.zeros_like(scores)

    # Negative importance features
    negative_features = [
        (f, s, sd) for f, s, sd in zip(features, scores, stds) if s < 0
    ]
    for feat, score, std in negative_features:
        interps.append(Interpretation(
            level=InterpretationLevel.WARNING,
            headline=f"{feat} has negative importance ({score:.4f}) — it may be hurting the model.",
            explanation=(
                f"Negative permutation importance means that shuffling {feat} actually "
                f"*improved* accuracy. This typically happens when a feature adds noise "
                f"that the model overfits to. It can also occur with highly correlated "
                f"features where the model compensates after one is shuffled."
            ),
            guidance=(
                f"Strong candidate for removal. Try training without {feat} — you may "
                f"see a small accuracy improvement. If the standard deviation ({std:.4f}) "
                f"is larger than the absolute score, the result is inconclusive."
            ),
            metric_name="permutation_negative",
            metric_value=score,
        ))

    # High-variance features (unstable importance)
    unstable = [
        (f, s, sd) for f, s, sd in zip(features, scores, stds)
        if sd > abs(s) and s >= 0
    ]
    if unstable:
        names = [f for f, _, _ in unstable[:5]]
        interps.append(Interpretation(
            level=InterpretationLevel.INFO,
            headline=f"{len(unstable)} feature(s) have unstable importance: {', '.join(names)}.",
            explanation=(
                f"For these features, the standard deviation across permutation repeats "
                f"is larger than the mean importance. This means the importance estimate "
                f"is unreliable — it could be zero or significant depending on the random shuffle."
            ),
            guidance=(
                f"Increase the number of permutation repeats (try 20-30) for a more stable "
                f"estimate. If the importance remains unstable, the feature likely has "
                f"weak or inconsistent predictive value."
            ),
            metric_name="permutation_unstable",
            metric_value=len(unstable),
        ))

    # Summary
    top_3 = [(f, s) for f, s in zip(features, scores)][:3]
    interps.append(Interpretation(
        level=InterpretationLevel.OK,
        headline=f"Top predictors by permutation: {', '.join(f for f, _ in top_3)}.",
        explanation=(
            f"Permutation importance measures the actual accuracy drop when each feature "
            f"is shuffled. Unlike Gini importance, it's unbiased — it doesn't favor "
            f"continuous features over binary ones. "
            + (f"The top feature ({top_3[0][0]}) causes a {top_3[0][1]:.4f} accuracy drop "
               f"when shuffled." if top_3 else "")
        ),
        guidance=(
            f"Compare this ranking with Gini importance. If they largely agree, "
            f"you can be confident about which features matter. If they disagree, "
            f"permutation importance is generally more trustworthy."
        ),
        metric_name="permutation_summary",
        tooltip=(
            "Permutation importance works by shuffling one feature at a time and "
            "measuring how much the model's accuracy drops. A big drop = the feature "
            "is important. No drop = the feature doesn't matter. Negative = the "
            "feature may be adding noise."
        ),
    ))

    return interps


# ---------------------------------------------------------------------------
# SHAP Importance
# ---------------------------------------------------------------------------

def interpret_shap_importance(result: ImportanceResult) -> list[Interpretation]:
    """Interpret SHAP-based feature importance."""
    interps = []
    df = result.to_dataframe()
    features = df["feature"].tolist()
    scores = df["importance"].to_numpy()

    top_feature = features[0]
    top_score = scores[0]

    interps.append(Interpretation(
        level=InterpretationLevel.OK,
        headline=f"SHAP analysis: {top_feature} has the strongest influence on predictions.",
        explanation=(
            f"Mean |SHAP| for {top_feature} is {top_score:.4f}, meaning that on average, "
            f"this feature shifts the model's prediction by ±{top_score:.4f} from the "
            f"baseline. SHAP values decompose each prediction into per-feature contributions, "
            f"so this is the most granular importance measure available."
        ),
        guidance=(
            f"Examine the SHAP value distribution plots below for {top_feature}. "
            f"If high feature values consistently push predictions one way, the "
            f"relationship is monotonic and easy to explain. If the spread is wide "
            f"with no clear pattern, the feature interacts with others in complex ways."
        ),
        metric_name="shap_summary",
        metric_value=top_score,
        tooltip=(
            "SHAP (SHapley Additive exPlanations) assigns each feature a contribution "
            "to each individual prediction. The mean absolute SHAP value across all "
            "samples gives the feature's overall importance. Unlike Gini or permutation, "
            "SHAP explains *how* each feature affects predictions, not just *whether* it does."
        ),
    ))

    # Check for features where SHAP disagrees with Gini
    if result.shap_values is not None:
        spread = np.std(np.abs(result.shap_values), axis=0)
        high_spread_idx = np.where(spread > np.median(spread) * 2)[0]
        if len(high_spread_idx) > 0:
            high_spread_features = [features[i] for i in high_spread_idx[:3]]
            interps.append(Interpretation(
                level=InterpretationLevel.INFO,
                headline=f"Features with high SHAP variability: {', '.join(high_spread_features)}.",
                explanation=(
                    f"These features have highly variable SHAP values — their importance "
                    f"differs a lot from sample to sample. This usually means the feature "
                    f"is important in some game situations but not others (interaction effects)."
                ),
                guidance=(
                    f"Look at the SHAP dependence plots for these features to understand "
                    f"when and how they influence predictions. This could reveal interesting "
                    f"domain insights (e.g., yards to go matters more in the 4th quarter)."
                ),
                metric_name="shap_variable_features",
            ))

    return interps


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def interpret_correlation_pair(
    feature_a: str, feature_b: str, correlation: float
) -> Interpretation:
    """Interpret a single correlation coefficient between two features."""
    abs_corr = abs(correlation)
    direction = "positively" if correlation > 0 else "negatively"

    if abs_corr >= 0.9:
        return Interpretation(
            level=InterpretationLevel.WARNING,
            headline=f"{feature_a} and {feature_b} are very highly {direction} correlated (r={correlation:.3f}).",
            explanation=(
                f"With |r| = {abs_corr:.3f}, these features carry nearly identical information. "
                f"The model is splitting its importance attribution between them, making both "
                f"appear less important than they really are. This also increases the variance "
                f"of model coefficients and can make the model less stable."
            ),
            guidance=(
                f"Remove one of these features. Keep whichever is more interpretable, more "
                f"reliable to compute at inference time, or ranks higher in permutation importance. "
                f"Check VIF scores — if both have high VIF, removing one should reduce the other's VIF."
            ),
            metric_name="correlation_pair",
            metric_value=correlation,
            tooltip="Pearson correlation measures linear relationship strength. |r| > 0.8 indicates substantial redundancy.",
        )
    elif abs_corr >= 0.7:
        return Interpretation(
            level=InterpretationLevel.WARNING,
            headline=f"{feature_a} and {feature_b} are moderately-to-highly correlated (r={correlation:.3f}).",
            explanation=(
                f"These features share substantial information (|r| = {abs_corr:.3f}). "
                f"While tree-based models like XGBoost handle correlation better than linear "
                f"models, it still affects importance scores — the model arbitrarily chooses "
                f"one over the other at each split, making importance unreliable for this pair."
            ),
            guidance=(
                f"Worth investigating but not urgent. Check if removing one improves or "
                f"maintains accuracy in the Feature Selection Lab. If accuracy is unchanged, "
                f"prefer the simpler model."
            ),
            metric_name="correlation_pair",
            metric_value=correlation,
        )
    elif abs_corr >= 0.4:
        return Interpretation(
            level=InterpretationLevel.INFO,
            headline=f"{feature_a} and {feature_b} have moderate correlation (r={correlation:.3f}).",
            explanation=(
                f"Some shared information, but each likely contributes unique signal too. "
                f"This level of correlation is common and usually not problematic for "
                f"tree-based models."
            ),
            guidance="No action needed — this is a normal level of correlation between related features.",
            metric_name="correlation_pair",
            metric_value=correlation,
        )
    else:
        return Interpretation(
            level=InterpretationLevel.OK,
            headline=f"{feature_a} and {feature_b} have low correlation (r={correlation:.3f}).",
            explanation="These features carry largely independent information.",
            guidance="No action needed.",
            metric_name="correlation_pair",
            metric_value=correlation,
        )


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------

def interpret_vif(vif_df: pd.DataFrame) -> list[Interpretation]:
    """Interpret Variance Inflation Factors for all features."""
    interps = []

    critical = vif_df[vif_df["vif"] > 10]
    moderate = vif_df[(vif_df["vif"] > 5) & (vif_df["vif"] <= 10)]
    healthy = vif_df[vif_df["vif"] <= 5]

    if len(critical) > 0:
        names = ", ".join(critical["feature"].tolist()[:5])
        worst = critical.iloc[0]
        interps.append(Interpretation(
            level=InterpretationLevel.CRITICAL,
            headline=f"Severe multicollinearity: {names} (VIF > 10).",
            explanation=(
                f"{worst['feature']} has VIF = {worst['vif']:.1f}, meaning that "
                f"{(1 - 1/worst['vif'])*100:.0f}% of its variance is explained by the other "
                f"features combined. The feature is almost entirely redundant — the model "
                f"can predict this feature's values from the others with high accuracy."
            ),
            guidance=(
                f"Removing the highest-VIF feature usually fixes it. After removing one, "
                f"re-check VIF — the others' VIF scores will drop. Prioritize removing "
                f"whichever has the lowest permutation importance."
            ),
            metric_name="vif_critical",
            metric_value=float(worst["vif"]),
            tooltip=(
                "VIF = 1/(1-R²) where R² is from regressing this feature on all others. "
                "VIF=1 means no collinearity. VIF=10 means 90% of this feature's variance "
                "is shared. Unlike pairwise correlation, VIF catches collinearity that "
                "involves 3+ features simultaneously."
            ),
        ))

    if len(moderate) > 0:
        names = ", ".join(moderate["feature"].tolist()[:5])
        interps.append(Interpretation(
            level=InterpretationLevel.WARNING,
            headline=f"Moderate multicollinearity: {names} (VIF 5-10).",
            explanation=(
                f"These features share meaningful information with others. Not severe enough "
                f"to require immediate action, but importance scores for these features "
                f"may be unreliable."
            ),
            guidance=(
                f"Keep an eye on these. If you're trying to reduce features, these are "
                f"good candidates — they may be partially redundant with other features."
            ),
            metric_name="vif_moderate",
            metric_value=len(moderate),
        ))

    if len(critical) == 0 and len(moderate) == 0:
        interps.append(Interpretation(
            level=InterpretationLevel.OK,
            headline="No multicollinearity issues detected (all VIF < 5).",
            explanation=(
                f"All {len(healthy)} features have VIF below 5, indicating each carries "
                f"substantially independent information. Feature importance scores should "
                f"be reliable and not distorted by collinearity."
            ),
            guidance="No action needed — feature set is well-conditioned.",
            metric_name="vif_summary",
        ))
    else:
        interps.append(Interpretation(
            level=InterpretationLevel.OK,
            headline=f"{len(healthy)} feature(s) have healthy VIF (< 5).",
            explanation="These features carry independent information and their importance scores are reliable.",
            guidance="No action needed for these features.",
            metric_name="vif_summary",
            metric_value=len(healthy),
        ))

    return interps


# ---------------------------------------------------------------------------
# Feature Subset Comparison
# ---------------------------------------------------------------------------

def interpret_feature_subset_comparison(
    all_eval: SubsetEvaluation,
    selected_eval: SubsetEvaluation,
) -> Interpretation:
    """Interpret the comparison between full and selected feature subsets."""
    cv_delta = selected_eval.cv_mean - all_eval.cv_mean
    n_removed = all_eval.n_features - selected_eval.n_features
    pct_removed = 100 * n_removed / all_eval.n_features if all_eval.n_features > 0 else 0

    # Statistical significance check: are the CV score distributions overlapping?
    # Using a rough rule: if delta < 2 * combined_std, it's within noise
    combined_std = np.sqrt(all_eval.cv_std**2 + selected_eval.cv_std**2)
    is_significant = abs(cv_delta) > 2 * combined_std

    if cv_delta >= -0.005:
        return Interpretation(
            level=InterpretationLevel.OK,
            headline=f"Selected features perform comparably (Δ accuracy: {cv_delta:+.1%}).",
            explanation=(
                f"Removing {n_removed} features ({pct_removed:.0f}% of the total) resulted "
                f"in a negligible accuracy change of {cv_delta:+.4f}. "
                f"The CV standard deviations overlap (all: ±{all_eval.cv_std:.4f}, "
                f"selected: ±{selected_eval.cv_std:.4f}), confirming the difference is within noise."
            ),
            guidance=(
                f"The simpler model with {selected_eval.n_features} features is preferred. "
                f"Fewer features means faster inference, less data to collect, and reduced "
                f"risk of overfitting. Deploy with the selected subset."
            ),
            metric_name="subset_comparison",
            metric_value=cv_delta,
        )
    elif cv_delta >= -0.02:
        return Interpretation(
            level=InterpretationLevel.WARNING if is_significant else InterpretationLevel.INFO,
            headline=f"Small accuracy tradeoff (Δ accuracy: {cv_delta:+.1%}).",
            explanation=(
                f"Removing {n_removed} features reduced CV accuracy by {abs(cv_delta):.1%}. "
                + (f"This difference {'is' if is_significant else 'is not'} statistically significant "
                   f"given the CV variance. ")
                + f"You're trading a small amount of accuracy for a {pct_removed:.0f}% reduction "
                f"in feature count."
            ),
            guidance=(
                f"This is a judgment call. If inference speed and simplicity matter, "
                f"the {abs(cv_delta):.1%} drop may be acceptable. If maximum accuracy is "
                f"critical, keep the full feature set. Try adding back just the 1-2 most "
                f"important removed features to find the sweet spot."
            ),
            metric_name="subset_comparison",
            metric_value=cv_delta,
        )
    else:
        return Interpretation(
            level=InterpretationLevel.CRITICAL,
            headline=f"Significant accuracy drop (Δ accuracy: {cv_delta:+.1%}).",
            explanation=(
                f"Removing {n_removed} features caused a {abs(cv_delta):.1%} accuracy loss, "
                f"which is substantial. Important predictive signals have been removed. "
                f"The CV scores confirm this is not just noise (all: {all_eval.cv_mean:.4f} ± "
                f"{all_eval.cv_std:.4f}, selected: {selected_eval.cv_mean:.4f} ± "
                f"{selected_eval.cv_std:.4f})."
            ),
            guidance=(
                f"Add features back until accuracy recovers. Use the importance rankings "
                f"to identify which removed features carry the most signal. "
                f"You likely removed a key predictor."
            ),
            metric_name="subset_comparison",
            metric_value=cv_delta,
        )


# ---------------------------------------------------------------------------
# Overall Feature Ranking Narrative
# ---------------------------------------------------------------------------

def interpret_feature_importance_ranking(
    result: ImportanceResult,
    top_n: int = 5,
) -> list[Interpretation]:
    """Generate a narrative summary of the feature importance ranking."""
    df = result.to_dataframe()
    top = df.head(top_n)
    bottom = df.tail(min(3, len(df)))

    top_names = top["feature"].tolist()
    top_scores = top["importance"].tolist()
    bottom_names = bottom["feature"].tolist()

    # Concentration: what % of total importance is in the top N?
    total = df["importance"].sum()
    top_pct = sum(top_scores) / total if total > 0 else 0

    interps = [
        Interpretation(
            level=InterpretationLevel.OK,
            headline=f"Top {top_n} features by {result.method}: {', '.join(top_names)}.",
            explanation=(
                f"These {top_n} features account for {top_pct:.0%} of total importance. "
                f"The strongest predictor is {top_names[0]} ({top_scores[0]:.4f}), followed by "
                f"{top_names[1]} ({top_scores[1]:.4f})"
                + (f" and {top_names[2]} ({top_scores[2]:.4f})" if len(top_names) > 2 else "")
                + ". "
                + f"The least important features are {', '.join(bottom_names)}."
            ),
            guidance=(
                f"If you need to reduce features, start by removing from the bottom of this "
                f"list. The top {top_n} should generally be kept unless VIF or domain knowledge "
                f"suggests otherwise."
            ),
            metric_name="ranking_summary",
        )
    ]

    return interps
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/ml/test_feature_interpret.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(ml): add plain-English interpretation for feature analysis statistics"
```

---

## Task 2: Data Quality Interpretation Engine

**Files:**
- Create: `packages/data_quality/src/data_quality/interpret.py`
- Test: `tests/data_quality/test_dq_interpret.py`

**Step 1: Write the failing test**

```python
# tests/data_quality/test_dq_interpret.py
import numpy as np
import pandas as pd
import pytest

from data_quality.interpret import (
    Interpretation,
    InterpretationLevel,
    interpret_schema_result,
    interpret_drift_column,
    interpret_drift_overall,
    interpret_rule_result,
    interpret_null_rate,
    interpret_overall_report,
)
from data_quality.schema import (
    SchemaValidationResult,
    ValidationError,
    Severity,
)
from data_quality.drift import ColumnDrift, DriftResult, DriftSeverity
from data_quality.rules import RuleResult
from data_quality.report import ValidationReport, OverallStatus


# --- Schema Interpretation ---

def test_interpret_schema_pass():
    result = SchemaValidationResult(is_valid=True, errors=[])
    interps = interpret_schema_result(result)
    assert len(interps) >= 1
    assert interps[0].level == InterpretationLevel.OK


def test_interpret_schema_missing_column():
    result = SchemaValidationResult(
        is_valid=False,
        errors=[
            ValidationError(column="epa", message="Missing column 'epa'", severity=Severity.ERROR)
        ],
    )
    interps = interpret_schema_result(result)
    critical = [i for i in interps if i.level == InterpretationLevel.CRITICAL]
    assert len(critical) >= 1
    assert "epa" in critical[0].headline


def test_interpret_schema_unknown_category():
    result = SchemaValidationResult(
        is_valid=True,
        errors=[
            ValidationError(
                column="posteam",
                message="Unknown categories: {'FAKE_TEAM'}",
                severity=Severity.WARNING,
                details={"unknown_values": ["FAKE_TEAM"]},
            )
        ],
    )
    interps = interpret_schema_result(result)
    warnings = [i for i in interps if i.level == InterpretationLevel.WARNING]
    assert len(warnings) >= 1


# --- Drift Interpretation ---

def test_interpret_drift_no_drift():
    drift = ColumnDrift(
        column="value",
        severity=DriftSeverity.NONE,
        ks_statistic=0.05,
        ks_pvalue=0.45,
        mean_shift=0.2,
    )
    interp = interpret_drift_column(drift)
    assert interp.level == InterpretationLevel.OK


def test_interpret_drift_significant():
    drift = ColumnDrift(
        column="value",
        severity=DriftSeverity.CRITICAL,
        ks_statistic=0.35,
        ks_pvalue=0.0001,
        mean_shift=4.5,
    )
    interp = interpret_drift_column(drift)
    assert interp.level == InterpretationLevel.CRITICAL
    assert "shifted" in interp.explanation.lower() or "different" in interp.explanation.lower()


def test_interpret_drift_new_categories():
    drift = ColumnDrift(
        column="team",
        severity=DriftSeverity.WARNING,
        new_categories=["NEW_TEAM"],
        missing_categories=["OLD_TEAM"],
    )
    interp = interpret_drift_column(drift)
    assert interp.level in (InterpretationLevel.WARNING, InterpretationLevel.CRITICAL)
    assert "NEW_TEAM" in interp.explanation


def test_interpret_drift_null_spike():
    drift = ColumnDrift(
        column="epa",
        severity=DriftSeverity.CRITICAL,
        null_rate_drift=0.35,
    )
    interp = interpret_drift_column(drift)
    assert interp.level == InterpretationLevel.CRITICAL
    assert "null" in interp.headline.lower() or "missing" in interp.headline.lower()


# --- Rule Interpretation ---

def test_interpret_rule_low_failure():
    result = RuleResult(
        rule_name="valid_down",
        column="down",
        severity=Severity.ERROR,
        total_rows=10000,
        failing_rows=3,
        failing_pct=0.03,
        message="3 rows failed",
    )
    interp = interpret_rule_result(result)
    assert interp.level == InterpretationLevel.WARNING  # Very low rate = likely data entry error


def test_interpret_rule_high_failure():
    result = RuleResult(
        rule_name="valid_wp",
        column="wp",
        severity=Severity.ERROR,
        total_rows=1000,
        failing_rows=250,
        failing_pct=25.0,
        message="250 rows failed",
    )
    interp = interpret_rule_result(result)
    assert interp.level == InterpretationLevel.CRITICAL


# --- Null Rate ---

def test_interpret_null_rate_healthy():
    interp = interpret_null_rate("value", null_pct=0.5, ref_null_pct=0.3)
    assert interp.level == InterpretationLevel.OK


def test_interpret_null_rate_spike():
    interp = interpret_null_rate("value", null_pct=25.0, ref_null_pct=0.1)
    assert interp.level == InterpretationLevel.CRITICAL


# --- Overall ---

def test_interpret_overall_pass():
    report = ValidationReport(
        status=OverallStatus.PASS,
        timestamp="2025-01-01T00:00:00",
        row_count=1000,
    )
    interp = interpret_overall_report(report)
    assert interp.level == InterpretationLevel.OK


def test_interpret_overall_fail():
    report = ValidationReport(
        status=OverallStatus.FAIL,
        timestamp="2025-01-01T00:00:00",
        row_count=1000,
    )
    interp = interpret_overall_report(report)
    assert interp.level == InterpretationLevel.CRITICAL
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/data_quality/test_dq_interpret.py -v
```

Expected: FAIL.

**Step 3: Implement the data quality interpretation engine**

```python
# packages/data_quality/src/data_quality/interpret.py
"""Plain-English interpretation of data quality statistics.

Turns KS p-values, drift severities, null rates, and rule failures
into sentences a non-statistician can act on.
"""

from dataclasses import dataclass
from enum import Enum

from data_quality.schema import SchemaValidationResult, Severity, ValidationError
from data_quality.drift import ColumnDrift, DriftResult, DriftSeverity
from data_quality.rules import RuleResult
from data_quality.report import ValidationReport, OverallStatus


class InterpretationLevel(Enum):
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Interpretation:
    """Structured interpretation of a data quality metric."""

    level: InterpretationLevel
    headline: str
    explanation: str
    guidance: str
    metric_name: str = ""
    metric_value: float | str | None = None
    tooltip: str | None = None

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "headline": self.headline,
            "explanation": self.explanation,
            "guidance": self.guidance,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
        }


# ---------------------------------------------------------------------------
# Schema Interpretation
# ---------------------------------------------------------------------------

def interpret_schema_result(result: SchemaValidationResult) -> list[Interpretation]:
    """Interpret schema validation results."""
    interps = []

    if result.is_valid and not result.errors:
        interps.append(Interpretation(
            level=InterpretationLevel.OK,
            headline="Schema validation passed — all columns, types, and ranges match expectations.",
            explanation=(
                "The incoming data has the same structure as the training data: "
                "all expected columns are present, data types are correct, numeric values "
                "are within observed ranges, and categorical values are recognized."
            ),
            guidance="No action needed. Data is structurally sound.",
            metric_name="schema_overall",
        ))
        return interps

    # Group errors by type
    missing_cols = [e for e in result.errors if "Missing column" in e.message]
    type_errors = [e for e in result.errors if "Expected numeric" in e.message or "type" in e.message.lower()]
    range_errors = [e for e in result.errors if "above range" in e.message or "below range" in e.message]
    null_errors = [e for e in result.errors if "null" in e.message.lower()]
    category_warnings = [e for e in result.errors if "Unknown categories" in e.message]

    if missing_cols:
        col_names = [e.column for e in missing_cols]
        interps.append(Interpretation(
            level=InterpretationLevel.CRITICAL,
            headline=f"Missing {len(missing_cols)} column(s): {', '.join(col_names)}.",
            explanation=(
                f"The incoming data is missing columns that the model requires. "
                f"Without these columns, feature engineering will fail and the model "
                f"cannot make predictions. This usually means the data source changed "
                f"its schema (renamed or dropped columns)."
            ),
            guidance=(
                f"Check the data source for schema changes. The missing columns are: "
                f"{', '.join(col_names)}. If columns were renamed, update the ingestion "
                f"pipeline. If they were dropped, the model needs retraining without them."
            ),
            metric_name="schema_missing_columns",
            metric_value=len(missing_cols),
        ))

    if range_errors:
        cols = list(set(e.column for e in range_errors))
        interps.append(Interpretation(
            level=InterpretationLevel.CRITICAL if len(range_errors) > 3 else InterpretationLevel.WARNING,
            headline=f"Values out of expected range in {len(cols)} column(s): {', '.join(cols[:5])}.",
            explanation=(
                f"Some values fall outside the range observed in training data. "
                f"This could be data entry errors (e.g., yardline_100 = 200), a genuine "
                f"shift in the data, or the range margins being too tight. "
                + "; ".join(
                    f"{e.column}: {e.message}" for e in range_errors[:3]
                )
            ),
            guidance=(
                f"Inspect the out-of-range values. If they're clearly errors (impossible "
                f"values like negative yards), filter them out. If they're plausible but "
                f"unseen in training (e.g., a 99-yard play), the model may handle them "
                f"poorly — monitor predictions for these cases."
            ),
            metric_name="schema_range_errors",
            metric_value=len(range_errors),
        ))

    if null_errors:
        cols = [e.column for e in null_errors]
        interps.append(Interpretation(
            level=InterpretationLevel.CRITICAL,
            headline=f"Unexpected nulls in {len(null_errors)} column(s): {', '.join(cols[:5])}.",
            explanation=(
                f"These columns were never null in training data but now contain missing "
                f"values. Null values will propagate through feature engineering and likely "
                f"cause errors or produce wrong predictions."
            ),
            guidance=(
                f"Determine why these columns have nulls now. Is it a data pipeline issue? "
                f"A schema change? If nulls are expected going forward, the feature "
                f"engineering and model need to be updated to handle them."
            ),
            metric_name="schema_null_errors",
            metric_value=len(null_errors),
        ))

    if category_warnings:
        for e in category_warnings:
            unknown = e.details.get("unknown_values", [])
            interps.append(Interpretation(
                level=InterpretationLevel.WARNING,
                headline=f"New categories in {e.column}: {', '.join(str(v) for v in unknown[:5])}.",
                explanation=(
                    f"The model has never seen these category values during training. "
                    f"For NFL data, this could mean a new team (expansion), a renamed "
                    f"category, or a data encoding change. The model will treat these "
                    f"as unknown and may produce unreliable predictions for these rows."
                ),
                guidance=(
                    f"If these are legitimate new values (e.g., team relocations), "
                    f"the model needs retraining. If they're typos, fix the data source."
                ),
                metric_name="schema_new_categories",
                metric_value=len(unknown),
            ))

    return interps


# ---------------------------------------------------------------------------
# Drift Interpretation
# ---------------------------------------------------------------------------

def interpret_drift_column(drift: ColumnDrift) -> Interpretation:
    """Interpret drift analysis for a single column."""

    # Null rate spike — check this first as it's the most actionable
    if drift.null_rate_drift is not None and drift.null_rate_drift > 0.10:
        return Interpretation(
            level=InterpretationLevel.CRITICAL,
            headline=f"Null rate spiked by {drift.null_rate_drift:.0%} in {drift.column}.",
            explanation=(
                f"The percentage of missing values in {drift.column} increased by "
                f"{drift.null_rate_drift:.0%} compared to the training data. "
                f"A jump this large usually indicates a data pipeline issue — "
                f"a source field stopped being populated, or an upstream join is failing."
            ),
            guidance=(
                f"Investigate the data pipeline immediately. Check if the source system "
                f"changed its schema or if an ETL job is failing. Do not use this data "
                f"for predictions until the null rate is resolved."
            ),
            metric_name="drift_null_spike",
            metric_value=drift.null_rate_drift,
            tooltip=(
                "Null rate drift = (new null %) - (reference null %). A large positive "
                "value means data completeness has degraded."
            ),
        )

    if drift.severity == DriftSeverity.NONE:
        detail = ""
        if drift.ks_pvalue is not None:
            detail = (
                f"The KS test p-value is {drift.ks_pvalue:.3f} (above 0.05), meaning "
                f"any differences from training data are within normal random variation."
            )
        if drift.mean_shift is not None:
            detail += f" The mean has shifted by only {drift.mean_shift:.1f} standard deviations."

        return Interpretation(
            level=InterpretationLevel.OK,
            headline=f"No significant drift in {drift.column}.",
            explanation=(
                f"The distribution of {drift.column} in the new data is statistically "
                f"consistent with the training data. " + detail
            ),
            guidance="No action needed.",
            metric_name="drift_column",
            metric_value=drift.ks_pvalue,
        )

    # Numeric drift
    if drift.ks_pvalue is not None:
        headline_parts = []
        explanation_parts = []

        if drift.mean_shift is not None and drift.mean_shift > 1.0:
            headline_parts.append(f"mean shifted {drift.mean_shift:.1f}σ")
            explanation_parts.append(
                f"The average value has moved {drift.mean_shift:.1f} standard deviations "
                f"from the training mean. "
                + ("This is a moderate shift. " if drift.mean_shift < 3 else
                   "This is a large shift — the model was trained on very different data. ")
            )

        if drift.ks_pvalue < 0.001:
            headline_parts.append(f"KS p<0.001")
            explanation_parts.append(
                f"The Kolmogorov-Smirnov test strongly rejects the hypothesis that these "
                f"distributions are the same (p={drift.ks_pvalue:.6f}). "
                f"The shape of the distribution has changed, not just the center."
            )
        elif drift.ks_pvalue < 0.05:
            headline_parts.append(f"KS p={drift.ks_pvalue:.3f}")
            explanation_parts.append(
                f"The KS test detects a statistically significant difference (p={drift.ks_pvalue:.3f}), "
                f"though the effect size may be small."
            )

        level = (
            InterpretationLevel.CRITICAL if drift.severity == DriftSeverity.CRITICAL
            else InterpretationLevel.WARNING
        )

        return Interpretation(
            level=level,
            headline=f"Distribution drift in {drift.column}: {', '.join(headline_parts)}.",
            explanation=" ".join(explanation_parts),
            guidance=(
                f"Compare the histogram overlays to understand what changed. "
                + ("The model's predictions for {drift.column}-dependent decisions may be unreliable. "
                   "Consider retraining on recent data. " if level == InterpretationLevel.CRITICAL else
                   "Monitor this — if the drift persists, retraining may be needed. ")
            ),
            metric_name="drift_column",
            metric_value=drift.ks_pvalue,
            tooltip=(
                "The Kolmogorov-Smirnov test compares two distributions without assuming "
                "they're normal. p < 0.05 means the distributions are statistically different. "
                "Smaller p = more confident the difference is real."
            ),
        )

    # Categorical drift
    if drift.new_categories or drift.missing_categories:
        parts = []
        if drift.new_categories:
            parts.append(f"new values: {', '.join(drift.new_categories[:5])}")
        if drift.missing_categories:
            parts.append(f"missing values: {', '.join(drift.missing_categories[:5])}")

        jsd_part = ""
        if drift.category_proportion_drift is not None:
            jsd_part = (
                f" The Jensen-Shannon divergence is {drift.category_proportion_drift:.4f}"
                + (" — substantial divergence." if drift.category_proportion_drift > 0.05
                   else " — minor divergence.")
            )

        return Interpretation(
            level=(
                InterpretationLevel.CRITICAL if drift.severity == DriftSeverity.CRITICAL
                else InterpretationLevel.WARNING
            ),
            headline=f"Category drift in {drift.column}: {'; '.join(parts)}.",
            explanation=(
                f"The set of values in {drift.column} has changed from training data. "
                f"The model has never seen the new values and will treat them as unknown. "
                + jsd_part
            ),
            guidance=(
                f"For new categories, check if the data encoding changed or if these are "
                f"genuinely new entities. For missing categories, verify the data is complete. "
                f"The model may need retraining if the category distribution has fundamentally changed."
            ),
            metric_name="drift_column_categorical",
            tooltip=(
                "Jensen-Shannon divergence measures how different two probability "
                "distributions are. 0 = identical, ~0.7 = completely different. "
                "Values above 0.05 indicate meaningful proportion shifts."
            ),
        )

    # Fallback
    return Interpretation(
        level=InterpretationLevel.INFO,
        headline=f"Minor changes detected in {drift.column}.",
        explanation="Some differences from training data, but within normal bounds.",
        guidance="No immediate action required.",
        metric_name="drift_column",
    )


def interpret_drift_overall(drift_result: DriftResult) -> Interpretation:
    """Summarize drift across all columns."""
    n_critical = sum(1 for d in drift_result.columns.values() if d.severity == DriftSeverity.CRITICAL)
    n_warning = sum(1 for d in drift_result.columns.values() if d.severity == DriftSeverity.WARNING)
    n_ok = sum(1 for d in drift_result.columns.values() if d.severity == DriftSeverity.NONE)
    total = len(drift_result.columns)

    if n_critical > 0:
        critical_names = [
            name for name, d in drift_result.columns.items()
            if d.severity == DriftSeverity.CRITICAL
        ]
        return Interpretation(
            level=InterpretationLevel.CRITICAL,
            headline=f"Critical drift in {n_critical} of {total} columns: {', '.join(critical_names[:5])}.",
            explanation=(
                f"The incoming data has significantly different distributions from training "
                f"data in {n_critical} column(s). Model predictions may be unreliable — "
                f"the model is being asked to make decisions about data it wasn't designed for."
            ),
            guidance=(
                f"Investigate the drifted columns in the Drift Monitor page. If the drift "
                f"is due to a data pipeline issue, fix the pipeline. If it's genuine "
                f"distribution change (e.g., new season, rule changes), retrain the model "
                f"on recent data."
            ),
            metric_name="drift_overall",
            metric_value=n_critical,
        )
    elif n_warning > 0:
        return Interpretation(
            level=InterpretationLevel.WARNING,
            headline=f"Drift warnings in {n_warning} of {total} columns.",
            explanation=(
                f"Some columns show statistical differences from training data. "
                f"This is common with real-world data — distributions shift over time. "
                f"The model should still perform reasonably, but monitor prediction quality."
            ),
            guidance=(
                f"Review the flagged columns. If predictions seem off, these drift "
                f"warnings may explain why. Plan a retraining cycle if drift persists."
            ),
            metric_name="drift_overall",
            metric_value=n_warning,
        )
    else:
        return Interpretation(
            level=InterpretationLevel.OK,
            headline=f"No drift detected across all {total} columns.",
            explanation=(
                f"The incoming data is statistically consistent with training data across "
                f"all features. The model should perform as expected."
            ),
            guidance="No action needed. Continue monitoring with each new data batch.",
            metric_name="drift_overall",
        )


# ---------------------------------------------------------------------------
# Rule Interpretation
# ---------------------------------------------------------------------------

def interpret_rule_result(result: RuleResult) -> Interpretation:
    """Interpret a business rule validation result."""
    pct = result.failing_pct

    if pct < 0.1:
        return Interpretation(
            level=InterpretationLevel.WARNING,
            headline=f"Rule '{result.rule_name}' failed for {result.failing_rows} rows ({pct:.2f}%).",
            explanation=(
                f"A tiny fraction of rows violate this rule. At {pct:.2f}%, this is "
                f"likely isolated data entry errors rather than a systematic issue. "
                f"These rows have physically impossible values for {result.column}."
            ),
            guidance=(
                f"Filter out the {result.failing_rows} invalid rows before feeding data "
                f"to the model. Log them for upstream data quality feedback."
            ),
            metric_name="rule_result",
            metric_value=pct,
        )
    elif pct < 5.0:
        return Interpretation(
            level=InterpretationLevel.WARNING,
            headline=f"Rule '{result.rule_name}' failed for {result.failing_rows} rows ({pct:.1f}%).",
            explanation=(
                f"About {pct:.1f}% of rows violate this rule. This is higher than "
                f"typical data entry error rates and may indicate a systematic issue "
                f"with the data source or transformation pipeline."
            ),
            guidance=(
                f"Investigate the failing rows — is there a pattern? A specific game, "
                f"week, or data source producing bad values? Fix upstream if possible, "
                f"otherwise filter and document."
            ),
            metric_name="rule_result",
            metric_value=pct,
        )
    else:
        return Interpretation(
            level=InterpretationLevel.CRITICAL,
            headline=f"Rule '{result.rule_name}' failed for {pct:.0f}% of rows — possible data corruption.",
            explanation=(
                f"{result.failing_rows} of {result.total_rows} rows ({pct:.1f}%) have "
                f"invalid values in {result.column}. A failure rate this high suggests "
                f"the data source has changed its encoding, the column was mismapped, "
                f"or there's a bug in the data pipeline."
            ),
            guidance=(
                f"Do NOT use this data for model training or inference without fixing the "
                f"underlying issue. Check the data pipeline for recent changes. Inspect "
                f"the raw data source to confirm the column mapping is correct."
            ),
            metric_name="rule_result",
            metric_value=pct,
        )


# ---------------------------------------------------------------------------
# Null Rate
# ---------------------------------------------------------------------------

def interpret_null_rate(
    column: str,
    null_pct: float,
    ref_null_pct: float | None = None,
) -> Interpretation:
    """Interpret the null rate for a column, optionally compared to reference."""

    if ref_null_pct is not None:
        delta = null_pct - ref_null_pct
        if delta > 10.0:
            return Interpretation(
                level=InterpretationLevel.CRITICAL,
                headline=f"Null rate for {column} jumped from {ref_null_pct:.1f}% to {null_pct:.1f}%.",
                explanation=(
                    f"Missing data increased by {delta:.1f} percentage points. This magnitude "
                    f"of change almost always indicates a data pipeline issue — the column "
                    f"stopped being populated, an upstream join is failing, or a source "
                    f"system changed its API."
                ),
                guidance=(
                    f"Investigate immediately. Check the data source, ETL logs, and any "
                    f"recent pipeline changes. Do not use this data until resolved."
                ),
                metric_name="null_rate",
                metric_value=null_pct,
            )
        elif delta > 2.0:
            return Interpretation(
                level=InterpretationLevel.WARNING,
                headline=f"Null rate for {column} increased from {ref_null_pct:.1f}% to {null_pct:.1f}%.",
                explanation=(
                    f"Missing data increased by {delta:.1f} percentage points. This could "
                    f"be a transient issue (partial data load) or the beginning of a "
                    f"data quality degradation."
                ),
                guidance=(
                    f"Monitor over the next few data batches. If the trend continues, "
                    f"investigate the data source."
                ),
                metric_name="null_rate",
                metric_value=null_pct,
            )

    if null_pct < 1.0:
        return Interpretation(
            level=InterpretationLevel.OK,
            headline=f"{column} has {null_pct:.1f}% missing values — within normal range.",
            explanation="A small amount of missing data is typical and usually handled well by the pipeline.",
            guidance="No action needed.",
            metric_name="null_rate",
            metric_value=null_pct,
        )
    elif null_pct < 10.0:
        return Interpretation(
            level=InterpretationLevel.INFO,
            headline=f"{column} has {null_pct:.1f}% missing values.",
            explanation=(
                f"This is a moderate null rate. Depending on the imputation strategy, "
                f"this should be manageable. Check that the feature engineering pipeline "
                f"handles nulls appropriately for this column."
            ),
            guidance="Verify null handling in the transform pipeline. Consider if imputation strategy is appropriate.",
            metric_name="null_rate",
            metric_value=null_pct,
        )
    else:
        return Interpretation(
            level=InterpretationLevel.WARNING,
            headline=f"{column} has {null_pct:.0f}% missing values — feature may be unreliable.",
            explanation=(
                f"With {null_pct:.0f}% of values missing, this feature provides incomplete "
                f"information for over {null_pct/100:.0%} of the data. Any imputation will "
                f"be speculative for a large portion of rows."
            ),
            guidance=(
                f"Consider whether this column should be used as a feature. If most values "
                f"are missing, the imputed values may introduce more noise than signal."
            ),
            metric_name="null_rate",
            metric_value=null_pct,
        )


# ---------------------------------------------------------------------------
# Overall Report
# ---------------------------------------------------------------------------

def interpret_overall_report(report: ValidationReport) -> Interpretation:
    """Top-level interpretation of the full validation report."""

    if report.status == OverallStatus.PASS:
        return Interpretation(
            level=InterpretationLevel.OK,
            headline="All validation checks passed — data is ready for use.",
            explanation=(
                f"Schema is valid, no significant drift detected, and all business rules "
                f"pass for the {report.row_count:,} rows in this batch."
            ),
            guidance="Data can proceed to the feature engineering pipeline.",
            metric_name="overall_status",
        )
    elif report.status == OverallStatus.WARNING:
        return Interpretation(
            level=InterpretationLevel.WARNING,
            headline="Data passed with warnings — review before using.",
            explanation=(
                f"The data is structurally valid but has some quality concerns. "
                f"These may affect model prediction quality. Review the specific "
                f"warnings below before proceeding."
            ),
            guidance=(
                "Data can be used with caution. Address the warnings if possible, "
                "and monitor model prediction quality for this batch."
            ),
            metric_name="overall_status",
        )
    else:
        return Interpretation(
            level=InterpretationLevel.CRITICAL,
            headline="Validation FAILED — do not use this data without fixing issues.",
            explanation=(
                f"Critical issues detected in the {report.row_count:,}-row batch. "
                f"Using this data will produce unreliable model predictions. "
                f"See the specific errors below."
            ),
            guidance=(
                "Fix the identified issues before proceeding. If the issues are in the "
                "data source, block this batch and alert the data engineering team."
            ),
            metric_name="overall_status",
        )
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/data_quality/test_dq_interpret.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(data-quality): add plain-English interpretation for all DQ metrics"
```

---

## Task 3: Reusable Interpretation Card Component

**Files:**
- Create: `packages/dashboard/src/dashboard/components/interpretation_card.py`

**Step 1: Implement the component**

```python
# packages/dashboard/src/dashboard/components/interpretation_card.py
"""Reusable Streamlit component for rendering Interpretation objects.

Displays a color-coded card with headline, expandable explanation,
guidance, and optional methodology tooltip. Consistent across all
pages that show interpreted statistics.
"""

import streamlit as st

# Import both interpretation types
try:
    from ml.feature_analysis.interpret import Interpretation as FeatureInterpretation
    from ml.feature_analysis.interpret import InterpretationLevel as FeatureLevel
except ImportError:
    FeatureInterpretation = None
    FeatureLevel = None

try:
    from data_quality.interpret import Interpretation as DQInterpretation
    from data_quality.interpret import InterpretationLevel as DQLevel
except ImportError:
    DQInterpretation = None
    DQLevel = None


def render_interpretation(interp, key_prefix: str = "") -> None:
    """Render a single Interpretation object as a Streamlit card.

    Works with both feature analysis and data quality interpretations
    (they share the same structure).
    """
    level = interp.level.value
    headline = interp.headline
    explanation = interp.explanation
    guidance = interp.guidance

    # Color-coded container
    level_config = {
        "ok": {"icon": "✅", "method": "success"},
        "info": {"icon": "💡", "method": "info"},
        "warning": {"icon": "⚠️", "method": "warning"},
        "critical": {"icon": "🚨", "method": "error"},
    }

    config = level_config.get(level, level_config["info"])

    # Headline — always visible
    getattr(st, config["method"])(f"{config['icon']} {headline}")

    # Expandable details
    unique_key = f"{key_prefix}_{interp.metric_name}_{id(interp)}"
    with st.expander("Why this matters & what to do", expanded=level in ("warning", "critical")):
        st.markdown(f"**What this means:** {explanation}")
        st.markdown(f"**What to do:** {guidance}")

        if interp.tooltip:
            st.caption(f"📖 **Methodology:** {interp.tooltip}")

        if interp.metric_value is not None:
            st.caption(f"Raw value: `{interp.metric_value}`")


def render_interpretation_list(
    interps: list,
    key_prefix: str = "",
    max_ok: int = 2,
) -> None:
    """Render a list of interpretations, collapsing OK-level ones.

    Shows all warnings and criticals expanded. Limits the number
    of OK/info items shown to avoid visual clutter.
    """
    # Sort: critical first, then warning, info, ok
    level_order = {"critical": 0, "warning": 1, "info": 2, "ok": 3}
    sorted_interps = sorted(interps, key=lambda i: level_order.get(i.level.value, 4))

    ok_count = 0
    for i, interp in enumerate(sorted_interps):
        if interp.level.value in ("ok", "info"):
            ok_count += 1
            if ok_count > max_ok:
                if ok_count == max_ok + 1:
                    remaining = len([x for x in sorted_interps[i:] if x.level.value in ("ok", "info")])
                    st.caption(f"...and {remaining} more passing checks")
                continue
        render_interpretation(interp, key_prefix=f"{key_prefix}_{i}")


def render_interpretation_sidebar(interps: list) -> None:
    """Render a compact summary in the sidebar."""
    n_critical = sum(1 for i in interps if i.level.value == "critical")
    n_warning = sum(1 for i in interps if i.level.value == "warning")
    n_ok = sum(1 for i in interps if i.level.value in ("ok", "info"))

    if n_critical > 0:
        st.sidebar.error(f"🚨 {n_critical} critical issue(s)")
    if n_warning > 0:
        st.sidebar.warning(f"⚠️ {n_warning} warning(s)")
    if n_ok > 0:
        st.sidebar.success(f"✅ {n_ok} check(s) OK")
```

**Step 2: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add reusable interpretation card component"
```

---

## Task 4: Integrate Interpretations into Feature Selection Pages

**Files:**
- Modify: `packages/dashboard/src/dashboard/pages/05_feature_importance.py`
- Modify: `packages/dashboard/src/dashboard/pages/06_feature_distributions.py`
- Modify: `packages/dashboard/src/dashboard/pages/07_feature_selection_lab.py`

> **Note for the engineer:** These modifications add interpretation calls *after* each statistical computation. The pattern is always: compute → interpret → render card → then show the chart. The chart still appears, but now has a plain-English explanation above it.

**Step 1: Modify 05_feature_importance.py**

Add after the importance computation and before the chart:

```python
# Add imports at top:
from ml.feature_analysis.interpret import (
    interpret_gini_importance,
    interpret_permutation_importance,
    interpret_shap_importance,
    interpret_feature_importance_ranking,
)
from dashboard.components.interpretation_card import render_interpretation_list

# After computing `result` and before the bar chart:
# --- Interpretation ---
if method == "Gini (built-in)":
    interps = interpret_gini_importance(result)
elif method == "Permutation" and has_data:
    interps = interpret_permutation_importance(result)
elif method == "SHAP" and has_data:
    interps = interpret_shap_importance(result)
else:
    interps = []

# Add ranking narrative
interps.extend(interpret_feature_importance_ranking(result, top_n=5))

render_interpretation_list(interps, key_prefix="importance")
```

**Step 2: Modify 06_feature_distributions.py**

Add to the Correlation Matrix tab, after `find_highly_correlated_pairs`:

```python
# Add imports:
from ml.feature_analysis.interpret import interpret_correlation_pair, interpret_vif as interpret_vif_scores
from dashboard.components.interpretation_card import render_interpretation_list, render_interpretation

# In the correlation tab, after showing pairs:
if pairs:
    st.subheader("What This Means")
    for feat_a, feat_b, corr_val in pairs[:5]:
        interp = interpret_correlation_pair(feat_a, feat_b, corr_val)
        render_interpretation(interp, key_prefix=f"corr_{feat_a}_{feat_b}")

# In the VIF tab, after showing VIF chart:
st.subheader("VIF Interpretation")
vif_interps = interpret_vif_scores(vif_df)
render_interpretation_list(vif_interps, key_prefix="vif")
```

**Step 3: Modify 07_feature_selection_lab.py**

Add after the comparison metrics display:

```python
# Add imports:
from ml.feature_analysis.interpret import interpret_feature_subset_comparison
from dashboard.components.interpretation_card import render_interpretation

# After displaying the comparison metrics, before CV score distribution:
st.subheader("What This Means")
comparison_interp = interpret_feature_subset_comparison(all_res, sel_res)
render_interpretation(comparison_interp, key_prefix="subset")
```

**Step 4: Manual verification**

```bash
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Navigate through pages 05, 06, 07. Verify interpretation cards appear above/below charts.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(dashboard): integrate interpretations into feature selection pages"
```

---

## Task 5: Integrate Interpretations into Data Quality Pages

**Files:**
- Modify: `packages/dashboard/src/dashboard/pages/08_data_quality_overview.py`
- Modify: `packages/dashboard/src/dashboard/pages/09_drift_monitor.py`
- Modify: `packages/dashboard/src/dashboard/pages/10_validation_rules.py`

**Step 1: Modify 08_data_quality_overview.py**

Add after the overall status banner:

```python
# Add imports:
from data_quality.interpret import (
    interpret_schema_result,
    interpret_drift_overall,
    interpret_rule_result,
    interpret_overall_report,
)
from dashboard.components.interpretation_card import render_interpretation, render_interpretation_list

# After the status banner, before the three-column layout:
overall_interp = interpret_overall_report(report)
render_interpretation(overall_interp, key_prefix="overall")

# Inside the Schema column (col1), after showing errors:
schema_interps = interpret_schema_result(report.schema_result)
render_interpretation_list(schema_interps, key_prefix="schema")

# Inside the Drift column (col2), after showing drift status:
if report.drift_result:
    drift_interp = interpret_drift_overall(report.drift_result)
    render_interpretation(drift_interp, key_prefix="drift_overall")

# Inside the Rules column (col3), after showing rule violations:
if report.rule_results:
    for r in report.rule_results[:5]:
        rule_interp = interpret_rule_result(r)
        render_interpretation(rule_interp, key_prefix=f"rule_{r.rule_name}")
```

**Step 2: Modify 09_drift_monitor.py**

Add inside each column's expander, after the metrics row:

```python
# Add imports:
from data_quality.interpret import interpret_drift_column, interpret_drift_overall, interpret_null_rate
from dashboard.components.interpretation_card import render_interpretation

# After the summary section:
overall_drift_interp = interpret_drift_overall(drift_result)
render_interpretation(overall_drift_interp, key_prefix="drift_summary")

# Inside each column's expander, after the metrics row and before the histogram:
interp = interpret_drift_column(drift)
render_interpretation(interp, key_prefix=f"drift_{col_name}")

# After the null rate markdown line:
if drift.null_rate_drift and abs(drift.null_rate_drift) > 0.001:
    null_interp = interpret_null_rate(
        col_name,
        null_pct=new_col.null_pct,
        ref_null_pct=ref_col.null_pct,
    )
    render_interpretation(null_interp, key_prefix=f"null_{col_name}")
```

**Step 3: Modify 10_validation_rules.py**

Add inside each rule's expander:

```python
# Add imports:
from data_quality.interpret import interpret_rule_result
from dashboard.components.interpretation_card import render_interpretation

# Inside each rule's expander, after showing the progress bar:
rule_interp = interpret_rule_result(result)
render_interpretation(rule_interp, key_prefix=f"rule_{result.rule_name}")
```

**Step 4: Manual verification**

```bash
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Navigate through pages 08, 09, 10. Verify interpretation cards appear with appropriate color-coding.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(dashboard): integrate interpretations into data quality pages"
```

---

## Task 6: Optional — LLM-Powered "Ask About This" Component

> **⚠️ SUPERSEDED — See Plan 6 (WebGPU LLM Advisor).** Do NOT implement this task alongside Plan 6. Plan 6 replaces this with a fully client-side WebGPU LLM (Qwen 2.5 1.5B) that requires zero API keys and zero server costs. Use Plan 6's `render_llm_advisor()` as the integration point — do NOT add both `render_ask_about()` and `render_llm_advisor()` to the same pages. This task is retained only as an alternative fallback for server-side deployments where WebGPU is unavailable.

This task adds a small Anthropic API-powered component that lets users ask follow-up questions about any specific statistic in context. For example, clicking "Ask about this" next to a VIF score opens a chat where the user can ask "Should I remove this feature even though it's my 2nd most important by SHAP?"

**Files:**
- Create: `packages/dashboard/src/dashboard/components/ask_about.py`

> **Note:** This requires an `ANTHROPIC_API_KEY` environment variable. If not set, the component gracefully degrades to a disabled state with a "Set API key to enable" message.

**Step 1: Implement the component**

```python
# packages/dashboard/src/dashboard/components/ask_about.py
"""Optional LLM-powered Q&A component for statistical interpretations.

Uses Claude Sonnet to answer follow-up questions about specific
statistics, with full context about what the stat means and the
current analysis state.

Requires ANTHROPIC_API_KEY environment variable.
"""

import os

import streamlit as st


def render_ask_about(
    context: str,
    stat_description: str,
    key: str = "ask_about",
) -> None:
    """Render an 'Ask about this' expandable Q&A section.

    Args:
        context: Background context about the analysis (feature names,
            model type, data description, etc.)
        stat_description: Description of the specific statistic being
            shown (e.g., "VIF for score_differential is 7.3")
        key: Unique Streamlit key for the component.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        st.caption("💬 _Set ANTHROPIC_API_KEY to enable AI-powered Q&A about these statistics._")
        return

    with st.expander("💬 Ask about this statistic", expanded=False):
        question = st.text_input(
            "Your question:",
            placeholder="e.g., Should I remove this feature?",
            key=f"{key}_question",
        )

        if question and st.button("Ask", key=f"{key}_button"):
            with st.spinner("Thinking..."):
                try:
                    import anthropic

                    client = anthropic.Anthropic(api_key=api_key)
                    message = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=500,
                        system=(
                            "You are a data science advisor helping a user understand "
                            "feature engineering and data quality statistics for an ML model. "
                            "Give concise, actionable advice. Respond in 2-4 sentences. "
                            "Be specific to their situation — don't give generic textbook answers."
                        ),
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    f"Context about my analysis:\n{context}\n\n"
                                    f"The specific statistic I'm looking at:\n{stat_description}\n\n"
                                    f"My question: {question}"
                                ),
                            }
                        ],
                    )
                    answer = message.content[0].text
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Error calling API: {e}")
```

**Step 2: Example integration (in 06_feature_distributions.py VIF tab)**

```python
from dashboard.components.ask_about import render_ask_about

# After VIF interpretation cards:
render_ask_about(
    context=(
        f"NFL 4th-down decision model with {len(MODEL_FEATURE_COLUMNS)} features. "
        f"XGBoost classifier predicting go/punt/field_goal. "
        f"Features: {', '.join(MODEL_FEATURE_COLUMNS)}"
    ),
    stat_description=f"VIF scores:\n{vif_df.to_string(index=False)}",
    key="vif_ask",
)
```

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add optional LLM-powered Q&A for statistics"
```

---

## Summary: Run Order

| Task | What it builds | Key files |
|------|---------------|-----------|
| 1 | Feature analysis interpretation engine | `ml/feature_analysis/interpret.py` |
| 2 | Data quality interpretation engine | `data_quality/interpret.py` |
| 3 | Reusable interpretation card component | `dashboard/components/interpretation_card.py` |
| 4 | Integrate interpretations into feature pages | Modify pages 05, 06, 07 |
| 5 | Integrate interpretations into DQ pages | Modify pages 08, 09, 10 |
| 6 | Optional LLM Q&A component | `dashboard/components/ask_about.py` |

Tasks 1-2 are pure backend (fully testable). Task 3 is a UI component. Tasks 4-5 are integration. Task 6 is optional enhancement.

## What Changes for Users

**Before:** User sees "VIF: 8.3" and has to know what VIF means, what thresholds matter, and what to do.

**After:** User sees:
> ⚠️ **Moderate multicollinearity: score_differential (VIF 5-10).**
>
> **What this means:** This feature shares meaningful information with others. While tree-based models like XGBoost handle correlation better than linear models, it still affects importance scores.
>
> **What to do:** Keep an eye on this. If you're trying to reduce features, this is a good candidate.
>
> 💬 _Ask about this statistic_: "Should I worry about this given that score_differential is my 3rd most important feature by SHAP?"

Every number on every page gets this treatment.
