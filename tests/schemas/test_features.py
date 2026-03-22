import pytest
from pydantic import ValidationError

from schemas.features import (
    ImportanceResult,
    ImportanceMethod,
    ImportanceEntry,
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
            ImportanceEntry(feature="ydstogo", importance=0.35),
            ImportanceEntry(feature="yardline_100", importance=0.25),
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
    with pytest.raises(ValidationError):
        CorrelationPair(feature_a="a", feature_b="b", correlation=1.5)


def test_vif_entry():
    entry = VIFEntry(feature="score_differential", vif=7.3)
    assert entry.vif >= 1.0


def test_vif_entry_validates_minimum():
    with pytest.raises(ValidationError):
        VIFEntry(feature="x", vif=0.5)


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
