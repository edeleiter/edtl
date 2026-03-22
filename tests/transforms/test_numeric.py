import ibis
import pandas as pd
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
def sample_table():
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
    assert "value_zscore" in result.columns
    assert abs(result["value_zscore"].mean()) < 1e-10


def test_log_transform(con, sample_table):
    expr = log_transform(sample_table, "value")
    result = con.execute(expr)
    assert "value_log" in result.columns
    assert result["value_log"].iloc[0] == pytest.approx(0.0, abs=1e-10)


def test_clip_outliers(con, sample_table):
    expr = clip_outliers(sample_table, "value", lower=1.0, upper=10.0)
    result = con.execute(expr)
    assert result["value_clipped"].max() == 10.0
    assert result["value_clipped"].min() == 1.0


def test_ratio_with_zero_denominator(con, sample_table):
    expr = ratio(sample_table, "numerator", "denominator")
    result = con.execute(expr)
    assert pd.isna(result["numerator_denominator_ratio"].iloc[-1])


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
