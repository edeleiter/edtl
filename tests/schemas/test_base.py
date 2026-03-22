import datetime

import polars as pl
import pytest
from pydantic import ValidationError

from schemas._base import (
    StrictModel,
    TimestampMixin,
    PolarsConvertible,
    Probability,
    YardLine,
    Quarter,
    Severity,
)


def test_strict_model_rejects_wrong_type():
    class Example(StrictModel):
        value: int

    with pytest.raises(ValidationError):
        Example(value="not_an_int")


def test_strict_model_accepts_correct_type():
    class Example(StrictModel):
        value: int

    obj = Example(value=42)
    assert obj.value == 42


def test_strict_model_is_frozen():
    class Example(StrictModel):
        value: int

    obj = Example(value=42)
    with pytest.raises(ValidationError):
        obj.value = 99


def test_timestamp_mixin_auto_populates():
    class Example(TimestampMixin, StrictModel):
        name: str

    obj = Example(name="test")
    assert obj.created_at is not None
    assert isinstance(obj.created_at, datetime.datetime)
    assert obj.created_at.tzinfo is not None


def test_probability_type_validates_range():
    class Example(StrictModel):
        p: Probability

    obj = Example(p=0.5)
    assert obj.p == 0.5

    with pytest.raises(ValidationError):
        Example(p=1.5)

    with pytest.raises(ValidationError):
        Example(p=-0.1)


def test_yard_line_validates_range():
    class Example(StrictModel):
        yl: YardLine

    Example(yl=50)
    with pytest.raises(ValidationError):
        Example(yl=0)
    with pytest.raises(ValidationError):
        Example(yl=100)


def test_quarter_validates_range():
    class Example(StrictModel):
        q: Quarter

    Example(q=1)
    Example(q=5)  # Overtime
    with pytest.raises(ValidationError):
        Example(q=0)
    with pytest.raises(ValidationError):
        Example(q=6)


def test_severity_enum():
    assert Severity.OK.value == "ok"
    assert Severity.WARNING.value == "warning"
    assert Severity.CRITICAL.value == "critical"


def test_polars_convertible_round_trip():
    class Row(PolarsConvertible, StrictModel):
        name: str
        value: float

    rows = [Row(name="a", value=1.0), Row(name="b", value=2.0)]
    df = Row.to_polars(rows)

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2
    assert df.columns == ["name", "value"]

    restored = Row.from_polars(df)
    assert len(restored) == 2
    assert restored[0].name == "a"
    assert restored[1].value == 2.0
