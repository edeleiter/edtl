import pytest
from pydantic import ValidationError

from schemas.transforms import (
    TransformStep,
    PipelineDefinition,
    TransformRequest,
    TransformResponse,
    ReferenceDataManifest,
)


def test_transform_step_valid():
    step = TransformStep(
        name="clip_outliers",
        params={"column": "value", "lower": 0.0, "upper": 100.0},
    )
    assert step.name == "clip_outliers"


def test_pipeline_definition():
    pipeline = PipelineDefinition(
        name="fourth_down_features",
        steps=[
            TransformStep(name="clip_outliers", params={"column": "value", "lower": 0.0, "upper": 100.0}),
            TransformStep(name="zscore_normalize", params={"column": "value"}),
        ],
        version="1.0.0",
    )
    assert len(pipeline.steps) == 2
    assert pipeline.version == "1.0.0"


def test_transform_request():
    req = TransformRequest(
        data=[{"value": 5.0, "numerator": 10.0, "denominator": 2.0}],
        transforms=["clip_outliers"],
        transform_params={"clip_outliers": {"column": "value", "lower": 0.0, "upper": 10.0}},
    )
    assert len(req.data) == 1


def test_transform_request_is_mutable():
    req = TransformRequest(
        data=[{"x": 1}],
        transforms=["a"],
    )
    req.transforms.append("b")
    assert len(req.transforms) == 2


def test_reference_data_manifest():
    manifest = ReferenceDataManifest(
        tables={"category_map": "/data/ref/category_map.parquet"},
    )
    assert "category_map" in manifest.tables
