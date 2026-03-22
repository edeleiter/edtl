"""Tests for model evaluation."""

import pytest

from ml.evaluate import evaluate_model
from schemas.prediction import EvaluationReport
from nfl.features import MODEL_FEATURE_COLUMNS


def test_evaluate_model_returns_report(trained_model):
    model, X, y = trained_model
    report = evaluate_model(model, X, y)
    assert isinstance(report, EvaluationReport)
    assert 0.0 <= report.accuracy <= 1.0
    assert len(report.confusion_matrix) == 3
    assert len(report.confusion_matrix[0]) == 3
    assert len(report.feature_importances) == len(MODEL_FEATURE_COLUMNS)


def test_report_to_dict(trained_model):
    model, X, y = trained_model
    report = evaluate_model(model, X, y)
    d = report.model_dump()
    assert "accuracy" in d
    assert "confusion_matrix" in d
    assert isinstance(d["confusion_matrix"], list)


def test_report_summary(trained_model):
    model, X, y = trained_model
    report = evaluate_model(model, X, y)
    summary = report.summary()
    assert "Accuracy" in summary
    assert "Confusion Matrix" in summary
