"""Model evaluation — accuracy, confusion matrix, classification report."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from nfl.target import INVERSE_LABEL_MAP
from schemas.prediction import EvaluationReport


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> EvaluationReport:
    """Evaluate model on test data, return structured report."""
    y_pred = model.predict(X)
    target_names = [INVERSE_LABEL_MAP[i] for i in sorted(INVERSE_LABEL_MAP.keys())]

    return EvaluationReport(
        accuracy=float(accuracy_score(y, y_pred)),
        confusion_matrix=confusion_matrix(y, y_pred).tolist(),
        class_report=classification_report(y, y_pred, target_names=target_names),
        feature_importances=model.feature_importances(),
    )
