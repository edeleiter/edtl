"""XGBoost model wrapper for 4th-down decision prediction."""

import os

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from nfl.features import MODEL_FEATURE_COLUMNS

DEFAULT_HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": min(os.cpu_count() or 4, 4),
}


class FourthDownModel:
    """XGBoost classifier for 4th-down decisions.

    Wraps XGBClassifier with sensible defaults for this problem.
    """

    def __init__(self, hyperparams: dict | None = None):
        params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self._model = XGBClassifier(**params)
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list | None = None,
        verbose: bool = False,
    ) -> "FourthDownModel":
        self._model.fit(X, y, eval_set=eval_set, verbose=verbose)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def feature_importances(self) -> dict[str, float]:
        importances = self._model.feature_importances_
        return {k: float(v) for k, v in zip(MODEL_FEATURE_COLUMNS, importances)}

    @property
    def raw_model(self):
        """Access the underlying XGBClassifier for serialization."""
        return self._model

    @classmethod
    def from_xgb(cls, xgb_model) -> "FourthDownModel":
        """Create a FourthDownModel from a pre-trained XGBClassifier."""
        instance = cls.__new__(cls)
        instance._model = xgb_model
        instance._is_fitted = True
        return instance

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
