"""Inference-time prediction — loads model, applies feature pipeline via DuckDB."""

from pathlib import Path

import ibis
import pandas as pd

from ml.serialize import load_model
from nfl.features import build_fourth_down_features, MODEL_FEATURE_COLUMNS
from nfl.target import INVERSE_LABEL_MAP


class FourthDownPredictor:
    """Loads a trained model and predicts 4th-down decisions.

    Uses DuckDB for feature engineering at inference time,
    ensuring the same transforms as training.
    """

    def __init__(self, model_dir: str | Path):
        self.model = load_model(model_dir)
        self._con = ibis.duckdb.connect()

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._con.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def predict_from_game_state(self, game_state: dict) -> dict:
        """Predict from a single game-state dict.

        Input keys: ydstogo, yardline_100, score_differential,
                    half_seconds_remaining, game_seconds_remaining,
                    quarter_seconds_remaining, qtr, goal_to_go, wp

        Returns: {"recommendation": str, "probabilities": {str: float}}
        """
        results = self.predict_batch([game_state])
        return results[0]

    def predict_batch(self, game_states: list[dict]) -> list[dict]:
        """Predict from multiple game-state dicts."""
        # Build Ibis table from input
        table = ibis.memtable(pd.DataFrame(game_states))

        # Apply the same feature pipeline used in training
        featured = build_fourth_down_features(table)

        # Execute on DuckDB and extract features
        df = self._con.execute(featured)
        X = df[MODEL_FEATURE_COLUMNS]

        # Predict
        predictions = self.model.predict(X)
        probas = self.model.predict_proba(X)

        results = []
        for pred, proba in zip(predictions, probas):
            pred_label = int(pred)
            results.append({
                "recommendation": INVERSE_LABEL_MAP[pred_label],
                "probabilities": {
                    INVERSE_LABEL_MAP[j]: float(proba[j])
                    for j in range(3)
                },
            })
        return results
