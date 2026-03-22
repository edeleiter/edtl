"""Model serialization — save/load with metadata."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from nfl.features import MODEL_FEATURE_COLUMNS
from nfl.target import INVERSE_LABEL_MAP


def save_model(
    model,
    output_dir: str | Path,
    report=None,
    version: str | None = None,
) -> Path:
    """Save model + metadata to a directory.

    Creates:
      - model.joblib (serialized XGBoost model)
      - metadata.json (version, features, evaluation metrics)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)

    if version is None:
        version = now.strftime("%Y%m%dT%H%M%S")

    # Save model
    model_path = output_dir / "model.joblib"
    joblib.dump(model.raw_model, model_path)

    # Save metadata
    metadata = {
        "version": version,
        "trained_at": now.isoformat(),
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "n_features": len(MODEL_FEATURE_COLUMNS),
        "n_classes": 3,
        "class_labels": [INVERSE_LABEL_MAP[i] for i in range(3)],
    }
    if report is not None:
        metadata["evaluation"] = report.model_dump()

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    return output_dir


def load_model(model_dir: str | Path):
    """Load a saved model from a directory.

    Returns a FourthDownModel with the loaded XGBoost model.
    """
    from ml.model import FourthDownModel

    model_dir = Path(model_dir)
    model_path = model_dir / "model.joblib"
    xgb_model = joblib.load(model_path)

    return FourthDownModel.from_xgb(xgb_model)
