"""Fourth-down prediction endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import app_state
from schemas.game import GameState

router = APIRouter(prefix="/fourth-down", tags=["fourth-down"])


class PredictionResponse(BaseModel):
    """Single prediction result."""

    recommendation: str
    probabilities: dict[str, float]


class BatchRequest(BaseModel):
    """Multiple game states for batch prediction."""

    game_states: list[GameState] = Field(max_length=1000)


class BatchResponse(BaseModel):
    """Batch prediction results."""

    predictions: list[PredictionResponse]


def _get_predictor():
    if app_state.fourth_down_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Set MODEL_DIR environment variable.",
        )
    return app_state.fourth_down_predictor


@router.post("/predict", response_model=PredictionResponse)
def predict(game_state: GameState):
    """Predict the optimal 4th-down decision for a single game state."""
    predictor = _get_predictor()
    result = predictor.predict_from_game_state(game_state.model_dump())
    return PredictionResponse(**result)


@router.post("/predict-batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    """Predict optimal 4th-down decisions for multiple game states."""
    predictor = _get_predictor()
    states = [gs.model_dump() for gs in batch.game_states]
    results = predictor.predict_batch(states)
    return BatchResponse(
        predictions=[PredictionResponse(**r) for r in results]
    )
