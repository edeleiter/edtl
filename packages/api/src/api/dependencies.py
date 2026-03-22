"""FastAPI dependency injection — app state and lifespan."""

import os

from ml.predict import FourthDownPredictor


class AppState:
    """Shared application state initialized at startup."""

    fourth_down_predictor: FourthDownPredictor | None = None

    def startup(self) -> None:
        model_dir = os.environ.get("MODEL_DIR")
        if model_dir:
            self.fourth_down_predictor = FourthDownPredictor(model_dir)

    def shutdown(self) -> None:
        if self.fourth_down_predictor is not None:
            self.fourth_down_predictor.close()
        self.fourth_down_predictor = None


app_state = AppState()
