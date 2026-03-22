"""FastAPI application — unified-etl inference service."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.dependencies import app_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    app_state.startup()
    yield
    app_state.shutdown()


app = FastAPI(
    title="Unified ETL — Inference API",
    version="0.1.0",
    lifespan=lifespan,
)


# Import and include routers
from api.routes import fourth_down  # noqa: E402

app.include_router(fourth_down.router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": app_state.fourth_down_predictor is not None,
    }
