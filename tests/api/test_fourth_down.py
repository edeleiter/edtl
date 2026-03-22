"""Tests for the fourth-down API endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(saved_model_dir, monkeypatch):
    """FastAPI test client with a loaded model."""
    monkeypatch.setenv("MODEL_DIR", str(saved_model_dir))
    # Import after setting env var
    from api.main import app
    from api.dependencies import app_state
    app_state.startup()
    yield TestClient(app)
    app_state.shutdown()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_single(client, sample_game_state):
    resp = client.post("/fourth-down/predict", json=sample_game_state)
    assert resp.status_code == 200
    data = resp.json()
    assert data["recommendation"] in ("go_for_it", "punt", "field_goal")
    assert len(data["probabilities"]) == 3
    total = sum(data["probabilities"].values())
    assert abs(total - 1.0) < 0.01


def test_predict_batch(client, sample_game_state):
    batch = {"game_states": [sample_game_state, {**sample_game_state, "ydstogo": 10}]}
    resp = client.post("/fourth-down/predict-batch", json=batch)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 2


def test_predict_validates_input(client):
    bad = {"ydstogo": 0, "yardline_100": 35}  # ydstogo < 1
    resp = client.post("/fourth-down/predict", json=bad)
    assert resp.status_code == 422  # Validation error


def test_health_no_model(monkeypatch):
    """Without MODEL_DIR, model is not loaded."""
    monkeypatch.delenv("MODEL_DIR", raising=False)
    from api.main import app
    from api.dependencies import app_state
    app_state.fourth_down_predictor = None
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.json()["model_loaded"] is False
