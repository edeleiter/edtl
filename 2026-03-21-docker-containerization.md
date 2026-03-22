# Docker Containerization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Containerize every component of the unified-etl platform — FastAPI inference API, Streamlit dashboard, training pipeline, validation harness, and monitoring — into Docker images with a docker-compose orchestration that spins up the full stack with a single command.

**Architecture:** Five service containers (API, dashboard, training worker, validation runner, monitoring data generator) plus two infrastructure containers (DuckDB doesn't need one — it's embedded, but we add a reverse proxy and a shared volume for Parquet data). Each service gets its own Dockerfile that installs the monorepo's shared packages. Docker Compose wires everything together with health checks, depends_on ordering, shared volumes for prediction logs and reference data, and environment variable configuration. A `.env.example` documents all configuration.

**Tech Stack:**
- **Containerization:** Docker, multi-stage builds
- **Orchestration:** Docker Compose v2
- **Base image:** python:3.11-slim
- **Proxy:** Caddy (lightweight, auto-HTTPS capable, simpler than nginx for this use case)
- **Volumes:** Named volumes for prediction logs, model artifacts, reference data

**Prerequisite:** Plans 1-2 (ETL Framework + NFL ML Pipeline) minimum. Plans 3-7 enhance the containerized stack but aren't required for the Docker setup to work.

---

## Design Decisions

**Why not Kubernetes?** This is a development/small-deployment platform. Docker Compose gives you single-command startup, easy local development, and simple configuration. If this grows to need multi-node orchestration, the Dockerfiles and health checks transfer directly to K8s manifests later.

**Why a shared base image?** The monorepo's packages (`transforms`, `backends`, `nfl`, `ml`, `monitoring`, `data_quality`) are interdependent. Building them once in a base stage and copying into service-specific images avoids reinstalling 500MB of dependencies in every image.

**Why Caddy over nginx?** Caddy has sane defaults (auto-HTTPS, HTTP/2, no config needed for reverse proxy), is a single binary, and the Caddyfile syntax is dramatically simpler. For a platform that routes to two backends (API on 8000, Streamlit on 8501), Caddy is ideal.

**Why named volumes over bind mounts?** Named volumes are portable across machines and don't break when the host path changes. For development, we add bind mount overrides in `docker-compose.override.yml` so you can edit code locally and see changes in the container.

---

## Extended Project Structure

```
unified-etl/
├── ... (existing monorepo structure)
│
├── docker/
│   ├── base.Dockerfile              # Shared base image with all packages
│   ├── api.Dockerfile                # FastAPI inference service
│   ├── dashboard.Dockerfile          # Streamlit dashboard
│   ├── training.Dockerfile           # Training pipeline runner
│   ├── validation.Dockerfile         # Validation + parity test runner
│   └── Caddyfile                     # Reverse proxy config
│
├── docker-compose.yml                # Production-like orchestration
├── docker-compose.override.yml       # Development overrides (bind mounts, hot reload)
├── .env.example                      # Environment variable documentation
├── .dockerignore                     # Keep images lean
│
├── scripts/
│   ├── ... (existing)
│   ├── docker-healthcheck.py         # Shared health check script
│   └── generate-monitoring-data.py   # Seed monitoring data for demo
│
└── monitoring_data/                  # Shared volume mount point
    └── .gitkeep
```

---

## Task 1: .dockerignore + Environment Configuration

**Files:**
- Create: `.dockerignore`
- Create: `.env.example`
- Create: `.env` (git-ignored, copied from .env.example)

**Step 1: Create .dockerignore**

```
# .dockerignore
# Keep images lean — exclude everything not needed at runtime

# Version control
.git
.gitignore

# Python artifacts
__pycache__
*.pyc
*.pyo
*.egg-info
dist/
build/
.eggs/

# Virtual environments
.venv
venv
.uv

# IDE
.vscode
.idea
*.swp
*.swo

# Test artifacts
.pytest_cache
.coverage
htmlcov/
.mypy_cache

# Documentation
docs/plans/
*.md
!README.md

# Docker (don't recursively copy docker context)
docker-compose*.yml

# OS
.DS_Store
Thumbs.db

# Large data files (mounted as volumes instead)
monitoring_data/
reference_data/
models/
data/
notebooks/

# Development overrides
docker-compose.override.yml
```

**Step 2: Create .env.example**

```bash
# .env.example
# Copy to .env and fill in values:
#   cp .env.example .env

# ─── Snowflake (optional — only needed for Snowflake backend) ───
SNOWFLAKE_ACCOUNT=
SNOWFLAKE_USER=
SNOWFLAKE_PASSWORD=
SNOWFLAKE_DATABASE=
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_WAREHOUSE=
SNOWFLAKE_ROLE=

# ─── DuckDB ───
DUCKDB_DATABASE=:memory:
DUCKDB_THREADS=4

# ─── API Service ───
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2
MODEL_DIR=/app/models/latest
REFERENCE_DATA_DIR=/app/reference_data
MONITORING_DATA_DIR=/app/monitoring_data/predictions

# ─── Dashboard ───
STREAMLIT_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# ─── Training ───
TRAIN_SEASONS=2019,2020,2021,2022
TEST_SEASONS=2023
CACHE_DATA=true
DATA_DIR=/app/data/pbp

# ─── Reverse Proxy ───
PROXY_PORT=80
PROXY_DOMAIN=localhost

# ─── Monitoring ───
MONITORING_ALERT_ACCURACY_WARNING=0.70
MONITORING_ALERT_ACCURACY_CRITICAL=0.60
MONITORING_ALERT_LATENCY_P95_WARNING=100
MONITORING_ALERT_LATENCY_P95_CRITICAL=500
```

**Step 3: Add .env to .gitignore**

```bash
echo ".env" >> .gitignore
```

**Step 4: Commit**

```bash
git add .dockerignore .env.example .gitignore
git commit -m "chore: add .dockerignore and environment configuration"
```

---

## Task 2: Shared Base Image

The base image installs all monorepo packages once. Service images copy from this stage.

**Files:**
- Create: `docker/base.Dockerfile`

**Step 1: Implement the base Dockerfile**

```dockerfile
# docker/base.Dockerfile
# Shared base image with all monorepo packages installed.
# Service-specific Dockerfiles use this as a build stage.
#
# Build: docker build -f docker/base.Dockerfile -t unified-etl-base .

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy workspace configuration first (cache layer)
COPY pyproject.toml uv.lock* ./

# Copy all package pyproject.toml files (cache layer — dependencies change less often)
COPY packages/transforms/pyproject.toml packages/transforms/pyproject.toml
COPY packages/backends/pyproject.toml packages/backends/pyproject.toml
COPY packages/api/pyproject.toml packages/api/pyproject.toml
COPY packages/dashboard/pyproject.toml packages/dashboard/pyproject.toml
COPY packages/nfl/pyproject.toml packages/nfl/pyproject.toml
COPY packages/ml/pyproject.toml packages/ml/pyproject.toml
COPY packages/data_quality/pyproject.toml packages/data_quality/pyproject.toml
COPY packages/monitoring/pyproject.toml packages/monitoring/pyproject.toml
COPY validation/pyproject.toml validation/pyproject.toml

# Install dependencies only (no source yet — cache-friendly)
RUN uv sync --no-install-workspace --frozen 2>/dev/null || uv sync --no-install-workspace

# Now copy all source code
COPY packages/ packages/
COPY validation/ validation/
COPY scripts/ scripts/
COPY snowflake/ snowflake/

# Install workspace packages in development mode
RUN uv sync --frozen 2>/dev/null || uv sync

# Create directories for runtime data (will be mounted as volumes)
RUN mkdir -p /app/models /app/reference_data /app/monitoring_data /app/data

# Default environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
```

**Step 2: Test the base image builds**

```bash
docker build -f docker/base.Dockerfile -t unified-etl-base .
```

Expected: Successful build.

**Step 3: Commit**

```bash
git add docker/base.Dockerfile
git commit -m "chore(docker): add shared base image with all monorepo packages"
```

---

## Task 3: API Service Dockerfile

**Files:**
- Create: `docker/api.Dockerfile`
- Create: `scripts/docker-healthcheck.py`

**Step 1: Implement the API Dockerfile**

```dockerfile
# docker/api.Dockerfile
# FastAPI inference service.
#
# Build:  docker build -f docker/api.Dockerfile -t unified-etl-api .
# Run:    docker run -p 8000:8000 -v models:/app/models unified-etl-api

FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy workspace config
COPY pyproject.toml uv.lock* ./
COPY packages/transforms/pyproject.toml packages/transforms/pyproject.toml
COPY packages/backends/pyproject.toml packages/backends/pyproject.toml
COPY packages/api/pyproject.toml packages/api/pyproject.toml
COPY packages/nfl/pyproject.toml packages/nfl/pyproject.toml
COPY packages/ml/pyproject.toml packages/ml/pyproject.toml
COPY packages/monitoring/pyproject.toml packages/monitoring/pyproject.toml
COPY packages/data_quality/pyproject.toml packages/data_quality/pyproject.toml
COPY validation/pyproject.toml validation/pyproject.toml

RUN uv sync --no-install-workspace --frozen 2>/dev/null || uv sync --no-install-workspace

# Copy source
COPY packages/ packages/
COPY validation/ validation/
COPY scripts/ scripts/

RUN uv sync --frozen 2>/dev/null || uv sync

# Create runtime directories
RUN mkdir -p /app/models /app/reference_data /app/monitoring_data/predictions

# Health check
COPY scripts/docker-healthcheck.py /app/scripts/
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD uv run python scripts/docker-healthcheck.py http://localhost:${API_PORT:-8000}/health

# Runtime configuration
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV API_WORKERS=2
ENV MODEL_DIR=/app/models/latest
ENV REFERENCE_DATA_DIR=/app/reference_data
ENV MONITORING_DATA_DIR=/app/monitoring_data/predictions

EXPOSE ${API_PORT:-8000}

CMD ["uv", "run", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]
```

**Step 2: Implement the health check script**

```python
# scripts/docker-healthcheck.py
"""Docker health check script.

Usage: python scripts/docker-healthcheck.py <url>
Exit code 0 = healthy, 1 = unhealthy.
"""

import sys
import urllib.request
import urllib.error


def check(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/health"
    sys.exit(0 if check(url) else 1)
```

**Step 3: Commit**

```bash
git add docker/api.Dockerfile scripts/docker-healthcheck.py
git commit -m "chore(docker): add API service Dockerfile with health check"
```

---

## Task 4: Dashboard Service Dockerfile

**Files:**
- Create: `docker/dashboard.Dockerfile`

**Step 1: Implement the dashboard Dockerfile**

```dockerfile
# docker/dashboard.Dockerfile
# Streamlit dashboard (all 16 pages).
#
# Build:  docker build -f docker/dashboard.Dockerfile -t unified-etl-dashboard .
# Run:    docker run -p 8501:8501 unified-etl-dashboard

FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY packages/transforms/pyproject.toml packages/transforms/pyproject.toml
COPY packages/backends/pyproject.toml packages/backends/pyproject.toml
COPY packages/api/pyproject.toml packages/api/pyproject.toml
COPY packages/dashboard/pyproject.toml packages/dashboard/pyproject.toml
COPY packages/nfl/pyproject.toml packages/nfl/pyproject.toml
COPY packages/ml/pyproject.toml packages/ml/pyproject.toml
COPY packages/monitoring/pyproject.toml packages/monitoring/pyproject.toml
COPY packages/data_quality/pyproject.toml packages/data_quality/pyproject.toml
COPY validation/pyproject.toml validation/pyproject.toml

RUN uv sync --no-install-workspace --frozen 2>/dev/null || uv sync --no-install-workspace

COPY packages/ packages/
COPY validation/ validation/

RUN uv sync --frozen 2>/dev/null || uv sync

RUN mkdir -p /app/models /app/reference_data /app/monitoring_data /app/data

# Streamlit config — disable CORS/XSRF for Docker networking
RUN mkdir -p /root/.streamlit
RUN echo '\
[server]\n\
headless = true\n\
port = 8501\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
fileWatcherType = "none"\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /root/.streamlit/config.toml

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", \
     "packages/dashboard/src/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
```

**Step 2: Commit**

```bash
git add docker/dashboard.Dockerfile
git commit -m "chore(docker): add Streamlit dashboard Dockerfile"
```

---

## Task 5: Training Pipeline Dockerfile

**Files:**
- Create: `docker/training.Dockerfile`

**Step 1: Implement the training Dockerfile**

```dockerfile
# docker/training.Dockerfile
# Training pipeline — downloads data, engineers features, trains model.
#
# Build:  docker build -f docker/training.Dockerfile -t unified-etl-training .
# Run:    docker run -v models:/app/models -v data:/app/data unified-etl-training
#
# This is a "run once" container, not a long-running service. It exits
# after training completes. Use docker-compose run or a job scheduler.

FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY packages/transforms/pyproject.toml packages/transforms/pyproject.toml
COPY packages/backends/pyproject.toml packages/backends/pyproject.toml
COPY packages/nfl/pyproject.toml packages/nfl/pyproject.toml
COPY packages/ml/pyproject.toml packages/ml/pyproject.toml
COPY packages/data_quality/pyproject.toml packages/data_quality/pyproject.toml
COPY packages/monitoring/pyproject.toml packages/monitoring/pyproject.toml
COPY packages/api/pyproject.toml packages/api/pyproject.toml
COPY packages/dashboard/pyproject.toml packages/dashboard/pyproject.toml
COPY validation/pyproject.toml validation/pyproject.toml

RUN uv sync --no-install-workspace --frozen 2>/dev/null || uv sync --no-install-workspace

COPY packages/ packages/
COPY validation/ validation/
COPY scripts/ scripts/

RUN uv sync --frozen 2>/dev/null || uv sync

RUN mkdir -p /app/models /app/data/pbp /app/reference_data

ENV PYTHONUNBUFFERED=1

# Default: train on 2019-2022, test on 2023, cache data, output to /app/models/latest
CMD ["uv", "run", "python", "scripts/train_fourth_down_model.py", \
     "--train-seasons", "2019", "2020", "2021", "2022", \
     "--test-seasons", "2023", \
     "--data-dir", "/app/data/pbp", \
     "--output-dir", "/app/models/latest", \
     "--cache-data"]
```

**Step 2: Commit**

```bash
git add docker/training.Dockerfile
git commit -m "chore(docker): add training pipeline Dockerfile"
```

---

## Task 6: Validation Runner Dockerfile

**Files:**
- Create: `docker/validation.Dockerfile`

**Step 1: Implement the validation Dockerfile**

```dockerfile
# docker/validation.Dockerfile
# Runs the full test suite including cross-backend parity tests.
#
# Build:  docker build -f docker/validation.Dockerfile -t unified-etl-validation .
# Run:    docker run unified-etl-validation
#
# Exit code 0 = all tests pass, non-zero = failures.

FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY packages/ packages/
COPY validation/ validation/
COPY tests/ tests/

RUN uv sync --frozen 2>/dev/null || uv sync

ENV PYTHONUNBUFFERED=1

# Run all tests except those requiring Snowflake credentials
CMD ["uv", "run", "pytest", "tests/", "-v", "--tb=short", \
     "-k", "not snowflake"]
```

**Step 2: Commit**

```bash
git add docker/validation.Dockerfile
git commit -m "chore(docker): add validation runner Dockerfile"
```

---

## Task 7: Caddy Reverse Proxy

**Files:**
- Create: `docker/Caddyfile`

**Step 1: Implement the Caddyfile**

```
# docker/Caddyfile
# Reverse proxy routing:
#   /          → Streamlit dashboard (port 8501)
#   /api/*     → FastAPI inference service (port 8000)
#   /health    → API health check (pass-through)
#
# Caddy handles TLS automatically in production.
# In development (localhost), it serves plain HTTP.

{
    # Disable admin API in production
    admin off
}

:{$PROXY_PORT:80} {
    # API routes
    handle /health {
        reverse_proxy api:8000
    }

    handle /inference/* {
        reverse_proxy api:8000
    }

    handle /fourth-down/* {
        reverse_proxy api:8000
    }

    handle /batch/* {
        reverse_proxy api:8000
    }

    handle /docs* {
        reverse_proxy api:8000
    }

    handle /openapi.json {
        reverse_proxy api:8000
    }

    # Everything else → Streamlit dashboard
    handle {
        reverse_proxy dashboard:8501
    }

    # Logging
    log {
        output stdout
        format json
    }
}
```

**Step 2: Commit**

```bash
git add docker/Caddyfile
git commit -m "chore(docker): add Caddy reverse proxy configuration"
```

---

## Task 8: Docker Compose — Production Configuration

**Files:**
- Create: `docker-compose.yml`

**Step 1: Implement docker-compose.yml**

```yaml
# docker-compose.yml
# Production-like orchestration for the unified-etl platform.
#
# Usage:
#   docker compose up -d              # Start all services
#   docker compose up -d api dashboard # Start specific services
#   docker compose run training        # Run training pipeline
#   docker compose run validation      # Run test suite
#   docker compose logs -f api         # Follow API logs
#   docker compose down                # Stop everything

name: unified-etl

services:
  # ─── FastAPI Inference Service ───
  api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    container_name: unified-etl-api
    restart: unless-stopped
    env_file: .env
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - MODEL_DIR=/app/models/latest
      - REFERENCE_DATA_DIR=/app/reference_data
      - MONITORING_DATA_DIR=/app/monitoring_data/predictions
    volumes:
      - models:/app/models
      - reference_data:/app/reference_data
      - monitoring_data:/app/monitoring_data
    ports:
      - "${API_PORT:-8000}:8000"
    healthcheck:
      test: ["CMD", "uv", "run", "python", "scripts/docker-healthcheck.py", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 15s

  # ─── Streamlit Dashboard ───
  dashboard:
    build:
      context: .
      dockerfile: docker/dashboard.Dockerfile
    container_name: unified-etl-dashboard
    restart: unless-stopped
    env_file: .env
    environment:
      - API_URL=http://api:8000
    volumes:
      - models:/app/models:ro
      - reference_data:/app/reference_data:ro
      - monitoring_data:/app/monitoring_data:ro
      - training_data:/app/data:ro
    ports:
      - "${STREAMLIT_PORT:-8501}:8501"
    depends_on:
      api:
        condition: service_healthy

  # ─── Reverse Proxy ───
  proxy:
    image: caddy:2-alpine
    container_name: unified-etl-proxy
    restart: unless-stopped
    ports:
      - "${PROXY_PORT:-80}:80"
      - "443:443"
    volumes:
      - ./docker/Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    depends_on:
      - api
      - dashboard

  # ─── Training Pipeline (run-once job) ───
  training:
    build:
      context: .
      dockerfile: docker/training.Dockerfile
    container_name: unified-etl-training
    env_file: .env
    environment:
      - DATA_DIR=/app/data/pbp
    volumes:
      - models:/app/models
      - reference_data:/app/reference_data
      - training_data:/app/data
    profiles:
      - training    # Only starts with: docker compose --profile training run training

  # ─── Validation Runner (run-once job) ───
  validation:
    build:
      context: .
      dockerfile: docker/validation.Dockerfile
    container_name: unified-etl-validation
    env_file: .env
    profiles:
      - ci          # Only starts with: docker compose --profile ci run validation

volumes:
  # Persistent data
  models:
    driver: local
  reference_data:
    driver: local
  monitoring_data:
    driver: local
  training_data:
    driver: local

  # Caddy internal
  caddy_data:
    driver: local
  caddy_config:
    driver: local
```

**Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "chore(docker): add docker-compose orchestration"
```

---

## Task 9: Docker Compose — Development Overrides

**Files:**
- Create: `docker-compose.override.yml`

**Step 1: Implement development overrides**

```yaml
# docker-compose.override.yml
# Development overrides — auto-loaded by docker compose.
#
# Adds:
#   - Bind mounts for live code editing (hot reload)
#   - Debug ports
#   - Verbose logging
#
# To use production config only:
#   docker compose -f docker-compose.yml up

services:
  api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    volumes:
      # Bind mount source for hot reload
      - ./packages/transforms/src:/app/packages/transforms/src:ro
      - ./packages/backends/src:/app/packages/backends/src:ro
      - ./packages/api/src:/app/packages/api/src:ro
      - ./packages/nfl/src:/app/packages/nfl/src:ro
      - ./packages/ml/src:/app/packages/ml/src:ro
      - ./packages/monitoring/src:/app/packages/monitoring/src:ro
      - ./packages/data_quality/src:/app/packages/data_quality/src:ro
      # Data volumes (bind mount for easy access)
      - ./models:/app/models
      - ./reference_data:/app/reference_data
      - ./monitoring_data:/app/monitoring_data
    command: >
      uv run uvicorn api.main:app
      --host 0.0.0.0 --port 8000
      --reload
      --reload-dir /app/packages/api/src
      --reload-dir /app/packages/transforms/src
      --reload-dir /app/packages/ml/src

  dashboard:
    volumes:
      - ./packages/dashboard/src:/app/packages/dashboard/src:ro
      - ./packages/transforms/src:/app/packages/transforms/src:ro
      - ./packages/ml/src:/app/packages/ml/src:ro
      - ./packages/monitoring/src:/app/packages/monitoring/src:ro
      - ./packages/data_quality/src:/app/packages/data_quality/src:ro
      - ./models:/app/models:ro
      - ./reference_data:/app/reference_data:ro
      - ./monitoring_data:/app/monitoring_data:ro
      - ./data:/app/data:ro
    environment:
      - STREAMLIT_SERVER_FILE_WATCHER_TYPE=auto  # Enable hot reload in dev
```

**Step 2: Commit**

```bash
git add docker-compose.override.yml
git commit -m "chore(docker): add development overrides with hot reload"
```

---

## Task 10: Monitoring Data Generator Script

This script generates realistic synthetic prediction logs so you can demo the monitoring dashboard without running real inference.

**Files:**
- Create: `scripts/generate-monitoring-data.py`

**Step 1: Implement the generator**

```python
# scripts/generate-monitoring-data.py
"""Generate synthetic prediction logs for monitoring dashboard demos.

Creates realistic 4th-down prediction data with:
- Gradual accuracy decay over time (simulates model degradation)
- Calibration drift
- Prediction distribution shifts
- Latency variations

Usage:
    uv run python scripts/generate-monitoring-data.py
    uv run python scripts/generate-monitoring-data.py --days 30 --predictions-per-day 100
    uv run python scripts/generate-monitoring-data.py --scenario decay
"""

import argparse
import datetime
import uuid

import numpy as np
import pandas as pd

from monitoring.prediction_log import log_prediction, log_batch, PredictionRecord
from monitoring.config import MonitoringConfig


def generate_game_state(rng: np.random.Generator) -> dict:
    """Generate a random but plausible game state."""
    qtr = rng.choice([1, 2, 3, 4], p=[0.15, 0.25, 0.25, 0.35])
    return {
        "ydstogo": int(rng.integers(1, 15)),
        "yardline_100": int(rng.integers(1, 99)),
        "score_differential": int(rng.integers(-21, 22)),
        "half_seconds_remaining": int(rng.integers(0, 1800)),
        "game_seconds_remaining": int(rng.integers(0, 3600)),
        "quarter_seconds_remaining": int(rng.integers(0, 900)),
        "qtr": int(qtr),
        "goal_to_go": int(rng.choice([0, 1], p=[0.85, 0.15])),
        "wp": round(float(rng.uniform(0.05, 0.95)), 3),
    }


def generate_stable(
    n_days: int = 14,
    per_day: int = 50,
    base_accuracy: float = 0.82,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Generate stable performance data (no decay)."""
    rng = rng or np.random.default_rng(42)
    records = []
    now = datetime.datetime.now(datetime.timezone.utc)

    for day in range(n_days):
        for i in range(per_day):
            ts = now - datetime.timedelta(days=n_days - day, hours=rng.integers(0, 24), minutes=rng.integers(0, 60))
            gs = generate_game_state(rng)

            prediction = rng.choice(["go_for_it", "punt", "field_goal"], p=[0.30, 0.45, 0.25])
            is_correct = bool(rng.binomial(1, base_accuracy))
            actual = prediction if is_correct else rng.choice([c for c in ["go_for_it", "punt", "field_goal"] if c != prediction])

            confidence = float(rng.uniform(0.50, 0.92)) if is_correct else float(rng.uniform(0.35, 0.70))
            probs = _generate_probs(prediction, confidence, rng)

            records.append({
                "request_id": f"req_{uuid.uuid4().hex[:12]}",
                "timestamp": ts,
                "model_version": "v1",
                "prediction": prediction,
                "confidence": confidence,
                "latency_ms": float(rng.exponential(12) + 3),
                "actual_decision": actual,
                "was_correct": is_correct,
                "epa_result": float(rng.normal(0.5 if is_correct else -0.5, 1.0)),
                **{f"input_{k}": v for k, v in gs.items()},
                **{f"prob_{k}": v for k, v in probs.items()},
            })

    return records


def generate_decay(
    n_days: int = 30,
    per_day: int = 50,
    start_accuracy: float = 0.85,
    end_accuracy: float = 0.55,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Generate data with gradual accuracy decay."""
    rng = rng or np.random.default_rng(42)
    records = []
    now = datetime.datetime.now(datetime.timezone.utc)

    for day in range(n_days):
        # Linear accuracy decay
        day_accuracy = start_accuracy + (end_accuracy - start_accuracy) * (day / n_days)

        for i in range(per_day):
            ts = now - datetime.timedelta(days=n_days - day, hours=rng.integers(0, 24), minutes=rng.integers(0, 60))
            gs = generate_game_state(rng)

            prediction = rng.choice(["go_for_it", "punt", "field_goal"], p=[0.30, 0.45, 0.25])
            is_correct = bool(rng.binomial(1, day_accuracy))
            actual = prediction if is_correct else rng.choice([c for c in ["go_for_it", "punt", "field_goal"] if c != prediction])

            # Confidence stays high even as accuracy drops (overconfidence)
            confidence = float(rng.uniform(0.60, 0.92))
            probs = _generate_probs(prediction, confidence, rng)

            records.append({
                "request_id": f"req_{uuid.uuid4().hex[:12]}",
                "timestamp": ts,
                "model_version": "v1",
                "prediction": prediction,
                "confidence": confidence,
                "latency_ms": float(rng.exponential(12 + day * 0.5) + 3),  # Latency creeps up too
                "actual_decision": actual,
                "was_correct": is_correct,
                "epa_result": float(rng.normal(0.5 if is_correct else -0.5, 1.0)),
                **{f"input_{k}": v for k, v in gs.items()},
                **{f"prob_{k}": v for k, v in probs.items()},
            })

    return records


def _generate_probs(prediction: str, confidence: float, rng) -> dict:
    """Generate class probabilities consistent with prediction and confidence."""
    classes = ["go_for_it", "punt", "field_goal"]
    remaining = 1.0 - confidence
    other_probs = rng.dirichlet([1, 1]) * remaining
    probs = {}
    j = 0
    for c in classes:
        if c == prediction:
            probs[c] = confidence
        else:
            probs[c] = float(other_probs[j])
            j += 1
    return probs


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic monitoring data")
    parser.add_argument("--scenario", choices=["stable", "decay"], default="decay")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--predictions-per-day", type=int, default=50)
    parser.add_argument("--output-dir", default="monitoring_data/predictions")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    if args.scenario == "stable":
        records = generate_stable(args.days, args.predictions_per_day, rng=rng)
    else:
        records = generate_decay(args.days, args.predictions_per_day, rng=rng)

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    from pathlib import Path
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_path = output_path / "predictions.parquet"
    df.to_parquet(parquet_path, index=False)

    print(f"Generated {len(df):,} predictions over {args.days} days")
    print(f"Scenario: {args.scenario}")
    print(f"Saved to: {parquet_path}")
    print(f"\nAccuracy: {df['was_correct'].mean():.1%}")
    print(f"Prediction distribution: {df['prediction'].value_counts(normalize=True).to_dict()}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/generate-monitoring-data.py
git commit -m "feat: add synthetic monitoring data generator for demos"
```

---

## Task 11: Makefile / Convenience Commands

**Files:**
- Create: `Makefile`

**Step 1: Implement the Makefile**

```makefile
# Makefile
# Convenience commands for the unified-etl platform.
#
# Usage:
#   make up              Start API + Dashboard + Proxy
#   make down            Stop everything
#   make train           Run training pipeline
#   make test            Run test suite in container
#   make demo            Generate demo data + start everything
#   make logs            Follow all service logs
#   make clean           Remove all containers, volumes, images

.PHONY: up down train test demo logs clean build status \
        seed-monitoring api dashboard proxy shell-api shell-dashboard

# ─── Core Lifecycle ───

build:
	docker compose build

up:
	docker compose up -d api dashboard proxy
	@echo "✅ Platform is running:"
	@echo "   Dashboard: http://localhost:$${PROXY_PORT:-80}"
	@echo "   API:       http://localhost:$${API_PORT:-8000}"
	@echo "   API Docs:  http://localhost:$${API_PORT:-8000}/docs"

down:
	docker compose down

status:
	docker compose ps

logs:
	docker compose logs -f

# ─── Individual Services ───

api:
	docker compose up -d api

dashboard:
	docker compose up -d dashboard

proxy:
	docker compose up -d proxy

# ─── Jobs ───

train:
	@echo "🏋️ Running training pipeline..."
	docker compose --profile training run --rm training
	@echo "✅ Training complete. Restart API to load new model:"
	@echo "   docker compose restart api"

test:
	@echo "🧪 Running test suite..."
	docker compose --profile ci run --rm validation

# ─── Demo / Development ───

seed-monitoring:
	@echo "📊 Generating synthetic monitoring data..."
	docker compose run --rm -e PYTHONPATH=/app/packages/monitoring/src:/app/packages/transforms/src:/app/packages/backends/src \
		api uv run python scripts/generate-monitoring-data.py --scenario decay --days 30
	@echo "✅ Monitoring data generated. Refresh the dashboard to see it."

demo: build seed-monitoring up
	@echo ""
	@echo "🎉 Demo is running!"
	@echo "   Dashboard: http://localhost:$${PROXY_PORT:-80}"
	@echo "   Monitoring: http://localhost:$${PROXY_PORT:-80} → Model Health page"

# ─── Debug / Shell Access ───

shell-api:
	docker compose exec api bash

shell-dashboard:
	docker compose exec dashboard bash

# ─── Cleanup ───

clean:
	docker compose down -v --rmi local --remove-orphans
	@echo "🧹 Cleaned up all containers, volumes, and images."
```

**Step 2: Commit**

```bash
git add Makefile
git commit -m "chore: add Makefile with convenience commands"
```

---

## Task 12: Update README with Docker Instructions

**Files:**
- Modify: `README.md`

**Step 1: Add Docker section to README**

Append to `README.md`:

```markdown

## Docker

### Quick Start (Demo)

```bash
# Clone and start everything with demo data
cp .env.example .env
make demo
```

This builds all containers, generates 30 days of synthetic monitoring data, and starts the API, dashboard, and proxy. Visit http://localhost to see the full platform.

### Production Deployment

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your Snowflake credentials, etc.

# 2. Train the model
make train

# 3. Start services
make up

# 4. Run tests
make test
```

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Caddy Proxy (:80)                 │
│              /api/* → API    /* → Dashboard          │
└────────┬──────────────────────────────┬──────────────┘
         │                              │
    ┌────▼─────┐                 ┌──────▼──────┐
    │  FastAPI  │                │  Streamlit   │
    │  (:8000)  │                │   (:8501)    │
    │           │                │  16 pages    │
    └────┬──────┘                └──────┬───────┘
         │                              │
    ┌────▼──────────────────────────────▼───────┐
    │          Shared Volumes (Parquet)          │
    │  models/  reference_data/  monitoring_data/│
    └───────────────────────────────────────────┘
```

### Available Commands

| Command | Description |
|---------|-------------|
| `make up` | Start API + Dashboard + Proxy |
| `make down` | Stop everything |
| `make train` | Run training pipeline (downloads NFL data, trains model) |
| `make test` | Run test suite in container |
| `make demo` | Build + seed demo data + start everything |
| `make logs` | Follow all service logs |
| `make status` | Show running containers |
| `make seed-monitoring` | Generate synthetic monitoring data |
| `make shell-api` | Shell into the API container |
| `make clean` | Remove all containers, volumes, and images |

### Services

| Service | Port | Description |
|---------|------|-------------|
| `proxy` | 80 | Caddy reverse proxy (routes to API/Dashboard) |
| `api` | 8000 | FastAPI inference service (DuckDB backend) |
| `dashboard` | 8501 | Streamlit dashboard (16 pages) |
| `training` | — | One-shot training pipeline (exits after completion) |
| `validation` | — | One-shot test runner (exits with pass/fail) |

### Volumes

| Volume | Purpose |
|--------|---------|
| `models` | Trained model artifacts (model.joblib, metadata.json) |
| `reference_data` | Parquet reference tables for inference |
| `monitoring_data` | Prediction logs for monitoring dashboard |
| `training_data` | Cached PBP Parquet files (avoid re-downloading) |
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Docker deployment instructions to README"
```

---

## Summary: Run Order

| Task | What it builds | Key files |
|------|---------------|-----------|
| 1 | .dockerignore + env config | `.dockerignore`, `.env.example` |
| 2 | Shared base image | `docker/base.Dockerfile` |
| 3 | API service container | `docker/api.Dockerfile` |
| 4 | Dashboard container | `docker/dashboard.Dockerfile` |
| 5 | Training pipeline container | `docker/training.Dockerfile` |
| 6 | Validation runner container | `docker/validation.Dockerfile` |
| 7 | Caddy reverse proxy | `docker/Caddyfile` |
| 8 | Docker Compose (production) | `docker-compose.yml` |
| 9 | Docker Compose (dev overrides) | `docker-compose.override.yml` |
| 10 | Monitoring data generator | `scripts/generate-monitoring-data.py` |
| 11 | Makefile convenience commands | `Makefile` |
| 12 | README Docker docs | `README.md` |

## End-to-End Workflow

```bash
# First time setup
git clone <repo>
cd unified-etl
cp .env.example .env

# Option A: Full demo (fastest)
make demo
# → Builds images, generates monitoring data, starts everything
# → Visit http://localhost

# Option B: Train your own model
make train
# → Downloads NFL PBP data, trains XGBoost, saves model
make up
# → Starts API + Dashboard + Proxy
# → Visit http://localhost

# Day-to-day development
make up                    # Start services
# Edit code locally — hot reload active in dev mode
make test                  # Run tests in container
make logs                  # Debug issues
make down                  # Stop when done

# CI/CD
make test                  # Exit code 0 = pass
docker compose build       # Build production images
docker compose push        # Push to registry (add registry config)
```
