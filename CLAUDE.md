# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

This is a **planning workspace** containing implementation plans for a unified-etl platform. There is no source code yet — only markdown plan documents and generated images. The plans describe a Python monorepo that predicts optimal NFL 4th-down decisions using ML, with dual-backend execution (Snowflake for training, DuckDB for inference).

## Plan Documents and Execution Order

Plans must be implemented in a phased dependency order. Plan 9 (schemas) uses a two-wave strategy — Wave 1 alongside Plan 1, Wave 2 after the domain plans are built.

| Phase | Plans | Description |
|-------|-------|-------------|
| 1 | Plan 9 Wave 1 (Tasks 1-3) + Plan 1 | Foundation: schemas base + monorepo + transforms + backends |
| 2 | Plan 2 | NFL data ingestion, XGBoost training, inference API |
| 3 | Plans 3 + 4 + Plan 9 Task 4 | Feature analysis UI + data quality UI (parallelizable) |
| 4 | Plans 5 + 6 + Plan 9 Tasks 5-6 | Interpretation engine + WebGPU LLM advisor |
| 5 | Plan 7 | Model monitoring dashboard |
| 6 | Plan 8 | Docker Compose containerization |
| 7 | Integration sweep | Replace all local dataclasses with `schemas` imports |

### File Index

| File | Plan # | Tasks |
|------|--------|-------|
| `2026-03-21-pydantic-schemas.md` | Plan 9 of 10 | 7 (two-wave) |
| `2026-03-21-unified-etl-framework.md` | Plan 1 of 10 | 11 |
| `2026-03-21-nfl-fourth-down-ml-pipeline.md` | Plan 2 of 10 | 11 |
| `2026-03-21-feature-selection-ui.md` | Plan 3 of 10 | 7 |
| `2026-03-21-data-quality-validation-ui.md` | Plan 4 of 10 | 8 |
| `2026-03-21-statistical-interpretation-engine.md` | Plan 5 of 10 | 6 |
| `2026-03-21-webgpu-llm-advisor.md` | Plan 6 of 10 | 4 |
| `2026-03-21-model-monitoring-dashboard.md` | Plan 7 of 10 | 13 |
| `2026-03-21-docker-containerization.md` | Plan 8 of 10 | 12 |
| `2026-03-21-consistency-review.md` | Doc 10 of 10 | Meta-review |

**Total: 79 tasks across 9 plans.**

## Key Architectural Decisions (From Plans)

- **Dual-backend via Ibis:** Transforms defined once as Ibis expressions, executed on Snowflake (batch/training) or DuckDB (real-time inference)
- **Monorepo with `uv` workspaces:** Packages at `packages/` — `transforms`, `backends`, `schemas`, `api`, `dashboard`, `nfl`, `ml`, `data_quality`, `monitoring`
- **Pydantic v2 strict mode** for all data contracts (Plan 9); no package defines its own models
- **Validate at boundaries, trust internally** — Pydantic validation at system edges only
- **Two-wave schema strategy** — Plan 9 Wave 1 (core models) alongside Plan 1, Wave 2 (domain models) after Plans 4/5/7
- **DuckDB-only development mode** — Snowflake is optional; all features work with DuckDB when `SNOWFLAKE_ACCOUNT` is not set
- **Tech stack:** Python 3.11+, Ibis 9.x, DuckDB, Snowflake, FastAPI, Streamlit, XGBoost, scikit-learn, SHAP, Plotly, Docker Compose, Caddy

## Working With These Plans

- Every plan starts with `> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans` — follow this directive when implementing
- Plans are self-contained with task breakdowns, file lists, test specifications, and acceptance criteria
- The consistency review (Doc 10) tracks all cross-plan issues — 30 issues identified, all breaking/high issues remediated
- Plan 6 (WebGPU) supersedes Plan 5 Task 6 (Anthropic API Q&A) — Plan 5 Task 6 is marked SUPERSEDED
- Permutation importance must use held-out test data (not training data) — see Plan 3 page 05
- Monitoring `load_predictions()` uses Ibis+DuckDB lazy loading, not pd.read_parquet (prevents OOM)
- Docker builds are independent per service; multi-stage optimization is optional (see Plan 8 Production Optimization section)
