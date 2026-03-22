# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

This is a **planning workspace** containing implementation plans for a unified-etl platform. There is no source code yet — only markdown plan documents and generated images. The plans describe a Python monorepo that predicts optimal NFL 4th-down decisions using ML, with dual-backend execution (Snowflake for training, DuckDB for inference).

## Plan Documents and Execution Order

Plans must be implemented in dependency order. The consistency review (`2026-03-21-consistency-review.md`) documents cross-plan conflicts that must be resolved during implementation.

| Order | File | Description |
|-------|------|-------------|
| 1 | `2026-03-21-pydantic-schemas.md` (Plan 9) | Central `schemas` package — implement FIRST, all other plans depend on it |
| 2 | `2026-03-21-unified-etl-framework.md` (Plan 1) | Core monorepo: Ibis transforms, backends, FastAPI, Streamlit |
| 3 | `2026-03-21-nfl-fourth-down-ml-pipeline.md` (Plan 2) | NFL data ingestion, XGBoost model training, inference API |
| 4 | `2026-03-21-feature-selection-ui.md` (Plan 3) | Feature importance (SHAP, Gini, permutation), interactive selection |
| 5 | `2026-03-21-data-quality-validation-ui.md` (Plan 4) | Schema validation, drift detection, business rules |
| 6 | `2026-03-21-statistical-interpretation-engine.md` (Plan 5) | Plain-English interpretation of all statistical outputs |
| 7 | `2026-03-21-webgpu-llm-advisor.md` (Plan 6) | In-browser Qwen 2.5 1.5B via WebGPU (replaces Plan 5 Task 6) |
| 8 | `2026-03-21-model-monitoring-dashboard.md` (Plan 7) | Performance tracking, PSI, drift detection, retrain advisor |
| 9 | `2026-03-21-docker-containerization.md` (Plan 8) | Docker Compose orchestration for full stack |
| 10 | `2026-03-21-consistency-review.md` | Meta-review: identifies breaking/inconsistent issues across all plans |

## Key Architectural Decisions (From Plans)

- **Dual-backend via Ibis:** Transforms defined once as Ibis expressions, executed on Snowflake (batch/training) or DuckDB (real-time inference)
- **Monorepo with `uv` workspaces:** Packages at `packages/` — `transforms`, `backends`, `schemas`, `api`, `dashboard`, `nfl`, `ml`, `data_quality`, `monitoring`
- **Pydantic v2 strict mode** for all data contracts (Plan 9); no package defines its own models
- **Validate at boundaries, trust internally** — Pydantic validation at system edges only
- **Tech stack:** Python 3.11+, Ibis 9.x, DuckDB, Snowflake, FastAPI, Streamlit, XGBoost, scikit-learn, SHAP, Plotly, Docker Compose, Caddy

## Working With These Plans

- Every plan starts with `> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans` — follow this directive when implementing
- Plans are self-contained with task breakdowns, file lists, test specifications, and acceptance criteria
- The consistency review identifies breaking issues (monitoring package has two incompatible designs, plan numbering gaps) — read it before implementing Plans 5-8
- Plan 6 (WebGPU) supersedes Plan 5 Task 6 (Anthropic API Q&A)
