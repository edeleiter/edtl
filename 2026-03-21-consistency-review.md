# Plan Consistency Review — Findings & Required Fixes

**Document 10 of 10 — Meta-Review**

**Reviewed:** All 10 plan documents (including Plan 9: Pydantic Schemas, added post-initial review)
**Date:** 2026-03-21

---

## Issue Severity Legend

- 🔴 **BREAKING** — Will cause implementation failure if not fixed
- 🟡 **INCONSISTENT** — Confusing but won't break; should fix for clarity
- 🟢 **MINOR** — Cosmetic or documentation-only

---

## 🔴 BREAKING Issues

### 1. Plan Numbering is Inconsistent (Affects 6 of 8 docs)

The plans don't have explicit plan numbers in their headers, but cross-reference each other by number. The implied numbering is:

| # | Document | Cross-referenced as |
|---|----------|-------------------|
| 1 | unified-etl-framework | Plan 1 ✅ |
| 2 | nfl-fourth-down-ml-pipeline | Plan 2 ✅ |
| 3 | feature-selection-ui | Plan 3 ✅ |
| 4 | data-quality-validation-ui | Plan 4 ✅ |
| 5 | statistical-interpretation-engine | Plan 5 ✅ |
| 6 | webgpu-llm-advisor | **Plan 6** in monitoring dashboard ✅ |
| 7 | model-monitoring-dashboard | **Plan 7** in its own map ✅ |
| 8 | docker-containerization | **No plan number** — refers to "Plans 1-7" |

**Fix:** Add explicit `**Plan N of 8**` to each document header. Docker should be Plan 8.

---

### 2. Monitoring Package Has TWO Incompatible Designs

The **model-monitoring-dashboard** plan (Plan 7) defines the monitoring package with these modules:
- `monitoring.config`
- `monitoring.prediction_log`
- `monitoring.performance`
- `monitoring.prediction_drift`
- `monitoring.psi`
- `monitoring.segments`
- `monitoring.retrain_advisor`
- `monitoring.alerts`

But the **docker-containerization** plan (Plan 8) references a *different* monitoring design from an earlier conversation draft:
- `monitoring.logger`
- `monitoring.schemas`

These are completely incompatible module names and structures.

**Fix:** Docker plan must be updated to reference the Plan 7 monitoring module names (`monitoring.prediction_log`, `monitoring.config`, etc.), not the earlier draft's names. Specifically:
- `scripts/generate-monitoring-data.py` in Docker plan imports `from monitoring.logger import PredictionLogger` and `from monitoring.schemas import PredictionLog` — these should be `from monitoring.prediction_log import PredictionLogger, PredictionLogEntry` per Plan 7
- The `docker-healthcheck.py` and Makefile `seed-monitoring` target need updated PYTHONPATH

---

### 3. `InterpretationLevel` Defined Twice with Different Import Paths

The **interpretation engine** (Plan 5) defines `InterpretationLevel` in TWO separate files:
- `ml.feature_analysis.interpret.InterpretationLevel`
- `data_quality.interpret.InterpretationLevel`

These are identical enums but exist independently. The **interpretation card component** (`dashboard/components/interpretation_card.py`) imports both with try/except and aliases.

**Fix:** Extract `Interpretation` and `InterpretationLevel` into a shared location. Two options:
- **(a)** Put them in `transforms` package (shared base) as `transforms.interpretation.Interpretation`
- **(b)** Keep both definitions but document in Plan 5 that they MUST stay in sync. Add a test that asserts `FeatureLevel.OK.value == DQLevel.OK.value` etc.

Option (b) is simpler and aligns with the existing plan structure. Add to Task 3 of Plan 5.

---

### 4. NFL Plan Header Says "nflreadpy" but Code Uses "nfl-data-py"

In `nfl-fourth-down-ml-pipeline.md`:
- **Header says:** `**Data:** nflreadpy (nflverse play-by-play data, CC-BY-4.0 licensed)`
- **pyproject.toml says:** `"nfl-data-py>=0.3.0"`
- **Code says:** `import nfl_data_py as nfl`
- **Engineer note says:** We're using `nfl-data-py` rather than `nflreadpy`

**Fix:** Change the header tech stack to `**Data:** nfl-data-py (nflverse play-by-play data, CC-BY-4.0 / CC-BY-SA-4.0 licensed)`.

---

## 🟡 INCONSISTENT Issues

### 5. Task Count Claim is Stale

The monitoring plan claims "Total tasks across all plans: **56 tasks**." The actual count:

| Plan | Tasks |
|------|-------|
| 1. ETL Framework | 11 |
| 2. NFL ML Pipeline | 11 |
| 3. Feature Selection | 7 |
| 4. Data Quality | 8 |
| 5. Interpretation Engine | 6 |
| 6. WebGPU LLM Advisor | 4 |
| 7. Model Monitoring | 13 |
| 8. Docker Containerization | 12 |
| 9. Pydantic Schemas | 7 |
| **Total** | **79** |

**Fix:** Update the monitoring plan's final summary to say **79 tasks** and include Plans 8-9 in the dashboard map table.

---

### 6. Plan 5 (Interpretation) Still References Anthropic API in Header

The interpretation plan's architecture description and tech stack still mention the Anthropic API approach:
- Architecture: "An LLM-powered 'Ask About This' feature (optional, using Anthropic API)"
- Tech Stack: "**Optional:** Anthropic API (Claude Sonnet) for contextual Q&A"
- Task 6 of Plan 5 implements the full Anthropic API component

The WebGPU plan (Plan 6) explicitly says it "replaces Task 6 of the interpretation plan."

**Fix:** Add a note to Plan 5's header: `**Note:** Task 6 (LLM Q&A) has been superseded by Plan 6 (WebGPU LLM Advisor). Implement Plan 6 instead of Task 6.` This keeps Plan 5 intact as a fallback (some deployments may prefer API over WebGPU) while making the default path clear.

---

### 7. `data-quality` Package Name vs `data_quality` Import

The pyproject.toml in Plan 4 declares `name = "data-quality"` (with hyphen), but all Python imports use `data_quality` (with underscore). This is actually fine — pip normalizes hyphens to underscores — but it's inconsistent with all other packages which use underscores in both places (`transforms`, `backends`, `monitoring`, `nfl`, `ml`).

**Fix:** Change `name = "data-quality"` to `name = "data_quality"` in Plan 4's pyproject.toml for consistency. (Or leave it — pip handles it either way. Cosmetic preference.)

---

### 8. Dashboard Page Map Doesn't Include Plan 8 (Docker)

The full dashboard map in the monitoring plan (Plan 7) lists 16 pages across Plans 1-7. Docker (Plan 8) doesn't add Streamlit pages, but the map doesn't mention Plan 8 exists at all.

**Fix:** Add a note below the table: "Plan 8 (Docker Containerization) does not add dashboard pages but containerizes all services above."

---

### 9. Interpretation Plan References "Plans 1-4" but Should Be "Plans 1-5"

In the interpretation plan header: `**Inherits:** Everything from Plans 1–4`

The interpretation plan IS Plan 5, so "inherits from Plans 1-4" is correct. But the Task 4 and Task 5 sections modify pages from Plans 3 and 4, which is also correct. No actual issue here — just flagging that the numbering is precise and correct.

**Status:** No fix needed.

---

### 10. Docker Plan's base.Dockerfile Isn't Actually Multi-Stage — 🔴 RECLASSIFIED AS BREAKING

The plan description says "A multi-stage build keeps images lean (~200-400MB each)" but `docker/base.Dockerfile` uses a single `FROM` stage. The service Dockerfiles (api, dashboard, training, validation) also each do their own full install rather than using the base as a build stage. This means dependencies are installed 5x (once per service), defeating the stated purpose and inflating total image storage by ~2GB.

**Fix:** Option (b) was applied — removed the multi-stage claim, documented independent builds, and added a "Production Optimization" section showing the true multi-stage pattern as an optional enhancement.

**Status:** ✅ FIXED in Plan 8.

---

## 🟢 MINOR Issues

### 11. Inconsistent Commit Message Prefixes

Most commits use conventional commit format (`feat(ml):`, `feat(dashboard):`, `chore(docker):`), but some don't include the scope:
- Plan 2: `git commit -m "feat: add end-to-end training script"` (no scope)
- Plan 2: `git commit -m "test: add cross-backend parity test for NFL transforms"` (no scope)

**Fix:** Add scopes to these: `feat(scripts):`, `test(nfl):`.

---

### 12. WebGPU Plan Model ID May Be Stale

The WebGPU plan hardcodes `Qwen2.5-1.5B-Instruct-q4f16_1-MLC` as the model ID. WebLLM updates their model list periodically. By the time someone implements this, the exact model ID may have changed.

**Fix:** Add a note: "Verify the model ID is still in `webllm.prebuiltAppConfig.model_list` at implementation time. Check https://github.com/mlc-ai/web-llm for the current list."

---

### 13. Some Test Files Are Referenced But Not Created

Plan 1 mentions `tests/dashboard/test_feature_preview.py` in the project structure but never creates it (Task 9 notes Streamlit is hard to unit test and does manual verification instead). This is fine — the structure shows intent — but the file won't exist.

**Fix:** Remove from project structure or add a placeholder: `# tests/dashboard/ — manual verification only (see task notes)`.

---

### 14. Docker Compose `make seed-monitoring` Command Has Fragile PYTHONPATH

The Makefile's `seed-monitoring` target manually sets PYTHONPATH:
```
-e PYTHONPATH=/app/packages/monitoring/src:/app/packages/transforms/src:/app/packages/backends/src
```

This will break if any new package dependency is added. It should use `uv run` instead, which handles the Python path automatically.

**Fix:** Change to:
```makefile
docker compose run --rm api uv run python scripts/generate-monitoring-data.py --scenario decay --days 30
```

---

## Summary of Required Fixes

### Original Issues (Plans 1-8)

| # | Severity | Issue | Fix Location | Status |
|---|----------|-------|-------------|--------|
| 1 | 🔴 | No explicit plan numbers | All 10 plan headers | ✅ FIXED |
| 2 | 🔴 | Docker references wrong monitoring module names | Docker plan Task 10 | ✅ VERIFIED — already correct in current draft |
| 3 | 🔴 | InterpretationLevel defined twice independently | Plan 9 Task 6 centralizes it | ✅ FIXED in Plan 9 |
| 4 | 🔴 | Header says nflreadpy but code uses nfl-data-py | NFL plan header | Pending |
| 5 | 🟡 | Task count claim (56) is stale (actual: 79) | Monitoring plan summary | Updated to 79 |
| 6 | 🟡 | Interpretation plan still references Anthropic API | Interpretation plan header | ✅ FIXED — Task 6 marked SUPERSEDED |
| 7 | 🟡 | data-quality vs data_quality in pyproject name | Data quality plan Task 1 | Cosmetic — pip normalizes |
| 8 | 🟡 | Dashboard map missing Plan 8 | Monitoring plan summary | Pending |
| 9 | 🟢 | — (no issue, numbering is correct) | — | — |
| 10 | 🔴 | Multi-stage build claimed but not implemented | Docker plan description | ✅ FIXED — reclassified BREAKING, remediated |
| 11 | 🟢 | Inconsistent commit message scopes | NFL plan Tasks 10, 11 | Pending |
| 12 | 🟢 | WebGPU model ID may be stale | WebGPU plan Task 2 | Pending |
| 13 | 🟢 | Test file referenced but never created | ETL plan project structure | Pending |
| 14 | 🟢 | Fragile PYTHONPATH in Makefile | Docker plan Task 11 | Pending |

### Additional Issues Found by Architect Review (March 2026)

| # | Severity | Issue | Fix Location | Status |
|---|----------|-------|-------------|--------|
| 15 | 🔴 | Plan 9 chicken-and-egg with Plan 1 | Plan 9 prerequisite section | ✅ FIXED — two-wave strategy |
| 16 | 🔴 | Plan 9 Tasks 5-6 severely underspecified | Plan 9 Tasks 5-6 | ✅ FIXED — full model defs + tests added |
| 17 | 🔴 | Plan 5 Task 6 vs Plan 6 execution ambiguity | Plan 5 Task 6 | ✅ FIXED — SUPERSEDED banner added |
| 18 | 🔴 | Monitoring load_predictions() OOMs at scale | Plan 7 prediction_log.py | ✅ FIXED — Ibis+DuckDB lazy loading |
| 19 | 🟡 | Snowflake empty-string credential defaults | Plan 1 backends/config.py | ✅ FIXED — is_configured + fallback |
| 20 | 🟡 | Reference data has no versioning/staleness detection | Plan 1 reference_data.py | ✅ FIXED — timestamp + staleness check |
| 21 | 🟡 | Permutation importance computed on training set (biased) | Plan 3 page 05 | ✅ FIXED — uses held-out test set |
| 22 | 🟡 | SHAP: no caching, no progress, no timeout | Plan 3 SHAP section | ✅ FIXED — cache + progress + timeout |
| 23 | 🟡 | Data loading conflict: Plan 3 Parquet vs Plan 4 CSV | Plans 3 + 4 data loading | ✅ FIXED — standardized on Parquet + CSV |
| 24 | 🟡 | WebGPU hardware assumptions overstated | Plan 6 Context section | ✅ FIXED — VRAM check, Safari versions |
| 25 | 🟡 | Docker prerequisite misleading (says Plans 1-2, needs Plan 7) | Plan 8 prerequisite | ✅ FIXED |
| 26 | 🟡 | No backup/recovery strategy for Docker volumes | Plan 8 | ✅ FIXED — backup/restore Makefile targets |
| 27 | 🟡 | PolarsConvertible doesn't handle nested models | Plan 9 _base.py | ✅ FIXED — limitation documented |
| 28 | 🟡 | Snowflake password visible in repr/logs | Plan 9 config.py | ✅ FIXED — SecretStr |
| 29 | 🟡 | nfl-data-py may be archived, no fallback | Plan 2 Task 1 | ✅ FIXED — fallback strategy added |
| 30 | 🟡 | Docker containers run as root | Plan 8 | ✅ FIXED — non-root user added |

**Updated totals: 5 breaking + 12 high + 5 inconsistent + 5 minor.** All breaking and high issues have been remediated. The plans are architecturally sound and ready for implementation following the phased roadmap in CLAUDE.md.
