# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the Flask dev server
PYTHONPATH=src .venv/bin/python -m flask --app 'ai4s_agent.app:create_app' run --port 8792

# Run all tests
PYTHONPATH=src .venv/bin/pytest

# Run a single test file
PYTHONPATH=src .venv/bin/pytest tests/test_planner.py -v

# Run a specific test
PYTHONPATH=src .venv/bin/pytest tests/test_planner.py::test_build_plan -v
```

The package uses `src/` layout. Always set `PYTHONPATH=src` when running Python commands. The virtual environment is at `.venv/`.

## Architecture

This is the **AI4S Agent** — a same-process (B1) orchestration layer for a molecular screening workflow. It bridges a legacy CLI pipeline (in a sibling `claude/scripts/` workspace) with a Flask API and web UI.

### Core flow

1. User submits a natural-language prompt (e.g. "optimize lambda_em/plqy/mw")
2. `planner.py` builds a plan of 8 **atomic tasks** (inspect → clean → check trainability → baseline → train → predict → filter/rank → render report)
3. `orchestrator.py` steps through the plan, pausing at **5 human-approval gates** before high-stakes steps like training
4. Adapters in `adapters/phase1.py` execute each task — some in-process (baseline ML, data inspection), others via subprocess calls to legacy scripts
5. Artifacts are persisted under `runs/<run_id>/` and `projects/<project>/`

### Key modules

- **`api.py`** — ~25 Flask endpoints registered via `register_routes()`. Covers plan creation, gate approval, job control (pause/resume/stop/retry), file upload, model registration, asset promotion, and report preview.
- **`planner.py`** — `AtomicTaskRegistry` defines the 8-step pipeline. `build_plan()` creates a Pydantic `PlanModel`; `expand_run_plan()` resolves artifact dependencies.
- **`schemas.py`** — All Pydantic v2 models: `GateName`, `RunStatus`, `PlanModel`, `RunPlan`, `AssetManifest`, etc. Single source of truth for data shapes.
- **`adapters/phase1.py`** — 15 adapter functions wrapping both in-process Python and legacy subprocess calls. This is the largest file (~1150 lines) and the primary integration surface.
- **`data_layer.py`** — CSV inspection, SMILES detection, duplicate detection, outlier detection, SMILES leakage checking. Optionally uses RDKit; gracefully degrades without it.
- **`trainability.py`** — Baseline ML: scaffold split, Morgan fingerprints, XGBoost/RandomForest. Includes backend recommendation (Uni-Mol vs baseline) and 3D relevance assessment.
- **`storage.py`** — `ArtifactStore` (run-scoped) and `ProjectStorage` (project-scoped with versioned assets, model registry, asset promotion).
- **`job_manager.py`** — In-memory job lifecycle state machine with JSONL persistence.
- **`memory.py`** — `ProjectMemory` for collecting run artifacts as persistent project knowledge, plus `PermissionPolicy` (auto / project-approved / confirm-each-time).
- **`ui_cards.py`** — Builds structured card data for the single-page Jinja2 UI (`templates/index.html`).

### The 8 atomic tasks and their gates

| # | Task | Risk | Gate |
|---|------|------|------|
| 1 | inspect_dataset | LOW | gate_1_task_parse (after plan creation) |
| 2 | clean_dataset | MEDIUM | gate_2_data_mining (after inspection) |
| 3 | check_trainability | LOW | — |
| 4 | run_baseline | LOW | — |
| 5 | train_model | HIGH | gate_3_train_config |
| 6 | predict_candidates | MEDIUM | gate_4_post_infer_stats |
| 7 | filter_rank | LOW | — |
| 8 | render_report | LOW | gate_5_final_threshold |

### Adapter pattern

Every adapter in `phase1.py` follows the same contract: accept a `dict` payload, return `{"status": "ok", ...}` or `{"status": "error", "error": "<message>"}`. `contract_validation.py` validates outputs against this shape. `runtime.py` provides subprocess helpers with timeout and error handling. Legacy scripts are invoked via paths resolved by `claude_scripts.py` (uses `AI4S_WORKSPACE` env var or walks up from `__file__`).

### Design constraints

- **B1 phase**: everything runs in-process. The architecture document (`docs/architecture-b1.md`) and migration plan (`docs/migration-b2-ready.md`) describe the planned extraction to separate services without changing the planning/gating logic.
- **Path safety**: all filesystem access uses `pathlib.Path` with `is_relative_to()` checks to prevent traversal outside `runs/` or `projects/`.
- **No ORM, no external DB**: storage is JSON files on disk. `ArtifactStore` and `ProjectStorage` handle all persistence.
- **RDKit is optional**: `data_layer.py` and `trainability.py` degrade gracefully when RDKit is not installed, falling back to hashed fingerprint generation and simplified canonicalization.
