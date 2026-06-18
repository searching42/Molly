# Long-Term Memory

Load this file only in main/direct sessions with Benton. Do not load or reveal
it in shared/group contexts.

## Project: AI4S Agent

`/Users/benton/openclaw-docker/workspace/agent` is the canonical mainline for
the AI4S Agent project. `../claude` and `../oled-agent` are legacy/reference
folders. Do not merge either folder wholesale into the mainline.

The product direction is a data-centric AI4S agent:

- Phase 1: local single-user Flask app for dataset inspection, cleaning,
  trainability, baseline/Uni-Mol training, prediction, ranking, reports, and
  asset promotion.
- Phase 2: candidate generation and inverse-design loop, including deterministic
  stub generation, REINVENT4 backend planning/execution paths, iterative
  generate-predict-filter, novelty/diversity, and frontier targets.
- Phase 3: evidence-grounded literature-to-dataset workflow, including source
  manifests, acquisition planning, MinerU/default parsing, parser fallbacks,
  table-aware retrieval, dense/multi-index retrieval, extraction, unit
  normalization, citation/license/provenance, conflict handling, human
  confirmation, leakage checks, and benchmark metrics.
- Phase 4: agentic decision layer over the audited workflow substrate:
  PlannerAgent, LLM provider abstraction, Observer/Verifier, RecoveryAgent,
  research/modeling/generation/report agents, agentic UI, remote worker planning,
  background-job budgets, and multi-user readiness checks.

Core principle: workflow/adapters/gates remain the execution authority. LLMs and
agents may propose, explain, verify, and replan, but high-risk actions require
explicit confirmation and structured traces.

## Current Status As Of 2026-06-15

Branch:

```text
release/oled-agent-v0.1.0
```

Latest known verification:

```text
PYTHONPATH=src .venv/bin/pytest -q
362 passed in 2.06s

git diff --check -- .
no output
```

Repository scale at that point:

- 39 Python source files under `src/ai4s_agent`.
- 29 top-level pytest files.
- 67 exported JSON Schema files under `docs/schemas`.

The worktree was dirty. Known local changes included Agent decision card and log
tail work in:

- `src/ai4s_agent/api.py`
- `src/ai4s_agent/templates/index.html`
- `tests/test_api_smoke.py`
- `tests/test_run_plan_executor.py`

There were also sibling workspace changes under `../claude` and `../TOOLS.md`.
Do not revert them without explicit approval.

## Real Acceptance Evidence

Phase 1 has real remote Uni-Mol acceptance evidence in:

```text
projects/phase1-acceptance/runs/phase1-acceptance-run/
```

That run records remote Uni-Mol training, candidate prediction, filtering/ranking,
and report generation as completed. Key artifacts include:

- `stage.json`
- `artifact_registry.json`
- `03_training/model_metadata.json`
- `03_training/phase1-acceptance-run_lambda_em_train_report.json`
- `04_screening/phase1-acceptance-run_lambda_em_scored.csv`
- `04_screening/phase1-acceptance-run_ranked.csv`
- `05_report/phase1-acceptance-run_final_summary.md`

Later OLED project runs under `projects/proj-oled` and `projects/proj-oled-2`
had completed inspect/clean/trainability and then paused at Uni-Mol training
plan-only `WAITING_USER` states. They did not continue into real remote training
in those runs.

## Important Commands

Use the repo virtual environment and `PYTHONPATH=src`:

```bash
PYTHONPATH=src .venv/bin/pytest -q
PYTHONPATH=src .venv/bin/python -m flask --app 'ai4s_agent.app:create_app' run --port 8792
PYTHONPATH=src .venv/bin/python -m compileall src tests
git diff --check -- .
```

Quick UI URL when the Flask app is running:

```text
http://127.0.0.1:8792/
```

## Design Decisions To Preserve

- `workspace/agent` is the mainline.
- Project storage is under `projects/<project_id>/runs/<run_id>` and
  `projects/<project_id>/assets`.
- JSON artifacts are the machine-readable source of truth; Markdown/HTML are
  human-facing summaries.
- User-confirmed project memory stores decisions, rules, preferences, and
  references, not raw datasets or secrets.
- External LLM is off by default for private data. If used, payload policy and
  confirmation records matter.
- Remote Uni-Mol, REINVENT4, network acquisition, expensive generation, model
  registration, and asset promotion require explicit gates or approvals.
- Existing deterministic adapters should remain usable without depending on
  hidden conversation state.
- Generated candidates must pass through the same prediction and filtering chain
  before ranking/reporting.
- Literature-derived data cannot become confirmed training data without human
  confirmation.
- The local UI should stay simple and conversation-first: project sidebar,
  project chat, review artifacts, response console, and collapsed advanced raw
  tools. Do not reintroduce the old wizard-card primary workflow.

## Documentation Notes

`to do list.md` is currently the most complete roadmap/status file.
`README.md` and `docs/architecture-b1.md` are useful but lag the codebase; they
still read like early B1 documentation while the code and TODO now cover Phase 4.

The next useful documentation pass should update:

- README quickstart and current capabilities,
- architecture docs from B1-only to audited substrate plus agentic layer,
- release notes for `release/oled-agent-v0.1.0`,
- a concise operator guide for local UI flow and safe remote backends.

## Recent Non-Code Deliverable

On 2026-06-14, a Chinese final review paper was generated about AI for Science
driven organic optoelectronic material discovery, tying together molecular
representation, generation models, multi-agent closed-loop workflows, Uni-Mol,
lambda_em/PLQY prediction, REINVENT4, and this AI4S Agent project.

Files:

- `output/pdf/ai4s_oled_review_paper.pdf`
- `output/pdf/ai4s_oled_review_paper.md`
- `output/pdf/generate_ai4s_oled_review_pdf.py`
