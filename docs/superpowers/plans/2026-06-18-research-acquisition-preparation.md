# Research Acquisition Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a review-only preparation artifact that turns a `ResearchSourceProposal` into explicit adapter payloads for source-manifest preparation and later acquisition.

**Architecture:** Keep `ResearchSourceProposal` as source discovery output and add `ResearchAcquisitionPreparation` as the next approval boundary. The preparation object will expose required gates/permissions and adapter payloads, but it will never execute adapters or perform network/database acquisition.

**Tech Stack:** Python 3.10+, Pydantic v2, Flask, pytest, JSON schema export.

---

### Task 1: Schema And Agent Preparation

**Files:**
- Modify: `tests/test_phase4_research_agent.py`
- Modify: `tests/test_schemas.py`
- Modify: `src/ai4s_agent/schemas.py`
- Modify: `src/ai4s_agent/agents/research.py`

- [x] **Step 1: Write failing tests**

Add tests for `ResearchAgent.prepare_acquisition()` and `ResearchAcquisitionPreparation` roundtrip/export. The agent test should build a `ResearchSourceProposal` with DOI/URL sources, call preparation with an output directory, and assert:
- `status == "needs_confirmation"`
- `executable is False`
- `source_manifest_adapter == "prepare_literature_corpus_sources_adapter"`
- `acquisition_adapter == "acquire_literature_sources_adapter"`
- `required_gates == ["gate_2_data_mining"]`
- `required_permissions` contains `external_acquisition_scope`

- [x] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_phase4_research_agent.py::test_research_agent_prepares_acquisition_payload_from_source_proposal tests/test_schemas.py::test_research_acquisition_preparation_schema_roundtrip tests/test_schemas.py::test_export_json_schemas -q
```

Expected: fail because the schema and method do not exist.

- [x] **Step 3: Implement minimal schema and agent method**

Add `ResearchAcquisitionPreparation` near `ResearchSourceProposal`. Add `ResearchAgent.prepare_acquisition()` and a write helper that stores `research_acquisition_preparation.json` and `.md` when project storage is supplied by the API.

- [x] **Step 4: Verify green**

Run the same pytest command. Expected: pass.

### Task 2: API Endpoint

**Files:**
- Modify: `tests/test_api_smoke.py`
- Modify: `src/ai4s_agent/api.py`

- [x] **Step 1: Write failing API test**

Add `POST /api/agent/research-acquisition/prepare` smoke coverage. The test should post a `ResearchSourceProposal`, receive a preparation object, and verify output artifact paths are written for a project run.

- [x] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_api_smoke.py::test_agent_research_acquisition_prepare_endpoint_writes_preparation -q
```

Expected: fail with 404.

- [x] **Step 3: Implement endpoint**

Parse `proposal`, `output_dir`, `local_mirror`, and `user_confirmed_external_acquisition`; call `ResearchAgent.prepare_acquisition()`; write artifacts when `project_id` is supplied; return `{"ok": true, "preparation": ..., "outputs": ...}`.

- [x] **Step 4: Verify green**

Run the same API test. Expected: pass.

### Task 3: Docs And Verification

**Files:**
- Modify: `docs/schemas/research_acquisition_preparation.schema.json`
- Modify: `README.md`
- Modify: `to do list.md`
- Modify: `memory/2026-06-18.md`

- [x] **Step 1: Export schemas**

```bash
PYTHONPATH=src .venv/bin/python -c 'from pathlib import Path; from ai4s_agent.schemas import export_json_schemas; export_json_schemas(Path("docs/schemas"))'
```

- [x] **Step 2: Update docs**

Document that acquisition preparation is review-only and does not execute adapter work.

- [x] **Step 3: Run verification**

```bash
PYTHONPATH=src .venv/bin/pytest -q
git diff --check -- .
```

- [x] **Step 4: Commit and push**

```bash
git add README.md "to do list.md" docs/schemas/research_acquisition_preparation.schema.json docs/superpowers/plans/2026-06-18-research-acquisition-preparation.md src/ai4s_agent/schemas.py src/ai4s_agent/agents/research.py src/ai4s_agent/api.py tests/test_phase4_research_agent.py tests/test_api_smoke.py tests/test_schemas.py
git commit -m "feat: add research acquisition preparation"
git push origin main
```

---

Self-review:
- Spec coverage: The plan adds the explicit approval boundary requested by the current project status without adding network execution.
- Placeholder scan: No placeholders remain.
- Type consistency: The schema, agent method, API response, and schema export all use `ResearchAcquisitionPreparation` / `research_acquisition_preparation`.
