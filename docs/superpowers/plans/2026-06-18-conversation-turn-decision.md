# Conversation Turn Decision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a structured next-turn decision layer that tells the agent whether a conversation needs clarification, evidence approval, or a modeling-plan proposal.

**Architecture:** Keep ordinary chat as the user interaction surface and reuse the existing `ConversationAgent.prepare_modeling_plan_payload()` bridge. Add a Pydantic `ConversationTurnDecision` contract, an agent method that wraps the current payload with decision metadata, and a Flask endpoint for callers that need the next action without immediately generating a modeling plan.

**Tech Stack:** Python 3.10+, Pydantic v2, Flask, pytest, JSON schema export.

---

### Task 1: Decision Contract And Agent Tests

**Files:**
- Modify: `tests/test_phase4_conversation_agent.py`
- Modify: `tests/test_schemas.py`
- Modify: `src/ai4s_agent/schemas.py`
- Modify: `src/ai4s_agent/agents/conversation.py`

- [x] **Step 1: Write failing tests**

Add tests asserting that `ConversationAgent.decide_next_turn()` returns:
- `ready_for_modeling_plan` after target and external evidence approval are present.
- `needs_evidence_approval` when a DOI/URL is cited but not approved.
- `needs_clarification` when the target property is missing.

Add a schema test that roundtrips `ConversationTurnDecision` and checks it is exported through `CORE_SCHEMA_MODELS`.

- [x] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_phase4_conversation_agent.py tests/test_schemas.py::test_conversation_turn_decision_schema_roundtrip tests/test_schemas.py::test_export_json_schemas -q
```

Expected: fails because `ConversationTurnDecision` and `decide_next_turn()` do not exist.

- [x] **Step 3: Implement minimal schema and agent method**

Add `ConversationTurnDecision` near `PlanQuestion` with `project_id`, `run_id`, `status`, `decision`, `summary`, `modeling_plan_payload`, `questions`, `pending_cited_target_evidence`, `next_actions`, `blocked_reasons`, `requires_user_response`, and `executable`.

Add `ConversationAgent.decide_next_turn()` that calls `prepare_modeling_plan_payload()`, carries optional `project_memory`, `previous_diagnostics`, and `available_inputs` into the payload, and chooses one of:
- `needs_clarification`
- `needs_evidence_approval`
- `ready_for_modeling_plan`

- [x] **Step 4: Verify green**

Run the same pytest command. Expected: all selected tests pass.

### Task 2: API Endpoint

**Files:**
- Modify: `tests/test_api_smoke.py`
- Modify: `src/ai4s_agent/api.py`

- [x] **Step 1: Write failing API test**

Add a smoke test for `POST /api/agent/conversation/next-turn` that posts chat messages and receives a `decision` payload with `ready_for_modeling_plan`.

- [x] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_api_smoke.py::test_agent_conversation_next_turn_endpoint_returns_decision -q
```

Expected: fails with 404 because the endpoint does not exist.

- [x] **Step 3: Implement minimal endpoint**

Add a Flask route that validates `run_id`, accepts `messages`, `project_memory`, `previous_diagnostics`, and `available_inputs`, calls `ConversationAgent.decide_next_turn()`, and returns `{"ok": true, "decision": ...}`.

- [x] **Step 4: Verify green**

Run the same API test. Expected: pass.

### Task 3: Schema Docs And Project Docs

**Files:**
- Modify: `docs/schemas/conversation_turn_decision.schema.json`
- Modify: `README.md`
- Modify: `to do list.md`
- Modify: `memory/2026-06-18.md`

- [x] **Step 1: Export schemas**

Run:

```bash
PYTHONPATH=src .venv/bin/python -c 'from pathlib import Path; from ai4s_agent.schemas import export_json_schemas; export_json_schemas(Path("docs/schemas"))'
```

Expected: `docs/schemas/conversation_turn_decision.schema.json` exists.

- [x] **Step 2: Update docs**

Document that the next-turn endpoint is the current agent decision boundary for chat-driven modeling. Mark the decision layer as implemented while keeping approved acquisition planning and model execution as separate later steps.

### Task 4: Verification And Commit

**Files:**
- All modified files from Tasks 1-3.

- [x] **Step 1: Run targeted tests**

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_phase4_conversation_agent.py tests/test_api_smoke.py::test_agent_conversation_next_turn_endpoint_returns_decision tests/test_schemas.py::test_conversation_turn_decision_schema_roundtrip tests/test_schemas.py::test_export_json_schemas tests/test_schemas.py::test_docs_schema_files_include_every_core_schema -q
```

- [x] **Step 2: Run broader regression**

```bash
PYTHONPATH=src .venv/bin/pytest -q
```

- [x] **Step 3: Check diff hygiene**

```bash
git diff --check -- .
```

- [x] **Step 4: Commit and push**

```bash
git add docs/superpowers/plans/2026-06-18-conversation-turn-decision.md docs/schemas/conversation_turn_decision.schema.json README.md "to do list.md" memory/2026-06-18.md src/ai4s_agent/schemas.py src/ai4s_agent/agents/conversation.py src/ai4s_agent/api.py tests/test_phase4_conversation_agent.py tests/test_api_smoke.py tests/test_schemas.py
git commit -m "feat: add conversation turn decisions"
git push origin main
```

---

Self-review:
- Spec coverage: The plan implements the next backend target after pausing UI work: conversation-level decisioning with approval and clarification states.
- Placeholder scan: All steps include exact file paths and verification commands.
- Type consistency: Tests, schema export key, and API response all use `ConversationTurnDecision` / `conversation_turn_decision`.
