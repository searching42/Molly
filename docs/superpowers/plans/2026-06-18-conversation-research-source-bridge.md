# Conversation Research Source Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let ordinary conversation produce a reviewable `ResearchSourceProposal` for optional external acquisition planning without performing network acquisition.

**Architecture:** Reuse `ConversationAgent` for dialogue parsing and `ResearchAgent.propose_sources()` for source ranking. Add a conversation-to-research payload method that extracts DOI/URL seed sources from chat, then expose a Flask endpoint that writes the existing research proposal artifacts when a project is supplied.

**Tech Stack:** Python 3.10+, Flask, Pydantic schemas already present, pytest.

---

### Task 1: Conversation Research Payload

**Files:**
- Modify: `tests/test_phase4_conversation_agent.py`
- Modify: `src/ai4s_agent/agents/conversation.py`

- [x] **Step 1: Write the failing test**

Add a test for `ConversationAgent.prepare_research_source_payload()` that supplies chat messages with a DOI and URL, then asserts the returned payload contains `run_id`, `project_id`, `goal`, and seed sources compatible with `LiteratureCorpusSource`.

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_phase4_conversation_agent.py::test_conversation_bridge_prepares_research_source_payload_from_dialogue -q
```

Expected: fail because `prepare_research_source_payload()` does not exist.

- [x] **Step 3: Implement minimal method**

Add `prepare_research_source_payload()` that validates messages through `_coerce_messages()`, extracts DOI/URL references, creates deterministic seed source dictionaries, keeps `user_approved_external_search`, and adds an approval question when query-only acquisition would need user review.

- [x] **Step 4: Verify green**

Run the same pytest command. Expected: pass.

### Task 2: API Bridge

**Files:**
- Modify: `tests/test_api_smoke.py`
- Modify: `src/ai4s_agent/api.py`

- [x] **Step 1: Write the failing API test**

Add a smoke test for `POST /api/agent/conversation/research-sources` that posts chat messages and receives a non-executable `ResearchSourceProposal` containing DOI/URL selected sources and output artifact paths.

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_api_smoke.py::test_agent_conversation_research_sources_endpoint_returns_proposal -q
```

Expected: fail with 404 because the endpoint does not exist.

- [x] **Step 3: Implement minimal endpoint**

Add a Flask route that validates `run_id`, calls `ConversationAgent.prepare_research_source_payload()`, calls `ResearchAgent.propose_sources()`, writes artifacts through `ResearchAgent.write_proposal()` when `project_id` is present, and returns `{"ok": true, "research_source_payload": ..., "proposal": ..., "outputs": ...}`.

- [x] **Step 4: Verify green**

Run the same API test. Expected: pass.

### Task 3: Documentation And Verification

**Files:**
- Modify: `README.md`
- Modify: `to do list.md`
- Modify: `memory/2026-06-18.md`

- [x] **Step 1: Update docs**

Document that conversation can now generate a dry-run research source proposal, but network/database acquisition remains a separate explicit action.

- [x] **Step 2: Run full verification**

```bash
PYTHONPATH=src .venv/bin/pytest -q
git diff --check -- .
```

- [x] **Step 3: Commit and push**

```bash
git add README.md "to do list.md" docs/superpowers/plans/2026-06-18-conversation-research-source-bridge.md src/ai4s_agent/agents/conversation.py src/ai4s_agent/api.py tests/test_phase4_conversation_agent.py tests/test_api_smoke.py
git commit -m "feat: bridge conversation to research source planning"
git push origin main
```

---

Self-review:
- Spec coverage: The plan advances optional approved acquisition planning without adding network execution.
- Placeholder scan: No placeholder requirements remain.
- Type consistency: The payload uses existing `LiteratureCorpusSource` dictionaries and the API returns existing `ResearchSourceProposal` JSON.
