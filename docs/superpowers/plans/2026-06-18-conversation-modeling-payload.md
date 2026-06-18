# Conversation Modeling Payload Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert ordinary user-agent conversation turns into a reviewable `/api/agent/modeling-plan` payload without adding a dedicated cited-evidence UI form.

**Architecture:** Add a small conversation bridge in the agent layer that extracts target property hints, cited DOI/URL/source summaries, user approval for external search, and follow-up questions from supplied conversation turns. Keep `/api/agent/modeling-plan` as the structured execution boundary and leave the frontend focused on review cards and artifacts.

**Tech Stack:** Python, Pydantic schemas, Flask API tests, pytest.

---

### Task 1: Conversation Bridge

**Files:**
- Create: `src/ai4s_agent/agents/conversation.py`
- Test: `tests/test_phase4_conversation_agent.py`

- [x] **Step 1: Write the failing tests**

```python
from ai4s_agent.agents.conversation import ConversationAgent


def test_conversation_bridge_builds_modeling_payload_from_dialogue() -> None:
    agent = ConversationAgent()

    payload = agent.prepare_modeling_plan_payload(
        run_id="run-dialogue-plqy",
        project_id="proj-dialogue-plqy",
        messages=[
            {"role": "user", "content": "I need an OLED PLQY model. Use DOI 10.1038/s41597-020-00634-8; it says PLQY is solvent-conditioned and high values compress."},
            {"role": "assistant", "content": "May I use that cited external source in the modeling brief?"},
            {"role": "user", "content": "Yes, use that external literature evidence."},
        ],
    )

    assert payload["run_id"] == "run-dialogue-plqy"
    assert payload["project_id"] == "proj-dialogue-plqy"
    assert payload["property_id"] == "plqy"
    assert payload["user_approved_external_search"] is True
    assert payload["cited_target_evidence"][0]["doi"] == "10.1038/s41597-020-00634-8"
    assert "solvent" in payload["cited_target_evidence"][0]["summary"].lower()
    assert payload["agent_questions"] == []


def test_conversation_bridge_asks_before_using_unapproved_external_evidence() -> None:
    agent = ConversationAgent()

    payload = agent.prepare_modeling_plan_payload(
        run_id="run-dialogue-needs-approval",
        messages=[
            {"role": "user", "content": "Train PLQY. DOI 10.1038/s41597-020-00634-8 says solvent matters."},
        ],
    )

    assert payload["property_id"] == "plqy"
    assert payload["user_approved_external_search"] is False
    assert payload["cited_target_evidence"] == []
    assert payload["pending_cited_target_evidence"][0]["doi"] == "10.1038/s41597-020-00634-8"
    assert payload["agent_questions"][0]["question_id"] == "approve_external_target_evidence"
```

- [x] **Step 2: Run tests to verify RED**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_phase4_conversation_agent.py -q`

Expected: FAIL because `ai4s_agent.agents.conversation` does not exist.

- [x] **Step 3: Implement minimal bridge**

Create `ConversationAgent.prepare_modeling_plan_payload()` with rule-based extraction for the MVP:

```python
payload = {
    "run_id": run_id,
    "project_id": project_id,
    "goal": latest_user_goal,
    "property_id": detected_property_id,
    "user_approved_external_search": approved,
    "cited_target_evidence": approved_evidence,
    "pending_cited_target_evidence": pending_evidence,
    "agent_questions": questions,
}
```

The bridge should detect common target aliases (`plqy`, `quantum yield`, `lambda_em`, `emission`) and DOI/URL/source references, but it must not fetch network content or claim uncited facts.

- [x] **Step 4: Run tests to verify GREEN**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_phase4_conversation_agent.py -q`

Expected: PASS.

### Task 2: API Wrapper And Docs

**Files:**
- Modify: `src/ai4s_agent/api.py`
- Modify: `README.md`
- Modify: `to do list.md`
- Test: `tests/test_api_smoke.py`

- [x] **Step 1: Write API smoke test**

```python
def test_agent_conversation_modeling_payload_endpoint_prepares_payload(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/conversation/modeling-payload",
        json={
            "project_id": "proj-conversation",
            "run_id": "run-conversation",
            "messages": [
                {"role": "user", "content": "Train OLED PLQY. DOI 10.1038/s41597-020-00634-8 says solvent matters."},
                {"role": "user", "content": "Yes, use this external literature evidence."},
            ],
        },
    )

    assert resp.status_code == 200
    assert resp.json["modeling_plan_payload"]["property_id"] == "plqy"
    assert resp.json["modeling_plan_payload"]["user_approved_external_search"] is True
```

- [x] **Step 2: Run smoke test to verify RED**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_api_smoke.py::test_agent_conversation_modeling_payload_endpoint_prepares_payload -q`

Expected: FAIL with 404 because the endpoint is not registered.

- [x] **Step 3: Implement API wrapper**

Add `POST /api/agent/conversation/modeling-payload` that validates `messages` as a list of role/content objects, calls `ConversationAgent.prepare_modeling_plan_payload()`, and returns:

```python
{"ok": True, "modeling_plan_payload": payload}
```

- [x] **Step 4: Update docs**

Document that ordinary dialogue is the primary interface for collecting user intent, cited evidence, approvals, and follow-up answers. Remove the stale plan to add a dedicated evidence input UI.

- [x] **Step 5: Verify and commit**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q
git diff --check -- .
git add docs/superpowers/plans/2026-06-18-conversation-modeling-payload.md README.md "to do list.md" src/ai4s_agent/agents/conversation.py src/ai4s_agent/api.py tests/test_phase4_conversation_agent.py tests/test_api_smoke.py
git commit -m "feat: prepare modeling payloads from conversation"
git push origin main
```

Expected: all tests pass, diff check is clean, and the commit is pushed to `main`.

---

## Self-Review

- Spec coverage: The plan keeps dialogue as the evidence collection interface, avoids new UI controls, and preserves `/api/agent/modeling-plan` as the structured downstream boundary.
- Placeholder scan: No open placeholders remain.
- Type consistency: The payload fields match the existing modeling-plan endpoint fields plus non-executable `pending_cited_target_evidence` and `agent_questions`; question objects use the existing `PlanQuestion.question_id` contract.
