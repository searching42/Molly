# Project Chat UI Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the Flask web console into a Codex-style project sidebar plus project conversation workspace while preserving existing workflow tools.

**Architecture:** Keep the implementation in `src/ai4s_agent/templates/index.html` for this first phase. Add smoke tests that lock down the new DOM markers and JavaScript calls before changing the template. Existing wizard forms remain in the DOM under a collapsed workflow/tools section so current endpoints and tests keep working.

**Tech Stack:** Flask/Jinja template, vanilla JavaScript, pytest smoke tests.

---

### Task 1: Lock Down New Layout Markers

**Files:**
- Modify: `tests/test_api_smoke.py`
- Modify: `src/ai4s_agent/templates/index.html`

- [x] **Step 1: Write failing layout smoke test**

Add a test that requests `/` and asserts:

```python
assert 'id="app-shell"' in html
assert 'id="project-sidebar"' in html
assert 'id="project-list"' in html
assert 'id="new-project-form"' in html
assert 'id="project-workspace"' in html
assert 'id="project-chat"' in html
assert 'id="conversation-stream"' in html
assert 'id="conversation-form"' in html
assert 'id="conversation-input"' in html
assert 'id="chat-review-artifacts"' in html
```

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_api_smoke.py::test_index_page_uses_project_sidebar_and_chat_workspace -q
```

Expected: FAIL because the new shell markers do not exist yet.

- [x] **Step 3: Add layout shell**

Modify `index.html` so the top-level page uses:

- `main#app-shell.app-shell`
- `aside#project-sidebar.project-sidebar`
- `section#project-workspace.project-workspace`
- `section#project-chat.chat-panel`
- `section#chat-review-artifacts.review-card-grid`

Keep `#primary-workflow`, `#response-console`, and `#advanced-tools` present.

- [x] **Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_api_smoke.py::test_index_page_uses_project_sidebar_and_chat_workspace -q
```

Expected: PASS.

### Task 2: Wire Project List And Conversation Payload Calls

**Files:**
- Modify: `tests/test_api_smoke.py`
- Modify: `src/ai4s_agent/templates/index.html`

- [x] **Step 1: Write failing JavaScript smoke test**

Add a test that requests `/` and asserts:

```python
assert "async function loadProjects" in html
assert 'getJSON("/api/projects")' in html
assert 'postJSON("/api/projects"' in html
assert "function selectProject" in html
assert "let conversationMessages" in html
assert 'postJSON("/api/agent/conversation/modeling-payload"' in html
assert 'postJSON("/api/agent/modeling-plan"' in html
assert "pending_cited_target_evidence" in html
assert "agent_questions" in html
```

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_api_smoke.py::test_index_page_wires_project_chat_to_agent_payload_bridge -q
```

Expected: FAIL because the new JavaScript functions are not present yet.

- [x] **Step 3: Add minimal JavaScript**

Add vanilla JS functions:

- `loadProjects()`
- `renderProjectList(projects)`
- `selectProject(project)`
- `appendConversationMessage(role, content)`
- `renderConversationArtifact(title, payload)`
- `renderAgentQuestions(questions)`
- conversation form submit handler that calls `/api/agent/conversation/modeling-payload`
- modeling plan button handler that calls `/api/agent/modeling-plan`

Do not auto-execute training, acquisition, promotion, or gate approval from a chat message.

- [x] **Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_api_smoke.py::test_index_page_wires_project_chat_to_agent_payload_bridge -q
```

Expected: PASS.

### Task 3: Verify Existing Workflow Compatibility

**Files:**
- Modify: `src/ai4s_agent/templates/index.html`
- Modify: `to do list.md`

- [x] **Step 1: Run focused UI smoke tests**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_api_smoke.py::test_index_page_available tests/test_api_smoke.py::test_index_page_uses_progressive_wizard_cards tests/test_api_smoke.py::test_index_page_submit_task_executes_current_run_plan tests/test_api_smoke.py::test_index_page_gate_approval_resumes_current_run_plan tests/test_api_smoke.py::test_index_page_renders_modeling_agent_review_card_sections -q
```

Expected: PASS.

- [x] **Step 2: Update roadmap status**

Update `to do list.md` UI TODO to record that phase 1 project sidebar/chat workspace is implemented and that remaining UI work is polish plus richer artifact rendering.

- [x] **Step 3: Run full verification**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q
git diff --check -- .
```

Expected: all tests pass and diff check is clean.

- [x] **Step 4: Commit**

Stage and commit:

```bash
git add docs/superpowers/plans/2026-06-18-project-chat-ui-phase1.md "to do list.md" src/ai4s_agent/templates/index.html tests/test_api_smoke.py
git commit -m "feat: add project chat workspace UI"
git push origin main
```

---

## Self-Review

- Spec coverage: Covers project sidebar, project chat, review artifacts, conversation payload bridge, and preservation of existing tools.
- Placeholder scan: No placeholders remain.
- Type consistency: DOM ids and endpoint names match the existing API and the design spec.
