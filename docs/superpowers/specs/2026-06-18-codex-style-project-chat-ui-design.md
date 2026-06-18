# Codex-Style Project Chat UI Design

## Purpose

Move the local web console from a wizard-first workflow into a simple project workspace:

- Left side: project navigation and project-level actions.
- Right side: current project conversation and review artifacts.

The UI should feel closer to Codex Desktop while remaining a lightweight Flask-served web page. The primary interaction is conversation. Forms remain available only for explicit approvals, file selection, and advanced/debug operations.

## Scope

This design covers the first web MVP iteration only.

In scope:

- A two-pane responsive layout in the existing Flask HTML template.
- Project list, project creation, and active project selection in the left sidebar.
- A project chat surface in the right pane with user/agent messages.
- A message composer that sends conversation turns to the existing agent payload bridge.
- Inline review areas for plan proposal, target evidence questions, run gates, logs, and artifacts.
- A collapsed advanced tools section that preserves existing wizard/tooling functions during migration.

Out of scope for this iteration:

- Native desktop packaging.
- A React/Vue/Svelte rewrite.
- Multi-user collaboration UI.
- Rich persistent chat history search.
- Autonomous execution from plain chat without a visible gate or review artifact.

## Layout

Desktop layout:

- `app-shell`: full-height two-column grid.
- `project-sidebar`: fixed 280px column.
- `project-workspace`: flexible main column.

Mobile layout:

- Sidebar collapses above the workspace.
- Project list becomes a compact stacked panel.
- Composer remains at the bottom of the workspace content, not fixed over unread content.

The visual style should be quiet and operational:

- No oversized hero section.
- No marketing-style cards.
- Cards are used only for repeated project rows, review cards, gates, and advanced tool panels.
- Prefer dense spacing, clear dividers, compact headings, and stable dimensions.

## Left Sidebar

The sidebar owns project selection, not scientific workflow execution.

Sections:

- Header: product name and current local mode.
- New project: compact form with project name/id.
- Project list: loaded from `GET /api/projects`.
- Current project summary: selected `project_id`, latest run id if available, memory enabled state if loaded.
- Utility links: memory, uploads, model assets, reports.

Project rows should show:

- Project name.
- Project id.
- Last modified date if available from `project.json`; otherwise omit the line.
- Active selection state.

Project creation uses the existing `POST /api/projects` endpoint. The UI should select the created project immediately after a successful response.

## Right Workspace

The right side is organized around a project conversation.

Top bar:

- Project title.
- Current run id.
- Compact status text for latest plan/job/gate state.

Conversation stream:

- User messages render right-aligned or visually distinct.
- Agent messages render left-aligned.
- Review artifacts render as inline system blocks, not as normal chat prose.
- Agent questions from `agent_questions` render as blocking follow-up blocks with explicit choices.

Composer:

- Multi-line text input.
- Primary submit button.
- Optional attach/upload button can reuse the existing upload endpoint, but file upload remains an explicit user action.
- The composer sends accumulated visible messages to `POST /api/agent/conversation/modeling-payload`.

After the conversation payload response:

- If `agent_questions` is non-empty, render those questions and wait for a user answer.
- If `cited_target_evidence` is available, the next explicit action can call `/api/agent/modeling-plan`.
- If only `pending_cited_target_evidence` exists, do not call `/api/agent/modeling-plan` with that evidence until the user approves.

## Review Artifacts

Review artifacts stay visible and auditable. They should not be hidden inside chat text.

Artifacts shown inline:

- `modeling_plan_payload`
- `TargetModelingBrief`
- `PlanRationale`
- `RunPlan`
- `ModelDiagnosticsReport`
- `RerunProposal`
- `ModelPackageReview`
- gate approval prompts
- run logs

Each artifact block should expose:

- short title
- status
- source labels when present
- primary action if any
- compact JSON/details toggle

Executable actions must still map to existing explicit endpoints and gates. A normal chat message must not silently trigger training, external acquisition, backend switches, model promotion, or data deletion.

## Data Flow

Project bootstrap:

1. Load projects with `GET /api/projects`.
2. Select the first project if none is selected.
3. If no projects exist, show an empty project state with a new-project form.

Conversation turn:

1. User submits text in composer.
2. UI appends the user message to the visible conversation.
3. UI posts `project_id`, `run_id`, and `messages` to `/api/agent/conversation/modeling-payload`.
4. UI renders returned `modeling_plan_payload`.
5. UI renders returned `agent_questions` as follow-up blocks.
6. UI does not execute the modeling plan automatically.

Modeling plan action:

1. User confirms the current payload is ready.
2. UI posts the payload fields required by `/api/agent/modeling-plan`.
3. UI renders the returned proposal and optional `target_modeling_brief`.
4. Follow-on actions use the existing run confirmation, submit, gate, status, and review endpoints.

Advanced tools:

- Existing wizard forms move under a collapsed advanced area.
- Advanced tools keep their current endpoint behavior.
- The main path should not require opening advanced tools for normal project chat planning.

## Error Handling

Project and conversation errors render inside the workspace, near the action that failed.

Rules:

- Missing active project: ask the user to create or select a project.
- Missing run id: generate a local run id before posting.
- API validation error: show the server error and preserve the composer content.
- Network/server failure: show a retryable error block.
- Unapproved external evidence: render the approval question and keep evidence pending.

## Testing Strategy

Unit/smoke tests should cover:

- Home page contains the two-pane shell markers.
- Project list endpoint data is renderable by expected DOM hooks.
- Conversation composer calls `/api/agent/conversation/modeling-payload`.
- Unapproved pending evidence appears as a follow-up question, not as approved evidence.
- Existing advanced forms remain present or reachable after the layout change.

Manual visual check should cover:

- Desktop viewport around 1440px wide.
- Narrow viewport around 390px wide.
- Text does not overflow buttons, project rows, review blocks, or composer.
- Main conversation remains usable with empty projects, one project, and several projects.

## Implementation Notes

Keep the first implementation inside `src/ai4s_agent/templates/index.html` unless the file becomes too hard to maintain. If the migration grows beyond a single focused patch, split static JavaScript/CSS into separate files as a separate refactor with its own tests.

The first implementation should preserve existing endpoint contracts and avoid backend schema churn. Backend additions are allowed only if the UI cannot list, create, or select projects with the current API.

## Open Decision Resolved

The UI will not add a dedicated cited-evidence input form. Evidence and approvals are collected through ordinary conversation, then rendered as structured review artifacts.
