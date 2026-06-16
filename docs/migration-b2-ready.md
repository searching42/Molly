# B2 Migration Readiness

## Goal

B2 moves orchestration out of the Flask API process without rewriting planning, gate, agent, or adapter logic.

## Service Boundaries

The following B1 modules are written as future service boundaries:

- `Planner`: request in, `PlanModel` out
- `Gatekeeper`: gate decision in, approval state out
- `Orchestrator`: run state transition in, status/artifacts out
- `ArtifactStore`: run id and filename in, JSON artifact out
- `Adapters`: typed command construction in, subprocess command out

## Transport Change

B1 calls Python modules in-process. B2 can expose the same operations over HTTP, queue messages, or an internal RPC layer.

The minimum transport payloads are:

- Create plan: `run_id`, `prompt`
- Approve gate: `run_id`, `gate`, `actor`, `note`
- Read status: `run_id`
- Launch step: `run_id`, `step_name`

## State Handoff

State handoff remains file-backed through:

```text
workspace/agent/runs/<run_id>/
```

This keeps B1 and B2 compatible with the same run artifacts.

## Compatibility Rules

- Do not change gate names without a migration script.
- Do not rename `plan.json` or `gate_decisions.json`.
- Keep adapter outputs serializable as JSON-friendly dictionaries.
- Keep generated candidate scoring routed through `ScreenerAgent`.
