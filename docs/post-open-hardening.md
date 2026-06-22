# Post-OPEN Hardening Roadmap

As of PR #35, the OPEN-001 through OPEN-024 backlog has resolved the known
MVP-blocking architecture, permission, execution-boundary, route-splitting,
state, snapshot, and storage issues tracked in `docs/open-issues.md`.

The next phase is stabilization and end-to-end validation:

```text
Prove that the agent framework can run a complete AI4Science/OLED workflow
with traceable permissions, state, logs, artifacts, retries, gates, and review
records.
```

This backlog uses `HARDEN-*` ids so post-OPEN production hardening does not
blur into the already-resolved OPEN series.

## Current Phase

- Status: OPEN backlog resolved for localhost MVP blockers.
- Resolved: HARDEN-004 localhost e2e safety net in PR #37.
- Resolved: HARDEN-001 route extension metadata and inspection observability in
  PR #39, PR #40, and PR #41.
- Resolved: HARDEN-002 upload, server permission, and project memory permission
  routes migrated to explicit hooks in PR #44, PR #45, and PR #46.
- Resolved: HARDEN-003 project job and project plan route overrides migrated to
  explicit hooks in PR #48 and PR #49.
- Resolved: HARDEN-005 permission grant expiry, revoke, and scope semantics in
  PR #51 and PR #52.
- Resolved: HARDEN-006 actor identity resolver and `confirmed_by` alias in
  PR #53 and PR #54.
- Resolved: HARDEN-007 production permission profile safety in PR #55.
- Focus: inventory mutable JSON state and remaining read-modify-write paths
  before adding storage consistency checks.
- Engineering priority: storage consistency, state migration design, and worker
  supervision.
- Science priority: a small but closed OLED demo with literature provenance,
  model training diagnostics, candidate generation, screening, and report
  artifacts.

## HARDEN-001: Introduce Explicit App Extension Registry

- Status: Resolved across PR #39, PR #40, and PR #41.
- Replace implicit route-extension monkeypatch chains with a first-class app
  extension registry.
- Preserve current installer order while exposing explicit extension metadata,
  dependency names, and registration hooks.
- Keep compatibility wrappers until each extension is migrated.

Evidence:

- PR #39 added `RouteExtensionSpec` metadata and metadata-driven installer
  ordering.
- PR #40 exposed installed route extension metadata on Flask app config and via
  a helper.
- PR #41 added the read-only `/api/system/route-extensions` inspection endpoint
  and route ownership metadata.
- Verification from PR #41: full suite passed with `560 passed`.

Acceptance:

- App creation can list installed extensions and their order.
- Tests assert extension order without relying on mutated function attributes.
- No route behavior changes.

## HARDEN-002: Migrate Permission And Upload Extensions

- Status: Resolved across PR #44, PR #45, and PR #46.
- Move `server_permissions`, `upload_assets`, and
  `project_memory_permissions` away from `api_module.register_routes`
  monkeypatching.
- Register their route overrides through explicit app extension hooks.
- Preserve existing permission audit and legacy fallback behavior.

Evidence:

- PR #44 migrated `immutable_upload_assets` and only the `upload_file` endpoint
  override to an explicit route hook.
- PR #45 migrated `server_permission_routes` and only the three permission
  routes: `create_permission_grant`, `list_permission_grants`, and
  `list_permission_audit`.
- PR #46 migrated `project_memory_permission_routes` and only the four memory
  endpoint overrides: `create_project_memory_record`,
  `update_project_memory_record`, `delete_project_memory_record`, and
  `set_project_memory_enabled`.
- Verification from PR #46: full suite passed with `565 passed`.

Acceptance:

- Existing permission, upload, and memory tests pass unchanged or with only
  naming updates.
- Route override ownership is visible from app construction.

## HARDEN-003: Migrate Project Plan And Job Route Overrides

- Status: Resolved across PR #48 and PR #49.
- Move `project_plan_routes` and `project_job_routes` away from route-function
  replacement via `app.view_functions[...]`.
- Prefer explicit route override hooks or service-level dependencies.
- Keep legacy route compatibility for clients without `project_id`.
- Split migration into smaller PRs because project plan/job routes are more
  coupled to gate approval, retry, job logs, project-scoped keys, and run state.
- Migrate project job routes before project plan routes.

Evidence:

- PR #48 migrated `project_scoped_job_routes` job, log, background, retry, and
  list endpoints to explicit route hooks while preserving `retry_run`
  ownership and behavior.
- PR #49 migrated `project_scoped_plan_routes` `create_plan`, `approve_gate`,
  and `project_run_status` to explicit route hooks without changing retry or
  job behavior.
- Verification from PR #49: full suite passed with `567 passed`.

Acceptance:

- Project-scoped and legacy run/job behavior remains covered by e2e tests.
- Ambiguous `run_id` behavior remains unchanged.
- Inspection endpoint reports explicit hook ownership for migrated job and plan
  route overrides.

## HARDEN-004: Add Localhost Project Workflow E2E Smoke

- Status: Resolved in PR #37 / merge commit
  `58c1b5d8bd4432e16877872d7ef9cd519b2cc224`.
- Evidence: `tests/test_harden_004_e2e_workflow.py` covers the lightweight
  localhost project workflow without real Uni-Mol, MinerU, or network
  acquisition.
- Verification from PR #37: full suite passed with `557 passed`.

Coverage:

```text
create project
-> create server permission grant
-> upload dataset asset
-> create project-scoped plan
-> generate run-plan preview
-> approve gate
-> execute stub or baseline task
-> register artifact
-> read project run status
-> generate report or decision card
```

Acceptance:

- Project namespace does not leak across runs.
- Job state and artifact registry are visible to later agent routes.
- Permission audit records key writes.
- Run-plan preview, execute, and resume do not bypass gate/snapshot checks.
- Conversation/modeling/report routes can consume prior artifacts.
- RunPlan execution/resume events are visible in project-scoped logs when
  `project_id` is present, while legacy run logs remain compatible.

## HARDEN-005: Add Permission Grant Expiry, Revoke, And Scope Semantics

- Status: Resolved across PR #51 and PR #52.
- Add expiry fields to server-side grants.
- Add revoke semantics and audit entries.
- Clarify project-scoped versus run-scoped grants.

Evidence:

- PR #51 added grant expiry fields, validation, and expired-grant denial/audit
  semantics.
- PR #52 added grant revoke behavior, revoke endpoint coverage, and audit
  records for revoked grants.
- Permission decisions now distinguish active server grants, expired grants,
  revoked grants, denied requests, and legacy fallbacks.

Acceptance:

- Expired or revoked grants cannot authorize writes.
- Audit records identify whether grant, fallback, or denial was used.

## HARDEN-006: Standardize Actor Identity Across Project Routes

- Status: Resolved across PR #53 and PR #54.
- Inventory all project routes that accept `actor`, `approved_by`, or similar
  identity fields from clients.
- Standardize request parsing and audit identity semantics.
- Prepare for future server/session-owned actor identity.

Evidence:

- PR #53 added the shared actor identity resolver for permission-related write
  routes, with source metadata such as `header:X-Actor`, `json:actor`,
  `json:approved_by`, `json:revoked_by`, `form:actor`, and `query:actor`.
- PR #54 added `confirmed_by` as a JSON/form actor alias so historical project
  memory payloads audit the confirming actor instead of an empty actor.
- Permission grant, revoke, upload, and project memory write audit records now
  persist `actor_source`.

Acceptance:

- All write routes use one shared actor resolver.
- Missing actor behavior is documented and tested.

## HARDEN-007: Disable Legacy Client Flags By Default In Production Profile

- Status: Resolved in PR #55.
- Keep legacy client flags available for local/dev compatibility.
- Add a production profile where client-declared approval flags are rejected
  unless backed by server grants.
- Consider guarding `/api/system/route-extensions` behind a read-only
  admin/debug switch in production profile, while keeping it available for
  localhost hardening observability.

Production behavior:

- `AI4S_PROFILE=production` or `AI4S_ENV=production` enables production profile.
- Legacy client permission flags are disabled in production, including explicit
  `AI4S_ALLOW_CLIENT_PERMISSION_FLAGS=true` and
  `AI4S_ALLOW_MEMORY_CLIENT_PERMISSION_FLAGS=true` overrides.
- Upload and project memory writes must rely on server grants in production.
- Valid server grants are still accepted in production.
- `/api/system/route-extensions` returns 404 in production by default.
- Route-extension inspection can be enabled explicitly with
  `AI4S_ENABLE_ROUTE_EXTENSION_INSPECTION=true` or app config for admin/debug
  inspection.

Acceptance:

- Production profile tests show upload, memory write, and privileged project
  actions require server-side grants.
- Route extension inspection remains read-only and is disabled or admin/debug
  gated in production profile.

## HARDEN-008: Inventory Mutable JSON State And Remaining RMW Paths

Inventory files such as:

- `job_state.json`
- `background_job_state.json`
- `plan.json`
- `run_plan.json`
- `stage.json`
- `artifact_registry.json`
- `gate_decisions.json`
- `permission_grants.json`
- `permission_audit.jsonl`
- project memory records
- asset manifests

Acceptance:

- Each mutable file is classified as locked RMW, append-only, immutable, or
  migration candidate.
- Unlocked RMW paths are tracked with follow-up items.

## HARDEN-009: Add Storage Consistency Checker

- Add a read-only checker for project/run storage.
- Validate artifact registry references, stage state, gate decisions, promoted
  assets, manifests, and job state coherence.

Acceptance:

- Checker reports missing files, dangling registry entries, malformed manifests,
  and incompatible state transitions.
- It can run in CI against test fixtures.

## HARDEN-010: Evaluate SQLite Migration For Job/Project/Artifact State

- Do not migrate immediately.
- First compare current JSON+lock behavior with a minimal SQLite state store.
- Identify which state should remain file/artifact based.

Acceptance:

- A short design note documents migration scope, risks, rollback path, and
  what stays as immutable artifacts.

## HARDEN-011: Add Local Process Worker Supervisor

- Decouple long-running execution from API request lifetime.
- Add a local supervisor that starts, monitors, and terminates worker processes.

Acceptance:

- API creates jobs; worker process executes callbacks and writes terminal state.
- Supervisor restart behavior is tested at control-plane level.

## HARDEN-012: Add Worker Queue Polling Loop

- Add queue polling around project-scoped jobs.
- Keep lease acquisition, heartbeat, cancel, and stale lease recovery semantics.

Acceptance:

- Multiple queued jobs are acquired in a deterministic order.
- Cancelled jobs do not start.

## HARDEN-013: Add Remote Worker Contract Test

- Define a contract test for remote worker assignment, lease, heartbeat,
  artifact handoff, and terminal state reporting.
- Avoid requiring real remote GPU resources in CI.

Acceptance:

- Fake remote worker satisfies the same state contract as local worker.

## HARDEN-014: Add Cancellation And Stale Lease Recovery E2E Test

- Cover job cancellation during execution.
- Cover stale lease takeover and terminal state recovery.

Acceptance:

- Cancelled jobs stop cleanly and record cancellation metadata.
- Stale leases can be recovered without corrupting artifacts or logs.

## Engineering Track

Recommended order:

```text
HARDEN-004 resolved
-> HARDEN-001 resolved
-> HARDEN-002 resolved
-> HARDEN-003 resolved
-> HARDEN-005 resolved
-> HARDEN-006 resolved
-> HARDEN-007 resolved
-> HARDEN-008 mutable JSON inventory
-> HARDEN-009 storage consistency checker
-> HARDEN-010 SQLite migration design note
-> HARDEN-011 / HARDEN-012 / HARDEN-013 / HARDEN-014
```

The route-extension cleanup and permission hardening layers are now behind the
e2e safety net. The next hardening phase should focus on storage inventory and
consistency before adding remote workers or broader science workflow
integrations.

## Science Track

Run in parallel with engineering hardening, but keep the first demo small:

```text
10-50 OLED papers
-> 20-100 extracted structure-property records
-> baseline predictor
-> 20 generated candidates
-> rule-based or xTB coarse screening
-> provenance-backed report
```

The goal is a closed, auditable demo rather than full automation.

## Near-Term PR Plan

- PR #36: completed. Document post-OPEN hardening roadmap and fix stale
  route-extension docs.
- PR #37: completed. Add localhost project workflow e2e smoke.
- PR #38: completed. Mark HARDEN-004 resolved in the roadmap.
- PR #39: completed. Add route extension metadata registry.
- PR #40: completed. Expose installed route extension metadata on app creation.
- PR #41: completed. Add route extension and route ownership inspection.
- PR #42: completed. Mark HARDEN-001 resolved in the roadmap.
- PR #43: completed. Add explicit route hook skeleton.
- PR #44: completed. Migrate immutable upload assets to explicit hook.
- PR #45: completed. Migrate server permission routes to explicit hook.
- PR #46: completed. Migrate project memory permission routes to explicit hook.
- PR #47: completed. Mark HARDEN-002 resolved in the roadmap.
- PR #48: completed. Migrate project job route overrides to explicit hook.
- PR #49: completed. Migrate project plan route overrides to explicit hook.
- PR #50: completed. Mark HARDEN-003 resolved and start permission hardening.
- PR #51: completed. Add permission grant expiry semantics.
- PR #52: completed. Add permission grant revoke and audit records.
- PR #53: completed. Standardize actor identity resolver.
- PR #54: completed. Add `confirmed_by` as an actor resolver alias.
- PR #55: completed. Add production permission profile safety.
- PR #56: mark permission hardening resolved.
- PR #57: inventory mutable JSON state and RMW paths.
- PR #58: add storage consistency checker.
