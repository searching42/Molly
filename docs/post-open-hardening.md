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
- Resolved: HARDEN-008 mutable JSON state inventory in PR #57.
- Resolved: HARDEN-009 storage consistency checker in PR #58 and PR #59.
- Resolved: HARDEN-010 SQLite migration design note in PR #60.
- Resolved: HARDEN-011 local process worker supervisor in PR #62 and PR #63.
- Resolved: HARDEN-012 queue control plane and local runner binding in PR #65
  through PR #73.
- Focus: add an explicit run-plan opt-in bridge on top of the file-backed local
  worker loop without jumping to remote workers or SQLite.
- Engineering priority: run-plan queue job schema, local worker loop binding,
  and opt-in `RunPlanExecutor` integration.
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

- Status: Covered by PR #57 inventory in `docs/storage-state-inventory.md`.

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
- The inventory distinguishes project-scoped storage from legacy compatibility
  paths where lock coverage differs.

## HARDEN-009: Add Storage Consistency Checker — RESOLVED

- Status: Resolved by PR #58 (checker API + CLI), PR #59 (e2e workflow binding).
- The `check_workspace_storage(workspace_dir)` API validates artifact registry
  references, stage state, gate decisions, promoted assets, manifests, and job
  state coherence.
- The e2e workflow test (`test_harden_004_e2e_workflow.py`) now calls the
  checker after a full upload → grant → plan → execute → resume → verify →
  report → feedback cycle and asserts `report.ok is True`.
- CLI entry point: `python -m ai4s_agent.storage_consistency <workspace_dir>`.

## HARDEN-010: Evaluate SQLite Migration — RESOLVED

- Status: Resolved by PR #60 (`docs/sqlite-migration-design.md`).
- The design note documents: current JSON+locking state (aligned with
  `storage-state-inventory.md`), per-project SQLite schema for 11 tables,
  what stays as immutable file artifacts, a three-phase migration strategy
  (dual-write → SQLite primary → JSON removal), and risks (WAL checkpointing,
  concurrency, rollback).
- No code migration starts before worker supervision and job durability are
  in place.

## HARDEN-011: Add Local Process Worker Supervisor — RESOLVED

- Status: Resolved across PR #62 and PR #63.
- PR #62 added the `WorkerSupervisor` skeleton with start / status / stop
  primitives, per-worker heartbeat JSON files, and a lifecycle test suite
  (pending → running → stopped/failed, SIGTERM/SIGKILL escalation, duplicate
  start rejection, cross-project same-run-id isolation).
- PR #63 hardened path/read semantics: `status()` is read-only and does not
  create directories; `project_id` and `run_id` are validated through
  `_safe_component()` which rejects empty strings, `.`, `..`, and path
  separators.
- `WorkerSupervisor` is still decoupled from `RunPlanExecutor`.
- No remote worker support yet.  No durable queue yet.

Evidence:

- `src/ai4s_agent/worker_supervisor.py` — `WorkerSupervisor` class
- `tests/test_harden_011_worker_supervisor.py` — 13 tests covering lifecycle,
  signal handling, composite key isolation, stale PID detection, path safety,
  and read-only status semantics.
- Verification from PR #63: full suite passed with `615 passed`.

## HARDEN-012: Add Worker Queue Polling Loop And Local Runner Binding — RESOLVED

- Status: Resolved across PR #65 through PR #73 for the file-backed queue,
  poller, task runner protocol, and local supervisor-backed runner adapter.
- Add queue polling around project-scoped jobs so that multiple queued jobs are
  acquired in a deterministic order and later executed by supervised workers.
- Keep lease acquisition, heartbeat update, cancellation, and stale lease
  recovery semantics without introducing remote workers.
- HARDEN-012 now includes optional task-runner binding and a local
  `WorkerSupervisorTaskRunner` adapter for dummy/process commands only.
- HARDEN-012 intentionally does not include `RunPlanExecutor`, real model
  training jobs, API routes, remote workers, or SQLite.

Skeleton evidence:

- PR #65: `src/ai4s_agent/worker_queue.py` — `WorkerQueue` and
  `JsonWorkerQueueStore`.
- PR #66: queue/lease record validation plus `storage_consistency.py` coverage
  for `worker_queue.json` and `worker_leases.json`.
- PR #67: `src/ai4s_agent/worker_queue_poller.py` — bounded polling skeleton
  for recover/acquire/heartbeat/cancellation visibility.
- PR #68: cancellation and stale recovery control transition tests.
- PR #70: `src/ai4s_agent/worker_task_runner.py` — `WorkerTaskRunner`
  protocol, `TaskRunResult`, and `FakeWorkerTaskRunner`.
- PR #71: optional `WorkerQueuePoller` runner binding while preserving
  control-plane-only behavior when no runner is configured.
- PR #72: `WorkerSupervisorTaskRunner` local adapter for supervised
  dummy/process commands.
- PR #73: `allowed_cwd_root` and task `cwd` fail-closed hardening for the local
  supervisor runner adapter.
- `docs/worker-queue-skeleton.md` — API, file layout, and out-of-scope
  integration points.
- `tests/test_harden_012_worker_queue.py` — queue ordering, lease acquisition,
  heartbeat, queued/running cancellation, terminal state, and stale lease
  recovery.
- `src/ai4s_agent/worker_queue_poller.py` — bounded poller skeleton for
  recover/acquire/heartbeat/cancellation visibility.
- `tests/test_harden_012_worker_queue_poller.py` — poller acquire, heartbeat,
  running cancellation, stale recovery, and bounded loop behavior.

Implemented skeleton semantics:

1. **Queued job schema** — A JSON-serializable record in `worker_queue.json`
   with `job_id`, `project_id`, `run_id`, `task`, `status`, `created_at`,
   `updated_at`, lease fields, cancellation state, attempts, and error state.

2. **Lease acquisition** — `WorkerQueue.acquire(worker_id)` claims the oldest
   non-cancelled queued job by `(created_at, job_id)`, writes a lease record to
   `worker_leases.json`, and updates the job to `running` under
   `.worker_queue.lock`.

3. **Heartbeat update** — `WorkerQueue.heartbeat(lease_id)` refreshes the lease
   heartbeat and expiry timestamp while keeping the job in `running` state.

4. **Cancellation** — Cancelling a queued job moves it to `cancelled` so it is
   skipped by acquire.  Cancelling a running job leaves the lease active but
   exposes `cancellation_requested=true` for a future polling worker.

5. **Stale lease recovery** — Expired active leases can be marked `stale`, and
   their running jobs are requeued for a later worker acquisition.

Implemented poller and runner semantics:

1. `WorkerQueuePoller.poll_once()` recovers stale leases before any worker
   action.

2. If the current worker has an active lease, the poller surfaces
   `cancellation_requested=true` before heartbeat.

3. If no active lease exists, the poller acquires the next queued job for the
   worker.

4. `WorkerQueuePoller.poll(max_iterations=N)` runs a bounded loop for tests and
   future supervisors.

5. When a `WorkerTaskRunner` is configured, the poller can start newly acquired
   jobs, poll active jobs, propagate succeeded/failed/cancelled terminal states
   back to the queue, and avoid heartbeating cancellation-requested jobs.

6. `WorkerSupervisorTaskRunner` adapts this protocol to `WorkerSupervisor` for
   local dummy/process commands.  It rejects shell string commands and can
   constrain task `cwd` under an explicit `allowed_cwd_root`.

Next phase: run-plan opt-in bridge:

1. **Queue + poller + supervisor runner integration test** — Prove the
   file-backed queue, bounded poller, and `WorkerSupervisorTaskRunner` can run a
   local dummy command from queued to terminal state.

2. **LocalWorkerLoop wrapper** — Add a small loop wrapper around queue, poller,
   and runner construction so supervisors can drive bounded iterations without
   API route coupling.

3. **Run-plan queue job schema** — Define the minimal queued task envelope for a
   run-plan execution request, including project/run identity, command-safe
   metadata, artifacts, and permission expectations.  PR #78 adds the schema,
   validator, and builder only; it does not execute `RunPlanExecutor`.

4. **RunPlanExecutor opt-in bridge** — Add an explicit opt-in bridge from
   run-plan jobs to worker queue execution only after the local loop and job
   schema are covered by tests.

Do not jump directly to remote worker support or SQLite migration.  Remote
worker contracts should wait until the local run-plan bridge is stable; SQLite
should wait until file-backed queue and runner semantics remain green under
run-plan opt-in integration.

Acceptance:

- Queue state is persisted in `worker_queue.json`; lease state is persisted in
  `worker_leases.json`.
- Queue and lease mutations use `.worker_queue.lock` and atomic JSON writes.
- Tests cover deterministic ordering, acquire → heartbeat → complete/fail,
  cancel before start, running cancellation visibility, and stale lease
  recovery.
- Poller tests cover recover → acquire, active lease heartbeat, runner start,
  runner poll, terminal state propagation, running cancellation visibility, and
  bounded loop behavior.
- Local runner tests cover exit 0/exit 1, SIGTERM/SIGKILL cancellation,
  shell-string rejection, and `allowed_cwd_root` cwd enforcement.

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
-> HARDEN-011 resolved
-> HARDEN-012 resolved
-> run-plan opt-in bridge
-> HARDEN-013 / HARDEN-014
```

The route-extension cleanup and permission hardening layers are now behind the
e2e safety net. Storage inventory/checking and the local worker control plane
are also in place. The next hardening phase should connect run-plan execution
through an explicit local opt-in bridge before adding remote workers, SQLite
migration, or broader science workflow integrations.

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
- PR #56: completed. Mark permission hardening resolved.
- PR #57: completed. Inventory mutable JSON state and RMW paths.
- PR #58: completed. Add storage consistency checker.
- PR #59: completed. Bind storage consistency checker into e2e workflow.
- PR #60: completed. Document SQLite migration design.
- PR #61: completed. Prepare worker supervision hardening.
- PR #62: completed. Add local process worker supervisor skeleton.
- PR #63: completed. Harden worker supervisor path/read semantics.
- PR #64: completed. Prepare worker queue control-plane skeleton.
- PR #65: completed. Add worker queue JSON control plane.
- PR #66: completed. Validate worker queue records and checker coverage.
- PR #67: completed. Add worker queue polling loop skeleton.
- PR #68: completed. Add cancellation and stale recovery transition tests.
- PR #69: completed. Mark HARDEN-012 control plane resolved and plan runner
  binding.
- PR #70: completed. Add WorkerTaskRunner protocol and fake runner.
- PR #71: completed. Bind WorkerQueuePoller to optional task runner protocol.
- PR #72: completed. Add local supervisor task runner adapter.
- PR #73: completed. Harden supervisor runner cwd constraints.
- PR #74: mark local runner binding complete and plan run-plan opt-in bridge.
- PR #78: define run-plan queue job schema without executing RunPlanExecutor.
