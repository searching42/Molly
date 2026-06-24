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
- Resolved: HARDEN-012 queue control plane, local runner binding, and internal
  run-plan queue route phase in PR #65 through PR #87.
- Focus: define default-route migration criteria before considering any
  replacement of the synchronous `/api/run-plan/execute` path.
- Engineering priority: harden permission/audit, queue lifecycle, observability,
  and waiting-user semantics around the opt-in run-plan queue bridge.
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

- Status: Resolved across PR #65 through PR #87 for the file-backed queue,
  poller, task runner protocol, local supervisor-backed runner adapter, run-plan
  queue bridge, internal CLI, stable summary schema, and feature-flagged
  internal API route.
- Add queue polling around project-scoped jobs so that multiple queued jobs are
  acquired in a deterministic order and later executed by supervised workers.
- Keep lease acquisition, heartbeat update, cancellation, and stale lease
  recovery semantics without introducing remote workers.
- HARDEN-012 now includes optional task-runner binding and a local
  `WorkerSupervisorTaskRunner` adapter for dummy/process commands only.
- HARDEN-012 intentionally does not replace the default synchronous
  `/api/run-plan/execute` route, does not add remote workers, does not migrate
  state to SQLite, and does not guarantee real training workloads.

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
- PR #75: integration test proving `WorkerQueue`, `WorkerQueuePoller`,
  `WorkerSupervisorTaskRunner`, and `WorkerSupervisor` can move local dummy
  commands from queued to terminal state.
- PR #76: runner exception handling so acquired jobs fail terminally instead of
  leaving active leases behind.
- PR #77: `LocalWorkerLoop` bounded wrapper for reusable local polling.
- PR #78: run-plan queue job schema and validator.
- PR #79: one-shot `RunPlanExecutorTaskRunner` adapter.
- PR #80: local worker loop integration test for the run-plan executor task
  runner.
- PR #81: internal enqueue helper for `run_plan_execute` worker jobs.
- PR #82: internal queued run-plan execution service helper.
- PR #83: internal run-plan queue CLI.
- PR #84: CLI `input_artifacts` and `task_options` JSON support.
- PR #85: low-risk CLI fixture demo.
- PR #86: stable `RunPlanQueueExecutionSummary` schema.
- PR #87: feature-flagged internal run-plan queue route.
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

Resolved run-plan queue bridge scope:

1. **Run-plan queue schema** — `run_plan_execute` envelopes are validated and
   intentionally exclude command/argv/cwd fields.
2. **Enqueue helper** — `RunPlan` plus project/input/options context can be
   converted into a queued worker job without starting execution.
3. **One-shot `RunPlanExecutorTaskRunner`** — The adapter can consume the
   queued envelope and synchronously call `RunPlanExecutor.execute(...)` behind
   explicit opt-in helpers.
4. **Local queue service** — `run_run_plan_via_local_queue(...)` composes the
   enqueue helper, `WorkerQueuePoller`, `LocalWorkerLoop`, and task runner.
5. **Internal CLI** — Local debugging can run the queued service from JSON
   files with stable exit-code behavior.
6. **Stable summary schema** — CLI, helper, and route responses share
   `RunPlanQueueExecutionSummary`.
7. **Feature-flagged internal route** —
   `POST /api/internal/run-plan/queue/execute` is available only behind
   `AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE` and uses an internal queue path.
8. **Actor and audit metadata** — The internal route requires the shared actor
   resolver, writes a pre-execution `requested` audit gate, and appends terminal
   audit records to
   `workspace/.ai4s_internal/audit/internal_run_plan_queue_audit.jsonl`.
9. **Permission gate** — The internal route requires an explicit
   `run_plan_queue_execute` server grant before writing the `requested` audit
   event or executing the queued run-plan helper.
10. **Lifecycle observability** — Internal helpers can read queue/lease status,
    recover stale leases, and clean terminal records without touching active
    queued/running jobs. A feature-flagged internal status route exposes
    read-only queue state behind the same actor and permission gate.
11. **Lifecycle helper hardening** — Cleanup/recovery helpers use locked queue
    state, return stable result dictionaries, fail closed on malformed JSON,
    and keep cleanup/recovery internal until mutating operations are audit-ready.
12. **Phase 1 queued workflow fixture** — The existing Phase 1 workflow is
    productized through the internal queued execution bridge in a low-risk
    fixture demo that writes cleaned dataset, baseline metrics, candidate
    predictions, ranking, report files, artifact registry entries, queue status,
    and audit records.
13. **Queued `WAITING_USER` semantics** — The first queued execution contract
    keeps `WAITING_USER` terminal-compatible by completing the queue job as
    `succeeded`, while `RunPlanQueueExecutionSummary`, queue status, and audit
    records explicitly expose `waiting_user`, `waiting_task`, and
    `required_gates` metadata.
14. **Phase 2 generation-to-screening fixture** — Deterministic candidate
    generation now feeds a generated candidate dataset into the Phase 1
    prediction, filtering/ranking, and report chain under the internal queued
    execution bridge.
15. **Phase 3 literature-to-dataset fixture** — Fixture parsed literature/table
    data now flows through structured extraction, PLQY unit normalization,
    duplicate merge/conflict reporting, confirmed dataset export, benchmark
    reporting, and Phase 1 trainability intake.

Still not default:

- `/api/run-plan/execute` remains synchronous and is not replaced.
- The internal execute and status routes require an explicit feature flag,
  actor identity, and a `run_plan_queue_execute` server grant.
- Remote workers are not connected.
- Queue state remains file-backed; no SQLite migration is included.
- The Phase 1 fixture uses lightweight baseline training/prediction only; it
  does not prove heavy Uni-Mol/DPA3 training, remote workers, or Phase 2/3/4
  workflow completion.
- The Phase 2 fixture uses deterministic local generation only; it is not full
  inverse design and does not execute REINVENT4 or other external generators.
- The Phase 3 fixture uses local parsed-table fixtures only; it is not full
  literature mining, Web Search, network acquisition, MinerU parsing, or
  large-scale corpus extraction.
- The Phase 3 confirmed dataset is intended to feed Phase 1 training workflows,
  but the fixture only verifies trainability intake and does not run heavy
  model training.
- Full queued resume semantics for waiting-user runs remain future work.

Default-route migration hard gates:

1. `RunPlanExecutorTaskRunner` must pass a real low-risk adapter demo, not only
   fake executor tests. PR #93 adds a low-risk Phase 1 queued workflow fixture
   demo with real artifact writes.
2. The internal route must have permission, actor identity, and audit
   constraints that match or exceed the synchronous route. PR #89 added the
   first actor/audit layer; PR #90 adds the route-level permission grant gate.
3. Queue lifecycle must include cleanup, stale recovery, and observability for
   stuck queued/running jobs. PR #91 starts this layer with read-only status,
   stale recovery helper, and terminal cleanup helper. PR #92 hardens helper
   semantics before any cleanup/recovery route is considered.
4. `RunPlanQueueExecutionSummary` must be validated consistently by route, CLI,
   service helper, and integration tests.
5. The dedicated-queue limitation must be resolved either with guaranteed
   per-request dedicated queues or target-job acquisition.
6. `waiting_user` semantics must stay explicit. PR #94 defines the first
   compatibility contract: queue jobs finish as `succeeded`, while summary,
   status, and audit surfaces carry waiting metadata. A full queued resume
   engine remains future work.

Do not jump directly to remote worker support or SQLite migration. Remote worker
contracts should wait until the local default-route migration gates are met;
SQLite should wait until file-backed queue and runner semantics remain green
under the opt-in bridge.

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
-> run-plan internal queue route resolved
-> default-route migration criteria
-> HARDEN-013 / HARDEN-014
```

The route-extension cleanup and permission hardening layers are now behind the
e2e safety net. Storage inventory/checking and the local worker control plane
are also in place. The run-plan queue bridge now has schema, local execution,
CLI, stable summary, and feature-flagged internal route coverage. The next
hardening phase should satisfy the default-route migration gates before
replacing `/api/run-plan/execute`, adding remote workers, migrating storage to
SQLite, or broadening science workflow integrations.

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
- PR #74: completed. Mark local runner binding complete and plan run-plan
  opt-in bridge.
- PR #75: completed. Add local worker queue supervisor runner integration
  coverage.
- PR #76: completed. Fail queued worker jobs when runner start/poll/cancel
  raises.
- PR #77: completed. Add `LocalWorkerLoop` bounded wrapper.
- PR #78: completed. Define run-plan queue job schema without executing
  RunPlanExecutor.
- PR #79: completed. Add one-shot RunPlanExecutorTaskRunner without API route
  wiring.
- PR #80: completed. Cover RunPlanExecutorTaskRunner through local worker loop.
- PR #81: completed. Add internal run-plan worker queue enqueue helper without
  API route wiring.
- PR #82: completed. Add internal opt-in queued run-plan execution helper
  without API route wiring.
- PR #83: completed. Add internal run-plan queue CLI without API route wiring.
- PR #84: completed. Add input_artifacts/task_options JSON support to the internal
  run-plan queue CLI.
- PR #85: completed. Add low-risk run-plan queue CLI fixture demo without API route
  wiring.
- PR #86: completed. Define stable RunPlanQueueExecutionSummary schema for
  queued helper and CLI output.
- PR #87: completed. Add feature-flagged internal run-plan queue route without
  replacing `/api/run-plan/execute`.
- PR #88: completed. Mark internal run-plan queue route phase complete and define
  default-route migration criteria.
- PR #89: completed. Add actor identity, pre-execution audit gate, and terminal audit
  metadata to the feature-flagged internal run-plan queue route.
- PR #90: completed. Add explicit `run_plan_queue_execute` permission grant
  requirement to the feature-flagged internal run-plan queue route.
- PR #91: completed. Add internal run-plan queue lifecycle helpers and feature-flagged
  read-only queue status route.
- PR #92: completed. Harden run-plan queue lifecycle helper semantics without exposing
  cleanup/recovery routes.
- PR #93: add Phase 1 queued workflow fixture demo without replacing
  `/api/run-plan/execute`.
- PR #94: define queued `WAITING_USER` compatibility semantics without adding a
  full resume queue engine.
- PR #95: connect deterministic generation candidates to the Phase 1
  screening/report chain without external generation backends.
- PR #96: add Phase 3 literature-to-dataset fixture pipeline without Web Search,
  MinerU crawling, remote workers, or heavy training.
