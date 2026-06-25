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
- Resolved: user-confirmed resume loop completion through PR #117 + PR #118:
  verifier/proposal → application → validation → actual one-time resume execution →
  post-resume review.
- Focus: define default-route migration criteria before considering any
  replacement of the synchronous `/api/run-plan/execute` path.
- Engineering priority: harden permission/audit, queue lifecycle, observability,
  and waiting-user semantics around the opt-in run-plan queue bridge.
- Science priority: a small but closed OLED demo with literature provenance,
  model training diagnostics, candidate generation, screening, and report
  artifacts.
- Resolved: PR #128 adds queue recovery and stale lease coverage for the
  feature-flagged queued execute canary without changing route behavior,
  allowlist scope, remote-worker posture, or SQLite/storage decisions.
- Resolved: PR #129 documents and guards the default-migration readiness
  checklist for the queued execute canary.
- Resolved: PR #130 adds a second allowlisted chain parity fixture for the
  feature-flagged queued execute canary, using the actual second fully
  allowlisted planner expansion instead of broadening the allowlist.
- Resolved: PR #131 adds cancellation coverage for the queued execute canary
  and documents that retry/requeue production semantics remain future work.
- Resolved: PR #132 documents and guards the production-sized fixture
  boundary for the queued execute canary, while explicitly keeping
  production-sized or nightly fixture proof out of the current PR.
- Resolved: PR #133 documents and guards the queued canary
  telemetry/observability checklist, while explicitly not implementing
  production telemetry or changing route behavior.
- Resolved: PR #134 documents and guards the optional nightly
  production-sized fixture lane design for the queued execute canary, while
  explicitly not enabling a workflow, not committing large fixture data, and
  not changing route behavior.
- Resolved: PR #135 adds minimal structured telemetry fields for queued
  canary runs at the local review/test level, while explicitly not
  implementing production telemetry sinks, dashboards, or alerting.
- Resolved: PR #136 adds an optional manual queued-canary evidence workflow
  skeleton, while explicitly not enabling scheduled nightly execution, not
  adding large fixture data, and not changing default CI behavior.
- Resolved: PR #137 defines explicit retry/requeue semantics for the local
  queued execute canary at the docs/test level, while explicitly not adding
  retry operations, queue mutations, or API routes.
- Next recommended queued-canary work: implement only narrow explicit retry
  behavior if needed, or deepen observability beyond the current minimal
  telemetry surface.
- Default-route migration is still not recommended.

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
16. **OLED property profile and multi-objective screening fixture** — A
    structured OLED property profile now drives a low-risk fixture where PLQY,
    emission wavelength, HOMO, LUMO, and singlet-triplet gap are recognized as
    configurable properties, predicted with multiple single-property baselines,
    combined into profile-driven score contributions, ranked, and reported.
17. **Run-plan artifact Observer-Verifier** — A fixed, read-only
    observer-verifier schema now evaluates queued execution summaries/status,
    audit outcomes, artifact registry entries, trainability reports, model
    metrics, generation reports, extraction benchmark reports, and
    multi-objective ranking outputs into one of `continue`, `needs_review`,
    `rerun_recommended`, or `blocked`.
18. **Reviewable replan proposal** — A deterministic proposal layer now maps
    `RunPlanArtifactVerification` results into a fixed, non-executable
    `RunPlanReplanProposal` with affected tasks, rationale, required user
    decisions, and an unapplied advisory run-plan patch.
19. **Verifier/replan review artifacts** — The verifier result and replan
    proposal can now be materialized as review-only artifacts:
    `observer_verification.json`, `replan_proposal.json`, and
    `replan_review.md`.
20. **Run-plan review card** — Previously written review artifacts can now be
    read as one `RunPlanReviewCard` schema for UI, report, or project-memory
    consumers through a helper and feature-flagged internal read route.
21. **Project memory review summary** — Run-plan review cards can now be saved
    as compact `ProjectMemoryRecord` entries containing only decision summary
    fields and artifact references for future Planner/Observer/Replanner use.
22. **User-confirmed replan application semantics** — PR #104 defines the
    design boundary for converting a user-confirmed reviewable proposal into a
    `ResumeIntent`, `RunPlanRevision`, or blocked acknowledgement without
    executing adapters, mutating `RunPlan`, enqueueing jobs, or letting
    proposal/LLM output apply itself automatically. See
    `docs/user-confirmed-replan-application-semantics.md`.
23. **Replan application schemas and compiler** — PR #105 and PR #106 define
    non-executable application request/record/result schemas and a
    deterministic validator/compiler for selected proposal operations.
24. **Replan application review artifacts** — PR #107 materializes compiled
    application drafts as review artifacts such as
    `replan_application_record.json`, `replan_resume_intent.json`,
    `run_plan_revision.json`, or `blocked_acknowledgement.json`.
25. **Replan application audit and memory summary** — PR #108 adds append-only
    audit records for requested/completed/failed replan application events and
    compact project-memory summaries containing only summary fields, artifact
    refs, and audit refs.
26. **Internal replan application review route** — PR #109 exposes
    `POST /api/internal/run-plan/replan/apply-review` behind the internal
    feature flag, actor identity, and `run_plan_replan_apply` permission grant.
    It writes requested/completed/failed audit records, materializes application
    review artifacts, saves compact project memory, and returns a review
    summary without executing or enqueueing anything.
27. **Resume intent validation semantics** — PR #110 defines how a future
    gate/resume path should validate `review/replan_resume_intent.json` before
    any execution bridge can consume it. The design covers source application
    linkage, proposal hash checks, artifact refs, current `RunPlan`
    compatibility, stale-intent detection, resume audit, and default-route
    compatibility without adding execution code.
28. **Internal resume intent validation route** — PR #111 through PR #114
    define resume-intent validation schemas, deterministic validation helper,
    validation audit/memory summaries, and a feature-flagged internal
    validation route. The route requires actor identity and a
    `run_plan_resume_intent_use` server grant, returns validation results, and
    records audit/memory summaries without resuming, enqueueing, writing gate
    decisions, executing adapters, mutating `RunPlan`, or replacing default
    routes.
29. **Resume intent state binding** — PR #115 binds user-confirmed resume
    intents to canonical `RunPlan` and stable `StageState` fingerprints. The
    application writer records the binding in both application and intent
    artifacts; validation recomputes current fingerprints and fails closed with
    `stale_intent` on drift. This is integrity/staleness hardening only, not
    resume authorization or execution.
30. **Strict resume stage/gate compatibility** — PR #116 requires resume
    intents to bind to the current `WAITING_USER` stage, validates the current
    execution snapshot material, checks the waiting task against
    `AtomicTaskRegistry`, separates application gates from executor gates, and
    rejects embedded executor gate approvals in resume intent artifacts. This
    remains validation-only: it does not call `RunPlanExecutor.resume_after_gate`,
    write gate decisions, enqueue work, execute adapters, mutate `RunPlan`, call
    LLMs, or replace default routes.
31. **Feature-flagged internal resume intent execution bridge** — PR #117 adds
    `POST /api/internal/run-plan/resume-intent/execute` behind
    `AI4S_ENABLE_INTERNAL_RESUME_INTENT_EXECUTE_ROUTE`, actor identity, and a
    `run_plan_resume_execute` server grant. The bridge server-loads current
    artifacts and state, reruns `validate_resume_intent(...)`, requires
    `decision="resume_eligible"`, writes a pre-execution consumed audit record
    fail-closed, calls the existing `RunPlanExecutor.resume_after_gate(...)`,
    and records completed/failed audit plus compact memory. It is one-time
    consumption only and still does not enqueue work, call LLMs, mutate
    `RunPlan`, write custom gate decisions, replace `/api/run-plan/resume`, or
    replace `/api/run-plan/execute`.
32. **User-confirmed resume loop e2e** — PR #118 adds
    `tests/test_user_confirmed_resume_loop_e2e.py` coverage of the closed-loop
    path: verifier findings → proposal → apply-review → validation →
    resume-after-approval execution → review artifacts/card refresh.

Still not default:

- `/api/run-plan/execute` remains synchronous and is not replaced.
- The internal execute and status routes require an explicit feature flag,
  actor identity, and a `run_plan_queue_execute` server grant.
- Target-job acquisition is now supported in queue acquire/poller selectors.
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
- The OLED property profile fixture is data configuration, not a hardcoded
  OLED-only core schema enum. It does not implement full multi-task model
  training and does not allow runtime LLM-generated executable code.
- The Observer-Verifier is a deterministic read-only layer. It does not execute
  adapters, call LLMs, mutate queues, or create revised plans. Future LLM or
  planner components should consume its fixed schema and propose only
  reviewable replans.
- The reviewable replan proposal layer is also deterministic and non-executing.
  It does not mutate `RunPlan`, apply its advisory patch, call LLMs, enqueue
  jobs, execute adapters, or automatically rerun tasks. User confirmation and a
  future gate/resume or modified-plan path are still required before execution.
- The review artifact writer only materializes verifier/proposal outputs for
  UI, report, or project-memory review. It does not execute the proposal,
  mutate `RunPlan`, call LLMs, enqueue jobs, or replace `/api/run-plan/execute`.
- The review card layer only aggregates existing review artifacts. It does not
  regenerate artifacts, execute proposals, apply patches, call LLMs, enqueue
  jobs, mutate `RunPlan`, or replace `/api/run-plan/execute`.
- The project memory review summary stores only the review decision,
  proposed action, affected tasks, required user decisions, and artifact
  references. It does not store raw datasets, full artifact contents, markdown
  bodies, complete verifier/proposal payloads, executable patches, or queued
  execution state.
- User-confirmed replan application semantics are design-only in PR #104.
  Confirmation should create a reviewable application record,
  `ResumeIntent`, `RunPlanRevision`, or blocked acknowledgement. It should not
  directly execute proposals, apply advisory patches, enqueue jobs, call LLMs,
  mutate `RunPlan`, or replace `/api/run-plan/execute`.
- Replan application schemas, compiler, artifacts, audit records, and memory
  summaries are still review-only. They do not execute adapters, apply
  patches, enqueue jobs, mutate `RunPlan`, call LLMs, or replace
  `/api/run-plan/execute`. Memory records store only compact summary fields,
  artifact references, and audit references.
- The internal replan application route is also review-only. It requires the
  existing internal feature flag, actor identity, and a
  `run_plan_replan_apply` server grant, but it does not resume runs, execute
  adapters, enqueue jobs, apply patches, mutate `RunPlan`, call LLMs, or
  replace `/api/run-plan/execute`.
- Resume intent validation remains design-only in PR #110. It does not add a
  resume route, call `RunPlanExecutor.resume_after_gate(...)`, enqueue work,
  write gate decisions, mutate `RunPlan`, call LLMs, or replace
  `/api/run-plan/resume` or `/api/run-plan/execute`.
- Resume intent validation schemas, helpers, audit/memory, and the internal
  validation route remain read-only validation surfaces. The internal route is
  feature-flagged, actor-gated, and permission-gated, but it does not call
  `RunPlanExecutor.resume_after_gate(...)`, write gate decisions, enqueue work,
  execute adapters, mutate `RunPlan`, call LLMs, or replace
  `/api/run-plan/resume` or `/api/run-plan/execute`.
- Resume intent state bindings are stale-state checks only. They do not
  authorize resume, do not replace actor/permission/gate checks, and do not
  execute adapters. Future resume execution must recompute fingerprints again
  immediately before consuming the intent.
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
5. The dedicated-queue limitation is addressed for the internal run-plan helper
   by low-level target acquisition plus service-level ownership of the newly
   created job. `WorkerQueue.acquire(...)` can target an arbitrary known job,
   but `run_run_plan_via_local_queue(...)` must only process the job it just
   enqueued.
6. `waiting_user` semantics must stay explicit. PR #94 defines the first
   compatibility contract: queue jobs finish as `succeeded`, while summary,
   status, and audit surfaces carry waiting metadata. A full queued resume
   engine remains future work.
7. User-confirmed replan application must remain separate from execution.
   A confirmed proposal should first produce a validated application record and
   either a `ResumeIntent`, `RunPlanRevision`, or blocked acknowledgement with
   actor, permission, audit, gate, and compact memory semantics. A separate
   gate/resume/execute path is still required before any adapter runs.
8. Resume intents must remain bound to current run state. PR #115 adds
   canonical run-plan and stable stage fingerprints; any future default route
   migration must recompute them at consumption time and fail closed on drift.
9. User-confirmed resume loop readiness is now established through PR #118:
   review findings → proposal → application → validation → one-time execute →
   post-resume review artifacts/card refresh.
10. Target-job acquisition now has selector-level support in queue/poller APIs;
    service-level hardening keeps `run_run_plan_via_local_queue(...)` scoped to
    its newly created job. Default-route migration still requires canary
    coverage for actor/permission/audit-safe target execution.

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
CLI, stable summary, feature-flagged internal route coverage, and a minimal
feature-flagged queued canary for `/api/run-plan/execute`. The default route
still uses synchronous execution unless `AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY`
is enabled. Canary observability now records backend markers in run logs, and
rollback evidence tests prove that disabling the flag returns the route to the
sync response shape without creating queue files for sync runs. The next
hardening phase should keep satisfying the migration gates before making queued
execution the default, adding remote workers, migrating storage to SQLite, or
broadening science workflow integrations.

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
- PR #97: add OLED property profile and multi-objective screening fixture using
  multiple single-property predictions plus weighted ranking, without hardcoding
  OLED-only core property enums or implementing full multi-task training.
- PR #98: add a read-only run-plan artifact Observer-Verifier that maps queue,
  audit, registry, trainability, model, generation, extraction, and ranking
  evidence into fixed four-state decisions for downstream reviewable replans.
- PR #99: add a deterministic, reviewable `RunPlanReplanProposal` layer from
  Observer-Verifier findings without mutating plans, executing adapters, calling
  LLMs, or auto-rerunning tasks.
- PR #100: write Observer-Verifier and replan proposal review artifacts
  (`observer_verification.json`, `replan_proposal.json`, `replan_review.md`)
  without executing proposals, mutating plans, enqueueing jobs, or replacing the
  default run-plan execution route.
- PR #101: expose a read-only `RunPlanReviewCard` helper and internal route for
  existing review artifacts without executing proposals, applying patches,
  enqueueing jobs, mutating plans, calling LLMs, or replacing the default
  run-plan execution route.
- PR #102: save `RunPlanReviewCard` summaries to project memory as compact
  `run_plan_review` records with decision fields and artifact references only,
  without storing raw data, full artifact contents, executing proposals,
  applying patches, enqueueing jobs, mutating plans, or calling LLMs.
- PR #103: completed. Consolidate Phase 1-4 milestone status and remaining
  non-goals.
- PR #104: completed. Define user-confirmed replan application semantics as a
  docs-only boundary.
- PR #105: completed. Define replan application request, record, resume intent,
  blocked acknowledgement, operation id, and selected-operation schemas without
  execution.
- PR #106: completed. Add deterministic replan patch validator/compiler without
  writing files, modifying `RunPlan`, enqueueing jobs, executing adapters, or
  calling LLMs.
- PR #107: completed. Write replan application review artifacts and register
  their refs without executing, applying patches, mutating `RunPlan`, or
  enqueueing jobs.
- PR #108: add replan application audit records and compact project-memory
  summary helpers without adding routes, executing proposals, applying patches,
  enqueueing jobs, mutating `RunPlan`, or calling LLMs.
- PR #109: add a feature-flagged internal replan application review route that
  writes application artifacts, audit records, and compact memory summaries
  behind actor and `run_plan_replan_apply` permission gates, without executing,
  enqueueing, applying patches, mutating `RunPlan`, calling LLMs, or replacing
  `/api/run-plan/execute`.
- PR #110: define resume intent validation semantics in docs only, covering
  source application, proposal hash, artifact refs, current `RunPlan`, stale
  intent, gates, audit, permission, and default-route compatibility without
  adding resume execution.
- PR #116: enforce strict waiting-stage, execution-snapshot, and executor-gate
  compatibility for resume intent validation without adding resume execution.
- PR #117: add a feature-flagged internal resume intent execution bridge with
  strict validation, audit, permission, and one-time consumption, without
  replacing default resume/execute routes.
- PR #118: add a full user-confirmed resume loop e2e test covering verifier
  findings, proposal generation, application, validation, one-time execute, and
  post-resume review artifact/card refresh.
- PR #119: completed. Implement target-job acquisition support in
  `WorkerQueue.acquire()` and `WorkerQueuePoller` before default-route queue
  migration.
- PR #120: completed. Harden run-plan queue target selector semantics so the service helper
  never processes an externally selected job and never leaves orphan queued
  jobs on selector mismatch.
- PR #121: completed. Add a feature-flagged `/api/run-plan/execute` queued
  canary. The flag can be turned off to immediately return to the synchronous
  route.
- PR #122: completed. Add queued execute canary observability and rollback
  evidence tests. The canary remains feature-flagged, preserves sync response
  compatibility when disabled, does not change `/api/run-plan/resume`, does not
  enable remote workers, and does not migrate queue storage to SQLite.
- PR #123: completed. Restrict the `/api/run-plan/execute` queued canary to
  selected low-risk task chains only. The canary flag remains a master switch,
  but only all-allowlisted chains can use the queue bridge. Non-allowlisted
  tasks such as `train_model`, generation, literature/mining, and unknown task
  ids fall back to the synchronous path with no `execution_backend` or
  `queue_summary` response fields.
- PR #124: completed. Define the queued execute canary rollout policy and
  decision matrix in `docs/queued-execute-canary-rollout-policy.md`, with
  green criteria, red disable conditions, allowlist expansion rules, and default
  migration exit criteria.
- PR #125: completed. Add artifact registry parity fixture coverage for an
  existing allowlisted queued execute chain, comparing sync and queued canary
  logical artifact ids and artifact file existence without expanding the
  allowlist.
- PR #126: completed. Add failure classification parity fixture coverage for an
  existing allowlisted queued execute chain, comparing sync and queued canary
  failed status, failed task, and useful error message fields without expanding
  the allowlist.
- PR #127: completed. Add repeated-run stability coverage for the queued
  canary, proving repeated allowlisted runs isolate queue state by
  project/run, keep response shape and logical artifacts stable, and do not let
  rollback touch existing queued jobs.
- PR #128: completed. Add queue recovery/stale lease coverage for the queued
  canary, proving stale running jobs are not mistaken for the target job,
  target-job selection remains valid after recovery, and sync fallback does not
  process queued jobs.
- PR #129: completed. Add a default-migration readiness checklist plus doc
  guard coverage for the queued canary, clarifying that current canary
  coverage is still insufficient for default migration.
- PR #130: completed. Add a second allowlisted chain parity fixture for the
  queued canary. The current planner expansion for `render_report` still
  reaches non-allowlisted tasks, so this parity fixture intentionally uses
  the actual second all-allowlisted chain, `check_trainability`, without
  expanding the allowlist.
- PR #131: completed. Add cancellation coverage for the queued canary and
  prove that sync fallback does not process or mutate cancelled queued jobs.
  Because the queue still has no explicit retry/requeue production API, retry
  semantics remain documented future work rather than new behavior in this PR.
- PR #132: completed. Document and guard the production-sized fixture boundary
  for the queued canary. This PR keeps the boundary explicit: current small
  deterministic fixtures are useful for control-plane confidence, but they are
  not production-sized proof. No medium or nightly fixture lane is enabled
  yet.
- PR #133: completed. Document and guard the telemetry/observability
  checklist for the queued canary. This PR does not implement production
  telemetry, does not change `/api/run-plan/execute`, and does not make queued
  execution default.
- PR #134: completed. Design and guard the optional nightly
  production-sized fixture lane for the queued canary. This PR does not enable
  a nightly workflow, does not commit large fixture data, and does not change
  default route behavior.
- PR #135: completed. Add minimal structured telemetry fields for queued
  canary runs, emitted as local review/test evidence only. This PR does not
  implement production telemetry sinks, dashboards, alerting, or default
  migration.
- PR #136: completed. Add an optional manual queued-canary workflow
  skeleton for bounded evidence collection. This PR does not enable a
  schedule, does not run on `pull_request` or `push`, and does not make
  queued execution default.
- PR #137: completed. Define retry/requeue semantics for the local queued
  execute canary without implementing retry behavior. This PR keeps lease
  attempts, stale recovery, explicit retry, and rerun/new execution as
  separate concepts and keeps default migration blocked.
- Next: implement only narrow explicit retry behavior if needed, or deepen
  observability beyond the current minimal telemetry surface. Do not proceed
  to default-route migration yet.
