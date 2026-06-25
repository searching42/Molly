# Worker Queue And Local Runner Skeleton

HARDEN-012 is resolved for the file-backed queue, bounded poller, optional task
runner binding, and local supervisor-backed runner adapter across PR #65 through
PR #73.  The queue can persist jobs, lease them to workers, poll for
heartbeat/cancellation/recovery state, validate queue files, optionally drive a
`WorkerTaskRunner`, and run local dummy/process commands through
`WorkerSupervisorTaskRunner`.

It still does not call `RunPlanExecutor`, expose API routes, connect to remote
workers, migrate state to SQLite, or run real model training jobs.

Resolved PRs:

- PR #65: `WorkerQueue` / `JsonWorkerQueueStore` JSON-backed queue and lease
  state.
- PR #66: queue/lease record validation and storage consistency checker
  coverage.
- PR #67: `WorkerQueuePoller` bounded polling skeleton.
- PR #68: cancellation and stale recovery control transition tests.
- PR #70: `WorkerTaskRunner` protocol, `TaskRunResult`, and
  `FakeWorkerTaskRunner`.
- PR #71: optional `WorkerQueuePoller` binding to `WorkerTaskRunner`.
- PR #72: `WorkerSupervisorTaskRunner` local adapter for supervised
  dummy/process commands.
- PR #73: `allowed_cwd_root` and task `cwd` fail-closed hardening for the local
  supervisor runner adapter.

Python API:

```python
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue

queue = WorkerQueue(JsonWorkerQueueStore("/path/to/control/state"))
job = queue.enqueue("project-a", "run-1", {"task_id": "train_model"})
lease = queue.acquire("worker-a")
queue.heartbeat(lease["lease_id"])
queue.complete(lease["lease_id"])
```

Persisted files:

- `worker_queue.json`: durable job records and queue status
- `worker_leases.json`: active and historical worker leases
- `.worker_queue.lock`: lock file used around queue/lease read-modify-write
  updates

Mutation semantics:

- All queue mutations acquire a process-local lock and same-directory lock file.
- JSON writes use the project atomic `write_json()` helper, which writes a temp
  file in the same directory and replaces the destination.
- The queue is intentionally store-backed and small so a future SQLite-backed
  store can implement the same API.

Implemented operations:

- `enqueue(project_id, run_id, task)`
- `enqueue_retry_of_failed_job(source_job_id, *, retry_request_id, requested_by, reason)`
- `acquire(worker_id, *, target_job_id=None, target_project_id=None, target_run_id=None)`
- `heartbeat(lease_id)`
- `complete(lease_id)`
- `fail(lease_id)`
- `cancel(job_id)`
- `recover_stale_leases()`
- `status(job_id)`
- `lease_status(lease_id)`
- `list_leases()`
- `list_jobs()`

Polling and runner skeleton:

```python
from ai4s_agent.worker_queue_poller import WorkerQueuePoller

poller = WorkerQueuePoller(queue, worker_id="worker-a")
result = poller.poll_once()
```

`WorkerQueuePoller` runs the control-plane sequence:

1. recover stale leases
2. if the worker has an active lease, surface `cancellation_requested` before
   refreshing the lease
3. heartbeat the active lease only when cancellation is not requested
4. if no active lease exists, acquire the first queued job that matches optional
   target selectors, then fall back to normal queued order when no selector is
   set.

`poll(max_iterations=N)` repeats this bounded sequence and returns the
per-iteration results.

The low-level `WorkerQueue.acquire(...)` and `WorkerQueuePoller` selectors are
queue-control primitives. They can target a known `job_id`, `project_id`, or
`run_id` when a caller already owns the queue semantics. Higher-level run-plan
helpers intentionally expose a narrower contract.

The poller can also bind to a task runner explicitly:

```python
from ai4s_agent.worker_queue_poller import WorkerQueuePoller
from ai4s_agent.worker_task_runner import FakeWorkerTaskRunner

poller = WorkerQueuePoller(queue, worker_id="worker-a", runner=FakeWorkerTaskRunner())
result = poller.poll_once()
```

When `runner` is omitted, the poller keeps the original control-plane-only
behavior. When `runner` is provided, the poller starts newly acquired jobs,
polls active jobs, propagates succeeded/failed/cancelled terminal states to the
queue, and does not heartbeat cancellation-requested jobs.

Local supervisor adapter:

```python
from ai4s_agent.worker_supervisor import WorkerSupervisor
from ai4s_agent.worker_task_runner import WorkerSupervisorTaskRunner

runner = WorkerSupervisorTaskRunner(
    supervisor=WorkerSupervisor(projects_root="/path/to/projects"),
    allowed_cwd_root="/path/to/workspace",
)
```

`WorkerSupervisorTaskRunner` adapts the runner protocol to
`WorkerSupervisor`. It is limited to local dummy/process commands. It rejects
shell string commands, maps exit 0 to succeeded and non-zero exits to failed,
handles SIGTERM/SIGKILL cancellation through the supervisor, and can require
task `cwd` values to resolve under `allowed_cwd_root`.

Skeleton behavior:

- Queued jobs are acquired in deterministic order by `created_at`, then
  `job_id`.
- Acquired jobs receive `lease_id`, `worker_id`, and `heartbeat_at`.
- Cancelled queued jobs move to `cancelled` and are skipped by acquire.
- Cancelled running jobs keep running status but expose
  `cancellation_requested=true` for future worker polling.
- Stale active leases are marked `stale`, and their jobs are requeued for a new
  worker to acquire.

Lease attempts versus explicit retry:

- `attempts` counts queue acquisitions only.
- Reacquisition after stale lease recovery increments `attempts`.
- Stale lease recovery keeps the same `job_id` and is not an explicit retry.
- Any future explicit retry for queued canary must create a new `job_id` and
  keep the original failed job immutable.
- `WAITING_USER` remains a resume/gate path, not a retry path.
- PR #138 adds atomic one-shot retry-child creation for failed local queue
  jobs. The original failed job remains immutable, the retry child receives a
  new `job_id`, and no automatic retry path is introduced.
- `enqueue_queued_canary_retry(...)` applies the existing run-plan queue
  envelope validation plus queued-canary allowlist checks before delegating to
  the low-level queue mutation.
- See `docs/queued-canary-retry-requeue-semantics.md` for the conservative
  queued-canary retry/requeue contract.

Out of scope:

- default `/api/run-plan/execute` integration
- API routes
- Remote worker contracts
- SQLite store implementation
- Real model training or adapter execution

Next phase:

1. Add a queue + poller + supervisor runner integration test proving a queued
   local dummy command can move from queued to terminal state.
2. Add a `LocalWorkerLoop` wrapper for bounded local polling iterations without
   API route coupling.
3. Define a run-plan queue job schema.
4. Add a one-shot `RunPlanExecutorTaskRunner` adapter for the run-plan queue
   task envelope.
5. Add an internal enqueue helper that converts a `RunPlan` into a queued
   `run_plan_execute` worker job.
6. Add an internal opt-in queued run-plan execution helper that composes the
   enqueue helper, `WorkerQueuePoller`, `LocalWorkerLoop`, and
   `RunPlanExecutorTaskRunner` without exposing API routes.

Run-plan queue job schema:

```python
from ai4s_agent.run_plan_queue import build_run_plan_execute_task

task = build_run_plan_execute_task(
    project_id="project-a",
    run_id="run-1",
    run_plan=run_plan,
    input_artifacts={"dataset": "datasets/input.csv"},
    task_options={"train_model": {"epochs": 1}},
)
```

The schema is a serializable queued task envelope only. It includes
`task_id="run_plan_execute"`, `kind="run_plan_execute"`, project/run identity,
the validated `RunPlan`, `input_artifacts`, and `task_options`. It deliberately
does not include `command`, local argv, cwd, shell strings, `RunPlanExecutor`
calls, API routing, remote worker fields, or SQLite behavior.

Run-plan queue enqueue helper:

```python
from ai4s_agent.run_plan_queue import enqueue_run_plan_execute_job

job = enqueue_run_plan_execute_job(
    queue,
    project_id="project-a",
    run_plan=run_plan,
    input_artifacts={"dataset": "datasets/input.csv"},
    task_options={"train_model": {"epochs": 1}},
)
```

The helper is internal plumbing only. It derives `run_id` from the `RunPlan`,
builds a validated `run_plan_execute` envelope, calls `WorkerQueue.enqueue(...)`,
and returns the queued job. It does not call `RunPlanExecutor`, start a
`LocalWorkerLoop`, add API routes, or change `/api/run-plan/execute`.

Internal queued run-plan execution helper:

```python
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue

summary = run_run_plan_via_local_queue(
    queue=queue,
    storage=project_storage,
    project_id="project-a",
    run_plan=run_plan,
    input_artifacts={"dataset": "datasets/input.csv"},
    task_options={"train_model": {"epochs": 1}},
    executor_factory=fake_or_real_executor_factory,
)
```

This helper is opt-in internal service plumbing. It enqueues a validated
`run_plan_execute` job, binds a `RunPlanExecutorTaskRunner` to a
`WorkerQueuePoller`, runs a bounded `LocalWorkerLoop`, and returns queue/lease
terminal state plus poll actions. By default the helper requires a dedicated
queue with no existing queued or running jobs. If `require_empty_queue=False`,
it still targets only the newly created job. Callers cannot provide an external
`target_job_id`; any optional project/run target selectors must match the
helper's own `project_id` and `run_plan.run_id` before enqueue. This prevents a
failed selector from leaving an orphan queued run-plan job. The service helper
does not add API routes, does not change the default synchronous
`/api/run-plan/execute` path, does not connect remote workers, and does not
change storage.

Internal queued run-plan CLI:

```bash
python -m ai4s_agent.run_plan_queue_cli \
  --workspace /path/to/workspace \
  --queue-dir /path/to/dedicated-queue \
  --project-id project-a \
  --run-plan-json /path/to/run_plan.json \
  --input-artifacts-json /path/to/input_artifacts.json \
  --task-options-json /path/to/task_options.json \
  --max-iterations 10
```

## Queued execute canary observability and rollback

`POST /api/run-plan/execute` remains synchronous by default. When
`AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY` is enabled, the flag acts as a
master switch only. The route uses the same local run-plan queue service helper
only when every `run_plan.tasks[].task_id` is on the low-risk canary allowlist.
The initial allowlist is:

- `inspect_dataset`
- `clean_dataset`
- `check_trainability`
- `run_baseline`
- `render_report`

Allowlisted queued responses return the normal `execution` field plus
`execution_backend="queued_canary"` and `queue_summary`. The canary uses the
service-level invariant that only the newly enqueued job can be processed, so a
non-empty queue cannot redirect the request to an older job.

If the flag is enabled but the task chain contains `train_model`, generation,
literature/mining tasks, unknown task ids, or any other non-allowlisted task,
the request falls back to the synchronous executor path. This fallback preserves
the sync response shape: no `execution_backend` and no `queue_summary`. It
records `RunPlan execution backend: sync_fallback_not_allowlisted` plus the
disallowed task ids in run logs. Turning the flag off immediately restores the
synchronous route for all task chains. The canary does not change
`/api/run-plan/resume`, does not enable remote workers, and does not migrate
queue storage to SQLite.

The rollout policy and decision matrix are documented in
`docs/queued-execute-canary-rollout-policy.md`. The allowlist is a policy gate,
not a statement that the queue bridge can safely run every task. Queued canary
execution is still not default migration, and sync fallback response
compatibility is part of the rollout criteria.

Failed queued executor results are not treated as successful route execution:
the canary returns `ok=false`, a failed execution payload, and the queue summary
for review.

Observability:

- With the flag off, the response remains backward compatible and does not
  include `execution_backend` or `queue_summary`.
- The sync path records a `RunPlan execution backend: sync` log marker while
  preserving the existing `RunPlan execution started` log.
- With the flag on and the task chain allowlisted, the response includes
  `execution_backend="queued_canary"` and `queue_summary`.
- The queued canary path records a `RunPlan execution backend: queued_canary`
  log marker and still logs terminal `WAITING_USER`, `FAILED`, or completed
  execution status.
- With the flag on and the task chain not allowlisted, the fallback path records
  `sync_fallback_not_allowlisted`, logs the disallowed task ids, and returns the
  same response shape as synchronous execution.

Rollback:

- Disable `AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY`.
- Requests immediately return to the synchronous executor path.
- Sync responses do not include `queue_summary`.
- Sync runs do not create per-run queue files under
  `.ai4s_internal/run_plan_queues/<project_id>/<run_id>`.
- No remote worker or SQLite migration is involved.

The CLI is an internal-only module entrypoint for local debugging and controlled
experiments. It reads a `RunPlan` JSON file, creates a file-backed dedicated
`WorkerQueue`, calls `run_run_plan_via_local_queue(...)`, prints a JSON summary,
and exits `0` only when the queued helper returns `ok=true` and `terminal=true`.
Optional `--input-artifacts-json` and `--task-options-json` files must have JSON
object roots. It does not add Flask/API routes, does not change
`/api/run-plan/execute`, does not connect remote workers, and does not change
storage.

The printed JSON summary is validated by `RunPlanQueueExecutionSummary` and has
a stable top-level shape:

```json
{
  "ok": true,
  "terminal": true,
  "queued_job_id": "job-demo-project-demo-run",
  "final_job": {},
  "final_lease": {},
  "loop_results": ["completed", "idle"],
  "error": null
}
```

Validation/input failures use the same schema with `ok=false`,
`terminal=false`, empty `queued_job_id`, null final state fields, an empty
`loop_results` list, and an `error` object containing `type` and `message`.

Internal queued run-plan API route:

```text
POST /api/internal/run-plan/queue/execute
```

This route is internal-only and disabled by default. It is available only when
`AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE` is truthy in app config or the
environment. It accepts `project_id`, `run_plan`, optional `input_artifacts`,
optional `task_options`, and optional `max_iterations`, then returns a
`RunPlanQueueExecutionSummary`. The route always uses an internal queue path
under `workspace/.ai4s_internal/run_plan_queues/<project_id>/<run_id>` and
rejects request-supplied `queue_dir` values. `project_id` and `run_id` must be
single safe path components, so traversal names and path separators are rejected
before the internal queue path is constructed. It intentionally does not replace
or alter the default synchronous `/api/run-plan/execute` route.

The route requires actor identity through the shared actor resolver. Supported
sources include `X-Actor`, JSON/form `actor`, `approved_by`, `revoked_by`, and
`confirmed_by`, plus query `actor`. It also requires an explicit server grant
for the `run_plan_queue_execute` permission action scoped to the project/run.
Actor-present requests without an active grant return a stable
`permission_denied` summary with HTTP 403 and do not call the executor. Each
feature-flag-enabled request writes an append-only audit record to
`workspace/.ai4s_internal/audit/internal_run_plan_queue_audit.jsonl` with actor,
actor source, project/run identity, outcome, status code, queued job id, and
error metadata when present. Permission decisions are recorded in the same event
with allowed/reason/action/resource/grant metadata, including
`permission_resource="project:<project_id>:run:<run_id>"`. Valid execution
requests write a `requested` audit event before queue execution starts; if that
write fails, the route fails closed without calling the executor or mutating the
worker queue. Terminal `succeeded`/`failed` events are written after queued
execution finishes.

Read-only internal queue status:

```text
GET /api/internal/run-plan/queue/status?project_id=<project_id>&run_id=<run_id>
```

This route is also internal-only and disabled unless
`AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE` is enabled. It requires actor
identity and the same `run_plan_queue_execute` server grant as execute. It does
not call the executor, mutate the queue, or expose cleanup/recovery operations.
The response contains `jobs`, `leases`, `counts`, `has_active_jobs`, and
`has_terminal_jobs`.

Lifecycle helpers:

- `internal_run_plan_queue_dir(workspace, project_id, run_id)` builds the
  internal queue path with the same safe path component and containment rules as
  the route.
- `read_run_plan_queue_status(queue)` returns job/lease records, status counts,
  and active/terminal booleans.
- `recover_stale_run_plan_queue(queue, now=...)` wraps
  `WorkerQueue.recover_stale_leases(...)` and returns a stable result with
  `ok`, `recovered_job_ids`, `recovered_count`, and `error`.
- `cleanup_terminal_run_plan_queue(queue, workspace=...)` removes terminal
  succeeded/failed/cancelled job records and terminal lease records under the
  queue lock. It returns a stable result with removed ids/counts,
  `deleted_files`, `has_active_jobs`, and `error`.

Cleanup/recovery helpers remain internal. Do not expose mutating cleanup or
recovery routes until their permission, actor, and audit behavior is explicitly
specified. Cleanup never deletes active queued/running jobs or active leases,
fails closed on malformed queue/lease JSON, and deletes queue/lease files only
when all records are terminal and safely removable.

Phase 1 queued workflow fixture demo:

- `tests/fixtures/phase1_queued_workflow_demo/` contains a small training CSV,
  candidate CSV, run-plan JSON, input artifact JSON, and task options JSON.
- `tests/test_phase1_queued_workflow_demo.py` invokes the feature-flagged
  internal queued execution route with actor identity and a
  `run_plan_queue_execute` server grant.
- The demo writes real Phase 1 artifacts: cleaned dataset, baseline metrics,
  lightweight baseline model metadata, candidate predictions, ranked
  candidates, final report files, artifact registry entries, queue status, and
  requested/succeeded audit records.
- The demo is a productization fixture for the existing Phase 1 path. It does
  not replace `/api/run-plan/execute`, does not expose cleanup/recovery routes,
  does not connect remote workers, does not use SQLite, and does not run heavy
  Uni-Mol/DPA3 training.

Phase 2 deterministic generation screening fixture demo:

- `tests/fixtures/phase2_generation_screening_demo/` contains a run-plan JSON,
  input artifact JSON, and task options JSON for a local generation-to-screening
  chain.
- `tests/test_phase2_generation_screening_demo.py` invokes the same
  feature-flagged internal queued execution route with actor identity and a
  `run_plan_queue_execute` server grant.
- The demo trains a lightweight Phase 1 baseline model from the fixture training
  dataset, runs `generate_candidates_stub_adapter`, registers the generated
  CSV as `candidate_dataset`, predicts candidate properties, ranks candidates,
  and renders a report.
- The demo writes real generation report, generated candidates, predictions,
  ranked candidates, final report files, queue status, and requested/succeeded
  audit records.
- This is a local low-risk Phase 2 bridge fixture. It does not execute
  REINVENT4, does not connect remote workers, does not replace
  `/api/run-plan/execute`, and does not claim full inverse-design automation.

Phase 3 literature-to-dataset fixture demo:

- `tests/fixtures/phase3_literature_dataset_demo/` contains a parsed document
  fixture, target property metadata, and extraction config for a small OLED-like
  PLQY table.
- `tests/test_phase3_literature_dataset_demo.py` runs a local fixture pipeline
  that extracts table rows into `ExtractedRecord` objects, normalizes PLQY
  percentages to fractions, merges duplicate molecule/property observations,
  writes conflict and benchmark reports, and exports a confirmed training CSV.
- The demo writes real Phase 3 artifacts: `extracted_records.jsonl`,
  `extracted_records.json`, `unit_normalization_report.json`,
  `conflict_report.json`, `merged_records.json`, `confirmed_dataset.csv`,
  `extraction_benchmark_report.json`, `report.md`, and `report.json`.
- The confirmed dataset is fed into Phase 1 `inspect_dataset_service` and
  `check_trainability_service` to verify that `plqy` is recognized as a
  trainable numeric property without running heavy model training.
- This is a local low-risk Phase 3 fixture. It does not perform Web Search,
  network acquisition, MinerU parsing, large-scale PDF crawling, remote worker
  execution, SQLite migration, or default route replacement.

OLED property profile and multi-objective screening fixture demo:

- `tests/fixtures/oled_property_profiles/oled_properties.json` defines a
  fixture OLED property profile with configurable property metadata, aliases,
  canonical units, optimization direction, ranking defaults, risk notes, and
  recommended task type.
- The profile includes `plqy`, `lambda_em_nm`, `homo_ev`, `lumo_ev`, and
  `delta_e_st_ev` as data-configured properties. These are not core schema enum
  restrictions and do not prevent future workflows from using other
  `property_id` values.
- `tests/fixtures/oled_multiobjective_screening_demo/` contains a small
  multi-property training CSV, candidate CSV, run-plan JSON, input artifact
  JSON, and task options JSON.
- `tests/test_oled_multiobjective_screening_demo.py` invokes the
  feature-flagged internal queued execution route with actor identity and a
  `run_plan_queue_execute` server grant. The test-local executor reuses
  existing Phase 1 adapters by training/predicting one property at a time,
  merges predictions into `multi_property_predictions.csv`, computes
  profile-driven objective score contributions, ranks candidates, and renders
  a report.
- This fixture uses multiple single-property predictions plus weighted
  multi-objective ranking. It does not implement full multi-task model
  training, does not let an LLM write executable code at runtime, does not
  connect REINVENT4/Web Search/MinerU/remote workers, and does not replace
  `/api/run-plan/execute`.

Run-plan artifact Observer-Verifier:

- `ai4s_agent.run_plan_artifact_verifier.verify_run_plan_artifacts(...)` is a
  read-only observer-verifier for queued workflow artifacts.
- It consumes optional queue execution summary/status payloads, internal
  run-plan queue audit records, the project artifact registry, and known
  artifact reports. It reads trainability reports, model metrics, generation
  reports, extraction benchmark reports, and multi-objective ranking CSVs when
  present.
- It returns a fixed `RunPlanArtifactVerification` schema with one decision:
  `continue`, `needs_review`, `rerun_recommended`, or `blocked`.
- This layer does not execute adapters, mutate queues, call LLMs, or author a
  revised plan. Downstream LLM/planner components should consume the fixed
  verifier result and produce only reviewable replan proposals.

Reviewable replan proposal:

- `ai4s_agent.run_plan_replan_proposal.propose_replan_from_verification(...)`
  consumes only a `RunPlanArtifactVerification` payload.
- It returns a fixed `RunPlanReplanProposal` schema with
  `decision_source="verifier"`, the original verifier decision, a deterministic
  `proposed_action`, affected tasks, rationale, required user decisions, and an
  unapplied `proposed_run_plan_patch`.
- `executable` is always `false`; attempts to validate a proposal with
  `executable=true` are rejected.
- The first implementation is deterministic and rule-based. It does not call an
  LLM, mutate a `RunPlan`, execute adapters, enqueue work, or automatically
  rerun tasks.
- Future planner/LLM layers may explain or refine these proposals, but the next
  executable step must still pass through explicit user review plus the
  existing gate/resume or modified-run-plan path.

Review artifacts:

- `ai4s_agent.run_plan_review_artifacts.write_run_plan_review_artifacts(...)`
  links the read-only Observer-Verifier and deterministic replan proposal into
  review artifacts under the run directory.
- It writes `review/observer_verification.json`,
  `review/replan_proposal.json`, and `review/replan_review.md`, then registers
  them as `observer_verification`, `replan_proposal`, and
  `replan_review_markdown` artifacts.
- These files are intended for UI, report, and project-memory consumption. They
  are not executable state and do not apply the proposed patch.
- The writer does not execute adapters, call LLMs, mutate `RunPlan`, enqueue
  work, auto-rerun tasks, or replace `/api/run-plan/execute`.

Review card aggregation:

- `ai4s_agent.run_plan_review_card.read_run_plan_review_card(...)` is a
  read-only aggregation helper for previously written review artifacts.
- It reads `observer_verification.json`, `replan_proposal.json`, and
  `replan_review.md`, then returns one `RunPlanReviewCard` schema for UI,
  report, or project-memory consumers.
- `GET /api/internal/run-plan/review-card?project_id=...&run_id=...` exposes
  the same card behind the internal run-plan queue feature flag, actor identity,
  and `run_plan_queue_execute` permission grant.
- The card route does not call an executor, write review artifacts, apply the
  proposed patch, enqueue work, mutate `RunPlan`, call LLMs, or replace
  `/api/run-plan/execute`.

Project memory summary:

- `ai4s_agent.run_plan_review_memory.save_run_plan_review_card_summary_to_memory(...)`
  stores a compact `ProjectMemoryRecord` from a `RunPlanReviewCard`.
- The saved memory record uses category `run_plan_review` and includes only the
  verifier decision, proposed action, affected tasks, required user decisions,
  and artifact references.
- It does not store raw CSV/data, full artifact contents, complete verifier or
  proposal payloads, markdown bodies, or executable patches.
- The memory integration does not execute proposals, apply patches, call LLMs,
  enqueue work, mutate `RunPlan`, or replace `/api/run-plan/execute`.

Replan application artifacts, audit, and memory:

- `ai4s_agent.run_plan_replan_application_artifacts.write_replan_application_artifacts(...)`
  materializes a compiled, user-confirmed application draft as review artifacts
  only.
- `ai4s_agent.run_plan_replan_application_audit_memory.append_replan_application_audit_record(...)`
  appends compact `replan_application_requested`,
  `replan_application_completed`, or `replan_application_failed` audit events
  under the run `review/` directory.
- `ai4s_agent.run_plan_replan_application_audit_memory.save_replan_application_summary_to_memory(...)`
  stores a compact project-memory summary with selected action, result type,
  selected operation ids, affected tasks, required gates, artifact refs, and
  audit refs.
- These helpers do not expose public routes, execute adapters, apply patches,
  call LLMs, enqueue work, mutate `RunPlan`, store raw data, or replace
  `/api/run-plan/execute`.

Internal replan application review route:

- `POST /api/internal/run-plan/replan/apply-review` is an internal-only route
  for materializing a user-confirmed replan application review draft.
- It requires `AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE`, actor identity, and
  a `run_plan_replan_apply` server permission grant.
- The route accepts a `ReplanApplicationRequest`, writes a requested audit
  record, calls `write_replan_application_artifacts(...)`, writes completed or
  failed audit records, saves a compact project-memory summary, and returns a
  compact application bundle summary.
- The route does not execute adapters, enqueue work, auto-resume, apply the
  proposed patch, mutate `RunPlan`, call LLMs, or replace
  `/api/run-plan/execute`.

Resume intent validation design:

- `docs/resume-intent-validation-semantics.md` defines how a future
  gate/resume path should validate `review/replan_resume_intent.json`.
- The design requires source application linkage, proposal hash verification,
  artifact registry checks, current `RunPlan` compatibility, stale-intent
  detection, permission, gate validation, and fail-closed resume audit.
- This is still design-only. It does not add a route, call
  `RunPlanExecutor.resume_after_gate(...)`, enqueue work, write gate decisions,
  mutate `RunPlan`, call LLMs, or replace `/api/run-plan/resume` or
  `/api/run-plan/execute`.

Resume intent validation schemas, helpers, audit, memory, and internal route:

- `ai4s_agent.run_plan_state_fingerprint.ResumeStateBinding` records compact
  run-plan and stage-state fingerprints when a user-confirmed replan
  application creates a resume intent.
- `run_plan_fingerprint(...)` hashes the complete schema-normalized `RunPlan`;
  `stage_state_fingerprint(...)` hashes stable stage semantics while ignoring
  volatile timestamps.
- `ai4s_agent.run_plan_resume_intent_validation.validate_resume_intent(...)`
  validates a materialized resume intent against the current `RunPlan`, review
  artifacts, optional stage state, optional audit records, and optional
  approved gates.
- `ai4s_agent.run_plan_resume_stage_gate.WaitingStageGateContext` validates
  that the current stage is still `WAITING_USER`, present in the current
  `RunPlan`, known to `AtomicTaskRegistry`, and backed by a complete execution
  snapshot whose material hash and required gates still match the current
  executor contract.
- `ResumeIntent.application_required_gates` records review/application gates;
  `ResumeIntent.required_gates` records executor gates for the waiting task.
  `ResumeIntent.approved_gates` must stay empty. The internal validation route
  accepts approved gates only from the request payload and rejects duplicates,
  non-string values, application gates, and gates outside the current executor
  gate set.
- `ai4s_agent.run_plan_resume_intent_validation_audit_memory.append_resume_intent_validation_audit_record(...)`
  appends compact requested/completed/failed validation audit records under the
  run `review/` directory.
- `ai4s_agent.run_plan_resume_intent_validation_audit_memory.save_resume_intent_validation_summary_to_memory(...)`
  stores a compact project-memory summary with the validation decision,
  intent/application/proposal identifiers, gates, rerun tasks, artifact refs,
  audit refs, and error shape.
- `POST /api/internal/run-plan/resume-intent/validate` exposes this validation
  as an internal-only route behind
  `AI4S_ENABLE_INTERNAL_RESUME_INTENT_VALIDATION_ROUTE`, actor identity, and a
  `run_plan_resume_intent_use` server permission grant.
- The route returns a `ResumeIntentValidationResult` wrapper and writes
  validation audit/memory summaries only. It does not call
  `RunPlanExecutor.resume_after_gate(...)`, write gate decisions, enqueue work,
  execute adapters, mutate `RunPlan`, call LLMs, or replace
  `/api/run-plan/resume` or `/api/run-plan/execute`.
- `POST /api/internal/run-plan/resume-intent/execute` is a separate
  feature-flagged internal bridge behind
  `AI4S_ENABLE_INTERNAL_RESUME_INTENT_EXECUTE_ROUTE`, actor identity, and a
  `run_plan_resume_execute` server grant. It server-loads the current
  `RunPlan`, `StageState`, and review artifacts, reruns
  `validate_resume_intent(...)`, requires `decision="resume_eligible"`, writes
  a pre-execution `resume_intent_consumed` audit record fail-closed, then calls
  the existing `RunPlanExecutor.resume_after_gate(...)`. Gate decisions are
  written only by the executor. The bridge does not enqueue work, mutate
  `RunPlan`, call LLMs, replace `/api/run-plan/resume`, or replace
  `/api/run-plan/execute`.
- Fingerprints are stale-state detection only. They are not signatures,
  permission grants, or execution authorization. Any future default-route or
  queued resume bridge must recompute them again immediately before execution.

Queued `WAITING_USER` contract:

- `RunPlanExecutorTaskRunner` treats `RunPlanExecutor` output with
  `status="WAITING_USER"` as terminal-compatible for the first queued bridge
  version.
- The underlying worker queue job is completed with `job.status="succeeded"`
  and the lease is completed, preserving current queue cleanup and status
  semantics.
- The job `result`, `RunPlanQueueExecutionSummary`, internal status route, and
  audit records expose `waiting_user=true`, `waiting_task`, and
  `required_gates`.
- Terminal audit outcome is `waiting_user`, not `failed`, when queued execution
  pauses for a gate/user decision.
- This contract does not implement a full queued resume engine. Resume behavior
  remains a separate future bridge after waiting-user state, actor, permission,
  and audit rules are stable.

Low-risk fixture demo:

```bash
python -m ai4s_agent.run_plan_queue_cli \
  --workspace /tmp/ai4s-demo-workspace \
  --queue-dir /tmp/ai4s-demo-queue \
  --project-id demo-project \
  --run-plan-json tests/fixtures/run_plan_queue_demo/run_plan.json \
  --input-artifacts-json tests/fixtures/run_plan_queue_demo/input_artifacts.json \
  --task-options-json tests/fixtures/run_plan_queue_demo/task_options.json \
  --max-iterations 10
```

The fixture files are intentionally command-free and low risk. The automated
demo test invokes the same CLI `main(...)` path with a fake executor, so it
proves the documented shape and JSON summary contract without running real
training or adding API routes.

Run-plan executor task runner:

```python
from ai4s_agent.run_plan_task_runner import RunPlanExecutorTaskRunner

runner = RunPlanExecutorTaskRunner(storage=project_storage)
result = runner.start(worker_job)
```

`RunPlanExecutorTaskRunner` is a one-shot `WorkerTaskRunner` adapter. It
validates `worker_job["task"]` as a `run_plan_execute` envelope and synchronously
calls `RunPlanExecutor.execute(...)`. It does not alter `/api/run-plan/execute`,
does not add API routes, does not support remote workers, and does not implement
SQLite behavior. `poll()` fails fast because the runner is one-shot; `cancel()`
reports cancellation as unsupported rather than pretending it can interrupt a
synchronous executor call.

Do not jump directly to remote workers or SQLite from this state. Remote worker
contracts should wait for the local run-plan bridge; SQLite should wait for
stable file-backed queue and runner semantics under the opt-in bridge.

Route phase status:

The internal run-plan queue bridge is complete for local opt-in use. It now has
a validated queue task schema, enqueue helper, one-shot
`RunPlanExecutorTaskRunner`, local queue service, internal CLI, stable
`RunPlanQueueExecutionSummary`, feature-flagged internal route, and minimal
actor/audit/permission metadata plus read-only lifecycle observability. These
pieces prove the route/CLI/service control path can write and inspect queue and
lease terminal state without changing the default run-plan execution route.

Still not default:

- `/api/run-plan/execute` remains synchronous.
- `POST /api/internal/run-plan/queue/execute` requires
  `AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE`, actor identity, and a
  `run_plan_queue_execute` server grant.
- No remote worker is connected.
- No SQLite queue store exists.
- Fake executor and low-risk CLI tests do not guarantee real model training
  success.
- `WAITING_USER` is terminal-compatible in the queue, but resumable queued
  execution is not implemented yet.

Default-route migration hard gates:

1. `RunPlanExecutorTaskRunner` must run a real low-risk adapter demo end to end.
2. The internal route must enforce permission, actor identity, and audit
   semantics suitable for queued execution.
3. Queue lifecycle must expose cleanup, stale recovery, and observability for
   queued/running jobs.
4. `RunPlanQueueExecutionSummary` must be validated consistently by route, CLI,
   service helper, and tests.
5. The dedicated queue limitation is resolved for the internal run-plan helper
   by targeting only the newly created job; the default execute route canary
   uses the same invariant while it remains feature-flagged.
6. `waiting_user` uses the current compatibility contract: terminal succeeded
   queue state plus explicit waiting metadata in summary, status, and audit.
   A full queued resume engine remains future work.
