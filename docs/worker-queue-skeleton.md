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
- `acquire(worker_id)`
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
4. acquire the next queued job if no active lease exists

`poll(max_iterations=N)` repeats this bounded sequence and returns the
per-iteration results.

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
terminal state plus poll actions. The helper requires a dedicated queue with no
existing queued or running jobs, because the current `WorkerQueuePoller` acquires
the next queued job rather than a specific target job. It does not add API
routes, does not change the default synchronous `/api/run-plan/execute` path,
does not connect remote workers, and does not change storage.

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
  `WorkerQueue.recover_stale_leases(...)` and returns recovered job ids/counts.
- `cleanup_terminal_run_plan_queue(queue, workspace=...)` removes terminal
  succeeded/failed/cancelled job records and terminal lease records. It never
  deletes active queued/running jobs, and deletes queue/lease files only when
  all records are terminal and safely removable.

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

Default-route migration hard gates:

1. `RunPlanExecutorTaskRunner` must run a real low-risk adapter demo end to end.
2. The internal route must enforce permission, actor identity, and audit
   semantics suitable for queued execution.
3. Queue lifecycle must expose cleanup, stale recovery, and observability for
   queued/running jobs.
4. `RunPlanQueueExecutionSummary` must be validated consistently by route, CLI,
   service helper, and tests.
5. The dedicated queue limitation must be resolved with dedicated per-request
   queues or target-job acquisition.
6. `waiting_user` must have an explicit contract: either terminal succeeded for
   compatibility or a non-terminal waiting state with resume semantics.
