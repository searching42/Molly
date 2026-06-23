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

- `RunPlanExecutor` integration
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
4. Add a `RunPlanExecutor` opt-in bridge after the local loop and job schema are
   covered by tests.

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

Do not jump directly to remote workers or SQLite from this state. Remote worker
contracts should wait for the local run-plan bridge; SQLite should wait for
stable file-backed queue and runner semantics under the opt-in bridge.
