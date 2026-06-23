# Worker Queue Skeleton

HARDEN-012 is control-plane resolved across PR #65 through PR #68.  The queue
can persist jobs, lease them to workers, poll for heartbeat/cancellation/recovery
state, and validate queue files.  It still does not execute tasks, start worker
processes, call `RunPlanExecutor`, connect to remote workers, or migrate state
to SQLite.

Resolved PRs:

- PR #65: `WorkerQueue` / `JsonWorkerQueueStore` JSON-backed queue and lease
  state.
- PR #66: queue/lease record validation and storage consistency checker
  coverage.
- PR #67: `WorkerQueuePoller` bounded polling skeleton.
- PR #68: cancellation and stale recovery control transition tests.

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

Polling skeleton:

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
per-iteration results. It does not call any task runner.

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

- Process supervision integration
- Remote worker contracts
- Real model training or adapter execution
- SQLite store implementation

Next phase:

1. Define a `WorkerTaskRunner` protocol.
2. Add fake runner tests for success, failure, cancellation, and heartbeat
   cadence.
3. Bind `WorkerQueuePoller` to the runner protocol behind an explicit opt-in
   path.
4. Add a `WorkerSupervisorTaskRunner` adapter after the fake runner contract is
   stable.
5. Add `RunPlanExecutor` opt-in integration only after the local adapter is
   covered.

Do not jump directly to remote workers or SQLite from this state.  Remote worker
contracts should wait for local runner binding; SQLite should wait for stable
file-backed runner semantics and checker coverage.
