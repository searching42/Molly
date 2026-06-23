# Worker Queue Skeleton

PR #65 starts HARDEN-012 with a read/write control-plane queue only. It does
not execute tasks, start worker processes, call `RunPlanExecutor`, or connect to
real training jobs.

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
- `list_jobs()`

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

- Worker polling loop
- Process supervision integration
- Remote worker contracts
- Real model training or adapter execution
- SQLite store implementation
