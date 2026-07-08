# OLED Local Demo Worker Loop CLI

`ai4s_agent.agents.oled_local_demo_worker_loop` runs a bounded local worker loop over existing queued OLED local demo jobs.

It uses:

```text
WorkerQueue
-> WorkerQueuePoller
-> LocalWorkerLoop
-> OLEDLocalDemoRunPlanWorkerTaskRunner
-> RunPlanExecutor
-> execute_oled_local_demo_adapter
```

## What It Does

The CLI opens an existing queue, polls with `OLEDLocalDemoRunPlanWorkerTaskRunner`, and stops when the queue is idle or `--max-iterations` is reached. It can consume queued jobs whose task is `execute_oled_local_demo_runplan`.

It does not enqueue jobs, start a daemon, run an infinite loop, sleep by default, spawn subprocesses, or execute scientific adapters.

## Example

Queue a job with the enqueue-only CLI:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_enqueue \
  --queue-root /tmp/molly-worker-queue \
  --project-root /tmp/molly-projects \
  --project-id demo-project \
  --run-id oled-local-demo \
  --input-bundle /tmp/oled_demo_bundle.json \
  --output-dir /tmp/oled-agent-demo \
  --overwrite
```

Or queue a job with Python:

```python
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue

queue = WorkerQueue(JsonWorkerQueueStore("/tmp/molly-worker-queue"))
queue.enqueue(
    "demo-project",
    "oled-local-demo",
    {
        "task_id": "execute_oled_local_demo_runplan",
        "project_root": "/tmp/molly-projects",
        "input_bundle": "/tmp/oled_demo_bundle.json",
        "output_dir": "/tmp/oled-agent-demo",
        "overwrite": True,
    },
)
```

Then consume the queued job with a bounded loop:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_worker_loop \
  --queue-root /tmp/molly-worker-queue \
  --worker-id local-worker-1 \
  --max-iterations 3
```

The command prints compact JSON with actions, completed/failed/cancelled job ids, recovered jobs, executed task ids, and whether the loop reached idle.

If a queued OLED local demo job fails, create a retry child with `ai4s_agent.agents.oled_local_demo_retry`, then run this bounded worker loop again to consume the retry.

If a queued or running OLED local demo job should be cancelled, use `ai4s_agent.agents.oled_local_demo_cancel`. Queued jobs are cancelled immediately; running jobs are marked `cancellation_requested` and this bounded worker loop handles the cancellation on a later poll.

Use `ai4s_agent.agents.oled_local_demo_status` to inspect existing OLED local demo jobs, leases, and optional ProjectStorage metadata before or after running the bounded loop.

## Targeted Polling

The bounded loop supports optional selectors:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_worker_loop \
  --queue-root /tmp/molly-worker-queue \
  --worker-id local-worker-1 \
  --max-iterations 3 \
  --target-run-id oled-local-demo
```

Selectors are passed to `WorkerQueuePoller`; non-matching jobs remain queued.

## Safety Boundary

This loop executes only already-queued OLED local demo jobs. Each executed job may read exactly one user-specified local summary bundle and write local demo reports, a manifest, RunPlanExecutor state, artifact registry entries, and worker queue job/lease state.

It does not read, open, hash, or validate artifact labels inside bundles. It does not call MinerU, parse PDFs/images, scan corpora, call LLMs, use network access, train models, predict candidates, approve gates, resume gates, spawn subprocesses, or mutate registry/promotion/publication/release/global append artifacts.

## Comparison

`ai4s_agent.agents.oled_local_demo_worker` is a one-shot helper that enqueues one job and immediately polls once. This bounded loop CLI is for consuming jobs that are already queued.
