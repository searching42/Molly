# OLED Local Demo Worker Queue Execution

`OLEDLocalDemoRunPlanWorkerTaskRunner` connects the OLED local demo RunPlanExecutor path to the existing local worker queue.

The execution chain is:

```text
WorkerQueue.enqueue(...)
-> WorkerQueuePoller.poll_once(...)
-> OLEDLocalDemoRunPlanWorkerTaskRunner.start(job)
-> execute_oled_local_demo_runplan(...)
-> RunPlanExecutor
-> execute_oled_local_demo_adapter
-> local demo reports + manifest + ProjectStorage state
-> queue marks job succeeded
```

## What It Executes

The runner executes the same low-risk `execute_oled_local_demo` task used by the direct CLI. It reads one local summary bundle and writes:

- `oled_agent_mvp_demo_bundle.json`
- `oled_agent_mvp_demo_bundle.md`
- `oled_local_demo_execution_manifest.json`
- RunPlanExecutor `stage.json`
- RunPlanExecutor `artifact_registry.json`
- `execute_oled_local_demo/adapter_result.json`
- worker queue job and lease state

## Safety Boundary

This worker runner does not execute scientific adapters, call MinerU, parse PDFs/images, scan corpora, call LLMs, use network access, train models, predict candidates, approve gates, resume gates, spawn subprocesses, or mutate registry/promotion/publication/release/global append artifacts.

Artifact labels inside the input bundle are summary placeholders. They are not opened, hashed, followed, or validated as real paths.

## Job Shape

```json
{
  "task_id": "execute_oled_local_demo_runplan",
  "project_root": "/tmp/molly-projects",
  "input_bundle": "/tmp/oled_demo_bundle.json",
  "output_dir": "/tmp/oled-agent-demo",
  "goal": "Find OLED emitters with high PLQY and red-shifted emission",
  "overwrite": true
}
```

The queue-level `project_id` and `run_id` are authoritative. Task payload values cannot override them.

## Example

```python
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePoller
from ai4s_agent.agents.oled_local_demo_worker import OLEDLocalDemoRunPlanWorkerTaskRunner

queue = WorkerQueue(JsonWorkerQueueStore("/tmp/molly-worker-queue"))
job = queue.enqueue(
    project_id="demo-project",
    run_id="oled-local-demo",
    task={
        "task_id": "execute_oled_local_demo_runplan",
        "project_root": "/tmp/molly-projects",
        "input_bundle": "/tmp/oled_demo_bundle.json",
        "output_dir": "/tmp/oled-agent-demo",
        "overwrite": True,
    },
)
poller = WorkerQueuePoller(
    queue,
    worker_id="local-worker-1",
    runner=OLEDLocalDemoRunPlanWorkerTaskRunner(),
)
result = poller.poll_once()
```

## One-Shot CLI

Run one bounded local queue execution from the command line:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_worker \
  --queue-root /tmp/molly-worker-queue \
  --project-root /tmp/molly-projects \
  --project-id demo-project \
  --run-id oled-local-demo \
  --input-bundle /tmp/oled_demo_bundle.json \
  --output-dir /tmp/oled-agent-demo \
  --worker-id local-worker-1 \
  --overwrite
```

This command creates a local `WorkerQueue`, enqueues one `execute_oled_local_demo_runplan` job, polls once with `OLEDLocalDemoRunPlanWorkerTaskRunner`, and prints compact JSON. It is a one-shot local queue execution command; it does not start a daemon, background worker, shell command, or long-running loop.

## Direct CLI Comparison

`ai4s_agent.agents.oled_local_demo_runplan` executes the same RunPlanExecutor path directly. `OLEDLocalDemoRunPlanWorkerTaskRunner` adds durable queue acquisition, lease completion, and queue result state around that same local task.
