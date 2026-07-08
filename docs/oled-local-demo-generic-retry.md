# OLED Local Demo Generic Retry

`ai4s_agent.agents.oled_local_demo_generic_retry` queues a retry child for a failed generic `run_plan_execute` OLED local demo job.

This command is retry enqueue-only. It accepts only failed generic queue jobs whose embedded RunPlan contains exactly `execute_oled_local_demo` in both `requested_tasks` and planned tasks. It does not execute the retry, poll the queue, read input bundles, open artifact labels, or modify the copied task payload.

## Example

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_generic_retry \
  --queue-root /tmp/molly-worker-queue \
  --source-job-id job-demo-project-oled-generic-queue-demo \
  --retry-request-id retry-001 \
  --requested-by benton \
  --reason "fixed stale output directory"
```

Then consume the retry child with the bounded generic worker loop:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_generic_worker_loop \
  --queue-root /tmp/molly-worker-queue \
  --project-root /tmp/molly-projects \
  --worker-id generic-worker-1 \
  --max-iterations 3
```

## Workflow

Pair this command with:

- `ai4s_agent.agents.oled_local_demo_generic_enqueue` to submit generic OLED jobs
- `ai4s_agent.agents.oled_local_demo_generic_worker_loop` to consume queued jobs and retry children
- `ai4s_agent.agents.oled_local_demo_generic_status` to inspect failed jobs and retry metadata

Duplicate `retry_request_id` values are idempotent for the same source job if the underlying worker queue returns the existing retry child. Reusing a retry request id for a different source job is rejected.

## Safety Boundary

This command does not poll the queue, execute `RunPlanExecutor`, instantiate `RunPlanExecutorTaskRunner`, instantiate ProjectStorage, use `LocalWorkerLoop`, use `WorkerQueuePoller`, execute adapters, read input bundle files, open artifact labels, modify the copied task payload, use the OLED-specific `execute_oled_local_demo_runplan` job type, call MinerU, parse PDFs/images, scan corpora, call LLMs, use network access, train models, predict candidates, approve gates, resume gates, spawn subprocesses, start a daemon, run an infinite loop, or mutate registry/promotion/publication/release/global append artifacts.
