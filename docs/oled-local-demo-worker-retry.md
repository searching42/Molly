# OLED Local Demo Worker Retry CLI

`ai4s_agent.agents.oled_local_demo_retry` enqueues a retry child for a failed OLED local demo worker job.

It opens the local worker queue, validates that the source job failed and uses `execute_oled_local_demo_runplan`, then delegates child creation to `WorkerQueue.enqueue_retry_of_failed_job(...)`. It prints compact JSON and leaves the retry child queued for a later bounded worker loop.

## Retry Flow

Retry a failed job:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_retry \
  --queue-root /tmp/molly-worker-queue \
  --source-job-id job-demo-project-oled-local-demo \
  --retry-request-id retry-001 \
  --requested-by benton \
  --reason "fixed stale output directory"
```

Then consume the retry child:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_worker_loop \
  --queue-root /tmp/molly-worker-queue \
  --worker-id local-worker-1 \
  --max-iterations 3
```

## What Is Preserved

The retry child copies the source task payload exactly. It records retry metadata such as:

- `retry_of_job_id`
- `retry_root_job_id`
- `retry_request_id`
- `retry_reason`
- `retry_requested_by`

The CLI does not modify `project_root`, `input_bundle`, `output_dir`, `goal`, `overwrite`, or any other task payload field.

## Safety Boundary

This command is retry enqueue-only. It does not poll the queue, run a loop, execute `RunPlanExecutor`, execute adapters, read the input bundle file, read artifact labels inside the bundle, modify the copied task payload, create ProjectStorage stage state, write output reports, spawn subprocesses, start a daemon, approve gates, call MinerU, parse PDFs/images, scan corpora, call LLMs, use network access, train models, predict candidates, or mutate registry/promotion/publication/release/global append artifacts.

Use `ai4s_agent.agents.oled_local_demo_enqueue` to submit new jobs and `ai4s_agent.agents.oled_local_demo_worker_loop` to consume queued jobs and retry children.

Use `ai4s_agent.agents.oled_local_demo_cancel` when a queued OLED local demo job should be cancelled or a running job should be marked for cancellation instead of retried.
