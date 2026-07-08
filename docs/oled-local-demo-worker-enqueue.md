# OLED Local Demo Worker Enqueue CLI

`ai4s_agent.agents.oled_local_demo_enqueue` enqueues one OLED local demo worker job without executing it.

It creates or opens a local `WorkerQueue`, writes one queued `execute_oled_local_demo_runplan` job, and prints compact JSON. It does not poll the queue, start a worker, call `RunPlanExecutor`, execute adapters, create ProjectStorage run state, read the input bundle, or inspect artifact labels inside the bundle.

## Two-Step Queue Flow

Submit the job:

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

Later, consume existing queued jobs with the bounded worker loop:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_worker_loop \
  --queue-root /tmp/molly-worker-queue \
  --worker-id local-worker-1 \
  --max-iterations 3
```

If the worker loop marks the job failed, use `ai4s_agent.agents.oled_local_demo_retry` to enqueue a retry child after fixing the local cause.

## Job Payload

The enqueued job has task id `execute_oled_local_demo_runplan` and stores:

- `project_root`
- `input_bundle`
- `output_dir`
- `goal`
- `overwrite`

The queue-level `project_id` and `run_id` remain authoritative. The command stores `input_bundle` as a string only; the worker loop validates and reads the bundle later when the job executes.

## Safety Boundary

This command is enqueue-only. It does not poll, run a loop, execute `RunPlanExecutor`, execute adapters, read the input bundle file, read artifact labels inside the bundle, spawn subprocesses, start a daemon, approve gates, call MinerU, parse PDFs/images, scan corpora, call LLMs, use network access, train models, predict candidates, or mutate registry/promotion/publication/release/global append artifacts.
