# OLED Local Demo Generic Enqueue

`ai4s_agent.agents.oled_local_demo_generic_enqueue` submits one OLED local demo job through the generic `run_plan_execute` queue envelope.

This command is enqueue-only. It builds a one-task RunPlan containing `execute_oled_local_demo`, stores it as a generic `run_plan_execute` queue job, and exits. It does not execute the job, poll the queue, instantiate ProjectStorage, instantiate `RunPlanExecutorTaskRunner`, or read the input bundle file.

## Example

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_generic_enqueue \
  --queue-root /tmp/molly-worker-queue \
  --project-id demo-project \
  --run-id oled-generic-queue-demo \
  --input-bundle /tmp/oled_demo_bundle.json \
  --output-dir /tmp/oled-agent-demo \
  --overwrite
```

Then consume the queued generic job with the bounded worker loop:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_generic_worker_loop \
  --queue-root /tmp/molly-worker-queue \
  --project-root /tmp/molly-projects \
  --worker-id generic-worker-1 \
  --max-iterations 3
```

Use `ai4s_agent.agents.oled_local_demo_generic_status` to inspect queued, running, succeeded, failed, or cancelled generic OLED local demo jobs without executing or mutating them.

Use `ai4s_agent.agents.oled_local_demo_generic_retry` when a failed generic OLED local demo job needs an explicit retry child queued for later worker-loop consumption.

Use `ai4s_agent.agents.oled_local_demo_generic_cancel` to cancel queued allowlisted generic OLED local demo jobs before they are consumed.

## Behavior

The enqueued queue task uses:

- `task_id: run_plan_execute`
- `kind: run_plan_execute`
- embedded RunPlan task: `execute_oled_local_demo`
- task options for `input_bundle`, `output_dir`, `goal`, `overwrite`, and `project_id`

The underlying queue controls timestamps. The CLI does not expose custom timestamp control.

## Safety Boundary

This command only enqueues a generic RunPlan queue job. It does not execute `RunPlanExecutor`, instantiate `RunPlanExecutorTaskRunner`, instantiate ProjectStorage, use `LocalWorkerLoop`, use `WorkerQueuePoller`, execute adapters, read input bundle files, open artifact labels, use the OLED-specific `execute_oled_local_demo_runplan` job type, call MinerU, parse PDFs/images, scan corpora, call LLMs, use network access, train models, predict candidates, approve gates, resume gates, spawn subprocesses, start a daemon, run an infinite loop, or mutate registry/promotion/publication/release/global append artifacts.
