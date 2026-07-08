# OLED Local Demo Generic Cancel

`ai4s_agent.agents.oled_local_demo_generic_cancel` cancels or requests cancellation for generic `run_plan_execute` OLED local demo jobs.

This command accepts only generic queue jobs whose embedded RunPlan contains exactly `execute_oled_local_demo` in both `requested_tasks` and planned tasks. Queued jobs are cancelled immediately. Running jobs are marked `cancellation_requested=true`, and the bounded generic worker loop handles cancellation on a later poll.

## Example

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_generic_cancel \
  --queue-root /tmp/molly-worker-queue \
  --job-id job-demo-project-oled-generic-queue-demo
```

If the job was already running, run the bounded generic worker loop to observe the cancellation:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_generic_worker_loop \
  --queue-root /tmp/molly-worker-queue \
  --project-root /tmp/molly-projects \
  --worker-id generic-worker-1 \
  --max-iterations 3
```

## Workflow

Pair this command with:

- `ai4s_agent.agents.oled_local_demo_generic_enqueue` to submit jobs
- `ai4s_agent.agents.oled_local_demo_generic_worker_loop` to consume jobs and observe running-job cancellation
- `ai4s_agent.agents.oled_local_demo_generic_status` to inspect job state before and after cancellation
- `ai4s_agent.agents.oled_local_demo_generic_retry` to recover failed jobs when retry is appropriate

## Safety Boundary

This command does not poll the queue, execute `RunPlanExecutor`, instantiate `RunPlanExecutorTaskRunner`, instantiate ProjectStorage, use `LocalWorkerLoop`, use `WorkerQueuePoller`, execute adapters, read input bundle files, open artifact labels, modify task payloads, use the OLED-specific `execute_oled_local_demo_runplan` job type, call MinerU, parse PDFs/images, scan corpora, call LLMs, use network access, train models, predict candidates, approve gates, resume gates, spawn subprocesses, start a daemon, run an infinite loop, or mutate registry/promotion/publication/release/global append artifacts.
