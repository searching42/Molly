# OLED Local Demo Worker Cancel CLI

`ai4s_agent.agents.oled_local_demo_cancel` cancels or requests cancellation for one OLED local demo worker job.

It opens the local worker queue, validates that the target job uses `execute_oled_local_demo_runplan`, calls `WorkerQueue.cancel(...)`, and prints compact JSON. It does not poll the queue, execute `RunPlanExecutor`, execute adapters, read the input bundle, inspect artifact labels, create output files, or create ProjectStorage run state.

## Cancel Flow

Cancel a queued job or request cancellation for a running job:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_cancel \
  --queue-root /tmp/molly-worker-queue \
  --job-id job-demo-project-oled-local-demo
```

Then run the bounded worker loop if a running job needs to observe the cancellation request:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_worker_loop \
  --queue-root /tmp/molly-worker-queue \
  --worker-id local-worker-1 \
  --max-iterations 3
```

Queued jobs are marked `cancelled` immediately. Running jobs keep their running status but are marked `cancellation_requested=true`; the worker loop handles that active lease on a later poll by invoking the OLED local demo worker runner's synchronous cancel path.

## What Is Preserved

The command does not modify the job task payload. Fields such as `project_root`, `input_bundle`, `output_dir`, `goal`, and `overwrite` remain unchanged. The command only updates queue cancellation state.

## Safety Boundary

This command is cancel-only. It does not poll the queue, run a loop, execute `RunPlanExecutor`, execute adapters, read the input bundle file, read artifact labels inside the bundle, modify the task payload, create ProjectStorage stage state, write output reports, spawn subprocesses, start a daemon, approve gates, call MinerU, parse PDFs/images, scan corpora, call LLMs, use network access, train models, predict candidates, or mutate registry/promotion/publication/release/global append artifacts.

Use `ai4s_agent.agents.oled_local_demo_enqueue` to submit new jobs, `ai4s_agent.agents.oled_local_demo_worker_loop` to consume queued jobs, and `ai4s_agent.agents.oled_local_demo_retry` to retry failed jobs after fixing the local cause.
