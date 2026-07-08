# OLED Local Demo Generic RunPlan Queue

`ai4s_agent.agents.oled_local_demo_generic_queue` runs the OLED local demo through the generic `run_plan_execute` queue envelope.

This path enqueues a generic RunPlan queue job whose run plan contains exactly one task: `execute_oled_local_demo`. The job is consumed by `WorkerQueuePoller`, `LocalWorkerLoop`, and `RunPlanExecutorTaskRunner`, then executed by `RunPlanExecutor` using the existing `execute_oled_local_demo_adapter`.

## Example

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_generic_queue \
  --queue-root /tmp/molly-worker-queue \
  --project-root /tmp/molly-projects \
  --project-id demo-project \
  --run-id oled-generic-queue-demo \
  --input-bundle /tmp/oled_demo_bundle.json \
  --output-dir /tmp/oled-agent-demo \
  --worker-id generic-worker-1 \
  --max-iterations 3 \
  --overwrite
```

## What It Proves

This command proves the OLED local demo can run through the repository's generic RunPlan queue infrastructure instead of the OLED-specific `execute_oled_local_demo_runplan` worker task envelope.

The execution chain is:

```text
run_plan_execute queue job
-> WorkerQueuePoller
-> LocalWorkerLoop
-> RunPlanExecutorTaskRunner
-> RunPlanExecutor
-> execute_oled_local_demo
-> execute_oled_local_demo_adapter
-> OLEDLocalDemoExecutionRunner
```

It writes local demo reports, a manifest, ProjectStorage stage state, artifact registry entries, and worker queue job/lease state.

## Safety Boundary

This command executes only the already-whitelisted local demo adapter. It reads exactly one user-specified local summary bundle during execution and does not read, open, hash, or validate referenced artifact labels inside the bundle.

It does not enqueue the OLED-specific `execute_oled_local_demo_runplan` job type. It does not execute scientific adapters, call MinerU, parse PDFs/images, scan corpora, call LLMs, use network access, train models, predict candidates, approve gates, resume gates, spawn subprocesses, start a daemon, run an infinite loop, or mutate registry/promotion/publication/release/global append artifacts.
