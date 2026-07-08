# OLED Local Demo Generic Worker Loop

`ai4s_agent.agents.oled_local_demo_generic_worker_loop` consumes existing generic `run_plan_execute` queue jobs for the OLED local demo.

This CLI does not enqueue jobs. It opens an existing worker queue, runs a bounded `LocalWorkerLoop`, and uses `RunPlanExecutorTaskRunner` only for generic RunPlan queue jobs whose embedded run plan contains exactly one task: `execute_oled_local_demo`.

## Example

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_generic_worker_loop \
  --queue-root /tmp/molly-worker-queue \
  --project-root /tmp/molly-projects \
  --worker-id generic-worker-1 \
  --max-iterations 3
```

Optional selectors can narrow the loop to one queued job, project, or run:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_generic_worker_loop \
  --queue-root /tmp/molly-worker-queue \
  --project-root /tmp/molly-projects \
  --worker-id generic-worker-1 \
  --max-iterations 3 \
  --target-run-id oled-generic-queue-demo
```

## Allowlist

The worker-loop runner validates each acquired job before delegating to `RunPlanExecutorTaskRunner`:

- queue task id must be `run_plan_execute`
- queue task kind must be `run_plan_execute`
- embedded run plan tasks must be exactly `["execute_oled_local_demo"]`
- embedded requested tasks must be exactly `["execute_oled_local_demo"]`

Any other generic run plan is failed with `generic_run_plan_not_allowlisted_for_oled_local_demo` before execution.

## Safety Boundary

This command consumes existing generic `run_plan_execute` jobs only. It does not enqueue jobs, create an OLED-specific worker task id, use `execute_oled_local_demo_runplan`, start a daemon, run forever, spawn subprocesses, approve gates, resume gates, call MinerU, parse PDFs/images, scan corpora, call LLMs, use network access, train models, predict candidates, or mutate registry/promotion/publication/release/global append artifacts.

For allowlisted jobs, it executes only the already-whitelisted OLED local demo adapter. The adapter reads exactly one user-specified local summary bundle during execution and does not read, open, hash, or validate referenced artifact labels inside the bundle.
