# OLED Local Demo Generic Status

`ai4s_agent.agents.oled_local_demo_generic_status` inspects generic `run_plan_execute` queue jobs for the OLED local demo.

This command is read-only. It reports only generic queue jobs whose embedded RunPlan contains exactly `execute_oled_local_demo` in both `requested_tasks` and planned tasks. It does not inspect the OLED-specific `execute_oled_local_demo_runplan` worker task envelope.

## Examples

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_generic_status \
  --queue-root /tmp/molly-worker-queue
```

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_generic_status \
  --queue-root /tmp/molly-worker-queue \
  --project-root /tmp/molly-projects \
  --status failed
```

Optional filters:

- `--job-id`
- `--project-id`
- `--run-id`
- `--status`

## Metadata

The status output includes queue job metadata, matching lease metadata, status counts, retry/cancel metadata, task options, and result `executed_tasks` when present.

Use `ai4s_agent.agents.oled_local_demo_generic_retry` to queue a retry child for failed allowlisted generic OLED local demo jobs after inspecting them here.

When `--project-root` is supplied, the command reads only metadata JSON files:

- `projects/<project_id>/runs/<run_id>/stage.json`
- `projects/<project_id>/runs/<run_id>/artifact_registry.json`

Artifact registry values are not opened. Only artifact ids are reported.

## Safety Boundary

This command does not enqueue jobs, poll the queue, retry jobs, cancel jobs, execute `RunPlanExecutor`, instantiate `RunPlanExecutorTaskRunner`, instantiate ProjectStorage, use `LocalWorkerLoop`, use `WorkerQueuePoller`, execute adapters, read input bundle files, open artifact paths, use the OLED-specific `execute_oled_local_demo_runplan` job type, call MinerU, parse PDFs/images, scan corpora, call LLMs, use network access, train models, predict candidates, approve gates, resume gates, spawn subprocesses, start a daemon, run an infinite loop, or mutate registry/promotion/publication/release/global append artifacts.
