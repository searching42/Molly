# OLED Local Demo Worker Health CLI

`ai4s_agent.agents.oled_local_demo_health` provides a read-only health check for the OLED local demo worker queue stack.

It checks queue readability, operation entrypoint availability, task-id consistency, optional project-root visibility, and the read-only status helper. It prints compact JSON and does not execute or mutate jobs.

## Example

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_health \
  --queue-root /tmp/molly-worker-queue \
  --project-root /tmp/molly-projects
```

## Checks

The health output includes:

- `queue_store_readable`: opens `JsonWorkerQueueStore` and reads job/lease metadata.
- `entrypoints_importable`: confirms enqueue, worker-loop, retry, cancel, status, and worker-runner entrypoints are available.
- `task_id_consistency`: confirms OLED local demo worker operation task ids use `execute_oled_local_demo_runplan`.
- `status_read_only_check`: calls the status helper and confirms it remains non-executable.
- `project_root_readable`: when supplied, reports whether the project root and `projects` directory exist.

## Safety Boundary

This command is health-only. It does not poll the queue, enqueue jobs, retry jobs, cancel jobs, run a worker loop, execute `RunPlanExecutor`, execute adapters, read input bundle files, open artifact paths, modify task payloads, create ProjectStorage stage state, write output reports, spawn subprocesses, start a daemon, approve gates, call MinerU, parse PDFs/images, scan corpora, call LLMs, use network access, train models, predict candidates, or mutate registry/promotion/publication/release/global append artifacts.

Use this command before `ai4s_agent.agents.oled_local_demo_enqueue`, `ai4s_agent.agents.oled_local_demo_worker_loop`, `ai4s_agent.agents.oled_local_demo_status`, `ai4s_agent.agents.oled_local_demo_retry`, or `ai4s_agent.agents.oled_local_demo_cancel` when diagnosing local setup.

Use `ai4s_agent.agents.oled_local_demo_generic_queue` to validate the same OLED local demo task through the generic `run_plan_execute` queue infrastructure.
