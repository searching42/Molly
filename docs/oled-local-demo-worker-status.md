# OLED Local Demo Worker Status CLI

`ai4s_agent.agents.oled_local_demo_status` provides a read-only view of OLED local demo worker jobs.

It opens the local worker queue, lists jobs and leases, filters to `execute_oled_local_demo_runplan`, and prints compact JSON. It can optionally read ProjectStorage `stage.json` and `artifact_registry.json` metadata when `--project-root` is supplied.

## Inspect Jobs

List OLED local demo jobs:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_status \
  --queue-root /tmp/molly-worker-queue
```

Inspect failed jobs with run metadata:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_status \
  --queue-root /tmp/molly-worker-queue \
  --project-root /tmp/molly-projects \
  --status failed
```

Optional filters:

- `--job-id`
- `--project-id`
- `--run-id`
- `--status`

## Output

The command returns:

- queue root and filter values
- job count and status counts
- compact job metadata
- matching lease metadata, when present
- retry and cancellation metadata
- optional stage status and artifact ids, when `--project-root` is supplied

Artifact ids are read from the registry metadata only. Artifact paths themselves are not opened.

## Safety Boundary

This command is status-only. It does not poll the queue, enqueue jobs, retry jobs, cancel jobs, run a loop, execute `RunPlanExecutor`, execute adapters, read the input bundle file, open artifact paths, modify task payloads, create ProjectStorage stage state, write output reports, spawn subprocesses, start a daemon, approve gates, call MinerU, parse PDFs/images, scan corpora, call LLMs, use network access, train models, predict candidates, or mutate registry/promotion/publication/release/global append artifacts.

Use `ai4s_agent.agents.oled_local_demo_enqueue` to submit jobs, `ai4s_agent.agents.oled_local_demo_worker_loop` to consume jobs, `ai4s_agent.agents.oled_local_demo_retry` to retry failed jobs, and `ai4s_agent.agents.oled_local_demo_cancel` to cancel queued or running jobs.

Use `ai4s_agent.agents.oled_local_demo_health` to check local worker stack readiness before submitting or consuming jobs.
