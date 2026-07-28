# OLED Local Demo RunPlanExecutor Task

`execute_oled_local_demo` is the first RunPlanExecutor-backed OLED local demo task. It runs exactly one low-risk adapter, `execute_oled_local_demo_adapter`, which calls `OLEDLocalDemoExecutionRunner.execute()`.

## What It Does

The task:

- reads exactly one user-specified local summary bundle JSON file
- writes the OLED MVP demo bundle JSON report
- writes the OLED MVP demo bundle Markdown report
- writes the local demo execution manifest
- records the adapter result and artifact registry entries under `ProjectStorage`

This differs from the standalone local runner by going through `RunPlanExecutor`, stage state, adapter result writing, and artifact registration.

## Safety Boundary

The adapter does not run scientific adapters, call MinerU, parse PDFs, scan corpora, call LLMs, use network access, train models, predict candidates, approve gates, resume gates, or mutate registry/promotion/publication/release/global append artifacts.

Artifact labels inside the input bundle are summary placeholders. They are not opened, hashed, followed, or validated as real paths.

## Example

```python
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.storage import ProjectStorage

run_plan = expand_run_plan(
    run_id="oled-local-demo",
    requested_tasks=["execute_oled_local_demo"],
)
result = RunPlanExecutor(storage=ProjectStorage("/tmp/molly-demo")).execute(
    project_id="demo-project",
    run_plan=run_plan,
    task_options={
        "execute_oled_local_demo": {
            "input_bundle": "/tmp/oled_demo_bundle.json",
            "output_dir": "/tmp/oled-agent-demo",
            "overwrite": True,
        }
    },
)
```

The task has no gates because it is low-risk and local-only. It still requires explicit `task_options` for the input bundle and may use either an explicit output directory or the default under the executor run directory.

For a direct command-line entrypoint around this one-task run plan, see `docs/oled-local-demo-runplan-cli.md`.

## Outputs

The adapter returns:

- `oled_demo_bundle_report`
- `oled_demo_bundle_markdown`
- `oled_local_demo_execution_manifest`

The adapter itself is executed. The `adapters_executed=false` field in the demo manifest and adapter summary means no scientific adapters were executed by the local demo runner.
