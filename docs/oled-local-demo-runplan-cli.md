# OLED Local Demo RunPlanExecutor CLI

`ai4s_agent.agents.oled_local_demo_runplan` is the direct command-line entrypoint for executing the OLED local demo through `RunPlanExecutor`.

It builds a one-task run plan for `execute_oled_local_demo`, runs the low-risk `execute_oled_local_demo_adapter`, and writes both the local demo outputs and the normal ProjectStorage executor state.

## Safety Boundary

This CLI executes exactly one local demo adapter. It reads one user-specified summary bundle, writes the requested local output directory, and writes RunPlanExecutor stage state, artifact registry entries, and the adapter result under `ProjectStorage`.

It does not execute scientific adapters, call MinerU, parse PDFs/images, scan corpora, call LLMs, use network access, train models, predict candidates, approve gates, resume gates, or mutate registry/promotion/publication/release/global append artifacts.

Artifact labels inside the input bundle are summary placeholders. They are not opened, hashed, followed, or validated as real paths.

## Example

Create a local summary bundle template:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_mvp_demo \
  --write-input-bundle-template /tmp/oled_demo_bundle.json
```

Run it through `RunPlanExecutor`:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_runplan \
  --project-root /tmp/molly-projects \
  --project-id demo-project \
  --run-id oled-local-demo \
  --input-bundle /tmp/oled_demo_bundle.json \
  --output-dir /tmp/oled-agent-demo \
  --overwrite
```

The CLI prints compact JSON only:

```json
{
  "ok": true,
  "status": "succeeded",
  "executed_tasks": ["execute_oled_local_demo"],
  "adapter": "execute_oled_local_demo_adapter",
  "executable": true,
  "adapters_executed": false
}
```

`executable=true` means this CLI performed local RunPlanExecutor file IO. `adapters_executed=false` means no scientific adapters were executed by the local demo runner.

## Outputs

The requested output directory contains:

- `oled_agent_mvp_demo_bundle.json`
- `oled_agent_mvp_demo_bundle.md`
- `oled_local_demo_execution_manifest.json`

The ProjectStorage run directory contains:

- `stage.json`
- `artifact_registry.json`
- `execute_oled_local_demo/adapter_result.json`
