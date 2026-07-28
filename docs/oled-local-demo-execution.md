# OLED Local Demo Execution

`OLEDLocalDemoExecutionRunner` is the first controlled local execution layer for the OLED Agent MVP demo. It executes only local demo file IO and report generation.

## What It Does

The runner performs one local command:

1. Read exactly one user-specified summary bundle JSON file.
2. Run the existing `OLEDAgentMVPDemoRunner.run_local_bundle()` chain.
3. Create the requested output directory if needed.
4. Write:
   - `oled_agent_mvp_demo_bundle.json`
   - `oled_agent_mvp_demo_bundle.md`
   - `oled_local_demo_execution_manifest.json`
5. Print compact JSON from the CLI.

The execution manifest is a local run log for this command, not a governance receipt.

## Safety Boundary

This runner does not execute scientific adapters, call or instantiate `RunPlanExecutor`, approve gates, mutate `stage.json`, mutate `gate_decisions.json`, mutate artifact registries, train models, predict, validate benchmarks, call LLMs, call MinerU, read PDFs/images, scan directories, read corpus files, use external network access, or mutate registry/promotion/publication/release/global append artifacts.

Artifact labels inside the summary bundle are placeholders. They are not opened, hashed, followed, or validated as real paths.

## CLI

Generate a local summary bundle template:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_mvp_demo \
  --write-input-bundle-template /tmp/oled_demo_bundle.json
```

Run controlled local demo execution:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_execution \
  --run-id local-demo \
  --input-bundle /tmp/oled_demo_bundle.json \
  --output-dir /tmp/oled-agent-demo
```

Use `--overwrite` to replace existing demo output files in the output directory.

## Output

Compact CLI output includes:

```json
{
  "source": "local_demo_execution",
  "scenario_count": 3,
  "executable": true,
  "adapters_executed": false
}
```

Here `executable=true` means the local demo runner performed file IO and report generation. It does not mean adapters were executed.

## RunPlanExecutor Task

The same local demo path can run through `RunPlanExecutor` via the low-risk `execute_oled_local_demo` task and `execute_oled_local_demo_adapter`. See `docs/oled-local-demo-runplan-executor.md` for the task-options example and artifact registration behavior.
