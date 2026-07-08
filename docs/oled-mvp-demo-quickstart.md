# OLED MVP Demo Quickstart

This quickstart shows the review-only OLED Agent MVP demo paths that can be run in a few minutes from a local checkout.

The quickstart paths are covered by lightweight smoke tests in `tests/test_oled_mvp_demo_quickstart_smoke.py`.

## Safety Boundary

The demo is review-only. It does not execute adapters, call or instantiate `RunPlanExecutor`, approve gates, mutate run state, train models, predict, validate benchmarks, call LLMs, call MinerU, read PDFs/images, read corpus files, scan directories, hash artifact paths, use external network access, or mutate registry/promotion/publication/release/global append artifacts.

Local bundle artifact values are labels/placeholders only. The runner does not open or read referenced artifact labels.

## Run One Built-In Scenario

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_mvp_demo \
  --run-id demo-one \
  --goal "Find OLED emitters with high PLQY and red-shifted emission" \
  --scenario acceptable_diagnostics
```

Expected compact JSON includes these fields. Exact field order may differ.

```json
{
  "critic_decision": "continue",
  "recommended_next_action": "candidate_generation_or_prediction",
  "selected_tool_id": "candidate_generation_or_prediction",
  "resolved_atomic_task_id": "generate_candidates",
  "executable": false
}
```

## Run All Built-In Scenarios

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_mvp_demo \
  --run-id demo-matrix \
  --goal "Find OLED emitters with high PLQY and red-shifted emission" \
  --all-scenarios \
  --output-dir /tmp/oled-agent-demo
```

Expected compact JSON includes:

```json
{
  "scenario_count": 4,
  "critic_decision_counts": {
    "continue": 1,
    "request_more_evidence": 1,
    "rerun_baseline": 1,
    "run_candidate_review": 1
  },
  "executable": false
}
```

The command writes:

- `oled_agent_mvp_demo_matrix.json`
- `oled_agent_mvp_demo_matrix.md`

## Print Or Write Local Bundle Template

Print the template:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_mvp_demo \
  --print-input-bundle-template
```

Write the template:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_mvp_demo \
  --write-input-bundle-template /tmp/oled_demo_bundle.json
```

Edit only summary fields. Do not paste raw MinerU output. Do not put real corpus paths expecting the runner to read them. The local bundle is a summary-only demo input file.

An example template is also available at `docs/examples/oled_demo_bundle.template.json`.

## Run Local Input Bundle

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_mvp_demo \
  --run-id local-demo \
  --input-bundle /tmp/oled_demo_bundle.json \
  --output-dir /tmp/oled-agent-demo
```

Expected compact JSON includes:

```json
{
  "source": "local_input_bundle",
  "scenario_count": 3,
  "executable": false
}
```

The command writes:

- `oled_agent_mvp_demo_bundle.json`
- `oled_agent_mvp_demo_bundle.md`

## Controlled Local Demo Execution

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_mvp_demo \
  --write-input-bundle-template /tmp/oled_demo_bundle.json

PYTHONPATH=src python -m ai4s_agent.agents.oled_local_demo_execution \
  --run-id local-demo \
  --input-bundle /tmp/oled_demo_bundle.json \
  --output-dir /tmp/oled-agent-demo
```

This controlled local runner reads one summary bundle and writes the bundle report plus `oled_local_demo_execution_manifest.json`. It still does not execute adapters or read referenced artifact labels.

## What This Proves

This proves the current review-only Agent loop can compose run-card state, tool recommendations, critic branching, action handoff, execution preview, dry-run packet, and bridge request summaries.

This does not prove real literature parsing, real model training, real prediction, benchmark validation, scientific performance validity, or publication readiness.

## Troubleshooting

- `--goal` is required unless `--input-bundle` or a template command is used.
- Local bundles must have `schema_version=1`.
- `scenarios` must be a nonempty list.
- Each scenario needs a nonempty `name` and a dict `payload`.
- Artifact labels inside bundle payloads are placeholders; missing files are not looked up by the demo runner.
