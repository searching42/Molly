# OLED Agent MVP Demo

## Purpose

`OLEDAgentMVPDemoRunner` provides a compact, deterministic demonstration of Molly's review-only OLED Agent loop. It exists to prove that the current Agent components can be used together end-to-end without adding another admission, receipt, readiness, preflight, writer, registry, promotion, or publication layer.

The demo is intentionally synthetic. It accepts a goal and scenario, builds minimal dictionaries, runs the existing review-only components, and emits one consolidated report.

## Component Chain

The runner uses the existing chain:

1. `OLEDDiscoveryReviewLoopAgent.build_review()`
2. `OLEDDiscoveryActionHandoffAgent.build_handoff()`
3. `OLEDDiscoveryExecutionPreviewAgent.build_preview()`
4. `OLEDDiscoveryDryRunPacketAgent.build_packet()`
5. `OLEDDiscoveryDryRunBridgeRequestAgent.build_request()`

It does not execute tools or adapters.

## Supported Scenarios

- `acceptable_diagnostics`: diagnostics are acceptable, provenance is present, and the loop recommends candidate generation.
- `weak_diagnostics`: diagnostics are weak and the critic recommends rerunning the baseline.
- `missing_provenance`: diagnostics are otherwise acceptable, but provenance is empty and the critic requests more evidence.
- `candidate_review_needed`: candidate artifacts exist and the critic recommends candidate review.

## Output

The runner returns a compact dictionary with:

- run id, project id, goal, and scenario
- current stage
- critic decision
- recommended next action
- selected tool
- resolved atomic task
- approval mode
- dry-run mode
- bridge mode
- bridge eligibility
- risk flags and blocked reasons
- `executable=false`

`write_demo_report()` writes:

- `oled_agent_mvp_demo.json`
- `oled_agent_mvp_demo.md`

## One-Scenario CLI

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_mvp_demo \
  --run-id demo \
  --goal "Find OLED emitters with high PLQY and red-shifted emission" \
  --scenario acceptable_diagnostics \
  --output-dir /tmp/oled-agent-demo
```

The CLI prints compact JSON only. If `--output-dir` is supplied, it also writes the JSON and Markdown demo report to that directory.

## All-Scenarios CLI

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_mvp_demo \
  --run-id demo \
  --goal "Find OLED emitters with high PLQY and red-shifted emission" \
  --all-scenarios \
  --output-dir /tmp/oled-agent-demo
```

With `--all-scenarios`, the runner executes every built-in synthetic scenario and writes:

- `oled_agent_mvp_demo_matrix.json`
- `oled_agent_mvp_demo_matrix.md`

The CLI still prints compact JSON only, including the run id, scenario count, critic-decision counts, and `executable=false`.

## Scenario Matrix

The scenario matrix compares each built-in path across:

- current stage
- critic decision
- recommended next action
- selected tool
- resolved atomic task
- approval mode
- dry-run mode
- bridge mode
- risk flags and blockers

The summary includes deterministic critic-decision counts, bridge-mode counts, and the scenarios that still have review blockers. This improves demoability and inspection of the integrated Agent loop; it is not a governance expansion and does not add any admission, receipt, readiness, preflight, writer, registry, promotion, or publication layer.

## Local JSON Input Bundle

The demo runner can also load one user-specified local JSON file containing scenario summaries:

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_mvp_demo \
  --run-id local-demo \
  --input-bundle /tmp/oled_demo_bundle.json \
  --output-dir /tmp/oled-agent-demo
```

The bundle may provide `goal` and `project_id`; explicit `--goal` and `--project-id` CLI values override the bundle values. When `--output-dir` is supplied, the runner writes:

- `oled_agent_mvp_demo_bundle.json`
- `oled_agent_mvp_demo_bundle.md`

Example bundle:

```json
{
  "schema_version": 1,
  "goal": "Find OLED emitters with high PLQY and red-shifted emission",
  "project_id": "demo-project",
  "scenarios": [
    {
      "name": "local_acceptable",
      "description": "Local acceptable diagnostics example.",
      "payload": {
        "dataset_artifacts": {"dataset_view_rows": "local_dataset_rows"},
        "training_package_artifacts": {"training_rows": "local_training_rows"},
        "baseline_artifacts": {"metrics": "local_metrics"},
        "diagnostics_report": {"status": "acceptable"},
        "provenance_summary": {"source_count": 2, "evidence_count": 8}
      }
    },
    {
      "name": "local_weak",
      "payload": {
        "dataset_artifacts": {"dataset_view_rows": "local_dataset_rows"},
        "training_package_artifacts": {"training_rows": "local_training_rows"},
        "baseline_artifacts": {"metrics": "local_metrics"},
        "diagnostics_report": {"status": "weak", "summary": "rerun recommended"},
        "provenance_summary": {"source_count": 2, "evidence_count": 8}
      }
    }
  ]
}
```

The local bundle path is the only file read by this mode. Artifact values inside the JSON are treated as labels/placeholders only; the runner does not follow, open, hash, scan, or validate referenced artifact paths. This improves demoability with local summaries without adding governance layers or execution behavior.

## What This Proves

This demo proves the Agent loop is now composable enough to show a reviewer:

- where the run stands
- what the critic decided
- what action is recommended
- which tool and atomic task would be considered
- what approval/dry-run/bridge mode applies
- what risks and blockers remain

It is a usability pivot away from over-optimizing governance layers and toward demonstrating the integrated Agent loop.

## Safety Boundary

The demo reads at most one user-specified local JSON bundle when `--input-bundle` is used. It does not execute adapters, call or instantiate `RunPlanExecutor`, approve or resume gates, mutate `stage.json`, mutate `gate_decisions.json`, read or hash referenced artifact paths, scan directories, read corpus files, train models, run prediction, validate benchmarks, call LLMs, call MinerU, read PDFs/images, use external network access, or mutate registry/promotion/publication/release/global append artifacts.

The report is review-only and always `executable=false`.
