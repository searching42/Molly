# OLED Discovery Action Handoff

## Purpose

`OLEDDiscoveryReviewLoopAgent` produces a review-only loop artifact with a recommended next action. `OLEDDiscoveryActionHandoffAgent` converts that recommendation into a structured, review-only action intent that a later gated planner or dry-run bridge can inspect.

This PR does not add execution. It adds the missing handoff object between Agent recommendation and future controlled execution planning.

## What It Maps

The handoff maps `recommended_next_action` to:

- selected tool id
- selected task id
- target stage
- required input artifacts
- missing inputs
- output artifacts
- required gates
- required permissions
- blocked reasons
- risk flags
- placeholder payload template
- rationale

If the action is a critic decision such as `rerun_baseline`, `revise_data`, or `run_candidate_review`, the handoff maps it to the closest review-only tool intent such as `baseline_runner`, `leakage_split`, `training_package`, or `critic_review`.

Resolver actions such as `resolve_missing_inputs:training_package_artifacts` produce no selected execution tool. They preserve the missing input or gate blocker for human review.

## Ready Semantics

`ready=true` means the handoff is ready for human review or future gated planner handoff. It never means execution is allowed.

If required gates are present, the handoff may still be ready as “ready for gated review, not execution.” The object remains `executable=false`.

If missing inputs or blocked reasons are present, the handoff is not ready.

## Payload Templates

Payload templates are placeholders only. Examples:

```json
{
  "run_id": "demo",
  "diagnostics_report": "<diagnostics_report>",
  "training_package_artifacts": "<training_package_artifacts>",
  "review_only": true
}
```

Templates never read files or artifact contents.

## Markdown And JSON

`write_handoff()` writes deterministic artifacts:

- `oled_discovery_action_handoff.json`
- `oled_discovery_action_handoff.md`

The Markdown summarizes inputs, missing inputs, gates, permissions, payload template, rationale, and safety boundary.

## CLI Example

```bash
PYTHONPATH=src python -m ai4s_agent.agents.action_handoff \
  --run-id demo \
  --goal "Find OLED emitters with high PLQY" \
  --recommended-next-action candidate_generation_or_prediction
```

The CLI prints compact JSON only and does not execute tools.

## Safety Boundary

The handoff does not execute adapters, call `RunPlanExecutor`, approve or resume gates, mutate stage state, run model training, run prediction, validate benchmarks, call LLMs, call MinerU, read PDFs/images, perform external network access, or mutate registry/promotion/publication/release/global append artifacts.

It prepares future controlled dry-run or executor bridge work while keeping this PR strictly review-only.
