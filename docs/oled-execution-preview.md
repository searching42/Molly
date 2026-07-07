# OLED Discovery Execution Preview

## Purpose

`OLEDDiscoveryActionHandoffAgent` turns a loop recommendation into a selected tool/task intent. `OLEDDiscoveryExecutionPreviewAgent` takes that handoff one step further by resolving known tools to read-only `AtomicTaskRegistry` metadata and summarizing the execution policy that a future controlled planner would need to respect.

The preview is still planning-only. It does not execute adapters, call `RunPlanExecutor`, approve gates, mutate project state, or install runtime execution policy patches.

## Handoff Versus Preview

The action handoff answers: what action did the Agent recommend, what tool does it map to, and what placeholder payload would a later planner need?

The execution preview answers: does the selected tool map to an atomic task, what adapter metadata applies, what risk level and gates are present, and whether the handoff is auto-eligible, gated-review-only, manual-review-only, or blocked for future controlled planning.

## Atomic Task Mapping

The preview uses deterministic mappings for OLED discovery tools:

- `baseline_runner` -> `run_baseline`
- `candidate_generation_or_prediction` -> `generate_candidates`
- `candidate_ranking` -> `filter_rank`
- `acquire_literature_sources` -> `acquire_literature_sources`
- `parse_document_mineru` -> `parse_document`
- `parse_document_pdfplumber` -> `parse_document_pdfplumber`
- `retrieve_evidence` -> `retrieve_evidence`

Some review-only planning tools, such as `research_source_proposal` and `critic_review`, intentionally have no atomic execution task. Those remain manual-review planning intents.

## Risk And Gates

The preview reads task risk, default adapter, required artifacts, output artifacts, and gates from `AtomicTaskRegistry`. It may also inspect `ExecutionPolicyRegistry` adapter metadata, but it never installs the policy registry or patches API/executor modules.

Approval modes are deterministic:

- `auto_eligible`: low risk, no gates, mapped atomic task, no missing inputs, and auto eligibility allowed.
- `gated_review_required`: mapped task requires gates and gated planning is allowed.
- `manual_review_required`: mapped medium/high risk task without gates, or reviewable action without an atomic task.
- `blocked`: missing inputs, handoff not ready, risk exceeds budget, gated tools are disallowed, or no selected tool exists.

`auto_eligible` is not execution. It only means a future controlled dry-run bridge may consider this kind of step for automatic approval after verifying artifacts and policy snapshots.

## Execution Preconditions

Every preview records preconditions a future executor bridge would need to verify:

- verify artifact paths exist
- verify payload still matches handoff
- verify gate approvals bind to snapshot
- verify no registry/promotion/publication mutation
- verify executor dry-run mode if used later

Gated tasks add a human gate approval precondition. Unmapped tools add a manual planner mapping precondition.

## Markdown And JSON

`write_preview()` writes deterministic review artifacts:

- `oled_discovery_execution_preview.json`
- `oled_discovery_execution_preview.md`

The Markdown summarizes selected tool/task, resolved atomic task, adapter, risk, approval mode, inputs, missing inputs, gates, preconditions, payload template, policy notes, and safety boundary.

## CLI Example

```bash
PYTHONPATH=src python -m ai4s_agent.agents.execution_preview \
  --run-id demo \
  --goal "Find OLED emitters with high PLQY" \
  --selected-tool candidate_generation_or_prediction
```

The CLI prints compact JSON only and does not execute tools.

## Safety Boundary

This preview does not execute adapters, call `RunPlanExecutor`, call `/api/run-plan/resume`, approve gates, mutate `stage.json`, mutate `gate_decisions.json`, run model training, run prediction, validate benchmarks, call LLMs, call MinerU, read PDFs/images, use external network access, or mutate registry/promotion/publication/release/global append artifacts.

It prepares future controlled dry-run execution work while keeping this PR strictly review-only.
