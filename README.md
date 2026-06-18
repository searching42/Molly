# AI4S Agent

Same-process B1 orchestration layer for the existing AI4S screening workflow.

## Quickstart

```bash
cd /Users/benton/openclaw-docker/workspace/agent
PYTHONPATH=src .venv/bin/python -m flask --app 'ai4s_agent.app:create_app' run --port 8792
```

Open the UI:

```text
http://127.0.0.1:8792/
```

The planning layer is intended to infer target properties from the user's natural-language goal plus the cleaned dataset property catalog. It should not assume a fixed `lambda_em/plqy/mw` target set.

For each requested training target, the agent should first prepare a target-aware modeling brief from project memory, previous run diagnostics, built-in domain rules, dataset statistics, and optional user-approved web/literature search. Structured `TargetEvidenceItem` records keep cited summaries, implications, recommended actions, and confidence visible inside that brief before they influence preprocessing, split strategy, target transforms, backend choice, or hyperparameters.

`ResearchAgent.prepare_target_evidence_items()` converts cited summaries into `TargetEvidenceItem` records for that handoff. External summaries must carry a source reference such as a DOI, URL, or source id and require explicit `user_approved_external_search=True`; the method does not perform network access.

Ordinary dialogue is the primary interface for collecting user intent, cited source summaries, approvals, and follow-up answers. `ConversationAgent.prepare_modeling_plan_payload()` turns those conversation turns into a non-executable modeling-plan payload, keeping unapproved external DOI/URL evidence in `pending_cited_target_evidence` until the user explicitly approves it. `/api/agent/conversation/modeling-payload` exposes that bridge for clients without requiring a dedicated evidence input form.

`ConversationAgent.decide_next_turn()` and `/api/agent/conversation/next-turn` wrap that payload in a `ConversationTurnDecision` so clients can distinguish `needs_clarification`, `needs_evidence_approval`, and `ready_for_modeling_plan` before calling `/api/agent/modeling-plan`. The decision object is review-only and never executes training, web acquisition, or model promotion by itself.

`ConversationAgent.prepare_research_source_payload()` and `/api/agent/conversation/research-sources` let chat messages generate a dry-run `ResearchSourceProposal` with DOI/URL seed sources and query scopes. This is acquisition planning only: it may write proposal artifacts for review, but network/database acquisition remains a separate explicit action.

`/api/agent/modeling-plan` accepts `property_id`, `cited_target_evidence`, `project_memory`, `previous_diagnostics`, `available_inputs`, and `user_approved_external_search`. When a target property or cited evidence is supplied, the endpoint returns and writes a `TargetModelingBrief` alongside the modeling plan proposal so preprocessing and hyperparameter decisions remain traceable to reviewable evidence.

After training, the agent should diagnose model quality against baselines and target-specific expectations before using the model for prediction. Weak results should produce a reviewable rerun proposal, not a silent rerun or an unqualified model promotion.

`/api/agent/review-card` exposes `TargetModelingBrief`, `ModelDiagnosticsReport`, `RerunProposal`, and `ModelPackageReview` as explicit review sections with source labels and approval controls. The local console renders these sections as lightweight cards while keeping the raw JSON response available for audit/debugging.

Successful training adapters write a promotable model package into the model directory, including `model_metadata.json`, `model_manifest.json`, and `domain_model_manifest.json`. These package manifests make later registration and promotion review reproducible, but they do not by themselves approve reuse.

`/api/agent/model-package-review` reviews those manifests plus optional diagnostics before any registry decision. The review can recommend `promote_candidate`, `rerun_recommended`, `memory_only`, or `blocked`; promotion recommendations still require the separate `promote_asset` confirmation path.

When `RunPlanExecutor` completes baseline training with model package manifests, it also writes `ModelDiagnosticsReport` and `ModelPackageReview` artifacts automatically, so every trained package has a simple review record before registration or promotion.

Historical training results are modeling priors for future agent decisions, not default MVP prediction weights. A model can be reused for prediction only after it is explicitly promoted as an asset for a compatible request, with applicability limits and user approval; otherwise fresh target-specific training remains the default.

`PromotedModelAsset` is the reuse contract for that exception: it records the approved model id, backend, runtime directory, required inputs, metrics, applicability notes, source run, and rollback asset. `PredictionPreparationAgent` will build a draft prediction payload only for a confirmed promoted asset, or for historical reuse that the user explicitly approves for a controlled run.

Registered model packages can be promoted into project assets via `ProjectStorage.promote_registered_model_asset()` or the `/models/promote` API/UI review path. The `/models/promote/draft` endpoint and local console draft button prefill review fields from registered model metadata and manifests before confirmation. Project-level prediction preparation can then load confirmed assets from storage before deciding whether a fresh training run is still required.

Create a plan:

```bash
curl -X POST http://127.0.0.1:8792/api/plan \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"demo","prompt":"find candidates with high emission wavelength and high photoluminescence quantum yield while keeping molecular weight manageable"}'
```

Approve the first gate:

```bash
curl -X POST http://127.0.0.1:8792/api/gates/approve \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"demo","gate":"gate_1_task_parse","actor":"user"}'
```
