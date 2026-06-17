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
