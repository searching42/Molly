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

For each requested training target, the agent should first prepare a target-aware modeling brief from project memory, previous run diagnostics, built-in domain rules, dataset statistics, and optional user-approved web/literature search. That brief should drive preprocessing, split strategy, target transforms, backend choice, and hyperparameters.

After training, the agent should diagnose model quality against baselines and target-specific expectations before using the model for prediction. Weak results should produce a reviewable rerun proposal, not a silent rerun or an unqualified model promotion.

Historical training results are modeling priors for future agent decisions, not default MVP prediction weights. A model can be reused for prediction only after it is explicitly promoted as an asset for a compatible request, with applicability limits and user approval; otherwise fresh target-specific training remains the default.

`PromotedModelAsset` is the reuse contract for that exception: it records the approved model id, backend, runtime directory, required inputs, metrics, applicability notes, source run, and rollback asset. `PredictionPreparationAgent` will build a draft prediction payload only for a confirmed promoted asset, or for historical reuse that the user explicitly approves for a controlled run.

Registered model packages can be promoted into project assets via `ProjectStorage.promote_registered_model_asset()` or the `/models/promote` API/UI review path. Project-level prediction preparation can then load those confirmed assets from storage before deciding whether a fresh training run is still required.

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
