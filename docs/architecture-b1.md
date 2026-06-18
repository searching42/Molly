# AI4S Agent B1 Architecture

## Runtime Boundary

B1 runs inside one Flask process. The process exposes a thin API layer and delegates orchestration state to `Orchestrator`.

Existing deterministic workflow scripts remain in `$AI4S_WORKSPACE/claude/scripts` (default: the parent workspace of this repository). The agent layer does not replace those scripts; it plans and gates when they should be invoked.

## Modules

- `ai4s_agent.api`: HTTP route registration for health, plan creation, and gate approval
- `ai4s_agent.app`: Flask app factory
- `ai4s_agent.planner`: converts prompt and run id into a `PlanModel`
- `ai4s_agent.gatekeeper`: in-process gate approval state
- `ai4s_agent.orchestrator`: coordinates planner, gatekeeper, and artifact writes
- `ai4s_agent.storage`: JSON artifact store under `runs/<run_id>/`
- `ai4s_agent.adapters.claude_scripts`: command construction for existing Claude scripts
- `ai4s_agent.error_taxonomy`: maps failures into `REMOTE`, `WF`, `VAL`, `DATA`, `PRED`, `GEN`, or `UNKNOWN`
- `ai4s_agent.agents.data_miner`: local data mining contract
- `ai4s_agent.agents.trainer`: training plan contract
- `ai4s_agent.agents.modeling`: target-aware modeling brief, backend recommendation, metric diagnosis, and rerun proposal contract
- `ai4s_agent.agents.screener`: screening plan contract
- `ai4s_agent.agents.generator_reinvent4`: REINVENT4 generation plan contract

## Five Gates

- `gate_1_task_parse`: confirm parsed objective, weights, constraints, model, and TopN
- `gate_2_data_mining`: confirm local data selection, mapping, and cleaning plan
- `gate_3_train_config`: confirm train properties, target columns, target modeling brief, preprocessing choices, split strategy, target transforms, backend, hyperparameters, and runtime setup
- `gate_4_post_train_diagnostics`: confirm model diagnostics, weak-metric handling, rerun proposal if any, and whether the model is acceptable for prediction
- `gate_5_final_threshold`: confirm final thresholds and output publication

## Artifact Contract

Every run stores agent-owned artifacts under:

```text
workspace/agent/runs/<run_id>/
```

Current B1 artifacts:

- `plan.json`: serialized `PlanModel`
- `gate_decisions.json`: ordered gate approval decisions
- `stage.json`: current `RunPlanExecutor` stage, including `execution_snapshot`
  when execution is paused at a gate

Planned downstream artifacts:

- `data_mining_report.json`
- `target_modeling_brief.json`
- `training_report.json`
- `model_diagnostics_report.json`
- `rerun_proposal.json`
- `screening_report.json`
- `generation_result.json`

## Execution Approval Boundary

The agentic `RunPlanExecutor` treats a gate approval as approval for a frozen
execution snapshot, not as approval for a mutable resume request. When a run
pauses at a gated atomic task, `stage.json` records the current task, default or
approved adapter, run plan, task options, normalized payload, required gates,
and snapshot hash. `/api/run-plan/resume` validates that the current task's
execution content still matches that snapshot before appending
`gate_decisions.json`; the gate decision records the approved snapshot id and
hash.

Direct adapter execution is closed to the atomic task registry. `/api/adapters/execute`
may call only adapters that map to a registered task policy, then applies the
same permission and gate checks. Exported legacy helpers remain importable for
tests or fallback code but are not API-executable unless registered.

Project-scoped run state is authoritative when a request includes `project_id`.
Status and direct adapter gate checks read `projects/<project_id>/runs/<run_id>`
for `stage.json`, `gate_decisions.json`, and `artifact_registry.json`, while
legacy `runs/<run_id>` remains a compatibility fallback for older endpoints.
JSON state writes use same-directory temporary files plus atomic replacement.
Project uploads reject duplicate filenames and enforce a configurable upload
size limit so existing artifacts are not silently overwritten.

## Target-Aware Modeling Loop

The planning layer should treat each requested target property as a modeling
problem with its own domain risks, not as a generic regression column.

Before training, `ModelingAgent` produces a `TargetModelingBrief` for each
resolved target. The brief may use project memory, previous run diagnostics,
built-in domain rules, current dataset statistics, and optional user-approved
web or literature search. It records target-specific cautions such as solvent or
device dependence, bounded labels, log-scale targets, replicate variance,
outliers, class imbalance, target leakage, recommended split strategy, target
transforms, backend candidates, hyperparameter ranges, and expected metric
scale.

After training, `ModelingAgent` and `VerifierAgent` produce a
`ModelDiagnosticsReport`. The report compares metrics against mean baselines,
fingerprint baselines, and target-specific expectations; checks fold stability,
prediction range collapse, target-bucket bias, high-error samples, featurization
or conformer failures, and learning curves; and assigns a use-case decision:
`ACCEPT`, `ACCEPT_LOW_CONFIDENCE`, `RERUN_RECOMMENDED`, or `BLOCKED`.

When a model is weak or degraded, the agent emits a reviewable `RerunProposal`
instead of silently continuing. Valid proposals include revised cleaning,
low-noise filtering, target transforms, solvent/context conditioning, split
correction, seed ensembles, hyperparameter sweeps, backend switches, collecting
more data, or using the model only as a low-confidence signal. Reruns that add
cost, external search, backend changes, data relaxation, or model promotion
remain gated by explicit user approval.

## REINVENT4 Minimal Loop

`GeneratorAgent` uses `reinvent4` as the backend label and should emit reward targets that match the parsed screening objective plus the trainable/available property set. The legacy `lambda_em` / `plqy` / `mw` trio is an early example, not the intended long-term restriction.

The generated candidates are marked with `rescore_with_screener=true`, so the discriminative screening path remains the single scoring authority.
