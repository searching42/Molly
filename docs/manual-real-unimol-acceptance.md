# Manual Uni-Mol Compatibility Acceptance Checklist

> **Compatibility-only acceptance.** This checklist covers the still-supported
> `train_model_unimol_legacy_adapter` and
> `predict_candidates_unimol_legacy_adapter`. It is not Molly's default
> architecture, an active roadmap, or permission to bypass `RunPlanExecutor`.
> Current priorities and status are defined only in `todo.md`.

This acceptance is manual because it can consume approved remote compute,
private runtime configuration, and project-specific data. Only sanitized
summaries may enter the public repository.

## Scope

- Exercise one approved Uni-Mol training run and one candidate prediction run
  through the existing compatibility adapters.
- Prove both actions remain bound to RunPlan policy, an immutable execution
  snapshot, explicit human approval, project-scoped state, and registered
  artifacts.
- Verify failure, retry, review, and asset-promotion boundaries without
  publishing private runtime locators or raw scientific data.

## Compatibility boundary

The legacy adapters remain imported and tested, but their external support
scripts are not the default Molly architecture. If this compatibility path is
being exercised, `AI4S_WORKSPACE` may point to a private support workspace that
contains the required legacy launcher and scoring scripts. Its resolved value,
script contents, connection details, and runtime outputs must remain outside
Git. Normal Molly development and packaged parser/cleaning fallbacks do not
require this layout.

## Preconditions

- [ ] Install the current public checkout with `python -m pip install -e
  ".[dev]"` in an isolated environment.
- [ ] Run the relevant local tests before using remote resources.
- [ ] Confirm the project, dataset, and intended Uni-Mol runtime are approved.
- [ ] Confirm private connection and environment profiles live in the
  user-level Molly configuration directory.
- [ ] If required for this compatibility path, set `AI4S_WORKSPACE` only in the
  private launch environment and verify the required scripts exist.
- [ ] Confirm SSH credentials, host verification data, usernames, and remote
  paths are supplied outside the repository.
- [ ] Confirm the training and candidate inputs are authorized for remote
  processing.
- [ ] Start Molly from the current public repository and create or select a
  project.

## Training acceptance

- [ ] Create a RunPlan whose `train_model` task selects
  `train_model_unimol_legacy_adapter`; do not invoke a gated adapter through a
  direct API shortcut.
- [ ] Review the frozen task options, input artifact digests, adapter policy,
  and required gate in the execution snapshot.
- [ ] Approve only the current snapshot through the normal resume boundary.
- [ ] Verify a changed dataset, option, script, or referenced artifact invalidates
  the approval instead of executing.
- [ ] Verify the run reaches a terminal StageState or a classified failure.
- [ ] Verify `model_metadata`, the training report, and model package manifests
  are registered in `artifact_registry.json`.
- [ ] Verify `stage.json` and execution confirmation records identify the
  approved snapshot without exposing a private command or credential.

## Prediction acceptance

- [ ] Register the candidate dataset as an immutable project artifact.
- [ ] Create a RunPlan whose prediction task selects
  `predict_candidates_unimol_legacy_adapter`.
- [ ] Review and approve the exact prediction snapshot through the normal gate
  and resume path.
- [ ] Verify the output contains the expected prediction or score column and is
  registered in `artifact_registry.json`.
- [ ] Run ranking and report generation only against the registered prediction
  artifact.
- [ ] Verify the report clearly labels model predictions as predictions, not
  computational or experimental validation.

## Asset promotion

- [ ] Review model diagnostics, limitations, prediction, ranking, and report
  artifacts before asset promotion.
- [ ] Promote only a confirmed, compatible model package through the controlled
  asset-promotion API/UI.
- [ ] Verify the promotion record contains logical identifiers, source artifact
  digests, approver, version, and rollback asset.
- [ ] Confirm promotion does not overwrite an earlier asset version.

## Failure, recovery, and rollback

- [ ] Verify remote or tool failures are classified with a reason, retryability,
  and suggested action.
- [ ] Retry only through the current project-scoped lifecycle and only when the
  latest failed state is explicitly eligible.
- [ ] Confirm recovery cannot infer success from stdout, mutable telemetry, or
  an output filename; authoritative artifacts and execution records must agree.
- [ ] If acceptance fails, do not promote the model, prediction, or report.

## Acceptance sign-off

- [ ] Training and prediction completed through `RunPlanExecutor` with exact
  snapshot-bound approvals.
- [ ] `stage.json`, `artifact_registry.json`, execution records, and publication
  or report artifacts reconcile.
- [ ] A fresh process can inspect the terminal state without repeating work.
- [ ] Scientific claims remain within the evidence actually produced.
- [ ] The public evidence summary contains no hostname, username, private path,
  credential, raw user data, or runtime bundle.
- [ ] A human reviewer recorded the decision and date in private acceptance
  records.

## Public evidence fields

Record only logical project/run IDs, schema versions, safe content digests,
aggregate outcomes, claim boundaries, and the reviewer decision. Keep dataset
paths, remote job IDs, command output, support-script paths, host verification
material, and full reports in the approved private environment.
