# Manual Real Uni-Mol Acceptance Checklist

This checklist is the Phase 1 manual acceptance gate for a real remote Uni-Mol run. It is intentionally not an automated pytest because it may use remote GPU resources, SSH credentials, and project-specific data.

## Scope

- Validate one real Uni-Mol training run through `train_model_unimol_legacy_adapter`.
- Validate one real Uni-Mol candidate prediction run through `predict_candidates_unimol_legacy_adapter`.
- Confirm the run produces standard `workspace/agent` artifacts and can be reviewed, retried, and promoted through the Phase 1 UI/API.

## Preconditions

- [ ] Set `AI4S_WORKSPACE` to the parent workspace that contains `claude/` and `scripts/`.
- [ ] Confirm the remote Uni-Mol runtime is approved for this project.
- [ ] Confirm any SSH credentials or remote runtime secrets are available outside the repository.
- [ ] Confirm the training CSV is approved for remote processing.
- [ ] Confirm the candidate CSV is approved for remote prediction.
- [ ] Confirm `workspace/agent` dependencies are installed and tests pass locally.
- [ ] Start the local Flask app from `workspace/agent`.

## Training Acceptance

- [ ] Create or select a project in the UI/API.
- [ ] Upload/register the training dataset as `train_dataset`.
- [ ] Build the data confirmation card and review all five sections.
- [ ] Build the run confirmation card and confirm the `train_model` gate.
- [ ] Call `train_model_unimol_legacy_adapter` with `execute=false` and review the planned command.
- [ ] Call `train_model_unimol_legacy_adapter` with `execute=true` only after confirmation.
- [ ] Save the raw training stdout/stderr tail or remote job id.
- [ ] Verify the Uni-Mol training report JSON exists.
- [ ] Verify `model_metadata` exists and includes run id, property id, backend, model path, and created time.
- [ ] Verify `stage.json` records the training stage status and history.
- [ ] Verify `artifact_registry.json` contains the training report and `model_metadata` artifacts.

## Prediction Acceptance

- [ ] Upload/register the candidate dataset as `candidate_dataset`.
- [ ] Call `predict_candidates_unimol_legacy_adapter` with `execute=false` and review the planned command.
- [ ] Call `predict_candidates_unimol_legacy_adapter` with `execute=true` only after confirmation.
- [ ] Verify the prediction CSV exists and contains at least one Uni-Mol prediction or score column.
- [ ] Verify the prediction artifact is registered in `artifact_registry.json`.
- [ ] Run `filter_rank` on the prediction CSV and verify TopN export.
- [ ] Run `render_report` and verify Markdown, HTML, and JSON reports exist.
- [ ] Open Report Preview in the UI and verify the final report content is readable.

## Asset Promotion

- [ ] Review the model, prediction, ranking, and report artifacts before asset promotion.
- [ ] Promote only confirmed assets through the asset promotion UI/API.
- [ ] Verify the asset promotion record contains asset id, asset type, version, source artifacts, approver, and timestamp.
- [ ] Confirm promoted assets do not overwrite previous versions.

## Failure And Rollback

- [ ] If remote training fails, verify `stage.json` includes the error category, reason, retryable flag, and suggested action.
- [ ] If the failure is retryable, use the retry endpoint only for the latest failed stage or explicitly retryable stage.
- [ ] If the run is not accepted, do not promote model or report artifacts.
- [ ] Record rollback notes in the run log or acceptance notes.

## Acceptance Sign-Off

Use this acceptance sign-off section to record the human decision after reviewing the real remote Uni-Mol evidence.

- [ ] Training completed on the real remote Uni-Mol path.
- [ ] Candidate prediction completed on the real remote Uni-Mol path.
- [ ] Standard Phase 1 artifacts were generated and registered.
- [ ] UI cards showed data confirmation, run confirmation, stage timeline, report preview, and asset promotion state.
- [ ] Human reviewer accepted the scientific output and recorded approver/date.

## Evidence To Attach

- Project id:
- Run id:
- Training dataset path:
- Candidate dataset path:
- Remote job id or stdout/stderr excerpt:
- `stage.json` path:
- `artifact_registry.json` path:
- `model_metadata` path:
- Final report path:
- Asset promotion record path:
- Approver and date:
