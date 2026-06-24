# Resume Intent Validation Semantics

Status: design only.

This document defines how a future gate/resume/execute path should consume
`review/replan_resume_intent.json` created by a user-confirmed replan
application. It does not add a resume route, enqueue jobs, execute adapters,
mutate `RunPlan`, call LLMs, apply patches, or replace `/api/run-plan/execute`.

## Context

Phase 4 now has a review-only replan application path:

```text
RunPlanReplanProposal
-> ReplanApplicationRequest
-> replan_application_record.json
-> replan_resume_intent.json | run_plan_revision.json | blocked_acknowledgement.json
-> replan application audit
-> compact project memory summary
```

`ResumeIntent` is the only application result that can eventually feed a
resume path without changing the task graph. It is still not executable. A
future resume bridge must treat it as a signed-off review artifact that needs
fresh validation against current run state before any adapter can run.

## Non-Goals

This design does not:

- add a new resume route
- call `RunPlanExecutor.resume_after_gate(...)`
- enqueue a worker job
- run adapters or model training
- mutate `RunPlan`
- apply a proposed patch
- call an LLM
- migrate the default `/api/run-plan/resume` or `/api/run-plan/execute` routes
- implement queued waiting-user resume
- add remote worker or SQLite behavior

## ResumeIntent Validity

A `ResumeIntent` is valid only when all of the following are true:

1. The JSON validates as the current `ResumeIntent` schema.
2. `executable` is exactly `false`.
3. `action` is not `block`.
4. `project_id` and `run_id` match the current request context.
5. `source_application_id` is non-empty and resolves to a matching
   `replan_application_record.json`.
6. The source application record has `result_type="resume_intent"`.
7. The source application record `result_ref` points to the same resume intent
   artifact being validated.
8. The source application record has `executable=false`.
9. The source application record links back to a proposal artifact and
   `proposal_hash`.
10. The proposal artifact still exists and rehashes to the recorded
    `proposal_hash`.
11. The proposal artifact still has `executable=false` and
    `proposed_run_plan_patch.applied=false`.
12. All selected operation ids in the application record still exist in the
    proposal artifact operations.
13. Every task referenced by `affected_tasks`, `rerun_tasks`, or
    `resume_from_task` still exists in the current `RunPlan`.
14. The current run state is still compatible with resume; for the existing
    executor this means the run is waiting for user action at a known stage.
15. Required gates are still pending or explicitly approved by the later
    resume request. `ResumeIntent` creation does not approve them.

If any condition fails, validation must fail closed and produce a reviewable
validation result. It must not attempt partial resume or infer missing links.

## Artifact Chain Validation

The future validator should read the run directory only through
`ProjectStorage` and artifact registry references. Expected artifacts:

```text
artifact_registry.json
review/replan_proposal.json
review/replan_application_record.json
review/replan_resume_intent.json
review/replan_application_audit.jsonl
```

Validation steps:

1. Resolve `replan_resume_intent` from the artifact registry when available.
2. Resolve `replan_application_record` from the artifact registry.
3. Ensure both relative paths stay under the run directory.
4. Load `replan_resume_intent.json`.
5. Load `replan_application_record.json`.
6. Confirm `resume_intent.source_application_id` equals
   `application_record.application_id`.
7. Confirm `application_record.result_ref` equals the relative resume-intent
   path.
8. Resolve `application_record.proposal_artifact_ref`.
9. Recompute the proposal hash and compare it with
   `application_record.proposal_hash`.
10. Confirm the proposal operation set includes all
    `application_record.selected_operation_ids`.

Do not trust a client-supplied artifact path. The client may submit an artifact
id or the known relative ref, but the service must resolve it against the run
directory and registry.

## Source Application And Proposal Hash

`source_application_id` is the stable link between `ResumeIntent` and
`ReplanApplicationRecord`. It is not permission to resume.

Validation must check:

- `resume_intent.source_application_id == application_record.application_id`
- `application_record.result_type == "resume_intent"`
- `application_record.result_ref == "review/replan_resume_intent.json"` or the
  registry-resolved equivalent
- `application_record.proposal_hash` starts with `sha256:`
- the proposal artifact hash still equals `application_record.proposal_hash`
- selected operation ids are present and not duplicated
- the selected action in the application record matches
  `resume_intent.action`

If the proposal hash no longer matches, the intent is stale. The system should
return `stale_intent` and require a new verifier/proposal/application cycle.

## Current RunPlan Compatibility

The resume validator must receive or load the current `RunPlan` before any
resume operation is considered.

Required checks:

- `current_run_plan.run_id == resume_intent.run_id`
- all `rerun_tasks` exist in `current_run_plan.tasks`
- all `affected_tasks` exist in `current_run_plan.tasks`
- `resume_from_task`, when present, exists in `current_run_plan.tasks`
- `block` actions are rejected because they cannot create a resume intent
- `adjust_targets` and `collect_more_data` intents are rejected because they
  should create `RunPlanRevision` artifacts instead
- for `rerun_task`, the task graph must not require new tasks, new inputs, or
  changed dependencies; otherwise the application should have produced a
  `RunPlanRevision`

First implementation should compare task ids and task order. A later
implementation should add a stable `run_plan_fingerprint` to application
artifacts so validation can detect dependency, option, and target changes more
precisely.

## Gate Requirements

`ResumeIntent` does not approve gates. It only records which gates are expected
before resume can proceed.

Gate validation should require:

- actor identity from the shared resolver
- permission grant for resume-intent consumption, for example
  `run_plan_resume_intent_use`
- confirmation that `resume_intent.required_gates` is a subset of the gates
  required by the current waiting stage
- no unknown or future gate approvals
- no missing current-stage gate approvals
- current executor snapshot validation remains in force

For current executor compatibility, the final resume call should still use the
existing gate approval semantics:

```text
approved_gates supplied by user
-> exact match against current waiting task gates
-> snapshot validation for artifacts and task options
-> only then resume_after_gate can run
```

The resume-intent validator must not write gate decisions by itself. Gate
decisions are written only by the actual gate/resume path after approval.

## Stale Intent Prevention

An intent is stale if any of these are true:

- proposal hash no longer matches the proposal artifact
- application record no longer points to this intent
- artifact registry points to different application or intent refs
- current `RunPlan` no longer contains referenced tasks
- current run is not in `WAITING_USER`
- current waiting stage does not match `resume_from_task`, when supplied
- current waiting stage is not one of `rerun_tasks` or `affected_tasks` for
  rerun intents
- another newer replan application record exists for the same run and selected
  operation set
- a newer `RunPlanRevision` has been accepted for the same run
- the intent was already consumed by a terminal resume audit event

The first implementation can detect staleness using artifact refs, proposal
hash, current `RunPlan`, stage state, and audit records. Later implementations
should add explicit `run_plan_fingerprint`, `intent_consumed_at`, and
`superseded_by_application_id` fields.

## Validation Result Shape

The future validator should return a fixed, non-executing result such as:

```json
{
  "ok": true,
  "project_id": "proj-a",
  "run_id": "run-a",
  "intent_id": "resume-replan-application-run-a-abc123",
  "source_application_id": "replan-application-run-a-abc123",
  "decision": "resume_eligible",
  "required_gates": ["gate_replan_rerun_task", "gate_3_train_config"],
  "rerun_tasks": ["train_model"],
  "resume_from_task": "train_model",
  "artifact_refs": {
    "replan_application_record": "review/replan_application_record.json",
    "replan_resume_intent": "review/replan_resume_intent.json",
    "replan_proposal": "review/replan_proposal.json"
  },
  "validation_findings": [
    "proposal_hash_valid",
    "application_record_matches_intent",
    "rerun_tasks_present_in_current_run_plan",
    "current_run_waiting_for_user"
  ],
  "error": null,
  "executable": false
}
```

Possible `decision` values:

```text
resume_eligible
needs_gate_approval
stale_intent
invalid_intent
blocked
```

`executable` remains `false` even for `resume_eligible`. The result authorizes
only the next review/gate step, not adapter execution.

## Resume Audit

Resume-intent validation should write audit records separate from the actual
executor resume audit.

Suggested audit events:

```text
resume_intent_validation_requested
resume_intent_validation_denied
resume_intent_validation_succeeded
resume_intent_validation_failed
resume_intent_consumed
```

Audit records should include:

- actor and actor_source
- project_id and run_id
- intent_id
- source_application_id
- proposal_hash
- proposal_artifact_ref
- resume_intent_ref
- application_record_ref
- permission action and grant id
- validation decision
- required gates
- approved gates, if supplied for validation
- rerun_tasks and resume_from_task
- stale reason, if any
- concise error type and message
- timestamp

Audit write failure should fail closed before any resume bridge calls the
executor. If validation audit cannot be written, no gate decision or resume
call may occur.

## Permission Model

The validation path should require a new permission action distinct from
application creation:

```text
run_plan_resume_intent_use
```

This separation matters because applying a proposal creates review artifacts,
while consuming a resume intent moves the system closer to execution.

The permission decision should be audited with:

- action
- project_id
- run_id
- actor
- actor_source
- allowed
- reason
- grant_id
- server_authorized

Legacy client approval flags should not authorize resume-intent consumption in
production or internal routes.

## Default Route Compatibility

The existing routes remain unchanged:

- `POST /api/run-plan/execute`
- `POST /api/run-plan/resume`

The future resume-intent validator should be an internal helper or
feature-flagged internal route first. It may produce a validated payload that a
separate user-confirmed resume path can later consume, but it must not silently
wrap or replace `/api/run-plan/resume`.

Default-route migration is out of scope until:

1. Resume-intent validation has fixed schema tests.
2. Resume-intent audit is fail-closed.
3. Current `RunPlan` compatibility checks cover task ids, task order, gates,
   and stage state.
4. Stale-intent detection is tested.
5. Existing `/api/run-plan/resume` snapshot and gate tests remain green.
6. A user confirmation gate explicitly authorizes the final resume.

## Future Implementation Order

Recommended next PRs:

1. Add `ResumeIntentValidationRequest` and `ResumeIntentValidationResult`
   schemas.
2. Add a deterministic `validate_resume_intent(...)` helper that reads only
   artifacts and current run state.
3. Add resume-intent audit writer and compact memory summary, still without
   execution.
4. Add a feature-flagged internal validation route that returns validation
   results only.
5. Add a separate user-confirmed resume bridge that calls the existing resume
   path only after gate, actor, permission, audit, stale-intent, and snapshot
   validation pass.

Do not combine these steps with queued execution migration, remote workers,
SQLite, heavy training, web acquisition, or default-route replacement.
