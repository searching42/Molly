# Resume Intent Validation Semantics

Status: validation contract is implemented, and PR #115 adds run-plan and
stage-state fingerprints for stale-intent detection. PR #116 adds
strict waiting-stage, execution-snapshot, and executor-gate compatibility checks.
PR #117 adds the feature-flagged one-time internal execution bridge, and PR #118
adds an end-to-end user-confirmed resume-loop fixture test. This document
describes the migration contract and remaining hardening gates for default-route
adoption.

The PR #117/118 execution path is intentionally narrow and internal; default
`/api/run-plan/execute` and `/api/run-plan/resume` replacement are still out of
scope here.

This document defines how a default-route migration path should consume
`review/replan_resume_intent.json` created by a user-confirmed replan
application. It does not add another resume route, enqueue jobs, execute adapters,
mutate `RunPlan`, call LLMs, apply patches, or replace `/api/run-plan/execute`.

## Context

Phase 4 now has a user-confirmed replan application path:

```text
RunPlanReplanProposal
-> ReplanApplicationRequest
-> replan_application_record.json
-> replan_resume_intent.json | run_plan_revision.json | blocked_acknowledgement.json
-> replan application audit
-> compact project memory summary
```

`ResumeIntent` is the only application result that can eventually feed a
resume path without changing the task graph. The PR #117 + PR #118 loop now
adds the current narrow execution bridge and closed-loop verification flow, while
default-route migration still requires this contract plus target-job-safe queue
acquisition.

## Current Closed-Loop Status

`PR #118` (plus the shared PR #117 execution bridge) proves the current user-confirmed
closed loop at feature-flagged scope:

```text
RunPlanReplanProposal
-> replan_application_record.json
-> replan_resume_intent.json
-> /api/internal/run-plan/replan/apply-review
-> internal resume-intent validation
-> /api/internal/run-plan/resume-intent/execute
-> post-resume review artifacts
```

This is a full review→application→validation→execution→re-review cycle inside the
opt-in path and is intended as the migration hard gate before default-route changes.

## Non-Goals

This design does not:

- add or change default-route resume/execute behavior
- enqueue a worker job
- run adapters or model training
- mutate `RunPlan`
- apply a proposed patch
- call an LLM
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
13. `resume_state_binding` exists on both `replan_application_record.json` and
    `replan_resume_intent.json`, and the two bindings are identical.
14. The current schema-normalized `RunPlan` fingerprint matches the recorded
    binding.
15. The current stable `StageState` fingerprint matches the recorded binding.
16. Every task referenced by `affected_tasks`, `rerun_tasks`, or
    `resume_from_task` still exists in the current `RunPlan`.
17. The current run state is still compatible with resume; for the existing
    executor this means the run is in `WAITING_USER` at a task present in both
    the current `RunPlan` and `AtomicTaskRegistry`.
18. The bound stage, current `StageState.stage`, and
    `resume_intent.resume_from_task` are identical.
19. The current stage has a complete `details.execution_snapshot` with
    `snapshot_id`, `snapshot_hash`, `run_id`, `task_id`, `run_plan`, and
    `approved_gates`.
20. The execution snapshot hash matches its canonical computational material.
    Existing executor snapshots may store a bare 64-character SHA-256 digest;
    resume validation canonicalizes it to `sha256:<digest>`.
21. The current stage required gates and execution snapshot gates match the
    gate list in `AtomicTaskRegistry` for the current waiting task.
22. Required executor gates are still pending or explicitly approved by the
    later validation/resume request. `ResumeIntent` creation does not approve
    them.

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

PR #115 adds a canonical `run_plan_fingerprint` to the application record and
resume intent. Validation recomputes the current fingerprint and fails closed
with `decision="stale_intent"` and
`error.type="run_plan_fingerprint_mismatch"` when the current plan differs.

## Run-Plan And Stage-State Binding

`ResumeStateBinding` records compact state identity when a user-confirmed
replan application creates a `ResumeIntent`:

```json
{
  "schema_version": "resume_state_binding.v1",
  "run_plan_fingerprint": "sha256:<64 hex>",
  "stage_fingerprint": "sha256:<64 hex>",
  "stage": "train_model",
  "stage_status": "WAITING_USER",
  "execution_snapshot_id": "snapshot-1",
  "execution_snapshot_hash": "sha256:<64 hex>"
}
```

The run-plan fingerprint is computed from the complete schema-normalized
`RunPlan` JSON using sorted JSON object keys and stable separators. List order
is preserved, so task ordering, dependencies, requested tasks, available
artifacts, missing artifacts, and any future schema fields included in
`RunPlan.model_dump(mode="json")` affect the fingerprint.

The stage fingerprint is computed from stable stage semantics only:

- `stage`
- `status`
- `next_stage`
- normalized sorted `details.required_gates`
- ordered `details.executed_tasks`
- `details.execution_snapshot.snapshot_id`
- `details.execution_snapshot.snapshot_hash`

It intentionally ignores volatile timestamps, stage history timestamps, audit
records, memory records, markdown/report contents, and raw execution snapshot
payloads. If `execution_snapshot_id` or `execution_snapshot_hash` is present,
both must be present.

The binding is stale-state detection only. It is not a digital signature, not
an authorization mechanism, and not a substitute for actor, permission, gate,
or executor snapshot checks. Any future actual resume bridge must recompute the
current fingerprints immediately before consuming the intent to avoid TOCTOU
drift between validation and execution.

## Gate Requirements

`ResumeIntent` does not approve executor gates. It records two distinct gate
domains:

- `application_required_gates`: review/application gates such as
  `gate_replan_rerun_task`. These explain why a user-confirmed application was
  needed.
- `required_gates`: executor gates for the current waiting task, derived from
  `AtomicTaskRegistry` and the current execution snapshot.

`ResumeIntent.approved_gates` must remain empty. If an artifact embeds executor
gate approvals, validation fails closed with
`resume_intent_embeds_gate_approval`. Gate approvals are supplied only by the
later validation/resume request and are never copied into the intent artifact.

Gate validation should require:

- actor identity from the shared resolver
- permission grant for resume-intent consumption, for example
  `run_plan_resume_intent_use`
- confirmation that `resume_intent.required_gates` equals the executor gates
  required by the current waiting stage
- confirmation that `resume_intent.application_required_gates` is not treated
  as executor approval
- no unknown, duplicate, non-string, or future gate approvals
- no missing current-stage executor gate approvals
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

The validation-only helper returns:

- `needs_gate_approval` when required executor gates are missing.
- `resume_eligible` when the request supplies exactly the current executor
  gates.
- `blocked` with `unexpected_gate_approval` when supplied approvals include an
  application gate or any gate outside the current executor gate set.

## Stale Intent Prevention

An intent is stale if any of these are true:

- proposal hash no longer matches the proposal artifact
- application record no longer points to this intent
- application record or resume intent is missing `resume_state_binding`
- application record and resume intent have different bindings
- current `RunPlan` fingerprint differs from the recorded binding
- current `StageState` is missing or fingerprints to a different semantic
  payload
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

The first implementation detects staleness using artifact refs, proposal hash,
current `RunPlan`, current `StageState`, recorded state fingerprints, and audit
records. Later implementations may add `intent_consumed_at` and
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
  "resume_state_binding": {
    "schema_version": "resume_state_binding.v1",
    "run_plan_fingerprint": "sha256:<64 hex>",
    "stage_fingerprint": "sha256:<64 hex>",
    "stage": "train_model",
    "stage_status": "WAITING_USER",
    "execution_snapshot_id": "snapshot-1",
    "execution_snapshot_hash": "sha256:<64 hex>"
  },
  "validation_findings": [
    "proposal_hash_valid",
    "run_plan_fingerprint_valid",
    "stage_fingerprint_valid",
    "application_intent_state_binding_match",
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

The feature-flagged internal execution bridge uses a separate permission:

```text
run_plan_resume_execute
```

That bridge still reruns this validator, requires `resume_eligible`, writes a
pre-execution `resume_intent_consumed` audit record before calling
`RunPlanExecutor.resume_after_gate(...)`, and treats the intent as one-time
consumed. It does not replace the default resume route.

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

The implemented internal execution path (PR #117) is feature-flagged and still an
opt-in narrow bridge. It may produce a validated payload that a separate
user-confirmed resume path consumes, but it must not replace
`/api/run-plan/resume` or `/api/run-plan/execute`.

Default-route migration is out of scope until:

1. Resume-intent validation has fixed schema tests.
2. User-confirmed loop closure from PR #118 (proposal/apply/validate/execute/review)
   remains green and one-time intent consumption is enforced.
3. Resume-intent audit remains fail-closed.
4. Current `RunPlan` compatibility checks cover task ids, task order, gates,
   and stage state.
5. `target-job` acquisition or strict dedicated-queue guarantees are in place.
   For the internal run-plan helper, this means the service targets only the
   job it just enqueued; low-level queue selectors must not let callers redirect
   the helper to an unrelated job.
6. Stale-intent detection and execution-snapshot consistency are tested.
7. Existing `/api/run-plan/resume` snapshot/gate tests remain green.
8. A user confirmation gate still explicitly authorizes final resume execution.
9. Full queued `WAITING_USER` resume behavior remains out-of-scope until a
   target-job-safe queue resume path is added.

## Remaining migration gates before default-route queue replacement

The current validation/execution loop is internal and opt-in only. Before default
queue execution can replace synchronous execution, the following hard gates remain:

1. target-job acquisition for queue workers plus service-level ownership checks
   that prevent externally selected run-plan jobs.
2. full queued `WAITING_USER` resume engine (resume semantics beyond terminal-
   compatible metadata).
3. queued resume should still include post-execute review/card refresh.
4. no remote worker support or SQLite migration before the local gates stay green.
5. no full MinerU/Web Search/MinerU crawl/mining and no REINVENT4 or heavy
   Uni-Mol/DPA3 in the default-route migration scope.

## Future Implementation Order

Remaining engineering milestones after PR #118 are migration-focused:

1. Keep target-job acquisition constrained to service-owned run-plan jobs in
   the internal queue path.
2. Add deterministic post-resume queue/job verification before any default-route
   queue migration.
3. Keep PR #117-style resume bridge as internal-only until the above migration
   gates are green.
4. Move to default-route migration only after queue resume and migration gates
   above are explicit and tested.

Do not combine these steps with queued execution migration, remote workers,
SQLite, heavy training, web acquisition, or default-route replacement.
