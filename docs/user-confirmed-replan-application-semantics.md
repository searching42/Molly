# User-Confirmed Replan Application Semantics

Status: design only.

This document defines how a user-confirmed `RunPlanReplanProposal` should move
from a review artifact into a safe resume or modified-plan path. It does not
introduce execution code, API routes, queue behavior, adapter execution, LLM
calls, or automatic proposal application.

## Context

The current Phase 4 review loop is intentionally non-executing:

- `RunPlanArtifactVerification` reads artifacts and emits a fixed verifier
  decision.
- `RunPlanReplanProposal` maps verifier findings to a deterministic,
  reviewable proposal.
- `RunPlanReplanProposal.executable` must remain `false`.
- `proposed_run_plan_patch.applied` must remain `false`.
- Review artifacts, review cards, and project memory summaries store only
  review information and artifact references.

The missing design question is what happens after a human user confirms one of
those proposals. The answer must keep proposal generation, proposal
application, and actual execution as separate boundaries.

## Principles

1. A proposal is advisory data, not an execution request.
2. User confirmation creates an application record, not direct execution.
3. Applying a proposal can produce either a `ResumeIntent` or a
   `RunPlanRevision`, depending on whether the task graph changes.
4. A `RunPlanRevision` is still non-executable until the required gates and
   explicit execution path approve it.
5. LLMs and rule-based proposal generators cannot set `applied=true`,
   `executable=true`, enqueue jobs, or mutate a `RunPlan`.
6. Every user-confirmed transition requires actor identity, permission
   decision, audit records, and compact memory summary semantics.

## Confirmable Actions

The current proposal action set is:

```text
continue
request_review
rerun_task
adjust_targets
collect_more_data
block
```

All actions can be acknowledged by a user, but only some can lead toward a
resume or revised run plan.

| Proposal action | User confirmation effect | Resume current RunPlan? | Create revised RunPlan? | Execution allowed immediately? |
| --- | --- | --- | --- | --- |
| `continue` | Confirm that no plan change is needed. | Yes, only if no unresolved gate remains. | No. | No. A separate resume/execute call is still required. |
| `request_review` | Record that the user reviewed the findings and selected a resolution. | Yes, if the current plan only needs gate approval or user review. | Only if the resolution changes tasks, targets, or inputs. | No. |
| `rerun_task` | Approve rerunning listed existing tasks. | Yes, if the task graph and task options are unchanged. | Yes, if dependencies, options, inputs, or task list must change. | No. |
| `collect_more_data` | Approve adding or completing data collection or preprocessing work. | Only if an existing pending collection task already exists. | Usually yes, because new artifacts or data tasks are added. | No. |
| `adjust_targets` | Approve target/property/objective changes. | No, except for a pure metadata clarification that does not affect tasks. | Yes, by default. | No. |
| `block` | Acknowledge that the run is blocked. | No. | No automatic revision. | No. |

`block` is confirmable only as an acknowledgement. It must not resume, rerun, or
create a revised plan unless a later user request or verifier result produces a
new non-blocking proposal.

## Application Lifecycle

The future application flow should be:

```text
review artifacts
-> user selects proposal action and operations
-> server validates actor, permission, proposal hash, and current run state
-> server writes requested audit record
-> server compiles selected operations with deterministic allowlisted rules
-> server writes a replan application record
-> server creates either ResumeIntent, RunPlanRevision, or BlockedAcknowledgement
-> server writes terminal audit and compact memory summary
-> user separately approves gates and resumes or executes the selected path
```

The application endpoint, helper, or service must stop before execution. It can
only produce records that a later gate/resume/execute path may consume.

## Proposed Patch Application Contract

`proposed_run_plan_patch` stays immutable as a review artifact:

```json
{
  "schema_version": "reviewable_run_plan_patch.v1",
  "applied": false,
  "operations": [
    {
      "operation_id": "op_000001",
      "op": "rerun_task",
      "task_id": "train_model",
      "source_finding_id": "poor_model_metrics_xxx",
      "category": "poor_model_metrics",
      "reason": "Model metrics are weak enough to recommend a rerun."
    }
  ]
}
```

User confirmation does not mutate this artifact. Instead, the server creates a
new application record such as:

```json
{
  "application_id": "replan-application-run-a-001",
  "project_id": "proj-a",
  "run_id": "run-a",
  "proposal_artifact_ref": "review/replan_proposal.json",
  "proposal_hash": "sha256:...",
  "schema_version": "reviewable_run_plan_patch.v1",
  "selected_action": "rerun_task",
  "selected_operation_ids": ["op_000001"],
  "applied": true,
  "result_type": "resume_intent",
  "result_ref": "review/replan_resume_intent.json",
  "executable": false
}
```

Important rules:

- `applied=true` exists only on the application record, not on the original
  proposal artifact.
- The server must re-load and hash the proposal artifact before applying it.
- The selected action must match the proposal action, or must be a stricter
  safe action such as `request_review` or `block`.
- Unknown patch operations are rejected.
- Patch operations are declarative. They are compiled by allowlisted server
  transformers and never interpreted as code.
- A stale proposal hash, stale current run revision, missing actor, missing
  permission, or missing gate context rejects the application.

## Operation Identity

Every `proposed_run_plan_patch.operations[]` entry must include an explicit,
stable `operation_id`.

Rules:

- `selected_operation_ids` can reference only `operation_id` values present in
  the proposal artifact.
- `operation_id` is a stable review, application, and audit selection anchor.
  It is not execution permission and does not by itself authorize a rerun,
  target change, data collection, or resume.
- The application service must re-load the proposal artifact, verify
  `proposal_hash`, and then verify that every `selected_operation_ids` entry
  exists in `proposed_run_plan_patch.operations[]`.
- Unknown `operation_id` values must be rejected.
- Duplicate `selected_operation_ids` should be rejected in the first
  implementation. This keeps audit and user-confirmation semantics explicit.
- `operation_id` should not be generated by the client at application time. It
  should be written by the proposal generator when creating the
  `RunPlanReplanProposal`.
- The first implementation can use deterministic sequence IDs such as
  `op_000001`, `op_000002`, or a short hash derived from a canonical operation
  payload. In either case, once an `operation_id` is written into a proposal
  artifact, it must remain immutable within that artifact.
- The application record should store `selected_operation_ids` and
  `proposal_hash`, not a copied operation payload. Consumers that need display
  details should read the original proposal artifact by reference.
- Historical proposal artifacts without `operation_id` must fail closed on the
  application path, or require proposal regeneration first. Application code
  must not infer operation identity from `task_id`, `category`, `reason`, or
  other operation fields.

## ResumeIntent

Use a `ResumeIntent` when the current `RunPlan` can be resumed without changing
its task graph.

Suggested fields for a future schema:

```text
intent_id
project_id
run_id
source_application_id
action
affected_tasks
approved_gates
rerun_tasks
resume_from_task
required_gates
reason
executable=false
created_at
created_by
actor_source
```

Valid `ResumeIntent` cases:

- `continue` with no task graph changes.
- `request_review` where the user only approves or resolves an existing gate.
- `rerun_task` where the affected task already exists and task options,
  dependencies, inputs, and targets do not change.

A `ResumeIntent` is not execution. It becomes eligible for execution only after
the normal gate/resume path validates it.

The follow-up validation contract is defined in
`docs/resume-intent-validation-semantics.md`. That document keeps validation
separate from execution: it covers source application id, proposal hash,
artifact refs, current `RunPlan` compatibility, stale-intent detection, gates,
and resume audit, but does not call the resume path.

## RunPlanRevision

Use `RunPlanRevision` when the proposal changes the task graph, run targets,
task options, data inputs, dependencies, or expected outputs.

Cases that should create a revised plan:

- `adjust_targets` changes target properties, objectives, thresholds, weights,
  domain profile, or ranking policy.
- `collect_more_data` adds acquisition, parsing, extraction, normalization, or
  confirmation tasks.
- `rerun_task` changes model hyperparameters, adapter options, input artifacts,
  dependencies, or output artifacts.
- `request_review` resolves into a substantive plan change.

The existing `RunPlanRevision` schema already captures previous plan, revised
plan, diff, reason, recovery actions, approvals required, questions, and
`executable=false`. Future application code should reuse it rather than
inventing a separate revised-plan object.

The revised plan should receive a new revision id. It can keep the same
`run_id` if treated as a revision of the same run, or use a child run id if the
storage model later needs side-by-side execution histories. The first
implementation should prefer same `run_id` plus explicit `revision_id` to align
with the current `RunPlanRevision` schema.

## BlockedAcknowledgement

Use a blocked acknowledgement when the confirmed proposal action is `block`.

It should record:

```text
application_id
project_id
run_id
blocked_reason
source_finding_ids
required_user_decisions
created_by
actor_source
created_at
```

It must not create a `ResumeIntent`, create a `RunPlanRevision`, approve gates,
enqueue work, or execute adapters.

## Gate Requirements

Proposal application needs gates separate from execution gates.

Always required:

- Actor identity from the shared resolver.
- Permission grant for proposal application, for example
  `run_plan_replan_apply`.
- A user confirmation gate for the selected action and selected operations.
- Requested and terminal audit records.

Action-specific gates:

- `request_review`: approve or reject each referenced existing gate before
  resume. If the review changes the task graph, require revised-plan approval.
- `rerun_task`: require a rerun approval gate listing affected tasks. Existing
  task gates, such as train configuration or external execution gates, must be
  re-evaluated.
- `collect_more_data`: require data collection approval. If network, PDF
  acquisition, web search, MinerU parsing, or external databases are involved,
  require external acquisition gates.
- `adjust_targets`: require target adjustment approval. If target changes imply
  retraining, generation, or ranking changes, those task gates must be
  re-evaluated.
- `block`: require only acknowledgement. No resume or execution gate is
  satisfied by this acknowledgement.

Memory writes remain permissioned. Saving an application summary to project
memory should use the existing project-memory permission model or a future
server-owned memory write path, not a client-supplied flag.

## Actor, Audit, And Memory

Every application attempt should record actor and audit context.

Actor fields:

- `actor`
- `actor_source`
- `confirmed_by`
- `project_id`
- `run_id`

Audit events:

- `replan_application_requested`
- `replan_application_denied`
- `replan_application_validated`
- `resume_intent_created`
- `run_plan_revision_created`
- `blocked_acknowledgement_created`
- `replan_application_completed`
- `replan_application_failed`

Audit records should include:

- proposal artifact references and hashes
- selected action
- selected operation ids
- affected tasks
- permission decision metadata
- gate requirements
- result type
- result artifact refs
- status code or helper status
- concise error type and message

Project memory should store only a compact application summary:

- verifier decision
- proposal action
- confirmed action
- affected tasks
- required user decisions
- result type
- result refs
- audit refs
- artifact refs

Project memory must not store raw CSVs, full model artifacts, full verifier
payloads, full proposal payloads, raw document text, or executable patches.

## Preventing Automatic Execution

The following invariants prevent proposal or LLM output from automatically
executing:

1. `RunPlanReplanProposal.executable` remains `false`.
2. `proposed_run_plan_patch.applied` remains `false`.
3. Application creates a separate application record and result artifact.
4. Application result artifacts also remain `executable=false`.
5. No proposal field is interpreted as Python code, shell command, adapter
   payload, local path, or queue task.
6. Only allowlisted patch operation names are accepted.
7. The server revalidates current run state, proposal hash, actor, permission,
   and gates before creating any result artifact.
8. Creating `ResumeIntent` or `RunPlanRevision` does not enqueue work.
9. A separate gate/resume/execute path is required for any execution.
10. LLMs can summarize or suggest, but cannot call the application path without
    an explicit user confirmation and server-side permission decision.

## Future Implementation Sequence

Recommended implementation order after this design PR:

1. Add fixed schemas for `ReplanApplicationRequest`,
   `ReplanApplicationRecord`, `ResumeIntent`, and `BlockedAcknowledgement`.
2. Add a deterministic patch validator/compiler with allowlisted operations.
3. Add tests that map each proposal action to the correct result type.
4. Add actor, permission, audit, and memory summary tests.
5. Add a feature-flagged internal review route for application only.
6. Add gate/resume integration tests without default-route migration.
7. Only after those pass, evaluate whether a confirmed `ResumeIntent` or
   `RunPlanRevision` can opt into queued execution.

Do not combine these steps with remote worker support, SQLite migration, heavy
training, web acquisition, or default `/api/run-plan/execute` replacement.

## Non-Goals

This design does not:

- execute proposals
- apply patches in code
- modify `RunPlan`
- enqueue jobs
- call adapters
- call LLMs
- introduce a new public API route
- replace `/api/run-plan/execute`
- implement full queued resume for `WAITING_USER`
- add remote workers
- migrate storage to SQLite
