# Scientific Agent Harness Controller v1

Status: PR-BN implementation with representative automated validation for
`M3H-008`; repository-owner review is still pending.

The Controller is the only PR-BN component allowed to turn an immutable
`AgentPlanStartIntent` into one bounded execution-side effect. It does not
replace the Planner, Permission Engine, plan authorization, Gate authority,
remote-resource authority, Executor, remote lifecycle, Artifact Registry,
StageState, recovery evidence, or trajectory projection.

```text
current exact start intent
  -> immutable controller execution binding
  -> deterministic inspection
  -> immutable action decision
  -> re-verify every decision source
  -> at most one bounded side effect
  -> exact-read authoritative result
  -> immutable action receipt
```

An HTTP response, trace span, mutable transport status, adapter return value,
or task name in a history list is never sufficient evidence that an action
succeeded.

## Authority matrix

| Object | Authority | Not authority for |
| --- | --- | --- |
| Plan proposal | Server-validated task graph, options, dispatch intents, and source snapshot | Permission, user approval, Gate approval, or execution |
| Permission decision | Deterministic policy result for exact proposal/authorization phase | User consent or dispatch |
| Plan authorization | User authority over one exact immutable plan and task-authority roster | Gate consumption, remote approval, or dispatch |
| Start intent | Request to let the Controller consider the exact authorization | Execution, Gate approval, remote request, or success |
| Controller execution | Immutable binding of the start intent, policy, plan, authorities, artifacts, Gates, budgets, and ordered tasks | A task-completion claim |
| Controller decision | Deterministic next-action selection from exact inspection inputs | Proof that its side effect committed |
| Controller receipt | Exact-read result of one attempted side effect plus all source/result digests | Scientific-result validity beyond the referenced authorities |
| `GateDecision` | Approval or rejection of one current Gate execution snapshot | Plan authorization or remote-compute approval |
| Remote approval | Approval of one exact server-derived remote request in one task-attempt slot | Gate approval, plan changes, or another task attempt |
| Execution snapshot | Executor authority for one exact local task payload and Gate roster | Plan-level or remote authority |
| Remote resource AuthoritySet | Current exact configured resource and aggregate budget authority | User authorization, request approval, or transport success |
| StageState | Authoritative task/run state transition anchor | Artifact contents or remote transport telemetry |
| Artifact Registry | Authoritative registered output path, digest, producer, and verification facts | Task-state success on its own |
| Remote publication | Immutable remote output proof bound to request, approval, manifest, and verifier | Another slot or logical artifact alias not explicitly registered |
| Recovery/dispatch evidence | Immutable evidence of an existing dispatch/recovery transition when present | A substitute for StageState, Registry, or publication |
| OpenTelemetry | Droppable operational observation | Any decision, idempotency, recovery, or success authority |

## Immutable controller execution

Creation consumes only the route-bound project and start-intent IDs plus a
strict `agent_harness_controller_start_request.v1` body. The body contains an
expected start-intent digest and canonical client request ID; it contains no
actor, plan, task, options, adapter, route, Gate, connection, resources,
command, path, SSH material, status, approval, or artifact assertion. The
actor is resolved by the same trusted server principal boundary used for plan
authorization.

Before publication the service exact-reads and current-verifies the start
intent and its complete chain. The controller execution binds at least:

- project, run, start intent, authorization, permission decision, proposal,
  semantic plan, observation, catalog, and RunPlan IDs/digests;
- the complete RunPlan-order task roster and digest;
- complete per-task compiled-options, dispatch-intent, task-authority,
  artifact-input, Gate, budget, and policy digests plus their aggregate
  bindings;
- for every local slot, the non-empty authorization-time local adapter
  execution binding (adapter ID plus path- and interpreter-version-independent
  callable wrapper-chain source/defaults/closure implementation digest);
  remote slots must leave that local binding empty;
- the current complete remote AuthoritySet ID/digest when any task is remote;
- a deterministic attempt-zero task-slot roster;
- trusted actor identity/source and the request binding; and
- fixed `scientific-agent-harness-controller-policy.v1` identity.

Time, retry count, trace/span IDs, exporter state, and HTTP metadata are
operational fields and are excluded from semantic identity. A controller
execution is immutable after manifest-last publication. Current verification
re-derives all source material; stale authority fails closed without an
execution side effect.

## Deterministic action table

One `advance` call evaluates the first incomplete task in RunPlan order. A
single call selects exactly one action. It never loops across tasks or polls a
remote worker.

| Exact inspection | Selected action | Maximum bounded side effect |
| --- | --- | --- |
| All tasks have authoritative success evidence | `COMPLETE_EXECUTION` | Publish final controller completion receipt |
| Current task is local, needs a Gate, and no current snapshot exists | `PREPARE_LOCAL_GATE` | Executor prepares one task and writes one WAITING_USER snapshot |
| Current task is local, Gate snapshot is waiting, no exact committed decision exists | `WAIT_FOR_GATE` | No execution mutation; publish observation receipt |
| Current task is local, required exact Gate decision is rejected | `STOP_GATE_REJECTED` | Publish terminal controller receipt only |
| Current task is local and executable without a Gate, or with exact committed Gate approval | `EXECUTE_LOCAL_TASK` | Executor executes exactly that one task |
| Current local task was completed through an exact manual/legacy seam without this Controller decision's dispatch authority | `ADOPT_COMPLETED_TASK` | Exact-verify outputs and publish an explicit adoption publication/receipt; never claim Controller dispatch |
| Current task is remote and no slot request exists | `PREPARE_REMOTE_REQUEST` | Create one server-derived request in the exact task-attempt slot |
| Exact remote request exists and no approval exists | `WAIT_FOR_REMOTE_APPROVAL` | No dispatch; publish observation receipt |
| Exact remote approval rejects | `STOP_REMOTE_REJECTED` | Publish terminal controller receipt only |
| Exact approved request is prepared | `DISPATCH_REMOTE_TASK` | Dispatch the already approved exact request once |
| Exact remote job is mutable running/submitted | `REFRESH_REMOTE_TASK` | One transport status refresh; no polling loop |
| Slot requires recovery during ordinary `advance` | `RECOVER_REMOTE_TASK` with `executable=false` | No lifecycle mutation; return inspection/decision and publish a WAITING observation receipt |
| Slot is cancelled or terminal failure is authoritative | `STOP_TASK_TERMINAL` | Publish terminal controller receipt only |
| Slot has immutable publication, exact registered outputs, and success StageState | `ADOPT_REMOTE_OUTPUTS` | Register frozen logical output bindings if required by the plan |

Input drift, a stale source, an impossible combination, a wrong slot, a
mismatched receipt, or ambiguous completion selects a fail-closed terminal or
conflict result; it does not fall through to a different action.

## Local one-task Executor seam

The existing legacy `RunPlanExecutor.execute()` and
`resume_after_gate()` behavior remains compatible. PR-BN adds server-only
one-task methods that accept the exact planned index plus the Controller's
expected route, callable binding, task-authority digest, dispatch-intent
digest, compiled-options digest, artifact-input digest, and Gate snapshot or
decision binding. The seam:

1. re-reads the registered `AtomicTaskSpec` and resolves its current default
   callable;
2. rejects anything except the frozen `local_executor` dispatch intent;
3. re-derives the task snapshot from Registry inputs and authorized compiled
   options;
4. executes or prepares only the requested task index; and
5. returns only after StageState and required Registry outputs can be
   exact-read.

The Permission Engine and Controller use the same pure local-task authority
projection. Controller creation recomputes the current task-authority material,
requires it to equal the authorization and permission-decision task digests,
and freezes the separately projected callable implementation binding in the
task slot. Gate preparation, Gate decision consumption, local execution, and
local adoption all recompute that material and compare it with the slot. A
default-adapter change or same-ID callable implementation replacement therefore
makes the old Controller execution stale before the new callable can run,
including after an earlier task has changed StageState and Registry.

Local Controller slots require implementation-bound Permission v3 or v4.
Historical Permission v1/v2 authority remains exact-verifiable by its recorded
reader, but its legacy name/presence binding cannot create a new local
Controller execution. The implementation reader starts at the callable export
that Executor will invoke and binds every bounded `__wrapped__` layer; it does
not use `inspect.unwrap()` as a substitute for the executed wrapper.

Mixed plans are therefore never passed to the legacy whole-plan loop. A remote
dispatch intent cannot reach a legacy local adapter through the Controller.

At the actual adapter boundary, the Executor first publishes its established
dispatch receipt/authority when that evidence contract applies, then invokes a
Controller-only recorder before calling the adapter. The Controller exact-reads
the before/after dispatch roster, requires exactly one new matching
`execution_started=true` authority, and publishes a decision-bound local
dispatch receipt. For legacy tasks without that older specialized roster, the
same recorder itself is the immutable adapter-boundary authority. A successful
local publication then binds the StageState digest, complete Registry digest,
planned output roster, each output path/size/SHA-256/producer, verifier
identity, and any immutable execution-record ID/digest. Before publishing the
successful StageState, Controller-driven Executor runs also place an exact
output-content roster in that StageState. If the process dies after successful
StageState/Registry commit but before the completion callback, re-entry may
create exactly one `recovered_controller_dispatch` publication only after it
replays the matching Controller dispatch receipt, StageState output roster,
complete Registry contract, current output hashes, producer/task binding, and
the Executor's task-specific exact verifier. Immutable execution-record tasks
invoke their established publication replay; they are not accepted from a
verification-class label alone.

Publication modes are disjoint: `controller_dispatch` is emitted by the normal
completion callback, `recovered_controller_dispatch` reconstructs the missing
publication for the same committed Controller dispatch, and
`adopt_completed_task` represents completion outside that decision and cannot
claim its dispatch. StageState plus logical Registry IDs alone is insufficient.

Gate approval is deliberately a separate route and transaction. It commits an
existing `GateDecision` against the exact current WAITING_USER execution
snapshot. A later `advance` exact-reads that decision before the one-task
Executor consumes it. Plan authorization and controller creation never imply
Gate approval.

## Remote task-attempt slots

Each remote task attempt has the stable identity
`controller_execution_id + planned_task_index + task_id + attempt`. The slot
binds the complete remote request and its server-derived transfer manifest,
output contract, connection/profile, AuthoritySet/task authority, configured
resources, dispatch intent, options, and input artifacts. See
`task-scoped-remote-execution-slot-v1.md`.

Remote approval is also a separate strict positive-approval route. The client
supplies only the expected request digest, canonical client request ID, and an
optional bounded note. It cannot select a decision or dispatch. The server
supplies the trusted actor and re-verifies the exact request and all current
authority immediately before immutable approval. A later `advance` re-reads
the approval; it cannot accept approval words from plan prose, authorization
notes, Gate decisions, query parameters, headers, or mutable transport state.

## Completion and inspection

The inspection response labels every fact as one of:

- `AUTHORITATIVE`: immutable control artifact, exact current source,
  StageState transition, Registry binding, or verified remote publication;
- `DERIVED`: deterministic conclusion whose complete source digest roster is
  returned;
- `OBSERVATIONAL`: mutable transport status or safe tracing/latency data; or
- `UNVERIFIED`: missing, stale, conflicting, or otherwise unusable evidence.

Remote inspection exposes separate exact bindings for the request, task-slot
binding, approval, slot StageState, mutable transport state, and publication.
The effective remote status is `DERIVED` from a digest of that complete source
roster. Transport state is always `OBSERVATIONAL`; once success StageState and
publication are exact-verified it cannot override terminal authority.

A local task completes only when the exact Executor dispatch authority,
decision-bound local execution publication, and every planned output have
current content-bound verifier evidence. A manual completion is represented by
`TASK_ADOPTED`, never `TASK_COMPLETED`, and cannot claim a dispatch receipt. A
reconstructed local completion retains the original decision and dispatch,
publishes `recovered_controller_dispatch`, and produces a `RECONCILED`
Controller receipt without invoking the adapter again. A remote task completes
only when its slot has the exact committed request and approval,
immutable verified publication, complete output registrations, and matching
success StageState. Earlier controller receipts may locate this evidence but
cannot replace it. `details.executed_tasks`, adapter return values, job state,
and trace spans are observational corroboration only.

## Crash safety and exactly-once effects

All mutating operations for one Controller execution share one
`controller_execution.lock`. The fixed order is create-only start-intent scope
lock, Controller-execution lock, client-request lock, then local/remote
lifecycle lock. The execution lock is held from current verification through
immutable receipt publication, so different client request IDs and different
operations cannot select the same predecessor or overlap effects.
Publications use private staging, bounded strict JSON, regular-file/no-symlink
checks, per-file fsync, staging-directory fsync, no-replace rename,
collection-directory fsync, manifest-last activation, and exact-byte reread.

Each advance has a deterministic action ID derived from the execution,
inspection digest, selected action, task slot, and expected source digests.
The immutable decision is committed before the side effect. Immediately before
execution, the Controller rebuilds the inspection and requires both its digest
and complete source-binding roster to equal the decision. A stale decision may
only enter exact-authority reconciliation; it is never executed against new
state. Re-entry then:

- returns the exact receipt if it exists;
- reconciles the action's authoritative target if the process crashed after
  the side effect but before its receipt;
- completes the receipt without repeating the effect when reconciliation
  proves the exact result; or
- stops in explicit recovery/conflict state when exact reconciliation is not
possible.

Explicit cancel and recover routes use the same decision-before-effect and
receipt-after-exact-read protocol. Cancel is limited to the current exact
remote slot. Recover invokes one existing remote lifecycle recovery transition
and never reruns an unknown local scientific effect. Ordinary `advance` never
calls recovery; only the explicit recover route creates an executable recovery
control decision.

This is exactly-once *effect selection and reconciliation*, not a claim that
an arbitrary external transport is transactional. Remote idempotency remains
owned by the existing request/job protocol and task-attempt slot.

## Routes

```text
POST /api/projects/<project_id>/agent-plan-start-intents/<start_intent_id>/controller-executions
GET  /api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>
POST /api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/advance
POST /api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/gates/<gate_id>/approve
POST /api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/remote-approvals
POST /api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/cancel
POST /api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/recover
```

Every body is a strict generated schema with `additionalProperties=false`.
IDs in a route are not accepted again in the body. Read responses return the
immutable execution, ordered decisions/receipts, and a freshly derived
inspection; they do not expose private paths, hosts, credentials, commands,
environment data, raw stdout/stderr, or document/dataset/model contents.

## Compatibility and scope

Legacy whole-plan local execution and the legacy run-scoped
`remote-execution` lifecycle remain byte/path compatible. They do not gain
Controller authority. Controller task slots use the shared generalized remote
lifecycle implementation and never duplicate its request, approval,
dispatch, output-verification, Registry, or recovery logic.

PR-BN does not start PR-BO, an autonomous Execution Agent, a Replanner, or a
new execution UI. It does not change the worker protocol or the trajectory
schema. It provides an explicit server API handoff: a future PR-BO may call
`advance`, inspect typed authority labels, and request user action, but may not
bypass any route or synthesize a decision object. `M3H-GATE-002` remains open
and `M3H-008` remains in progress until repository-owner review.
