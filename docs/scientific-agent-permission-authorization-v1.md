# Scientific Agent Permission and Authorization Contract v1

Status: PR-BM control-plane contract for `M3H-004` through `M3H-007`.

This contract extends the merged PR-BL review-only planning layer with three
new and deliberately separate objects:

```text
verified PR-BL proposal
  -> deterministic permission decision
  -> explicit immutable user authorization
  -> immutable start intent
  -> future PR-BN Harness Controller
```

PR-BM stops at the start intent. It does not dispatch, execute, resume, cancel,
create a remote request, consume a Gate, or write execution state.

## Existing PR-BL binding

The input is not client JSON. The server calls
`ScientificAgentPlanProposalStore.read(project_id, proposal_id,
verify_current=True)` and re-verifies the merged PR-BL publication. That read
binds and checks:

- the immutable proposal publication ID and full proposal digest;
- semantic plan ID and digest, separately from invocation and request identity;
- project and run identity;
- the complete privacy-safe observation and observation digest;
- all observation source bindings: StageState summary, Artifact Registry,
  existing RunPlan, confirmed dataset manifests, tool catalog, and resource
  profile snapshot;
- the deterministic tool catalog and catalog digest;
- the validated planning LLM response;
- the canonical dependency-expanded `RunPlan`, including its complete ordered
  task roster, dependencies, inputs, outputs, available artifacts, and missing
  artifacts;
- LLM-facing planner options;
- server-materialized effective planner options for every expanded task;
- server-compiled task options for every expanded task;
- the per-task logical dispatch intents, including local/remote route, logical
  remote task type, logical profile, and nullable resource intent;
- selected artifacts and profiles;
- limits, stop conditions, success criteria, questions, and required Gates;
- immutable publication bytes, source binding, verification file, and
  manifest; and
- current artifact bytes/trust, profile capabilities, catalog, StageState,
  Artifact Registry, and existing RunPlan, so a stale source fails closed.

The proposal remains `status=review_required` and `executable=false`. External
LLM consent only permits the dedicated planning data transfer and is not part
of any authorization material.

## Authority matrix

| Object | Decides or records | Exact scope | Can dispatch or mutate execution? |
| --- | --- | --- | --- |
| PR-BL proposal | Server-validated plan proposal | Observation, catalog, canonical RunPlan, options, dispatch intents | No |
| `ServerPermissionStore` grant | Broad route/action access | Project or run plus an action name | No exact plan authority |
| Permission decision | Deterministic policy evaluation | Exact proposal/source plus policy identity and phase | No |
| Plan authorization | Explicit user authority | Exact proposal, plan semantics, all task parameters and source bindings | No |
| Start intent | User request for future Controller action | Exact authorization plus authorized-start decision | No; `not_dispatched` only |
| `GateDecision` | Domain/risk Gate approval | One existing Gate and execution snapshot | Only the existing resume path consumes it |
| `ExecutionConfirmation` | Audit of confirmed snapshot execution | One task/adapter snapshot after resume validation | Not plan authorization |
| Execution snapshot | Executor replay/TOCTOU authority | Exact task payload, artifacts, Gate set, and options | Consumed only by Executor |
| Future Controller | Dispatch decision | Valid authorization and start intent plus current authority | Out of PR-BM |

`ServerPermissionStore` is mutable broad route/action policy. A grant such as
`upload_dataset` or `scientific_agent_plan_authorize` does not bind a proposal
digest, RunPlan, task roster, options, artifact lineage, profile capabilities,
budget, Gate roster, actor request, or start slot. It may later be used to
control entry to an authorization route, but it can never substitute for the
exact immutable `AgentPlanAuthorization`.

`GateDecision`, `ExecutionConfirmation`, plan authorization, and start intent
are four independent authorities:

- `GateDecision` approves a current execution Gate and snapshot through the
  existing Executor resume path.
- `ExecutionConfirmation` is a separate append-only audit written after that
  path validates and resumes an exact snapshot.
- plan authorization records the user's exact plan-level decision before any
  dispatch exists.
- start intent records only that the user requests a future Controller to
  consider starting the exact authorization.

None can be inferred from another. Plan authorization does not create or
consume a Gate, and start intent does not claim execution occurred.

## Permission Engine

The decision schema is `agent_permission_decision.v1`. Outcomes are:

```text
DENY > REQUIRE_APPROVAL > ALLOW
```

The precedence is global and deterministic. Any task-level or proposal-level
`DENY` makes the overall result `DENY`. The phases are:

- `proposal_review`: a complete current proposal returns
  `REQUIRE_APPROVAL`; invalid, stale, incomplete, or hostile input returns
  `DENY`.
- `authorization_candidate`: validates actor, mode, requested Gate
  preauthorization, and all current proposal bindings. A valid candidate still
  returns `REQUIRE_APPROVAL` because the explicit user commit has not yet been
  made.
- `authorized_start`: only an exact re-verified authorization can return
  `ALLOW`. `ALLOW` permits start-intent creation only.
- `shadow_comparison`: computes the new proposal-review result for an
  independent audit comparison.

### Policy identity

The fixed policy semantic material is versioned as:

```text
scientific-agent-permission-policy.v1
```

At this PR HEAD its canonical digest is:

```text
sha256:08f76cffc76dd1c0ba58cfcc401198c12106ef19abd0b29e75129ca5b510dd7c
```

The digest covers recognized effect classes, permissions, local/remote routes,
logical remote task types, risk rules, semantic/operational Gate rules, budget
dimensions and upper-bound rules, artifact trust rules, profile/resource
completeness, the explicit hidden-internal-dependency contract, recognized
idempotency/verifier policies, callable local execution binding rules, task
authority digest versions, authorization modes, outcome precedence, and the
reason-code vocabulary. Dictionary order and `PYTHONHASHSEED` do not affect
canonical bytes. A semantic rule change must change this digest and normally
upgrades the policy version.

### Decision rules

Representative `REQUIRE_APPROVAL` reasons are:

- `PLAN_AUTHORIZATION_REQUIRED`;
- `HIGH_RISK_TASK_REQUIRES_USER`;
- `REMOTE_COMPUTE_REQUIRES_USER`;
- `OPERATIONAL_GATE_REQUIRES_USER`; and
- `SEMANTIC_GATE_REMAINS_PENDING`.

Representative fail-closed `DENY` cases are:

- failed/no-replace/stale proposal verification or expected digest mismatch;
- proposal `executable` other than false;
- duplicate/unknown tasks or incomplete task/options/dispatch/Gate coverage;
- blocking questions or missing artifacts;
- unknown effect, risk, permission, Gate, route, or remote task type;
- a planner-hidden dependency without an explicit, fixed local Registry
  permission contract;
- a hidden local dependency without a non-empty callable registered default
  adapter binding;
- an unrecognized hidden-task idempotency or verification policy;
- changed selected artifact content, trust, or producer lineage;
- changed/missing profile availability or capability digest;
- incomplete remote profile or resource intent;
- non-empty limits without budget authority or a limit exceeding authority;
- missing actor or invalid authorization mode;
- actor identity supplied by an untrusted client-controlled source;
- stepwise Gate preauthorization;
- frozen-plan Gate outside the plan, a semantic Gate, or a task whose
  `supports_plan_preapproval` is false;
- client authority/task/option/adapter/command/path/SSH/status/GateDecision
  injection;
- non-literal `confirmed=true`; and
- one client request ID bound to different bytes.

## Exact immutable plan authorization

The request schema is `agent_plan_authorization_request.v1`, and the persisted
authority is `agent_plan_authorization.v1`. The project and proposal IDs come
from the route. The only body fields accepted by authorization and
approve-and-start are:

```text
expected_proposal_digest
authorization_mode
requested_preauthorized_gate_ids
confirmed
client_request_id
note
```

Unknown fields are rejected. In particular the client cannot supply RunPlan,
task/options, artifacts, profiles, permission/Gate policy, adapter, command,
path, SSH, status, approval, actor, authorization digest, or start-intent
contents. `confirmed` must be the literal JSON boolean `true`. Actor and actor
source are obtained from a PR-BM-specific authenticated resolver. It accepts an
authenticated middleware principal in
`flask.g.ai4s_authenticated_principal`, a private server-populated WSGI
principal in `ai4s.authenticated_principal`, or the fixed
`AI4S_AGENT_AUTHORIZATION_OWNER` used by a local single-user deployment. The
resolver ignores `X-Actor` and all body/query/form actor assertions. With no
trusted principal configured, authorization is unavailable by default.

The authorization binds:

- project ID and run ID;
- proposal publication ID and digest;
- semantic plan ID and digest;
- observation ID and digest;
- tool catalog digest;
- a digest of the complete canonical RunPlan plus the full RunPlan object;
- the complete ordered task roster;
- a complete task-keyed authority-digest roster, where each digest binds the
  task's permission policy, fixed caller option contract, and server-only
  execution binding digest;
- task-keyed effective planner options;
- task-keyed compiled task options;
- every per-task dispatch intent;
- selected artifact IDs, content digests, trust classes, verification states,
  and producer task bindings;
- selected logical profile IDs/types, capability digests, availability,
  verified capabilities, and logical task types;
- limits, stop conditions, and success criteria;
- complete required Gate roster and task/effect/Gate-class/preapproval
  bindings;
- preauthorized operational Gates and still-pending Gates;
- permission policy version and digest;
- authorization-candidate decision ID and digest;
- mode, actor, actor source, note, and client request identity; and
- `executable=false`.

`created_at` is preserved in the exact immutable publication but is excluded
from the semantic authorization digest. Actor, mode, request identity, exact
scope, and all proposal bindings participate. The authorization ID derives
from the semantic digest. The authorization is not registered as a scientific
result, GateDecision, execution confirmation, execution record, or Artifact
Registry item, and it has no mutable status field.

Any exact field drift causes re-verification failure. A requested task/option,
artifact, profile, backend, budget, success criterion, route, resource, or Gate
change requires a new PR-BL proposal; authorization never patches a proposal.

PR-BL may expand a planner-visible task through a non-planner-visible internal
dependency. Such a dependency is not exposed to the LLM and still has fixed
empty effective/compiled caller options. PR-BM accepts it only when its
`AtomicTaskRegistry` entry explicitly declares risk, effect, permissions,
Gates, fixed `local_executor` routing, empty option/default maps,
preauthorization capability, idempotency policy, verification policy, and
`planner_visible=false`. Its `default_adapter` must also be explicitly set and
resolve to a callable export accepted by the existing Executor. Resolution is
read-only and never invokes the adapter. Missing permission metadata fails as
`INTERNAL_TASK_PERMISSION_METADATA_INCOMPLETE`; an unknown policy fails as
`INTERNAL_TASK_POLICY_UNRECOGNIZED`; and a missing/unknown adapter fails as
`INTERNAL_TASK_EXECUTION_BINDING_INCOMPLETE`.

Each per-task decision stores two server-derived digests:

- `execution_binding_digest` binds the server-only default-adapter identity for
  local execution, or the logical route/type for non-local execution; and
- `task_authority_digest` binds task ID, visibility, effect, risk, permissions,
  Gates, route/type, preapproval capability, exact idempotency and verification
  policy values, fixed/effective/compiled option contract, and the execution
  binding digest.

Adapter names remain absent from proposal, authorization request, and LLM
material. The authorization persists the exact task-to-authority-digest map;
its digest and the start-intent verifier transitively bind every value. A
non-empty-to-non-empty policy or registered-adapter change therefore changes
the decision and authorization authority rather than passing completeness
unchanged.

## Stepwise, frozen plan, and Gates

`stepwise` authorizes the exact complete plan but preauthorizes no Gate. Every
Gate remains in `pending_gates` and continues through the existing
GateDecision/Executor snapshot authority.

`frozen_plan` also authorizes only the exact complete plan. It may preauthorize
a Gate only if the Gate belongs to the current RunPlan, its task declares
`supports_plan_preapproval=true`, the effect is not semantic, the policy marks
it operational, and proposal/artifact/profile/resource/budget bindings still
match exactly.

A required Gate is a unique roster entry but may have multiple task bindings.
PR-BM preserves every binding. If any task binding has a semantic effect, the
shared Gate is semantic for plan-preauthorization purposes. A shared Gate is
preauthorizable only when every binding is operational and every bound task
sets `supports_plan_preapproval=true`; sharing a Gate is not a catalog
conflict.

At the merged PR-BL catalog used by this PR, every planner-visible task has
`supports_plan_preapproval=false`. The current legal frozen-plan
`preauthorized_operational_gates` roster is therefore empty. PR-BM does not
change any task merely to demonstrate preauthorization.

Gates on `scientific_confirm`, `change_objective`, and `publish_or_promote`
effects are always semantic and always pending. Dataset confirmation, target
or scientific constraint changes, ranking objective changes, new sources,
retry, task graph changes, profile/resource/budget expansion, final promotion,
and experiment batches require their existing user Gate and/or a new exact
authorization. External LLM consent is never copied into a Gate roster.

Authorization does not create or update `GateDecision`, write approved Gates
to StageState, consume a Gate, or call the resume route.

## Approve-and-start durability

One HTTP operation performs two visibly distinct immutable commits:

```text
validate literal confirmation and resolve trusted server principal
  -> reserve and lock exact client request
  -> exact-read and verify current proposal
  -> evaluate authorization candidate
  -> checkpoint exact authorization bytes
  -> exact-read and compare current source binding again
  -> publish and fsync authorization
  -> verify authorization and current proposal
  -> write AUTHORIZATION_COMMITTED marker only after final verification
  -> evaluate authorized-start and verify proposal start slot
  -> checkpoint exact start-intent bytes
  -> reverify authorization and current proposal immediately before commit
  -> publish and fsync start intent
  -> verify start intent, authorization, and current proposal
  -> write START_INTENT_COMMITTED marker only after final verification
  -> return dispatched=false
```

The request states are:

```text
RESERVED
AUTHORIZATION_COMMITTED
START_INTENT_COMMITTED
```

Each request has a cross-process advisory lock and an immutable digest over
project, proposal, operation, request body, actor, and actor source. Request
markers and checkpoints use exclusive `O_NOFOLLOW` creation and per-file and
directory fsync. Publications use a request-private staging directory,
data/verification files, manifest-last, staging-directory fsync, atomic rename,
collection-directory fsync, and exact byte re-verification. Publication IDs are
no-replace identities.

Current-source checks are deliberately consumed by the creation path, not left
only to later GET requests. Fault injection covers source drift after the
initial read, after candidate evaluation, after authorization staging, after
authorization publication, between authorization and start intent, and after
start-intent rename but before the response. A publication may remain as an
immutable audit object if the source changes after its atomic rename, but the
request receives no final success marker and the API does not return success.

Recovery uses the immutable checkpoints:

- before authorization commit: rebuild or reuse the checkpoint and commit once;
- after authorization publication but before its marker: verify the existing
  exact publication, then complete the marker;
- during start-intent staging: ignore incomplete private staging, reuse the
  checkpoint, and atomically publish one target;
- after start publication but before its marker: verify the one start intent
  and complete the marker;
- a new process with the same request replays exact bytes;
- a different payload, actor, operation, or proposal under the same request ID
  conflicts before new authority is created; and
- the proposal has one deterministic start-intent slot, so a different request
  cannot create a second start intent.

The response is explicit:

```json
{
  "authorized": true,
  "start_intent_created": true,
  "dispatched": false
}
```

It never returns `started`, `running`, `job_id`, `remote_job_id`, or
`execution_started`.

## Start intent is not dispatch

The schema is `agent_plan_start_intent.v1`. Fixed values are:

```text
intent_type = start_authorized_plan
handoff_target = scientific_agent_harness_controller.v1
dispatch_state = not_dispatched
executable = false
```

It binds the exact authorization and the distinct `authorized_start`
permission decision, plus actor, mode, proposal, and client request. It has no
adapter, command, queue job, remote job, execution snapshot, Gate decision, or
execution status. Only the future PR-BN Controller may reverify and consume it.

## Current remote resource-authority limitation

The merged PR-BL projection currently emits nullable `gpu_count` and
`cpu_threads` and a resource status of `partial` or `not_configured` for its
remote Uni-Mol, REINVENT4, and MinerU routes. PR-BM requires a server-owned
remote profile plus `requested_resources.status=configured`; it does not infer
authority from profile ceilings or fill nullable fields with defaults.

Consequently, PR-BM v1 has a successful authorization path only for complete
local plans. Current Uni-Mol, REINVENT4, and MinerU proposals fail closed with
`REMOTE_RESOURCE_INTENT_INCOMPLETE`. Remote Controller integration is blocked
until a separate server-owned resource-authority contract can produce and
verify configured resource bindings. That prerequisite is recorded for the
PR-BN handoff; PR-BM does not create it and does not relax the deny rule.

## Shadow mode

The explicit shadow endpoint creates `agent_permission_shadow_record.v1` and
does not intercept an existing route. It compares the deterministic
proposal-review result with a server-derived legacy expectation:

- local plans without Gates map to the current synchronous execute
  expectation;
- local plans with Gates map to the current pause/resume Gate expectation;
- all-remote logical plans map to the separate current remote approval path;
  and
- mixed legacy routes are `INCOMPARABLE` rather than inventing one boolean.

Alignment is `MATCH`, `NEW_STRICTER`, `NEW_LOOSER`, or `INCOMPARABLE`. The
client cannot submit a legacy outcome. The record is a separate no-replace,
`executable=false` audit object and creates neither authorization nor start
intent.

Automatic observation of existing execute/resume routes is not installed;
`AI4S_ENABLE_AGENT_PERMISSION_SHADOW_OBSERVATION` is documented and remains
unset/disabled by default. Therefore shadow on/off and evaluator failure cannot
change legacy HTTP responses, scientific artifact bytes, Gate state, StageState,
or queue state.

## Storage

```text
projects/<project_id>/agent_plan_control/
  permission_decisions/<decision_id>/
  authorizations/<authorization_id>/
  start_intents/<start_intent_id>/
  shadow_records/<shadow_record_id>/
  requests/<client_request_id>/
```

Every publication directory contains one typed JSON object,
`verification.json`, and `publication_manifest.json`. Identifiers are canonical
single path components. Project, collection, request, target, staging, and file
objects reject symlinks and scope escape. Stable canonical JSON and semantic
digests contain no absolute local path.

## API

The additive project-scoped endpoints are:

```text
POST /api/projects/<project_id>/agent-plan-proposals/<proposal_id>/permission-evaluations
GET  /api/projects/<project_id>/agent-permission-decisions/<decision_id>

POST /api/projects/<project_id>/agent-plan-proposals/<proposal_id>/authorizations
GET  /api/projects/<project_id>/agent-plan-authorizations/<authorization_id>

POST /api/projects/<project_id>/agent-plan-proposals/<proposal_id>/approve-and-start
GET  /api/projects/<project_id>/agent-plan-start-intents/<start_intent_id>

POST /api/projects/<project_id>/agent-plan-proposals/<proposal_id>/permission-shadow-evaluations
GET  /api/projects/<project_id>/agent-permission-shadow/<shadow_record_id>
```

The permission and shadow evaluation bodies accept only the expected proposal
digest. The server reloads the proposal rather than accepting uploaded proposal
JSON.

## Threat model and privacy

The contract fails closed for stale/replaced proposal or source bytes,
symlink/path escape, partial publications, request-ID reuse, concurrent
duplicate requests, source changes between authorization and start intent,
unknown catalog/policy terms, incomplete remote authority, budget expansion,
artifact/profile drift, hidden adapter/policy drift, missing/replaced task
authority digests, and all client authority injection.

An untrusted client actor assertion and an authenticated server principal are
different inputs. Raw `X-Actor`, body/query/form actor aliases, and broad
project grants cannot create authorization. Middleware or trusted proxy
deployments must write the private server principal after authenticating the
request; production must not translate an arbitrary client header into that
principal. Local single-user mode may instead configure one fixed owner.

Ordinary chat text such as `approved` or `继续`, assistant prose, LLM output
approval fields, `external_llm_approved=true`, proposal
`status=review_required`, legacy client flags, broad grants, GateDecision,
ExecutionConfirmation, ResumeIntent, and start-intent-like JSON cannot create an
authorization.

Persisted semantic material contains logical IDs, safe actor projection,
digests, typed high-level options, and privacy-safe PR-BL observation bindings.
It does not copy raw data/document content, absolute paths, host/IP, SSH/SCP,
known-hosts, username/email, command/argv, environment, credentials,
stdout/stderr, provider raw response, or hidden reasoning. Schema safe-value
validation and repository privacy tests remain in force.

## Non-goals

PR-BM does not:

- call `RunPlanExecutor.execute()` or `resume_after_gate()`;
- call remote `prepare`, `approve`, `refresh`, `recover`, or `cancel`;
- call an adapter, SSH/SCP, `molly-worker`, or a worker queue;
- create a remote execution request or queue job;
- create/consume a GateDecision or ExecutionConfirmation;
- write StageState or any `RUNNING`, `WAITING_USER`, `SUCCEEDED`, `FAILED`, or
  `CANCELLED` execution claim;
- implement the Harness Controller, Execution Agent, or Replanner;
- change the main UI or replace `/api/run-plan/execute` or
  `/api/run-plan/resume`; or
- claim that a plan has started or a scientific task is running.

## PR-BN handoff

PR-BN may reuse, but must reverify immediately before any Controller action:

- deterministic permission-decision verifier and policy version/digest;
- exact authorization verifier and authorization digest;
- exact canonical RunPlan and ordered task roster;
- per-task execution-binding and task-authority digests;
- effective and compiled options;
- dispatch-intent binding;
- artifact digest/trust/producer bindings;
- profile capability bindings;
- resource and budget bindings;
- required, pending, and preauthorized Gate rosters;
- start-intent verifier and one-per-proposal start slot;
- request lock/checkpoint/crash recovery; and
- independent shadow mismatch records.

PR-BN remains responsible for Controller dispatch into the existing
RunPlanExecutor/RemoteExecutionService and for preserving Verifier, Artifact
Registry, GateDecision, execution snapshot, worker, and StageState authority.
Before PR-BN can dispatch a remote start intent, a server-owned resource
authority must replace the current `partial`/`not_configured` remote projection
with an exact `configured` binding. This is a prerequisite, not authority
granted by the current start-intent schema.
