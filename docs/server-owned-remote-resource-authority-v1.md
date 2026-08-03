# Server-Owned Configured Remote Resource Authority Contract v1

Status: PR-BM2 control-plane contract for `M3H-007A`.

This contract closes the remote resource-authority prerequisite without
creating an execution request or dispatching work.

## Authority boundary

The server resource policy configures. The capability probe verifies
availability. PR-BM2 derives and binds. The user authorizes the exact plan.
PR-BN will dispatch later. PR-BM2 does not create a remote execution request.

| Object | Authority | Not authority for |
| --- | --- | --- |
| PR-BL proposal | Reviewed task, profile, route, options, and upper-bound intent | Configured GPU/CPU/walltime |
| Resource Authority Policy | Private owner-selected resources and budget ceilings | Availability or execution |
| Connection Profile | Private logical connection identity and declared capability | Configured resources or dispatch |
| Execution Profile | Fixed task/worker/environment/device contract and ceilings | A resource grant |
| Capability Probe | Current availability and verified capability facts for one exact connection digest | Resource selection |
| Per-task resource authority | Exact configured resources for one proposal task; inert without its complete set | User plan approval or dispatch |
| Resource AuthoritySet | Manifest-last activation of the complete current remote roster and aggregate budget | User plan approval or dispatch |
| Plan authorization | Exact user authority over the proposal and task-authority digest roster | GateDecision or execution |
| Start intent | Request for a future Controller action | RemoteExecutionRequest or running state |

All public PR-BM2 objects have `executable=false`. The authority outcome is
`CONFIGURED` or `DENY`; `CONFIGURED` never means `ALLOW` or dispatch permission.

## Private policy

`RemoteResourceAuthorityPolicyStore` stores
`molly_remote_resource_authority_policy.v1` at
`<MOLLY_CONFIG_DIR>/resource_authority_policies.json`. The directory is `0700`,
the file and process lock are `0600`, and reads/writes use dirfd traversal,
`O_NOFOLLOW`, regular-file checks, bounded strict JSON, atomic replacement,
file fsync, and directory fsync.

Each enabled entry binds a unique `policy_id`, `connection_id`, fixed
`execution_profile_id`, `remote_task_type`, explicit task allowlist, complete
integer `gpu_count`/`cpu_threads`/`walltime_sec`, and server budget limits.
`enabled` is a strict JSON boolean; coercible integers and strings are rejected.
Unknown keys are rejected. Hostnames, SSH aliases, remote roots, paths,
commands, argv, environment activation, credentials, tokens, and worker
overrides are not policy fields and never appear in public authority bytes.

There is intentionally no project API for changing this private owner policy.
The resource-authority creation API can submit only the exact proposal digest
and a canonical client request ID.

The repository README contains the reusable operator configuration pattern.
Configure one entry per exact Planner task and execution profile through
`RemoteResourceAuthorityPolicyStore`; do not hand-edit the private JSON. A
two-tool workflow such as Uni-Mol plus REINVENT4 therefore needs two entries.
This policy cannot convert a local Planner route into a remote route and cannot
repair a worker-protocol or provider/profile mismatch.

## Ceiling versus authority

`ExecutionProfile.resource_limits` remains a ceiling. It cannot select actual
resources and cannot prove that a resource is currently available. A
configured authority exists only when one unique enabled private policy entry,
one enabled exact connection, the fixed execution profile, and the last exact
available probe all agree.

The shared pure validator
`validate_requested_resources_against_execution_profile()` applies the same
ceiling and device rules used by `build_remote_execution_request()` without
building a request:

- `gpu_count <= gpu_count_max`;
- `cpu_threads <= cpu_threads_max`;
- `walltime_sec <= walltime_sec_max`;
- `cpu_only` requires zero GPU;
- `gpu_required` requires at least one GPU.

When the probe reports CPU threads, the configured count cannot exceed it. A
positive GPU request requires the `gpu` verified capability and an `available`
CUDA probe. Capabilities from different connections are never combined.

## Deterministic derivation

One canonical remote roster is projected from RunPlan dependency order, never
from the proposal's task-ID-sorted dispatch serialization. For every ordered
`remote_execution_service` task:

1. exact-read and current-verify the immutable PR-BL publication;
2. bind the ordered RunPlan roster and unique dispatch intent;
3. select exactly one enabled policy by task, type, and logical profile;
4. read the connection and all relevant probes under one profile-store process
   lock and recompute the PR-BL profile capability digest;
5. require an enabled connection, exact probe connection digest, `available`
   status, and all profile-required capabilities;
6. apply the shared profile ceiling/device validator and probe limits;
7. treat proposal resource values only as reviewed constraints, never defaults;
8. validate per-task runtime and derived GPU hours against private policy
   ceilings;
9. derive immutable authority bytes, digest, and ID.

After every per-task authority is derived, v1 constructs a plan-level
aggregate budget in the same RunPlan order:

```text
total_derived_gpu_hours = sum(per-task derived_gpu_hours)
total_walltime_upper_bound_sec = sum(per-task walltime_sec)
walltime_aggregation_policy = sequential_sum.v1
```

The conservative sequential walltime and total GPU hours are compared once
against proposal-level `max_runtime_sec` and `max_gpu_hours`. Individual tasks
cannot each reuse the whole plan limit. `total_configured_cpu_threads` is
bound as an informational aggregate and is not treated as concurrent CPU
allocation.

`derived_gpu_hours = gpu_count * walltime_sec / 3600`. A non-null proposal or
policy `max_cost_usd` is denied because v1 has no versioned server cost model;
cost is never assumed to be zero. For Permission v2, `max_gpu_hours` is owned
by the current exact AuthoritySet. The set always validates the remote
`max_runtime_sec` subtotal. If a mixed plan also contains a `local_executor`
task whose Registry contract declares `max_runtime_sec`, that same plan-level
limit must additionally be covered by configured legacy server budget
authority; otherwise Permission denies with
`MIXED_PLAN_RUNTIME_AUTHORITY_REQUIRED`. This prevents AuthoritySet from
silently turning a global runtime limit into a remote-only limit. Other local
dimensions such as `max_steps` and `max_records` retain the existing PR-BM
budget semantics.

The compiler-generated `remote_resources_<task_id>` question is satisfied only
for Permission policy v2 by a current exact authority for that same task. No
question or proposal byte is deleted or rewritten. Every other blocking
question and every missing artifact remains a denial.

## Exact identity and source binding

One `agent_remote_resource_authority.v1` object covers one remote RunPlan task.
Its semantic digest excludes only `created_at` and covers:

- project/run/proposal, semantic plan, observation, catalog, and RunPlan
  identities and digests;
- the complete ordered task roster and roster digest;
- task ID and dispatch-intent digest;
- remote task type and logical profile;
- logical connection ID and exact connection profile digest;
- execution profile ID/digest;
- capability probe digest/status and verified capability roster;
- complete configured resources;
- budget limits, budget digest, and derived GPU hours;
- selected resource-policy ID/digest and complete policy version/digest;
- sorted source-binding roster.

The ID is derived from the authority digest. The publication never includes
host, path, command, credential, request/job ID, or runtime status.

Per-task files are audit material until one
`agent_remote_resource_authority_set.v1` manifest exact-binds the decision,
RunPlan-order remote roster, every authority ID/digest, the complete roster
digest, every per-task budget binding, aggregate totals, aggregation policy,
and aggregate budget digest. The set ID is derived from its semantic digest;
`created_at` is excluded and `executable=false` is fixed. A stale set remains
an immutable audit but is not current authority.

## Permission, authorization, and start-intent integration

The frozen PR-BM v1 and PR-BM2 v2 permission materials and digests remain
unchanged for exact replay. Persisted local-only v1 and resource-aware v2
decisions continue to regenerate with their recorded readers. New remote
authorizations use `scientific-agent-permission-policy.v4`, which preserves
the v2 remote AuthoritySet and budget semantics while selecting the
implementation-bound local callable algorithm introduced by policy v3.

For a remote task:

```text
no complete current published AuthoritySet
  -> DENY / REMOTE_RESOURCE_AUTHORITY_REQUIRED

stale or mismatched authority
  -> DENY with its stable resource-authority reason

current exact authority in a complete current set
  -> execution_binding_digest = digest(route + type + profile + authority digest + set digest)
  -> proposal review = REQUIRE_APPROVAL
```

The existing chain then binds the resource facts without changing
`agent_plan_authorization.v1` or `agent_plan_start_intent.v1`:

```text
resource authority digest + AuthoritySet digest
-> remote execution_binding_digest
-> task_authority_digest
-> permission decision digest
-> authorization.task_authority_digests
-> authorization digest
-> start-intent current verifier
```

Resource-aware evaluations use `agent-task-authority-binding.v2` for every
planner-visible and planner-hidden task. In addition to the frozen v1 task
material, v2 exact-binds the Registry task's canonical, sorted, unique
`budget_dimensions` roster. A hidden dependency must explicitly declare that
field, including an explicit empty roster when it owns no budget dimension;
an omitted field or an unrecognized dimension is denied. Consequently a
change such as `max_runtime_sec -> none`, or the reverse, changes the task
authority digest and makes the old authorization and start intent stale.
Planner visibility does not create two different budget-identity rules.

The local-only policy continues to use the byte-identical
`agent-task-authority-binding.v1`; its policy, decision, authorization, and
start-intent replay identities are not migrated.

Authorization and approve-and-start payloads remain unchanged and cannot name
an authority, connection, profile override, or resource value. Current
verification re-reads the proposal, private policy, connection, execution
profile, probe, capability digest, resources, budget, and full task roster.
Any drift makes the old authorization and start intent stale.

Persisted v1 local decisions are regenerated with v1. Persisted v2 remote
decisions are regenerated with v2. New local and resource-aware decisions use
v3 and v4 respectively. Existing proposal, authorization, and start-intent
bytes are never migrated or rewritten.

The reviewed v2 policy semantic digest for this contract is
`sha256:e5279fe137409cf3490beac8b29c32c3c3212537f67e924fdd875aebe4d6d124`;
it covers complete-set execution binding, mixed/aggregate remote budget
ownership, and the v2 exact task budget-dimension roster. The frozen
local-only v1 digest remains unchanged.

The implementation-bound resource-aware v4 policy digest is
`sha256:f7793b493ba2d28194df21e8993651031d40c5f2c3edcca3d8dc8db39f7f027f`.
It changes no remote resource-authority or AuthoritySet schema; its additional
identity is limited to the version-selected complete local callable wrapper
chain used by mixed plans.

## API and durable publication

The explicit project-scoped endpoints are:

```text
POST remote-resource-authority-evaluations
POST remote-resource-authorities
GET  agent-remote-resource-authority-decisions/<decision_id>
GET  agent-remote-resource-authorities/<authority_id>
GET  agent-remote-resource-authority-sets/<authority_set_id>
```

The request schema permits only `expected_proposal_digest` and
`client_request_id`. Responses expose reviewable decision/authority objects,
`executable=false`, and `dispatched=false`; they never return a remote request,
job, host, SSH locator, absolute path, or command. A local-only evaluation has
an empty authority roster and `REMOTE_RESOURCE_AUTHORITY_NOT_REQUIRED`.

Project control publication uses private staging, manifest-last file order,
per-file fsync, staging-directory fsync, no-replace atomic rename, collection
fsync, and exact-byte reread. Per-request process locking freezes:

```text
RESERVED
-> DECISION_COMMITTED
-> AUTHORITIES_COMMITTED
```

Authorities for multiple remote tasks are published in deterministic RunPlan
order. Bare per-task publications are inert. After the full roster is
current-verified, the no-replace AuthoritySet is published manifest-last and
then current-verified again. The mutation/fault opportunity precedes the last
source re-derive; only after that barrier does the request success marker bind
the set ID/digest, complete roster digest, and aggregate budget digest.
Same request/same bytes recovers idempotently; same request/different bytes is
a conflict. A crash after any per-task rename but before AuthoritySet
publication cannot be consumed by Permission. Source drift after set rename
may leave a stale immutable audit set, but cannot write the request success
marker or return success. Re-entry derives the same IDs and completes exactly
one roster.

## Threat model and privacy

The contract fails closed against client/LLM resource injection, profile
ceiling inference, missing/disabled/ambiguous policy, policy corruption or
symlink substitution, connection/probe/profile drift, cross-connection
capability joining, incomplete or incorrectly typed resources, budget/cost
omission, stale sources, request replay conflicts, path escape, and partial
publication.

Connection IDs may be retained as server-managed logical IDs. Private host,
username, SSH, known-hosts path, remote root, environment command, credential,
and probe hostname fields are excluded from authority semantics and public
responses. Absolute paths are not persisted in semantic material.

## Non-goals and PR-BN handoff

PR-BM2 does not call an Executor, `RemoteExecutionService`, worker queue,
transport, SSH/SCP/subprocess, adapter, Gate writer, or StageState writer. It
does not create `RemoteExecutionRequest`, transfer manifest, queue job,
execution record, GateDecision, or running status. It implements no Controller,
Execution Agent, Replanner, UI, cancellation, retry, or dispatch.

PR-BN may consume only after re-verification: AuthoritySet ID/digest and
complete roster/aggregate budget digests, per-task authority ID/digest, task
execution-binding digest, configured resources, connection/profile/probe
digests, budget and policy digests, full task roster, and the exact
proposal/authorization/start-intent relationship. Time-based probe expiry and
a versioned monetary cost model remain future prerequisites if those policies
are required.
