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
| Resource authority | Exact configured resources for one proposal task | User plan approval or dispatch |
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
Unknown keys are rejected. Hostnames, SSH aliases, remote roots, paths,
commands, argv, environment activation, credentials, tokens, and worker
overrides are not policy fields and never appear in public authority bytes.

There is intentionally no project API for changing this private owner policy.
The resource-authority creation API can submit only the exact proposal digest
and a canonical client request ID.

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

For every ordered `remote_execution_service` dispatch intent:

1. exact-read and current-verify the immutable PR-BL publication;
2. bind the ordered RunPlan roster and unique dispatch intent;
3. select exactly one enabled policy by task, type, and logical profile;
4. read the connection and all relevant probes under one profile-store process
   lock and recompute the PR-BL profile capability digest;
5. require an enabled connection, exact probe connection digest, `available`
   status, and all profile-required capabilities;
6. apply the shared profile ceiling/device validator and probe limits;
7. treat proposal resource values only as reviewed constraints, never defaults;
8. validate runtime and derived GPU hours against both proposal and private
   budget ceilings;
9. derive immutable authority bytes, digest, and ID.

`derived_gpu_hours = gpu_count * walltime_sec / 3600`. A non-null proposal or
policy `max_cost_usd` is denied because v1 has no versioned server cost model;
cost is never assumed to be zero. `max_steps` and `max_records` retain the
existing PR-BM budget semantics.

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

## Permission, authorization, and start-intent integration

The frozen PR-BM v1 permission material and digest remain unchanged for exact
replay. Local-only proposals continue to produce byte-identical policy v1
decisions. New remote evaluations use
`scientific-agent-permission-policy.v2`.

For a remote task:

```text
no current published authority
  -> DENY / REMOTE_RESOURCE_AUTHORITY_REQUIRED

stale or mismatched authority
  -> DENY with its stable resource-authority reason

current exact authority
  -> execution_binding_digest = digest(route + type + profile + authority digest)
  -> proposal review = REQUIRE_APPROVAL
```

The existing chain then binds the resource facts without changing
`agent_plan_authorization.v1` or `agent_plan_start_intent.v1`:

```text
resource authority digest
-> remote execution_binding_digest
-> task_authority_digest
-> permission decision digest
-> authorization.task_authority_digests
-> authorization digest
-> start-intent current verifier
```

Authorization and approve-and-start payloads remain unchanged and cannot name
an authority, connection, profile override, or resource value. Current
verification re-reads the proposal, private policy, connection, execution
profile, probe, capability digest, resources, budget, and full task roster.
Any drift makes the old authorization and start intent stale.

Persisted v1 local decisions are regenerated with v1. Persisted v2 remote
decisions are regenerated with v2. Existing proposal, authorization, and start
intent bytes are never migrated or rewritten.

## API and durable publication

The explicit project-scoped endpoints are:

```text
POST remote-resource-authority-evaluations
POST remote-resource-authorities
GET  agent-remote-resource-authority-decisions/<decision_id>
GET  agent-remote-resource-authorities/<authority_id>
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
order. Only the final marker binds the complete authority ID/digest roster.
Same request/same bytes recovers idempotently; same request/different bytes is
a conflict. A crash or source drift may leave an immutable audit publication,
but cannot write the success marker or return success. Re-entry derives the
same IDs and completes exactly one roster.

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

PR-BN may consume only after re-verification: authority ID/digest, task
execution-binding digest, configured resources, connection/profile/probe
digests, budget and policy digests, full task roster, and the exact
proposal/authorization/start-intent relationship. Time-based probe expiry and
a versioned monetary cost model remain future prerequisites if those policies
are required.
