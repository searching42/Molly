# Scientific Agent Harness: plan proposal contract v1

Status: PR-BL, read-only planning and review artifact layer.

This contract is the first (planning) segment of the M3.5 authority chain:

```text
authoritative project state
  -> AgentProjectObservation
  -> dedicated JSON planning call
  -> strict validation
  -> AtomicTaskRegistry dependency expansion
  -> immutable review-only AgentExecutionPlanProposal
```

`LLM proposes. The server validates and compiles. No user authorization exists
in PR-BL. No task is executed. Executor/verifier authority is unchanged.`

## Current architecture and the boundary of this PR

`AtomicTaskRegistry` in `src/ai4s_agent/planner.py` is the current executable
task fact source. Each `AtomicTaskSpec` defines the canonical task ID,
required and output artifact contracts, risk level, required gates, adapter
binding, and dependency hints. `expand_run_plan()` resolves the registered
dependency graph and computes missing artifacts. The new
`ScientificToolCatalog` is a deterministic, privacy-safe projection of these
registered specs; it is not a second task registry and contains no adapter
callable or adapter name.

The existing `AgentToolRegistry` and action handoff remain review-only. Their
schemas force `executable=false`, and the handoff path produces review data
without invoking an adapter, `RunPlanExecutor`, Gate mutation, remote
transport, SSH, or `molly-worker`. PR-BL does not turn them into an execution
authority.

The conversation assistant still receives a deterministic server decision and
may use a text LLM only to explain that decision. Its prose, Markdown, cards,
and ordinary approval words are not parsed as actions. The dedicated planning
call introduced here is a separate `complete_json()` call with a strict
`AgentExecutionPlanLLMResponse` schema.

PR-BL does not modify `RunPlanExecutor`, Gate snapshots or exact approval,
`RemoteExecutionService`/remote lifecycle, worker protocol, StageState,
queue jobs, adapters, or the execution UI.

## Contracts

The five published JSON schema files are generated from the Pydantic models in
`src/ai4s_agent/schemas.py`:

| Contract | Version | Purpose |
| --- | --- | --- |
| `ScientificToolSpec` | `scientific_tool_spec.v1` | One LLM-visible logical projection of a registered task. |
| `ScientificToolCatalog` | `scientific_tool_catalog.v1` | Stable sorted catalog with canonical digest and excluded-task roster. |
| `AgentProjectObservation` | `agent_project_observation.v1` | Fixed server-derived, privacy-safe planner input. |
| `AgentExecutionPlanLLMResponse` | `agent_execution_plan_llm_response.v1` | Strict high-level task/profile/options proposal from the planning LLM. |
| `AgentExecutionPlanProposal` | `agent_execution_plan_proposal.v1` | Immutable review/control artifact with compiled `RunPlan`. |

`ScientificToolSpec` includes the schema version, canonical tool/task IDs,
human label and description, logical artifact IDs, effect class, risk,
required permissions and gates, a closed high-level option schema, logical
profile requirements, budget dimensions, preapproval declaration,
idempotency policy, verification policy, and planner visibility. Its
`effect_class` vocabulary is deliberately limited to `observe`,
`derive_local`, `mutate_artifacts`, `external_io`, `compute`,
`scientific_confirm`, `change_objective`, and `publish_or_promote`.

The LLM response contains only requested tool IDs, selected input artifact
IDs, typed task options, logical profile IDs, limits, stop conditions, success
criteria, concise rationales, assumptions, and questions. `extra="forbid"`
rejects authority, execution, adapter, command, path, SSH, credential,
status, and unknown fields. Unknown tools, artifacts, profiles, options, and
capabilities fail closed during server compilation.

## Observation sources and privacy boundary

`AgentProjectObservationBuilder` reads only fixed server-side projections:

- validated `StageState` summary: stage/status/next stage, registered task IDs,
  required Gate IDs, safe failure family/error code, and verified artifact IDs;
- artifact registry roster plus content digests, with relative paths used only
  internally for verification;
- allowlisted confirmed-dataset manifest summaries;
- the logical execution-profile/capability snapshot;
- existing `RunPlan` summary and digest;
- the generated tool catalog;
- no configured budget authority unless one exists in the current system.

Every observation binds project/run identity, StageState, artifact registry,
confirmed dataset manifests, catalog, logical profile snapshot, and existing
RunPlan through source digests. Semantic identity excludes wall-clock
`created_at`; canonical JSON sorting makes equal source snapshots produce equal
semantic bytes and digests. The builder rechecks source files, artifact inodes
and bytes, catalog digest, and profile snapshot before publishing.

The observation schema and projection do not contain absolute paths, host/IP,
SSH aliases, usernames or emails, `known_hosts`, commands/argv, environment
assignments, credentials, raw stderr/stdout, raw CSV/PDF/model/document text,
or private chain-of-thought. Artifact content is streamed for hashing; only a
small allowlisted JSON manifest summary can enter the observation. The
recursive schema/privacy checks reject forbidden keys and authority fields
rather than silently removing them.

## Planning and deterministic compilation

The service sends only validated observation material, the goal, explicit user
constraints, and the LLM-facing catalog to `provider.complete_json()` using
`scientific-agent-long-horizon-plan.v1`. It never uses assistant prose or a raw
provider response as an action source. External provider use follows the
existing `external_llm_approved=true` consent check; that consent permits
planning data transfer only and is not execution authorization.

`AgentExecutionPlanCompiler` then:

1. binds the parsed response to the observation and catalog digests;
2. validates tool, artifact, option-schema, logical-profile, and capability
   references;
3. maps tool IDs to registered task IDs;
4. calls the existing `expand_run_plan()` dependency resolver;
5. derives missing artifacts and required gates from registered task specs;
6. rejects LLM-provided dependencies, output artifacts, gates, adapter names,
   status claims, and permissions;
7. creates the canonical `RunPlan` and proposal semantic digest.

The resulting proposal has `status=review_required` and an immutable
`executable=false` literal. It carries no authorization, approval state,
dispatch state, or current execution status claim. It is a review/control
artifact, not a scientific result trust anchor.

## Storage and API contract

Published proposals are stored under the existing project/run scope:

```text
projects/<project_id>/runs/<run_id>/agent_plans/<proposal_id>/
  observation.json
  tool_catalog.json
  llm_response.json
  proposal.json
  proposal_summary.md
  source_binding.json
  verification.json
```

Files are created with no-replace, `O_NOFOLLOW`, exclusive creation, fsync,
and exact canonical-byte replay checks. A repeated publication of identical
bytes is idempotent. Reusing a proposal ID for different bytes is a conflict.
Reads verify all projections, digests, source bindings, and (by default) the
current authoritative source snapshot. A stale StageState, artifact digest,
catalog, profile snapshot, or run plan fails closed. Publication never updates
the Artifact Registry or any execution state.

The additive non-execution endpoints are:

```text
POST /api/projects/<project_id>/agent-plan-proposals
GET  /api/projects/<project_id>/agent-plan-proposals/<proposal_id>
```

POST accepts only `run_id`, `goal`, `user_constraints`, the existing
`external_llm_approved` consent flag, and an optional existing `llm_provider`
override. Observation, catalog, RunPlan, Gate, status, and execution fields
submitted by a client are rejected. GET performs exact verification and returns
safe JSON; it does not refresh state, approve, start, execute, resume, dispatch,
or create a queue job.

Only minimal invocation metadata is persisted: provider/model, prompt version,
response ID, source digests, validated output digest, and safe latency/cost
when available. Raw messages, raw provider responses, headers, keys, and hidden
reasoning are not persisted.

## Threat model and fail-closed cases

The contract treats the following as hostile or stale input:

- tool/task/option/profile/artifact injection and duplicate catalog mappings;
- adapter, callable/module, Python expression, command, shell, argv, path,
  SSH/SCP, worker, environment, or credential injection;
- `approved`, `execute`, `dispatch`, `start_now`, `RUNNING`, `SUCCEEDED`,
  `FAILED`, or status-override injection;
- missing or replaced artifacts, StageState races, stale catalog/profile
  capability, and changed existing RunPlan;
- symlink/replacement attacks against source and proposal files;
- duplicate proposal publication or same-ID different-byte replay;
- raw-data, private-document, infrastructure, email, IP, token, and exception
  leakage to an external LLM, API response, log, or Markdown artifact;
- drift between a planner catalog and the single `AtomicTaskRegistry` source.

Unknown fields and hostile authority fields are rejected; they are not
silently sanitized into a publishable proposal. The server owns dependency,
output artifact, Gate, adapter, verifier, and execution bindings.

## Non-goals

PR-BL does not provide a Permission Engine, authorization, approve-and-start,
stepwise/frozen-plan execution, Gate consumption, remote dispatch, SSH/SCP,
worker calls, adapter execution, queue jobs, Execution Agent, Replanner, UI
execution changes, or end-to-end Harness completion. In particular, external
LLM consent must never be interpreted as execution consent.

## PR-BM handoff

The next PR can reuse the catalog builder/API, observation verifier and source
bindings, proposal verifier and digest, strict option schemas, logical profile
capability bindings, and canonical dependency-expanded `RunPlan`. PR-BM must
add a separate Permission Engine and explicit user authorization; it must not
reinterpret this proposal artifact as authorization or alter the existing
Executor/Gate/remote/worker authority without a separate reviewed change.
