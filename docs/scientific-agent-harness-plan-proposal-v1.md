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
required/optional/alternative input and output artifact contracts, risk level,
required gates, adapter binding, dependency hints, and the planner-facing
local/remote execution-route projection. `expand_run_plan()` resolves the registered
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

PR-BL does not call `RunPlanExecutor.execute()`/`resume_after_gate()`, consume a
Gate, or invoke `RemoteExecutionService`, remote lifecycle, transport, worker,
or adapters. The existing Executor authority and adapter bindings are unchanged.
The only Executor-side change aligns deterministic payload/snapshot construction
for already registered local tasks with their declared artifact contracts; it
does not add an execution entry point or dispatch path. Worker protocol,
StageState, queue jobs, and the execution UI are unchanged.

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
human label and description, required and optional logical artifact IDs,
input-artifact alternative groups, effect class, risk, required permissions and
gates, a closed high-level option schema, an option-compiler version, static and
backend-conditioned logical profile requirements, static/backend-conditioned
`local_executor` or `remote_execution_service` routes, a logical remote task
type, server-owned general/backend-conditioned planner defaults, option IDs
that require an explicit scientific review value, a server-owned default
backend for dependency expansion, per-artifact accepted trust classes, budget
dimensions, preapproval declaration,
idempotency policy, verification policy, and planner visibility. `AtomicTaskSpec.planner_visible`
defaults to `false`: PR-BL freezes
a narrow v1 roster in `DEFAULT_ATOMIC_TASKS`, and every visible task explicitly
sets all projection metadata. New execution tasks remain hidden until a review
adds an explicit projection; no task-ID heuristic infers effect, profile, or
planner visibility. Its `effect_class` vocabulary is deliberately limited to `observe`,
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
RunPlan through source digests. An execution profile is `available` only when
one enabled connection has a digest-matched successful probe whose verified
capabilities independently cover that profile's whole requirement; capability
fragments from disabled or separate connections are never combined. Semantic identity excludes wall-clock
`created_at`; canonical JSON sorting makes equal source snapshots produce equal
semantic bytes and digests. The builder rechecks source files, artifact inodes
and bytes, catalog digest, and profile snapshot before publishing.

The observation schema and projection do not contain absolute paths, host/IP,
SSH aliases, usernames or emails, `known_hosts`, commands/argv, environment
assignments, credentials, raw stderr/stdout, raw CSV/PDF/model/document text,
or private chain-of-thought. Artifact content is streamed for hashing; only a
small allowlisted JSON manifest summary can enter the observation. A registry-
bound input with a content digest but no server-recorded producer is classified
as `content_bound_input`; a registered produced artifact is
`registered_intermediate`; terminal StageState output is `verified_output`; and
only a terminal `confirm_extracted_dataset` output is
`confirmed_scientific_input`. A JSON field such as `confirmed: true` cannot
promote itself. Each tool declares trust classes by logical input ID, so raw
PDF/CSV inputs can be proposed only for a matching registered task. In
particular, `inspect_dataset` accepts one content-bound/registered
`uploaded_dataset` or a confirmed `confirmed_training_dataset`; that alternative
is part of the registered task contract and the selected content digest is bound
into the observation and proposal.

The recursive schema/privacy checks reject exact forbidden structural keys and
authority fields rather than silently removing them. Natural-language goal and
rationale text is not treated as an authority parser: normal OLED wording such
as “host–dopant”, “failed validation”, and “authorization review” is allowed.
Prose is still rejected for concrete absolute paths, IPs, email addresses,
credential formats, and environment assignments.

## Planning and deterministic compilation

The service sends only validated observation material, the goal, explicit user
constraints, and the LLM-facing catalog to `provider.complete_json()` using
`scientific-agent-long-horizon-plan.v1`. It never uses assistant prose or a raw
provider response as an action source. External provider use follows the
existing `external_llm_approved=true` consent check; that consent permits
planning data transfer only and is not execution authorization.

`AgentExecutionPlanCompiler` then:

1. binds the parsed response to the observation and catalog digests;
2. validates tool, artifact, logical-profile, and capability references;
3. maps tool IDs to registered task IDs;
4. calls the existing `expand_run_plan()` dependency resolver, including a
   deterministic choice among registered input alternatives;
5. materializes `effective_planner_options` for every expanded task from the
   catalog-bound defaults plus the validated LLM option patch, and emits a
   blocking question for each unresolved review-required scientific option;
6. compiles every expanded task through the registered, versioned
   `PlannerOptionCompiler` into canonical scientific task options; even a task
   with no options has an explicit `{}` entry;
7. derives backend-conditioned profile requirements after expansion, so an
   implicit MinerU/Uni-Mol/REINVENT4 dependency may bind an appropriate selected
   logical profile without appearing in `requested_tool_ids`;
8. derives an immutable per-task dispatch intent: local tasks bind
   `local_executor`; MinerU, Uni-Mol, and REINVENT4 bind
   `remote_execution_service`, logical task type, selected logical profile, and
   a nullable resource-request intent;
9. derives missing artifacts and required gates from registered task specs;
10. rejects LLM-provided dependencies, output artifacts, gates, adapter names,
   status claims, and permissions;
11. creates the canonical `RunPlan` and proposal semantic digest.

The proposal persists raw `planner_options`, task-keyed
`effective_planner_options`, and task-keyed `compiled_task_options`, plus
`scientific-planner-option-compiler.v1`. The former is the validated LLM
suggestion. The effective map records server defaults and explicit unresolved
review placeholders. The compiled map is the exact deterministic scientific-parameter
representation that a future authorization must bind together with
`dispatch_intents`. Both task-keyed maps and `dispatch_intents` exactly cover
the complete expanded `RunPlan`; a missing key is invalid, including for an
implicit dependency. Non-planner-visible internal dependencies are permitted
only with fixed empty caller options. Local options map only to fields consumed
by the existing Executor payload/snapshot path. Remote options never contain an adapter name,
SSH/SCP material, command, environment, or connection detail: Uni-Mol,
REINVENT4, and MinerU freeze only a `RemoteExecutionService` dispatch intent.
Incomplete remote profile or resource authority becomes a blocking review
question and cannot be interpreted as dispatch permission. Tasks whose current
Executor branch does not consume configurable options expose a closed empty v1
schema rather than claiming unsupported semantics.

The artifact roster is tested per tool with only its required inputs and one
selected optional/alternative input; no global registry artifact union is used.
The registered data chain supports both content-bound `uploaded_dataset ->
inspect_dataset -> clean_dataset -> check_trainability -> baseline train_model`
and `confirmed_training_dataset -> inspect_dataset -> check_trainability ->
baseline train_model` without inventing `uploaded_dataset` as missing. Local
dependency expansion selects artifact producers from the bound source snapshot
and the explicitly requested upstream task roster: cleaning is authoritative
for a raw upload, while inspection is authoritative for an already confirmed
dataset. Explicit granular fallback producers remain authoritative over a
monolithic workflow that happens to emit the same logical artifact.
`clean_dataset` therefore
declares the `property_catalog` that the Executor actually registers; the
Planner does not model the dataset itself as an unused `check_trainability`
input.

Phase 3 payload snapshots likewise bind exactly the registered logical inputs.

The frozen permission vocabulary is `read_content_bound_input`,
`derive_project_artifact`, `external_document_processing`,
`model_training_compute`, `model_inference_compute`,
`candidate_generation_compute`, and
`scientific_dataset_confirmation`. Every planner-visible task declares at least
one meaningful permission. PR-BL only describes these permissions; it does not
decide or grant them.

The resulting proposal has `status=review_required` and an immutable
`executable=false` literal. It carries no authorization, approval state,
dispatch state, or current execution status claim. It is a review/control
artifact, not a scientific result trust anchor.

The contract deliberately separates identities:

- `semantic_plan_digest` / `semantic_plan_id` bind deterministic planning
  semantics and exclude time, provider response IDs, latency, and invocation
  metadata;
- `invocation_id` records one dedicated LLM call;
- `client_request_id` binds one client retry request; and
- `publication_id` (also returned as compatibility `proposal_id`) names the
  immutable persisted envelope and its full `proposal_digest`.

A repeated `client_request_id` with identical request material replays the
exact stored publication without another LLM call. A request-scoped advisory
file lock covers reservation, planning, checkpoint, publication, and commit
across processes. Reuse with different request material fails before a second
LLM call. Separate requests may publish distinct envelopes for the same semantic
plan without a no-replace byte conflict.

## Storage and API contract

Published proposals use a project-scoped planning-only area; a run ID remains a
logical observation/RunPlan binding but does not need to pre-exist:

```text
projects/<project_id>/agent_plan_proposals/<publication_id>/
  observation.json
  tool_catalog.json
  llm_response.json
  proposal.json
  proposal_summary.md
  source_binding.json
  verification.json
  publication_manifest.json

projects/<project_id>/agent_plan_requests/<client_request_id>/
  request.lock
  reservation.json
  planning.json
  planning_checkpoint.json
  publication_pending.json
  committed.json
```

Request markers are immutable and advance through `RESERVED`, `PLANNING`,
`PUBLICATION_PENDING`, and `COMMITTED`. A validated post-LLM checkpoint supports
exact recovery without another provider call. If a process dies after the
provider returned but before a validated checkpoint can be persisted, retry
enters a typed `PLANNING` recovery state instead of ambiguously repeating an
external call.

Files are created in a request-private staging directory with no-replace,
`O_NOFOLLOW`, exclusive creation and fsync; the manifest is written last, the
directory is fsynced, and an atomic rename publishes the complete directory.
The proposal root is then fsynced before the immutable request binding is
committed. Recovery after a partial staging write, completed rename, or rename
without the final request binding verifies and reuses the exact checkpoint and
publication. A repeated publication of identical bytes is idempotent. Reusing
a publication ID for different bytes is a conflict.
Creating the first proposal for a project never creates a run directory,
StageState, GateDecision, queue job, or execution authority.
Reads verify all projections, digests, source bindings, and (by default) the
current authoritative source snapshot. A stale StageState, artifact digest,
catalog, profile snapshot, or run plan fails closed. Publication never updates
the Artifact Registry or any execution state.

The additive non-execution endpoints are:

```text
POST /api/projects/<project_id>/agent-plan-proposals
GET  /api/projects/<project_id>/agent-plan-proposals/<proposal_id>
```

POST accepts only `run_id`, `goal`, `user_constraints`, optional
`client_request_id`, the existing
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
  capability, a probe whose digest no longer matches its connection, split
  capabilities across connections, and changed existing RunPlan;
- symlink/replacement attacks against source and proposal files;
- duplicate publication, request-ID reuse with different request material, or
  same-publication-ID different-byte replay;
- concurrent same-request planning across Flask workers, interrupted staging
  writes, publication-before-binding crashes, and ambiguous post-provider
  failures;
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
bindings, proposal verifier and digest, strict option schemas,
`effective_planner_options`, `compiled_task_options`, option-compiler version,
server defaults/review-required option IDs, artifact alternatives and
per-input trust bindings, explicit permission metadata, backend-conditioned
logical profile capability bindings, `dispatch_intents`, remote logical task
types, nullable resource-request intents, and canonical dependency-expanded
`RunPlan`. A future exact authorization must bind `effective_planner_options`,
`compiled_task_options`, and
`dispatch_intents`, not
recompile or reinterpret planner prose/options. PR-BM must add a separate
Permission Engine and explicit user authorization; it must not reinterpret this
proposal artifact as authorization or alter the existing
Executor/Gate/remote/worker authority without a separate reviewed change.

## Generated schema propagation

All schema files under `docs/schemas/` are regenerated by the same
`export_json_schemas()` implementation from the Pydantic source models. Several
non-Harness composite schemas reference `AtomicTaskSpec`, so regeneration may
propagate its new optional/default planning metadata into Critic, Gate, OLED, or
other aggregate schema documents. That propagation does not change those
components' authority or runtime semantics; `planner_visible` remains false by
default and the fields are inert outside the explicit catalog projection. The
schema regeneration test requires a clean byte-for-byte workspace after running
the generator.
