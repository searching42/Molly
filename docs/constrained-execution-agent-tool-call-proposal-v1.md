# Constrained Execution Agent and ToolCallProposal v1

Status: PR-BO design freeze for `M3H-009`.

## Authority boundary

PR-BO adds a bounded selector between the current deterministic Controller
inspection and one existing Controller advance. It does not add a task
scheduler or an execution backend.

```text
locked current Controller execution + inspection
  -> privacy-safe execution observation
  -> server-derived bounded tool catalog
  -> one schema-constrained LLM call
  -> immutable review-only ToolCallProposal
  -> explicit current-verified application
  -> at most one existing Controller advance
  -> immutable application receipt
```

The Controller remains the sole authority for the current task, task order,
route, action, adapter, task attempt, Gate boundary, remote lifecycle, and
terminal status. `RunPlanExecutor`, `RemoteExecutionLifecycleService`,
`GateDecision`, `StageState`, Artifact Registry, and verified publications
remain the only execution and result authorities. An LLM response, proposal,
application receipt, conversation message, or trace span cannot assert task
success or grant execution authority.

## Locked Controller handoff

The Controller exposes a server-only read seam that holds the existing
`controller_execution.lock`, current-verifies the immutable execution and its
authority chain, constructs one fresh inspection, and reads the current receipt
leaf. It creates no decision or receipt and writes no execution state.

Proposal application calls the existing `Controller.advance` with both the
expected execution digest and the proposal inspection digest. The inspection
comparison occurs while both the Controller execution lock and request lock
are held. A first call must match the frozen inspection; an exact replay with
an already committed decision remains readable after its own effect changes
that inspection. This keeps the established lock order:

```text
proposal application lock
  -> Controller execution lock
  -> Controller request lock
  -> existing local or remote lifecycle lock
```

The public Controller request schemas and
`scientific-agent-harness-controller-policy.v1` bytes are unchanged.

## Observation and privacy

`agent_execution_agent_observation.v1` is an explicit allowlist projection. It
contains only validated logical identifiers, fixed enums, bounded integers,
opaque SHA-256 digests, safe reason codes, and the current task slot's already
authorized digest bindings. Inspection facts retain their authority class,
logical source identity, digest, and fixed state, but not free-form `detail`.

The projection excludes goal and conversation text, user and authorization
notes, Gate notes, artifact names and contents, paths, hosts, connection
locators, commands, argv, environment data, credentials, stdout/stderr, raw
exceptions, raw provider material, PDFs, CSV rows, molecule strings, model
bytes, and private reasoning. The complete Controller execution or inspection
objects are never sent to the provider.

Authoritative facts are immutable control artifacts, current StageState,
Registry bindings, Gate/remote approvals, and verified publications. Derived
facts bind the digest roster from which they were calculated. Mutable remote
transport state remains observational and cannot override terminal authority.

## Shared Controller action boundary

A pure Controller helper classifies the current action as:

```text
ORDINARY_ADVANCE
USER_GATE_APPROVAL
USER_REMOTE_APPROVAL
EXPLICIT_RECOVERY
TERMINAL_OBSERVATION
```

Ordinary Controller decision creation and Execution Agent catalog construction
reuse that helper. A terminal Controller action remains `ORDINARY_ADVANCE`
until its exact terminal receipt has been committed; afterward the same stable
snapshot is `TERMINAL_OBSERVATION`. Explicit cancellation is never exposed in
the Execution Agent catalog.

## Server-owned tool catalog

`agent_execution_tool_catalog.v1` contains a state-dependent subset of this
fixed roster:

| Tool | Current boundary | Application effect |
| --- | --- | --- |
| `controller.advance_current.v1` | ordinary advance | Calls `Controller.advance` once with no LLM/client arguments. |
| `agent.pause_current.v1` | every valid snapshot | No Controller mutation. |
| `user.request_gate_approval.v1` | waiting Gate | Records user action required; never writes a Gate decision. |
| `user.request_remote_approval.v1` | waiting remote approval | Records user action required; never approves or dispatches. |
| `user.request_recovery.v1` | recovery required | Records user action required; never calls recovery. |
| `agent.observe_terminal.v1` | stable terminal | Records terminal observation; never calls Controller. |

Tool specifications contain no task, adapter, route URL, profile, resource,
path, host, command, argv, or arbitrary argument object. The server compiles a
selected tool ID to one fixed operation enum; the LLM cannot supply that enum.

## LLM call and prompt

The prompt version is `scientific-agent-execution-selection.v1`. The system
message instructs the model to select exactly one advertised tool and return
only the strict response object. The user message is canonical JSON containing
only the observation and catalog. The prompt digest binds the exact system
text, observation digest, catalog digest, response-schema digest, and execution
agent policy digest.

`AgentExecutionLLMResponse` permits only `selected_tool_id` and an optional
bounded `decision_summary`. Unknown fields, arguments, multiple tools,
authority fields, task/adapter/profile/resource data, chain-of-thought, and
non-object output fail strict Pydantic and JSON-Schema validation. The provider
is called through `LLMProvider.complete_json()`. PR-BO additionally rejects
markdown-wrapped or embedded JSON instead of relying on the provider's legacy
permissive extraction. Unsafe summaries fail the whole response; they are not
redacted into an executable proposal.

Provider configuration and consent reuse the existing `LLMSettingsStore`,
`LLMProviderManager`, and external-endpoint consent resolver. The API requires
the literal JSON boolean `external_llm_approved=true`; that consent permits
only the bounded observation transfer and is not execution authority. No API
key, endpoint credential, raw prompt, raw request, raw response, or raw
provider exception is persisted.

## Proposal identity and publication

`agent_tool_call_proposal.v1` binds the Controller execution and inspection,
safe observation, exact catalog and selected tool, current task/attempt/slot,
server-compiled operation, execution-agent policy, prompt identity, safe
provider/model/response metadata, parsed response, and exact source roster.
It always has:

```text
status = review_only
executable = false
```

It contains no HTTP route/body, task options, adapter, profile, resources,
host, path, command, argv, credential, approval, recovery/cancel/retry flag,
raw provider material, conversation, or trace identity.

The publication uses a request-private staging directory, exclusive no-follow
files, file and directory fsync, manifest-last activation, no-replace atomic
rename, collection fsync, and exact byte reread. Historical stale proposals
remain readable audit artifacts but cannot be applied.

## Proposal request crash semantics

Each `(Controller execution, client request)` is process-locked and advances
through immutable checkpoints:

```text
RESERVED
  -> OBSERVATION_FROZEN
  -> LLM_REQUEST_STARTED
  -> LLM_RESPONSE_COMMITTED
  -> PROPOSAL_COMMITTED
```

The observation is current-verified again before provider invocation and after
the response checkpoint. Same request and same binding replays; different
binding conflicts before a second provider call. If the process dies or the
provider fails after `LLM_REQUEST_STARTED` but before a safe response
checkpoint, the outcome is `execution_agent_llm_outcome_unknown` and is never
automatically retried. A committed response is recoverable without another
provider call. If its inspection is no longer current, the request is marked
stale/aborted and no applicable proposal is published.

## Application and exactly-once behavior

Application accepts only the expected proposal digest and a canonical client
request ID. It exact-verifies the proposal publication and current Controller
snapshot under a proposal-level application lock. Any execution, inspection,
source, catalog, authorization, adapter-authority, Gate, Registry, StageState,
remote approval/state/publication, or AuthoritySet drift fails closed before a
Controller effect.

For `controller.advance_current.v1`, the server derives the Controller request
ID from the proposal ID, proposal digest, and selected tool. It calls the
existing Controller once. A crash after the Controller decision or receipt is
re-entered with that same request ID; Controller evidence is replayed or
reconciled and no second effect is selected. Every other tool is no-effect and
only creates an application receipt. No-effect application retains the
Controller execution lock from current snapshot verification through receipt
publication, so a concurrent Gate/remote approval, advance, cancel, or recovery
cannot be interleaved into a receipt that claims an unchanged inspection.

`agent_tool_call_application_receipt.v1` binds before/after inspection,
proposal and operation identity, exact Controller decision/receipt when called,
side-effect flags, fixed outcome, user boundary, safe reason codes, and source
bindings. `applied` means only that the server applied the proposal and bound
the Controller result; it is not a task-success claim. One proposal has at most
one application receipt even when different client request IDs race.

## Tracing

Execution Agent spans and events use the existing `HarnessTracer` seam and its
closed allowlists. OpenTelemetry remains lazy, optional, fail-open, and
non-authoritative. Prompts, summaries, provider responses, project content,
paths, hosts, endpoints, credentials, exceptions, conversation text, and
artifact content are forbidden attributes. Tracing on/off and exporter failure
must produce byte-identical observations, catalogs, proposals, Controller
artifacts, StageState/Registry/remote evidence, and application receipts.

## Conversation and non-goals

PR-BO does not modify ordinary conversation messages,
`/api/agent/conversation/next-turn`, `ConversationAgent`, `ConversationStore`,
or the conversation execution-request freeze contract. Chat text cannot create
or apply a ToolCallProposal.

PR-BO adds no arbitrary shell, SSH, path, argv, adapter selection, worker
protocol, connection/profile/resource selection, task selection, retry,
recovery, cancellation, approval, plan mutation, Replanner, execution loop,
background runner, complete Harness UI, or real-infrastructure canary.

## PR-BP handoff

PR-BP may consume immutable proposals, user-action-required outcomes,
Controller terminal/failure observations, and explicit user feedback. Any
material change must produce an explicit plan diff, a new plan digest, and new
authorization. It may not mutate or reinterpret an existing proposal or
authorization.
