# Scientific Agent Replanner and plan revision v1

The Replanner is a review-only control-plane component. It consumes exact,
current-verified PR-BL through PR-BO artifacts and can ask one configured LLM
for a bounded revision suggestion. The LLM does not edit a `RunPlan`, choose
dependencies, grant resources, approve a Gate, authorize work, select a
Controller action, or dispatch execution.

## Authority boundary

Explicit feedback enters only through the dedicated feedback API. The raw text
is stored in project-private storage; public/control artifacts, logs, and trace
data contain only the receipt ID, digest, source kind, and fixed reason code.
A feedback receipt is advisory: it is not approval, execution instruction,
GateDecision, or scientific truth. Ordinary conversation is not converted to a
feedback receipt.

The server builds `agent_replanner_observation.v1` from the immutable baseline
proposal and authorization, the exact Permission decision, current tool catalog,
current Controller inspection/receipt when present, and current verified plan
sources. Paths, hosts, commands, credentials, raw artifacts, stdout/stderr,
provider payloads, exceptions, conversation history, and private reasoning are
not projected. Unsafe LLM text is rejected rather than redacted.

The v1 trigger enum accepts dedicated feedback, Controller failure/terminal,
plan-source drift, and user-requested revision triggers. A standalone
`verifier_outcome` trigger is intentionally not exposed in v1: it will require
a future versioned binding that exact-identifies and current-verifies the
project/run/task, StageState, Registry lineage, verified publication, and full
source roster. A client-supplied outcome label is never verifier evidence.

All LLM-controlled prose uses one concrete-payload policy across rationale,
question prompt/reason, stop conditions, and success criteria. Private paths,
endpoints, credential assignments, execution output, and shell payloads fail
closed. Scientific domain language remains valid; in particular, OLED terms
such as `host material` and `host–dopant pair` are not treated as hostnames.

The validated LLM response is only a suggestion. The existing PR-BL catalog,
`AtomicTaskRegistry`, dependency expansion, option compiler, artifact trust
checks, route/profile selection, resource and budget checks, and Gate bindings
compile a complete candidate again on the server.

## Canonical complete diff

`agent_plan_diff.v1` compares the full baseline and successor compiled plan.
Its stable, ordered changes cover:

- requested, added, removed, retained, reordered, visible, and hidden tasks;
- registry-derived dependency edges and hidden dependency expansion;
- raw planner, effective/default-derived, and compiled task options;
- selected, available, missing, required, optional, and output artifacts;
- local/remote routes, logical profiles, remote task types, and resource intents;
- limits and budget semantics;
- required and pending Gate semantics;
- goal constraints, stop conditions, success criteria, blocking questions,
  catalog digest, option compiler version, dispatch intents, and RunPlan digest.

Each entry records presence separately from its value, so absence and explicit
`null` differ. Maps use canonical JSON key order and change entries use stable
dimension/path order. `created_at` is excluded from the diff digest. A rebuild
from the two proposals must reproduce the same projection and diff bytes; an
unknown semantic dimension fails closed.

Rationale-only changes do not alter plan semantics. An empty canonical diff
publishes `no_material_change`, creates no successor proposal or authorization,
and cannot be applied.

## Proposal, application, authorization, and dispatch

A revision proposal is immutable, `review_only=true`, `executable=false`,
`authorized=false`, and `applied=false`. Creating it cannot call Controller,
Executor, RemoteExecutionService, adapters, workers, Gate writers, StageState,
Registry/publication writers, retry, recovery, or cancellation.

Application is a separate explicit operation. Before first publication it
exact-reads the revision, re-verifies every baseline/source binding, rebuilds
the current observation, recompiles the successor, and regenerates the diff.
Only exact byte-equivalent candidate and diff material can be published.
Application then uses the existing PR-BL immutable proposal store and records
parent/supersedes bindings in an immutable application receipt. It never
authorizes or dispatches.

Crash reconciliation is historical rather than a new current-plan decision.
If the revision-determined successor publication already exists, application
first exact-reads every immutable publication file without consulting the
current Registry or observation sources, proves the stored proposal equals the
revision candidate and its digest/parent/diff bindings, and adopts it into the
single application receipt. A completed application replay follows the same
immutable reader. Later catalog, artifact, or profile drift can therefore make
a new application stale, but cannot orphan an already-published successor or
invalidate exact replay of its already-committed receipt.

The old proposal and authorization stay immutable and remain bound only to the
old digest. A material revision has a new proposal and semantic-plan digest.
The existing PR-BM Permission Engine must freshly evaluate that proposal, and a
trusted authenticated user must explicitly create a new authorization. The old
authorization fails closed when presented for the successor digest. The usual
non-dispatched start-intent and Controller path remains the only way forward.

## Provider, recovery, privacy, and tracing

Each replan request has a durable reservation and provider-started checkpoint.
The provider is called at most once. A safe parsed outcome is checkpointed
without raw prompt, raw response, endpoint, credential, or exception. If the
process cannot establish whether the provider completed, replay returns the
typed unknown-outcome state and never retries automatically. Once the safe
outcome exists, candidate compilation and publication are adopt/reconcile-only.

Revision create and apply use project-scoped process locks, immutable request
digests, `O_NOFOLLOW` regular-file checks, private staging, fsync, manifest-last
publications, atomic rename, and exact-byte rereads. Same request/same content
replays exactly; same request/different content conflicts. Applying requests are
reserved by revision before successor publication so competing processes cannot
claim the crash window. Recovery never re-executes a scientific task.

Harness tracing remains optional, lazy, fail-open, privacy-allowlisted, and
non-authoritative. Trace/span identities and exporter outcomes do not enter any
request, observation, diff, proposal, Permission, authorization, or application
digest. Feedback text, prompts, provider responses, artifact contents, paths,
hosts, commands, credentials, exceptions, and private reasoning are forbidden.

## Compatibility, limitations, and rollback

This version adds no UI, autonomous loop, automatic retry/recovery/cancel,
resource/budget expansion, connection switching, Gate or remote approval,
Controller action, Executor call, scientific-result authority, or real
infrastructure canary. PR-BL proposals, PR-BM Permission/authorization/start
intent, PR-BM2 AuthoritySet, PR-BN Controller, PR-BO Execution Agent, legacy
execution, conversation, Gate, StageState, Registry, and publication semantics
remain unchanged.

Rollback is an ordinary code revert with no data migration. Already published
feedback receipts, revision proposals, successor proposals, and application
receipts remain immutable read-only audit artifacts.
