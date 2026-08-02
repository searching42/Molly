# Scientific Agent run inspection v1

`agent_run_inspection.v1` is Molly's one canonical, reconstructable, read-only
projection for a current verified Scientific Agent run. Its only HTTP boundary
is:

```text
GET /api/projects/{project_id}/agent-runs/{run_id}/inspection
```

The projection is derived (`authoritative=false`) and cannot be used as an
execution, authorization, Gate, StageState, Registry, publication, verifier, or
scientific-success authority. It is the stable backend handoff for PR-BQ2,
PR-BR1, PR-BR2, PR-BQ3, and PR-BR3.

## Authority chain

The service composes existing exact readers rather than parsing authority files
itself. Before Controller start, the Planner publication current reader and the
Permission/authorization/start-intent verifiers establish currentness. After
Controller start, `ScientificAgentHarnessController.get()` establishes the
stronger post-start chain:

```text
Planner proposal + observation + tool catalog
  -> Permission decision
  -> trusted-user authorization (+ configured AuthoritySet when required)
  -> start intent
  -> Controller execution
  -> Controller decision and action receipt chain
  -> current StageState + Artifact Registry
  -> exact local execution publication or remote verified publication
```

Execution Agent proposals and application receipts are exact-read and bound to
the same Controller execution/inspection/decision/receipt. Replanner feedback
receipts, revisions, complete plan diffs, successor publications, and
application receipts are exact-read as a separate review/application chain. An
applied successor remains non-executable until fresh Permission and trusted-user
authorization exist.

The run-level outcome never comes from a client outcome label, conversation,
LLM response, trace, telemetry, stdout, stderr, exception, or UI state. Terminal
and in-flight outcomes are derived from the verified Controller inspection,
whose completion checks require the exact StageState, Artifact Registry,
verified publication, and Controller/execution receipt bindings.

## Source roster and deterministic digest

Every response contains a canonically sorted `source_roster`. Each entry has a
privacy-safe source name/kind, exact ID, exact digest, and `current` or
`historical` currentness. The roster covers the Planner sources, control
authority, Controller facts and immutable receipt chain, Execution Agent facts,
and Replanner facts that affect the projection.

Tasks, artifacts, tool calls, revisions, Gate rosters, consumers, dependencies,
and sources are sorted before the projection is hashed. `inspection_digest` is
the canonical SHA-256 digest of the semantic projection. `created_at`,
`inspection_id`, and `inspection_digest` are excluded from semantic material,
so wall-clock time and process identity do not alter the digest. The inspection
ID is derived from the digest. No persistent inspection cache exists.

## Current versus historical semantics

The Planner current reader is used for a plan that has not started. Once a
Controller execution exists, Controller verification permits only dynamic
StageState/Registry changes anchored by its immutable receipt chain. This is
necessary because the pre-start observation is intentionally no longer
byte-current after a committed task.

The existing immutable Planner publication reader is used only to reconstruct
published history and Replanner successor ancestry. An applied Execution Agent
proposal or completed Replanner application is marked historical in the source
roster. A historical exact reader never establishes a new request's
currentness, Permission decision, authorization, dispatch, or current scientific
state.

## Task, artifact, and lineage boundary

Task views expose logical task IDs, dependency and artifact rosters, logical
route/profile/resource digests, Gate requirements, exact StageState/Registry/
publication bindings, verifier-supported outcome, and recovery requirement.
Artifact views expose logical ID/digest/type/role, producer and consumer IDs,
Registry/publication bindings, provenance digest, and currentness.

Artifact bytes are never returned. Registry relative paths are never returned.
The service does not read raw artifact contents; existing Planner and Controller
verifiers perform any required exact content verification behind their existing
authority boundary.

## Privacy boundary

The response excludes local paths, host names, IP addresses, usernames, SSH
material, endpoints, credentials, commands/argv, stdout/stderr, raw exceptions,
provider prompts/responses, conversation history, private feedback, private
reasoning, and raw artifact contents. Trusted actor identity is represented only
by a domain-separated digest. Provider metadata is not projected.

Domain prose such as OLED “host material” and “host–dopant” remains valid; the
projection copies no Planner/LLM prose, so infrastructure-host filtering cannot
misclassify these scientific terms.

## HTTP semantics and failure taxonomy

The GET route rejects every query parameter and every request body. Headers,
conversation state, telemetry, and trace metadata are ignored and cannot
override any field. Project and run IDs use the existing canonical lowercase
single-component validation.

| Condition | HTTP | Stable status/code |
|---|---:|---|
| Current exact projection | 200 | `current` / `RUN_INSPECTION_CURRENT` |
| Verified recovery-required state | 409 | `recovery_required` / `RUN_INSPECTION_RECOVERY_REQUIRED` |
| Current source drift | 409 | `stale_source` / `RUN_INSPECTION_SOURCE_STALE` |
| Replacement, competing head, or conflicting immutable slot | 409 | `replaced_source` / fixed replacement/conflict code |
| Missing project/run/source | 404 | `missing_source` / `RUN_INSPECTION_SOURCE_MISSING` |
| Corrupt, truncated, unsafe, or schema-invalid source | 422 | `damaged_source` / `RUN_INSPECTION_SOURCE_DAMAGED` |
| Broken cross-binding | 409 | `incomplete_authority_chain` / fixed binding code |
| Invalid scope/query/body | 400 | fixed request code |

Errors are fixed safe JSON and set `authoritative_status_available=false`. Raw
exceptions and partially verified success/lineage are not returned. A verified
recovery-required projection remains available but is returned with HTTP 409 so
callers cannot silently treat it as ordinary success.

## Read-only and compatibility guarantees

Inspection calls do not create or change a proposal, Permission decision,
authorization, start intent, Controller decision/receipt, Gate, StageState,
Registry, publication, Execution Agent artifact, or Replanner artifact. They do
not call an LLM provider, Executor, RemoteExecutionService mutation, worker, or
adapter. They do not dispatch, resume, retry, recover, cancel, approve, register,
or publish. Tests snapshot authoritative bytes before and after reads.

The route is additive. Existing manual/legacy execute and resume APIs,
Controller, Execution Agent, Replanner, UI, and `molly-worker` protocol are not
changed. Tracing remains optional and non-authoritative; no OpenTelemetry or
LangSmith package is required.

## Non-goals and handoff

This version does not deploy observability, implement an OTel/LangSmith adapter,
run the Structured Dataset or PDF–MinerU–LLM canaries, build the unified UI,
change frontend frameworks, add automatic loops/retry/recovery/cancellation,
approve Gates, expand resources or budgets, or claim scientific validation.

PR-BQ2 may translate the stable projection into non-authoritative observability.
PR-BR1 and PR-BR2 must bind their canary evidence to this inspection identity and
the underlying exact sources. PR-BQ3 may render the projection without becoming
authority. PR-BR3 remains responsible for representative runtime, recovery,
privacy, replay, and final v1 acceptance.

This PR does not create execution, authorization, Gate, StageState, Registry,
publication, verifier, or scientific-success authority.

No M3H Gate V, M3.5 completion, or Molly v1 completion is claimed.
