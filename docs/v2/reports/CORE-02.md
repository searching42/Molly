# Molly Core v2 CORE-02 single AgentLoop and bounded tool execution

## Scope and base

This report records the CORE-02 implementation only.  CORE-03 and every later
milestone are not started.

```text
repository: searching42/Molly
base: origin/main@7b3ed82a880712f97b626d365d481295d01e8f3a
branch: codex/molly-core-v2-core-02-agent-loop
draft PR: #67 (https://github.com/searching42/Molly/pull/67)
```

The base is the merged CORE-01 main line.  The accepted CORE-01 commit
`b31513f39e2f36c69eff55ecec941278556a8f52` is an ancestor of the base.  The
readiness manifest still reports C0-C7 PASS, `core_goal_mode_ready=true`, and
`core_cutover_ready=false`.

## Implementation commits and files

The coherent implementation commit is:

```text
1b86a439d42658cc69beb15888d2edddf08aac51
```

It adds:

```text
src/molly/core/agent_loop.py
src/molly/core/approvals.py
src/molly/core/runs.py
src/molly/core/tools.py
```

Small CORE-01 extensions preserve the existing data foundation:

```text
src/molly/core/errors.py
src/molly/core/ids.py
src/molly/core/ledger.py
src/molly/core/lineage.py
src/molly/core/__init__.py
```

The focused production test module is:

```text
tests/molly/test_core02_agent_loop.py
```

No `pyproject.toml` or `uv.lock` change is required.  No legacy v1 module was
modified.

## Authority and bounded contracts

`AgentLoop` is the only v2 execution authority.  `RunEngine` is only a
descriptive alias of that same implementation; there is no second controller,
planner, replanner, recovery agent, permission graph, lease, or authority
stack.

`RunRequest` and `RunBudget` are frozen canonical values.  A server-generated
run ID and a request SHA-256 are durably bound by `RUN_STARTED`.  Resuming the
same run with a changed request or active policy digest fails closed.  Decision,
tool-call, step, and budget consumption are reconstructed from the append-only
`RunLedger`; no mutable run-state file is introduced.

`ToolSpec` validates its input/output JSON Schemas with the repository's
`jsonschema` dependency and rejects indirect `$ref` resolution.  A closed
`ToolRegistry` keeps the executor host-owned and exposes only name,
description, and input schema to the DecisionProvider.  `ToolPolicy` binds
exact tool names and side-effect classes and can only make a ToolSpec stricter
about human approval.

The model-owned action is limited to `ToolCallProposal`, `StopAction`, and
`RequestReviewAction`.  The server materializes `step_id`, `call_id`, tool
version/spec digest, policy digest, input IDs, and the deterministic
`tool_call_digest`.  The latter is the idempotency key passed to
`ToolExecutionContext`; model actions cannot supply identity, executor,
backend, path, or credential fields.

`ApprovalRecord` binds one `APPROVED` or `REJECTED` decision to exactly one
materialized call digest.  Approval-required calls are materialized and
persisted before returning `WAITING_APPROVAL`.  Approval resume reconstructs
the persisted call from the ledger and does not ask the DecisionProvider to
recreate it.  A rejected call is durably recorded and is not executed; only a
later normal loop turn may obtain another provider action.

`ToolExecutionContext` exposes only the declared input artifact reader and
bounded run/step/call/idempotency identities.  It has no public store root,
ledger, lineage, registry, policy, filesystem, network, or credential
authority.  `ArtifactDraft` contains only intrinsic publication bytes and
media/schema metadata.  Tool output data is validated against the ToolSpec
schema before any successful execution event is written.

## CORE-01 authority preservation

`ArtifactStore` remains the immutable content authority and produces exact
SHA-256 artifact IDs.  `RunLedger` remains execution-occurrence truth.
`ArtifactLineage` remains a bounded provenance projection; it records
`PRODUCED_BY`, `DERIVED_FROM`, and `CONSUMED_BY` relations only after a durable
`TOOL_EXECUTION_SUCCEEDED` event.  Relation IDs for execution projections are
derived deterministically from the success event and relation identities, so
missing relations can be repaired idempotently and conflicting semantics fail
closed.

Artifact visibility is run-scoped: initial RunRequest inputs plus successful
outputs from that same run.  An existing artifact ID alone is not sufficient
authority.  Cross-run content reuse is possible only when explicitly supplied
as an initial input.  The loop never uses aggregated lineage parents to infer a
particular run occurrence; the ledger event, run ID, and step ID remain
authoritative.

The CORE-01A content/provenance separation is unchanged.  `ArtifactRecord`
contains intrinsic content metadata only; producer/input occurrence provenance
is represented by ledger events and lineage relations.  Identical bytes may
therefore have one content identity and multiple correct production occurrences
across runs.

## Event and crash behavior

The bounded event vocabulary is:

```text
RUN_STARTED, DECISION_RECORDED, TOOL_CALL_MATERIALIZED,
APPROVAL_REQUIRED, APPROVAL_RECORDED, TOOL_CALL_REJECTED,
TOOL_EXECUTION_STARTED, TOOL_EXECUTION_SUCCEEDED, TOOL_EXECUTION_FAILED,
REVIEW_REQUESTED, RUN_STOPPED, RUN_FAILED, BUDGET_EXHAUSTED
```

On restart, successful ledger events are projected into missing lineage before
another provider action is requested.  Repeating reconciliation does not
duplicate relations.  A `TOOL_EXECUTION_STARTED` event without a success or
failure terminal is projected as `INTERRUPTED` and is never automatically
reexecuted, including for a PURE tool.  Executor exceptions produce a bounded
sanitized failure event and are not automatically retried.

## Tests and verification

The dedicated test module covers:

```text
request/policy binding and server-owned IDs
closed registry and sanitized provider view
JSON Schema validation and closed policy checks
exact approval waiting, restart, rejection, and digest mismatch
run-scoped artifact visibility and explicit cross-run inputs
bounded context access and no raw store authority
ArtifactStore/RunLedger/ArtifactLineage integration
same-content production in distinct runs
restart lineage reconciliation and deterministic conflict handling
interrupted-call handling and ledger-reconstructed budgets
StopAction, RequestReviewAction, unknown actions, and import boundaries
```

Local checks completed before final CI:

```text
git diff --check: PASS
python -m compileall -q src tests prototypes: PASS
CORE-01 + CORE-02 + readiness + C4 regression: 50 passed
```

The final local PR Fast result on the implementation commit was:

```text
1559 passed, 5664 deselected in 190.34s (0:03:10)
```

The required GitHub Full CI result was:

```text
run: 33310620640
tested HEAD: 1b86a439d42658cc69beb15888d2edddf08aac51
compile and shard policy: PASS
weighted shard 0: PASS
weighted shard 1: PASS
weighted shard 2: PASS
weighted shard 3: PASS
```

Full CI is therefore PASS for the exact implementation/test HEAD.  A later
report-only commit, if present, must not be treated as the executable/test
commit validated by that run.

The final executable/test evidence HEAD after the implementation was:

```text
a1795b3054ad7c3ecae0f7e6b45c4b5826bc79f3
```

The GitHub checks for that evidence HEAD are:

```text
PR Fast workflow: 33312183371 — PASS
  compile and diff: PASS
  pytest (PR fast): PASS
CodeQL workflow: 33312181638 — PASS
  Analyze (actions): PASS
  Analyze (javascript-typescript): PASS
  Analyze (python): PASS
```

The subsequent report synchronization commit is documentation-only and does
not alter the executable or test tree.  Full CI remains the authoritative
complete-suite result for implementation/test HEAD
`1b86a439d42658cc69beb15888d2edddf08aac51`.

## Known limitations and next milestone dependencies

CORE-02 uses deterministic host-local tools in tests only.  It does not add an
LLM provider, network/acquisition access, shell or subprocess authority,
remote/GPU execution, document parsing, OLED logic, BR1, observability, UI,
API, scheduler, or automatic interrupted-call replay.  There is no new
scientific error-propagation model.

CORE-03 must separately introduce the conservative acquisition contracts and
security implementation already frozen by the readiness evidence.  It must
continue to use `AgentLoop`, `ToolRegistry`, `ToolPolicy`, and the CORE-01
stores without adding a second authority or exposing credentials to a model.

CORE-02 does not establish BR1 v2 parity.  B2 = `PENDING`, B3 = `PENDING`, and
B4 = `PENDING`.  `core_cutover_ready = false`.  The immutable v1 tag
`molly-v1-pre-core-v2-20260829` and branch `legacy/molly-v1` remain bound to
`ae7892dbf8a6bfe85dd909056eadc2afecc40d9`.
