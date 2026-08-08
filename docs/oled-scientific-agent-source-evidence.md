# Authoritative scientific-agent source evidence v1

This document freezes the narrow PR-BJ source contract that unblocks the
representative M3 validation cases. It does not make validation claims;
[`roadmap.md`](roadmap.md) remains authoritative for evidence maturity and
execution order.

## Compatibility decision

PR-BJ retains the existing observer contract versions:

```text
scientific_agent_trajectory_projection.v1
scientific_agent_trajectory_audit_metrics.v1
scientific_agent_failure_attribution.v1
scientific_agent_trajectory_inspection.v1
```

The compatibility matrix is frozen before implementation:

| Compatibility condition | Decision | Required proof |
|---|---|---|
| No new trajectory event kind | retain v1 | only existing `task_dispatched` and `stage_failed` events are emitted |
| No new event top-level field | retain v1 | source-backed data is limited to existing `reason_codes`, `source`, and `outcome` members |
| Canonical JSON/JSONL unchanged | retain v1 | existing serializers and event envelope remain unchanged |
| New information is opt-in | retain v1 | only explicit versioned failure evidence or immutable receipts activate the new path |
| Previously publishable legacy source behavior unchanged | retain v1 | frozen legacy single-round, multi-round, action-telemetry, and recovered-action sources replay byte-for-byte |
| Existing v1 publications remain verifiable | retain v1 | the external verifier rebuilds legacy publications to the same IDs and bytes |
| Any legacy byte drift or required top-level field | stop | do not silently revise v1; document and implement side-by-side v1/v2 compatibility instead |

This decision is conditional on the required byte-equality tests. A passing
behavioral suite alone is not sufficient evidence of compatibility.

At baseline `b2c1cbe77faca4309350ee25531ed84ed5fe0a9e`, PR-BD could not
publish a terminal `FAILED` Session because it unconditionally required
`session_result.json`, while PR-AV correctly forbids that file for `FAILED`.
PR-BJ therefore has no pre-existing generic-failure publication bytes to
preserve. Accepting a validated failed Session is an opt-in input expansion;
it does not rewrite an old publication or alter any previously accepted input.

## Source contracts

The source layer separates four immutable contracts:

- `scientific_agent_failure_source_evidence.v1` records bounded, typed failure
  reason codes in `StageState.details.failure_evidence` only when the facts are
  known. Historical StageState JSON gains no automatic null field.
- `scientific_agent_dispatch_receipt.v1` records an actual dispatch-boundary
  observation under the child run. It distinguishes real dispatch from
  duplicate rejection, idempotent replay, and recovery adoption.
- `scientific_agent_dispatch_authority.v1` is published atomically beside each
  dispatch receipt. The receipt binds the exact `authority.json` SHA-256, while
  terminal `StageState.details.dispatch_authority_roster` freezes the complete
  projectable receipt/authority roster. PR-BD recomputes both files and rejects
  an otherwise self-consistent appended receipt that is absent from that
  terminal authority roster.
- `scientific_agent_recovery_receipt.v1` records only the adoption of an
  already-completed child by PR-AW. It is not evidence of a new dispatch.

The causal-link sub-contract is
`scientific_agent_failure_causal_link.v1`. It binds a symptom to an explicit
cause child run; timestamp or adjacent revision is never a causal link.

The v1 failure reason allowlist is:

```text
known_hosts_verification_failed
ssh_connection_failed
remote_endpoint_verification_failed
remote_output_retrieval_failed
scp_transfer_failed
gate_snapshot_mismatch
authorization_mismatch
tool_runtime_failure
adapter_runtime_failed
output_parse_failed
duplicate_dispatch_detected
reconciliation_failed
stale_ownership_detected
stale_state_detected
```

Taxonomy families are deliberately absent. PR-BG derives a family only from
these semantic codes after exact replay.

## Authority matrix

| Fact | Authoritative source |
|---|---|
| failure reason | validated `StageState.details.failure_evidence` |
| dispatch attempt | immutable dispatch authority plus its exact-bound receipt and terminal StageState roster |
| duplicate dispatch | two distinct authority-bound receipts in the terminal roster, with the latter explicitly marked duplicate |
| successful recovery | immutable recovery receipt |
| causal link | validated failure evidence with an explicit typed link; a recovery receipt alone cannot invent one |
| mutable action status | telemetry only |

PR-BD remains post-hoc and observer-only. Executor, adapter, coordinator, and
action-service code may record the fact at the boundary where it becomes
known, but none of them may invoke PR-BD, PR-BF, PR-BG, or inspection.

Dispatch ordinal allocation is serialized by a per-run cross-process lock.
Roster verification, ordinal allocation, predecessor selection, authority
construction, and no-replace publication are one critical section. The lock
file contains no identity or infrastructure data. `initial` versus `retry` is
also selected within that critical section; `idempotent_replay` and
`recovery_adoption` remain non-dispatch facts.

A recovery receipt freezes only the projectable dispatch-authority roster:
`initial`, `retry`, and `duplicate_rejected`. Later `idempotent_replay` or
`recovery_adoption` receipts do not invalidate an already-proven recovery,
while any later projectable authority changes that roster and fails closed.

PR-AW recovery also covers the crash window where reconciliation committed
`expected_revision + 1` but the recovery receipt was not yet published. A
later invocation exact-replays the predecessor and current Session revisions,
requires the same single completed child transition and verified StageState,
then deterministically reconstructs the same receipt without invoking
reconciliation or scientific execution again. Other revision shapes fail
closed.

## Privacy boundary

Source evidence stores semantic reason codes and opaque content-bound IDs. It
must not store or infer from raw exception text, stderr, commands, environment
variables, paths, hostnames, IP addresses, usernames, email accounts, SSH
aliases, known-hosts locations or bytes, credentials, or private scientific
text. Unknown codes fail validation rather than being exposed because they
look syntactically safe.

## Versioning boundary

Adding a source reason code requires a reviewed allowlist change and tests.
Changing dispatch or recovery semantics, adding a trajectory event kind or
top-level field, or changing legacy projection bytes requires a new version.
The source contracts are execution facts, not new scientific trust anchors.

## Legacy byte-compatibility evidence

The baseline commit and current implementation projected the same source trees
independently. `cmp` reported exact equality for all four artifacts and both
sides produced the same publication ID.

| Legacy case | Publication ID | `events.jsonl` | `source_bindings.json` | `telemetry_findings.jsonl` | `trajectory.json` |
|---|---|---|---|---|---|
| single round | `scientific-agent-trajectory-publication:0dd5fa85027c79c52ba10495c6c14ff1e3f15d8007bd75a802e09df77bdc80c6` | `a09b197a…49d2f` | `9994340b…e2852` | `e3b0c442…b855` | `bbf1a399…e41d7` |
| multi round | `scientific-agent-trajectory-publication:99ff714b67ed82617cfa55980ff1e93ccba3afe9bdb95d59a582776b1ee51050` | `2bf45f72…34855` | `4e00dfea…09df` | `e3b0c442…b855` | `8d81581b…7ca2` |
| recovered action telemetry | `scientific-agent-trajectory-publication:3c5887425fa4ca4ee1a039461f55d33e87732317f7b21f5bd4010d591aa599d3` | `7d84a7b0…eeaf1` | `ffca56d4…10c9fc` | `e3b0c442…b855` | `04defdce…a250` |

The baseline single-round PR-BF and PR-BG publications were then consumed by
the new nested verifier and passed exact external replay. No raw source bundle,
local path, user name, hostname, or temporary locator is committed as evidence.

The final local targeted validation comprised:

```text
source + PR-BD/BE/BF/BG + PR-AW: 134 passed
RunPlanExecutor + docs/privacy:      89 passed
PR Fast:                            900 passed, 5,241 deselected
```

The source suite includes a barrier-released two-process dispatch test, a
fully re-signed appended-receipt rejection test, and a recovery test that
crashes after Session commit but before receipt publication. The latter
proves the next invocation reconstructs the receipt without a second
reconciliation call.

These are implementation and test results (`I/T/—`), not representative
runtime validation. PR #10 remains Draft, M3 remains without `V`, and M4 stays
locked until PR-BI is rebased, regenerates all eight cases, and receives owner
review.
