# Scientific agent trajectory audit metrics v1

This document defines the durable PR-BF artifact and metric contract. It does
not define roadmap status; [`../todo.md`](../todo.md) remains authoritative for
scope, evidence maturity, priority, and execution order.

## Authority boundary

The publisher accepts a trajectory publication only through PR-BE's
context-bound verifier. PR-BE externally rebuilds PR-BD, compares the complete
file roster and every byte, holds the verified publication directory open, and
checks the same inode and bytes again after the metrics consumer returns.

PR-BF computes entirely from the read-only byte mapping yielded inside that
context. It does not:

- re-open a projection artifact by path;
- implement a second projection or external verifier;
- inspect the original Session, action, gate, StageState, or scientific
  publication while calculating metrics;
- write to the Session, control plane, projection, or scientific publication;
- register the audit as scientific evidence or a scientific trust anchor.

All metric bytes are prepared while the PR-BE directory is pinned. The output
root is not created until PR-BE's post-consumer stability check succeeds.

## Publication contract

One deterministic publication directory contains exactly:

```text
<audit-publication-id>/
├── audit_metrics.json
├── audit_findings.jsonl
├── source_binding.json
├── audit_manifest.json
└── report.md
```

The publisher writes a complete temporary directory and atomically commits it
without replacing an existing name. Re-publishing the same ID fails instead of
overwriting the prior bytes.

`source_binding.json` binds the audit to:

- the PR-BE-verified trajectory ID;
- the PR-BE-verified projection publication ID;
- the SHA-256 of all four exact projection payloads;
- the context-bound, exact-roster, exact-byte, external-replay verification
  claims.

The audit ID is a stable hash over the metrics version and those source
identities and digests. The publication ID additionally binds the exact bytes
of metrics, findings, source binding, and report. `audit_manifest.json` records
those digests without attempting a circular self-digest.

JSON is UTF-8, Unicode NFC, sorted-key, finite integer/boolean/string/null data
with a final newline. JSONL uses one compact sorted-key object per line and a
final newline when non-empty. The report contains no local path or timestamp,
so identical verified bytes produce identical output across processes.

## Provenance coverage

Coverage is event-level. An event is covered only when its complete `source`
object exactly matches one entry in `source_bindings.json`. Each category
reports eligible and covered counts, integer basis points, and one of
`complete`, `partial`, `none`, or `not_applicable`.

| Category | v1 eligible events |
|---|---|
| action | `action_requested`, `task_dispatched`, `stage_completed`, `stage_failed` |
| evidence | `publication_verified` |
| authorization | `action_authorized` |
| observation-to-decision | non-initial `state_committed` events (the frozen v1 committed-state transition proxy) |
| recovery | `state_committed` events explicitly carrying `RECOVERY_REQUIRED` |
| terminal | `terminal_result_committed`; a failure-form projection without a result uses its final matching terminal `state_committed` event |

A zero denominator is `not_applicable`, not 100 percent. Missing bindings lower
coverage and create findings; they never cause the auditor to fabricate a
replacement source.

## Deterministic metric groups

`audit_metrics.json` freezes these v1 groups:

- trajectory length: event and distinct Session revision counts;
- action outcome: immutable request, dispatch, success, failure,
  integrity-failure, and unresolved child counts;
- tool failure: projected `stage_failed` count, child IDs, and persisted reason
  codes;
- retry: repeated dispatches of the same child run ID;
- reconciliation: only explicit `RECOVERY_REQUIRED` state markers, with an
  `inferred_reconciliation_count` fixed to zero;
- gate: authorization, approved, rejected, and unknown counts;
- budget consumption: exact non-negative `iterations`, `generation_rounds`,
  and `generated_candidates` from the terminal event;
- Top-N completion: the persisted terminal boolean when applicable;
- bounded-search correct stop: exact consistency between terminal status,
  completion boolean, and the frozen PR-AU stop-reason set.

Two requested concepts are intentionally represented without invented values:

- latency is `unavailable` because projection v1 has no wall-clock event fields;
- wasted computation is `not_derivable` because projection v1 does not prove
  cost or whether a failed computation was reused. The metric may expose the
  factual failed-child count but never equates it with waste.

Changing eligibility, denominators, stop-reason semantics, serialization, or
availability semantics requires a new metrics version.

## Findings

`audit_findings.jsonl` is limited to deterministic inconsistencies in the
consumed bytes, such as:

- receipt/artifact digest or count mismatch;
- source-manifest or trajectory identity mismatch;
- duplicate event or source-binding identity;
- event sequence mismatch;
- an event whose exact source binding is absent;
- missing, duplicate, or status-inconsistent terminal anchors.

Every finding contains only a reason code, bounded structural details, and
source references consisting of projection artifact name, exact artifact
SHA-256, and an event record ID when available. Findings explicitly make no
root-cause claim and affect audit metrics only. Root-cause taxonomy and
first-cause attribution are outside this contract.

## Public entry point

```python
from ai4s_agent.oled_scientific_agent_trajectory_audit_metrics import (
    publish_oled_scientific_agent_trajectory_audit_metrics,
)

result = publish_oled_scientific_agent_trajectory_audit_metrics(
    storage=storage,
    project_id=project_id,
    session_id=session_id,
    actions_root=actions_root,
    trajectory_publication_dir=verified_projection_dir,
    output_root=audit_root,  # optional
)
```

The default output root is the project's `trajectory-audits/` directory.
Source/output overlap and symlink-component redirection are rejected.

## Non-goals

v1 does not perform failure root-cause inference, LLM review, counterfactual
generation, inspect API or timeline UI, recovery automation, Session control,
or scientific validity assessment.
