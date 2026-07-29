# Scientific agent trajectory inspection v1

## Scope and task mapping

`scientific_agent_trajectory_inspection.v1` completes the implementation and
test evidence for `M3-023` through `M3-028`:

- `M3-023`: project-scoped read-only inspection API;
- `M3-024`: deterministic allowlisted queries and bounded result pages;
- `M3-025`: minimal timeline in canonical PR-BD event order;
- `M3-026`: source-backed evidence references;
- `M3-027`: only alternatives actually persisted by observer sources;
- `M3-028`: sensitive-field, path, and infrastructure-information controls.

This is an observer view. It is not a publication, scientific trust anchor,
cache, search index, or control-plane input.

## Verified source chain

The view consumes exactly this chain:

```text
scientific_agent_trajectory_projection.v1 (PR-BD / PR-BE exact replay)
  -> scientific_agent_trajectory_audit_metrics.v1 (PR-BF exact replay)
  -> scientific_agent_failure_attribution.v1 (PR-BG exact replay)
  -> ephemeral scientific_agent_trajectory_inspection.v1 response
```

PR-BG exposes one internal context-bound seam. It nests the existing PR-BF
and PR-BE seams, pins all three publication directory inodes, and yields
read-only byte mappings. The inspection payload is fully constructed inside
that context. On context exit, each existing verifier repeats its source
stability checks. Named-file replacement, directory replacement, inode
replacement, roster change, byte change, source-binding mismatch, or failed
exact replay prevents the HTTP response from returning a partial timeline.

No source file is reopened by path after a verifier context closes. The
consumer does not duplicate or weaken the PR-BE, PR-BF, or PR-BG verifier.

## HTTP contract

```text
GET /api/projects/<project_id>/oled-bounded-sessions/<session_id>/trajectory-inspect
```

Required query parameters are:

```text
trajectory_publication_id
audit_publication_id
attribution_publication_id
```

The client supplies stable IDs only. The server constructs fixed paths below
the selected project and obtains `actions_root` from
`OledBoundedDiscoverySessionActionService`. Publication IDs require the exact
versioned prefix and a 64-character lowercase hexadecimal identity. Repeated
parameters, separators, whitespace, traversal, encoded traversal, arbitrary
paths, output roots, and runtime locators are rejected.

All responses set `Cache-Control: no-store`. Inspection does not persist the
response or a derived identifier.

## Response schema

The stable top-level fields are:

```text
ok
inspection_version
project_id
session_id
verified_chain
summary
applied_filters
page
timeline
unattached_findings
alternatives
claims
```

`verified_chain` binds trajectory, audit, and attribution IDs and publication
IDs. `summary` is computed from the complete verified source and does not
change under filtering. `page` distinguishes `returned_count`,
`total_matching_count`, and `truncated`. The claims explicitly state that the
response is read-only, observer-only, does not modify scientific execution,
offers no control action, makes no scientific-validation claim, and includes
no private chain of thought.

## Deterministic timeline and joins

`events.jsonl` is the sole timeline skeleton. Its persisted `sequence_index`
must be contiguous and is already derived by PR-BD from the frozen order:

```text
Session revision -> event-kind order -> stable source/event ID
```

Inspection never sorts by timestamp, filesystem order, map order, or mutable
telemetry. It does not infer a general causal graph.

Failure attribution joins first through `affected.event_id`, then through a
unique persisted `source_refs.record_id`. Audit findings join only through a
unique source record ID. Findings without a unique event anchor remain in
`unattached_findings`. Telemetry can join through its persisted action ID but
is always labelled `non_authoritative_telemetry`; it never overrides Session,
publication, audit, or attribution facts.

The response preserves `first_cause`, `downstream_symptom`, `determined`, and
`undetermined` values. Attribution-level ambiguity remains available as
`multiple_equal_first_cause_candidates`; evidence gaps remain
`insufficient_causal_evidence` or `causal_link_not_proven`. The UI never
renders an undetermined result as a known cause.

## Safe outcome and evidence fields

Timeline outcomes use this fixed allowlist only:

```text
status
current_step
approved
gate
task_id
has_complete_top_n
stop_reason
selected_candidate_count
```

The field name allowlist is not sufficient by itself. String values for
`status`, `current_step`, `gate`, `task_id`, `stop_reason`, and projected
`reason_codes` must also belong to the frozen M2/M3 semantic enums. An unknown
safe-looking identifier is omitted; it is never exposed merely because it
matches an identifier regular expression. This includes hostname-shaped
values such as dotted names and internal node labels.

Evidence uses only persisted safe reference fields:

```text
artifact_name
sha256
record_id
record_digest
source_binding_sha256
logical_role
```

The API does not pass through complete outcome objects, audit details,
exceptions, commands, environments, remote metadata, mutable locators, or
source filesystem paths.

## Filter allowlist

Filters execute only after all sources have passed exact replay:

- `event_kind`: a frozen PR-BD event kind;
- `taxonomy_family`: one of the nine PR-BG families;
- `finding_code`: one of the five PR-BG public codes;
- `attribution_role`: `first_cause` or `downstream_symptom`;
- `attribution_status`: `determined`, `undetermined`, or `no_failure`;
- `source_artifact`: a fixed observer artifact name;
- `limit`: canonical integer, default `200`, maximum `500`.

Filters use exact equality. Regular expressions, arbitrary sorting, unknown
fields, and free-form source names are forbidden. Limit truncation affects the
returned page only, not `verified_chain` or the total summary.

`attribution_status=no_failure` is a publication-level predicate. It matches
only when `attribution_manifest.result.attribution_status` is exactly
`no_failure`; it does not mean that an individual timeline event happens to
have no attached failure finding. A determined or undetermined failure
publication therefore returns zero matching items for this filter.

## Alternatives policy

Projection, audit, and attribution v1 do not persist real alternatives.
Therefore v1 returns:

```json
{"available":false,"items":[],"reason":"source_observer_publications_do_not_persist_alternatives"}
```

This is the correct result. Inspection does not use an LLM, action vocabulary,
fallback behavior, retry behavior, or current state to invent counterfactuals.
A future schema can expose alternatives only after an upstream observer
contract durably persists and source-binds them.

## Sensitive-field policy

Long-lived or returned inspection identity never contains an absolute path,
workspace path, actions root, known-hosts bytes or path, SSH alias, hostname,
username, IP address, email-like account, remote repository path, interpreter
path, raw exception, shell command, credential, API key, or private paper path.
The builder uses both field and semantic allowlists rather than redaction or
format-only identifier checks. The UI renders all dynamic inspection text with
`textContent` and does not insert source content through `innerHTML`.

## Error contract

The API returns fixed, privacy-safe codes and never `str(exc)`:

```text
400 invalid_inspection_request
400 inspection_response_limit_exceeded
404 observer_publication_unavailable
409 observer_publication_chain_mismatch
409 observer_publication_integrity_failure
```

Internal logs may retain an exception for local diagnosis; the browser sees
only the fixed public message. No error path returns a partial response.

## Observer-only proof and claim boundary

Inspection performs no POST, PUT, PATCH, or DELETE operation. It does not
create, advance, approve, reject, recover, retry, or dispatch a Session or
action. It does not update Session revisions, StageState, action telemetry,
Gate snapshots, approval decisions, scientific child publications, Artifact
Registry, trajectory, audit, or attribution publications. It registers no
scientific artifact and writes no cache.

Tests compare the complete source workspace tree before and after repeated API
inspection, exercise context mapping immutability and source replacement, and
reuse the full PR-BE/PR-BF/PR-BG adversarial verifier suites. The UI states the
claim boundary explicitly:

> 该视图是 observer-only 审计结果，不代表实验或高保真计算验证。

M3 remains `I/T/—` until a separate representative evidence PR exercises one
single-round success, one multi-round success, and at least one real or
representative failure through this same API in a fresh process and receives
human review. M4 may evaluate attribution accuracy only after that evidence.

## Versioning and rollback

Changing field meaning, join authority, filter semantics, ordering, or privacy
policy requires a new inspection version. Additive implementation fixes that
preserve this contract remain v1. Because v1 is ephemeral and observer-only,
ordinary code revert removes it without Session, storage, registry, or
scientific-publication migration.
