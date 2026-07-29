# Scientific agent failure taxonomy and first-cause attribution v1

This document freezes the PR-BG contract for tasks `M3-018` through `M3-022`.
It extends the observer-only trajectory audit; it does not change scientific
execution, recovery, authorization, or roadmap authority. [`../todo.md`](../todo.md)
remains the sole source for milestone status and execution order.

## Authority and observer-only boundary

Attribution consumes two sources in one nested verified-byte context:

1. PR-BE externally rebuilds the trajectory projection, compares its complete
   roster and exact bytes, and pins the accepted directory inode.
2. While those projection bytes remain pinned, PR-BF is rebuilt from the same
   mapping. Its complete publication is exact-compared and pinned.
3. PR-BG computes only from the two read-only byte mappings yielded inside that
   context.
4. PR-BF and PR-BE repeat their inode, named-directory, roster, and byte checks
   after the PR-BG consumer returns. Existing exception chaining is preserved
   when both the consumer and a source stability check fail.
5. Attribution output is not created until both post-consumer checks succeed.

PR-BG never reopens a source by an ordinary path after verification. It does
not write Session revisions, StageState, action telemetry, Gate decisions,
PR-AU decisions, Artifact Registry entries, trajectory projections, audit
metrics, or scientific publications. It triggers no retry, recovery, dispatch,
approval, or other control action. The publication is not a scientific trust
anchor.

## Frozen taxonomy

Taxonomy version: `scientific_agent_failure_taxonomy.v1`.

| Family | Required evidence and meaning | Boundary |
|---|---|---|
| `input_integrity` | A bound scientific input identity, source binding, manifest, roster, path scope, or digest is invalid before or during execution. | Projection or audit corruption is `audit_integrity`; a valid-input execution exception is `tool_runtime`. |
| `authorization_mismatch` | A persisted Gate, approval snapshot, controller, predecessor, actor, or authorization-scope mismatch. | An explicit policy rejection is `policy_constraint`; remote endpoint identity is `transport`. |
| `transport` | A persisted known-hosts, expected-hostname, SSH/SCP, transfer, endpoint-verification, or remote-output-retrieval reason. | A remote program failure after transport succeeds is `tool_runtime`. Runtime locators are never published. |
| `tool_runtime` | A persisted invocation, execution, adapter, or output-parsing failure after valid input and authorization. | A generic `stage_failed` is insufficient to claim this as a first cause. |
| `model_inadequacy` | Explicit persisted applicability or capability evidence shows that the model is inadequate. | Missing Top-N alone never proves model inadequacy; candidate scarcity is `candidate_supply`. |
| `candidate_supply` | Explicit persisted evidence shows that the legal property-qualified pool cannot supply the target Top-N. | The claim is bounded to the observed search and never states that chemical space has no solution. A budget stop is `policy_constraint`. |
| `policy_constraint` | An explicit hard constraint, bounded-search rule, or frozen budget prevents an action or complete Top-N. | Usage alone does not prove a budget limit without a persisted bound or stop reason. |
| `recovery` | Persisted recovery, reconciliation, distinct duplicate-dispatch, stale ownership, or interrupted-action evidence. | Successful recovery is not automatically a failure. Mutable telemetry cannot override typed Session authority. |
| `audit_integrity` | Projection, audit, source roster, history, serialization, or attribution publication integrity is invalid. | It must not be presented as a scientific, model, or candidate failure. |

Every family has a stable ID, evidence requirements, prohibited uses, adjacent
boundary, and explicit first-cause/symptom permissions in
`failure_taxonomy.json`. Changing a family, boundary, or mapping requires a new
taxonomy version.

## Finding-code boundary

External findings use only this allowlist:

```text
BOUNDED_SEARCH_NO_COMPLETE_TOP_N
MODEL_INADEQUACY_DETECTED
BUDGET_LIMIT_REACHED
REVIEW_RECOMMENDED
INTEGRITY_FAILURE
```

Taxonomy family and finding code are independent dimensions. Families classify
the evidence domain; finding codes state the bounded public conclusion.

| Family | Allowed public codes |
|---|---|
| `input_integrity` | `INTEGRITY_FAILURE`, `REVIEW_RECOMMENDED` |
| `authorization_mismatch` | `REVIEW_RECOMMENDED`, `INTEGRITY_FAILURE` |
| `transport` | `REVIEW_RECOMMENDED` |
| `tool_runtime` | `REVIEW_RECOMMENDED` |
| `model_inadequacy` | `MODEL_INADEQUACY_DETECTED`, `REVIEW_RECOMMENDED` |
| `candidate_supply` | `BOUNDED_SEARCH_NO_COMPLETE_TOP_N`, `REVIEW_RECOMMENDED` |
| `policy_constraint` | `BOUNDED_SEARCH_NO_COMPLETE_TOP_N`, `BUDGET_LIMIT_REACHED`, `REVIEW_RECOMMENDED` |
| `recovery` | `REVIEW_RECOMMENDED`, `INTEGRITY_FAILURE` |
| `audit_integrity` | `INTEGRITY_FAILURE` |

Findings are read-only observations. No code in this set authorizes a control
action.

## First cause, symptom, and sufficiency

A `first_cause` requires all of the following:

- a concrete source reference and exact source SHA-256;
- an explicit frozen reason or structural fact sufficient for its family;
- a position in canonical causal order before any linked symptom;
- a typed-authority link, such as the same immutable child ID or the Session's
  terminal result following the bound child failure;
- no dependence on mutable telemetry overriding an authoritative source.

A `downstream_symptom` is a persisted later manifestation whose link is proven,
such as the terminal failure following a source-bound child failure. Temporal
adjacency alone is not a causal link. An unrelated later observation remains
`undetermined`, even when another primary first cause exists.

At most one primary first cause is published. Candidates at the same earliest
Session revision are equal causal candidates: event-kind or lexicographic order
must not break the tie. The result is deterministically
`multiple_equal_first_cause_candidates`, no primary is selected, and public
findings request review. If no sufficiently evidenced candidate exists, the
result is `insufficient_causal_evidence`. It uses `REVIEW_RECOMMENDED`, except
that the factual bounded-search symptom may retain
`BOUNDED_SEARCH_NO_COMPLETE_TOP_N` without claiming a cause.

Clean successful trajectories publish `no_failure` and an empty JSONL file.
An explicitly recovered trajectory that completes Top-N does not acquire a
failure solely because an intermediate authoritative state recorded recovery
required.

## Canonical order and determinism

PR-BG follows the frozen M2 order:

```text
Session revision
→ canonical event sequence/event-kind order
→ stable source or event ID
```

Timestamps never select or break a causal tie. Publication identity uses only
schema versions, verified trajectory/audit identities, exact source artifact
digests, and canonical output bytes. JSON is normalized UTF-8, sorted-key,
finite integer/boolean/string/null data with a final newline. JSONL is one
compact sorted-key object per line with a final newline when non-empty. Floats,
NaN, and Infinity are rejected by the shared canonical serializer.

Unordered Python mapping construction and internal observation input order do
not affect output. Serialized projection and audit roster order is already part
of the upstream verified-byte identity; PR-BG does not reinterpret or weaken
that contract. LLMs, randomness, current time, wall-clock ordering, filesystem
enumeration order, mutable environment data, and private reasoning are absent.

## Source-backed record

Each attribution contains:

- taxonomy family and taxonomy version;
- `first_cause` or `downstream_symptom` role;
- one allowlisted finding code;
- safe affected event/action/child/stage identifiers when available;
- Session revision and event kind when available;
- source artifact name, exact source SHA-256, safe record ID, and record digest;
- a frozen deterministic reason code;
- `sufficient` or `insufficient` evidence status;
- `determined` or `undetermined` attribution status;
- a fixed non-private rationale summary.

Unsafe identifiers containing paths, whitespace, email-style runtime accounts,
or path separators are omitted. Raw known-hosts content, expected hostname,
remote endpoint, username, absolute path, local temporary path, timestamp, and
exception text are never copied into long-term attribution bytes. Their exact
verified source record remains bound through a digest.

## Immutable publication and exact replay

Schema version: `scientific_agent_failure_attribution.v1`.

```text
<publication-id>/
├── failure_taxonomy.json
├── failure_attributions.jsonl
├── source_binding.json
├── attribution_manifest.json
└── report.md
```

The fixed roster is assembled in a complete temporary directory and committed
atomically without replacement. Reusing an existing publication ID fails
closed. `source_binding.json` binds every exact PR-BE and PR-BF source artifact
digest. `attribution_manifest.json` binds every non-manifest attribution
artifact. The attribution ID binds both verified source publications and both
complete source rosters; the publication ID additionally binds all generated
bytes.

The verifier repeats PR-BE and PR-BF exact replay while both sources are pinned,
rebuilds PR-BG, pins the attribution directory, and compares its directory
identity, complete roster, and every byte. Modified family/code mappings,
source-reference rebinding, first-cause/symptom inversion, roster changes,
content replacement, and fully re-signed manifests therefore fail closed.

## Frozen standard cases

- **known-hosts propagation:** an explicit transport verification reason is the
  first cause; later stage/terminal failure is separated. No runtime locator or
  raw known-hosts content is copied.
- **history truncation:** structural audit findings produce
  `audit_integrity / INTEGRITY_FAILURE`; no scientific cause is inferred.
  External replay rejects a fully re-signed missing event before publication.
- **duplicate dispatch:** only distinct persisted dispatch records plus the
  explicit duplicate reason are sufficient. Exact duplicate event replay,
  recovery adoption, terminal replay, and repeated dispatch without execution
  proof are not labeled as duplicate computation.
- **stale state:** typed Session authority wins. Mutable telemetry conflict is a
  recovery-layer symptom and cannot replace a source-backed tool/runtime first
  cause or modify the scientific result.

The fixtures freeze exact input bytes, expected family/code, primary result,
symptoms, source bindings, sufficiency, and ambiguity. Fixture names do not
drive classification.

## Claim boundary and versioning

Attribution v1 classifies deterministic evidence patterns. It does not measure
classification accuracy, scientific validity, model quality, recovery utility,
or counterfactual correctness. Projection v1 can legitimately lack causal
evidence; `undetermined` is an expected result, not an implementation failure.
M4 reviewed benchmarks are required before making accuracy claims.

Any change to taxonomy families, public code mappings, causal sufficiency,
tie-breaking, source authority, artifact roster, serialization, or identity
inputs requires a new version. Readers must continue to treat v1 publications
as immutable historical observer artifacts.

## Non-goals

PR-BG does not add the PR-BH inspect API or timeline UI, an LLM auditor, Critic
Agent, counterfactuals, adaptive planning, recovery automation, retries, Gate
decisions, scientific models, candidate sources, benchmark labels, storage
migration, Session schema changes, projection v2, or scientific/control-plane
refactoring.
