# OLED reviewed-evidence staging preflight (PR-R)

## Purpose

PR-R is the read-only boundary between PR-Q observation candidates and a future
append-only reviewed-evidence ledger. It consumes one exact PR-Q artifact plus
one immutable ledger snapshot, classifies every candidate, and emits a
deterministic staging plan.

It does not stage reviewed evidence, mutate the ledger, assign a confidence
score, create Gold, or write dataset/training rows.

## Three distinct data layers

PR-R keeps three concepts separate:

1. an immutable source claim records what one exact source cell reported;
2. a versioned semantic projection records how the current Registry, ontology,
   unit rules, and comparison-context policy interpret that source claim; and
3. a future Gold/training view may select eligible projections for one defined
   task.

An incomplete or conflicting source claim remains preservable and queryable.
Quarantine changes its comparison/admission status; it does not erase the
reported evidence.

## Exact inputs

The file entry requires two distinct regular JSON files:

1. `oled_observation_materialization_candidate.v1`; and
2. `oled_reviewed_evidence_ledger_snapshot.v1`.

The output records the exact SHA-256 of both input files and embeds both
validated models. It also revalidates the semantic artifact digests carried by
the embedded models. Standalone model validation cannot recover the original
external bytes, so
`standalone_input_bytes_revalidation_supported=false` remains explicit.

The preflight timestamp must not predate either PR-Q or the ledger snapshot.

## Stable identity

PR-Q `candidate_id` remains a local source-cell locator. It is not used as the
global ledger key.

PR-R derives separate identifiers:

- `source_claim_id` binds the source PDF SHA-256 and source-cell digest;
- `projection_id` additionally binds the PR-Q candidate digest, selected
  Registry entity, Registry entry digest, reviewed cell disposition, and
  semantic-contract digest; and
- `conflict_key` binds material ID, property ID, causal layer, and normalized
  comparison context.

The same source under a changed identity or semantic mapping therefore becomes
a proposed revision, not an in-place update. Two papers that report the same
value remain two source claims even when they are consistent duplicates.

## Semantic-contract snapshot

Each preflight embeds and hashes the exact:

- property ontology definitions;
- representation contract;
- property and condition unit-normalization rules; and
- photophysical comparison-context fields and policy.

This is the first downstream point that explicitly versions the semantic
projection. A later ontology change should produce a new projection rather
than rewrite the immutable source claim.

## Classification

Each candidate receives one disposition:

- `new_claim_ready`: no prior live matching claim or conflict key;
- `exact_replay`: the exact projection already exists, so the future writer
  must perform a no-op;
- `consistent_duplicate_ready`: another live source claim reports the same
  normalized value and unit under the same conflict key; both claims remain;
- `value_conflict_quarantine`: a comparable live claim reports another value;
- `incomplete_context_quarantine`: required comparison context is missing;
- `revision_requires_review`: the same immutable source claim already has a
  different live semantic projection; or
- `semantic_contract_migration_required`: a potentially comparable live claim
  uses another pinned semantic contract and must not be compared silently.

Clean claims do not require another human review. Value conflicts and revisions
require a later exception decision. Incomplete-context claims are staged only
to quarantine and remain `comparison_ready=false`.

## Source-row grouping

Observation candidates remain cell-level, but PR-R groups them by the exact
PR-P staging item, identity group, Registry material, source PDF, table, and
row. It never groups solely by alias or material ID. This prevents the five
properties from one paper016-shaped source row from becoming five unrelated
record-level objects.

## Confidence and Gold boundary

PR-R records verified facets for exact source binding, transcription review,
property mapping, Registry identity, and comparison-context assessment. It
does not convert those facets into a probability or invent the old fallback
confidence value `0.5`.

Every item therefore retains at least these Gold blockers:

- `missing_confidence_assessment`; and
- `scientific_consistency_not_reviewed`.

Conflict, revision, and incomplete-context blockers are added when applicable.
These flags describe what has and has not been reviewed; they do not claim that
the reported scientific value is physically correct.

## Device-only boundary

The ledger snapshot model and PR-R derivation both reject device-layer entries
or candidates. `device_only_cell_count=0` is a model invariant, not only an
upstream convention.

## Controlled workflow

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_reviewed_evidence_staging_preflight \
  --materialization-candidates /operator/local/pr-q-candidates.json \
  --ledger-snapshot /operator/local/reviewed-evidence-ledger-snapshot.json \
  --output /operator/local/reviewed-evidence-staging-preflight.json
```

The output must be fresh and distinct from both inputs. Symbolic input paths,
input overwrite, duplicate-key or non-finite JSON, timestamp reversal, changed
output parents, and partial publication fail closed. CLI failures emit only a
stable redacted error object.

## paper016-shaped automated boundary

The existing exact-chain fixture produces:

```text
5 source observation candidates
1 exact source-row group
5 new claims ready against the genesis ledger
14 ontology-review-pending cells still excluded upstream
0 device-only cells admitted
```

The tests also cover exact replay, consistent cross-source duplicates, value
conflicts, source-claim revision, candidate-ID collision, incomplete context,
semantic-contract migration, semantic tamper, timestamp reversal, input
overwrite, redaction, and output-parent replacement.

This remains fixture-level validation. A real paper016 operator artifact and a
multi-paper canary are later acceptance requirements.

## Next boundary

The next safe step is a compare-and-swap, append-only reviewed-evidence ledger
writer. It should consume the exact PR-R file and exact prior snapshot, make
exact replay a no-op, publish quarantine entries without activating them for
comparison, and refuse revisions without a roster-bound human exception
decision. Gold conversion remains later.
