# OLED categorical Gold successor preflight (PR-AE)

## Purpose

PR-AE is the read-only publication preflight after PR-AD. It consumes one
exact verified Gold candidate publication and one explicit current categorical
Gold snapshot, then constructs the complete deterministic successor snapshot
that a later writer may publish.

It does not write Gold, activate a Gold head, construct the legacy
`OledGoldDatasetRecord`, generate dataset views, or enable training.

## Categorical Gold contract

PR-AE introduces `oled_categorical_gold_snapshot.v1`. Each entry embeds the
exact PR-AB admission candidate carried through PR-AC and independently
verified by PR-AD. It preserves:

- scientific consistency as the categorical value `consistent`;
- confidence sufficiency as the categorical value `sufficient`;
- the complete Registry, property/value/unit/context, PDF/table/cell, and
  human-review lineage;
- the source candidate ID and digest; and
- the source immutable candidate snapshot ID and digest.

No numeric confidence is inferred. In particular, PR-AE does not reuse the
legacy reviewed-candidate conversion fallback that may construct a numeric
score when one is missing.

## Exact inputs

The controlled file entry consumes three distinct regular JSON files:

1. one `oled_gold_candidate_postwrite_verification.v1` artifact;
2. the separately published `oled_gold_candidate_snapshot.v1` verified by that
   artifact; and
3. one explicit `oled_categorical_gold_snapshot.v1` representing current Gold.

Construction records the exact SHA-256 of all three files. The candidate
snapshot is independently replayed against the PR-AC receipt carried by PR-AD,
including its exact publication-file SHA. Reformatting that candidate snapshot
therefore fails closed.

The current Gold file SHA and semantic digest form the later compare-and-swap
parent boundary. No separate Gold-head or activation-receipt contract exists
yet, so `current_snapshot_lineage_receipt_bound=false` remains explicit.

## Explicit genesis

Initial publication must not infer an empty Gold state from a missing file.
Operators must provide a valid generation-zero categorical Gold snapshot:

- zero entries;
- no parent snapshot digest;
- no source verification digest; and
- a deterministic snapshot ID and digest.

`build_oled_categorical_gold_genesis_snapshot()` constructs this explicit
input.

## Conflict replay

PR-AE rejects the complete batch on any collision with current Gold or within
the candidate batch involving:

- Gold entry ID;
- source candidate ID or candidate digest;
- adjudicated observation digest;
- source-cell digest; or
- the semantic observation tuple of material, property, causal layer,
  normalized value/unit, and comparison-context hash.

It never overwrites, merges, silently drops, or assigns a new confidence score
to a conflicting observation.

## Deterministic successor plan

For a clean roster, the preflight constructs the complete expected successor:

- every current Gold entry preserved exactly;
- only verified candidate entries appended and entry-ID sorted;
- generation incremented by one;
- exact parent snapshot digest bound;
- exact PR-AD verification digest bound;
- deterministic successor snapshot ID and digest; and
- exact prior, planned, and expected entry counts.

The result is a plan only. The expected successor is not authoritative until a
later writer rechecks all exact inputs and publishes it.

## CLI

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_gold_successor_preflight \
  --verification-artifact /operator/local/pr-ad-verification.json \
  --candidate-snapshot /operator/local/gold_candidate_snapshot.json \
  --current-gold-snapshot /operator/local/current_gold_snapshot.json \
  --output /operator/local/gold_successor_preflight.json
```

Inputs and output must be distinct. Symbolic path components, input overwrite,
output-parent replacement, timestamp reversal, exact candidate publication
tamper, lineage tamper, and partial publication fail closed. CLI failures
expose only a stable error code and exception type.

## Explicitly false after PR-AE

- categorical Gold snapshot publication;
- mutable Gold-head activation;
- numeric confidence assignment;
- legacy numeric-confidence Gold-record construction;
- curated dataset or training eligibility;
- reviewed-evidence or Material Registry mutation; and
- network, external-service, LLM, or MinerU calls.

## Next boundary

PR-AF should consume this exact preflight, the exact PR-AD verification and
candidate snapshot, and the exact current Gold snapshot. Immediately before
publication it must re-read every input, compare-and-swap against the current
snapshot file SHA and digest, then publish the expected successor with fresh
files, fsync, inode binding, and atomic no-replace rename. Gold-head activation
must remain explicit in the publication receipt; dataset views remain later.
