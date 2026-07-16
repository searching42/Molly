# OLED categorical Gold successor writer (PR-AF)

## Purpose

PR-AF consumes one exact PR-AE successor preflight and every exact external
input bound by that preflight. It compare-and-swaps against the current
categorical Gold snapshot, publishes the complete expected successor as an
immutable fresh directory, and records explicit snapshot activation in the
publication receipt.

It does not write a mutable Gold-head pointer, mutate the prior snapshot,
construct legacy numeric-confidence Gold records, generate dataset views, or
enable training.

## Exact inputs

The writer accepts four distinct, non-symlinked JSON files:

1. `oled_gold_successor_preflight.v1`;
2. the exact `oled_gold_candidate_postwrite_verification.v1` used by PR-AE;
3. the exact published `oled_gold_candidate_snapshot.v1` used by PR-AE; and
4. the exact current `oled_categorical_gold_snapshot.v1` used by PR-AE.

The verification, candidate, and current snapshot file SHA-256 values must
equal the values bound by PR-AE. Every supplied model must also equal its
embedded PR-AE model and reproduce its semantic digest. A semantically
equivalent but reformatted file is a different publication input and fails
closed.

Immediately before publication, PR-AF rereads all four files. Any byte or
parsed-payload change since initial derivation aborts publication.

## Compare-and-swap and transition replay

The current categorical Gold file SHA and semantic digest form the exact CAS
parent. PR-AF independently verifies:

- the current snapshot equals PR-AE's complete embedded current snapshot;
- every prior Gold entry is preserved exactly;
- the successor roster is exactly prior entries plus PR-AE planned entries;
- every planned entry is unchanged;
- generation increments by one;
- parent snapshot and source-verification lineage are exact;
- prior, added, and published counts are complete; and
- the published snapshot equals PR-AE's expected successor and digest.

PR-AE's deterministic entry-ID and current-snapshot internal-uniqueness
validators are replayed during all model reconstruction.

## Atomic publication

PR-AF creates a private sibling directory and writes exactly:

```text
<fresh-output-dir>/
  categorical_gold_successor_write.json
  categorical_gold_snapshot.json
```

Both files use deterministic publication bytes and are created with
exclusive/no-follow semantics. Files and the temporary directory are fsynced.
The directory is then published with atomic no-replace rename, the parent
directory is fsynced, and the still-open directory descriptor revalidates the
published inode, exact filenames, and exact bytes.

Existing targets, concurrent targets, symbolic path components, output-parent
replacement, partial writes, and unsupported no-replace runtimes fail closed.
Cleanup removes only objects whose inodes belong to the current invocation.

## Publication and activation receipt

`oled_gold_successor_write.v1` binds:

- exact construction-time file SHA and semantic digest for all inputs;
- all four validated input models;
- the exact published successor and serialized file SHA;
- prior/published generation and entry counts;
- added entry IDs and digests;
- successor and activated snapshot ID/digest; and
- fsync, CAS, atomic publication, inode, and payload-revalidation assertions.

`categorical_gold_snapshot_activated=true` means this immutable publication
unit is explicitly activated by the receipt. It does not mean a mutable
pointer was written:

```text
activation_receipt_created = true
categorical_gold_snapshot_activated = true
gold_head_activated = false
mutable_gold_head_pointer_written = false
```

## Explicitly false after PR-AF

- mutable Gold-head creation or activation;
- prior Gold snapshot mutation;
- numeric confidence assignment;
- legacy numeric-confidence Gold-record construction;
- curated dataset or training eligibility;
- reviewed-evidence or Material Registry mutation; and
- source PDF, network, external-service, LLM, or MinerU access.

## CLI

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_gold_successor_writer \
  --successor-preflight /operator/local/pr-ae-preflight.json \
  --verification-artifact /operator/local/pr-ad-verification.json \
  --candidate-snapshot /operator/local/gold_candidate_snapshot.json \
  --current-gold-snapshot /operator/local/current_gold_snapshot.json \
  --output-dir /operator/local/gold-successor-publication
```

CLI failures expose only a stable error code and exception type.

## Next boundary

PR-AG should independently consume the receipt and separately published
categorical Gold snapshot. It must reconstruct the exact receipt/snapshot
publication bytes and replay the complete CAS transition, activation binding,
entry coverage, ordering, counts, and lineage without trusting PR-AF booleans.
Dataset admission remains a later boundary.
