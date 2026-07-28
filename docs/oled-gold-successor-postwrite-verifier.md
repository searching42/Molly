# OLED categorical Gold successor post-write verifier (PR-AG)

## Purpose

PR-AG is the independent read-only verification boundary after PR-AF. It
consumes the exact PR-AF publication/activation receipt and the separately
published categorical Gold snapshot, then replays the complete publication,
compare-and-swap transition, and snapshot-activation binding.

It does not trust PR-AF safety booleans as evidence. It does not write or
activate Gold, create a mutable head, generate datasets, or enable training.

## Exact inputs

The controlled entry consumes two distinct regular JSON files:

1. `oled_gold_successor_write.v1`; and
2. the separately published `oled_categorical_gold_snapshot.v1`.

PR-AG records the exact SHA-256 of both files and independently reconstructs
their deterministic publication bytes. Added whitespace, key reformatting, or
a separately valid but different snapshot fails closed.

The receipt and published snapshot models are fully revalidated, including the
PR-AE deterministic entry-ID and snapshot-internal uniqueness invariants.

## Independent transition replay

Without relying on PR-AF booleans such as
`append_only_transition_verified`, `published_payloads_revalidated`, or
`categorical_gold_snapshot_activated`, PR-AG independently verifies:

- the receipt semantic digest and exact receipt-file SHA;
- the snapshot semantic digest and exact snapshot-file SHA;
- equality of the standalone snapshot, PR-AF embedded snapshot, and PR-AE
  expected successor;
- exact PR-AE current-snapshot model, digest, and file-SHA CAS binding;
- preservation of every prior entry;
- exact entry coverage of prior plus planned additions;
- every planned entry payload and digest;
- deterministic entry ordering and snapshot internal uniqueness;
- prior, added, and published counts;
- registry identity, generation increment, parent digest, PR-AD verification
  lineage, snapshot ID/digest, and timestamp lineage; and
- equality of activated snapshot ID/digest with the separately published
  snapshot.

## Verification result

On success:

```text
status = categorical_gold_successor_publication_verified
published_categorical_gold_snapshot_verified = true
snapshot_activation_receipt_verified = true
eligible_for_explicit_dataset_admission_input = true
curated_dataset_written = false
training_eligible = false
```

Eligibility means only that a later explicit admission boundary may consume
this exact verified snapshot. It is not dataset admission, dataset
materialization, or training approval.

The verifier records that the PR-AF activation receipt is valid, but performs
no activation itself:

```text
gold_snapshot_written = false
categorical_gold_snapshot_activated = false
gold_head_activated = false
mutable_gold_head_pointer_written = false
```

## CLI

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_gold_successor_postwrite_verifier \
  --write-artifact \
    /operator/local/categorical_gold_successor_write.json \
  --published-categorical-gold-snapshot \
    /operator/local/categorical_gold_snapshot.json \
  --output /operator/local/gold_successor_verification.json
```

Inputs and output must be distinct. Symbolic path components, input overwrite,
output-parent replacement, timestamp reversal, exact-byte reformatting,
transition/activation tamper, and partial output publication fail closed. CLI
failures expose only a stable error code and exception type.

## Explicitly false after PR-AG

- Gold snapshot write or activation by the verifier;
- mutable Gold-head creation or activation;
- prior snapshot mutation;
- numeric confidence or legacy Gold-record construction;
- curated dataset or training eligibility;
- reviewed-evidence or Material Registry mutation; and
- source PDF, network, external-service, LLM, or MinerU access.

## Next boundary

The next project decision should define explicit dataset admission over one
exact PR-AG-verified categorical Gold snapshot. Admission should decide which
Gold entries may enter which dataset views while preserving categorical
confidence and causal-layer/context constraints. Dataset materialization and
training-package generation must remain later boundaries.
