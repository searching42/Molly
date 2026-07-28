# OLED Gold candidate post-write verifier (PR-AD)

## Purpose

PR-AD is a read-only independent verification boundary over one exact PR-AC
write receipt and the separately published Gold candidate snapshot.

It does not trust PR-AC success booleans as evidence and does not create Gold
records, activate a Gold head, write a dataset, or enable training.

## Exact files

The controlled entry consumes two distinct regular JSON files:

1. `oled_gold_candidate_write.v1`; and
2. `oled_gold_candidate_snapshot.v1`.

It records the exact SHA-256 of both files and independently rebuilds their
deterministic publication bytes. Semantically equivalent reformatting, added
whitespace, or a separately valid but different snapshot fails closed.

## Independent replay

PR-AD derives the expected snapshot from the receipt's embedded exact PR-AB
preflight and the PR-AC publication timestamp. It verifies:

- the separately supplied snapshot equals both the receipt snapshot and the
  independently rebuilt expected snapshot;
- candidate IDs are sorted and unique;
- the snapshot roster exactly equals the PR-AB eligible candidate roster;
- every candidate payload and candidate digest is unchanged;
- receipt candidate IDs, digests, and counts equal the published snapshot;
- snapshot ID, source-preflight digest, semantic digest, timestamp, and exact
  file SHA preserve lineage; and
- categorical confidence remains categorical, with no numeric score, legacy
  numeric-confidence record, Gold record, dataset, or training eligibility.

The replay does not use writer booleans such as
`exact_candidate_roster_replayed` or `published_payloads_revalidated`.

## Verification result

On success:

```text
status = gold_candidate_publication_verified
published_gold_candidate_snapshot_verified = true
eligible_for_explicit_gold_publication_input = true
gold_records_created = false
curated_dataset_written = false
```

Eligibility means only that a later, separately authorized boundary may consume
this exact immutable candidate snapshot. It is not Gold publication itself.

## CLI

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_gold_candidate_postwrite_verifier \
  --write-artifact /operator/local/gold_candidate_write.json \
  --published-gold-candidate-snapshot \
    /operator/local/gold_candidate_snapshot.json \
  --output /operator/local/gold_candidate_verification.json
```

Inputs and output must be distinct. Symbolic paths, input overwrite, output
parent replacement, timestamp reversal, byte reformatting, roster/payload/
digest/count/lineage tamper, and partial publication fail closed. CLI failures
expose only a stable error code and exception type.

## Explicitly false after PR-AD

- Gold record creation or mutable Gold-head activation;
- numeric confidence assignment or legacy Gold-record construction;
- dataset or training eligibility;
- reviewed-evidence or Registry mutation;
- source PDF reads; and
- network, external-service, LLM, or MinerU calls.

## Next boundary

The next project decision should define what “Gold publication” means under the
categorical-confidence contract. It should not reuse the legacy numeric
confidence writer unchanged. A small explicit schema/policy migration may be
needed before publishing final Gold records; dataset views must remain later.
