# OLED immutable Gold candidate writer (PR-AC)

## Purpose

PR-AC is the first boundary allowed to publish the candidate-only Gold roster
derived by PR-AB. It consumes one exact Gold admission preflight and publishes
one immutable snapshot plus one write receipt.

It does not create legacy `OledGoldDatasetRecord` objects, activate a mutable
Gold head, write a curated dataset, or enable training.

## Exact input recheck

The writer validates the complete PR-AB model and records the exact input file
SHA-256. Immediately before publication it rereads the input and requires both
its bytes and parsed payload to remain unchanged. Any change aborts without a
partial output.

An empty PR-AB eligible roster cannot be published. `no eligible evidence` is a
valid preflight outcome, but it is not a Gold candidate snapshot.

## Published unit

The fresh output directory contains exactly:

```text
gold_candidate_write.json
gold_candidate_snapshot.json
```

The snapshot contains the exact sorted PR-AB candidate roster, its source
preflight digest, a deterministic snapshot ID, and a semantic snapshot digest.
The receipt embeds the complete PR-AB artifact and published snapshot and
records the exact snapshot-file SHA-256.

Both files continue to declare:

```text
categorical_confidence_only = true
numeric_confidence_score_assigned = false
legacy_numeric_confidence_record_constructed = false
gold_records_created = false
curated_dataset_written = false
training_eligible = false
```

Publication does not reinterpret categorical sufficiency as a calibrated
probability.

## Atomic publication

PR-AC uses the same verified filesystem protocol as the Registry and
reviewed-evidence writers:

- pin the output parent by directory descriptor;
- create a private temporary sibling directory;
- write both files with fresh-file semantics and fsync them;
- fsync the temporary directory;
- publish it with a true atomic no-replace directory rename;
- fsync the parent directory; and
- revalidate the published directory inode, exact filenames, and exact bytes
  through the still-open directory descriptor.

Existing outputs, symbolic parents, temporary-directory replacement, targets
created in the check-to-rename window, and changed input bytes fail closed.
Cleanup removes only files and directories whose inodes are still owned by the
current invocation.

## CLI

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_gold_candidate_writer \
  --gold-admission-preflight /operator/local/pr-ab.json \
  --output-dir /operator/local/pr-ac-publication
```

CLI failures expose only a stable error code and exception type.

## Explicitly false after PR-AC

- independent post-write verification;
- Gold record creation or mutable Gold-head activation;
- numeric confidence assignment;
- curated dataset or training eligibility;
- reviewed-evidence, Registry, or alias mutation;
- source PDF reads; and
- network, external-service, LLM, or MinerU calls.

## Next boundary

The next safe step is an independent post-write verifier that consumes the
exact receipt and separately published snapshot, replays the complete
candidate roster and file hashes without trusting writer booleans, and only
then marks that immutable snapshot eligible for a later Gold publication
decision.
