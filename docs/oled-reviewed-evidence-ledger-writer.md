# OLED reviewed-evidence ledger writer (PR-S)

## Purpose

PR-S is the first boundary allowed to publish reviewed evidence. It consumes
one exact PR-R staging preflight plus the exact prior ledger snapshot used by
that preflight and produces one immutable successor snapshot and one write
receipt.

It does not create Gold records, write a dataset, assign confidence, correct a
source value, mutate the Registry or aliases, or make an item training-eligible.

## Compare-and-swap binding

The file entry validates both input models and records the exact SHA-256 of
their bytes. The current ledger bytes must equal the SHA-256 pinned by PR-R,
and the complete current snapshot must equal PR-R's embedded prior snapshot.
Both inputs are read again immediately before publication; any byte or parsed
payload change aborts the write.

The output is a fresh directory. The receipt and next snapshot are written to
a private temporary sibling directory, fsynced, and renamed into place as one
directory commit. Existing outputs and symbolic output parents fail closed.

## Disposition-to-ledger rules

- `new_claim_ready` and `consistent_duplicate_ready` append `active` entries;
- `value_conflict_quarantine` and `incomplete_context_quarantine` append
  `quarantined` entries, never active entries;
- `exact_replay` appends nothing and preserves the prior snapshot exactly; and
- `revision_requires_review` and `semantic_contract_migration_required` abort
  the entire write because PR-S does not consume a roster-bound exception
  decision.

Every appended entry is rebuilt from the exact PR-R source candidate and
semantic-contract snapshot. The complete pre-existing entry set is preserved.

## Outputs

```text
<output-dir>/
  reviewed_evidence_ledger_write.json
  reviewed_evidence_ledger_snapshot.json
```

The write receipt embeds PR-R, the prior snapshot, and the successor snapshot.
Standalone validation deterministically replays the transition and verifies
all counts, entry IDs, status mappings, snapshot content, and artifact digest.
It also requires the prior-ledger file hash recorded by PR-S to equal the hash
pinned by PR-R. A standalone model cannot recover either original input's
external bytes, so `standalone_input_bytes_revalidation_supported=false`
remains explicit.

## CLI

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_reviewed_evidence_ledger_writer \
  --staging-preflight /operator/local/pr-r.json \
  --current-ledger-snapshot /operator/local/current-ledger.json \
  --output-dir /operator/local/pr-s-ledger-write
```

Failures emit only a stable redacted error object.

## Automated boundary

The paper016-shaped exact-chain fixture appends five active entries to the
genesis snapshot. Tests also cover exact-replay no-op behavior, conflict
quarantine, stale compare-and-swap inputs, refusal of unreviewed revisions,
fresh-output protection, receipt tamper, atomic two-file publication, and CLI
redaction.

This remains fixture-level validation. A real paper016 operator run and a
multi-paper append sequence remain later acceptance evidence.

## Next boundary

The next safe step is a read-only post-write verifier over the exact PR-S
receipt and published snapshot. It should independently replay append-only
preservation, disposition-to-status mapping, byte bindings, and snapshot
lineage before any confidence/scientific-consistency review or Gold admission.
