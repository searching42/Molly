# OLED reviewed-evidence ledger post-write verifier (PR-T)

## Purpose

PR-T is a read-only verification boundary over one exact PR-S write receipt and
the separately published successor ledger snapshot. It independently replays
the transition before any confidence or scientific-consistency review and
before any Gold, dataset, or training admission step.

It does not write the ledger, change evidence, assign confidence, create Gold,
or mutate the Registry or aliases.

## Exact byte inputs

The file entry requires two distinct regular JSON files:

1. `oled_reviewed_evidence_ledger_write.v1`; and
2. `oled_reviewed_evidence_ledger_snapshot.v1`.

PR-T records the exact SHA-256 of both files, validates both models, and requires
the published snapshot to equal the exact successor embedded by PR-S. The
verification artifact embeds both inputs and their semantic digests. Standalone
validation cannot recover the original external bytes, so
`standalone_input_bytes_revalidation_supported=false` remains explicit.

## Independent replay

PR-T does not accept PR-S boundary flags as sufficient evidence. It independently
checks:

- every prior entry remains present and byte-for-model identical;
- every prior semantic-contract snapshot remains unchanged;
- the only added entries are those planned by writable PR-R items;
- each added entry is rebuilt from the exact PR-R candidate, semantic contract,
  PR-S timestamp, and required active/quarantined status;
- clean and consistent items are active;
- conflicts and incomplete-context items are quarantined, never active;
- exact-replay projections already exist in the prior snapshot and add nothing;
- no unplanned or non-writable projection crosses the write boundary;
- the successor snapshot ID binds the prior snapshot digest and exact PR-R
  digest; and
- no-op writes preserve the prior snapshot exactly.

## Controlled workflow

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_reviewed_evidence_ledger_postwrite_verifier \
  --write-artifact /operator/local/pr-s/reviewed_evidence_ledger_write.json \
  --published-ledger-snapshot \
    /operator/local/pr-s/reviewed_evidence_ledger_snapshot.json \
  --output /operator/local/pr-t-postwrite-verification.json
```

The output must be fresh and distinct from both inputs. Symbolic input/output
components, duplicate-key or non-finite JSON, input overwrite, changed output
parents, timestamp reversal, and partial publication fail closed. CLI failures
emit only a stable redacted error object.

## Automated boundary

The paper016-shaped exact-chain fixture verifies five active additions from the
genesis ledger. Tests also cover exact-replay no-op, conflict quarantine,
different valid snapshots, timestamp reversal, derived-count tamper, input
overwrite, exact file hashes, standalone validation, and CLI redaction.

This remains fixture-level validation. A real paper016 PR-S output and a
multi-paper append sequence remain later acceptance evidence.

## Next boundary

After PR-T, the ledger write path is mechanically verified but records still
carry `missing_confidence_assessment` and
`scientific_consistency_not_reviewed`. The next safe step is a bounded human
review request over those two facets, grouped by source row and restricted to
active, comparison-ready reviewed evidence. It must not convert quarantined
claims or device-only records into Gold.
