# OLED Global Append Release-Readiness Preflight

## Purpose

`oled_curated_global_append_release_preflight.py` checks whether local OLED global-append candidate artifacts are ready for a later release/publication gate.

This preflight is read-only. Passing it does not publish, validate, externally release, globally register, mutate a registry, or create scientific conclusions.

## Inputs

- Global-append writer manifest JSON.
- Global-append candidate entry JSON.
- Global-append delta JSONL.
- Newly written global registry snapshot JSONL.
- Optional prior registry snapshot JSONL.

The manifest points to the entry, delta, and new snapshot artifacts. The preflight verifies SHA256 values when the manifest provides them.

## SHA256 Checks

The file runner loads artifacts through the writer manifest and checks SHA256 digests for:

- `global_append_entry_json`
- `global_append_delta_jsonl`
- `global_registry_snapshot_jsonl`

Mismatch errors use the `global_append_entry_sha256_mismatch`, `global_append_delta_sha256_mismatch`, and `global_registry_snapshot_sha256_mismatch` reason codes.

## Delta And Snapshot Consistency

The preflight checks that:

- the global-append entry is present in the delta records
- delta records have `global_append_candidate` status
- the new snapshot contains the appended source chain
- the snapshot does not contain raw prediction payloads, feature dictionaries, raw text, or absolute paths

By default, exactly one delta record is expected for this local candidate append package.

## Prior Snapshot Preservation

When a prior registry snapshot is supplied, the preflight checks that every prior record is preserved in the same relative order at the beginning of the new snapshot.

This protects the release path from accidentally dropping or reordering existing registry records while still keeping this step read-only.

## Source Chain Checks

The report verifies source identifiers for the final-registry writer, final-registry entry, global-append preflight, publication candidate, promoted entry, registry entry, candidate report, and benchmark report manifest.

## Safety Boundary

The report metadata keeps the boundary explicit:

- `global_append_release_preflight_only=True`
- `global_registry_mutated=False`
- `external_publication_written=False`
- `benchmark_validated=False`
- `scientific_claim_validated=False`
- `baseline_backend_run=False`

The preflight rejects benchmark validation, scientific validation, external publication, and global mutation claims in manifest, entry, delta, and snapshot metadata.

## CLI Example

```bash
python -m ai4s_agent.domains.oled_curated_global_append_release_preflight \
  --global-append-writer-manifest /path/to/global_append_writer_manifest.json \
  --global-append-base-dir /path/to/global_append_candidate \
  --prior-registry-snapshot /path/to/existing_registry_snapshot.jsonl \
  --output-report /path/to/global_append_release_preflight_report.json
```

The CLI prints only a compact summary and never prints full registry records, metrics payloads, prediction payloads, feature dictionaries, or raw text.
