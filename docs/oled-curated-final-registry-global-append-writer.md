# OLED Final Registry Global Append Writer Gate

## Purpose

`oled_curated_final_registry_global_append_writer.py` materializes local global-append candidate artifacts from OLED final-registry candidate artifacts that passed the global-append-readiness preflight.

The output is still a local candidate package. It is not an in-place append to an existing global registry, not external publication, not benchmark validation, and not scientific performance validation.

## Inputs

- Final-registry candidate writer manifest JSON.
- Final-registry candidate entry JSON.
- Final-registry candidate index JSONL.
- Global-append-readiness preflight report JSON.
- Optional existing final/global registry snapshot JSONL for building a new candidate snapshot.

The writer loads source artifacts through the manifest and verifies SHA256 values when present. It does not mutate source artifacts or existing snapshots.

## Relationship To Global-Append Preflight

The writer uses the global-append-readiness preflight report as the gate. By default, failed preflight blocks selection, while warnings are allowed for sparse pilot data. Policies can disallow warnings.

The preflight remains read-only. This writer is the first gate that may create local global-append candidate outputs, but it still does not update a real global registry.

## Confirmation Requirement

Write mode requires `--confirm-final-registry-global-append-write`.

Without confirmation, the selection API raises:

```text
confirmation_required:final_registry_global_append_write
```

## Dry-Run Mode

`--dry-run` builds the candidate entry and delta records in memory. It writes no entry, delta, or snapshot files. If `--output-manifest` is supplied, the manifest can be written with `dry_run_no_files_written`.

## Output Files

Default write mode can produce:

- `oled_global_append_candidate_entry.json`
- `oled_global_append_candidate_delta.jsonl`
- `oled_global_registry_snapshot.jsonl`
- a writer manifest JSON at the requested path

The snapshot writer preserves existing snapshot records first and appends candidate delta records into a new output file. It never edits the input snapshot in place.

## SHA256 Manifest

The writer manifest records output file paths, SHA256 digests, status, reason codes, policy, source manifest identifiers, source preflight status, and safety metadata.

## CLI Example

```bash
python -m ai4s_agent.domains.oled_curated_final_registry_global_append_writer \
  --final-registry-writer-manifest /path/to/final_registry_candidate_manifest.json \
  --global-append-preflight-report /path/to/global_append_preflight_report.json \
  --final-registry-candidate-base-dir /path/to/final_registry_candidate \
  --existing-registry-snapshot /path/to/existing_registry_snapshot.jsonl \
  --output-dir /path/to/global_append_candidate \
  --output-manifest /path/to/global_append_writer_manifest.json \
  --confirm-final-registry-global-append-write
```

## Safety Boundary

This gate does not append to or mutate a global registry, publish externally, mark outputs benchmark validated, validate scientific claims, rerun baselines or model backends, call LLMs or MinerU, or read PDFs or images.
