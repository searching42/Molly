# OLED Global Append Release Writer Gate

## Purpose

`oled_curated_global_append_release_writer.py` is the controlled writer gate after global-append release-readiness preflight.

It materializes local release-candidate artifacts from OLED global-append candidate artifacts that passed the read-only release preflight. These artifacts are still local review/release candidates. They are not externally published, benchmark validated, or scientific-performance claims.

## Inputs

- Global-append writer manifest JSON
- Global-append candidate entry JSON
- Global-append delta JSONL
- Newly written global registry snapshot JSONL
- Global-append release-readiness preflight report JSON
- Optional prior registry snapshot JSONL for release snapshot construction

The writer verifies source artifact SHA256 values through the existing global-append artifact loader where available.

## Relationship To Release Preflight

The release writer uses the release preflight report as its gate.

By default it requires the preflight to be valid. Warnings are allowed by default, but can be disallowed by policy. A failed preflight blocks release-candidate selection with `release_preflight_failed`.

## Confirmation Requirement

Write mode requires `--confirm-global-append-release-write`.

Without confirmation, the selection API raises:

```text
confirmation_required:global_append_release_write
```

## Dry Run

Dry-run mode builds the release-candidate entry and delta records in memory but does not write release entry, delta, or snapshot files.

If `--output-manifest` is provided, dry-run mode may write a manifest with `dry_run_no_files_written` in the file-result reason codes.

## Output Files

Default output names:

- `oled_release_candidate_entry.json`
- `oled_release_candidate_delta.jsonl`
- `oled_release_candidate_snapshot.jsonl`
- optional release writer manifest JSON

The snapshot writer preserves supplied snapshot records before appending release delta records.

## CLI Example

```bash
python -m ai4s_agent.domains.oled_curated_global_append_release_writer \
  --global-append-writer-manifest /path/to/global_append_writer_manifest.json \
  --release-preflight-report /path/to/global_append_release_preflight_report.json \
  --global-append-base-dir /path/to/global_append_candidate \
  --prior-registry-snapshot /path/to/prior_registry_snapshot.jsonl \
  --output-dir /path/to/release_candidate \
  --output-manifest /path/to/release_candidate_manifest.json \
  --confirm-global-append-release-write
```

## Safety Boundary

This gate does not mutate global registry files in place, publish externally, create GitHub releases, upload artifacts, mark benchmark validation, validate scientific claims, rerun baselines/models, call LLMs or MinerU, or read PDFs/images.

External publication / release remains a later explicit gate.
