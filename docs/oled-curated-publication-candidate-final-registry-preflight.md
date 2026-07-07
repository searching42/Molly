# OLED Publication Candidate Final Registry Preflight

This read-only Phase 1 preflight checks whether local OLED publication-candidate registry artifacts are ready for a later final/global registry writer gate.

Passing this preflight does not publish benchmark results, write final/global registry files, mutate any existing registry, mark benchmark results as validated, validate scientific claims, rerun backends, train models, predict, call LLMs or MinerU, or read PDFs/images.

## Inputs

- Publication writer manifest JSON from `oled_curated_promoted_registry_publication_writer.py`
- Publication-candidate entry JSON
- Publication-candidate index JSONL

The file runner resolves entry and index paths from the publication writer manifest, relative to `--publication-candidate-base-dir` or the manifest parent.

## Relationship To Publication Writer

`oled_curated_promoted_registry_publication_writer.py` writes local publication-candidate artifacts only. This preflight reads those artifacts and checks whether their manifest, entry, and index are internally consistent and still preserve the final-registry safety boundary.

It does not write final/global registry files. The final/global registry writer remains a later explicit gate.

## SHA256 Checks

When manifest file results include SHA256 values, the loader verifies:

- `publication_candidate_entry_json`
- `publication_candidate_index_jsonl`

Mismatches raise:

- `publication_candidate_entry_sha256_mismatch:`
- `publication_candidate_index_sha256_mismatch:`

## Consistency Checks

The preflight checks that:

- publication entry and index artifacts are present when required
- `publication_status=="publication_candidate"`
- the publication index references the publication entry id
- a single publication index record is present by default
- source chain identifiers are present
- source publication preflight status is `passed` or `passed_with_warnings`
- required caveats are present
- run-card and metric-card counts are nonzero when required

## Safety Checks

The preflight rejects source metadata that claims:

- `benchmark_validated=True`
- `scientific_claim_validated=True`
- `benchmark_published=True`
- `benchmark_registered=True`
- `globally_registered=True`
- `global_registry_mutated=True`
- `final_registry_written=True`

It also rejects raw prediction payloads, feature dictionaries, raw paper text markers, and absolute local path leakage.

## CLI Example

```bash
python -m ai4s_agent.domains.oled_curated_publication_candidate_final_registry_preflight \
  --publication-writer-manifest /path/to/publication_candidate_registry_manifest.json \
  --publication-candidate-base-dir /path/to/publication_candidate_registry \
  --output-report /path/to/final_registry_preflight_report.json
```

Optional repeated or comma-separated filters:

- `--baseline-kind`
- `--target-property-id`
- `--feature-view`

Use `--allow-multiple-publication-index-records` to relax the default single-index-record check.

## Output Report

The report JSON is deterministic and redacted. It includes artifact summaries, entry summaries, finding code counts, status counts, caveats, source identifiers, and safety metadata.

This preflight does not write final/global registry files, publish benchmark registry entries, validate benchmark performance, or create scientific conclusions.
