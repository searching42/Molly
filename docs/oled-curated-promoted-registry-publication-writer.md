# OLED Curated Promoted Registry Publication Writer

This writer is a controlled Phase 1 gate for local OLED publication-candidate registry artifacts. It consumes promoted candidate registry artifacts only after the publication-readiness preflight has passed, then materializes a standalone publication-candidate entry JSON, a standalone publication-candidate index JSONL, and an audit manifest.

It does not publish to an external service, mutate a global registry, mark benchmark outputs as validated, validate scientific performance claims, rerun backends, train models, predict, call LLMs or MinerU, or read PDFs/images.

## Inputs

- Promotion writer manifest JSON from `oled_curated_benchmark_registry_promotion_writer.py`
- Promoted registry entry JSON
- Promoted registry index JSONL
- Promoted registry publication-readiness preflight report JSON from `oled_curated_promoted_registry_publication_preflight.py`

The file runner loads promoted entry and index artifacts through the promotion writer manifest, using paths relative to `--promoted-registry-base-dir` or the manifest parent.

## Relationship To Publication Preflight

`oled_curated_promoted_registry_publication_preflight.py` is read-only and verifies that promoted candidate artifacts are publication-ready. This writer treats that report as the gate:

- failed publication preflight blocks writing by default
- warnings are allowed by default, but can be disallowed by policy
- source metadata claiming benchmark validation, scientific validation, publication, or global registration is rejected

Passing the writer gate still creates only local publication-candidate artifacts.

## Confirmation

Writing requires explicit confirmation by default:

```bash
--confirm-promoted-registry-publication-write
```

Without confirmation, selection raises:

```text
confirmation_required:promoted_registry_publication_write
```

## Dry Run

`--dry-run` performs selection and manifest assembly without writing publication entry or index artifacts. If `--output-manifest` is provided, only the manifest is written and includes `dry_run_no_files_written`.

## Output Files

Default filenames:

- `oled_publication_candidate_registry_entry.json`
- `oled_publication_candidate_registry_index.jsonl`
- caller-provided publication writer manifest JSON

The entry and index preserve source identifiers, selected baseline kinds, target property ids, feature views, caveats, run-card count, metric-card count, and SHA256 provenance for source promoted artifacts.

## Publication-Candidate Status

The writer uses `publication_status="publication_candidate"` only. This status means the artifacts are local review candidates for a later final/global registry gate. It does not mean external publication, global registration, benchmark validation, or scientific claim validation.

## Safety Boundary

All report, manifest, and entry metadata keeps:

- `final_registry_written=False`
- `global_registry_mutated=False`
- `benchmark_published=False`
- `benchmark_registered=False`
- `benchmark_validated=False`
- `scientific_claim_validated=False`
- `baseline_backend_rerun=False`
- `models_fitted=False`
- `predictions_written=False`
- `metrics_written=False`
- `llm_called=False`
- `mineru_called=False`
- `pdfs_read=False`
- `images_read=False`

## CLI Example

```bash
python -m ai4s_agent.domains.oled_curated_promoted_registry_publication_writer \
  --promotion-writer-manifest /path/to/promoted_registry_manifest.json \
  --publication-preflight-report /path/to/promoted_registry_publication_preflight_report.json \
  --promoted-registry-base-dir /path/to/promoted_registry \
  --output-dir /path/to/publication_candidate_registry \
  --output-manifest /path/to/publication_candidate_registry_manifest.json \
  --confirm-promoted-registry-publication-write
```

Use `--entry-only` or `--index-only` to restrict output artifact kinds. Use repeated or comma-separated `--baseline-kind`, `--target-property-id`, and `--feature-view` to select subsets.

This writer does not publish, validate benchmark performance, globally register, or create scientific conclusions.
