# Custom Corpus Governance Stage Summary - 2026-06-28

## Summary

Molly now has a full custom corpus governance chain through package validation:

```text
custom_corpus_manifest.v1
-> custom_corpus_dry_run.v1
-> custom_corpus_review.v1
-> custom_corpus_admission.v1
-> custom_corpus_admission_package_validation.v1
-> future dataset materialization
```

The chain supports controlled custom corpus intake, unconfirmed dry-runs,
human review artifacts, admission-intent validation, and cross-artifact package
binding validation. It does not materialize records into datasets and does not
run Phase 1.

## Completed PRs

- **#155 intake contract**: defined the custom/private corpus intake contract,
  corpus class policy, redaction expectations, and dry-run boundary. It did
  not implement parsing, review, admission, or dataset materialization.
- **#156 dry-run runner**: implemented the controlled local PDF dry-run path
  that keeps `DatasetConfirmation.confirmed=false` and verifies Phase 1 remains
  `not_run`. It did not admit training data.
- **#157 public dry-run evidence**: recorded redacted evidence for a small
  public custom corpus dry-run. It did not commit PDFs, full reports, MinerU
  bundles, or ParsedDocument outputs.
- **#158 human review artifact schema**: added `custom_corpus_review.v1` and
  an offline validator for redacted review metadata. Review artifacts still do
  not admit training data.
- **#159 admission request gate contract**: added `custom_corpus_admission.v1`
  and an offline validator for admission intent. It did not create candidate or
  training datasets.
- **#160 package binding validator**: added
  `custom_corpus_admission_package_validation.v1` summary generation across
  manifest, dry-run report, review manifest, and admission request. It did not
  materialize datasets or run Phase 1.

## Current Trust Boundary

Custom corpora can be described, parsed in dry-run mode, reviewed, and checked
for admission intent. The package binding validator can verify that the review
and admission request are tied to the expected manifest and dry-run report.

No records are yet materialized into training data. Phase 1 remains protected
by the absence of any custom corpus dataset materialization implementation and
by the unchanged `DatasetConfirmation` boundary.

## Why This Matters

This governance chain separates parsing from review, review from admission
intent, and admission intent from materialization. It prevents silent
training-data admission, preserves reproducibility through artifact hashes, and
keeps redaction requirements explicit before any future dataset builder can
consume custom corpus records.

## Remaining Gaps

- no materialization implementation
- no reviewed-record-to-dataset transform
- no production data deletion/rollback story
- no full real scientific extraction quality benchmark
- no private corpus operational certification
- no MinerU Cloud API provider

## Recommended Next PR

```text
docs: design custom corpus dataset materialization boundary
```

The next step should be a design-only PR that defines how package-validated
admission records could become materialized candidate/training artifacts,
which explicit operator gates are required, and how rollback/deletion and
provenance binding will work before any runtime implementation is added.

## Post-Runbook Design Note

The materialization boundary design was added after the #155-#161 governance
runbook work:

```text
docs/custom-corpus-dataset-materialization-boundary.md
```

It documents the future materialization boundary, but still does not implement
materialization. The custom corpus path remains protected before candidate or
training artifacts are created.

## Post-Design Schema Note

The materialization plan schema was added after the materialization boundary
design:

```text
docs/custom-corpus-materialization-schema.md
```

It validates candidate-only materialization intent and source binding, but
still does not implement a materializer or create candidate/training artifacts.
