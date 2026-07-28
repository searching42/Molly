# Custom Corpus Property Training Dataset Materialization Dry-Run

The property training dataset materialization dry-run consumes a
row-contract-precheck-passed package and emits safe row preview evidence before
any future dataset writer or materializer work.

It answers whether the row contract and upstream materialization evidence can
produce preview-shaped row references without writing a training dataset.

It does not serialize training rows.
It does not create training dataset files.

## Relationship To Row Contract Precheck

The upstream row contract precheck is documented in:

```text
docs/custom-corpus-property-training-dataset-row-contract-precheck.md
```

The precheck validates the row contract package. The dry-run reads the same
contract, precheck, materialization plan, planner summary, ledger evidence,
execution dry-run evidence, execution request evidence, request draft evidence,
request plan/preflight, training admission readiness, and quarantine candidate
evidence. It validates the full ID/SHA/status/record chain again and writes
dry-run-only preview summaries.

Future dataset writing remains separate.

## Output Artifacts

The CLI writes artifacts under a run-specific clean directory:

```text
<output-dir>/<materialization-dry-run-id>/
  property_training_dataset_materialization_dry_run_report.json
  property_training_dataset_materialization_dry_run_summary.json
  redacted_property_training_dataset_materialization_dry_run_evidence.md
```

The report schema is:

```text
custom_corpus_property_training_dataset_materialization_dry_run.v1
```

The summary schema is:

```text
custom_corpus_property_training_dataset_materialization_dry_run_summary.v1
```

## Dry-Run Rules

The dry-run validates:

- row contract precheck status and optional needs-review allowance
- row contract schema, status, field contract, quality flags, split/dedup keys,
  model-family labels, and output-format labels
- materialization plan and planner summary status
- ledger, dry-run, execution request, request draft, request plan/preflight,
  readiness, and quarantine candidate evidence status
- SHA-256 bindings across the full package
- corpus, materialization plan, execution ledger, execution request, row
  contract, and dataset name consistency
- planned dataset record ids and contract record references
- excluded, blocked, and needs-review candidate leakage
- dry-run boundary fields
- output redaction

The post-contract boundary must remain:

- `training_admitted=true`
- `training_dataset_materialized=false`
- `dataset_artifact_created=false`
- `phase1_status=not_run`
- `dataset_confirmation_changed=false`

## Row Previews

Row previews are summaries only. They contain safe ids, safe field names,
model-family labels, output-format labels, counts, and SHA-256 hashes.

Row previews must not include raw property values, raw table rows, raw article
text, PDF names or paths, local paths, serialized training rows, or future
output paths.

## Status Values

The dry-run status is:

- `passed` when all checks pass and no needs-review evidence remains
- `needs_review` when no hard error exists and explicitly allowed upstream
  needs-review evidence is present
- `blocked` when a schema, status, hash, id, record, boundary, or redaction
  check fails

CLI return codes:

- `0` for `passed` or `needs_review`
- `1` for `blocked`

## After Materialization Dry-Run: Dry-Run Precheck

After a dry-run writes row preview summaries, the package can be independently
checked before any future dataset writer request:

```text
docs/custom-corpus-property-training-dataset-materialization-dry-run-precheck.md
```

Future dry-run precheck evidence should use:

```text
docs/evidence/templates/custom-corpus-property-training-dataset-materialization-dry-run-precheck-evidence-template.md
```

The dry-run precheck validates dry-run package consistency, source hashes,
row-preview summaries, field coverage, model-family labels, output-format
labels, and upstream evidence. It still does not serialize training rows,
create dataset artifacts, write CSV/JSONL/Parquet/LMDB files, generate
conformers or DPA3 structures, run Phase 1, or change `DatasetConfirmation`.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_materialization_dry_run \
  --training-dataset-row-contract-precheck /tmp/property_training_dataset_row_contract_precheck_summary.json \
  --training-dataset-row-contract /tmp/property_training_dataset_row_contract.json \
  --training-dataset-row-contract-summary /tmp/property_training_dataset_row_contract_summary.json \
  --training-dataset-materialization-plan-precheck /tmp/property_training_dataset_materialization_plan_precheck_summary.json \
  --training-dataset-materialization-plan /tmp/property_training_dataset_materialization_plan.json \
  --training-dataset-materialization-planner-summary /tmp/property_training_dataset_materialization_planner_summary.json \
  --training-admission-execution-ledger-precheck /tmp/property_training_admission_execution_ledger_precheck_summary.json \
  --training-admission-execution-ledger /tmp/property_training_admission_execution_ledger.json \
  --training-admission-execution-ledger-summary /tmp/property_training_admission_execution_ledger_summary.json \
  --training-admission-execution-dry-run-precheck /tmp/property_training_admission_execution_dry_run_precheck_summary.json \
  --training-admission-execution-dry-run-report /tmp/property_training_admission_execution_dry_run_report.json \
  --training-admission-execution-request /tmp/property_training_admission_execution_request.json \
  --training-admission-execution-request-summary /tmp/property_training_admission_execution_request_summary.json \
  --training-admission-execution-request-preflight /tmp/property_training_admission_execution_request_preflight_summary.json \
  --training-admission-request-draft /tmp/property_training_admission_request.draft.json \
  --training-admission-request-draft-summary /tmp/property_training_admission_request_draft_summary.json \
  --training-admission-request-draft-precheck /tmp/property_training_admission_request_draft_precheck_summary.json \
  --training-admission-request-plan /tmp/property_training_admission_request_plan_summary.json \
  --training-admission-request-preflight /tmp/property_training_admission_request_preflight_summary.json \
  --training-admission-readiness-summary /tmp/property_training_admission_readiness_summary.json \
  --quarantine-candidate-preflight-summary /tmp/property_quarantine_candidate_preflight_summary.json \
  --quarantine-candidate-records /tmp/property_quarantine_candidate_records.json \
  --output-dir /tmp/property-training-dataset-materialization-dry-run \
  --materialization-dry-run-id property-training-dataset-dry-run-example-001 \
  --created-by operator-redacted \
  --confirm-training-dataset-materialization-dry-run
```

## Redaction

The report, summary, and Markdown evidence may include safe ids, field names,
SHA-256 hashes, counts, status labels, model-family labels, output-format
labels, and safe error codes.

They must not include local absolute paths, private paths, PDF names or paths,
raw article text, raw table rows, serialized training rows, future dataset
output paths, tokens, Authorization headers, cookies, signed URLs, or private
emails.

## Boundaries

- This is a training dataset materialization dry-run only.
- Row previews are summaries only.
- No serialized training rows are created.
- No training dataset artifact is created.
- No training CSV/JSONL/Parquet/LMDB artifact is created.
- No candidate CSV/JSONL/Parquet/LMDB artifact is created.
- No conformers are generated.
- No DPA3 structures are generated.
- No Phase 1 execution occurs.
- `DatasetConfirmation` is not changed.
- No model training or evaluation is run.
- No LLM, agent, MinerU, PDF, ParsedDocument, or corpus workflow call is made.
