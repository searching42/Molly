# Custom Corpus Property Training Dataset Row Contract Precheck

The property training dataset row contract precheck validates an existing
`custom_corpus_property_training_dataset_row_contract.v1` package before any
future materialization dry-run or dataset writer can use it.

It answers whether the row contract package is internally consistent, safe,
and hash-bound to the materialization plan and upstream evidence.

It does not write row previews or training dataset files.

## Relationship To Row Contract

The upstream row contract is documented in:

```text
docs/custom-corpus-property-training-dataset-row-contract.md
```

The row contract defines future row semantics. The precheck reads that
contract, the row contract summary, the materialization plan precheck, the
materialization plan and planner summary, ledger evidence, dry-run evidence,
execution request evidence, request draft evidence, request plan/preflight,
training admission readiness, and quarantine candidate evidence. It validates
the full chain again and emits safe precheck evidence.

Future materialization dry-runs and dataset writers remain separate.

## Validation Rules

The precheck validates:

- row contract and summary schema versions
- row contract status and needs-review gating
- source SHA-256 bindings across the full package
- safe ids and safe basenames
- required and optional row fields
- field type descriptors
- provenance contract requirements
- quality flag labels
- split and dedup key requirements
- model-family compatibility labels
- output-format compatibility labels
- contract record reference counts and ids
- planned dataset record and ledger record consistency
- excluded, blocked, and needs-review candidate leakage
- redaction boundaries

The post-contract boundary must remain:

- `training_admitted=true`
- `training_dataset_materialized=false`
- `dataset_artifact_created=false`
- `phase1_status=not_run`
- `dataset_confirmation_changed=false`

## Summary Schema

The precheck emits:

```text
custom_corpus_property_training_dataset_row_contract_precheck.v1
```

Status values:

- `passed`
- `needs_review`
- `blocked`

`passed` means all checks passed and no needs-review evidence remains.
`needs_review` means no hard error was found, but allowed needs-review evidence
is present. `blocked` means a schema, status, hash, id, record, field contract,
boundary, or redaction check failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_row_contract_precheck \
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
  --output-summary /tmp/property_training_dataset_row_contract_precheck_summary.json \
  --output-markdown /tmp/property_training_dataset_row_contract_precheck_summary.md
```

Return codes:

- `0` for `passed` or `needs_review`
- `1` for `blocked`

## Redaction

Summary and Markdown evidence may include safe field names, schema/status
fields, safe ids, SHA-256 hashes, aggregate counts, allowed quality flag
labels, model-family labels, output-format labels, and safe error codes.

They must not include raw property values, raw table rows, raw article text,
local absolute paths, private paths, PDF names or paths, serialized dataset
rows, output artifact paths, or full upstream payloads.

## Boundaries

- This is a training dataset row contract precheck only.
- No training dataset artifact is created.
- No row preview is generated.
- No training CSV/JSONL/Parquet/LMDB artifact is created.
- No candidate CSV/JSONL/Parquet/LMDB artifact is created.
- No conformers are generated.
- No DPA3 structures are generated.
- No Phase 1 execution occurs.
- `DatasetConfirmation` is not changed.
- No model training or evaluation is run.
- No LLM, agent, MinerU, PDF, ParsedDocument, or corpus workflow call is made.
