from __future__ import annotations

from pathlib import Path


DESIGN_DOC = Path("docs/custom-corpus-property-training-dataset-controlled-writer-execution-request-design.md")
TEMPLATE_DOC = Path(
    "docs/evidence/templates/"
    "custom-corpus-property-training-dataset-controlled-writer-execution-request-design-evidence-template.md"
)

CHAIN_PHRASE = (
    "property training dataset controlled writer value resolution dry-run\n"
    "-> property training dataset controlled writer value resolution dry-run precheck\n"
    "-> small public quarantine materialization evidence\n"
    "-> property training dataset quarantined candidate admission boundary\n"
    "-> property training dataset domain validation boundary\n"
    "-> property training dataset controlled writer design plan\n"
    "-> property training dataset controlled writer design plan preflight\n"
    "-> property training dataset controlled writer dry-run design\n"
    "-> property training dataset controlled writer dry-run\n"
    "-> property training dataset controlled writer dry-run precheck\n"
    "-> property training dataset controlled writer execution request design\n"
    "-> property training dataset controlled writer execution request\n"
    "-> future controlled writer execution request preflight\n"
    "-> future explicitly confirmed controlled writer execution"
)

GOVERNANCE_DOCS = (
    Path("docs/custom-corpus-dataset-materialization-boundary.md"),
    Path("docs/custom-corpus-governance-runbook.md"),
    Path("docs/custom-corpus-governance-stage-summary-20260628.md"),
    Path("docs/custom-corpus-materialization-schema.md"),
    Path("docs/custom-corpus-property-training-dataset-controlled-writer-design-plan.md"),
    Path("docs/custom-corpus-property-training-dataset-controlled-writer-design-plan-preflight.md"),
    Path("docs/custom-corpus-property-training-dataset-controlled-writer-dry-run-design.md"),
    Path("docs/custom-corpus-property-training-dataset-controlled-writer-dry-run.md"),
    Path("docs/custom-corpus-property-training-dataset-controlled-writer-dry-run-precheck.md"),
    Path("docs/phase-1-4-milestone-status.md"),
)

REQUIRED_SECTIONS = (
    "# Custom Corpus Property Training Dataset Controlled Writer Execution Request Design",
    "## Purpose",
    "## Position in the Governance Chain",
    "## Required Upstream Evidence",
    "## Execution Request Design Scope",
    "## Future Execution Request Input Contract",
    "## Future Execution Request Schema",
    "## Future Execution Request Summary Schema",
    "## Authorization Boundary",
    "## Explicit Confirmation Boundary",
    "## Allowed Future Request Fields",
    "## Disallowed Current Outputs",
    "## Disallowed Future Request Fields",
    "## Hash and Basename Policy",
    "## Redaction and Non-Leakage Policy",
    "## Request Status Semantics",
    "## Future Execution Request Preflight Expectations",
    "## Implementation Blockers",
    "## Pass Criteria",
    "## Needs-Review Criteria",
    "## Fail Criteria",
    "## Residual Risks",
    "## Next Step",
)

REQUIRED_PLACEHOLDERS = (
    "<controlled_writer_execution_request_design_evidence_id>",
    "<date>",
    "<operator>",
    "<corpus_id>",
    "<dataset_name>",
    "<controlled_writer_dry_run_precheck_status>",
    "<controlled_writer_dry_run_status>",
    "<controlled_writer_dry_run_report_sha256>",
    "<controlled_writer_dry_run_report_basename>",
    "<controlled_writer_dry_run_summary_basename>",
    "<controlled_writer_design_plan_preflight_status>",
    "<domain_validation_boundary_status>",
    "<controlled_writer_value_resolution_dry_run_precheck_status>",
    "<accepted_candidate_record_count>",
    "<needs_review_candidate_record_count>",
    "<blocked_candidate_record_count>",
    "<missing_required_field_count>",
    "<redaction_status>",
    "<future_execution_request_schema>",
    "<future_execution_request_summary_schema>",
    "<future_execution_request_preflight_schema>",
    "<future_explicit_confirmation_schema>",
    "<next_gate_decision>",
    "<residual_risks>",
)

FUTURE_SCHEMA_LABELS = (
    "custom_corpus_property_training_dataset_controlled_writer_execution_request.v1",
    "custom_corpus_property_training_dataset_controlled_writer_execution_request_summary.v1",
    "custom_corpus_property_training_dataset_controlled_writer_execution_request_preflight.v1",
    "custom_corpus_property_training_dataset_controlled_writer_explicit_confirmation.v1",
)

BOUNDARY_STATEMENTS = (
    "This controlled writer execution request design does not create an execution request.",
    "This controlled writer execution request design does not implement execution request creation.",
    "This controlled writer execution request design does not implement execution request preflight.",
    "This controlled writer execution request design does not explicitly confirm execution.",
    "This controlled writer execution request design does not execute the controlled writer.",
    "This controlled writer execution request design does not emit raw values.",
    "This controlled writer execution request design does not materialize values.",
    "This controlled writer execution request design does not serialize training rows.",
    "This controlled writer execution request design does not create training dataset artifacts.",
    "This controlled writer execution request design does not create CSV/JSONL/Parquet/LMDB artifacts.",
    "This controlled writer execution request design does not generate conformers.",
    "This controlled writer execution request design does not generate DPA3 structures.",
    "This controlled writer execution request design does not run Phase 1.",
    "This controlled writer execution request design does not modify DatasetConfirmation.",
    "This controlled writer execution request design does not run model training or evaluation.",
)

REQUIRED_AUTHORIZATION_PHRASES = (
    "A controlled writer execution request is not a controlled writer execution.",
    "A controlled writer execution request is not explicit confirmation.",
    "A controlled writer execution request does not authorize execution by itself.",
    "A controlled writer execution request must be separately prechecked before any confirmation gate.",
    "A controlled writer execution request must not be inferred from a passed dry-run precheck alone.",
    "A controlled writer execution request must not be inferred from CI success alone.",
    "A controlled writer execution request must not be inferred from merge status alone.",
    "stale, missing, mismatched, blocked, or needs-review upstream evidence is not execution-ready by default",
)

FUTURE_STATUS_LABELS = (
    "request_designed",
    "request_ready_for_preflight",
    "request_needs_review",
    "request_blocked",
)

FORBIDDEN_CATEGORY_PHRASES = (
    "raw property values",
    "exact numeric extracted values",
    "molecular strings",
    "SMILES",
    "InChI",
    "InChIKey",
    "row payloads",
    "serialized rows",
    "table payloads",
    "article text",
    "paper titles",
    "PDF names",
    "source payloads",
    "local paths",
    "absolute paths",
    "output artifact paths",
    "conformer data",
    "DPA3 structures",
    "model input tensors",
    "credentials",
)

FORBIDDEN_MARKERS = (
    ".csv",
    ".jsonl",
    ".parquet",
    ".lmdb",
    ".pdf",
    "/home/",
    "/Users/",
    "C:\\",
    "InChI=",
    "InChIKey=",
    "SMILES=",
    "C1=CC",
    "0.72",
    "Authorization:",
    "Bearer ",
    "token=",
    "secret=",
    "password=",
    "cookie=",
    "raw article text:",
    "raw table:",
    "serialized training row:",
    "serialized dataset row:",
    "conformer block:",
    "dpa3 structure block:",
)


def test_execution_request_design_document_and_template_exist() -> None:
    assert DESIGN_DOC.exists()
    assert TEMPLATE_DOC.exists()


def test_execution_request_design_document_has_required_sections_chain_and_boundaries() -> None:
    text = DESIGN_DOC.read_text(encoding="utf-8")

    for section in REQUIRED_SECTIONS:
        assert section in text
    assert CHAIN_PHRASE in text
    for statement in BOUNDARY_STATEMENTS:
        assert statement in text


def test_execution_request_design_template_has_placeholders_boundaries_and_schema_labels() -> None:
    text = TEMPLATE_DOC.read_text(encoding="utf-8")

    for placeholder in REQUIRED_PLACEHOLDERS:
        assert placeholder in text
    for statement in BOUNDARY_STATEMENTS:
        assert statement in text
    for schema_label in FUTURE_SCHEMA_LABELS:
        assert schema_label in text


def test_governance_docs_link_execution_request_design_chain() -> None:
    for doc in GOVERNANCE_DOCS:
        assert CHAIN_PHRASE in doc.read_text(encoding="utf-8")


def test_dry_run_precheck_doc_points_to_execution_request_design_next() -> None:
    text = Path("docs/custom-corpus-property-training-dataset-controlled-writer-dry-run-precheck.md").read_text(
        encoding="utf-8"
    )

    assert "property training dataset controlled writer execution request design" in text
    assert "property training dataset controlled writer execution request" in text


def test_execution_request_design_contains_schema_authorization_and_status_semantics() -> None:
    text = DESIGN_DOC.read_text(encoding="utf-8")

    for schema_label in FUTURE_SCHEMA_LABELS:
        assert schema_label in text
    for phrase in REQUIRED_AUTHORIZATION_PHRASES:
        assert phrase in text
    for status_label in FUTURE_STATUS_LABELS:
        assert status_label in text
    for category in FORBIDDEN_CATEGORY_PHRASES:
        assert category in text


def test_execution_request_design_docs_do_not_include_sensitive_markers() -> None:
    combined = "\n".join(
        [
            DESIGN_DOC.read_text(encoding="utf-8"),
            TEMPLATE_DOC.read_text(encoding="utf-8"),
        ]
    )
    sanitized = combined.replace(
        "This controlled writer execution request design does not create CSV/JSONL/Parquet/LMDB artifacts.",
        "This controlled writer execution request design does not create FORMAT-LABEL artifacts.",
    )
    sanitized = sanitized.replace("CSV/JSONL/Parquet/LMDB artifacts", "FORMAT-LABEL artifacts")
    sanitized = sanitized.replace("candidate CSV/JSONL/Parquet/LMDB paths", "candidate FORMAT-LABEL paths")
    sanitized = sanitized.replace("training CSV/JSONL/Parquet/LMDB paths", "training FORMAT-LABEL paths")

    for marker in FORBIDDEN_MARKERS:
        assert marker not in sanitized
