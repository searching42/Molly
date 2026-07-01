from __future__ import annotations

from pathlib import Path


DESIGN_DOC = Path("docs/custom-corpus-property-training-dataset-controlled-writer-dry-run-design.md")
TEMPLATE_DOC = Path(
    "docs/evidence/templates/"
    "custom-corpus-property-training-dataset-controlled-writer-dry-run-design-evidence-template.md"
)

REQUIRED_SECTIONS = (
    "# Custom Corpus Property Training Dataset Controlled Writer Dry-Run Design",
    "## Purpose",
    "## Position in the Governance Chain",
    "## Required Upstream Evidence",
    "## Dry-Run Design Scope",
    "## Future Dry-Run Input Contract",
    "## Future Dry-Run Report Contract",
    "## Future Dry-Run Summary Contract",
    "## Allowed Future Dry-Run Outputs",
    "## Disallowed Current Outputs",
    "## Side-Effect Boundary",
    "## Redaction and Non-Leakage Policy",
    "## Dry-Run Status Semantics",
    "## Future Dry-Run Precheck Expectations",
    "## Implementation Blockers",
    "## Pass Criteria",
    "## Needs-Review Criteria",
    "## Fail Criteria",
    "## Residual Risks",
    "## Next Step",
)

REQUIRED_PLACEHOLDERS = (
    "<controlled_writer_dry_run_design_evidence_id>",
    "<date>",
    "<operator>",
    "<corpus_id>",
    "<dataset_name>",
    "<controlled_writer_design_plan_preflight_status>",
    "<domain_validation_boundary_status>",
    "<controlled_writer_value_resolution_dry_run_precheck_status>",
    "<accepted_candidate_record_count>",
    "<needs_review_candidate_record_count>",
    "<blocked_candidate_record_count>",
    "<future_dry_run_report_schema>",
    "<future_dry_run_summary_schema>",
    "<redaction_status>",
    "<next_gate_decision>",
    "<residual_risks>",
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
    "-> future controlled writer dry-run precheck\n"
    "-> future controlled writer execution request\n"
    "-> future explicitly confirmed controlled writer execution"
)

BOUNDARY_STATEMENTS = (
    "This controlled writer dry-run design does not implement the dry-run.",
    "This controlled writer dry-run design does not execute a dry-run.",
    "This controlled writer dry-run design does not implement the controlled writer.",
    "This controlled writer dry-run design does not execute the controlled writer.",
    "This controlled writer dry-run design does not emit raw values.",
    "This controlled writer dry-run design does not materialize values.",
    "This controlled writer dry-run design does not serialize training rows.",
    "This controlled writer dry-run design does not create training dataset artifacts.",
    "This controlled writer dry-run design does not create CSV/JSONL/Parquet/LMDB artifacts.",
    "This controlled writer dry-run design does not generate conformers.",
    "This controlled writer dry-run design does not generate DPA3 structures.",
    "This controlled writer dry-run design does not run Phase 1.",
    "This controlled writer dry-run design does not modify DatasetConfirmation.",
    "This controlled writer dry-run design does not run model training or evaluation.",
)

GOVERNANCE_DOCS = (
    Path("docs/custom-corpus-dataset-materialization-boundary.md"),
    Path("docs/custom-corpus-governance-runbook.md"),
    Path("docs/custom-corpus-governance-stage-summary-20260628.md"),
    Path("docs/custom-corpus-materialization-schema.md"),
    Path("docs/custom-corpus-property-training-dataset-controlled-writer-design-plan.md"),
    Path("docs/custom-corpus-property-training-dataset-controlled-writer-design-plan-preflight.md"),
    Path("docs/phase-1-4-milestone-status.md"),
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
    "InChIKey",
    "SMILES",
    "C1=CC",
    "0.72",
    "Authorization",
    "Bearer",
    "token",
    "secret",
    "password",
    "cookie",
    "raw article text",
    "raw table",
    "serialized training row",
    "serialized dataset row",
    "conformer block",
    "dpa3 structure block",
)

FUTURE_REPORT_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_dry_run_report.v1"
FUTURE_SUMMARY_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_dry_run_summary.v1"


def test_dry_run_design_document_and_template_exist() -> None:
    assert DESIGN_DOC.exists()
    assert TEMPLATE_DOC.exists()


def test_dry_run_design_document_has_required_sections_boundaries_and_schema_labels() -> None:
    text = DESIGN_DOC.read_text(encoding="utf-8")

    for section in REQUIRED_SECTIONS:
        assert section in text
    for statement in BOUNDARY_STATEMENTS:
        assert statement in text
    assert FUTURE_REPORT_SCHEMA in text
    assert FUTURE_SUMMARY_SCHEMA in text


def test_dry_run_design_template_has_placeholders_boundaries_and_schema_labels() -> None:
    text = TEMPLATE_DOC.read_text(encoding="utf-8")

    for placeholder in REQUIRED_PLACEHOLDERS:
        assert placeholder in text
    for statement in BOUNDARY_STATEMENTS:
        assert statement in text
    assert FUTURE_REPORT_SCHEMA in text
    assert FUTURE_SUMMARY_SCHEMA in text


def test_governance_docs_link_dry_run_design_chain() -> None:
    for doc in GOVERNANCE_DOCS:
        assert CHAIN_PHRASE in doc.read_text(encoding="utf-8")


def test_dry_run_design_docs_do_not_include_forbidden_markers() -> None:
    combined = "\n".join(
        [
            DESIGN_DOC.read_text(encoding="utf-8"),
            TEMPLATE_DOC.read_text(encoding="utf-8"),
        ]
    )
    sanitized = combined.replace(
        "This controlled writer dry-run design does not create CSV/JSONL/Parquet/LMDB artifacts.",
        "This controlled writer dry-run design does not create FORMAT-LABEL artifacts.",
    )
    sanitized = sanitized.replace(
        "CSV/JSONL/Parquet/LMDB artifacts",
        "FORMAT-LABEL artifacts",
    )
    sanitized = sanitized.replace(
        "would_create_csv_jsonl_parquet_lmdb=false",
        "would_create_format_label=false",
    )

    for marker in FORBIDDEN_MARKERS:
        assert marker not in sanitized
