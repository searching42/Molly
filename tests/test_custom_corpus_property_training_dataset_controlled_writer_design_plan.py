from __future__ import annotations

from pathlib import Path


DESIGN_PLAN_DOC = Path(
    "docs/custom-corpus-property-training-dataset-controlled-writer-design-plan.md"
)
TEMPLATE_DOC = Path(
    "docs/evidence/templates/"
    "custom-corpus-property-training-dataset-controlled-writer-design-plan-evidence-template.md"
)

REQUIRED_SECTIONS = (
    "# Custom Corpus Property Training Dataset Controlled Writer Design Plan",
    "## Purpose",
    "## Position in the Governance Chain",
    "## Required Upstream Evidence",
    "## Writer Design Scope",
    "## Input Package Contract",
    "## Admission and Domain Validation Contract",
    "## Value Resolution Contract",
    "## Output Artifact Policy",
    "## Dry-Run-First Policy",
    "## Confirmation and Operator Control",
    "## Redaction and Non-Leakage Requirements",
    "## Allowed Future Writer Outputs",
    "## Disallowed Current Outputs",
    "## Implementation Blockers",
    "## Pass Criteria",
    "## Needs-Review Criteria",
    "## Fail Criteria",
    "## Residual Risks",
    "## Next Step",
)

REQUIRED_PLACEHOLDERS = (
    "<controlled_writer_design_plan_evidence_id>",
    "<date>",
    "<operator>",
    "<corpus_id>",
    "<dataset_name>",
    "<quarantined_candidate_admission_boundary_status>",
    "<domain_validation_boundary_status>",
    "<controlled_writer_value_resolution_dry_run_precheck_status>",
    "<accepted_candidate_record_count>",
    "<needs_review_candidate_record_count>",
    "<blocked_candidate_record_count>",
    "<row_contract_id>",
    "<materialization_plan_id>",
    "<writer_execution_request_id>",
    "<writer_input_binding_plan_id>",
    "<value_source_manifest_id>",
    "<controlled_writer_execution_plan_id>",
    "<value_resolution_dry_run_id>",
    "<property_unit_compatibility_status>",
    "<numeric_plausibility_status>",
    "<provenance_consistency_status>",
    "<compound_alias_association_status>",
    "<duplicate_conflict_status>",
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
    "-> property training dataset controlled writer dry-run precheck\n"
    "-> property training dataset controlled writer execution request design\n"
    "-> property training dataset controlled writer execution request\n"
    "-> future controlled writer execution request preflight\n"
    "-> future explicitly confirmed controlled writer execution"
)

BOUNDARY_STATEMENTS = (
    "This controlled writer design plan does not implement the controlled writer.",
    "This controlled writer design plan does not execute the controlled writer.",
    "This controlled writer design plan does not emit raw values.",
    "This controlled writer design plan does not materialize values.",
    "This controlled writer design plan does not serialize training rows.",
    "This controlled writer design plan does not create training dataset artifacts.",
    "This controlled writer design plan does not create CSV/JSONL/Parquet/LMDB artifacts.",
    "This controlled writer design plan does not generate conformers.",
    "This controlled writer design plan does not generate DPA3 structures.",
    "This controlled writer design plan does not run Phase 1.",
    "This controlled writer design plan does not modify DatasetConfirmation.",
    "This controlled writer design plan does not run model training or evaluation.",
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

GOVERNANCE_DOCS = (
    Path("docs/custom-corpus-dataset-materialization-boundary.md"),
    Path("docs/custom-corpus-governance-runbook.md"),
    Path("docs/custom-corpus-governance-stage-summary-20260628.md"),
    Path("docs/custom-corpus-materialization-schema.md"),
    Path("docs/custom-corpus-property-training-dataset-domain-validation-boundary.md"),
    Path("docs/phase-1-4-milestone-status.md"),
)


def test_controlled_writer_design_plan_document_and_template_exist() -> None:
    assert DESIGN_PLAN_DOC.exists()
    assert TEMPLATE_DOC.exists()


def test_controlled_writer_design_plan_has_required_sections_and_statements() -> None:
    text = DESIGN_PLAN_DOC.read_text(encoding="utf-8")

    for section in REQUIRED_SECTIONS:
        assert section in text
    for statement in BOUNDARY_STATEMENTS:
        assert statement in text
    assert "domain_validation_boundary_status=passed" in text
    assert "controlled_writer_value_resolution_dry_run_precheck_status=passed" in text
    assert "missing_required_field_count=0" in text
    assert "dataset_confirmation_changed=false" in text


def test_controlled_writer_design_plan_template_has_placeholders_and_statements() -> None:
    text = TEMPLATE_DOC.read_text(encoding="utf-8")

    for placeholder in REQUIRED_PLACEHOLDERS:
        assert placeholder in text
    for statement in BOUNDARY_STATEMENTS:
        assert statement in text


def test_governance_docs_link_controlled_writer_design_plan_chain() -> None:
    for doc in GOVERNANCE_DOCS:
        assert CHAIN_PHRASE in doc.read_text(encoding="utf-8")


def test_controlled_writer_design_plan_docs_do_not_include_forbidden_markers() -> None:
    combined = "\n".join(
        [
            DESIGN_PLAN_DOC.read_text(encoding="utf-8"),
            TEMPLATE_DOC.read_text(encoding="utf-8"),
        ]
    )
    sanitized = combined.replace(
        "This controlled writer design plan does not create CSV/JSONL/Parquet/LMDB artifacts.",
        "This controlled writer design plan does not create FORMAT-LABEL artifacts.",
    )
    sanitized = sanitized.replace(
        "candidate or training CSV/JSONL/Parquet/LMDB",
        "candidate or training FORMAT-LABEL",
    )

    for marker in FORBIDDEN_MARKERS:
        assert marker not in sanitized
