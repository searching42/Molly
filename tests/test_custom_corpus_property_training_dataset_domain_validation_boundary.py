from __future__ import annotations

from pathlib import Path


BOUNDARY_DOC = Path("docs/custom-corpus-property-training-dataset-domain-validation-boundary.md")
TEMPLATE_DOC = Path(
    "docs/evidence/templates/custom-corpus-property-training-dataset-domain-validation-boundary-evidence-template.md"
)

REQUIRED_SECTIONS = (
    "# Custom Corpus Property Training Dataset Domain Validation Boundary",
    "## Purpose",
    "## Position in the Governance Chain",
    "## Required Upstream Evidence",
    "## Domain Validation Scope",
    "## Property-Unit Compatibility Boundary",
    "## Numeric Plausibility Boundary",
    "## Provenance and Condition Boundary",
    "## Compound and Alias Association Boundary",
    "## Duplicate and Conflict Boundary",
    "## Allowed Evidence",
    "## Disallowed Evidence",
    "## Pass Criteria",
    "## Needs-Review Criteria",
    "## Fail Criteria",
    "## Residual Risks",
    "## Next Step",
)

REQUIRED_PLACEHOLDERS = (
    "<domain_validation_boundary_evidence_id>",
    "<date>",
    "<operator>",
    "<corpus_id>",
    "<dataset_name>",
    "<quarantined_candidate_admission_boundary_status>",
    "<candidate_record_count>",
    "<accepted_candidate_record_count>",
    "<needs_review_candidate_record_count>",
    "<blocked_candidate_record_count>",
    "<property_unit_compatibility_status>",
    "<property_unit_compatibility_pass_count>",
    "<property_unit_compatibility_needs_review_count>",
    "<property_unit_compatibility_fail_count>",
    "<numeric_plausibility_status>",
    "<numeric_plausibility_pass_count>",
    "<numeric_plausibility_needs_review_count>",
    "<numeric_plausibility_fail_count>",
    "<provenance_consistency_status>",
    "<condition_completeness_status>",
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
    "-> future controlled training dataset writer implementation"
)

BOUNDARY_STATEMENTS = (
    "This domain validation boundary does not execute a controlled writer.",
    "This domain validation boundary does not emit raw values.",
    "This domain validation boundary does not materialize values.",
    "This domain validation boundary does not serialize training rows.",
    "This domain validation boundary does not create training dataset artifacts.",
    "This domain validation boundary does not create CSV/JSONL/Parquet/LMDB artifacts.",
    "This domain validation boundary does not generate conformers.",
    "This domain validation boundary does not generate DPA3 structures.",
    "This domain validation boundary does not run Phase 1.",
    "This domain validation boundary does not modify DatasetConfirmation.",
    "This domain validation boundary does not run model training or evaluation.",
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
    Path("docs/custom-corpus-property-training-dataset-quarantined-candidate-admission-boundary.md"),
    Path("docs/phase-1-4-milestone-status.md"),
)


def test_domain_boundary_document_and_template_exist() -> None:
    assert BOUNDARY_DOC.exists()
    assert TEMPLATE_DOC.exists()


def test_domain_boundary_document_has_required_sections_and_statements() -> None:
    text = BOUNDARY_DOC.read_text(encoding="utf-8")

    for section in REQUIRED_SECTIONS:
        assert section in text
    for statement in BOUNDARY_STATEMENTS:
        assert statement in text
    assert "property_unit_compatibility_status=passed" in text
    assert "numeric_plausibility_status=passed" in text
    assert "compound_alias_association_status=passed" in text
    assert "controlled_writer_executed=false" in text
    assert "dataset_confirmation_changed=false" in text


def test_domain_boundary_template_has_placeholders_and_statements() -> None:
    text = TEMPLATE_DOC.read_text(encoding="utf-8")

    for placeholder in REQUIRED_PLACEHOLDERS:
        assert placeholder in text
    for statement in BOUNDARY_STATEMENTS:
        assert statement in text


def test_governance_docs_link_domain_boundary_chain() -> None:
    for doc in GOVERNANCE_DOCS:
        assert CHAIN_PHRASE in doc.read_text(encoding="utf-8")


def test_domain_boundary_docs_do_not_include_forbidden_markers() -> None:
    combined = "\n".join(
        [
            BOUNDARY_DOC.read_text(encoding="utf-8"),
            TEMPLATE_DOC.read_text(encoding="utf-8"),
        ]
    )
    sanitized = combined.replace(
        "This domain validation boundary does not create CSV/JSONL/Parquet/LMDB artifacts.",
        "This domain validation boundary does not create FORMAT-LABEL artifacts.",
    )
    sanitized = sanitized.replace(
        "candidate or training CSV/JSONL/Parquet/LMDB",
        "candidate or training FORMAT-LABEL",
    )

    for marker in FORBIDDEN_MARKERS:
        assert marker not in sanitized
