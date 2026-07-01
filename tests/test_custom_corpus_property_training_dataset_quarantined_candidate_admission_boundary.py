from __future__ import annotations

from pathlib import Path


BOUNDARY_DOC = Path("docs/custom-corpus-property-training-dataset-quarantined-candidate-admission-boundary.md")
TEMPLATE_DOC = Path(
    "docs/evidence/templates/"
    "custom-corpus-property-training-dataset-quarantined-candidate-admission-boundary-evidence-template.md"
)

REQUIRED_SECTIONS = (
    "# Custom Corpus Property Training Dataset Quarantined Candidate Admission Boundary",
    "## Purpose",
    "## Position in the Governance Chain",
    "## Required Upstream Evidence",
    "## Eligible Quarantined Candidate Criteria",
    "## Training Admission Boundary Criteria",
    "## Value Resolution Boundary Criteria",
    "## Public Evidence Boundary",
    "## Disallowed Outputs",
    "## Pass Criteria",
    "## Needs-Review Criteria",
    "## Fail Criteria",
    "## Residual Risks",
    "## Next Step",
)

REQUIRED_PLACEHOLDERS = (
    "<boundary_evidence_id>",
    "<date>",
    "<operator>",
    "<corpus_id>",
    "<dataset_name>",
    "<small_public_quarantine_evidence_status>",
    "<quarantine_candidate_preflight_status>",
    "<quarantine_candidate_record_count>",
    "<accepted_candidate_record_count>",
    "<needs_review_candidate_record_count>",
    "<blocked_candidate_record_count>",
    "<training_admission_readiness_status>",
    "<training_dataset_materialization_plan_precheck_status>",
    "<training_dataset_row_contract_precheck_status>",
    "<training_dataset_materialization_dry_run_precheck_status>",
    "<writer_execution_request_preflight_status>",
    "<writer_input_binding_plan_preflight_status>",
    "<writer_value_source_manifest_preflight_status>",
    "<controlled_writer_execution_plan_preflight_status>",
    "<controlled_writer_value_resolution_dry_run_precheck_status>",
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
    "This boundary evidence does not execute a controlled writer.",
    "This boundary evidence does not materialize values.",
    "This boundary evidence does not serialize training rows.",
    "This boundary evidence does not create training dataset artifacts.",
    "This boundary evidence does not create CSV/JSONL/Parquet/LMDB artifacts.",
    "This boundary evidence does not generate conformers.",
    "This boundary evidence does not generate DPA3 structures.",
    "This boundary evidence does not run Phase 1.",
    "This boundary evidence does not modify DatasetConfirmation.",
    "This boundary evidence does not run model training or evaluation.",
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
    Path("docs/phase-1-4-milestone-status.md"),
)


def test_boundary_document_and_template_exist() -> None:
    assert BOUNDARY_DOC.exists()
    assert TEMPLATE_DOC.exists()


def test_boundary_document_has_required_sections_and_boundary_statements() -> None:
    text = BOUNDARY_DOC.read_text(encoding="utf-8")

    for section in REQUIRED_SECTIONS:
        assert section in text
    for statement in BOUNDARY_STATEMENTS:
        assert statement in text
    assert "small_public_quarantine_evidence_status=passed" in text
    assert "controlled_writer_value_resolution_dry_run_precheck_status=passed" in text
    assert "controlled_writer_executed=false" in text
    assert "dataset_confirmation_changed=false" in text


def test_evidence_template_has_required_placeholders_and_boundaries() -> None:
    text = TEMPLATE_DOC.read_text(encoding="utf-8")

    for placeholder in REQUIRED_PLACEHOLDERS:
        assert placeholder in text
    for statement in BOUNDARY_STATEMENTS:
        assert statement in text


def test_governance_docs_link_new_boundary_chain() -> None:
    for doc in GOVERNANCE_DOCS:
        assert CHAIN_PHRASE in doc.read_text(encoding="utf-8")


def test_boundary_docs_do_not_include_forbidden_markers() -> None:
    combined = "\n".join(
        [
            BOUNDARY_DOC.read_text(encoding="utf-8"),
            TEMPLATE_DOC.read_text(encoding="utf-8"),
        ]
    )
    sanitized = combined.replace(
        "This boundary evidence does not create CSV/JSONL/Parquet/LMDB artifacts.",
        "This boundary evidence does not create FORMAT-LABEL artifacts.",
    )
    sanitized = sanitized.replace(
        "candidate or training CSV/JSONL/Parquet/LMDB",
        "candidate or training FORMAT-LABEL",
    )

    for marker in FORBIDDEN_MARKERS:
        assert marker not in sanitized
