from __future__ import annotations

from pathlib import Path


EVIDENCE_PATH = Path("docs/evidence/custom-corpus-small-public-quarantine-materialization-evidence-20260701.md")
TEMPLATE_PATH = Path("docs/evidence/templates/custom-corpus-small-public-quarantine-materialization-evidence-template.md")

REQUIRED_SECTIONS = (
    "# Small Public Quarantine Materialization Evidence",
    "## Scope",
    "## Public/Low-Risk Corpus Boundary",
    "## Input Evidence Chain",
    "## Quarantine Materialization Evidence",
    "## Training Dataset Boundary",
    "## Value Resolution Readiness",
    "## Redaction Review",
    "## Residual Risks",
    "## Next Gate",
    "## Operator Checklist",
)

FORBIDDEN_MARKERS = (
    ".csv",
    ".jsonl",
    ".parquet",
    ".lmdb",
    ".pdf",
    "/home/",
    "/Users/",
    "InChI=",
    "C1=CC",
    "0.72",
    "Authorization",
    "Bearer",
    "token",
    "secret",
    "serialized training row",
    "raw article text",
    "raw table",
)


def test_small_public_quarantine_evidence_has_required_sections_and_boundaries() -> None:
    evidence = EVIDENCE_PATH.read_text(encoding="utf-8")

    for section in REQUIRED_SECTIONS:
        assert section in evidence
    assert "| training_dataset_materialized | false |" in evidence
    assert "| dataset_artifact_created | false |" in evidence
    assert "| phase1_status | not_run |" in evidence
    assert "| dataset_confirmation_changed | false |" in evidence
    assert "This evidence packet does not create a training dataset." in evidence
    assert "This evidence packet does not execute a controlled writer." in evidence
    assert "This evidence packet does not serialize training rows." in evidence
    assert "This evidence packet does not create CSV/JSONL/Parquet/LMDB artifacts." in evidence
    assert "This evidence packet does not run Phase 1." in evidence
    assert "This evidence packet does not modify DatasetConfirmation." in evidence
    assert "This evidence packet does not run model training or evaluation." in evidence


def test_small_public_quarantine_evidence_and_template_are_redacted() -> None:
    combined = "\n".join(
        [
            EVIDENCE_PATH.read_text(encoding="utf-8"),
            TEMPLATE_PATH.read_text(encoding="utf-8"),
        ]
    )

    for marker in FORBIDDEN_MARKERS:
        assert marker not in combined


def test_small_public_quarantine_evidence_template_has_reusable_placeholders() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    for placeholder in (
        "<evidence_id>",
        "<date>",
        "<operator>",
        "<corpus_id>",
        "<dataset_name>",
        "<candidate_record_count>",
        "<quarantined_record_count>",
        "<admitted_record_count>",
        "<materialization_dry_run_status>",
        "<value_resolution_dry_run_status>",
        "<value_resolution_dry_run_precheck_status>",
        "<redaction_status>",
        "<residual_risks>",
        "<next_gate_decision>",
    ):
        assert placeholder in template
    assert "public source boundary confirmed" in template
    assert "no private paths" in template
    assert "no exact property values" in template
    assert "no canonical SMILES" in template
    assert "no InChI/InChIKey" in template
    assert "no PDF names" in template
    assert "no article/table text" in template
    assert "no row serialization" in template
    assert "no dataset artifact paths" in template
    assert "no conformer/DPA3 artifacts" in template
    assert "no model training" in template
    assert "no Phase 1" in template
    assert "no DatasetConfirmation mutation" in template
