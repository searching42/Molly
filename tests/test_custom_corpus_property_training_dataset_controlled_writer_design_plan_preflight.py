from __future__ import annotations

import inspect
import io
import json
from pathlib import Path
from typing import Any

import pytest

import ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_design_plan_preflight as preflight_module
from ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_design_plan_preflight import (
    main,
    preflight_property_training_dataset_controlled_writer_design_plan,
)


_PLAN_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_design_plan.v1"
_PREFLIGHT_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_design_plan_preflight.v1"
_DOC = Path("docs/custom-corpus-property-training-dataset-controlled-writer-design-plan-preflight.md")
_TEMPLATE = Path(
    "docs/evidence/templates/"
    "custom-corpus-property-training-dataset-controlled-writer-design-plan-preflight-evidence-template.md"
)
_GOVERNANCE_DOCS = (
    Path("docs/custom-corpus-dataset-materialization-boundary.md"),
    Path("docs/custom-corpus-governance-runbook.md"),
    Path("docs/custom-corpus-governance-stage-summary-20260628.md"),
    Path("docs/custom-corpus-materialization-schema.md"),
    Path("docs/custom-corpus-property-training-dataset-controlled-writer-design-plan.md"),
    Path("docs/phase-1-4-milestone-status.md"),
)
_CHAIN_PHRASE = (
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
    "-> future controlled writer execution request\n"
    "-> future controlled writer execution request preflight\n"
    "-> future explicitly confirmed controlled writer execution"
)
_REQUIRED_DOC_SECTIONS = (
    "# Custom Corpus Property Training Dataset Controlled Writer Design Plan Preflight",
    "## Purpose",
    "## Position in the Governance Chain",
    "## Input Package",
    "## Preflight Checks",
    "## Status Semantics",
    "## Redaction Policy",
    "## CLI Usage",
    "## Outputs",
    "## Blocked Conditions",
    "## Out of Scope",
    "## Next Step",
)
_REQUIRED_TEMPLATE_PLACEHOLDERS = (
    "<controlled_writer_design_plan_preflight_evidence_id>",
    "<date>",
    "<operator>",
    "<corpus_id>",
    "<dataset_name>",
    "<design_plan_id>",
    "<design_plan_status>",
    "<preflight_status>",
    "<quarantined_candidate_admission_boundary_status>",
    "<domain_validation_boundary_status>",
    "<controlled_writer_value_resolution_dry_run_precheck_status>",
    "<accepted_candidate_record_count>",
    "<needs_review_candidate_record_count>",
    "<blocked_candidate_record_count>",
    "<values_resolved>",
    "<missing_required_field_count>",
    "<redaction_status>",
    "<next_gate_decision>",
    "<residual_risks>",
)
_PREFLIGHT_BOUNDARY_STATEMENTS = (
    "This controlled writer design plan preflight does not implement the controlled writer.",
    "This controlled writer design plan preflight does not execute the controlled writer.",
    "This controlled writer design plan preflight does not run a writer dry-run.",
    "This controlled writer design plan preflight does not emit raw values.",
    "This controlled writer design plan preflight does not materialize values.",
    "This controlled writer design plan preflight does not serialize training rows.",
    "This controlled writer design plan preflight does not create training dataset artifacts.",
    "This controlled writer design plan preflight does not create CSV/JSONL/Parquet/LMDB artifacts.",
    "This controlled writer design plan preflight does not generate conformers.",
    "This controlled writer design plan preflight does not generate DPA3 structures.",
    "This controlled writer design plan preflight does not run Phase 1.",
    "This controlled writer design plan preflight does not modify DatasetConfirmation.",
    "This controlled writer design plan preflight does not run model training or evaluation.",
)
_REQUIRED_UPSTREAM_STATUSES = (
    "quarantined_candidate_admission_boundary_status",
    "domain_validation_boundary_status",
    "controlled_writer_value_resolution_dry_run_precheck_status",
    "property_unit_compatibility_status",
    "numeric_plausibility_status",
    "provenance_consistency_status",
    "compound_alias_association_status",
    "duplicate_conflict_status",
    "redaction_status",
)
_SOURCE_PACKAGE_REFS = (
    "row_contract_id",
    "materialization_plan_id",
    "writer_execution_request_id",
    "writer_input_binding_plan_id",
    "value_source_manifest_id",
    "controlled_writer_execution_plan_id",
    "value_resolution_dry_run_id",
)
_BOUNDARY_FLAGS = (
    "controlled_writer_implemented",
    "controlled_writer_executed",
    "writer_dry_run_executed",
    "values_materialized",
    "serialized_rows_created",
    "training_dataset_materialized",
    "dataset_artifact_created",
    "dataset_confirmation_changed",
    "model_training_run",
    "evaluation_run",
)
_FORBIDDEN_MARKERS = (
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


def test_valid_design_plan_package_returns_passed(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["schema_version"] == _PREFLIGHT_SCHEMA
    assert summary["preflight_status"] == "passed"
    assert summary["design_plan_id"] == "property-training-controlled-writer-design-plan-001"
    assert summary["design_plan_status"] == "passed"
    assert summary["corpus_id"] == "safe-corpus-id"
    assert summary["dataset_name"] == "safe-dataset-name"
    assert summary["accepted_candidate_record_count"] == 3
    assert summary["needs_review_candidate_record_count"] == 0
    assert summary["blocked_candidate_record_count"] == 0
    assert summary["redaction_status"] == "passed"
    assert summary["controlled_writer_implemented"] is False
    assert summary["controlled_writer_executed"] is False
    assert summary["writer_dry_run_executed"] is False
    assert summary["values_materialized"] is False
    assert summary["serialized_rows_created"] is False
    assert summary["training_dataset_materialized"] is False
    assert summary["dataset_artifact_created"] is False
    assert summary["phase1_status"] == "not_run"
    assert summary["dataset_confirmation_changed"] is False
    assert summary["model_training_run"] is False
    assert summary["evaluation_run"] is False
    assert summary["preflight_errors"] == []
    assert summary["preflight_warnings"] == []


def test_design_plan_preflight_docs_and_template_exist() -> None:
    assert _DOC.exists()
    assert _TEMPLATE.exists()


def test_design_plan_preflight_doc_has_required_sections_and_boundaries() -> None:
    text = _DOC.read_text(encoding="utf-8")

    for section in _REQUIRED_DOC_SECTIONS:
        assert section in text
    for statement in _PREFLIGHT_BOUNDARY_STATEMENTS:
        assert statement in text
    assert "controlled writer dry-run design" in text


def test_design_plan_preflight_template_has_placeholders_and_boundaries() -> None:
    text = _TEMPLATE.read_text(encoding="utf-8")

    for placeholder in _REQUIRED_TEMPLATE_PLACEHOLDERS:
        assert placeholder in text
    for statement in _PREFLIGHT_BOUNDARY_STATEMENTS:
        assert statement in text


def test_governance_docs_link_design_plan_preflight_chain() -> None:
    for doc in _GOVERNANCE_DOCS:
        assert _CHAIN_PHRASE in doc.read_text(encoding="utf-8")


def test_valid_package_writes_summary_json_and_redacted_markdown(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    output_summary = tmp_path / "design-plan-preflight-summary.json"
    output_markdown = tmp_path / "design-plan-preflight-evidence.md"

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path,
        output_summary_path=output_summary,
        output_markdown_path=output_markdown,
    )

    written = json.loads(output_summary.read_text(encoding="utf-8"))
    markdown = output_markdown.read_text(encoding="utf-8")
    serialized = json.dumps(summary, sort_keys=True)

    assert written == summary
    assert summary["controlled_writer_design_plan_path"] == plan_path.name
    assert str(tmp_path) not in serialized
    assert "controlled writer design plan preflight only" in markdown
    assert "controlled writer was not implemented" in markdown
    assert "writer dry-run was not executed" in markdown
    assert str(tmp_path) not in markdown


def test_summary_uses_safe_ids_and_basenames_only(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["controlled_writer_design_plan_path"] == "controlled-writer-design-plan.json"
    assert str(tmp_path) not in serialized
    assert "/" not in summary["controlled_writer_design_plan_path"]


def test_wrong_design_plan_schema_blocks(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(plan_path, lambda payload: payload.__setitem__("schema_version", "wrong"))

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "controlled_writer_design_plan_schema_invalid" in summary["preflight_errors"]


@pytest.mark.parametrize("field", ["design_plan_id", "upstream_evidence", "candidate_counts"])
def test_missing_required_top_level_fields_block(tmp_path: Path, field: str) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(plan_path, lambda payload: payload.pop(field))

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert f"{field}_missing" in summary["preflight_errors"]


def test_unsafe_design_plan_id_blocks(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(plan_path, lambda payload: payload.__setitem__("design_plan_id", "unsafe id"))

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "design_plan_id_unsafe" in summary["preflight_errors"]


@pytest.mark.parametrize("status", ["unknown", "", None])
def test_invalid_design_plan_status_blocks(tmp_path: Path, status: object) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(plan_path, lambda payload: payload.__setitem__("design_plan_status", status))

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "design_plan_status_invalid" in summary["preflight_errors"]


@pytest.mark.parametrize("status", ["blocked", "failed"])
def test_blocked_design_plan_status_blocks(tmp_path: Path, status: str) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(plan_path, lambda payload: payload.__setitem__("design_plan_status", status))

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "design_plan_blocked" in summary["preflight_errors"]


def test_needs_review_design_plan_status_blocks_by_default(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(plan_path, lambda payload: payload.__setitem__("design_plan_status", "needs_review"))

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "design_plan_needs_review" in summary["preflight_errors"]


def test_needs_review_design_plan_status_allowed_with_flag(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(plan_path, lambda payload: payload.__setitem__("design_plan_status", "needs_review"))

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path,
        allow_design_plan_needs_review=True,
    )

    assert summary["preflight_status"] == "needs_review"
    assert "design_plan_needs_review" in summary["preflight_warnings"]


def test_missing_upstream_evidence_blocks(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(plan_path, lambda payload: payload["upstream_evidence"].pop("redaction_status"))

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "redaction_status_missing" in summary["preflight_errors"]


@pytest.mark.parametrize("status", ["blocked", "failed", None])
def test_domain_validation_boundary_status_missing_or_blocked_blocks(tmp_path: Path, status: object) -> None:
    plan_path = _write_plan(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        if status is None:
            payload["upstream_evidence"].pop("domain_validation_boundary_status")
        else:
            payload["upstream_evidence"]["domain_validation_boundary_status"] = status

    _mutate_plan(plan_path, mutate)

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert (
        "domain_validation_boundary_status_missing" in summary["preflight_errors"]
        or "domain_validation_boundary_status_blocked" in summary["preflight_errors"]
    )


def test_domain_validation_boundary_status_needs_review_blocks_by_default(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _set_upstream(plan_path, "domain_validation_boundary_status", "needs_review")

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "domain_validation_boundary_status_needs_review" in summary["preflight_errors"]


def test_domain_validation_boundary_status_needs_review_allowed_with_flag(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _set_upstream(plan_path, "domain_validation_boundary_status", "needs_review")

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path,
        allow_design_plan_needs_review=True,
    )

    assert summary["preflight_status"] == "needs_review"
    assert "domain_validation_boundary_status_needs_review" in summary["preflight_warnings"]


def test_controlled_writer_value_resolution_dry_run_precheck_status_blocked_blocks(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _set_upstream(plan_path, "controlled_writer_value_resolution_dry_run_precheck_status", "blocked")

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "controlled_writer_value_resolution_dry_run_precheck_status_blocked" in summary["preflight_errors"]


def test_values_resolved_false_blocks_by_default(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(
        plan_path,
        lambda payload: payload["value_resolution_contract"].__setitem__("values_resolved", False),
    )

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "values_not_resolved" in summary["preflight_errors"]


def test_values_resolved_false_allowed_only_when_not_required(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(
        plan_path,
        lambda payload: payload["value_resolution_contract"].__setitem__("values_resolved", False),
    )

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path,
        require_values_resolved=False,
    )

    assert summary["preflight_status"] == "needs_review"
    assert "values_not_resolved" in summary["preflight_warnings"]


def test_missing_required_field_count_blocks_by_default(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(
        plan_path,
        lambda payload: payload["value_resolution_contract"].__setitem__("missing_required_field_count", 1),
    )

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "missing_required_fields" in summary["preflight_errors"]


def test_accepted_candidate_record_count_below_minimum_blocks(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(
        plan_path,
        lambda payload: payload["candidate_counts"].__setitem__("accepted_candidate_record_count", 0),
    )

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "minimum_accepted_candidate_records_not_met" in summary["preflight_errors"]


def test_blocked_candidate_record_count_blocks(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(
        plan_path,
        lambda payload: payload["candidate_counts"].__setitem__("blocked_candidate_record_count", 1),
    )

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "blocked_candidate_records_present" in summary["preflight_errors"]


def test_needs_review_candidate_count_blocks_by_default(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(
        plan_path,
        lambda payload: payload["candidate_counts"].__setitem__("needs_review_candidate_record_count", 1),
    )

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "needs_review_candidate_records_present" in summary["preflight_errors"]


def test_needs_review_candidate_count_allowed_with_explicit_allowance(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(
        plan_path,
        lambda payload: payload["candidate_counts"].__setitem__("needs_review_candidate_record_count", 1),
    )

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path,
        allow_design_plan_needs_review=True,
    )

    assert summary["preflight_status"] == "needs_review"
    assert "needs_review_candidate_records_present" in summary["preflight_warnings"]


def test_missing_source_package_refs_block(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(plan_path, lambda payload: payload["source_package_refs"].pop("row_contract_id"))

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "row_contract_id_missing" in summary["preflight_errors"]


def test_unsafe_source_package_ref_id_blocks(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(
        plan_path,
        lambda payload: payload["source_package_refs"].__setitem__("row_contract_id", "unsafe id"),
    )

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "row_contract_id_unsafe" in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("field", "error_code"),
    [
        ("controlled_writer_implemented", "controlled_writer_implemented"),
        ("controlled_writer_executed", "controlled_writer_executed"),
        ("writer_dry_run_executed", "writer_dry_run_executed"),
        ("values_materialized", "values_materialized"),
        ("serialized_rows_created", "serialized_rows_created"),
        ("training_dataset_materialized", "training_dataset_materialized"),
        ("dataset_artifact_created", "dataset_artifact_created"),
        ("dataset_confirmation_changed", "dataset_confirmation_changed"),
        ("model_training_run", "model_training_run"),
        ("evaluation_run", "evaluation_run"),
    ],
)
def test_true_boundary_flags_block(tmp_path: Path, field: str, error_code: str) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(plan_path, lambda payload: payload["boundary_flags"].__setitem__(field, True))

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


def test_phase1_status_other_than_not_run_blocks(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(plan_path, lambda payload: payload["boundary_flags"].__setitem__("phase1_status", "ran"))

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )

    assert summary["preflight_status"] == "blocked"
    assert "phase1_ran" in summary["preflight_errors"]


@pytest.mark.parametrize("marker", _FORBIDDEN_MARKERS)
def test_every_forbidden_marker_blocks_without_echoing_sensitive_value(tmp_path: Path, marker: str) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(plan_path, lambda payload: payload.__setitem__("unsafe_note", f"unsafe {marker}"))

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path
    )
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["preflight_status"] == "blocked"
    assert "controlled_writer_design_plan_contains_unsafe_material" in summary["preflight_errors"]
    assert marker not in serialized


def test_redaction_failure_returns_minimal_summary_and_no_unsafe_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = _write_plan(tmp_path)
    output_summary = tmp_path / "summary.json"
    output_markdown = tmp_path / "evidence.md"
    monkeypatch.setattr(preflight_module, "_markdown", lambda summary: "unsafe .pdf")

    summary = preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path,
        output_summary_path=output_summary,
        output_markdown_path=output_markdown,
    )

    assert summary == {
        "schema_version": _PREFLIGHT_SCHEMA,
        "preflight_status": "blocked",
        "preflight_errors": [
            "property_training_dataset_controlled_writer_design_plan_preflight_redaction_failed"
        ],
        "redaction_status": "failed",
    }
    assert json.loads(output_summary.read_text(encoding="utf-8")) == summary
    assert not output_markdown.exists()


def test_cli_returns_zero_for_passed_and_stdout_is_valid_json(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    stdout = io.StringIO()

    exit_code = main(["--controlled-writer-design-plan", str(plan_path)], stdout=stdout, stderr=io.StringIO())
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["preflight_status"] == "passed"


def test_cli_returns_one_for_blocked(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(plan_path, lambda payload: payload.__setitem__("schema_version", "wrong"))
    stdout = io.StringIO()

    exit_code = main(["--controlled-writer-design-plan", str(plan_path)], stdout=stdout, stderr=io.StringIO())
    payload = json.loads(stdout.getvalue())

    assert exit_code == 1
    assert payload["preflight_status"] == "blocked"


def test_cli_allows_needs_review_with_flag(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    _mutate_plan(plan_path, lambda payload: payload.__setitem__("design_plan_status", "needs_review"))
    stdout = io.StringIO()

    exit_code = main(
        ["--controlled-writer-design-plan", str(plan_path), "--allow-design-plan-needs-review"],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["preflight_status"] == "needs_review"


def test_no_training_or_candidate_artifacts_are_created(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    output_summary = tmp_path / "summary.json"
    output_markdown = tmp_path / "evidence.md"

    preflight_property_training_dataset_controlled_writer_design_plan(
        controlled_writer_design_plan_path=plan_path,
        output_summary_path=output_summary,
        output_markdown_path=output_markdown,
    )

    forbidden_suffixes = {".csv", ".jsonl", ".parquet", ".lmdb"}
    assert {path.suffix for path in tmp_path.iterdir()} <= {".json", ".md"}
    assert not any(path.suffix in forbidden_suffixes for path in tmp_path.iterdir())


def test_no_conformer_dpa3_or_external_workflow_imports() -> None:
    source = inspect.getsource(preflight_module)
    import_lines = "\n".join(
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ).lower()

    for forbidden in (
        "mineru",
        "openai",
        "pdf",
        "workflow",
        "rdkit",
        "chem",
        "controlled_writer_dry_run",
        "controlled_writer_execution",
        "training_dataset_writer",
        "evaluation",
    ):
        assert forbidden not in import_lines


def _write_plan(tmp_path: Path) -> Path:
    path = tmp_path / "controlled-writer-design-plan.json"
    path.write_text(json.dumps(_valid_plan(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _valid_plan() -> dict[str, Any]:
    return {
        "schema_version": _PLAN_SCHEMA,
        "design_plan_id": "property-training-controlled-writer-design-plan-001",
        "design_plan_status": "passed",
        "corpus_id": "safe-corpus-id",
        "dataset_name": "safe-dataset-name",
        "upstream_evidence": {field: "passed" for field in _REQUIRED_UPSTREAM_STATUSES},
        "candidate_counts": {
            "accepted_candidate_record_count": 3,
            "needs_review_candidate_record_count": 0,
            "blocked_candidate_record_count": 0,
        },
        "source_package_refs": {
            "row_contract_id": "row-contract-001",
            "materialization_plan_id": "materialization-plan-001",
            "writer_execution_request_id": "writer-execution-request-001",
            "writer_input_binding_plan_id": "writer-input-binding-plan-001",
            "value_source_manifest_id": "value-source-manifest-001",
            "controlled_writer_execution_plan_id": "controlled-writer-execution-plan-001",
            "value_resolution_dry_run_id": "value-resolution-dry-run-001",
        },
        "value_resolution_contract": {
            "controlled_writer_value_resolution_dry_run_status": "passed",
            "values_resolved": True,
            "missing_required_field_count": 0,
        },
        "boundary_flags": {
            "controlled_writer_implemented": False,
            "controlled_writer_executed": False,
            "writer_dry_run_executed": False,
            "values_materialized": False,
            "serialized_rows_created": False,
            "training_dataset_materialized": False,
            "dataset_artifact_created": False,
            "phase1_status": "not_run",
            "dataset_confirmation_changed": False,
            "model_training_run": False,
            "evaluation_run": False,
        },
        "redaction_status": "passed",
    }


def _mutate_plan(path: Path, mutator: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _set_upstream(path: Path, field: str, value: object) -> None:
    _mutate_plan(path, lambda payload: payload["upstream_evidence"].__setitem__(field, value))
