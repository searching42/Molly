from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from ai4s_agent import custom_corpus_property_training_dataset_controlled_writer_dry_run_precheck as precheck_module
from ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_dry_run_precheck import (
    main,
    precheck_property_training_dataset_controlled_writer_dry_run,
)


REPORT_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_dry_run_report.v1"
SUMMARY_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_dry_run_summary.v1"
PREFLIGHT_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_dry_run_precheck.v1"

REPORT_BASENAME = "property_training_dataset_controlled_writer_dry_run_report.json"
SUMMARY_BASENAME = "property_training_dataset_controlled_writer_dry_run_summary.json"
EVIDENCE_BASENAME = "redacted_property_training_dataset_controlled_writer_dry_run_evidence.md"

DESIGN_DOC = Path("docs/custom-corpus-property-training-dataset-controlled-writer-dry-run-precheck.md")
TEMPLATE_DOC = Path(
    "docs/evidence/templates/"
    "custom-corpus-property-training-dataset-controlled-writer-dry-run-precheck-evidence-template.md"
)
MODULE_PATH = Path("src/ai4s_agent/custom_corpus_property_training_dataset_controlled_writer_dry_run_precheck.py")

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
    Path("docs/phase-1-4-milestone-status.md"),
)

DOC_SECTIONS = (
    "# Custom Corpus Property Training Dataset Controlled Writer Dry-Run Precheck",
    "## Purpose",
    "## Position in the Governance Chain",
    "## Input Package",
    "## Precheck Checks",
    "## Status Semantics",
    "## Hash and Basename Policy",
    "## Redaction Policy",
    "## CLI Usage",
    "## Outputs",
    "## Blocked Conditions",
    "## Out of Scope",
    "## Next Step",
)

TEMPLATE_PLACEHOLDERS = (
    "<controlled_writer_dry_run_precheck_evidence_id>",
    "<date>",
    "<operator>",
    "<corpus_id>",
    "<dataset_name>",
    "<dry_run_id>",
    "<dry_run_status>",
    "<precheck_status>",
    "<dry_run_report_basename>",
    "<dry_run_report_sha256>",
    "<dry_run_summary_basename>",
    "<accepted_candidate_record_count>",
    "<needs_review_candidate_record_count>",
    "<blocked_candidate_record_count>",
    "<required_field_count>",
    "<resolved_required_field_count>",
    "<missing_required_field_count>",
    "<would_write_row_count>",
    "<would_write_field_count>",
    "<redaction_status>",
    "<next_gate_decision>",
    "<residual_risks>",
)

BOUNDARY_STATEMENTS = (
    "This controlled writer dry-run precheck does not rerun the dry-run.",
    "This controlled writer dry-run precheck does not execute the controlled writer.",
    "This controlled writer dry-run precheck does not emit raw values.",
    "This controlled writer dry-run precheck does not materialize values.",
    "This controlled writer dry-run precheck does not serialize training rows.",
    "This controlled writer dry-run precheck does not create training dataset artifacts.",
    "This controlled writer dry-run precheck does not create CSV/JSONL/Parquet/LMDB artifacts.",
    "This controlled writer dry-run precheck does not generate conformers.",
    "This controlled writer dry-run precheck does not generate DPA3 structures.",
    "This controlled writer dry-run precheck does not run Phase 1.",
    "This controlled writer dry-run precheck does not modify DatasetConfirmation.",
    "This controlled writer dry-run precheck does not run model training or evaluation.",
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


def _valid_report() -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA,
        "dry_run_id": "controlled-writer-dry-run-001",
        "dry_run_status": "passed",
        "corpus_id": "safe-corpus-id",
        "dataset_name": "safe-dataset-name",
        "controlled_writer_design_plan_preflight_status": "passed",
        "domain_validation_boundary_status": "passed",
        "controlled_writer_value_resolution_dry_run_precheck_status": "passed",
        "accepted_candidate_record_count": 3,
        "needs_review_candidate_record_count": 0,
        "blocked_candidate_record_count": 0,
        "required_field_count": 5,
        "resolved_required_field_count": 5,
        "missing_required_field_count": 0,
        "would_write_row_count": 3,
        "would_write_field_count": 15,
        "would_create_training_dataset_artifact": False,
        "would_create_csv_jsonl_parquet_lmdb": False,
        "would_serialize_rows": False,
        "would_materialize_values": False,
        "controlled_writer_executed": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "serialized_rows_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "model_training_run": False,
        "evaluation_run": False,
        "redaction_status": "passed",
        "dry_run_errors": [],
        "dry_run_warnings": [],
    }


def _valid_summary(report_sha: str) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA,
        "dry_run_id": "controlled-writer-dry-run-001",
        "dry_run_status": "passed",
        "dry_run_report_basename": REPORT_BASENAME,
        "dry_run_report_sha256": report_sha,
        "corpus_id": "safe-corpus-id",
        "dataset_name": "safe-dataset-name",
        "accepted_candidate_record_count": 3,
        "needs_review_candidate_record_count": 0,
        "blocked_candidate_record_count": 0,
        "required_field_count": 5,
        "resolved_required_field_count": 5,
        "missing_required_field_count": 0,
        "would_write_row_count": 3,
        "would_write_field_count": 15,
        "redaction_status": "passed",
        "controlled_writer_executed": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "serialized_rows_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "model_training_run": False,
        "evaluation_run": False,
        "dry_run_errors": [],
        "dry_run_warnings": [],
    }


def _safe_evidence() -> str:
    return "\n".join(
        [
            "# Controlled Writer Dry-Run Evidence",
            "",
            "This controlled writer dry-run does not execute the controlled writer.",
            "Values were not materialized.",
            "No row payloads were emitted.",
        ]
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_package(
    tmp_path: Path,
    *,
    report: dict[str, Any] | None = None,
    summary_mutator: Any | None = None,
    evidence: str | None = None,
) -> dict[str, Path]:
    package_dir = tmp_path / "package"
    package_dir.mkdir(parents=True, exist_ok=True)
    report_path = package_dir / REPORT_BASENAME
    summary_path = package_dir / SUMMARY_BASENAME
    evidence_path = package_dir / EVIDENCE_BASENAME
    report_payload = report or _valid_report()
    _write_json(report_path, report_payload)
    summary_payload = _valid_summary(_sha256(report_path))
    if summary_mutator is not None:
        summary_mutator(summary_payload)
    _write_json(summary_path, summary_payload)
    evidence_path.write_text(evidence if evidence is not None else _safe_evidence(), encoding="utf-8")
    return {"report": report_path, "summary": summary_path, "evidence": evidence_path}


def _precheck(paths: dict[str, Path], **kwargs: Any) -> dict[str, Any]:
    return precheck_property_training_dataset_controlled_writer_dry_run(
        controlled_writer_dry_run_report_path=paths["report"],
        controlled_writer_dry_run_summary_path=paths["summary"],
        controlled_writer_dry_run_evidence_path=paths.get("evidence"),
        **kwargs,
    )


def _assert_no_artifact_files(root: Path) -> None:
    forbidden_suffixes = (".csv", ".jsonl", ".parquet", ".lmdb")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        assert not lowered.endswith(forbidden_suffixes)
        assert "conformer" not in lowered
        assert "dpa3" not in lowered


def test_valid_report_and_summary_return_passed_and_write_outputs(tmp_path: Path) -> None:
    paths = _write_package(tmp_path)
    output_summary = tmp_path / "precheck-summary.json"
    output_markdown = tmp_path / "precheck-evidence.md"

    summary = precheck_property_training_dataset_controlled_writer_dry_run(
        controlled_writer_dry_run_report_path=paths["report"],
        controlled_writer_dry_run_summary_path=paths["summary"],
        controlled_writer_dry_run_evidence_path=paths["evidence"],
        output_summary_path=output_summary,
        output_markdown_path=output_markdown,
    )

    assert summary["schema_version"] == PREFLIGHT_SCHEMA
    assert summary["precheck_status"] == "passed"
    assert summary["dry_run_report_basename"] == REPORT_BASENAME
    assert summary["dry_run_summary_basename"] == SUMMARY_BASENAME
    assert summary["dry_run_report_sha256"] == _sha256(paths["report"])
    assert json.loads(output_summary.read_text(encoding="utf-8")) == summary
    markdown = output_markdown.read_text(encoding="utf-8")
    assert "This controlled writer dry-run precheck does not rerun the dry-run." in markdown
    assert str(tmp_path) not in json.dumps(summary, sort_keys=True)
    _assert_no_artifact_files(tmp_path)


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("report", lambda payload: payload.__setitem__("schema_version", "wrong"), "controlled_writer_dry_run_report_schema_invalid"),
        ("summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "controlled_writer_dry_run_summary_schema_invalid"),
    ],
)
def test_wrong_schema_blocks(tmp_path: Path, target: str, mutator: Any, error_code: str) -> None:
    report = _valid_report()
    paths = _write_package(
        tmp_path,
        report=report,
        summary_mutator=(mutator if target == "summary" else None),
    )
    if target == "report":
        mutator(report)
        _write_json(paths["report"], report)

    summary = _precheck(paths)

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


@pytest.mark.parametrize(("target", "text", "error_code"), [
    ("report", "{", "controlled_writer_dry_run_report_invalid_json"),
    ("summary", "{", "controlled_writer_dry_run_summary_invalid_json"),
])
def test_invalid_json_blocks_safely(tmp_path: Path, target: str, text: str, error_code: str) -> None:
    paths = _write_package(tmp_path)
    paths[target].write_text(text, encoding="utf-8")

    summary = _precheck(paths)

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]
    assert "Expecting property name" not in json.dumps(summary)


@pytest.mark.parametrize(("target", "error_code"), [
    ("report", "controlled_writer_dry_run_report_missing"),
    ("summary", "controlled_writer_dry_run_summary_missing"),
])
def test_missing_paths_block_safely(tmp_path: Path, target: str, error_code: str) -> None:
    paths = _write_package(tmp_path)
    paths[target].unlink()

    summary = _precheck(paths)

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("dry_run_report_sha256", "sha256:" + "0" * 64, "dry_run_report_sha256_mismatch"),
        ("dry_run_report_basename", "other.json", "dry_run_report_basename_mismatch"),
    ],
)
def test_report_hash_and_basename_mismatches_block(
    tmp_path: Path,
    field: str,
    value: str,
    error_code: str,
) -> None:
    paths = _write_package(tmp_path, summary_mutator=lambda payload: payload.__setitem__(field, value))

    summary = _precheck(paths)

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("dry_run_id", "other-dry-run", "dry_run_id_mismatch"),
        ("dry_run_status", "needs_review", "dry_run_status_mismatch"),
        ("corpus_id", "other-corpus", "corpus_id_mismatch"),
        ("dataset_name", "other-dataset", "dataset_name_mismatch"),
        ("accepted_candidate_record_count", 4, "accepted_candidate_record_count_mismatch"),
        ("needs_review_candidate_record_count", 1, "needs_review_candidate_record_count_mismatch"),
        ("blocked_candidate_record_count", 1, "blocked_candidate_record_count_mismatch"),
        ("required_field_count", 6, "required_field_count_mismatch"),
        ("resolved_required_field_count", 4, "resolved_required_field_count_mismatch"),
        ("missing_required_field_count", 1, "missing_required_field_count_mismatch"),
        ("would_write_row_count", 4, "would_write_row_count_mismatch"),
        ("would_write_field_count", 16, "would_write_field_count_mismatch"),
        ("controlled_writer_executed", True, "controlled_writer_executed_mismatch"),
        ("training_dataset_materialized", True, "training_dataset_materialized_mismatch"),
        ("dataset_artifact_created", True, "dataset_artifact_created_mismatch"),
        ("serialized_rows_created", True, "serialized_rows_created_mismatch"),
        ("phase1_status", "ran", "phase1_status_mismatch"),
        ("dataset_confirmation_changed", True, "dataset_confirmation_changed_mismatch"),
        ("model_training_run", True, "model_training_run_mismatch"),
        ("evaluation_run", True, "evaluation_run_mismatch"),
    ],
)
def test_report_summary_mismatches_block(
    tmp_path: Path,
    field: str,
    value: Any,
    error_code: str,
) -> None:
    paths = _write_package(tmp_path, summary_mutator=lambda payload: payload.__setitem__(field, value))

    summary = _precheck(paths)

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


@pytest.mark.parametrize("status", ["blocked", "failed"])
def test_dry_run_status_blocked_or_failed_blocks(tmp_path: Path, status: str) -> None:
    report = _valid_report()
    report["dry_run_status"] = status
    paths = _write_package(tmp_path, report=report, summary_mutator=lambda payload: payload.__setitem__("dry_run_status", status))

    summary = _precheck(paths)

    assert summary["precheck_status"] == "blocked"
    assert "dry_run_blocked" in summary["precheck_errors"]


def test_dry_run_needs_review_blocks_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    report = _valid_report()
    report["dry_run_status"] = "needs_review"
    paths = _write_package(
        tmp_path,
        report=report,
        summary_mutator=lambda payload: payload.__setitem__("dry_run_status", "needs_review"),
    )

    blocked = _precheck(paths)
    allowed = _precheck(paths, allow_dry_run_needs_review=True)

    assert blocked["precheck_status"] == "blocked"
    assert "dry_run_needs_review" in blocked["precheck_errors"]
    assert allowed["precheck_status"] == "needs_review"
    assert "dry_run_needs_review" in allowed["precheck_warnings"]


def test_needs_review_count_blocks_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    report = _valid_report()
    report["needs_review_candidate_record_count"] = 1
    paths = _write_package(
        tmp_path,
        report=report,
        summary_mutator=lambda payload: payload.__setitem__("needs_review_candidate_record_count", 1),
    )

    blocked = _precheck(paths)
    allowed = _precheck(paths, allow_dry_run_needs_review=True)

    assert blocked["precheck_status"] == "blocked"
    assert "needs_review_candidate_records_present" in blocked["precheck_errors"]
    assert allowed["precheck_status"] == "needs_review"
    assert "needs_review_candidate_records_present" in allowed["precheck_warnings"]


def test_blocked_count_blocks(tmp_path: Path) -> None:
    report = _valid_report()
    report["blocked_candidate_record_count"] = 1
    paths = _write_package(
        tmp_path,
        report=report,
        summary_mutator=lambda payload: payload.__setitem__("blocked_candidate_record_count", 1),
    )

    summary = _precheck(paths)

    assert summary["precheck_status"] == "blocked"
    assert "blocked_candidate_records_present" in summary["precheck_errors"]


def test_missing_required_fields_block_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    report = _valid_report()
    report["resolved_required_field_count"] = 4
    report["missing_required_field_count"] = 1
    paths = _write_package(
        tmp_path,
        report=report,
        summary_mutator=lambda payload: (
            payload.__setitem__("resolved_required_field_count", 4),
            payload.__setitem__("missing_required_field_count", 1),
        ),
    )

    blocked = _precheck(paths)
    allowed = _precheck(paths, require_zero_missing_required_fields=False)

    assert blocked["precheck_status"] == "blocked"
    assert "missing_required_fields" in blocked["precheck_errors"]
    assert allowed["precheck_status"] == "needs_review"
    assert "missing_required_fields" in allowed["precheck_warnings"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("would_write_row_count", 0, "minimum_would_write_row_count_not_met"),
        ("would_write_field_count", 0, "would_write_field_count_invalid"),
        ("would_create_training_dataset_artifact", True, "would_create_training_dataset_artifact"),
        ("would_create_csv_jsonl_parquet_lmdb", True, "would_create_csv_jsonl_parquet_lmdb"),
        ("would_serialize_rows", True, "would_serialize_rows"),
        ("would_materialize_values", True, "would_materialize_values"),
        ("controlled_writer_executed", True, "controlled_writer_executed"),
        ("training_dataset_materialized", True, "training_dataset_materialized"),
        ("dataset_artifact_created", True, "dataset_artifact_created"),
        ("serialized_rows_created", True, "serialized_rows_created"),
        ("phase1_status", "ran", "phase1_ran"),
        ("dataset_confirmation_changed", True, "dataset_confirmation_changed"),
        ("model_training_run", True, "model_training_run"),
        ("evaluation_run", True, "evaluation_run"),
        ("redaction_status", "failed", "redaction_status_failed"),
    ],
)
def test_status_count_and_boundary_violations_block(
    tmp_path: Path,
    field: str,
    value: Any,
    error_code: str,
) -> None:
    report = _valid_report()
    report[field] = value
    paths = _write_package(tmp_path, report=report, summary_mutator=lambda payload: payload.__setitem__(field, value))

    summary = _precheck(paths)

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


def test_optional_evidence_markdown_unsafe_blocks(tmp_path: Path) -> None:
    paths = _write_package(tmp_path, evidence="unsafe .pdf")

    summary = _precheck(paths)

    assert summary["precheck_status"] == "blocked"
    assert "controlled_writer_dry_run_evidence_contains_unsafe_material" in summary["precheck_errors"]


@pytest.mark.parametrize("marker", FORBIDDEN_MARKERS)
def test_forbidden_markers_block_without_echoing_sensitive_value(tmp_path: Path, marker: str) -> None:
    report = _valid_report()
    report["unsafe_note"] = f"unsafe {marker}"
    paths = _write_package(tmp_path, report=report)

    summary = _precheck(paths)
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["precheck_status"] == "blocked"
    assert "controlled_writer_dry_run_package_contains_unsafe_material" in summary["precheck_errors"]
    assert marker not in serialized


@pytest.mark.parametrize("target", ["report", "summary", "evidence"])
def test_absolute_path_blocks(tmp_path: Path, target: str) -> None:
    paths = _write_package(tmp_path)
    if target == "evidence":
        paths["evidence"].write_text("unsafe /tmp/absolute/path", encoding="utf-8")
    else:
        payload = json.loads(paths[target].read_text(encoding="utf-8"))
        payload["unsafe_path"] = "/tmp/absolute/path"
        _write_json(paths[target], payload)

    summary = _precheck(paths)

    assert summary["precheck_status"] == "blocked"


def test_redaction_failure_returns_minimal_summary_and_no_unsafe_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_package(tmp_path)
    output_markdown = tmp_path / "unsafe-precheck.md"
    monkeypatch.setattr(precheck_module, "_markdown", lambda summary: "unsafe .pdf")

    summary = precheck_property_training_dataset_controlled_writer_dry_run(
        controlled_writer_dry_run_report_path=paths["report"],
        controlled_writer_dry_run_summary_path=paths["summary"],
        controlled_writer_dry_run_evidence_path=paths["evidence"],
        output_markdown_path=output_markdown,
    )

    assert summary == {
        "schema_version": PREFLIGHT_SCHEMA,
        "precheck_status": "blocked",
        "precheck_errors": ["property_training_dataset_controlled_writer_dry_run_precheck_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not output_markdown.exists()


def test_cli_returns_zero_for_passed_and_stdout_is_valid_json(tmp_path: Path) -> None:
    paths = _write_package(tmp_path)
    stdout = io.StringIO()

    code = main(
        [
            "--controlled-writer-dry-run-report",
            str(paths["report"]),
            "--controlled-writer-dry-run-summary",
            str(paths["summary"]),
            "--controlled-writer-dry-run-evidence",
            str(paths["evidence"]),
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["precheck_status"] == "passed"


def test_cli_returns_one_for_blocked(tmp_path: Path) -> None:
    paths = _write_package(tmp_path, summary_mutator=lambda payload: payload.__setitem__("schema_version", "wrong"))
    stdout = io.StringIO()

    code = main(
        [
            "--controlled-writer-dry-run-report",
            str(paths["report"]),
            "--controlled-writer-dry-run-summary",
            str(paths["summary"]),
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    payload = json.loads(stdout.getvalue())
    assert code == 1
    assert payload["precheck_status"] == "blocked"


def test_no_disallowed_artifacts_or_imports_are_created(tmp_path: Path) -> None:
    paths = _write_package(tmp_path)
    output_summary = tmp_path / "precheck-summary.json"
    output_markdown = tmp_path / "precheck-evidence.md"

    precheck_property_training_dataset_controlled_writer_dry_run(
        controlled_writer_dry_run_report_path=paths["report"],
        controlled_writer_dry_run_summary_path=paths["summary"],
        controlled_writer_dry_run_evidence_path=paths["evidence"],
        output_summary_path=output_summary,
        output_markdown_path=output_markdown,
    )

    _assert_no_artifact_files(tmp_path)
    source = MODULE_PATH.read_text(encoding="utf-8")
    import_lines = "\n".join(
        line for line in source.splitlines() if line.startswith("import ") or line.startswith("from ")
    ).lower()
    for forbidden in (
        "custom_corpus_property_training_dataset_controlled_writer_dry_run",
        "controlled_writer_execution",
        "dataset_writer",
        "mineru",
        "openai",
        "pypdf",
        "pdfplumber",
        "parseddocument",
        "corpus_workflow",
        "workflow",
        "rdkit",
        "chem",
        "training",
        "evaluation",
    ):
        assert forbidden not in import_lines


def test_docs_and_template_exist_and_are_linked() -> None:
    assert DESIGN_DOC.exists()
    assert TEMPLATE_DOC.exists()
    design_text = DESIGN_DOC.read_text(encoding="utf-8")
    template_text = TEMPLATE_DOC.read_text(encoding="utf-8")
    for section in DOC_SECTIONS:
        assert section in design_text
    for placeholder in TEMPLATE_PLACEHOLDERS:
        assert placeholder in template_text
    for statement in BOUNDARY_STATEMENTS:
        assert statement in design_text
        assert statement in template_text
    for doc in GOVERNANCE_DOCS:
        assert CHAIN_PHRASE in doc.read_text(encoding="utf-8")


def test_dry_run_precheck_docs_do_not_include_forbidden_markers() -> None:
    combined = "\n".join(
        [
            DESIGN_DOC.read_text(encoding="utf-8"),
            TEMPLATE_DOC.read_text(encoding="utf-8"),
        ]
    )
    sanitized = combined.replace(
        "This controlled writer dry-run precheck does not create CSV/JSONL/Parquet/LMDB artifacts.",
        "This controlled writer dry-run precheck does not create FORMAT-LABEL artifacts.",
    )
    sanitized = sanitized.replace("CSV/JSONL/Parquet/LMDB", "FORMAT-LABEL")
    sanitized = sanitized.replace("would_create_csv_jsonl_parquet_lmdb", "would_create_format_label")
    for marker in FORBIDDEN_MARKERS:
        assert marker not in sanitized
