from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from ai4s_agent import custom_corpus_property_training_dataset_controlled_writer_execution_request_preflight as preflight_module
from ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_request_preflight import (
    main,
    preflight_property_training_dataset_controlled_writer_execution_request,
)


REQUEST_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_execution_request.v1"
SUMMARY_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_execution_request_summary.v1"
PREFLIGHT_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_execution_request_preflight.v1"

REQUEST_BASENAME = "property_training_dataset_controlled_writer_execution_request.json"
SUMMARY_BASENAME = "property_training_dataset_controlled_writer_execution_request_summary.json"
EVIDENCE_BASENAME = "redacted_property_training_dataset_controlled_writer_execution_request_evidence.md"

DESIGN_DOC = Path("docs/custom-corpus-property-training-dataset-controlled-writer-execution-request-preflight.md")
TEMPLATE_DOC = Path(
    "docs/evidence/templates/"
    "custom-corpus-property-training-dataset-controlled-writer-execution-request-preflight-evidence-template.md"
)
MODULE_PATH = Path(
    "src/ai4s_agent/custom_corpus_property_training_dataset_controlled_writer_execution_request_preflight.py"
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
    "-> property training dataset controlled writer execution request preflight\n"
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
    Path("docs/custom-corpus-property-training-dataset-controlled-writer-execution-request-design.md"),
    Path("docs/custom-corpus-property-training-dataset-controlled-writer-execution-request.md"),
    Path("docs/phase-1-4-milestone-status.md"),
)

DOC_SECTIONS = (
    "# Custom Corpus Property Training Dataset Controlled Writer Execution Request Preflight",
    "## Purpose",
    "## Position in the Governance Chain",
    "## Input Package",
    "## Preflight Checks",
    "## Status Semantics",
    "## Hash and Basename Policy",
    "## Redaction Policy",
    "## Authorization Boundary",
    "## Explicit Confirmation Boundary",
    "## CLI Usage",
    "## Outputs",
    "## Blocked Conditions",
    "## Out of Scope",
    "## Next Step",
)

TEMPLATE_PLACEHOLDERS = (
    "<controlled_writer_execution_request_preflight_evidence_id>",
    "<date>",
    "<operator>",
    "<corpus_id>",
    "<dataset_name>",
    "<request_id>",
    "<request_status>",
    "<preflight_status>",
    "<request_basename>",
    "<request_sha256>",
    "<request_summary_basename>",
    "<requested_by>",
    "<dry_run_precheck_summary_basename>",
    "<dry_run_precheck_summary_sha256>",
    "<dry_run_precheck_status>",
    "<dry_run_status>",
    "<accepted_candidate_record_count>",
    "<needs_review_candidate_record_count>",
    "<blocked_candidate_record_count>",
    "<missing_required_field_count>",
    "<would_write_row_count>",
    "<would_write_field_count>",
    "<redaction_status>",
    "<writer_execution_authorized>",
    "<explicit_confirmation_required>",
    "<next_gate_decision>",
    "<residual_risks>",
)

BOUNDARY_STATEMENTS = (
    "This controlled writer execution request preflight does not explicitly confirm execution.",
    "This controlled writer execution request preflight does not execute the controlled writer.",
    "This controlled writer execution request preflight does not authorize writer execution by itself.",
    "This controlled writer execution request preflight keeps explicit confirmation required.",
    "This controlled writer execution request preflight does not emit raw values.",
    "This controlled writer execution request preflight does not materialize values.",
    "This controlled writer execution request preflight does not serialize training rows.",
    "This controlled writer execution request preflight does not create training dataset artifacts.",
    "This controlled writer execution request preflight does not create CSV/JSONL/Parquet/LMDB artifacts.",
    "This controlled writer execution request preflight does not generate conformers.",
    "This controlled writer execution request preflight does not generate DPA3 structures.",
    "This controlled writer execution request preflight does not run Phase 1.",
    "This controlled writer execution request preflight does not modify DatasetConfirmation.",
    "This controlled writer execution request preflight does not run model training or evaluation.",
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
    "SMILES=",
    "C1=CC",
    "0.72",
    "Authorization",
    "Bearer",
    "token=",
    "secret=",
    "password=",
    "cookie=",
    "raw article text",
    "raw table",
    "serialized training row",
    "serialized dataset row",
    "conformer block",
    "dpa3 structure block",
)


def _valid_request() -> dict[str, Any]:
    return {
        "schema_version": REQUEST_SCHEMA,
        "request_id": "controlled-writer-execution-request-001",
        "request_status": "request_ready_for_preflight",
        "requested_by": "safe-operator-id",
        "request_purpose": "controlled_writer_execution_request_for_preflight",
        "corpus_id": "safe-corpus-id",
        "dataset_name": "safe-dataset-name",
        "dry_run_precheck_summary_basename": "property_training_dataset_controlled_writer_dry_run_precheck_summary.json",
        "dry_run_precheck_summary_sha256": "sha256:" + "b" * 64,
        "dry_run_precheck_status": "passed",
        "dry_run_status": "passed",
        "dry_run_report_basename": "property_training_dataset_controlled_writer_dry_run_report.json",
        "dry_run_report_sha256": "sha256:" + "a" * 64,
        "dry_run_summary_basename": "property_training_dataset_controlled_writer_dry_run_summary.json",
        "accepted_candidate_record_count": 3,
        "needs_review_candidate_record_count": 0,
        "blocked_candidate_record_count": 0,
        "required_field_count": 5,
        "resolved_required_field_count": 5,
        "missing_required_field_count": 0,
        "would_write_row_count": 3,
        "would_write_field_count": 15,
        "controlled_writer_executed": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "serialized_rows_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "model_training_run": False,
        "evaluation_run": False,
        "redaction_status": "passed",
        "requested_next_gate": "controlled_writer_execution_request_preflight",
        "explicit_confirmation_required": True,
        "writer_execution_authorized": False,
        "request_errors": [],
        "request_warnings": [],
    }


def _valid_summary(request_sha: str) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA,
        "request_id": "controlled-writer-execution-request-001",
        "request_status": "request_ready_for_preflight",
        "request_basename": REQUEST_BASENAME,
        "request_sha256": request_sha,
        "requested_by": "safe-operator-id",
        "corpus_id": "safe-corpus-id",
        "dataset_name": "safe-dataset-name",
        "dry_run_precheck_summary_basename": "property_training_dataset_controlled_writer_dry_run_precheck_summary.json",
        "dry_run_precheck_summary_sha256": "sha256:" + "b" * 64,
        "dry_run_precheck_status": "passed",
        "dry_run_status": "passed",
        "accepted_candidate_record_count": 3,
        "needs_review_candidate_record_count": 0,
        "blocked_candidate_record_count": 0,
        "would_write_row_count": 3,
        "would_write_field_count": 15,
        "missing_required_field_count": 0,
        "redaction_status": "passed",
        "explicit_confirmation_required": True,
        "writer_execution_authorized": False,
        "controlled_writer_executed": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "serialized_rows_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "model_training_run": False,
        "evaluation_run": False,
        "request_errors": [],
        "request_warnings": [],
    }


def _safe_evidence() -> str:
    return "\n".join(
        [
            "# Controlled Writer Execution Request Evidence",
            "",
            "This controlled writer execution request does not execute the controlled writer.",
            "Explicit confirmation remains required.",
            "Writer execution is not authorized.",
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
    request: dict[str, Any] | None = None,
    summary_mutator: Any | None = None,
    evidence: str | None = None,
) -> dict[str, Path]:
    package_dir = tmp_path / "package"
    package_dir.mkdir(parents=True, exist_ok=True)
    request_path = package_dir / REQUEST_BASENAME
    summary_path = package_dir / SUMMARY_BASENAME
    evidence_path = package_dir / EVIDENCE_BASENAME
    request_payload = request or _valid_request()
    _write_json(request_path, request_payload)
    summary_payload = _valid_summary(_sha256(request_path))
    if summary_mutator is not None:
        summary_mutator(summary_payload)
    _write_json(summary_path, summary_payload)
    evidence_path.write_text(evidence if evidence is not None else _safe_evidence(), encoding="utf-8")
    return {"request": request_path, "summary": summary_path, "evidence": evidence_path}


def _preflight(paths: dict[str, Path], **kwargs: Any) -> dict[str, Any]:
    return preflight_property_training_dataset_controlled_writer_execution_request(
        controlled_writer_execution_request_path=paths["request"],
        controlled_writer_execution_request_summary_path=paths["summary"],
        controlled_writer_execution_request_evidence_path=paths.get("evidence"),
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


def test_valid_request_and_summary_return_preflight_passed_and_write_outputs(tmp_path: Path) -> None:
    paths = _write_package(tmp_path)
    output_summary = tmp_path / "request-preflight-summary.json"
    output_markdown = tmp_path / "request-preflight-evidence.md"

    summary = preflight_property_training_dataset_controlled_writer_execution_request(
        controlled_writer_execution_request_path=paths["request"],
        controlled_writer_execution_request_summary_path=paths["summary"],
        controlled_writer_execution_request_evidence_path=paths["evidence"],
        output_summary_path=output_summary,
        output_markdown_path=output_markdown,
    )

    assert summary["schema_version"] == PREFLIGHT_SCHEMA
    assert summary["preflight_status"] == "preflight_passed"
    assert summary["request_basename"] == REQUEST_BASENAME
    assert summary["request_summary_basename"] == SUMMARY_BASENAME
    assert summary["request_sha256"] == _sha256(paths["request"])
    assert summary["next_gate"] == "future_explicit_confirmation"
    assert summary["writer_execution_authorized"] is False
    assert summary["explicit_confirmation_required"] is True
    assert json.loads(output_summary.read_text(encoding="utf-8")) == summary
    markdown = output_markdown.read_text(encoding="utf-8")
    for statement in BOUNDARY_STATEMENTS:
        assert statement in markdown
    assert str(tmp_path) not in json.dumps(summary, sort_keys=True)
    _assert_no_artifact_files(tmp_path)


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("request", lambda payload: payload.__setitem__("schema_version", "wrong"), "controlled_writer_execution_request_schema_invalid"),
        ("summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "controlled_writer_execution_request_summary_schema_invalid"),
    ],
)
def test_wrong_schema_blocks(tmp_path: Path, target: str, mutator: Any, error_code: str) -> None:
    request = _valid_request()
    paths = _write_package(tmp_path, request=request, summary_mutator=(mutator if target == "summary" else None))
    if target == "request":
        mutator(request)
        _write_json(paths["request"], request)

    summary = _preflight(paths)

    assert summary["preflight_status"] == "preflight_blocked"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(("target", "text", "error_code"), [
    ("request", "{", "controlled_writer_execution_request_invalid_json"),
    ("summary", "{", "controlled_writer_execution_request_summary_invalid_json"),
])
def test_invalid_json_blocks_safely(tmp_path: Path, target: str, text: str, error_code: str) -> None:
    paths = _write_package(tmp_path)
    paths[target].write_text(text, encoding="utf-8")

    summary = _preflight(paths)

    assert summary["preflight_status"] == "preflight_blocked"
    assert error_code in summary["preflight_errors"]
    assert "Expecting property name" not in json.dumps(summary)


@pytest.mark.parametrize(("target", "error_code"), [
    ("request", "controlled_writer_execution_request_missing"),
    ("summary", "controlled_writer_execution_request_summary_missing"),
])
def test_missing_paths_block_safely(tmp_path: Path, target: str, error_code: str) -> None:
    paths = _write_package(tmp_path)
    paths[target].unlink()

    summary = _preflight(paths)

    assert summary["preflight_status"] == "preflight_blocked"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("request_sha256", "sha256:" + "0" * 64, "request_sha256_mismatch"),
        ("request_basename", "other.json", "request_basename_mismatch"),
    ],
)
def test_request_hash_and_basename_mismatches_block(tmp_path: Path, field: str, value: str, error_code: str) -> None:
    paths = _write_package(tmp_path, summary_mutator=lambda payload: payload.__setitem__(field, value))

    summary = _preflight(paths)

    assert summary["preflight_status"] == "preflight_blocked"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("request_id", "other-request", "request_id_mismatch"),
        ("request_status", "request_needs_review", "request_status_mismatch"),
        ("requested_by", "other-operator", "requested_by_mismatch"),
        ("corpus_id", "other-corpus", "corpus_id_mismatch"),
        ("dataset_name", "other-dataset", "dataset_name_mismatch"),
        ("dry_run_precheck_summary_basename", "other.json", "dry_run_precheck_summary_basename_mismatch"),
        ("dry_run_precheck_summary_sha256", "sha256:" + "c" * 64, "dry_run_precheck_summary_sha256_mismatch"),
        ("dry_run_precheck_status", "needs_review", "dry_run_precheck_status_mismatch"),
        ("dry_run_status", "needs_review", "dry_run_status_mismatch"),
        ("accepted_candidate_record_count", 4, "accepted_candidate_record_count_mismatch"),
        ("needs_review_candidate_record_count", 1, "needs_review_candidate_record_count_mismatch"),
        ("blocked_candidate_record_count", 1, "blocked_candidate_record_count_mismatch"),
        ("would_write_row_count", 4, "would_write_row_count_mismatch"),
        ("would_write_field_count", 16, "would_write_field_count_mismatch"),
        ("missing_required_field_count", 1, "missing_required_field_count_mismatch"),
        ("writer_execution_authorized", True, "writer_execution_authorized_mismatch"),
        ("explicit_confirmation_required", False, "explicit_confirmation_required_mismatch"),
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
def test_request_summary_mismatches_block(tmp_path: Path, field: str, value: Any, error_code: str) -> None:
    paths = _write_package(tmp_path, summary_mutator=lambda payload: payload.__setitem__(field, value))

    summary = _preflight(paths)

    assert summary["preflight_status"] == "preflight_blocked"
    assert error_code in summary["preflight_errors"]


def test_request_status_needs_review_blocks_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    request = _valid_request()
    request["request_status"] = "request_needs_review"
    paths = _write_package(
        tmp_path,
        request=request,
        summary_mutator=lambda payload: payload.__setitem__("request_status", "request_needs_review"),
    )

    blocked = _preflight(paths)
    allowed = _preflight(paths, allow_request_needs_review=True)

    assert blocked["preflight_status"] == "preflight_blocked"
    assert "request_needs_review" in blocked["preflight_errors"]
    assert allowed["preflight_status"] == "preflight_needs_review"
    assert "request_needs_review" in allowed["preflight_warnings"]


@pytest.mark.parametrize("status", ["request_blocked", "failed"])
def test_request_status_blocked_or_failed_blocks(tmp_path: Path, status: str) -> None:
    request = _valid_request()
    request["request_status"] = status
    paths = _write_package(tmp_path, request=request, summary_mutator=lambda payload: payload.__setitem__("request_status", status))

    summary = _preflight(paths)

    assert summary["preflight_status"] == "preflight_blocked"
    assert "request_blocked" in summary["preflight_errors"]


def test_needs_review_count_blocks_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    request = _valid_request()
    request["needs_review_candidate_record_count"] = 1
    paths = _write_package(tmp_path, request=request, summary_mutator=lambda payload: payload.__setitem__("needs_review_candidate_record_count", 1))

    blocked = _preflight(paths)
    allowed = _preflight(paths, allow_request_needs_review=True)

    assert blocked["preflight_status"] == "preflight_blocked"
    assert "needs_review_candidate_records_present" in blocked["preflight_errors"]
    assert allowed["preflight_status"] == "preflight_needs_review"
    assert "needs_review_candidate_records_present" in allowed["preflight_warnings"]


def test_missing_required_fields_block_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    request = _valid_request()
    request["missing_required_field_count"] = 1
    paths = _write_package(tmp_path, request=request, summary_mutator=lambda payload: payload.__setitem__("missing_required_field_count", 1))

    blocked = _preflight(paths)
    allowed = _preflight(paths, require_zero_missing_required_fields=False)

    assert blocked["preflight_status"] == "preflight_blocked"
    assert "missing_required_fields" in blocked["preflight_errors"]
    assert allowed["preflight_status"] == "preflight_needs_review"
    assert "missing_required_fields" in allowed["preflight_warnings"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("accepted_candidate_record_count", 0, "minimum_accepted_candidate_records_not_met"),
        ("blocked_candidate_record_count", 1, "blocked_candidate_records_present"),
        ("would_write_row_count", 0, "minimum_would_write_row_count_not_met"),
        ("would_write_field_count", 0, "would_write_field_count_invalid"),
        ("writer_execution_authorized", True, "writer_execution_authorized"),
        ("explicit_confirmation_required", False, "explicit_confirmation_required"),
        ("controlled_writer_executed", True, "controlled_writer_executed"),
        ("training_dataset_materialized", True, "training_dataset_materialized"),
        ("dataset_artifact_created", True, "dataset_artifact_created"),
        ("serialized_rows_created", True, "serialized_rows_created"),
        ("phase1_status", "ran", "phase1_ran"),
        ("dataset_confirmation_changed", True, "dataset_confirmation_changed"),
        ("model_training_run", True, "model_training_run"),
        ("evaluation_run", True, "evaluation_run"),
        ("redaction_status", "failed", "redaction_status_failed"),
        ("requested_next_gate", "other-gate", "requested_next_gate_invalid"),
    ],
)
def test_status_count_authorization_and_boundary_violations_block(
    tmp_path: Path,
    field: str,
    value: Any,
    error_code: str,
) -> None:
    request = _valid_request()
    request[field] = value
    paths = _write_package(tmp_path, request=request, summary_mutator=lambda payload: payload.__setitem__(field, value))

    summary = _preflight(paths)

    assert summary["preflight_status"] == "preflight_blocked"
    assert error_code in summary["preflight_errors"]


def test_optional_evidence_markdown_unsafe_blocks(tmp_path: Path) -> None:
    paths = _write_package(tmp_path, evidence="unsafe .pdf")

    summary = _preflight(paths)

    assert summary["preflight_status"] == "preflight_blocked"
    assert "controlled_writer_execution_request_evidence_contains_unsafe_material" in summary["preflight_errors"]


@pytest.mark.parametrize("marker", FORBIDDEN_MARKERS)
def test_forbidden_markers_block_without_echoing_sensitive_value(tmp_path: Path, marker: str) -> None:
    request = _valid_request()
    request["unsafe_note"] = f"unsafe {marker}"
    paths = _write_package(tmp_path, request=request)

    summary = _preflight(paths)
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["preflight_status"] == "preflight_blocked"
    assert "controlled_writer_execution_request_package_contains_unsafe_material" in summary["preflight_errors"]
    assert marker not in serialized


@pytest.mark.parametrize("target", ["request", "summary", "evidence"])
def test_absolute_path_blocks(tmp_path: Path, target: str) -> None:
    paths = _write_package(tmp_path)
    if target == "evidence":
        paths["evidence"].write_text("unsafe /tmp/absolute/path", encoding="utf-8")
    else:
        payload = json.loads(paths[target].read_text(encoding="utf-8"))
        payload["unsafe_path"] = "/tmp/absolute/path"
        _write_json(paths[target], payload)

    summary = _preflight(paths)

    assert summary["preflight_status"] == "preflight_blocked"


def test_redaction_failure_returns_minimal_summary_and_no_unsafe_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_package(tmp_path)
    output_markdown = tmp_path / "unsafe-preflight.md"
    monkeypatch.setattr(preflight_module, "_markdown", lambda summary: "unsafe .pdf")

    summary = preflight_property_training_dataset_controlled_writer_execution_request(
        controlled_writer_execution_request_path=paths["request"],
        controlled_writer_execution_request_summary_path=paths["summary"],
        controlled_writer_execution_request_evidence_path=paths["evidence"],
        output_markdown_path=output_markdown,
    )

    assert summary == {
        "schema_version": PREFLIGHT_SCHEMA,
        "preflight_status": "preflight_blocked",
        "preflight_errors": [
            "property_training_dataset_controlled_writer_execution_request_preflight_redaction_failed"
        ],
        "redaction_status": "failed",
        "writer_execution_authorized": False,
        "explicit_confirmation_required": True,
    }
    assert not output_markdown.exists()


def test_cli_return_codes_and_stdout_are_valid_json(tmp_path: Path) -> None:
    passed_paths = _write_package(tmp_path / "passed")
    needs_review_request = _valid_request()
    needs_review_request["request_status"] = "request_needs_review"
    needs_review_paths = _write_package(
        tmp_path / "needs-review",
        request=needs_review_request,
        summary_mutator=lambda payload: payload.__setitem__("request_status", "request_needs_review"),
    )
    blocked_paths = _write_package(tmp_path / "blocked", summary_mutator=lambda payload: payload.__setitem__("schema_version", "wrong"))

    passed_stdout = io.StringIO()
    needs_review_stdout = io.StringIO()
    blocked_stdout = io.StringIO()
    passed_code = main(
        [
            "--controlled-writer-execution-request",
            str(passed_paths["request"]),
            "--controlled-writer-execution-request-summary",
            str(passed_paths["summary"]),
            "--controlled-writer-execution-request-evidence",
            str(passed_paths["evidence"]),
        ],
        stdout=passed_stdout,
        stderr=io.StringIO(),
    )
    needs_review_code = main(
        [
            "--controlled-writer-execution-request",
            str(needs_review_paths["request"]),
            "--controlled-writer-execution-request-summary",
            str(needs_review_paths["summary"]),
            "--allow-request-needs-review",
        ],
        stdout=needs_review_stdout,
        stderr=io.StringIO(),
    )
    blocked_code = main(
        [
            "--controlled-writer-execution-request",
            str(blocked_paths["request"]),
            "--controlled-writer-execution-request-summary",
            str(blocked_paths["summary"]),
        ],
        stdout=blocked_stdout,
        stderr=io.StringIO(),
    )

    assert passed_code == 0
    assert json.loads(passed_stdout.getvalue())["preflight_status"] == "preflight_passed"
    assert needs_review_code == 0
    assert json.loads(needs_review_stdout.getvalue())["preflight_status"] == "preflight_needs_review"
    assert blocked_code == 1
    assert json.loads(blocked_stdout.getvalue())["preflight_status"] == "preflight_blocked"


def test_no_disallowed_artifacts_or_imports_are_created(tmp_path: Path) -> None:
    paths = _write_package(tmp_path)
    output_summary = tmp_path / "preflight-summary.json"
    output_markdown = tmp_path / "preflight-evidence.md"

    preflight_property_training_dataset_controlled_writer_execution_request(
        controlled_writer_execution_request_path=paths["request"],
        controlled_writer_execution_request_summary_path=paths["summary"],
        controlled_writer_execution_request_evidence_path=paths["evidence"],
        output_summary_path=output_summary,
        output_markdown_path=output_markdown,
    )

    _assert_no_artifact_files(tmp_path)
    source = MODULE_PATH.read_text(encoding="utf-8")
    import_lines = "\n".join(
        line for line in source.splitlines() if line.startswith("import ") or line.startswith("from ")
    ).lower()
    for forbidden in (
        "controlled_writer_execution",
        "explicit_confirmation",
        "controlled_writer_dry_run",
        "custom_corpus_property_training_dataset_controlled_writer_execution_request",
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


def test_execution_request_preflight_docs_do_not_include_forbidden_markers() -> None:
    combined = "\n".join(
        [
            DESIGN_DOC.read_text(encoding="utf-8"),
            TEMPLATE_DOC.read_text(encoding="utf-8"),
        ]
    )
    sanitized = combined.replace(
        "This controlled writer execution request preflight does not create CSV/JSONL/Parquet/LMDB artifacts.",
        "This controlled writer execution request preflight does not create FORMAT-LABEL artifacts.",
    )
    sanitized = sanitized.replace("CSV/JSONL/Parquet/LMDB", "FORMAT-LABEL")
    sanitized = sanitized.replace("## Authorization Boundary", "## Auth Boundary")
    sanitized = sanitized.replace("## Authorization State", "## Auth State")
    sanitized = sanitized.replace("would_create_csv_jsonl_parquet_lmdb", "would_create_format_label")
    for marker in FORBIDDEN_MARKERS:
        assert marker not in sanitized
