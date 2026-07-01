from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from ai4s_agent import custom_corpus_property_training_dataset_controlled_writer_execution_request as request_module
from ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_request import (
    create_property_training_dataset_controlled_writer_execution_request,
    main,
)


PREFLIGHT_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_dry_run_precheck.v1"
REQUEST_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_execution_request.v1"
SUMMARY_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_execution_request_summary.v1"

INPUT_BASENAME = "property_training_dataset_controlled_writer_dry_run_precheck_summary.json"
REQUEST_BASENAME = "property_training_dataset_controlled_writer_execution_request.json"
SUMMARY_BASENAME = "property_training_dataset_controlled_writer_execution_request_summary.json"
EVIDENCE_BASENAME = "redacted_property_training_dataset_controlled_writer_execution_request_evidence.md"

DESIGN_DOC = Path("docs/custom-corpus-property-training-dataset-controlled-writer-execution-request.md")
TEMPLATE_DOC = Path(
    "docs/evidence/templates/"
    "custom-corpus-property-training-dataset-controlled-writer-execution-request-evidence-template.md"
)
MODULE_PATH = Path("src/ai4s_agent/custom_corpus_property_training_dataset_controlled_writer_execution_request.py")

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
    Path("docs/custom-corpus-property-training-dataset-controlled-writer-execution-request-design.md"),
    Path("docs/phase-1-4-milestone-status.md"),
)

DOC_SECTIONS = (
    "# Custom Corpus Property Training Dataset Controlled Writer Execution Request",
    "## Purpose",
    "## Position in the Governance Chain",
    "## Input Summary",
    "## Request Creation Checks",
    "## Request Status Semantics",
    "## Output Files",
    "## Hash and Basename Policy",
    "## Redaction Policy",
    "## Authorization Boundary",
    "## Explicit Confirmation Boundary",
    "## CLI Usage",
    "## Blocked Conditions",
    "## Out of Scope",
    "## Next Step",
)

TEMPLATE_PLACEHOLDERS = (
    "<controlled_writer_execution_request_evidence_id>",
    "<date>",
    "<operator>",
    "<corpus_id>",
    "<dataset_name>",
    "<request_id>",
    "<request_status>",
    "<requested_by>",
    "<request_purpose>",
    "<request_basename>",
    "<request_sha256>",
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
    "This controlled writer execution request does not implement execution request preflight.",
    "This controlled writer execution request does not explicitly confirm execution.",
    "This controlled writer execution request does not execute the controlled writer.",
    "This controlled writer execution request does not authorize writer execution by itself.",
    "This controlled writer execution request keeps explicit confirmation required.",
    "This controlled writer execution request does not emit raw values.",
    "This controlled writer execution request does not materialize values.",
    "This controlled writer execution request does not serialize training rows.",
    "This controlled writer execution request does not create training dataset artifacts.",
    "This controlled writer execution request does not create CSV/JSONL/Parquet/LMDB artifacts.",
    "This controlled writer execution request does not generate conformers.",
    "This controlled writer execution request does not generate DPA3 structures.",
    "This controlled writer execution request does not run Phase 1.",
    "This controlled writer execution request does not modify DatasetConfirmation.",
    "This controlled writer execution request does not run model training or evaluation.",
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


def _valid_precheck_summary() -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "precheck_status": "passed",
        "dry_run_id": "controlled-writer-dry-run-001",
        "dry_run_status": "passed",
        "dry_run_report_basename": "property_training_dataset_controlled_writer_dry_run_report.json",
        "dry_run_report_sha256": "sha256:" + "a" * 64,
        "dry_run_summary_basename": "property_training_dataset_controlled_writer_dry_run_summary.json",
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
        "precheck_errors": [],
        "precheck_warnings": [],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_input(tmp_path: Path, payload: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "input" / INPUT_BASENAME
    _write_json(path, payload or _valid_precheck_summary())
    return path


def _create(input_path: Path, tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    return create_property_training_dataset_controlled_writer_execution_request(
        controlled_writer_dry_run_precheck_summary_path=input_path,
        output_dir=tmp_path / "out",
        request_id=kwargs.pop("request_id", "controlled-writer-execution-request-001"),
        requested_by=kwargs.pop("requested_by", "safe-operator-id"),
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


def test_valid_precheck_summary_creates_request_summary_and_markdown(tmp_path: Path) -> None:
    input_path = _write_input(tmp_path)

    summary = _create(input_path, tmp_path)

    run_dir = tmp_path / "out" / "controlled-writer-execution-request-001"
    request_path = run_dir / REQUEST_BASENAME
    summary_path = run_dir / SUMMARY_BASENAME
    markdown_path = run_dir / EVIDENCE_BASENAME
    request = json.loads(request_path.read_text(encoding="utf-8"))
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert request["schema_version"] == REQUEST_SCHEMA
    assert summary["schema_version"] == SUMMARY_SCHEMA
    assert summary["request_status"] == "request_ready_for_preflight"
    assert summary["request_basename"] == REQUEST_BASENAME
    assert summary["request_sha256"] == _sha256(request_path)
    assert summary["dry_run_precheck_summary_basename"] == INPUT_BASENAME
    assert summary["dry_run_precheck_summary_sha256"] == _sha256(input_path)
    assert written_summary == summary
    assert request["writer_execution_authorized"] is False
    assert request["explicit_confirmation_required"] is True
    for statement in BOUNDARY_STATEMENTS:
        assert statement in markdown
    assert str(tmp_path) not in json.dumps(summary, sort_keys=True)
    _assert_no_artifact_files(tmp_path)


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    input_path = _write_input(tmp_path)
    run_dir = tmp_path / "out" / "controlled-writer-execution-request-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = _create(input_path, tmp_path)

    assert summary["request_status"] == "request_blocked"
    assert "controlled_writer_execution_request_output_dir_not_clean" in summary["request_errors"]


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda payload: payload.__setitem__("schema_version", "wrong"), "controlled_writer_dry_run_precheck_schema_invalid"),
        (lambda payload: payload.__setitem__("precheck_status", "blocked"), "dry_run_precheck_blocked"),
        (lambda payload: payload.__setitem__("dry_run_status", "blocked"), "dry_run_blocked"),
        (lambda payload: payload.__setitem__("accepted_candidate_record_count", 0), "minimum_accepted_candidate_records_not_met"),
        (lambda payload: payload.__setitem__("blocked_candidate_record_count", 1), "blocked_candidate_records_present"),
        (lambda payload: payload.__setitem__("would_write_row_count", 0), "minimum_would_write_row_count_not_met"),
        (lambda payload: payload.__setitem__("would_write_field_count", 0), "would_write_field_count_invalid"),
        (lambda payload: payload.__setitem__("controlled_writer_executed", True), "controlled_writer_executed"),
        (lambda payload: payload.__setitem__("training_dataset_materialized", True), "training_dataset_materialized"),
        (lambda payload: payload.__setitem__("dataset_artifact_created", True), "dataset_artifact_created"),
        (lambda payload: payload.__setitem__("serialized_rows_created", True), "serialized_rows_created"),
        (lambda payload: payload.__setitem__("phase1_status", "ran"), "phase1_ran"),
        (lambda payload: payload.__setitem__("dataset_confirmation_changed", True), "dataset_confirmation_changed"),
        (lambda payload: payload.__setitem__("model_training_run", True), "model_training_run"),
        (lambda payload: payload.__setitem__("evaluation_run", True), "evaluation_run"),
        (lambda payload: payload.__setitem__("redaction_status", "failed"), "redaction_status_failed"),
    ],
)
def test_invalid_precheck_summary_blocks(tmp_path: Path, mutator: Any, error_code: str) -> None:
    payload = _valid_precheck_summary()
    mutator(payload)
    input_path = _write_input(tmp_path, payload)

    summary = _create(input_path, tmp_path)

    assert summary["request_status"] == "request_blocked"
    assert error_code in summary["request_errors"]


@pytest.mark.parametrize(
    ("kwargs", "error_code"),
    [
        ({"request_id": "bad/id"}, "request_id_unsafe"),
        ({"requested_by": "bad/operator"}, "requested_by_unsafe"),
        ({"request_purpose": "bad/purpose"}, "request_purpose_unsafe"),
    ],
)
def test_unsafe_request_metadata_blocks(tmp_path: Path, kwargs: dict[str, Any], error_code: str) -> None:
    input_path = _write_input(tmp_path)

    summary = _create(input_path, tmp_path, **kwargs)

    assert summary["request_status"] == "request_blocked"
    assert error_code in summary["request_errors"]


def test_missing_input_and_invalid_json_block_safely(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")

    missing_summary = _create(missing, tmp_path / "missing-case")
    invalid_summary = _create(invalid, tmp_path / "invalid-case")

    assert missing_summary["request_status"] == "request_blocked"
    assert "controlled_writer_dry_run_precheck_summary_missing" in missing_summary["request_errors"]
    assert invalid_summary["request_status"] == "request_blocked"
    assert "controlled_writer_dry_run_precheck_summary_invalid_json" in invalid_summary["request_errors"]
    assert "Expecting property name" not in json.dumps(invalid_summary)


def test_precheck_needs_review_and_dry_run_needs_review_block_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    precheck_payload = _valid_precheck_summary()
    precheck_payload["precheck_status"] = "needs_review"
    precheck_input = _write_input(tmp_path / "precheck", precheck_payload)
    dry_run_payload = _valid_precheck_summary()
    dry_run_payload["dry_run_status"] = "needs_review"
    dry_run_input = _write_input(tmp_path / "dryrun", dry_run_payload)

    precheck_blocked = _create(precheck_input, tmp_path / "precheck-blocked")
    precheck_allowed = _create(precheck_input, tmp_path / "precheck-allowed", require_dry_run_precheck_passed=False)
    dry_run_blocked = _create(dry_run_input, tmp_path / "dryrun-blocked")
    dry_run_allowed = _create(dry_run_input, tmp_path / "dryrun-allowed", require_dry_run_passed=False)

    assert precheck_blocked["request_status"] == "request_blocked"
    assert "dry_run_precheck_needs_review" in precheck_blocked["request_errors"]
    assert precheck_allowed["request_status"] == "request_needs_review"
    assert "dry_run_precheck_needs_review" in precheck_allowed["request_warnings"]
    assert dry_run_blocked["request_status"] == "request_blocked"
    assert "dry_run_needs_review" in dry_run_blocked["request_errors"]
    assert dry_run_allowed["request_status"] == "request_needs_review"
    assert "dry_run_needs_review" in dry_run_allowed["request_warnings"]


def test_needs_review_candidates_and_missing_required_fields_need_allowance(tmp_path: Path) -> None:
    needs_review_payload = _valid_precheck_summary()
    needs_review_payload["needs_review_candidate_record_count"] = 1
    needs_review_input = _write_input(tmp_path / "needs-review", needs_review_payload)
    missing_payload = _valid_precheck_summary()
    missing_payload["missing_required_field_count"] = 1
    missing_payload["resolved_required_field_count"] = 4
    missing_input = _write_input(tmp_path / "missing-fields", missing_payload)

    needs_review_blocked = _create(needs_review_input, tmp_path / "needs-review-blocked")
    needs_review_allowed = _create(needs_review_input, tmp_path / "needs-review-allowed", allow_needs_review_candidates=True)
    missing_blocked = _create(missing_input, tmp_path / "missing-blocked")
    missing_allowed = _create(missing_input, tmp_path / "missing-allowed", require_zero_missing_required_fields=False)

    assert needs_review_blocked["request_status"] == "request_blocked"
    assert "needs_review_candidate_records_present" in needs_review_blocked["request_errors"]
    assert needs_review_allowed["request_status"] == "request_needs_review"
    assert "needs_review_candidate_records_present" in needs_review_allowed["request_warnings"]
    assert missing_blocked["request_status"] == "request_blocked"
    assert "missing_required_fields" in missing_blocked["request_errors"]
    assert missing_allowed["request_status"] == "request_needs_review"
    assert "missing_required_fields" in missing_allowed["request_warnings"]


@pytest.mark.parametrize("marker", FORBIDDEN_MARKERS)
def test_forbidden_markers_block_without_echoing_sensitive_value(tmp_path: Path, marker: str) -> None:
    payload = _valid_precheck_summary()
    payload["unsafe_note"] = f"unsafe {marker}"
    input_path = _write_input(tmp_path, payload)

    summary = _create(input_path, tmp_path)
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["request_status"] == "request_blocked"
    assert "controlled_writer_dry_run_precheck_summary_contains_unsafe_material" in summary["request_errors"]
    assert marker not in serialized


def test_redaction_failure_returns_minimal_blocked_summary_and_no_unsafe_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = _write_input(tmp_path)
    monkeypatch.setattr(request_module, "_markdown", lambda request, summary: "unsafe .pdf")

    summary = _create(input_path, tmp_path)

    assert summary == {
        "schema_version": SUMMARY_SCHEMA,
        "request_status": "request_blocked",
        "request_errors": ["property_training_dataset_controlled_writer_execution_request_redaction_failed"],
        "redaction_status": "failed",
        "writer_execution_authorized": False,
        "explicit_confirmation_required": True,
    }
    assert not list((tmp_path / "out").rglob("*.md"))


def test_cli_return_codes_and_stdout_are_valid_json(tmp_path: Path) -> None:
    passed_input = _write_input(tmp_path / "passed")
    needs_review_payload = _valid_precheck_summary()
    needs_review_payload["needs_review_candidate_record_count"] = 1
    needs_review_input = _write_input(tmp_path / "needs-review", needs_review_payload)
    blocked_payload = _valid_precheck_summary()
    blocked_payload["schema_version"] = "wrong"
    blocked_input = _write_input(tmp_path / "blocked", blocked_payload)

    passed_stdout = io.StringIO()
    needs_review_stdout = io.StringIO()
    blocked_stdout = io.StringIO()
    passed_code = main(
        [
            "--controlled-writer-dry-run-precheck-summary",
            str(passed_input),
            "--output-dir",
            str(tmp_path / "passed-out"),
            "--request-id",
            "request-passed",
            "--requested-by",
            "safe-operator-id",
        ],
        stdout=passed_stdout,
        stderr=io.StringIO(),
    )
    needs_review_code = main(
        [
            "--controlled-writer-dry-run-precheck-summary",
            str(needs_review_input),
            "--output-dir",
            str(tmp_path / "needs-review-out"),
            "--request-id",
            "request-needs-review",
            "--requested-by",
            "safe-operator-id",
            "--allow-needs-review-candidates",
        ],
        stdout=needs_review_stdout,
        stderr=io.StringIO(),
    )
    blocked_code = main(
        [
            "--controlled-writer-dry-run-precheck-summary",
            str(blocked_input),
            "--output-dir",
            str(tmp_path / "blocked-out"),
            "--request-id",
            "request-blocked",
            "--requested-by",
            "safe-operator-id",
        ],
        stdout=blocked_stdout,
        stderr=io.StringIO(),
    )

    assert passed_code == 0
    assert json.loads(passed_stdout.getvalue())["request_status"] == "request_ready_for_preflight"
    assert needs_review_code == 0
    assert json.loads(needs_review_stdout.getvalue())["request_status"] == "request_needs_review"
    assert blocked_code == 1
    assert json.loads(blocked_stdout.getvalue())["request_status"] == "request_blocked"


def test_no_disallowed_artifacts_or_imports_are_created(tmp_path: Path) -> None:
    input_path = _write_input(tmp_path)

    _create(input_path, tmp_path)

    _assert_no_artifact_files(tmp_path)
    source = MODULE_PATH.read_text(encoding="utf-8")
    import_lines = "\n".join(
        line for line in source.splitlines() if line.startswith("import ") or line.startswith("from ")
    ).lower()
    for forbidden in (
        "controlled_writer_execution",
        "execution_request_preflight",
        "explicit_confirmation",
        "controlled_writer_dry_run",
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


def test_execution_request_docs_do_not_include_forbidden_markers() -> None:
    combined = "\n".join(
        [
            DESIGN_DOC.read_text(encoding="utf-8"),
            TEMPLATE_DOC.read_text(encoding="utf-8"),
        ]
    )
    sanitized = combined.replace(
        "This controlled writer execution request does not create CSV/JSONL/Parquet/LMDB artifacts.",
        "This controlled writer execution request does not create FORMAT-LABEL artifacts.",
    )
    sanitized = sanitized.replace("CSV/JSONL/Parquet/LMDB", "FORMAT-LABEL")
    sanitized = sanitized.replace("## Authorization Boundary", "## Auth Boundary")
    sanitized = sanitized.replace("## Authorization State", "## Auth State")
    sanitized = sanitized.replace("would_create_csv_jsonl_parquet_lmdb", "would_create_format_label")
    for marker in FORBIDDEN_MARKERS:
        assert marker not in sanitized
