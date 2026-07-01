from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from ai4s_agent import custom_corpus_property_training_dataset_controlled_writer_dry_run as dry_run_module
from ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_dry_run import (
    main,
    run_property_training_dataset_controlled_writer_dry_run,
)


INPUT_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_dry_run_input.v1"
REPORT_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_dry_run_report.v1"
SUMMARY_SCHEMA = "custom_corpus_property_training_dataset_controlled_writer_dry_run_summary.v1"

DESIGN_DOC = Path("docs/custom-corpus-property-training-dataset-controlled-writer-dry-run.md")
TEMPLATE_DOC = Path(
    "docs/evidence/templates/custom-corpus-property-training-dataset-controlled-writer-dry-run-evidence-template.md"
)
MODULE_PATH = Path(
    "src/ai4s_agent/custom_corpus_property_training_dataset_controlled_writer_dry_run.py"
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
    "-> future controlled writer execution request\n"
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
    Path("docs/phase-1-4-milestone-status.md"),
)

DOC_SECTIONS = (
    "# Custom Corpus Property Training Dataset Controlled Writer Dry-Run",
    "## Purpose",
    "## Position in the Governance Chain",
    "## Input Package",
    "## Dry-Run Checks",
    "## Status Semantics",
    "## Output Files",
    "## Redaction Policy",
    "## Side-Effect Boundary",
    "## CLI Usage",
    "## Blocked Conditions",
    "## Out of Scope",
    "## Next Step",
)

TEMPLATE_PLACEHOLDERS = (
    "<controlled_writer_dry_run_evidence_id>",
    "<date>",
    "<operator>",
    "<corpus_id>",
    "<dataset_name>",
    "<dry_run_id>",
    "<dry_run_status>",
    "<controlled_writer_design_plan_preflight_status>",
    "<domain_validation_boundary_status>",
    "<controlled_writer_value_resolution_dry_run_precheck_status>",
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
    "This controlled writer dry-run does not execute the controlled writer.",
    "This controlled writer dry-run does not emit raw values.",
    "This controlled writer dry-run does not materialize values.",
    "This controlled writer dry-run does not serialize training rows.",
    "This controlled writer dry-run does not create training dataset artifacts.",
    "This controlled writer dry-run does not create CSV/JSONL/Parquet/LMDB artifacts.",
    "This controlled writer dry-run does not generate conformers.",
    "This controlled writer dry-run does not generate DPA3 structures.",
    "This controlled writer dry-run does not run Phase 1.",
    "This controlled writer dry-run does not modify DatasetConfirmation.",
    "This controlled writer dry-run does not run model training or evaluation.",
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


def _valid_input() -> dict[str, Any]:
    return {
        "schema_version": INPUT_SCHEMA,
        "dry_run_id": "controlled-writer-dry-run-001",
        "corpus_id": "safe-corpus-id",
        "dataset_name": "safe-dataset-name",
        "controlled_writer_design_plan_preflight_status": "passed",
        "controlled_writer_design_plan_preflight_id": "design-plan-preflight-001",
        "domain_validation_boundary_status": "passed",
        "controlled_writer_value_resolution_dry_run_precheck_status": "passed",
        "accepted_candidate_record_count": 3,
        "needs_review_candidate_record_count": 0,
        "blocked_candidate_record_count": 0,
        "field_coverage": {
            "required_field_count": 5,
            "resolved_required_field_count": 5,
            "missing_required_field_count": 0,
        },
        "would_write": {
            "would_write_row_count": 3,
            "would_write_field_count": 15,
            "would_create_training_dataset_artifact": False,
            "would_create_csv_jsonl_parquet_lmdb": False,
            "would_serialize_rows": False,
            "would_materialize_values": False,
        },
        "boundary_flags": {
            "controlled_writer_executed": False,
            "training_dataset_materialized": False,
            "dataset_artifact_created": False,
            "serialized_rows_created": False,
            "phase1_status": "not_run",
            "dataset_confirmation_changed": False,
            "model_training_run": False,
            "evaluation_run": False,
        },
        "redaction_status": "passed",
    }


def _write_input(tmp_path: Path, payload: dict[str, Any] | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "controlled_writer_dry_run_input.json"
    path.write_text(json.dumps(payload or _valid_input(), indent=2), encoding="utf-8")
    return path


def _run(
    tmp_path: Path,
    payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], Path]:
    input_path = _write_input(tmp_path, payload)
    output_dir = tmp_path / "out"
    summary = run_property_training_dataset_controlled_writer_dry_run(
        controlled_writer_dry_run_input_path=input_path,
        output_dir=output_dir,
        **kwargs,
    )
    return summary, output_dir / (payload or _valid_input()).get("dry_run_id", "")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_no_artifact_files(root: Path) -> None:
    forbidden_suffixes = (".csv", ".jsonl", ".parquet", ".lmdb")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        assert not lowered.endswith(forbidden_suffixes)
        assert "conformer" not in lowered
        assert "dpa3" not in lowered


def test_valid_input_returns_passed_and_writes_report_summary_markdown(tmp_path: Path) -> None:
    summary, run_dir = _run(tmp_path)

    assert summary["dry_run_status"] == "passed"
    report_path = run_dir / "property_training_dataset_controlled_writer_dry_run_report.json"
    summary_path = run_dir / "property_training_dataset_controlled_writer_dry_run_summary.json"
    markdown_path = run_dir / "redacted_property_training_dataset_controlled_writer_dry_run_evidence.md"
    assert report_path.exists()
    assert summary_path.exists()
    assert markdown_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert report["schema_version"] == REPORT_SCHEMA
    assert written_summary["schema_version"] == SUMMARY_SCHEMA
    assert written_summary["dry_run_status"] == "passed"
    assert written_summary["dry_run_report_basename"] == report_path.name
    assert written_summary["dry_run_report_sha256"] == _sha256(report_path)
    assert "This controlled writer dry-run does not execute the controlled writer." in markdown
    assert "raw values" in markdown
    assert "/Users/" not in json.dumps(written_summary)
    _assert_no_artifact_files(tmp_path)


def test_report_and_summary_are_aggregate_only_and_preserve_boundary_flags(tmp_path: Path) -> None:
    summary, run_dir = _run(tmp_path)
    report = json.loads((run_dir / "property_training_dataset_controlled_writer_dry_run_report.json").read_text())

    assert summary["controlled_writer_executed"] is False
    assert summary["training_dataset_materialized"] is False
    assert summary["dataset_artifact_created"] is False
    assert summary["serialized_rows_created"] is False
    assert summary["phase1_status"] == "not_run"
    assert summary["dataset_confirmation_changed"] is False
    assert summary["model_training_run"] is False
    assert summary["evaluation_run"] is False
    assert report["would_write_row_count"] == 3
    emitted = json.dumps({"summary": summary, "report": report})
    assert "raw article text" not in emitted
    assert "C1=CC" not in emitted
    assert "InChI=" not in emitted
    assert "serialized dataset row" not in emitted


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    payload = _valid_input()
    run_dir = tmp_path / "out" / payload["dry_run_id"]
    run_dir.mkdir(parents=True)

    summary, _ = _run(tmp_path, payload)

    assert summary["dry_run_status"] == "blocked"
    assert "controlled_writer_dry_run_output_dir_not_clean" in summary["dry_run_errors"]


def test_nonempty_output_directory_must_be_clean(tmp_path: Path) -> None:
    payload = _valid_input()
    run_dir = tmp_path / "out" / payload["dry_run_id"]
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("occupied", encoding="utf-8")

    summary, _ = _run(tmp_path, payload)

    assert summary["dry_run_status"] == "blocked"
    assert "controlled_writer_dry_run_output_dir_not_clean" in summary["dry_run_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema_version", "wrong.schema", "controlled_writer_dry_run_input_schema_invalid"),
        ("dry_run_id", "unsafe/id", "dry_run_id_unsafe"),
        ("corpus_id", "unsafe id", "corpus_id_unsafe"),
        ("dataset_name", "../unsafe", "dataset_name_unsafe"),
    ],
)
def test_schema_and_safe_id_violations_block(
    tmp_path: Path,
    field: str,
    value: Any,
    error: str,
) -> None:
    payload = _valid_input()
    payload[field] = value

    summary, run_dir = _run(tmp_path, payload)

    assert summary["dry_run_status"] == "blocked"
    assert error in summary["dry_run_errors"]
    assert not (run_dir / "property_training_dataset_controlled_writer_dry_run_report.json").exists()


def test_missing_top_level_required_field_blocks(tmp_path: Path) -> None:
    payload = _valid_input()
    del payload["field_coverage"]

    summary, _ = _run(tmp_path, payload)

    assert summary["dry_run_status"] == "blocked"
    assert "field_coverage_missing" in summary["dry_run_errors"]


@pytest.mark.parametrize(
    ("field", "flag_name", "expected"),
    [
        (
            "controlled_writer_design_plan_preflight_status",
            "require_design_plan_preflight_passed",
            "controlled_writer_design_plan_preflight_status_needs_review",
        ),
        (
            "domain_validation_boundary_status",
            "require_domain_validation_passed",
            "domain_validation_boundary_status_needs_review",
        ),
        (
            "controlled_writer_value_resolution_dry_run_precheck_status",
            "require_value_resolution_precheck_passed",
            "controlled_writer_value_resolution_dry_run_precheck_status_needs_review",
        ),
    ],
)
def test_needs_review_upstream_blocks_by_default_and_can_be_allowed(
    tmp_path: Path,
    field: str,
    flag_name: str,
    expected: str,
) -> None:
    payload = _valid_input()
    payload[field] = "needs_review"

    blocked, _ = _run(tmp_path / "blocked", payload)
    allowed, _ = _run(tmp_path / "allowed", payload, **{flag_name: False})

    assert blocked["dry_run_status"] == "blocked"
    assert expected in blocked["dry_run_errors"]
    assert allowed["dry_run_status"] == "needs_review"
    assert expected in allowed["dry_run_warnings"]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        (
            "controlled_writer_design_plan_preflight_status",
            "blocked",
            "controlled_writer_design_plan_preflight_status_blocked",
        ),
        ("domain_validation_boundary_status", "blocked", "domain_validation_boundary_status_blocked"),
        (
            "controlled_writer_value_resolution_dry_run_precheck_status",
            "blocked",
            "controlled_writer_value_resolution_dry_run_precheck_status_blocked",
        ),
    ],
)
def test_upstream_blocked_status_blocks(tmp_path: Path, field: str, value: str, expected: str) -> None:
    payload = _valid_input()
    payload[field] = value

    summary, _ = _run(tmp_path, payload)

    assert summary["dry_run_status"] == "blocked"
    assert expected in summary["dry_run_errors"]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("accepted_candidate_record_count", 0, "minimum_accepted_candidate_records_not_met"),
        ("blocked_candidate_record_count", 1, "blocked_candidate_records_present"),
        ("needs_review_candidate_record_count", 1, "needs_review_candidate_records_present"),
    ],
)
def test_candidate_count_violations_block(tmp_path: Path, field: str, value: int, expected: str) -> None:
    payload = _valid_input()
    payload[field] = value

    summary, _ = _run(tmp_path, payload)

    assert summary["dry_run_status"] == "blocked"
    assert expected in summary["dry_run_errors"]


def test_needs_review_candidates_can_be_allowed_only_with_explicit_flag(tmp_path: Path) -> None:
    payload = _valid_input()
    payload["needs_review_candidate_record_count"] = 1

    summary, _ = _run(tmp_path, payload, allow_needs_review_candidates=True)

    assert summary["dry_run_status"] == "needs_review"
    assert "needs_review_candidate_records_present" in summary["dry_run_warnings"]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("required_field_count", None, "required_field_count_invalid"),
        ("resolved_required_field_count", -1, "resolved_required_field_count_invalid"),
        ("missing_required_field_count", 1, "missing_required_fields"),
    ],
)
def test_field_coverage_violations_block(
    tmp_path: Path,
    field: str,
    value: Any,
    expected: str,
) -> None:
    payload = _valid_input()
    payload["field_coverage"][field] = value

    summary, _ = _run(tmp_path, payload)

    assert summary["dry_run_status"] == "blocked"
    assert expected in summary["dry_run_errors"]


def test_missing_required_fields_can_be_needs_review_when_values_not_required(tmp_path: Path) -> None:
    payload = _valid_input()
    payload["field_coverage"]["resolved_required_field_count"] = 4
    payload["field_coverage"]["missing_required_field_count"] = 1

    summary, _ = _run(tmp_path, payload, require_values_resolved=False)

    assert summary["dry_run_status"] == "needs_review"
    assert "missing_required_fields" in summary["dry_run_warnings"]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("would_write_row_count", None, "would_write_row_count_invalid"),
        ("would_write_field_count", 0, "would_write_field_count_invalid"),
        ("would_create_training_dataset_artifact", True, "would_create_training_dataset_artifact"),
        ("would_create_csv_jsonl_parquet_lmdb", True, "would_create_csv_jsonl_parquet_lmdb"),
        ("would_serialize_rows", True, "would_serialize_rows"),
        ("would_materialize_values", True, "would_materialize_values"),
    ],
)
def test_would_write_violations_block(
    tmp_path: Path,
    field: str,
    value: Any,
    expected: str,
) -> None:
    payload = _valid_input()
    payload["would_write"][field] = value

    summary, _ = _run(tmp_path, payload)

    assert summary["dry_run_status"] == "blocked"
    assert expected in summary["dry_run_errors"]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("controlled_writer_executed", True, "controlled_writer_executed"),
        ("training_dataset_materialized", True, "training_dataset_materialized"),
        ("dataset_artifact_created", True, "dataset_artifact_created"),
        ("serialized_rows_created", True, "serialized_rows_created"),
        ("phase1_status", "ran", "phase1_ran"),
        ("dataset_confirmation_changed", True, "dataset_confirmation_changed"),
        ("model_training_run", True, "model_training_run"),
        ("evaluation_run", True, "evaluation_run"),
    ],
)
def test_boundary_violations_block(
    tmp_path: Path,
    field: str,
    value: Any,
    expected: str,
) -> None:
    payload = _valid_input()
    payload["boundary_flags"][field] = value

    summary, _ = _run(tmp_path, payload)

    assert summary["dry_run_status"] == "blocked"
    assert expected in summary["dry_run_errors"]


@pytest.mark.parametrize("marker", FORBIDDEN_MARKERS)
def test_forbidden_markers_block_without_echoing_sensitive_value(tmp_path: Path, marker: str) -> None:
    payload = _valid_input()
    payload["unsafe_note"] = f"unsafe {marker}"

    summary, run_dir = _run(tmp_path, payload)
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["dry_run_status"] == "blocked"
    assert "controlled_writer_dry_run_input_contains_unsafe_material" in summary["dry_run_errors"]
    assert marker not in serialized
    assert not (run_dir / "redacted_property_training_dataset_controlled_writer_dry_run_evidence.md").exists()


def test_redaction_failure_returns_minimal_summary_and_no_unsafe_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dry_run_module, "_markdown", lambda report, summary: "unsafe .pdf")

    summary, run_dir = _run(tmp_path)

    assert summary == {
        "schema_version": SUMMARY_SCHEMA,
        "dry_run_status": "blocked",
        "dry_run_errors": ["property_training_dataset_controlled_writer_dry_run_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "redacted_property_training_dataset_controlled_writer_dry_run_evidence.md").exists()
    assert not (run_dir / "property_training_dataset_controlled_writer_dry_run_report.json").exists()


def test_cli_returns_zero_for_passed_and_stdout_is_valid_json(tmp_path: Path) -> None:
    input_path = _write_input(tmp_path)
    stdout = io.StringIO()

    code = main(
        [
            "--controlled-writer-dry-run-input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["dry_run_status"] == "passed"


def test_cli_returns_one_for_blocked(tmp_path: Path) -> None:
    payload = _valid_input()
    payload["schema_version"] = "wrong.schema"
    input_path = _write_input(tmp_path, payload)
    stdout = io.StringIO()

    code = main(
        [
            "--controlled-writer-dry-run-input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    result = json.loads(stdout.getvalue())
    assert code == 1
    assert result["dry_run_status"] == "blocked"


def test_no_disallowed_runtime_imports_or_calls_are_added() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    import_lines = "\n".join(
        line for line in source.splitlines() if line.startswith("import ") or line.startswith("from ")
    ).lower()

    for forbidden in (
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
        "controlled_writer_execution",
        "dry_run_precheck",
        "dataset_writer",
    ):
        assert forbidden not in import_lines


def test_docs_and_evidence_template_exist_and_are_linked() -> None:
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
    assert REPORT_SCHEMA in design_text
    assert SUMMARY_SCHEMA in design_text
    for doc in GOVERNANCE_DOCS:
        assert CHAIN_PHRASE in doc.read_text(encoding="utf-8")


def test_dry_run_docs_do_not_include_forbidden_markers() -> None:
    combined = "\n".join(
        [
            DESIGN_DOC.read_text(encoding="utf-8"),
            TEMPLATE_DOC.read_text(encoding="utf-8"),
        ]
    )
    sanitized = combined.replace(
        "This controlled writer dry-run does not create CSV/JSONL/Parquet/LMDB artifacts.",
        "This controlled writer dry-run does not create FORMAT-LABEL artifacts.",
    )
    sanitized = sanitized.replace("CSV/JSONL/Parquet/LMDB", "FORMAT-LABEL")
    sanitized = sanitized.replace(
        "would_create_csv_jsonl_parquet_lmdb",
        "would_create_format_label",
    )

    for marker in FORBIDDEN_MARKERS:
        assert marker not in sanitized
