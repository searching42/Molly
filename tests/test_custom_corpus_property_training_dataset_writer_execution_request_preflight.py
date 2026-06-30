from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_dataset_writer_execution_request import (
    build_property_training_dataset_writer_execution_request,
)
from ai4s_agent.custom_corpus_property_training_dataset_writer_execution_request_preflight import (
    main,
    preflight_property_training_dataset_writer_execution_request,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_dataset_writer_execution_request import (
    _kwargs as _writer_request_kwargs,
)
from test_custom_corpus_property_training_dataset_writer_execution_request import (
    _write_writer_request_package as _write_base_writer_request_package,
)


def test_valid_full_package_returns_passed_and_writes_outputs(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    summary_path = tmp_path / "property-training-dataset-writer-request-preflight-summary.json"
    markdown_path = tmp_path / "property-training-dataset-writer-request-preflight-summary.md"

    summary = preflight_property_training_dataset_writer_execution_request(
        **_kwargs(paths),
        output_summary_path=summary_path,
        output_markdown_path=markdown_path,
    )
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["schema_version"] == "custom_corpus_property_training_dataset_writer_execution_request_preflight.v1"
    assert summary["preflight_status"] == "passed"
    assert written_summary == summary
    assert summary["writer_execution_request_id"] == "property-training-dataset-writer-request-001"
    assert summary["writer_request_record_count"] == 1
    assert summary["row_preview_count"] == 1
    assert summary["planned_dataset_record_count"] == 1
    assert set(summary["requested_output_formats"]) == {"jsonl", "parquet", "lmdb", "csv"}
    assert summary["writer_executed"] is False
    assert summary["training_admitted"] is True
    assert summary["training_dataset_materialized"] is False
    assert summary["dataset_artifact_created"] is False
    assert summary["phase1_status"] == "not_run"
    assert summary["dataset_confirmation_changed"] is False
    assert summary["preflight_errors"] == []
    assert str(tmp_path) not in serialized
    assert "this is a training dataset writer execution request preflight only" in markdown
    assert "no dataset writer was executed" in markdown
    assert "no training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert "no conformers were generated" in markdown
    assert "no DPA3 structures were generated" in markdown


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_dataset_writer_execution_request", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_writer_execution_request_schema_invalid"),
        ("training_dataset_writer_execution_request", lambda payload: payload.__setitem__("request_status", "blocked"), "training_dataset_writer_execution_request_blocked"),
        ("training_dataset_writer_execution_request_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_writer_execution_request_summary_schema_invalid"),
        ("training_dataset_materialization_dry_run_precheck", lambda payload: payload.__setitem__("precheck_status", "blocked"), "training_dataset_materialization_dry_run_precheck_blocked"),
        ("training_dataset_materialization_dry_run_report", lambda payload: payload.__setitem__("dry_run_status", "blocked"), "training_dataset_materialization_dry_run_blocked"),
        ("training_dataset_row_contract", lambda payload: payload.__setitem__("contract_status", "blocked"), "training_dataset_row_contract_blocked"),
        ("training_dataset_materialization_plan", lambda payload: payload.__setitem__("plan_status", "blocked"), "training_dataset_materialization_plan_blocked"),
        ("training_execution_ledger", lambda payload: payload.__setitem__("execution_status", "blocked"), "training_admission_execution_ledger_blocked"),
    ],
)
def test_schema_and_status_failures_block(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = preflight_property_training_dataset_writer_execution_request(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


def test_writer_request_needs_review_blocks_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path, needs_review=True)

    blocked = preflight_property_training_dataset_writer_execution_request(**_kwargs(paths))
    allowed = preflight_property_training_dataset_writer_execution_request(
        **_kwargs(paths),
        allow_writer_request_needs_review=True,
    )

    assert blocked["preflight_status"] == "blocked"
    assert "training_dataset_writer_execution_request_needs_review" in blocked["preflight_errors"]
    assert allowed["preflight_status"] == "needs_review"
    assert "training_dataset_writer_execution_request_needs_review" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "field", "error_code"),
    [
        ("training_dataset_writer_execution_request_summary", "training_dataset_writer_execution_request_sha256", "training_dataset_writer_execution_request_sha256_mismatch"),
        ("training_dataset_writer_execution_request", "training_dataset_materialization_dry_run_precheck_sha256", "training_dataset_materialization_dry_run_precheck_sha256_mismatch"),
        ("training_dataset_writer_execution_request", "training_dataset_materialization_dry_run_report_sha256", "training_dataset_materialization_dry_run_report_sha256_mismatch"),
        ("training_dataset_materialization_dry_run_report", "training_dataset_row_contract_sha256", "training_dataset_row_contract_sha256_mismatch"),
        ("training_dataset_row_contract", "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256_mismatch"),
        ("training_dataset_materialization_plan", "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256_mismatch"),
    ],
)
def test_sha_mismatches_block(tmp_path: Path, target: str, field: str, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = preflight_property_training_dataset_writer_execution_request(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("target", "field", "value", "error_code"),
    [
        ("training_dataset_writer_execution_request", "writer_executed", True, "writer_executed"),
        ("training_dataset_writer_execution_request", "training_admitted", False, "training_not_admitted"),
        ("training_dataset_writer_execution_request", "training_dataset_materialized", True, "training_dataset_materialized"),
        ("training_dataset_writer_execution_request", "dataset_artifact_created", True, "dataset_artifact_created"),
        ("training_dataset_writer_execution_request", "phase1_status", "ran", "phase1_ran"),
        ("training_dataset_writer_execution_request", "dataset_confirmation_changed", True, "dataset_confirmation_changed"),
        ("training_dataset_writer_execution_request", "requested_output_formats", ["xml"], "requested_output_format_invalid"),
        ("training_dataset_writer_execution_request", "requested_writer_mode", "writer_now", "requested_writer_mode_invalid"),
        ("training_dataset_writer_execution_request", "writer_request_record_count", 2, "writer_request_record_count_mismatch"),
        ("training_dataset_writer_execution_request", "writer_request_record_ids", ["other-record"], "writer_request_record_ids_mismatch"),
        ("training_dataset_writer_execution_request", "row_preview_ids", ["other-preview"], "row_preview_ids_mismatch"),
        ("training_dataset_writer_execution_request", "planned_dataset_record_ids", ["other-dataset-record"], "planned_dataset_record_ids_mismatch"),
        ("training_dataset_writer_execution_request_summary", "writer_execution_request_id", "other-request", "writer_execution_request_id_mismatch"),
    ],
)
def test_request_record_id_and_boundary_failures_block(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
    error_code: str,
) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, value))

    summary = preflight_property_training_dataset_writer_execution_request(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("exclude_record_ids", ["property-candidate-001"], "planned_candidate_from_excluded_record"),
        ("blocked_from_training_admission_record_ids", ["property-candidate-001"], "planned_candidate_from_blocked_record"),
        ("needs_review_record_ids", ["property-candidate-001"], "planned_candidate_from_needs_review_record"),
    ],
)
def test_excluded_blocked_or_needs_review_candidate_leakage_blocks(
    tmp_path: Path,
    field: str,
    value: list[str],
    error_code: str,
) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, value))

    summary = preflight_property_training_dataset_writer_execution_request(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda payload: payload["writer_request_records"][0].__setitem__("property_value", "0.72"), "writer_request_record_contains_unsafe_value"),
        (lambda payload: payload["writer_request_records"][0].__setitem__("notes", "serialized dataset row"), "writer_request_record_contains_unsafe_value"),
        (lambda payload: payload["writer_request_records"][0].__setitem__("output_path", "future.jsonl"), "writer_request_record_contains_unsafe_value"),
        (lambda payload: payload["writer_request_records"][0].__setitem__("requested_action", "write_now"), "writer_request_record_action_invalid"),
        (lambda payload: payload["writer_request_records"][0].__setitem__("writer_executed", True), "writer_executed"),
    ],
)
def test_writer_request_record_safety_failures_block(tmp_path: Path, mutator: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths["training_dataset_writer_execution_request"], mutator)

    summary = preflight_property_training_dataset_writer_execution_request(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


def test_summary_uses_safe_basenames_only(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)

    summary = preflight_property_training_dataset_writer_execution_request(**_kwargs(paths))
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["preflight_status"] == "passed"
    assert summary["training_dataset_writer_execution_request_path"] == "property_training_dataset_writer_execution_request.json"
    assert str(tmp_path) not in serialized


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths["training_dataset_writer_execution_request"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_unsafe_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_preflight_package(tmp_path)
    summary_path = tmp_path / "preflight-summary.json"
    markdown_path = tmp_path / "preflight-summary.md"
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_dataset_writer_execution_request_preflight._contains_forbidden_material",
        lambda value: True,
    )

    summary = preflight_property_training_dataset_writer_execution_request(
        **_kwargs(paths),
        output_summary_path=summary_path,
        output_markdown_path=markdown_path,
    )

    assert summary == {
        "schema_version": "custom_corpus_property_training_dataset_writer_execution_request_preflight.v1",
        "preflight_status": "blocked",
        "preflight_errors": ["property_training_dataset_writer_execution_request_preflight_redaction_failed"],
        "redaction_status": "failed",
    }
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    assert not markdown_path.exists()


def test_cli_stdout_valid_json_and_no_dataset_artifacts_created(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["preflight_status"] == "passed"
    assert stderr.getvalue() == ""
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_forbidden_runners_are_not_imported_or_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_property_training_dataset_writer_execution_request",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_dry_run_precheck",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_dry_run",
            "ai4s_agent.custom_corpus_property_training_dataset_row_contract_precheck",
            "ai4s_agent.custom_corpus_property_training_dataset_row_contract",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_planner",
            "ai4s_agent.custom_corpus_property_training_admission_execution_ledger",
            "ai4s_agent.custom_corpus_property_training_admission_execution_dry_run",
            "ai4s_agent.custom_corpus_property_quarantine_materializer",
            "ai4s_agent.custom_corpus_materialization_planner",
            "ai4s_agent.workflows.corpus_to_phase1_workflow",
            "ai4s_agent.document_parse_service",
            "ai4s_agent.mineru",
            "openai",
            "pdfplumber",
        )
        if name.startswith(forbidden):
            raise AssertionError(f"forbidden import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracking_import)

    summary = preflight_property_training_dataset_writer_execution_request(**_kwargs(paths))

    assert summary["preflight_status"] == "passed"
    assert not any("custom_corpus_property_training_dataset_writer_execution_request" in name for name in imported_modules)


def _write_preflight_package(tmp_path: Path, *, needs_review: bool = False) -> dict[str, Path]:
    paths = _write_base_writer_request_package(tmp_path, needs_review=needs_review)
    writer_summary = build_property_training_dataset_writer_execution_request(
        **_writer_request_kwargs(paths),
        confirm_training_dataset_writer_execution_request=True,
        allow_dry_run_precheck_needs_review=needs_review,
    )
    assert writer_summary["request_status"] in {"written", "needs_review"}
    run_dir = paths["writer_execution_request_output_dir"] / "property-training-dataset-writer-request-001"
    paths["training_dataset_writer_execution_request"] = run_dir / "property_training_dataset_writer_execution_request.json"
    paths["training_dataset_writer_execution_request_summary"] = (
        run_dir / "property_training_dataset_writer_execution_request_summary.json"
    )
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "training_dataset_writer_execution_request_path": paths["training_dataset_writer_execution_request"],
        "training_dataset_writer_execution_request_summary_path": paths["training_dataset_writer_execution_request_summary"],
        "training_dataset_materialization_dry_run_precheck_path": paths["training_dataset_materialization_dry_run_precheck"],
        "training_dataset_materialization_dry_run_report_path": paths["training_dataset_materialization_dry_run_report"],
        "training_dataset_materialization_dry_run_summary_path": paths["training_dataset_materialization_dry_run_summary"],
        "training_dataset_row_contract_precheck_path": paths["training_dataset_row_contract_precheck"],
        "training_dataset_row_contract_path": paths["training_dataset_row_contract"],
        "training_dataset_row_contract_summary_path": paths["training_dataset_row_contract_summary"],
        "training_dataset_materialization_plan_precheck_path": paths["training_dataset_materialization_plan_precheck"],
        "training_dataset_materialization_plan_path": paths["training_dataset_materialization_plan"],
        "training_dataset_materialization_planner_summary_path": paths["training_dataset_materialization_planner_summary"],
        "training_admission_execution_ledger_precheck_path": paths["training_execution_ledger_precheck_summary"],
        "training_admission_execution_ledger_path": paths["training_execution_ledger"],
        "training_admission_execution_ledger_summary_path": paths["training_execution_ledger_summary"],
        "training_admission_execution_dry_run_precheck_path": paths["training_execution_dry_run_precheck_summary"],
        "training_admission_execution_dry_run_report_path": paths["training_execution_dry_run_report"],
        "training_admission_execution_request_path": paths["training_execution_request"],
        "training_admission_execution_request_summary_path": paths["training_execution_request_summary"],
        "training_admission_execution_request_preflight_path": paths["training_execution_request_preflight_summary"],
        "training_admission_request_draft_path": paths["training_request_draft"],
        "training_admission_request_draft_summary_path": paths["training_request_draft_summary"],
        "training_admission_request_draft_precheck_path": paths["training_request_draft_precheck_summary"],
        "training_admission_request_plan_path": paths["training_request_plan_summary"],
        "training_admission_request_preflight_path": paths["training_request_preflight_summary"],
        "training_admission_readiness_summary_path": paths["training_admission_readiness_summary"],
        "quarantine_candidate_preflight_summary_path": paths["quarantine_candidate_preflight_summary"],
        "quarantine_candidate_records_path": paths["quarantine_candidate_records"],
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--training-dataset-writer-execution-request",
        str(paths["training_dataset_writer_execution_request"]),
        "--training-dataset-writer-execution-request-summary",
        str(paths["training_dataset_writer_execution_request_summary"]),
        "--training-dataset-materialization-dry-run-precheck",
        str(paths["training_dataset_materialization_dry_run_precheck"]),
        "--training-dataset-materialization-dry-run-report",
        str(paths["training_dataset_materialization_dry_run_report"]),
        "--training-dataset-materialization-dry-run-summary",
        str(paths["training_dataset_materialization_dry_run_summary"]),
        "--training-dataset-row-contract-precheck",
        str(paths["training_dataset_row_contract_precheck"]),
        "--training-dataset-row-contract",
        str(paths["training_dataset_row_contract"]),
        "--training-dataset-row-contract-summary",
        str(paths["training_dataset_row_contract_summary"]),
        "--training-dataset-materialization-plan-precheck",
        str(paths["training_dataset_materialization_plan_precheck"]),
        "--training-dataset-materialization-plan",
        str(paths["training_dataset_materialization_plan"]),
        "--training-dataset-materialization-planner-summary",
        str(paths["training_dataset_materialization_planner_summary"]),
        "--training-admission-execution-ledger-precheck",
        str(paths["training_execution_ledger_precheck_summary"]),
        "--training-admission-execution-ledger",
        str(paths["training_execution_ledger"]),
        "--training-admission-execution-ledger-summary",
        str(paths["training_execution_ledger_summary"]),
        "--training-admission-execution-dry-run-precheck",
        str(paths["training_execution_dry_run_precheck_summary"]),
        "--training-admission-execution-dry-run-report",
        str(paths["training_execution_dry_run_report"]),
        "--training-admission-execution-request",
        str(paths["training_execution_request"]),
        "--training-admission-execution-request-summary",
        str(paths["training_execution_request_summary"]),
        "--training-admission-execution-request-preflight",
        str(paths["training_execution_request_preflight_summary"]),
        "--training-admission-request-draft",
        str(paths["training_request_draft"]),
        "--training-admission-request-draft-summary",
        str(paths["training_request_draft_summary"]),
        "--training-admission-request-draft-precheck",
        str(paths["training_request_draft_precheck_summary"]),
        "--training-admission-request-plan",
        str(paths["training_request_plan_summary"]),
        "--training-admission-request-preflight",
        str(paths["training_request_preflight_summary"]),
        "--training-admission-readiness-summary",
        str(paths["training_admission_readiness_summary"]),
        "--quarantine-candidate-preflight-summary",
        str(paths["quarantine_candidate_preflight_summary"]),
        "--quarantine-candidate-records",
        str(paths["quarantine_candidate_records"]),
    ]
