from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding_plan_preflight import (
    main,
    preflight_property_training_dataset_writer_input_binding_plan,
)
from ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding_planner import (
    build_property_training_dataset_writer_input_binding_plan,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_dataset_writer_input_binding_planner import (
    _kwargs as _binding_plan_kwargs,
)
from test_custom_corpus_property_training_dataset_writer_input_binding_planner import (
    _write_binding_package as _write_binding_base_package,
)


def test_valid_package_returns_passed_and_writes_optional_markdown(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    summary_path = tmp_path / "writer-input-binding-plan-preflight-summary.json"
    markdown_path = tmp_path / "writer-input-binding-plan-preflight-summary.md"

    summary = preflight_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        output_summary_path=summary_path,
        output_markdown_path=markdown_path,
    )
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["schema_version"] == "custom_corpus_property_training_dataset_writer_input_binding_plan_preflight.v1"
    assert summary["preflight_status"] == "passed"
    assert written == summary
    assert summary["writer_input_binding_plan_id"] == "property-writer-input-binding-plan-001"
    assert summary["binding_record_count"] == 1
    assert summary["values_materialized"] is False
    assert summary["writer_executed"] is False
    assert summary["training_dataset_materialized"] is False
    assert summary["dataset_artifact_created"] is False
    assert summary["phase1_status"] == "not_run"
    assert summary["dataset_confirmation_changed"] is False
    assert summary["preflight_errors"] == []
    assert "this is a training dataset writer input binding plan preflight only" in markdown
    assert "no dataset writer was executed" in markdown
    assert "no values were materialized" in markdown
    assert "no training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert str(tmp_path) not in serialized


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_dataset_writer_input_binding_plan", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_writer_input_binding_plan_schema_invalid"),
        ("training_dataset_writer_input_binding_plan", lambda payload: payload.__setitem__("planner_status", "blocked"), "training_dataset_writer_input_binding_plan_blocked"),
        ("training_dataset_writer_input_binding_planner_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_writer_input_binding_planner_summary_schema_invalid"),
        ("training_dataset_writer_execution_request_preflight", lambda payload: payload.__setitem__("preflight_status", "blocked"), "training_dataset_writer_execution_request_preflight_blocked"),
        ("training_dataset_writer_execution_request", lambda payload: payload.__setitem__("request_status", "blocked"), "training_dataset_writer_execution_request_blocked"),
        ("training_dataset_materialization_dry_run_report", lambda payload: payload.__setitem__("dry_run_status", "blocked"), "training_dataset_materialization_dry_run_blocked"),
        ("training_dataset_row_contract", lambda payload: payload.__setitem__("contract_status", "blocked"), "training_dataset_row_contract_blocked"),
        ("training_dataset_materialization_plan", lambda payload: payload.__setitem__("plan_status", "blocked"), "training_dataset_materialization_plan_blocked"),
        ("training_execution_ledger", lambda payload: payload.__setitem__("execution_status", "blocked"), "training_admission_execution_ledger_blocked"),
    ],
)
def test_schema_and_status_failures_block(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = preflight_property_training_dataset_writer_input_binding_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


def test_binding_plan_needs_review_blocks_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path, plan_needs_review=True)

    blocked = preflight_property_training_dataset_writer_input_binding_plan(**_kwargs(paths))
    allowed = preflight_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        allow_binding_plan_needs_review=True,
        require_all_required_fields_bound=False,
    )

    assert blocked["preflight_status"] == "blocked"
    assert "training_dataset_writer_input_binding_plan_needs_review" in blocked["preflight_errors"]
    assert allowed["preflight_status"] == "needs_review"
    assert "training_dataset_writer_input_binding_plan_needs_review" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "field", "error_code"),
    [
        ("training_dataset_writer_input_binding_planner_summary", "training_dataset_writer_input_binding_plan_sha256", "training_dataset_writer_input_binding_plan_sha256_mismatch"),
        ("training_dataset_writer_input_binding_plan", "training_dataset_writer_execution_request_preflight_sha256", "training_dataset_writer_execution_request_preflight_sha256_mismatch"),
        ("training_dataset_writer_input_binding_plan", "training_dataset_writer_execution_request_sha256", "training_dataset_writer_execution_request_sha256_mismatch"),
        ("training_dataset_writer_execution_request", "training_dataset_materialization_dry_run_precheck_sha256", "training_dataset_materialization_dry_run_precheck_sha256_mismatch"),
        ("training_dataset_materialization_dry_run_report", "training_dataset_row_contract_sha256", "training_dataset_row_contract_sha256_mismatch"),
        ("training_dataset_row_contract", "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256_mismatch"),
        ("training_dataset_materialization_plan", "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256_mismatch"),
    ],
)
def test_sha_mismatches_block(tmp_path: Path, target: str, field: str, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = preflight_property_training_dataset_writer_input_binding_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("target", "field", "value", "error_code"),
    [
        ("training_dataset_writer_input_binding_plan", "writer_executed", True, "writer_executed"),
        ("training_dataset_writer_input_binding_plan", "values_materialized", True, "values_materialized"),
        ("training_dataset_writer_input_binding_plan", "training_admitted", False, "training_not_admitted"),
        ("training_dataset_writer_input_binding_plan", "training_dataset_materialized", True, "training_dataset_materialized"),
        ("training_dataset_writer_input_binding_plan", "dataset_artifact_created", True, "dataset_artifact_created"),
        ("training_dataset_writer_input_binding_plan", "phase1_status", "ran", "phase1_ran"),
        ("training_dataset_writer_input_binding_plan", "dataset_confirmation_changed", True, "dataset_confirmation_changed"),
        ("training_dataset_writer_input_binding_plan", "binding_record_count", 2, "binding_record_count_mismatch"),
        ("training_dataset_writer_input_binding_plan", "binding_record_ids", ["other-binding"], "binding_record_ids_mismatch"),
        ("training_dataset_writer_input_binding_plan", "writer_request_record_ids", ["other-request-record"], "writer_request_record_ids_mismatch"),
        ("training_dataset_writer_input_binding_plan", "row_preview_ids", ["other-preview"], "row_preview_ids_mismatch"),
        ("training_dataset_writer_input_binding_plan", "row_contract_id", "other-row-contract", "row_contract_id_mismatch"),
    ],
)
def test_boundary_count_and_id_mismatches_block(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
    error_code: str,
) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, value))

    summary = preflight_property_training_dataset_writer_input_binding_plan(**_kwargs(paths))

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
def test_planned_candidate_from_excluded_blocked_or_needs_review_record_blocks(
    tmp_path: Path,
    field: str,
    value: list[str],
    error_code: str,
) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, value))

    summary = preflight_property_training_dataset_writer_input_binding_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda record: record.__setitem__("binding_record_status", "blocked"), "binding_record_status_invalid"),
        (lambda record: record.__setitem__("writer_executed", True), "writer_executed"),
        (lambda record: record.__setitem__("training_admitted", False), "training_not_admitted"),
        (lambda record: record.__setitem__("dataset_artifact_created", True), "dataset_artifact_created"),
        (lambda record: record.__setitem__("notes", "serialized training row"), "binding_record_contains_unsafe_value"),
        (lambda record: record.__setitem__("output_path", "future.jsonl"), "binding_record_contains_unsafe_value"),
    ],
)
def test_binding_record_safety_failures_block(tmp_path: Path, mutator: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths["training_dataset_writer_input_binding_plan"], lambda payload: mutator(payload["binding_records"][0]))

    summary = preflight_property_training_dataset_writer_input_binding_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda bindings: bindings.pop(), "required_field_bindings_missing"),
        (lambda bindings: bindings[0].__setitem__("binding_status", "invalid"), "required_field_binding_status_invalid"),
        (lambda bindings: bindings[0].__setitem__("source_artifact_label", "invalid_source"), "source_artifact_label_invalid"),
        (lambda bindings: bindings[0].__setitem__("derivation_rule", "derive value now"), "derivation_rule_label_invalid"),
        (lambda bindings: bindings[0].__setitem__("value_materialized", True), "value_materialized"),
    ],
)
def test_required_field_binding_failures_block(tmp_path: Path, mutator: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(
        paths["training_dataset_writer_input_binding_plan"],
        lambda payload: mutator(payload["binding_records"][0]["required_field_bindings"]),
    )

    summary = preflight_property_training_dataset_writer_input_binding_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


def test_required_missing_source_blocks_by_default_and_can_be_needs_review(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path, plan_needs_review=True)

    blocked = preflight_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        allow_binding_plan_needs_review=True,
    )
    allowed = preflight_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        allow_binding_plan_needs_review=True,
        require_all_required_fields_bound=False,
    )

    assert blocked["preflight_status"] == "blocked"
    assert "required_field_source_missing" in blocked["preflight_errors"]
    assert allowed["preflight_status"] == "needs_review"
    assert "required_field_source_missing" in allowed["warnings"]


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda binding: binding.__setitem__("value_materialized", True), "value_materialized"),
        (lambda binding: binding.__setitem__("source_artifact_label", "invalid_source"), "source_artifact_label_invalid"),
    ],
)
def test_optional_field_binding_failures_block(tmp_path: Path, mutator: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(
        paths["training_dataset_writer_input_binding_plan"],
        lambda payload: mutator(payload["binding_records"][0]["optional_field_bindings"][0]),
    )

    summary = preflight_property_training_dataset_writer_input_binding_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("dedup_key_materialized", True, "dedup_key_materialized"),
        ("split_group_key_materialized", True, "split_group_key_materialized"),
        ("split_group_key_default", "row_id", "split_group_key_default_invalid"),
        ("row_id_split_forbidden", False, "row_id_split_not_forbidden"),
    ],
)
def test_dedup_split_binding_failures_block(tmp_path: Path, field: str, value: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(
        paths["training_dataset_writer_input_binding_plan"],
        lambda payload: payload["binding_records"][0]["dedup_split_binding"].__setitem__(field, value),
    )

    summary = preflight_property_training_dataset_writer_input_binding_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    "unsafe_value",
    ["C1=CC=CC=C1", "0.72", "InChI=1S/example", "raw table", "future.csv"],
)
def test_value_and_path_leaks_block(tmp_path: Path, unsafe_value: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(
        paths["training_dataset_writer_input_binding_plan"],
        lambda payload: payload["binding_records"][0]["required_field_bindings"][0].__setitem__("leak", unsafe_value),
    )

    summary = preflight_property_training_dataset_writer_input_binding_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert "binding_record_contains_unsafe_value" in summary["preflight_errors"]


def test_summary_uses_safe_basenames_only(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)

    summary = preflight_property_training_dataset_writer_input_binding_plan(**_kwargs(paths))
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["preflight_status"] == "passed"
    assert summary["training_dataset_writer_input_binding_plan_path"] == "property_training_dataset_writer_input_binding_plan.json"
    assert str(tmp_path) not in serialized


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(
        paths["training_dataset_writer_input_binding_plan"],
        lambda payload: payload.__setitem__("notes", "token abc123"),
    )
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
    summary_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding_plan_preflight._contains_forbidden_material",
        lambda value: True,
    )

    summary = preflight_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        output_summary_path=summary_path,
        output_markdown_path=markdown_path,
    )

    assert summary == {
        "schema_version": "custom_corpus_property_training_dataset_writer_input_binding_plan_preflight.v1",
        "preflight_status": "blocked",
        "preflight_errors": ["property_training_dataset_writer_input_binding_plan_preflight_redaction_failed"],
        "redaction_status": "failed",
    }
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    assert not markdown_path.exists()


def test_cli_stdout_valid_json_and_no_dataset_or_structure_artifacts_created(tmp_path: Path) -> None:
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
    assert not any(tmp_path.glob("**/*conformer*"))
    assert not any(tmp_path.glob("**/*dpa3*"))


def test_forbidden_runners_are_not_imported_or_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding_planner",
            "ai4s_agent.custom_corpus_property_training_dataset_writer_execution_request_preflight",
            "ai4s_agent.custom_corpus_property_training_dataset_writer_execution_request",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_dry_run_precheck",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_dry_run",
            "ai4s_agent.custom_corpus_property_training_dataset_row_contract_precheck",
            "ai4s_agent.custom_corpus_property_training_dataset_row_contract",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_planner",
            "ai4s_agent.custom_corpus_property_training_admission_execution_ledger",
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

    summary = preflight_property_training_dataset_writer_input_binding_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "passed"
    assert not any("custom_corpus_property_training_dataset_writer_input_binding_planner" in name for name in imported_modules)


def _write_preflight_package(tmp_path: Path, *, plan_needs_review: bool = False) -> dict[str, Path]:
    paths = _write_binding_base_package(tmp_path, declare_sources=not plan_needs_review)
    plan_summary = build_property_training_dataset_writer_input_binding_plan(
        **_binding_plan_kwargs(paths),
        confirm_training_dataset_writer_input_binding_plan=True,
        require_all_required_fields_bound=not plan_needs_review,
    )
    assert plan_summary["planner_status"] in {"planned", "needs_review"}
    run_dir = paths["writer_input_binding_output_dir"] / "property-writer-input-binding-plan-001"
    paths["training_dataset_writer_input_binding_plan"] = run_dir / "property_training_dataset_writer_input_binding_plan.json"
    paths["training_dataset_writer_input_binding_planner_summary"] = (
        run_dir / "property_training_dataset_writer_input_binding_planner_summary.json"
    )
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "training_dataset_writer_input_binding_plan_path": paths["training_dataset_writer_input_binding_plan"],
        "training_dataset_writer_input_binding_planner_summary_path": paths[
            "training_dataset_writer_input_binding_planner_summary"
        ],
        "training_dataset_writer_execution_request_preflight_path": paths[
            "training_dataset_writer_execution_request_preflight"
        ],
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
        "--training-dataset-writer-input-binding-plan",
        str(paths["training_dataset_writer_input_binding_plan"]),
        "--training-dataset-writer-input-binding-planner-summary",
        str(paths["training_dataset_writer_input_binding_planner_summary"]),
        "--training-dataset-writer-execution-request-preflight",
        str(paths["training_dataset_writer_execution_request_preflight"]),
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
