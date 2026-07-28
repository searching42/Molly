from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_dataset_writer_value_source_manifest_planner import (
    plan_property_training_dataset_writer_value_source_manifest,
)
from ai4s_agent.custom_corpus_property_training_dataset_writer_value_source_manifest_preflight import (
    main,
    preflight_property_training_dataset_writer_value_source_manifest,
)
from ai4s_agent.custom_corpus_materialization import sha256_file
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_dataset_writer_value_source_manifest_planner import (
    _kwargs as _manifest_kwargs,
)
from test_custom_corpus_property_training_dataset_writer_value_source_manifest_planner import (
    _write_value_source_package as _write_manifest_base_package,
)


def test_valid_package_returns_passed_and_writes_optional_markdown(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    summary_path = tmp_path / "value-source-manifest-preflight-summary.json"
    markdown_path = tmp_path / "value-source-manifest-preflight-summary.md"

    summary = preflight_property_training_dataset_writer_value_source_manifest(
        **_kwargs(paths),
        output_summary_path=summary_path,
        output_markdown_path=markdown_path,
    )
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["schema_version"] == "custom_corpus_property_training_dataset_writer_value_source_manifest_preflight.v1"
    assert summary["preflight_status"] == "passed"
    assert written == summary
    assert summary["value_source_manifest_id"] == "property-value-source-manifest-001"
    assert summary["value_source_record_count"] == 7
    assert summary["source_payloads_read"] is False
    assert summary["values_materialized"] is False
    assert summary["writer_executed"] is False
    assert summary["training_dataset_materialized"] is False
    assert summary["dataset_artifact_created"] is False
    assert summary["phase1_status"] == "not_run"
    assert summary["dataset_confirmation_changed"] is False
    assert summary["preflight_errors"] == []
    assert "this is a training dataset writer value source manifest preflight only" in markdown
    assert "source payloads were not read" in markdown
    assert "no values were materialized" in markdown
    assert "no training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert str(tmp_path) not in serialized


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_dataset_writer_value_source_manifest", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_writer_value_source_manifest_schema_invalid"),
        ("training_dataset_writer_value_source_manifest_planner_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_writer_value_source_manifest_planner_summary_schema_invalid"),
        ("training_dataset_writer_value_source_manifest", lambda payload: payload.__setitem__("planner_status", "blocked"), "training_dataset_writer_value_source_manifest_blocked"),
        ("training_dataset_writer_input_binding_plan_preflight", lambda payload: payload.__setitem__("preflight_status", "blocked"), "training_dataset_writer_input_binding_plan_preflight_blocked"),
        ("training_dataset_writer_input_binding_plan", lambda payload: payload.__setitem__("planner_status", "blocked"), "training_dataset_writer_input_binding_plan_blocked"),
        ("training_dataset_writer_execution_request", lambda payload: payload.__setitem__("request_status", "blocked"), "training_dataset_writer_execution_request_blocked"),
        ("training_dataset_materialization_dry_run_report", lambda payload: payload.__setitem__("dry_run_status", "blocked"), "training_dataset_materialization_dry_run_blocked"),
        ("training_dataset_row_contract", lambda payload: payload.__setitem__("contract_status", "blocked"), "training_dataset_row_contract_blocked"),
        ("training_dataset_materialization_plan", lambda payload: payload.__setitem__("plan_status", "blocked"), "training_dataset_materialization_plan_blocked"),
        ("training_execution_ledger", lambda payload: payload.__setitem__("execution_status", "blocked"), "training_admission_execution_ledger_blocked"),
    ],
)
def test_schema_and_status_mismatches_block(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = preflight_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


def test_manifest_needs_review_blocks_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path, manifest_needs_review=True)

    blocked = preflight_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))
    allowed = preflight_property_training_dataset_writer_value_source_manifest(
        **_kwargs(paths),
        allow_value_source_manifest_needs_review=True,
        require_all_value_fields_covered=False,
    )

    assert blocked["preflight_status"] == "blocked"
    assert "training_dataset_writer_value_source_manifest_needs_review" in blocked["preflight_errors"]
    assert allowed["preflight_status"] == "needs_review"
    assert "training_dataset_writer_value_source_manifest_needs_review" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "field", "error_code"),
    [
        ("training_dataset_writer_value_source_manifest_planner_summary", "training_dataset_writer_value_source_manifest_sha256", "training_dataset_writer_value_source_manifest_sha256_mismatch"),
        ("training_dataset_writer_value_source_manifest", "training_dataset_writer_input_binding_plan_preflight_sha256", "training_dataset_writer_input_binding_plan_preflight_sha256_mismatch"),
        ("training_dataset_writer_value_source_manifest", "training_dataset_writer_input_binding_plan_sha256", "training_dataset_writer_input_binding_plan_sha256_mismatch"),
        ("training_dataset_writer_input_binding_plan", "training_dataset_writer_execution_request_preflight_sha256", "training_dataset_writer_execution_request_preflight_sha256_mismatch"),
        ("training_dataset_writer_execution_request", "training_dataset_materialization_dry_run_precheck_sha256", "training_dataset_materialization_dry_run_precheck_sha256_mismatch"),
        ("training_dataset_materialization_dry_run_report", "training_dataset_row_contract_sha256", "training_dataset_row_contract_sha256_mismatch"),
        ("training_dataset_row_contract", "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256_mismatch"),
        ("training_dataset_materialization_plan", "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256_mismatch"),
    ],
)
def test_sha_mismatches_block(tmp_path: Path, target: str, field: str, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = preflight_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("writer_executed", True, "writer_executed"),
        ("source_payloads_read", True, "source_payloads_read"),
        ("values_materialized", True, "values_materialized"),
        ("training_admitted", False, "training_not_admitted"),
        ("training_dataset_materialized", True, "training_dataset_materialized"),
        ("dataset_artifact_created", True, "dataset_artifact_created"),
        ("phase1_status", "ran", "phase1_ran"),
        ("dataset_confirmation_changed", True, "dataset_confirmation_changed"),
        ("value_source_record_count", 99, "value_source_record_count_mismatch"),
        ("value_source_record_ids", ["other-value-source"], "value_source_record_ids_mismatch"),
        ("binding_record_ids", ["other-binding"], "binding_record_ids_mismatch"),
        ("writer_request_record_ids", ["other-writer-record"], "writer_request_record_ids_mismatch"),
        ("row_contract_id", "other-row-contract", "row_contract_id_mismatch"),
    ],
)
def test_boundary_count_and_id_mismatches_block(tmp_path: Path, field: str, value: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths["training_dataset_writer_value_source_manifest"], lambda payload: payload.__setitem__(field, value))

    summary = preflight_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("field", "error_code"),
    [
        ("exclude_record_ids", "planned_candidate_from_excluded_record"),
        ("blocked_from_training_admission_record_ids", "planned_candidate_from_blocked_record"),
        ("needs_review_record_ids", "planned_candidate_from_needs_review_record"),
    ],
)
def test_planned_candidate_from_excluded_blocked_or_needs_review_record_blocks(
    tmp_path: Path,
    field: str,
    error_code: str,
) -> None:
    paths = _write_preflight_package(tmp_path)
    manifest = json.loads(paths["training_dataset_writer_value_source_manifest"].read_text(encoding="utf-8"))
    planned_candidate_id = manifest["planned_training_admission_candidate_record_ids"][0]
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, [planned_candidate_id]))

    summary = preflight_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda record: record.__setitem__("value_field_name", "invalid_field"), "value_field_name_invalid"),
        (lambda record: record.__setitem__("source_artifact_label", "invalid_source"), "source_artifact_label_invalid"),
        (lambda record: record.__setitem__("source_artifact_basename", "dir/source.json"), "source_artifact_basename_not_safe"),
        (lambda record: record.__setitem__("source_artifact_sha256", "sha256:" + "0" * 64), "source_artifact_sha256_mismatch"),
        (lambda record: record.__setitem__("source_payload_read", True), "source_payload_read"),
        (lambda record: record.__setitem__("value_materialized", True), "value_materialized"),
        (lambda record: record.__setitem__("leak", "0.72"), "value_source_record_contains_unsafe_value"),
        (lambda record: record.__setitem__("leak", "C1=CC=CC=C1"), "value_source_record_contains_unsafe_value"),
        (lambda record: record.__setitem__("leak", "InChI=1S/example"), "value_source_record_contains_unsafe_value"),
        (lambda record: record.__setitem__("leak", "/tmp/private/source"), "value_source_record_contains_unsafe_value"),
        (lambda record: record.__setitem__("leak", "serialized training row"), "value_source_record_contains_unsafe_value"),
    ],
)
def test_value_source_record_safety_failures_block(tmp_path: Path, mutator: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths["training_dataset_writer_value_source_manifest"], lambda payload: mutator(payload["value_source_records"][0]))

    summary = preflight_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


def test_missing_value_source_coverage_blocks_by_default_and_can_be_needs_review(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        if "value_source_records" in payload:
            payload["value_source_records"] = [
                record for record in payload["value_source_records"] if record["value_field_name"] != "property_value"
            ]
            payload["value_source_record_count"] = len(payload["value_source_records"])
            payload["value_source_record_ids"] = [
                record["value_source_record_id"] for record in payload["value_source_records"]
            ]
        else:
            payload["value_source_record_count"] = int(payload["value_source_record_count"]) - 1
            payload["value_source_record_ids"] = [
                record_id
                for record_id in payload["value_source_record_ids"]
                if not str(record_id).endswith("-value-source-property_value")
            ]
        payload["value_field_coverage_summary"]["property_value"] = 0
        payload["missing_value_source_field_counts"]["property_value"] = 1

    _mutate_json(paths["training_dataset_writer_value_source_manifest"], mutate)
    _mutate_json(paths["training_dataset_writer_value_source_manifest_planner_summary"], mutate)
    _mutate_json(
        paths["training_dataset_writer_value_source_manifest_planner_summary"],
        lambda payload: payload.__setitem__(
            "training_dataset_writer_value_source_manifest_sha256",
            sha256_file(paths["training_dataset_writer_value_source_manifest"]),
        ),
    )

    blocked = preflight_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))
    allowed = preflight_property_training_dataset_writer_value_source_manifest(
        **_kwargs(paths),
        require_all_value_fields_covered=False,
    )

    assert blocked["preflight_status"] == "blocked"
    assert "missing_value_source_for_bound_required_field" in blocked["preflight_errors"]
    assert allowed["preflight_status"] == "needs_review"
    assert "missing_value_source_for_bound_required_field" in allowed["warnings"]


def test_summary_uses_safe_basenames_only(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)

    summary = preflight_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["preflight_status"] == "passed"
    assert summary["training_dataset_writer_value_source_manifest_path"] == "property_training_dataset_writer_value_source_manifest.json"
    assert str(tmp_path) not in serialized


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(
        paths["training_dataset_writer_value_source_manifest"],
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
        "ai4s_agent.custom_corpus_property_training_dataset_writer_value_source_manifest_preflight._contains_forbidden_material",
        lambda value: True,
    )

    summary = preflight_property_training_dataset_writer_value_source_manifest(
        **_kwargs(paths),
        output_summary_path=summary_path,
        output_markdown_path=markdown_path,
    )

    assert summary == {
        "schema_version": "custom_corpus_property_training_dataset_writer_value_source_manifest_preflight.v1",
        "preflight_status": "blocked",
        "preflight_errors": ["property_training_dataset_writer_value_source_manifest_preflight_redaction_failed"],
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
            "ai4s_agent.custom_corpus_property_training_dataset_writer_value_source_manifest_planner",
            "ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding_plan_preflight",
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

    summary = preflight_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["preflight_status"] == "passed"
    assert not any("value_source_manifest_planner" in name for name in imported_modules)


def _write_preflight_package(tmp_path: Path, *, manifest_needs_review: bool = False) -> dict[str, Path]:
    paths = _write_manifest_base_package(tmp_path, plan_needs_review=manifest_needs_review)
    manifest_summary = plan_property_training_dataset_writer_value_source_manifest(
        **_manifest_kwargs(paths),
        allow_input_binding_preflight_needs_review=manifest_needs_review,
        require_all_bound_value_fields_covered=not manifest_needs_review,
    )
    assert manifest_summary["planner_status"] in {"planned", "needs_review"}
    run_dir = paths["value_source_output_dir"] / "property-value-source-manifest-001"
    paths["training_dataset_writer_value_source_manifest"] = (
        run_dir / "property_training_dataset_writer_value_source_manifest.json"
    )
    paths["training_dataset_writer_value_source_manifest_planner_summary"] = (
        run_dir / "property_training_dataset_writer_value_source_manifest_planner_summary.json"
    )
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "training_dataset_writer_value_source_manifest_path": paths["training_dataset_writer_value_source_manifest"],
        "training_dataset_writer_value_source_manifest_planner_summary_path": paths[
            "training_dataset_writer_value_source_manifest_planner_summary"
        ],
        "training_dataset_writer_input_binding_plan_preflight_path": paths[
            "training_dataset_writer_input_binding_plan_preflight"
        ],
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
        "--training-dataset-writer-value-source-manifest",
        str(paths["training_dataset_writer_value_source_manifest"]),
        "--training-dataset-writer-value-source-manifest-planner-summary",
        str(paths["training_dataset_writer_value_source_manifest_planner_summary"]),
        "--training-dataset-writer-input-binding-plan-preflight",
        str(paths["training_dataset_writer_input_binding_plan_preflight"]),
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
