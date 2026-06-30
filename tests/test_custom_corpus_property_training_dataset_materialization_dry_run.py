from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_dataset_materialization_dry_run import (
    main,
    run_property_training_dataset_materialization_dry_run,
)
from ai4s_agent.custom_corpus_property_training_dataset_row_contract_precheck import (
    precheck_property_training_dataset_row_contract,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_dataset_row_contract import OPTIONAL_ROW_FIELDS, REQUIRED_ROW_FIELDS
from test_custom_corpus_property_training_dataset_row_contract_precheck import (
    _kwargs as _row_contract_precheck_kwargs,
)
from test_custom_corpus_property_training_dataset_row_contract_precheck import (
    _write_precheck_package as _write_row_contract_precheck_package,
)


def test_valid_full_package_writes_report_summary_and_markdown(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)

    summary = run_property_training_dataset_materialization_dry_run(
        **_kwargs(paths),
        confirm_training_dataset_materialization_dry_run=True,
    )
    run_dir = paths["training_dataset_materialization_dry_run_output_dir"] / "property-training-dataset-dry-run-001"
    report_path = run_dir / "property_training_dataset_materialization_dry_run_report.json"
    summary_path = run_dir / "property_training_dataset_materialization_dry_run_summary.json"
    markdown_path = run_dir / "redacted_property_training_dataset_materialization_dry_run_evidence.md"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    serialized_report = json.dumps(report, sort_keys=True)
    serialized_summary = json.dumps(summary, sort_keys=True)

    assert written_summary == summary
    assert report["schema_version"] == "custom_corpus_property_training_dataset_materialization_dry_run.v1"
    assert summary["schema_version"] == "custom_corpus_property_training_dataset_materialization_dry_run_summary.v1"
    assert report["dry_run_status"] == "passed"
    assert summary["dry_run_status"] == "passed"
    assert report["dry_run_mode"] == "training_dataset_materialization_dry_run_only"
    assert report["training_admitted"] is True
    assert report["training_dataset_materialized"] is False
    assert report["dataset_artifact_created"] is False
    assert report["phase1_status"] == "not_run"
    assert report["dataset_confirmation_changed"] is False
    assert report["row_preview_count"] == 1
    assert report["planned_dataset_record_count"] == 1
    assert report["contract_record_reference_count"] == 1
    assert len(report["row_previews"]) == 1
    assert summary["row_preview_count"] == 1
    assert summary["training_dataset_materialization_dry_run_report_path"] == report_path.name
    assert str(tmp_path) not in serialized_report
    assert str(tmp_path) not in serialized_summary
    assert "this is a training dataset materialization dry-run only" in markdown
    assert "row previews are summaries only, not serialized training rows" in markdown
    assert "no training dataset artifact was created" in markdown
    assert "no conformers were generated" in markdown
    assert "no DPA3 structures were generated" in markdown
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_missing_confirmation_exits_1_and_writes_no_report(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    stdout = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=io.StringIO())
    run_dir = paths["training_dataset_materialization_dry_run_output_dir"] / "property-training-dataset-dry-run-001"

    assert code == 1
    assert json.loads(stdout.getvalue())["dry_run_status"] == "blocked"
    assert not (run_dir / "property_training_dataset_materialization_dry_run_report.json").exists()


def test_row_contract_precheck_needs_review_blocks_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path, needs_review=True)

    blocked = run_property_training_dataset_materialization_dry_run(
        **_kwargs(paths),
        confirm_training_dataset_materialization_dry_run=True,
    )
    allowed = run_property_training_dataset_materialization_dry_run(
        **_kwargs(paths, materialization_dry_run_id="property-training-dataset-dry-run-002"),
        confirm_training_dataset_materialization_dry_run=True,
        allow_row_contract_precheck_needs_review=True,
    )

    assert blocked["dry_run_status"] == "blocked"
    assert "training_dataset_row_contract_precheck_needs_review" in blocked["dry_run_errors"]
    assert allowed["dry_run_status"] == "needs_review"
    assert "training_dataset_row_contract_precheck_needs_review" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_dataset_row_contract_precheck", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_row_contract_precheck_schema_invalid"),
        ("training_dataset_row_contract_precheck", lambda payload: payload.__setitem__("precheck_status", "blocked"), "training_dataset_row_contract_precheck_blocked"),
        ("training_dataset_row_contract", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_row_contract_schema_invalid"),
        ("training_dataset_materialization_plan", lambda payload: payload.__setitem__("plan_status", "blocked"), "training_dataset_materialization_plan_blocked"),
        ("training_execution_ledger", lambda payload: payload.__setitem__("execution_status", "blocked"), "training_admission_execution_ledger_blocked"),
    ],
)
def test_schema_and_status_failures_block(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_dry_run_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = run_property_training_dataset_materialization_dry_run(
        **_kwargs(paths),
        confirm_training_dataset_materialization_dry_run=True,
    )

    assert summary["dry_run_status"] == "blocked"
    assert error_code in summary["dry_run_errors"]


@pytest.mark.parametrize(
    ("target", "field", "error_code"),
    [
        ("training_dataset_row_contract_precheck", "training_dataset_row_contract_sha256", "training_dataset_row_contract_sha256_mismatch"),
        ("training_dataset_row_contract", "training_dataset_materialization_plan_precheck_sha256", "training_dataset_materialization_plan_precheck_sha256_mismatch"),
        ("training_dataset_materialization_plan_precheck", "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256_mismatch"),
        ("training_dataset_materialization_plan", "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256_mismatch"),
    ],
)
def test_sha_mismatches_block(tmp_path: Path, target: str, field: str, error_code: str) -> None:
    paths = _write_dry_run_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = run_property_training_dataset_materialization_dry_run(
        **_kwargs(paths),
        confirm_training_dataset_materialization_dry_run=True,
    )

    assert summary["dry_run_status"] == "blocked"
    assert error_code in summary["dry_run_errors"]


@pytest.mark.parametrize(
    ("target", "field", "value", "error_code"),
    [
        ("training_dataset_row_contract", "row_contract_id", "other-row-contract", "row_contract_id_mismatch"),
        ("training_dataset_materialization_plan", "planned_dataset_records", [], "no_planned_dataset_records"),
        ("training_dataset_row_contract", "contract_record_references", [], "no_contract_record_references"),
        ("training_dataset_materialization_plan", "planned_training_admission_candidate_record_ids", [], "no_planned_candidates"),
        ("training_dataset_row_contract", "training_dataset_materialized", True, "training_dataset_materialized"),
        ("training_dataset_row_contract", "dataset_artifact_created", True, "dataset_artifact_created"),
        ("training_dataset_row_contract", "phase1_status", "ran", "phase1_ran"),
        ("training_dataset_row_contract", "dataset_confirmation_changed", True, "dataset_confirmation_changed"),
    ],
)
def test_id_record_and_boundary_failures_block(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
    error_code: str,
) -> None:
    paths = _write_dry_run_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, value))

    summary = run_property_training_dataset_materialization_dry_run(
        **_kwargs(paths),
        confirm_training_dataset_materialization_dry_run=True,
    )

    assert summary["dry_run_status"] == "blocked"
    assert error_code in summary["dry_run_errors"]


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
    paths = _write_dry_run_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, value))

    summary = run_property_training_dataset_materialization_dry_run(
        **_kwargs(paths),
        confirm_training_dataset_materialization_dry_run=True,
    )

    assert summary["dry_run_status"] == "blocked"
    assert error_code in summary["dry_run_errors"]


def test_row_previews_are_safe_id_hash_field_name_label_summaries_only(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)

    run_property_training_dataset_materialization_dry_run(
        **_kwargs(paths),
        confirm_training_dataset_materialization_dry_run=True,
    )
    report = json.loads(
        (
            paths["training_dataset_materialization_dry_run_output_dir"]
            / "property-training-dataset-dry-run-001"
            / "property_training_dataset_materialization_dry_run_report.json"
        ).read_text(encoding="utf-8")
    )
    preview = report["row_previews"][0]
    serialized = json.dumps(preview, sort_keys=True).lower()

    assert preview["row_preview_status"] == "would_materialize"
    assert preview["would_materialize_row"] is True
    assert set(preview["required_row_fields"]) == REQUIRED_ROW_FIELDS
    assert set(preview["optional_row_fields"]) == OPTIONAL_ROW_FIELDS
    assert preview["missing_required_fields"] == []
    assert "raw" not in serialized
    assert "property_value" in serialized
    assert "serialized dataset row" not in serialized
    assert ".csv" not in serialized
    assert ".jsonl" not in serialized
    assert ".parquet" not in serialized
    assert ".lmdb" not in serialized


def test_field_model_and_output_format_summaries_are_present(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)

    run_property_training_dataset_materialization_dry_run(
        **_kwargs(paths),
        confirm_training_dataset_materialization_dry_run=True,
    )
    report = json.loads(
        (
            paths["training_dataset_materialization_dry_run_output_dir"]
            / "property-training-dataset-dry-run-001"
            / "property_training_dataset_materialization_dry_run_report.json"
        ).read_text(encoding="utf-8")
    )

    assert report["field_coverage_summary"]["required_field_count"] == len(REQUIRED_ROW_FIELDS)
    assert report["field_coverage_summary"]["optional_field_count"] == len(OPTIONAL_ROW_FIELDS)
    assert report["field_coverage_summary"]["missing_required_field_counts"] == {}
    assert set(report["model_family_compatibility_summary"]["counts_by_model_family"]) == {
        "generic_property_predictor",
        "unimol",
        "dpa3",
    }
    assert report["model_family_compatibility_summary"]["unimol_requires_future_conformer_generation"] is True
    assert report["model_family_compatibility_summary"]["dpa3_requires_future_structure_generation"] is True
    assert report["model_family_compatibility_summary"]["conformers_generated"] is False
    assert report["model_family_compatibility_summary"]["dpa3_structures_generated"] is False
    assert set(report["output_format_compatibility_summary"]["counts_by_output_format"]) == {
        "jsonl",
        "parquet",
        "lmdb",
        "csv",
    }
    assert report["output_format_compatibility_summary"]["jsonl_created"] is False
    assert report["output_format_compatibility_summary"]["parquet_created"] is False
    assert report["output_format_compatibility_summary"]["lmdb_created"] is False
    assert report["output_format_compatibility_summary"]["csv_created"] is False


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    run_dir = paths["training_dataset_materialization_dry_run_output_dir"] / "property-training-dataset-dry-run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = run_property_training_dataset_materialization_dry_run(
        **_kwargs(paths),
        confirm_training_dataset_materialization_dry_run=True,
    )

    assert summary["dry_run_status"] == "blocked"
    assert "output_directory_not_clean" in summary["dry_run_errors"]


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    _mutate_json(paths["training_dataset_row_contract"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-dataset-materialization-dry-run"], stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_unsafe_report_or_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_dry_run_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_dataset_materialization_dry_run._contains_forbidden_material",
        lambda value: True,
    )

    summary = run_property_training_dataset_materialization_dry_run(
        **_kwargs(paths),
        confirm_training_dataset_materialization_dry_run=True,
    )
    run_dir = paths["training_dataset_materialization_dry_run_output_dir"] / "property-training-dataset-dry-run-001"

    assert summary == {
        "schema_version": "custom_corpus_property_training_dataset_materialization_dry_run_summary.v1",
        "dry_run_status": "blocked",
        "dry_run_errors": ["property_training_dataset_materialization_dry_run_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "property_training_dataset_materialization_dry_run_report.json").exists()
    assert not (run_dir / "redacted_property_training_dataset_materialization_dry_run_evidence.md").exists()


def test_cli_stdout_valid_json_and_no_dataset_artifacts_created(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-dataset-materialization-dry-run"], stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["dry_run_status"] == "passed"
    assert stderr.getvalue() == ""
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_forbidden_runners_are_not_imported_or_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_property_training_dataset_row_contract_precheck",
            "ai4s_agent.custom_corpus_property_training_dataset_row_contract",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_plan_precheck",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_planner",
            "ai4s_agent.custom_corpus_property_training_admission_execution_ledger",
            "ai4s_agent.custom_corpus_property_training_admission_execution_dry_run",
            "ai4s_agent.custom_corpus_property_training_admission_execution_request",
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

    summary = run_property_training_dataset_materialization_dry_run(
        **_kwargs(paths),
        confirm_training_dataset_materialization_dry_run=True,
    )

    assert summary["dry_run_status"] == "passed"
    assert not any("custom_corpus_property_training_dataset_row_contract_precheck" in name for name in imported_modules)


def _write_dry_run_package(tmp_path: Path, *, needs_review: bool = False) -> dict[str, Path]:
    paths = _write_row_contract_precheck_package(tmp_path, needs_review=needs_review)
    precheck_path = tmp_path / "property-training-dataset-row-contract-precheck-summary.json"
    precheck = precheck_property_training_dataset_row_contract(
        **_row_contract_precheck_kwargs(paths),
        output_summary_path=precheck_path,
        allow_row_contract_needs_review=needs_review,
    )
    assert precheck["precheck_status"] in {"passed", "needs_review"}
    paths["training_dataset_row_contract_precheck"] = precheck_path
    paths["training_dataset_materialization_dry_run_output_dir"] = tmp_path / "property-training-dataset-dry-run-output"
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
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
        "output_dir": paths["training_dataset_materialization_dry_run_output_dir"],
        "materialization_dry_run_id": "property-training-dataset-dry-run-001",
        "created_by": "operator-redacted",
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
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
        "--output-dir",
        str(paths["training_dataset_materialization_dry_run_output_dir"]),
        "--materialization-dry-run-id",
        "property-training-dataset-dry-run-001",
        "--created-by",
        "operator-redacted",
    ]
