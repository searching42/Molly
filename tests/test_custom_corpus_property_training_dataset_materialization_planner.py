from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_admission_execution_ledger_precheck import (
    precheck_property_training_admission_execution_ledger_package,
)
from ai4s_agent.custom_corpus_property_training_dataset_materialization_planner import (
    main,
    plan_property_training_dataset_materialization,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_admission_execution_ledger_precheck import (
    _kwargs as _ledger_precheck_kwargs,
)
from test_custom_corpus_property_training_admission_execution_ledger_precheck import (
    _write_ledger_precheck_package,
)


def test_valid_full_package_writes_plan_summary_and_markdown(tmp_path: Path) -> None:
    paths = _write_planner_package(tmp_path)

    summary = plan_property_training_dataset_materialization(
        **_kwargs(paths),
        confirm_training_dataset_materialization_plan=True,
        planned_output_formats=["jsonl", "parquet"],
        target_model_families=["generic_property_predictor", "unimol"],
    )

    run_dir = paths["training_dataset_materialization_output_dir"] / "property-training-dataset-materialization-plan-001"
    plan_path = run_dir / "property_training_dataset_materialization_plan.json"
    summary_path = run_dir / "property_training_dataset_materialization_planner_summary.json"
    markdown_path = run_dir / "redacted_property_training_dataset_materialization_plan_evidence.md"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    serialized_plan = json.dumps(plan, sort_keys=True)
    serialized_summary = json.dumps(summary, sort_keys=True)

    assert written_summary == summary
    assert plan["schema_version"] == "custom_corpus_property_training_dataset_materialization_plan.v1"
    assert summary["schema_version"] == "custom_corpus_property_training_dataset_materialization_planner.v1"
    assert plan["plan_status"] == "planned"
    assert summary["plan_status"] == "planned"
    assert plan["plan_mode"] == "training_dataset_materialization_plan_only"
    assert plan["planned_output_formats"] == ["jsonl", "parquet"]
    assert plan["target_model_families"] == ["generic_property_predictor", "unimol"]
    assert plan["training_admitted"] is True
    assert plan["training_dataset_materialized"] is False
    assert plan["dataset_artifact_created"] is False
    assert plan["phase1_status"] == "not_run"
    assert plan["dataset_confirmation_changed"] is False
    assert plan["planned_dataset_record_count"] == 1
    assert summary["planned_dataset_record_count"] == 1
    assert plan["planning_errors"] == []
    assert summary["planning_errors"] == []
    assert summary["training_dataset_materialization_plan_path"] == plan_path.name
    assert str(tmp_path) not in serialized_plan
    assert str(tmp_path) not in serialized_summary
    assert "this is a training dataset materialization plan only" in markdown
    assert "no training dataset artifact was created" in markdown
    assert "no training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert "DatasetConfirmation was not changed" in markdown
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_missing_confirmation_exits_1_and_writes_no_plan(tmp_path: Path) -> None:
    paths = _write_planner_package(tmp_path)
    stdout = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=io.StringIO())
    run_dir = paths["training_dataset_materialization_output_dir"] / "property-training-dataset-materialization-plan-001"

    assert code == 1
    assert json.loads(stdout.getvalue())["plan_status"] == "blocked"
    assert not (run_dir / "property_training_dataset_materialization_plan.json").exists()


def test_ledger_precheck_needs_review_blocks_by_default_and_can_write_needs_review(tmp_path: Path) -> None:
    paths = _write_planner_package(tmp_path, needs_review=True)

    blocked = plan_property_training_dataset_materialization(
        **_kwargs(paths),
        confirm_training_dataset_materialization_plan=True,
    )
    allowed = plan_property_training_dataset_materialization(
        **_kwargs(paths, materialization_plan_id="property-training-dataset-materialization-plan-002"),
        confirm_training_dataset_materialization_plan=True,
        allow_ledger_precheck_needs_review=True,
    )

    assert blocked["plan_status"] == "blocked"
    assert "training_admission_execution_ledger_precheck_needs_review" in blocked["planning_errors"]
    assert allowed["plan_status"] == "needs_review"
    assert "training_admission_execution_ledger_precheck_needs_review" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_execution_ledger_precheck_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_admission_execution_ledger_precheck_schema_invalid"),
        ("training_execution_ledger_precheck_summary", lambda payload: payload.__setitem__("precheck_status", "blocked"), "training_admission_execution_ledger_precheck_blocked"),
        ("training_execution_ledger", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_admission_execution_ledger_schema_invalid"),
        ("training_execution_ledger", lambda payload: payload.__setitem__("execution_status", "blocked"), "training_admission_execution_ledger_blocked"),
        ("training_execution_ledger_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_admission_execution_ledger_summary_schema_invalid"),
        ("training_execution_ledger_summary", lambda payload: payload.__setitem__("execution_status", "blocked"), "training_admission_execution_ledger_summary_blocked"),
        ("training_execution_dry_run_precheck_summary", lambda payload: payload.__setitem__("preflight_status", "blocked"), "training_admission_execution_dry_run_precheck_blocked"),
        ("training_execution_dry_run_report", lambda payload: payload.__setitem__("dry_run_status", "blocked"), "training_admission_execution_dry_run_blocked"),
        ("training_execution_request_preflight_summary", lambda payload: payload.__setitem__("preflight_status", "blocked"), "training_admission_execution_request_preflight_blocked"),
        ("training_execution_request", lambda payload: payload.__setitem__("request_status", "blocked"), "training_admission_execution_request_blocked"),
        ("training_request_draft_precheck_summary", lambda payload: payload.__setitem__("precheck_status", "blocked"), "training_admission_request_draft_precheck_blocked"),
        ("training_request_plan_summary", lambda payload: payload.__setitem__("planner_status", "blocked"), "training_admission_request_plan_blocked"),
        ("training_admission_readiness_summary", lambda payload: payload.__setitem__("readiness_status", "blocked"), "training_admission_readiness_blocked"),
    ],
)
def test_blocking_input_failures(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_planner_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = plan_property_training_dataset_materialization(
        **_kwargs(paths),
        confirm_training_dataset_materialization_plan=True,
    )

    assert summary["plan_status"] == "blocked"
    assert error_code in summary["planning_errors"]


@pytest.mark.parametrize(
    ("target", "field", "error_code"),
    [
        ("training_execution_ledger_precheck_summary", "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256_mismatch"),
        ("training_execution_ledger_summary", "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256_mismatch"),
        ("training_execution_ledger", "training_admission_execution_dry_run_report_sha256", "training_admission_execution_dry_run_report_sha256_mismatch"),
        ("training_execution_dry_run_precheck_summary", "training_admission_execution_dry_run_report_sha256", "training_admission_execution_dry_run_report_sha256_mismatch"),
        ("training_execution_request", "source_training_admission_request_draft_sha256", "training_admission_request_draft_sha256_mismatch"),
    ],
)
def test_sha_mismatches_block(tmp_path: Path, target: str, field: str, error_code: str) -> None:
    paths = _write_planner_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = plan_property_training_dataset_materialization(
        **_kwargs(paths),
        confirm_training_dataset_materialization_plan=True,
    )

    assert summary["plan_status"] == "blocked"
    assert error_code in summary["planning_errors"]


def test_id_mismatch_blocks(tmp_path: Path) -> None:
    paths = _write_planner_package(tmp_path)
    _mutate_json(paths["training_execution_ledger"], lambda payload: payload.__setitem__("corpus_id", "other-corpus"))

    summary = plan_property_training_dataset_materialization(
        **_kwargs(paths),
        confirm_training_dataset_materialization_plan=True,
    )

    assert summary["plan_status"] == "blocked"
    assert "corpus_id_mismatch" in summary["planning_errors"]


@pytest.mark.parametrize(
    ("target", "field", "value", "error_code"),
    [
        ("training_execution_ledger", "ledger_records", [], "no_ledger_records"),
        ("training_execution_ledger", "ledger_record_count", 2, "ledger_record_count_mismatch"),
        ("training_execution_ledger", "planned_training_admission_candidate_record_ids", [], "no_planned_candidates"),
        ("training_execution_ledger_precheck_summary", "planned_candidate_count", 2, "planned_candidate_count_mismatch"),
        ("training_execution_dry_run_report", "dry_run_records", [], "no_dry_run_records"),
        ("training_execution_request", "execution_records", [], "no_execution_records"),
    ],
)
def test_record_consistency_failures(tmp_path: Path, target: str, field: str, value: object, error_code: str) -> None:
    paths = _write_planner_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, value))

    summary = plan_property_training_dataset_materialization(
        **_kwargs(paths),
        confirm_training_dataset_materialization_plan=True,
    )

    assert summary["plan_status"] == "blocked"
    assert error_code in summary["planning_errors"]


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
    paths = _write_planner_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, value))

    summary = plan_property_training_dataset_materialization(
        **_kwargs(paths),
        confirm_training_dataset_materialization_plan=True,
    )

    assert summary["plan_status"] == "blocked"
    assert error_code in summary["planning_errors"]


@pytest.mark.parametrize(
    ("target", "field", "value", "error_code"),
    [
        ("training_execution_ledger", "dataset_confirmation_changed", True, "dataset_confirmation_changed"),
        ("training_execution_ledger", "phase1_status", "success", "phase1_ran"),
        ("training_execution_ledger", "training_dataset_materialized", True, "training_dataset_materialized"),
        ("training_execution_ledger", "dataset_artifact_created", True, "dataset_artifact_created"),
        ("training_execution_dry_run_report", "training_admitted", True, "training_admitted_before_ledger"),
    ],
)
def test_boundary_violations_block(tmp_path: Path, target: str, field: str, value: object, error_code: str) -> None:
    paths = _write_planner_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, value))

    summary = plan_property_training_dataset_materialization(
        **_kwargs(paths),
        confirm_training_dataset_materialization_plan=True,
    )

    assert summary["plan_status"] == "blocked"
    assert error_code in summary["planning_errors"]


def test_planned_dataset_records_are_safe_id_hash_and_label_only(tmp_path: Path) -> None:
    paths = _write_planner_package(tmp_path)

    summary = plan_property_training_dataset_materialization(
        **_kwargs(paths),
        confirm_training_dataset_materialization_plan=True,
    )
    run_dir = paths["training_dataset_materialization_output_dir"] / "property-training-dataset-materialization-plan-001"
    plan = json.loads((run_dir / "property_training_dataset_materialization_plan.json").read_text(encoding="utf-8"))
    record = plan["planned_dataset_records"][0]

    assert summary["plan_status"] == "planned"
    assert record["planned_action"] == "materialize_training_dataset_record"
    assert record["planned_record_status"] == "planned"
    assert record["training_admitted"] is True
    assert record["training_dataset_materialized"] is False
    assert record["dataset_artifact_created"] is False
    assert record["phase1_status"] == "not_run"
    assert record["dataset_confirmation_changed"] is False
    assert set(record) == {
        "planned_dataset_record_id",
        "ledger_record_id",
        "dry_run_record_id",
        "execution_record_id",
        "draft_record_id",
        "candidate_record_id",
        "record_id",
        "materialization_record_id",
        "admission_record_id",
        "review_id",
        "document_id",
        "field_name",
        "planned_action",
        "planned_record_status",
        "training_admitted",
        "training_dataset_materialized",
        "dataset_artifact_created",
        "phase1_status",
        "dataset_confirmation_changed",
        "target_model_families",
        "planned_output_formats",
        "source_artifact_sha256",
        "review_artifact_sha256",
        "admission_request_sha256",
        "package_validation_sha256",
        "materialization_plan_sha256",
        "quarantine_candidate_records_sha256",
        "training_admission_readiness_sha256",
        "training_admission_request_plan_sha256",
        "training_admission_request_preflight_sha256",
        "training_admission_request_draft_sha256",
        "training_admission_request_draft_precheck_sha256",
        "training_admission_execution_request_sha256",
        "training_admission_execution_request_preflight_sha256",
        "training_admission_execution_dry_run_sha256",
        "training_admission_execution_dry_run_precheck_sha256",
        "training_admission_execution_ledger_sha256",
        "training_admission_execution_ledger_precheck_sha256",
    }
    serialized = json.dumps(record, sort_keys=True)
    assert "raw table" not in serialized.lower()
    assert "article text" not in serialized.lower()
    assert ".pdf" not in serialized.lower()
    assert ".csv" not in serialized.lower()
    assert ".jsonl" not in serialized.lower()
    assert ".parquet" not in serialized.lower()
    assert ".lmdb" not in serialized.lower()
    assert str(tmp_path) not in serialized


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_planner_package(tmp_path)
    run_dir = paths["training_dataset_materialization_output_dir"] / "property-training-dataset-materialization-plan-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = plan_property_training_dataset_materialization(
        **_kwargs(paths),
        confirm_training_dataset_materialization_plan=True,
    )

    assert summary["plan_status"] == "blocked"
    assert "output_directory_not_clean" in summary["planning_errors"]


def test_cli_stdout_valid_json_and_no_dataset_artifacts_created(tmp_path: Path) -> None:
    paths = _write_planner_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-dataset-materialization-plan"], stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["plan_status"] == "planned"
    assert stderr.getvalue() == ""
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_planner_package(tmp_path)
    _mutate_json(paths["training_execution_ledger"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-dataset-materialization-plan"], stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_unsafe_plan_or_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_planner_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_dataset_materialization_planner._contains_forbidden_material",
        lambda value: True,
    )

    summary = plan_property_training_dataset_materialization(
        **_kwargs(paths),
        confirm_training_dataset_materialization_plan=True,
    )
    run_dir = paths["training_dataset_materialization_output_dir"] / "property-training-dataset-materialization-plan-001"

    assert summary == {
        "schema_version": "custom_corpus_property_training_dataset_materialization_planner.v1",
        "plan_status": "blocked",
        "planning_errors": ["property_training_dataset_materialization_planner_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "property_training_dataset_materialization_plan.json").exists()
    assert not (run_dir / "redacted_property_training_dataset_materialization_plan_evidence.md").exists()


def test_forbidden_runners_are_not_imported_or_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_planner_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_property_training_admission_execution_ledger_precheck",
            "ai4s_agent.custom_corpus_property_training_admission_execution_ledger",
            "ai4s_agent.custom_corpus_property_training_admission_execution_dry_run_precheck",
            "ai4s_agent.custom_corpus_property_training_admission_execution_dry_run",
            "ai4s_agent.custom_corpus_property_training_admission_execution_request_preflight",
            "ai4s_agent.custom_corpus_property_training_admission_execution_request",
            "ai4s_agent.custom_corpus_property_training_admission_request_draft_precheck",
            "ai4s_agent.custom_corpus_property_training_admission_request_draft",
            "ai4s_agent.custom_corpus_property_training_admission_request_preflight",
            "ai4s_agent.custom_corpus_property_training_admission_request_planner",
            "ai4s_agent.custom_corpus_property_training_admission_readiness",
            "ai4s_agent.custom_corpus_property_quarantine_candidate_preflight",
            "ai4s_agent.custom_corpus_property_quarantine_materializer",
            "ai4s_agent.custom_corpus_materialization_planner",
            "ai4s_agent.workflows.corpus_to_phase1_workflow",
            "ai4s_agent.document_parse_service",
            "ai4s_agent.document_parse",
            "ai4s_agent.mineru",
            "openai",
            "pdfplumber",
        )
        if name.startswith(forbidden):
            raise AssertionError(f"forbidden import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracking_import)

    summary = plan_property_training_dataset_materialization(
        **_kwargs(paths),
        confirm_training_dataset_materialization_plan=True,
    )

    assert summary["plan_status"] == "planned"
    assert not any("custom_corpus_property_training_admission_execution_ledger" in name for name in imported_modules)
    assert not any("custom_corpus_property_training_admission_execution_dry_run" in name for name in imported_modules)
    assert not any("custom_corpus_property_quarantine_materializer" in name for name in imported_modules)


def _write_planner_package(tmp_path: Path, *, needs_review: bool = False) -> dict[str, Path]:
    paths = _write_ledger_precheck_package(tmp_path, needs_review=needs_review)
    precheck = precheck_property_training_admission_execution_ledger_package(
        **_ledger_precheck_kwargs(paths),
        allow_ledger_needs_review=needs_review,
    )
    assert precheck["precheck_status"] in {"passed", "needs_review"}
    paths["training_dataset_materialization_output_dir"] = tmp_path / "property-training-dataset-materialization-output"
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
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
        "output_dir": paths["training_dataset_materialization_output_dir"],
        "materialization_plan_id": "property-training-dataset-materialization-plan-001",
        "created_by": "operator-redacted",
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
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
        str(paths["training_dataset_materialization_output_dir"]),
        "--materialization-plan-id",
        "property-training-dataset-materialization-plan-001",
        "--created-by",
        "operator-redacted",
    ]
