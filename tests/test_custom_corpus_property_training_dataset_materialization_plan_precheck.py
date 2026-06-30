from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_dataset_materialization_plan_precheck import (
    main,
    precheck_property_training_dataset_materialization_plan,
)
from ai4s_agent.custom_corpus_property_training_dataset_materialization_planner import (
    plan_property_training_dataset_materialization,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_dataset_materialization_planner import (
    _kwargs as _planner_kwargs,
)
from test_custom_corpus_property_training_dataset_materialization_planner import (
    _write_planner_package,
)


def test_valid_full_package_returns_passed_summary_and_markdown(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    output_summary = tmp_path / "property-training-dataset-materialization-plan-precheck-summary.json"
    output_markdown = tmp_path / "property-training-dataset-materialization-plan-precheck-summary.md"

    summary = precheck_property_training_dataset_materialization_plan(
        **_kwargs(paths),
        output_summary_path=output_summary,
        output_markdown_path=output_markdown,
    )
    written = json.loads(output_summary.read_text(encoding="utf-8"))
    markdown = output_markdown.read_text(encoding="utf-8")
    serialized = json.dumps(summary, sort_keys=True)

    assert written == summary
    assert summary["schema_version"] == "custom_corpus_property_training_dataset_materialization_plan_precheck.v1"
    assert summary["precheck_status"] == "passed"
    assert summary["plan_status"] == "planned"
    assert summary["ledger_precheck_status"] == "passed"
    assert summary["training_admitted"] is True
    assert summary["training_dataset_materialized"] is False
    assert summary["dataset_artifact_created"] is False
    assert summary["phase1_status"] == "not_run"
    assert summary["dataset_confirmation_changed"] is False
    assert summary["planned_dataset_record_count"] == 1
    assert summary["planned_candidate_count"] == 1
    assert summary["precheck_errors"] == []
    assert summary["warnings"] == []
    assert summary["training_dataset_materialization_plan_path"] == paths["training_dataset_materialization_plan"].name
    assert str(tmp_path) not in serialized
    assert "this is a training dataset materialization plan precheck only" in markdown
    assert "no training dataset artifact was created" in markdown
    assert "no training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert "DatasetConfirmation was not changed" in markdown
    assert str(tmp_path) not in markdown


def test_plan_needs_review_blocks_by_default_and_can_return_needs_review(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path, needs_review=True)

    blocked = precheck_property_training_dataset_materialization_plan(**_kwargs(paths))
    allowed = precheck_property_training_dataset_materialization_plan(
        **_kwargs(paths),
        allow_plan_needs_review=True,
    )

    assert blocked["precheck_status"] == "blocked"
    assert "training_dataset_materialization_plan_needs_review" in blocked["precheck_errors"]
    assert allowed["precheck_status"] == "needs_review"
    assert "training_dataset_materialization_plan_needs_review" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_dataset_materialization_plan", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_materialization_plan_schema_invalid"),
        ("training_dataset_materialization_plan", lambda payload: payload.__setitem__("plan_status", "blocked"), "training_dataset_materialization_plan_blocked"),
        ("training_dataset_materialization_planner_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_materialization_planner_summary_schema_invalid"),
        ("training_execution_ledger_precheck_summary", lambda payload: payload.__setitem__("precheck_status", "blocked"), "training_admission_execution_ledger_precheck_blocked"),
        ("training_execution_ledger", lambda payload: payload.__setitem__("execution_status", "blocked"), "training_admission_execution_ledger_blocked"),
        ("training_execution_ledger_summary", lambda payload: payload.__setitem__("execution_status", "blocked"), "training_admission_execution_ledger_summary_blocked"),
        ("training_execution_dry_run_precheck_summary", lambda payload: payload.__setitem__("preflight_status", "blocked"), "training_admission_execution_dry_run_precheck_blocked"),
        ("training_execution_dry_run_report", lambda payload: payload.__setitem__("dry_run_status", "blocked"), "training_admission_execution_dry_run_blocked"),
        ("training_execution_request", lambda payload: payload.__setitem__("request_status", "blocked"), "training_admission_execution_request_blocked"),
        ("training_execution_request_preflight_summary", lambda payload: payload.__setitem__("preflight_status", "blocked"), "training_admission_execution_request_preflight_blocked"),
        ("training_request_draft_precheck_summary", lambda payload: payload.__setitem__("precheck_status", "blocked"), "training_admission_request_draft_precheck_blocked"),
        ("training_request_plan_summary", lambda payload: payload.__setitem__("planner_status", "blocked"), "training_admission_request_plan_blocked"),
        ("training_admission_readiness_summary", lambda payload: payload.__setitem__("readiness_status", "blocked"), "training_admission_readiness_blocked"),
    ],
)
def test_blocking_input_failures(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = precheck_property_training_dataset_materialization_plan(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("target", "field", "error_code"),
    [
        ("training_dataset_materialization_planner_summary", "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256_mismatch"),
        ("training_dataset_materialization_plan", "training_admission_execution_ledger_precheck_sha256", "training_admission_execution_ledger_precheck_sha256_mismatch"),
        ("training_dataset_materialization_plan", "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256_mismatch"),
        ("training_dataset_materialization_plan", "training_admission_execution_dry_run_precheck_sha256", "training_admission_execution_dry_run_precheck_sha256_mismatch"),
        ("training_dataset_materialization_plan", "training_admission_execution_request_sha256", "training_admission_execution_request_sha256_mismatch"),
        ("training_dataset_materialization_plan", "training_admission_request_plan_sha256", "training_admission_request_plan_sha256_mismatch"),
        ("training_dataset_materialization_plan", "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256_mismatch"),
        ("training_dataset_materialization_plan", "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256_mismatch"),
    ],
)
def test_sha_mismatches_block(tmp_path: Path, target: str, field: str, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = precheck_property_training_dataset_materialization_plan(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("target", "field", "value", "error_code"),
    [
        ("training_dataset_materialization_plan", "corpus_id", "other-corpus", "corpus_id_mismatch"),
        ("training_dataset_materialization_plan", "source_dry_run_id", "other-dry-run", "source_dry_run_id_mismatch"),
        ("training_dataset_materialization_plan", "review_manifest_id", "other-review-manifest", "review_manifest_id_mismatch"),
        ("training_dataset_materialization_plan", "admission_request_id", "other-admission", "admission_request_id_mismatch"),
        ("training_dataset_materialization_plan", "source_execution_request_id", "other-execution", "source_execution_request_id_mismatch"),
        ("training_dataset_materialization_plan", "quarantine_run_id", "other-quarantine", "quarantine_run_id_mismatch"),
        ("training_dataset_materialization_plan", "property_candidate_manifest_id", "other-candidates", "property_candidate_manifest_id_mismatch"),
    ],
)
def test_id_mismatches_block(tmp_path: Path, target: str, field: str, value: str, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, value))

    summary = precheck_property_training_dataset_materialization_plan(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("training_admitted", False, "training_not_admitted"),
        ("training_dataset_materialized", True, "training_dataset_materialized"),
        ("dataset_artifact_created", True, "dataset_artifact_created"),
        ("phase1_status", "success", "phase1_ran"),
        ("dataset_confirmation_changed", True, "dataset_confirmation_changed"),
        ("plan_mode", "training_dataset_writer", "training_dataset_materialization_plan_mode_invalid"),
        ("planning_errors", ["bad"], "training_dataset_materialization_plan_has_errors"),
        ("dataset_name", "unsafe/name", "dataset_name_invalid"),
        ("target_model_families", ["unimol", "train_now"], "target_model_family_invalid"),
        ("planned_output_formats", ["jsonl", "rows.csv"], "planned_output_format_invalid"),
        ("planned_dataset_records", [], "no_planned_dataset_records"),
        ("ledger_record_ids", [], "no_ledger_records"),
        ("planned_training_admission_candidate_record_ids", [], "no_planned_candidates"),
        ("planned_dataset_record_count", 2, "planned_dataset_record_count_mismatch"),
        ("planned_dataset_record_ids", ["different-record"], "planned_dataset_record_ids_mismatch"),
        ("ledger_record_ids", ["different-ledger-record"], "ledger_record_ids_mismatch"),
    ],
)
def test_plan_field_failures_block(tmp_path: Path, field: str, value: object, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["training_dataset_materialization_plan"], lambda payload: payload.__setitem__(field, value))

    summary = precheck_property_training_dataset_materialization_plan(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("planned_action", "write_row", "planned_dataset_record_action_invalid"),
        ("planned_record_status", "written", "planned_dataset_record_status_invalid"),
        ("training_admitted", False, "planned_dataset_record_not_training_admitted"),
        ("training_dataset_materialized", True, "training_dataset_materialized"),
        ("dataset_artifact_created", True, "dataset_artifact_created"),
        ("phase1_status", "success", "phase1_ran"),
        ("dataset_confirmation_changed", True, "dataset_confirmation_changed"),
        ("candidate_record_id", "unknown-candidate", "planned_candidate_ids_mismatch"),
        ("ledger_record_id", "unknown-ledger-record", "planned_dataset_record_ledger_id_unknown"),
    ],
)
def test_planned_dataset_record_failures_block(tmp_path: Path, field: str, value: object, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        records = payload["planned_dataset_records"]
        assert isinstance(records, list)
        records[0][field] = value

    _mutate_json(paths["training_dataset_materialization_plan"], mutate)

    summary = precheck_property_training_dataset_materialization_plan(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


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
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, value))

    summary = precheck_property_training_dataset_materialization_plan(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


def test_output_format_labels_must_not_be_paths(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(
        paths["training_dataset_materialization_plan"],
        lambda payload: payload.__setitem__("planned_output_formats", ["jsonl", "tmp/output.csv"]),
    )

    summary = precheck_property_training_dataset_materialization_plan(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert "planned_output_format_invalid" in summary["precheck_errors"]


def test_summary_and_markdown_use_safe_basenames_only(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    output_summary = tmp_path / "summary.json"
    output_markdown = tmp_path / "summary.md"

    summary = precheck_property_training_dataset_materialization_plan(
        **_kwargs(paths),
        output_summary_path=output_summary,
        output_markdown_path=output_markdown,
    )

    assert summary["precheck_status"] == "passed"
    serialized = json.dumps(summary, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert str(tmp_path) not in output_markdown.read_text(encoding="utf-8")


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["training_dataset_materialization_plan"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_minimal_summary_and_no_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_precheck_package(tmp_path)
    output_summary = tmp_path / "summary.json"
    output_markdown = tmp_path / "summary.md"
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_dataset_materialization_plan_precheck._contains_forbidden_material",
        lambda value: True,
    )

    summary = precheck_property_training_dataset_materialization_plan(
        **_kwargs(paths),
        output_summary_path=output_summary,
        output_markdown_path=output_markdown,
    )

    assert summary == {
        "schema_version": "custom_corpus_property_training_dataset_materialization_plan_precheck.v1",
        "precheck_status": "blocked",
        "precheck_errors": ["property_training_dataset_materialization_plan_precheck_redaction_failed"],
        "redaction_status": "failed",
    }
    assert json.loads(output_summary.read_text(encoding="utf-8")) == summary
    assert not output_markdown.exists()


def test_cli_stdout_valid_json_and_no_dataset_artifacts_created(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["precheck_status"] == "passed"
    assert stderr.getvalue() == ""
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_forbidden_runners_are_not_imported_or_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_planner",
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

    summary = precheck_property_training_dataset_materialization_plan(**_kwargs(paths))

    assert summary["precheck_status"] == "passed"
    assert not any("custom_corpus_property_training_dataset_materialization_planner" in name for name in imported_modules)
    assert not any("custom_corpus_property_training_admission_execution_ledger" in name for name in imported_modules)


def _write_precheck_package(tmp_path: Path, *, needs_review: bool = False) -> dict[str, Path]:
    paths = _write_planner_package(tmp_path, needs_review=needs_review)
    summary = plan_property_training_dataset_materialization(
        **_planner_kwargs(paths),
        confirm_training_dataset_materialization_plan=True,
        allow_ledger_precheck_needs_review=needs_review,
    )
    assert summary["plan_status"] in {"planned", "needs_review"}
    run_dir = paths["training_dataset_materialization_output_dir"] / "property-training-dataset-materialization-plan-001"
    paths["training_dataset_materialization_plan"] = run_dir / "property_training_dataset_materialization_plan.json"
    paths["training_dataset_materialization_planner_summary"] = (
        run_dir / "property_training_dataset_materialization_planner_summary.json"
    )
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
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
