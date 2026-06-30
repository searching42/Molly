from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_admission_execution_ledger import (
    build_property_training_admission_execution_ledger,
)
from ai4s_agent.custom_corpus_property_training_admission_execution_ledger_precheck import (
    main,
    precheck_property_training_admission_execution_ledger_package,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_admission_execution_ledger import (
    _kwargs as _ledger_kwargs,
)
from test_custom_corpus_property_training_admission_execution_ledger import (
    _write_ledger_package,
)


def test_valid_full_pipeline_passes_and_writes_safe_evidence(tmp_path: Path) -> None:
    paths = _write_ledger_precheck_package(tmp_path)

    summary = precheck_property_training_admission_execution_ledger_package(**_kwargs(paths))

    markdown = paths["training_execution_ledger_precheck_markdown"].read_text(encoding="utf-8")
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["schema_version"] == "custom_corpus_property_training_admission_execution_ledger_precheck.v1"
    assert summary["precheck_status"] == "passed"
    assert summary["execution_ledger_status"] == "committed"
    assert summary["ledger_record_count"] == 1
    assert summary["planned_candidate_count"] == 1
    assert summary["training_admitted"] is True
    assert summary["phase1_status"] == "not_run"
    assert summary["dataset_confirmation_changed"] is False
    assert summary["training_dataset_materialized"] is False
    assert summary["dataset_artifact_created"] is False
    assert summary["precheck_errors"] == []
    assert summary["warnings"] == []
    assert summary["training_admission_execution_ledger_path"] == "property_training_admission_execution_ledger.json"
    assert str(tmp_path) not in serialized
    assert "this is a training admission execution ledger precheck only" in markdown
    assert "future training dataset materialization was not run" in markdown
    assert "no training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert "DatasetConfirmation was not changed" in markdown
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_ledger_needs_review_blocks_by_default_and_can_return_needs_review(tmp_path: Path) -> None:
    paths = _write_ledger_precheck_package(tmp_path, needs_review=True)

    blocked = precheck_property_training_admission_execution_ledger_package(**_kwargs(paths))
    allowed = precheck_property_training_admission_execution_ledger_package(
        **_kwargs(paths, allow_ledger_needs_review=True)
    )

    assert blocked["precheck_status"] == "blocked"
    assert "training_admission_execution_ledger_needs_review" in blocked["precheck_errors"]
    assert allowed["precheck_status"] == "needs_review"
    assert "training_admission_execution_ledger_needs_review" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_execution_ledger", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_admission_execution_ledger_schema_invalid"),
        ("training_execution_ledger_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_admission_execution_ledger_summary_schema_invalid"),
        ("training_execution_ledger", lambda payload: payload.__setitem__("execution_status", "blocked"), "training_admission_execution_ledger_blocked"),
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
    paths = _write_ledger_precheck_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = precheck_property_training_admission_execution_ledger_package(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("target", "field", "error_code"),
    [
        ("training_execution_ledger_summary", "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256_mismatch"),
        ("training_execution_ledger", "training_admission_execution_dry_run_report_sha256", "training_admission_execution_dry_run_report_sha256_mismatch"),
        ("training_execution_dry_run_precheck_summary", "training_admission_execution_dry_run_report_sha256", "training_admission_execution_dry_run_report_sha256_mismatch"),
        ("training_execution_request", "source_training_admission_request_draft_sha256", "training_admission_request_draft_sha256_mismatch"),
    ],
)
def test_sha_mismatches_block(tmp_path: Path, target: str, field: str, error_code: str) -> None:
    paths = _write_ledger_precheck_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = precheck_property_training_admission_execution_ledger_package(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


def test_id_mismatch_blocks(tmp_path: Path) -> None:
    paths = _write_ledger_precheck_package(tmp_path)
    _mutate_json(paths["training_execution_ledger"], lambda payload: payload.__setitem__("corpus_id", "other-corpus"))

    summary = precheck_property_training_admission_execution_ledger_package(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert "corpus_id_mismatch" in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("target", "field", "value", "error_code"),
    [
        ("training_execution_ledger", "ledger_records", [], "no_ledger_records"),
        ("training_execution_ledger", "ledger_record_count", 2, "ledger_record_count_mismatch"),
        ("training_execution_ledger", "planned_training_admission_candidate_record_ids", [], "no_planned_candidates"),
        ("training_execution_dry_run_report", "dry_run_records", [], "no_dry_run_records"),
        ("training_execution_request", "execution_records", [], "no_execution_records"),
    ],
)
def test_record_consistency_failures(tmp_path: Path, target: str, field: str, value: object, error_code: str) -> None:
    paths = _write_ledger_precheck_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, value))

    summary = precheck_property_training_admission_execution_ledger_package(**_kwargs(paths))

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
    paths = _write_ledger_precheck_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, value))

    summary = precheck_property_training_admission_execution_ledger_package(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


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
    paths = _write_ledger_precheck_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, value))

    summary = precheck_property_training_admission_execution_ledger_package(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


def test_cli_outputs_valid_json_and_markdown_boundary(tmp_path: Path) -> None:
    paths = _write_ledger_precheck_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())
    markdown = paths["training_execution_ledger_precheck_markdown"].read_text(encoding="utf-8")

    assert code == 0
    assert summary["precheck_status"] == "passed"
    assert stderr.getvalue() == ""
    assert "this is a training admission execution ledger precheck only" in markdown
    assert "no model training or evaluation was run" in markdown


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_ledger_precheck_package(tmp_path)
    _mutate_json(paths["training_execution_ledger"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_unsafe_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_ledger_precheck_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_admission_execution_ledger_precheck._contains_forbidden_material",
        lambda value: True,
    )

    summary = precheck_property_training_admission_execution_ledger_package(**_kwargs(paths))

    assert summary == {
        "schema_version": "custom_corpus_property_training_admission_execution_ledger_precheck.v1",
        "precheck_status": "blocked",
        "precheck_errors": ["property_training_admission_execution_ledger_precheck_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not paths["training_execution_ledger_precheck_markdown"].exists()


def test_forbidden_runners_are_not_imported_or_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_ledger_precheck_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
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

    summary = precheck_property_training_admission_execution_ledger_package(**_kwargs(paths))

    assert summary["precheck_status"] == "passed"
    assert not any("custom_corpus_property_training_admission_execution_ledger" in name for name in imported_modules)
    assert not any("custom_corpus_property_training_admission_execution_dry_run" in name for name in imported_modules)
    assert not any("custom_corpus_property_quarantine_materializer" in name for name in imported_modules)


def _write_ledger_precheck_package(tmp_path: Path, *, needs_review: bool = False) -> dict[str, Path]:
    paths = _write_ledger_package(
        tmp_path,
        package_binding_status="needs_review" if needs_review else "passed",
        allow_quarantine_needs_review=needs_review,
        allow_preflight_partial=needs_review,
        allow_draft_needs_review=needs_review,
        allow_execution_request_needs_review=needs_review,
        allow_execution_preflight_needs_review=needs_review,
        allow_dry_run_needs_review=needs_review,
        allow_dry_run_precheck_needs_review=needs_review,
    )
    ledger_summary = build_property_training_admission_execution_ledger(
        **_ledger_kwargs(paths),
        confirm_training_admission_ledger_write=True,
        allow_dry_run_precheck_needs_review=needs_review,
    )
    assert ledger_summary["execution_status"] in {"committed", "needs_review"}
    run_dir = paths["training_execution_ledger_output_dir"] / "property-training-admission-execution-ledger-001"
    paths["training_execution_ledger"] = run_dir / "property_training_admission_execution_ledger.json"
    paths["training_execution_ledger_summary"] = run_dir / "property_training_admission_execution_ledger_summary.json"
    paths["training_execution_ledger_precheck_summary"] = tmp_path / "property_training_admission_execution_ledger_precheck_summary.json"
    paths["training_execution_ledger_precheck_markdown"] = tmp_path / "property_training_admission_execution_ledger_precheck_summary.md"
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
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
        "output_summary_path": paths["training_execution_ledger_precheck_summary"],
        "output_markdown_path": paths["training_execution_ledger_precheck_markdown"],
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
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
        "--output-summary",
        str(paths["training_execution_ledger_precheck_summary"]),
        "--output-markdown",
        str(paths["training_execution_ledger_precheck_markdown"]),
    ]
