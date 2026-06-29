from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_admission_execution_dry_run_precheck import (
    preflight_property_training_admission_execution_dry_run_package,
)
from ai4s_agent.custom_corpus_property_training_admission_execution_ledger import (
    build_property_training_admission_execution_ledger,
    main,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_admission_execution_dry_run_precheck import (
    _kwargs as _dry_run_precheck_kwargs,
)
from test_custom_corpus_property_training_admission_execution_dry_run_precheck import (
    _write_precheck_package,
)


def test_valid_full_package_writes_ledger_summary_and_markdown(tmp_path: Path) -> None:
    paths = _write_ledger_package(tmp_path)

    summary = build_property_training_admission_execution_ledger(
        **_kwargs(paths),
        confirm_training_admission_ledger_write=True,
    )

    run_dir = paths["training_execution_ledger_output_dir"] / "property-training-admission-execution-ledger-001"
    ledger_path = run_dir / "property_training_admission_execution_ledger.json"
    summary_path = run_dir / "property_training_admission_execution_ledger_summary.json"
    markdown_path = run_dir / "redacted_property_training_admission_execution_ledger_evidence.md"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert written_summary == summary
    assert ledger["schema_version"] == "custom_corpus_property_training_admission_execution_ledger.v1"
    assert summary["schema_version"] == "custom_corpus_property_training_admission_execution_ledger_summary.v1"
    assert ledger["execution_status"] == "committed"
    assert summary["execution_status"] == "committed"
    assert ledger["execution_mode"] == "training_admission_ledger_only"
    assert ledger["training_admitted"] is True
    assert ledger["phase1_status"] == "not_run"
    assert ledger["dataset_confirmation_changed"] is False
    assert ledger["training_dataset_materialized"] is False
    assert ledger["dataset_artifact_created"] is False
    assert ledger["ledger_record_count"] == 1
    assert summary["ledger_record_count"] == 1
    assert ledger["execution_errors"] == []
    assert summary["execution_errors"] == []
    assert ledger["warnings"] == []
    assert summary["warnings"] == []
    assert summary["training_admission_execution_ledger_path"] == ledger_path.name
    assert str(tmp_path) not in json.dumps(ledger, sort_keys=True)
    assert str(tmp_path) not in json.dumps(summary, sort_keys=True)
    assert "this is a training admission execution ledger only" in markdown
    assert "training candidates were admitted to the ledger" in markdown
    assert "no training dataset artifact was created" in markdown
    assert "no training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert "DatasetConfirmation was not changed" in markdown
    assert "no model training or evaluation was run" in markdown


def test_missing_confirmation_writes_no_ledger(tmp_path: Path) -> None:
    paths = _write_ledger_package(tmp_path)
    stdout = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=io.StringIO())
    run_dir = paths["training_execution_ledger_output_dir"] / "property-training-admission-execution-ledger-001"

    assert code == 1
    assert json.loads(stdout.getvalue())["execution_status"] == "blocked"
    assert not (run_dir / "property_training_admission_execution_ledger.json").exists()


def test_dry_run_precheck_needs_review_blocks_by_default_and_can_write_needs_review(tmp_path: Path) -> None:
    paths = _write_ledger_package(
        tmp_path,
        package_binding_status="needs_review",
        allow_quarantine_needs_review=True,
        allow_preflight_partial=True,
        allow_draft_needs_review=True,
        allow_execution_request_needs_review=True,
        allow_execution_preflight_needs_review=True,
        allow_dry_run_needs_review=True,
        allow_dry_run_precheck_needs_review=True,
    )

    blocked = build_property_training_admission_execution_ledger(
        **_kwargs(paths),
        confirm_training_admission_ledger_write=True,
    )
    allowed = build_property_training_admission_execution_ledger(
        **_kwargs(paths, execution_ledger_id="property-training-admission-execution-ledger-002"),
        confirm_training_admission_ledger_write=True,
        allow_dry_run_precheck_needs_review=True,
    )

    assert blocked["execution_status"] == "blocked"
    assert "training_admission_execution_dry_run_precheck_needs_review" in blocked["execution_errors"]
    assert allowed["execution_status"] == "needs_review"
    assert "training_admission_execution_dry_run_precheck_needs_review" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_execution_dry_run_precheck_summary", lambda payload: payload.__setitem__("preflight_status", "blocked"), "training_admission_execution_dry_run_precheck_blocked"),
        ("training_execution_dry_run_precheck_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_admission_execution_dry_run_precheck_schema_invalid"),
        ("training_execution_dry_run_report", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_admission_execution_dry_run_schema_invalid"),
        ("training_execution_dry_run_report", lambda payload: payload.__setitem__("dry_run_status", "blocked"), "training_admission_execution_dry_run_blocked"),
        ("training_execution_dry_run_report", lambda payload: payload.__setitem__("training_admitted", True), "training_admitted_before_ledger"),
        ("training_execution_dry_run_report", lambda payload: payload.__setitem__("phase1_status", "success"), "phase1_ran"),
        ("training_execution_dry_run_report", lambda payload: payload.__setitem__("dataset_confirmation_changed", True), "dataset_confirmation_changed"),
        ("training_execution_request_preflight_summary", lambda payload: payload.__setitem__("preflight_status", "blocked"), "training_admission_execution_request_preflight_blocked"),
        ("training_execution_request", lambda payload: payload.__setitem__("request_status", "blocked"), "training_admission_execution_request_blocked"),
        ("training_request_draft_precheck_summary", lambda payload: payload.__setitem__("precheck_status", "blocked"), "training_admission_request_draft_precheck_blocked"),
        ("training_request_plan_summary", lambda payload: payload.__setitem__("planner_status", "blocked"), "training_admission_request_plan_blocked"),
        ("training_admission_readiness_summary", lambda payload: payload.__setitem__("readiness_status", "blocked"), "training_admission_readiness_blocked"),
    ],
)
def test_blocking_input_failures(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_ledger_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = build_property_training_admission_execution_ledger(
        **_kwargs(paths),
        confirm_training_admission_ledger_write=True,
    )

    assert summary["execution_status"] == "blocked"
    assert error_code in summary["execution_errors"]


def test_sha_and_id_mismatches_block(tmp_path: Path) -> None:
    paths = _write_ledger_package(tmp_path)
    _mutate_json(
        paths["training_execution_dry_run_precheck_summary"],
        lambda payload: payload.__setitem__("training_admission_execution_dry_run_report_sha256", "sha256:" + "0" * 64),
    )
    sha_summary = build_property_training_admission_execution_ledger(
        **_kwargs(paths),
        confirm_training_admission_ledger_write=True,
    )

    paths = _write_ledger_package(tmp_path / "id")
    _mutate_json(paths["training_execution_dry_run_report"], lambda payload: payload.__setitem__("corpus_id", "other-corpus"))
    id_summary = build_property_training_admission_execution_ledger(
        **_kwargs(paths),
        confirm_training_admission_ledger_write=True,
    )

    assert sha_summary["execution_status"] == "blocked"
    assert "training_admission_execution_dry_run_report_sha256_mismatch" in sha_summary["execution_errors"]
    assert id_summary["execution_status"] == "blocked"
    assert "corpus_id_mismatch" in id_summary["execution_errors"]


def test_record_consistency_failures(tmp_path: Path) -> None:
    paths = _write_ledger_package(tmp_path)
    _mutate_json(paths["training_execution_dry_run_report"], lambda payload: payload.__setitem__("dry_run_records", []))
    no_dry_run_records = build_property_training_admission_execution_ledger(
        **_kwargs(paths),
        confirm_training_admission_ledger_write=True,
    )

    paths = _write_ledger_package(tmp_path / "execution-records")
    _mutate_json(paths["training_execution_request"], lambda payload: payload.__setitem__("execution_records", []))
    no_execution_records = build_property_training_admission_execution_ledger(
        **_kwargs(paths),
        confirm_training_admission_ledger_write=True,
    )

    paths = _write_ledger_package(tmp_path / "planned")
    _mutate_json(
        paths["training_execution_dry_run_report"],
        lambda payload: payload.__setitem__("planned_training_admission_candidate_record_ids", []),
    )
    no_planned = build_property_training_admission_execution_ledger(
        **_kwargs(paths),
        confirm_training_admission_ledger_write=True,
    )

    paths = _write_ledger_package(tmp_path / "count")
    _mutate_json(paths["training_execution_dry_run_report"], lambda payload: payload.__setitem__("dry_run_record_count", 2))
    count_mismatch = build_property_training_admission_execution_ledger(
        **_kwargs(paths),
        confirm_training_admission_ledger_write=True,
    )

    assert "no_dry_run_records" in no_dry_run_records["execution_errors"]
    assert "no_execution_records" in no_execution_records["execution_errors"]
    assert "no_planned_candidates" in no_planned["execution_errors"]
    assert "dry_run_record_count_mismatch" in count_mismatch["execution_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("exclude_record_ids", ["property-candidate-001"], "planned_candidate_from_excluded_record"),
        ("blocked_from_training_admission_record_ids", ["property-candidate-001"], "planned_candidate_from_blocked_record"),
        ("needs_review_record_ids", ["property-candidate-001"], "planned_candidate_from_needs_review_record"),
    ],
)
def test_disallowed_candidate_leakage_blocks(tmp_path: Path, field: str, value: list[str], error_code: str) -> None:
    paths = _write_ledger_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, value))

    summary = build_property_training_admission_execution_ledger(
        **_kwargs(paths),
        confirm_training_admission_ledger_write=True,
    )

    assert summary["execution_status"] == "blocked"
    assert error_code in summary["execution_errors"]


def test_ledger_records_are_safe_id_hash_only_and_set_ledger_boundary(tmp_path: Path) -> None:
    paths = _write_ledger_package(tmp_path)

    summary = build_property_training_admission_execution_ledger(
        **_kwargs(paths),
        confirm_training_admission_ledger_write=True,
    )
    run_dir = paths["training_execution_ledger_output_dir"] / "property-training-admission-execution-ledger-001"
    ledger = json.loads((run_dir / "property_training_admission_execution_ledger.json").read_text(encoding="utf-8"))
    record = ledger["ledger_records"][0]

    assert summary["execution_status"] == "committed"
    assert record["training_admitted"] is True
    assert record["phase1_status"] == "not_run"
    assert record["dataset_confirmation_changed"] is False
    assert record["admission_action"] == "admit_training_candidate"
    assert record["ledger_record_status"] == "admitted_to_training_ledger"
    assert set(record) == {
        "ledger_record_id",
        "dry_run_record_id",
        "execution_record_id",
        "draft_record_id",
        "candidate_record_id",
        "record_id",
        "materialization_record_id",
        "execution_record_id_from_materializer",
        "admission_record_id",
        "review_id",
        "document_id",
        "field_name",
        "admission_action",
        "ledger_record_status",
        "training_admitted",
        "phase1_status",
        "dataset_confirmation_changed",
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
    }
    serialized = json.dumps(ledger, sort_keys=True)
    assert "raw table" not in serialized.lower()
    assert "article text" not in serialized.lower()
    assert ".pdf" not in serialized.lower()
    assert ".csv" not in serialized.lower()
    assert ".jsonl" not in serialized.lower()
    assert ".parquet" not in serialized.lower()
    assert ".lmdb" not in serialized.lower()
    assert str(tmp_path) not in serialized


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_ledger_package(tmp_path)
    run_dir = paths["training_execution_ledger_output_dir"] / "property-training-admission-execution-ledger-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = build_property_training_admission_execution_ledger(
        **_kwargs(paths),
        confirm_training_admission_ledger_write=True,
    )

    assert summary["execution_status"] == "blocked"
    assert "output_directory_not_clean" in summary["execution_errors"]


def test_redaction_fail_closed_writes_no_unsafe_ledger_or_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_ledger_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_admission_execution_ledger._contains_forbidden_material",
        lambda value: True,
    )

    summary = build_property_training_admission_execution_ledger(
        **_kwargs(paths),
        confirm_training_admission_ledger_write=True,
    )
    run_dir = paths["training_execution_ledger_output_dir"] / "property-training-admission-execution-ledger-001"

    assert summary == {
        "schema_version": "custom_corpus_property_training_admission_execution_ledger_summary.v1",
        "execution_status": "blocked",
        "execution_errors": ["property_training_admission_execution_ledger_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "property_training_admission_execution_ledger.json").exists()
    assert not (run_dir / "redacted_property_training_admission_execution_ledger_evidence.md").exists()


def test_cli_stdout_valid_json_and_no_training_artifacts_created(tmp_path: Path) -> None:
    paths = _write_ledger_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-admission-ledger-write"], stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["execution_status"] == "committed"
    assert stderr.getvalue() == ""
    assert not (tmp_path / "training.csv").exists()
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_ledger_package(tmp_path)
    _mutate_json(paths["training_execution_dry_run_precheck_summary"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-admission-ledger-write"], stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_forbidden_runners_are_not_imported_or_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_ledger_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
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

    summary = build_property_training_admission_execution_ledger(
        **_kwargs(paths),
        confirm_training_admission_ledger_write=True,
    )

    assert summary["execution_status"] == "committed"
    assert not any("custom_corpus_property_training_admission_execution_dry_run" in name for name in imported_modules)
    assert not any("custom_corpus_property_training_admission_request_planner" in name for name in imported_modules)
    assert not any("custom_corpus_property_quarantine_materializer" in name for name in imported_modules)


def _write_ledger_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
    allow_quarantine_needs_review: bool = False,
    allow_preflight_partial: bool = False,
    allow_draft_needs_review: bool = False,
    allow_execution_request_needs_review: bool = False,
    allow_execution_preflight_needs_review: bool = False,
    allow_dry_run_needs_review: bool = False,
    allow_dry_run_precheck_needs_review: bool = False,
) -> dict[str, Path]:
    paths = _write_precheck_package(
        tmp_path,
        package_binding_status=package_binding_status,
        allow_quarantine_needs_review=allow_quarantine_needs_review,
        allow_preflight_partial=allow_preflight_partial,
        allow_draft_needs_review=allow_draft_needs_review,
        allow_execution_request_needs_review=allow_execution_request_needs_review,
        allow_execution_preflight_needs_review=allow_execution_preflight_needs_review,
        allow_dry_run_needs_review=allow_dry_run_needs_review,
    )
    precheck = preflight_property_training_admission_execution_dry_run_package(
        **_dry_run_precheck_kwargs(paths),
        output_summary_path=paths["training_execution_dry_run_precheck_summary"],
        output_markdown_path=paths["training_execution_dry_run_precheck_markdown"],
        allow_dry_run_needs_review=allow_dry_run_precheck_needs_review,
    )
    assert precheck["preflight_status"] in {"passed", "needs_review"}
    paths["training_execution_ledger_output_dir"] = tmp_path / "property-training-admission-execution-ledger-output"
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
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
        "output_dir": paths["training_execution_ledger_output_dir"],
        "execution_ledger_id": "property-training-admission-execution-ledger-001",
        "created_by": "operator-redacted",
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
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
        str(paths["training_execution_ledger_output_dir"]),
        "--execution-ledger-id",
        "property-training-admission-execution-ledger-001",
        "--created-by",
        "operator-redacted",
    ]
