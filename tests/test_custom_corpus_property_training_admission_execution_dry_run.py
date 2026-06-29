from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_admission_execution_dry_run import (
    dry_run_property_training_admission_execution,
    main,
)
from ai4s_agent.custom_corpus_property_training_admission_execution_request_preflight import (
    preflight_property_training_admission_execution_request_package,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_admission_execution_request_preflight import (
    _kwargs as _preflight_kwargs,
)
from test_custom_corpus_property_training_admission_execution_request_preflight import (
    _write_preflight_package,
)


def test_valid_full_package_writes_dry_run_report_and_markdown(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)

    report = dry_run_property_training_admission_execution(
        **_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
    )

    run_dir = paths["training_execution_dry_run_output_dir"] / "property-training-admission-execution-dry-run-001"
    report_path = run_dir / "property_training_admission_execution_dry_run_report.json"
    markdown_path = run_dir / "redacted_property_training_admission_execution_dry_run_evidence.md"
    written = json.loads(report_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert written == report
    assert report["schema_version"] == "custom_corpus_property_training_admission_execution_dry_run.v1"
    assert report["dry_run_status"] == "passed"
    assert report["dry_run_mode"] == "execution_simulation_only"
    assert report["training_admitted"] is False
    assert report["phase1_status"] == "not_run"
    assert report["dataset_confirmation_changed"] is False
    assert report["execution_request_preflight_status"] == "passed"
    assert report["execution_request_status"] == "written"
    assert report["draft_precheck_status"] == "passed"
    assert report["dry_run_record_count"] == 1
    assert report["execution_record_count"] == 1
    assert report["planned_candidate_count"] == 1
    assert report["dry_run_record_ids"][0].startswith("property-training-admission-execution-dry-run-001-")
    assert report["dry_run_records"][0]["would_execute_action"] == "would_admit_training_candidate"
    assert report["dry_run_records"][0]["dry_run_record_status"] == "would_admit"
    assert report["dry_run_records"][0]["training_admitted"] is False
    assert report["dry_run_errors"] == []
    assert report["warnings"] == []
    assert report["redaction_status"] == "passed"
    assert report["training_admission_execution_request_path"] == paths["training_execution_request"].name
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)
    assert "this is a training admission execution dry-run only" in markdown
    assert "no training admission was executed" in markdown
    assert "no training data was admitted" in markdown
    assert "no training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert "no candidate CSV/JSONL/Parquet/LMDB was created" in markdown
    assert "DatasetConfirmation was not changed" in markdown
    assert "no model training or evaluation was run" in markdown


def test_missing_confirmation_writes_no_report(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    stdout = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=io.StringIO())
    run_dir = paths["training_execution_dry_run_output_dir"] / "property-training-admission-execution-dry-run-001"

    assert code == 1
    assert json.loads(stdout.getvalue())["dry_run_status"] == "blocked"
    assert not (run_dir / "property_training_admission_execution_dry_run_report.json").exists()


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_execution_request_preflight_summary", lambda payload: payload.__setitem__("preflight_status", "blocked"), "training_admission_execution_request_preflight_blocked"),
        ("training_execution_request", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_admission_execution_request_schema_invalid"),
        ("training_execution_request_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_admission_execution_request_summary_schema_invalid"),
        ("training_execution_request", lambda payload: payload.__setitem__("request_status", "blocked"), "training_admission_execution_request_blocked"),
        ("training_execution_request_summary", lambda payload: payload.__setitem__("training_admission_execution_request_sha256", "sha256:" + "0" * 64), "training_admission_execution_request_sha256_mismatch"),
        ("training_execution_request_preflight_summary", lambda payload: payload.__setitem__("training_admission_request_draft_sha256", "sha256:" + "0" * 64), "training_admission_request_draft_sha256_mismatch"),
        ("training_request_draft_precheck_summary", lambda payload: payload.__setitem__("training_admission_request_plan_sha256", "sha256:" + "0" * 64), "training_admission_request_plan_sha256_mismatch"),
        ("training_request_preflight_summary", lambda payload: payload.__setitem__("training_admission_readiness_summary_sha256", "sha256:" + "0" * 64), "training_admission_readiness_summary_sha256_mismatch"),
        ("training_request_plan_summary", lambda payload: payload.__setitem__("quarantine_candidate_preflight_summary_sha256", "sha256:" + "0" * 64), "quarantine_candidate_preflight_summary_sha256_mismatch"),
        ("training_execution_request", lambda payload: payload.__setitem__("corpus_id", "other-corpus"), "corpus_id_mismatch"),
        ("training_execution_request", lambda payload: payload.__setitem__("training_admitted", True), "training_admitted"),
        ("training_execution_request", lambda payload: payload.__setitem__("phase1_status", "success"), "phase1_ran"),
        ("training_execution_request", lambda payload: payload.__setitem__("dataset_confirmation_changed", True), "dataset_confirmation_changed"),
    ],
)
def test_blocking_input_failures(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_dry_run_package(tmp_path)
    _mutate_json(paths[target], mutator)

    report = dry_run_property_training_admission_execution(
        **_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
    )

    assert report["dry_run_status"] == "blocked"
    assert error_code in report["dry_run_errors"]


def test_execution_preflight_needs_review_blocks_by_default_and_can_write_needs_review(tmp_path: Path) -> None:
    paths = _write_dry_run_package(
        tmp_path,
        package_binding_status="needs_review",
        allow_quarantine_needs_review=True,
        allow_preflight_partial=True,
        allow_draft_needs_review=True,
        allow_execution_request_needs_review=True,
        allow_execution_preflight_needs_review=True,
        dry_run_id="property-training-admission-execution-dry-run-002",
    )

    blocked = dry_run_property_training_admission_execution(
        **_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
    )
    allowed = dry_run_property_training_admission_execution(
        **_kwargs(paths, dry_run_id="property-training-admission-execution-dry-run-003"),
        confirm_training_admission_execution_dry_run=True,
        allow_execution_preflight_needs_review=True,
    )

    assert blocked["dry_run_status"] == "blocked"
    assert "training_admission_execution_request_preflight_needs_review" in blocked["dry_run_errors"]
    assert allowed["dry_run_status"] == "needs_review"
    assert "training_admission_execution_request_preflight_needs_review" in allowed["warnings"]


def test_record_consistency_failures(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    _mutate_json(paths["training_execution_request"], lambda payload: payload.__setitem__("execution_records", []))
    no_records = dry_run_property_training_admission_execution(
        **_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
    )

    paths = _write_dry_run_package(tmp_path / "no-planned")
    _mutate_json(paths["training_execution_request"], lambda payload: payload.__setitem__("planned_training_admission_candidate_record_ids", []))
    no_planned = dry_run_property_training_admission_execution(
        **_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
    )

    paths = _write_dry_run_package(tmp_path / "count")
    _mutate_json(paths["training_execution_request"], lambda payload: payload.__setitem__("execution_record_count", 2))
    count_mismatch = dry_run_property_training_admission_execution(
        **_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
    )

    paths = _write_dry_run_package(tmp_path / "ids")
    _mutate_json(paths["training_execution_request"], lambda payload: payload.__setitem__("execution_record_ids", ["wrong-id"]))
    id_mismatch = dry_run_property_training_admission_execution(
        **_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
    )

    paths = _write_dry_run_package(tmp_path / "candidate")
    _mutate_json(paths["training_execution_request"], lambda payload: payload.__setitem__("planned_training_admission_candidate_record_ids", ["unknown-candidate"]))
    candidate_mismatch = dry_run_property_training_admission_execution(
        **_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
    )

    assert "no_execution_records" in no_records["dry_run_errors"]
    assert "no_planned_candidates" in no_planned["dry_run_errors"]
    assert "execution_record_count_mismatch" in count_mismatch["dry_run_errors"]
    assert "execution_record_ids_mismatch" in id_mismatch["dry_run_errors"]
    assert "planned_candidate_not_in_execution_request" in candidate_mismatch["dry_run_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("exclude_record_ids", ["property-candidate-001"], "planned_candidate_from_excluded_record"),
        ("blocked_from_training_admission_record_ids", ["property-candidate-001"], "planned_candidate_from_blocked_record"),
        ("needs_review_record_ids", ["property-candidate-001"], "planned_candidate_from_needs_review_record"),
    ],
)
def test_planned_candidate_from_disallowed_source_blocks(tmp_path: Path, field: str, value: list[str], error_code: str) -> None:
    paths = _write_dry_run_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, value))

    report = dry_run_property_training_admission_execution(
        **_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
    )

    assert report["dry_run_status"] == "blocked"
    assert error_code in report["dry_run_errors"]


def test_dry_run_records_are_safe_id_hash_only(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)

    report = dry_run_property_training_admission_execution(
        **_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
    )

    record = report["dry_run_records"][0]
    assert set(record) == {
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
        "would_execute_action",
        "dry_run_record_status",
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
    }
    serialized = json.dumps(report, sort_keys=True)
    assert "raw table" not in serialized.lower()
    assert "article text" not in serialized.lower()
    assert ".pdf" not in serialized.lower()
    assert ".csv" not in serialized.lower()
    assert ".jsonl" not in serialized.lower()
    assert ".parquet" not in serialized.lower()
    assert ".lmdb" not in serialized.lower()
    assert str(tmp_path) not in serialized


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    run_dir = paths["training_execution_dry_run_output_dir"] / "property-training-admission-execution-dry-run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    report = dry_run_property_training_admission_execution(
        **_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
    )

    assert report["dry_run_status"] == "blocked"
    assert "output_directory_not_clean" in report["dry_run_errors"]


def test_summary_uses_safe_basenames_only(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)

    report = dry_run_property_training_admission_execution(
        **_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
    )

    serialized = json.dumps(report, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert report["training_admission_execution_request_path"] == paths["training_execution_request"].name
    assert report["training_admission_execution_request_preflight_path"] == paths["training_execution_request_preflight_summary"].name


def test_redaction_fail_closed_writes_no_report_or_unsafe_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_dry_run_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_admission_execution_dry_run._contains_forbidden_material",
        lambda value: True,
    )

    report = dry_run_property_training_admission_execution(
        **_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
    )
    run_dir = paths["training_execution_dry_run_output_dir"] / "property-training-admission-execution-dry-run-001"

    assert report == {
        "schema_version": "custom_corpus_property_training_admission_execution_dry_run.v1",
        "dry_run_status": "blocked",
        "dry_run_errors": ["property_training_admission_execution_dry_run_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "property_training_admission_execution_dry_run_report.json").exists()
    assert not (run_dir / "redacted_property_training_admission_execution_dry_run_evidence.md").exists()


def test_cli_stdout_valid_json_and_no_training_artifacts_created(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-admission-execution-dry-run"], stdout=stdout, stderr=stderr)
    report = json.loads(stdout.getvalue())

    assert code == 0
    assert report["dry_run_status"] == "passed"
    assert stderr.getvalue() == ""
    assert not (tmp_path / "training_admission_execution.json").exists()
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    _mutate_json(paths["training_execution_request"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-admission-execution-dry-run"], stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_forbidden_runners_are_not_imported_or_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_property_training_admission_execution_request",
            "ai4s_agent.custom_corpus_property_training_admission_execution_request_preflight",
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

    report = dry_run_property_training_admission_execution(
        **_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
    )

    assert report["dry_run_status"] == "passed"
    assert not any("custom_corpus_property_training_admission_execution_request" in name for name in imported_modules)
    assert not any("custom_corpus_property_training_admission_request_planner" in name for name in imported_modules)
    assert not any("custom_corpus_property_quarantine_materializer" in name for name in imported_modules)


def _write_dry_run_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
    allow_quarantine_needs_review: bool = False,
    allow_preflight_partial: bool = False,
    allow_draft_needs_review: bool = False,
    allow_execution_request_needs_review: bool = False,
    allow_execution_preflight_needs_review: bool = False,
    dry_run_id: str = "property-training-admission-execution-dry-run-001",
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    execution_request_id = dry_run_id.replace("execution-dry-run", "execution-request")
    paths = _write_preflight_package(
        tmp_path,
        package_binding_status=package_binding_status,
        allow_quarantine_needs_review=allow_quarantine_needs_review,
        allow_preflight_partial=allow_preflight_partial,
        allow_draft_needs_review=allow_draft_needs_review,
        allow_execution_request_needs_review=allow_execution_request_needs_review,
        execution_request_id=execution_request_id,
    )
    preflight = preflight_property_training_admission_execution_request_package(
        **_preflight_kwargs(paths),
        output_summary_path=paths["training_execution_request_preflight_summary"],
        output_markdown_path=paths["training_execution_request_preflight_markdown"],
        allow_execution_request_needs_review=allow_execution_preflight_needs_review,
    )
    assert preflight["preflight_status"] in {"passed", "needs_review"}
    paths["training_execution_dry_run_output_dir"] = tmp_path / "property-training-admission-execution-dry-run-output"
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
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
        "output_dir": paths["training_execution_dry_run_output_dir"],
        "dry_run_id": "property-training-admission-execution-dry-run-001",
        "created_by": "operator-redacted",
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
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
        str(paths["training_execution_dry_run_output_dir"]),
        "--dry-run-id",
        "property-training-admission-execution-dry-run-001",
        "--created-by",
        "operator-redacted",
    ]
