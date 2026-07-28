from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_materializer_execution_preflight import (
    main,
    preflight_property_materializer_execution_request,
)
from test_custom_corpus_property_materialization_plan_preflight import (
    _mutate_json,
    _sha256_file,
)
from test_custom_corpus_property_materializer_execution_request import (
    _kwargs as _request_kwargs,
)
from test_custom_corpus_property_materializer_execution_request import (
    _write_request_package,
)
from test_custom_corpus_property_materializer_execution_request import (
    build_property_materializer_execution_request,
)


def test_valid_full_package_returns_passed_and_writes_optional_outputs(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    summary_path = tmp_path / "execution_preflight_summary.json"
    markdown_path = tmp_path / "execution_preflight_summary.md"

    summary = preflight_property_materializer_execution_request(
        **_kwargs(paths),
        output_summary_path=summary_path,
        output_markdown_path=markdown_path,
    )

    written = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert written == summary
    assert summary["schema_version"] == "custom_corpus_property_materializer_execution_preflight.v1"
    assert summary["preflight_status"] == "passed"
    assert summary["execution_request_id"] == "property-materializer-execution-request-001"
    assert summary["materialization_plan_id"] == "property-materialization-plan-001"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["source_dry_run_id"] == "custom-dry-run-example-001"
    assert summary["review_manifest_id"] == "review-example-001"
    assert summary["admission_request_id"] == "property-admission-draft-001"
    assert summary["review_queue_id"] == "property-review-queue-001"
    assert summary["property_candidate_manifest_id"] == "property-candidates-001"
    assert summary["dataset_target"] == "example-candidate-target"
    assert summary["request_status"] == "written"
    assert summary["dry_run_status"] == "passed"
    assert summary["planner_status"] == "planned"
    assert summary["offline_planner_status"] == "planned"
    assert summary["preflight_input_status"] == "passed"
    assert summary["package_binding_status"] == "passed"
    assert summary["formal_package_validation_status"] == "passed"
    assert summary["materialization_decision"] == "planned"
    assert summary["materializer_status"] == "not_run"
    assert summary["phase1_status"] == "not_run"
    assert summary["training_admitted"] is False
    assert summary["dataset_confirmation_changed"] is False
    assert summary["admission_record_count"] == 2
    assert summary["admit_count"] == 1
    assert summary["exclude_count"] == 1
    assert summary["blocked_record_count"] == 1
    assert summary["materialization_record_count"] == 1
    assert summary["execution_record_count"] == 1
    assert summary["materialization_record_ids"] == ["property-materialization-plan-001-property-candidate-001"]
    assert summary["execution_record_ids"] == [
        "property-materializer-execution-request-001-property-materialization-plan-001-property-candidate-001"
    ]
    assert summary["admit_record_ids"] == ["property-candidate-001"]
    assert summary["exclude_record_ids"] == ["property-candidate-002"]
    assert summary["blocked_record_ids"] == ["property-candidate-003"]
    assert summary["preflight_errors"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"
    assert "this is a materializer execution request preflight only" in markdown
    assert "No real materializer was run" in markdown


def test_execution_request_records_are_safe_id_and_hash_only(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)

    summary = preflight_property_materializer_execution_request(**_kwargs(paths))

    request = json.loads(paths["execution_request"].read_text(encoding="utf-8"))
    record = request["execution_records"][0]
    assert summary["preflight_status"] == "passed"
    assert set(record) == {
        "execution_record_id",
        "materialization_record_id",
        "record_id",
        "admission_record_id",
        "review_id",
        "document_id",
        "field_name",
        "planned_action",
        "source_artifact_sha256",
        "review_artifact_sha256",
        "admission_request_sha256",
        "package_validation_sha256",
        "materialization_plan_sha256",
        "offline_planner_output_sha256",
        "dry_run_report_sha256",
    }
    serialized = json.dumps(record, sort_keys=True)
    assert "normalized_value_summary" not in serialized
    assert "provenance_summary" not in serialized
    assert "short provenance" not in serialized


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("execution_request", lambda payload: payload.__setitem__("schema_version", "wrong"), "execution_request_schema_invalid"),
        ("execution_request_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "execution_request_summary_schema_invalid"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("schema_version", "wrong"), "materialization_dry_run_report_schema_invalid"),
        ("property_planner_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "property_planner_schema_invalid"),
        ("offline_planner_output", lambda payload: payload.__setitem__("schema_version", "wrong"), "offline_planner_schema_invalid"),
        ("materialization_plan_preflight_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "preflight_schema_invalid"),
        ("property_package_binding_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "property_package_binding_schema_invalid"),
        ("formal_package_validation", lambda payload: payload.__setitem__("schema_version", "wrong"), "formal_package_validation_schema_invalid"),
        ("materialization_plan_draft", lambda payload: payload.__setitem__("schema_version", "wrong"), "materialization_plan_schema_invalid"),
    ],
)
def test_schema_mismatch_in_each_input_fails(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], mutator)
    if target == "execution_request":
        _refresh_execution_request_summary_sha(paths)

    summary = preflight_property_materializer_execution_request(**_kwargs(paths))

    assert summary["preflight_status"] == "failed"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("execution_request", lambda payload: payload.__setitem__("request_status", "blocked"), "execution_request_not_written"),
        ("execution_request_summary", lambda payload: payload.__setitem__("request_status", "blocked"), "execution_request_summary_not_written"),
        ("execution_request_summary", lambda payload: payload.__setitem__("request_errors", ["error"]), "execution_request_summary_has_errors"),
        ("execution_request", lambda payload: payload.__setitem__("execution_mode", "materialize"), "execution_mode_invalid"),
        ("execution_request", lambda payload: payload.__setitem__("materializer_status", "success"), "materializer_status_not_run"),
        ("execution_request", lambda payload: payload.__setitem__("phase1_status", "success"), "phase1_ran"),
        ("execution_request", lambda payload: payload.__setitem__("training_admitted", True), "training_admitted"),
        ("execution_request", lambda payload: payload.__setitem__("dataset_confirmation_changed", True), "dataset_confirmation_changed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("dry_run_status", "failed"), "materialization_dry_run_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("dry_run_errors", ["error"]), "materialization_dry_run_has_errors"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("planner_status", "failed"), "planner_summary_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("offline_planner_status", "failed"), "offline_planner_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("preflight_status", "failed"), "preflight_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("package_binding_status", "failed"), "package_binding_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("formal_package_validation_status", "failed"), "formal_package_validation_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("materialization_decision", "blocked"), "materialization_decision_not_planned"),
    ],
)
def test_status_failures(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], mutator)
    if target == "execution_request":
        _refresh_execution_request_summary_sha(paths)

    summary = preflight_property_materializer_execution_request(**_kwargs(paths))

    assert summary["preflight_status"] == "failed"
    assert error_code in summary["preflight_errors"]


def test_dry_run_needs_review_returns_needs_review_by_default_and_fails_strict(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path, package_binding_status="needs_review")

    summary = preflight_property_materializer_execution_request(**_kwargs(paths))
    strict = preflight_property_materializer_execution_request(
        **_kwargs(paths),
        require_dry_run_passed=True,
    )

    assert summary["preflight_status"] == "needs_review"
    assert "materialization_dry_run_needs_review_allowed" in summary["warnings"]
    assert strict["preflight_status"] == "failed"
    assert "materialization_dry_run_needs_review" in strict["preflight_errors"]


@pytest.mark.parametrize(
    ("target", "field", "error_code"),
    [
        ("execution_request", "source_manifest_sha256", "manifest_sha256_mismatch"),
        ("execution_request", "source_dry_run_report_sha256", "dry_run_report_sha256_mismatch"),
        ("execution_request", "source_review_manifest_sha256", "review_manifest_sha256_mismatch"),
        ("execution_request", "source_admission_request_sha256", "admission_request_sha256_mismatch"),
        ("execution_request", "source_formal_package_validation_sha256", "formal_package_validation_sha256_mismatch"),
        ("execution_request", "source_property_package_binding_summary_sha256", "property_package_binding_summary_sha256_mismatch"),
        ("execution_request", "source_materialization_plan_sha256", "materialization_plan_sha256_mismatch"),
        ("execution_request", "source_materialization_preflight_summary_sha256", "materialization_plan_preflight_summary_sha256_mismatch"),
        ("execution_request", "source_offline_planner_output_sha256", "offline_planner_output_sha256_mismatch"),
        ("execution_request", "source_property_planner_summary_sha256", "property_planner_summary_sha256_mismatch"),
        ("execution_request", "source_materialization_dry_run_report_sha256", "materialization_dry_run_report_sha256_mismatch"),
        ("execution_request_summary", "execution_request_sha256", "execution_request_sha256_mismatch"),
        ("execution_request_summary", "manifest_sha256", "manifest_sha256_mismatch"),
        ("execution_request_summary", "dry_run_report_sha256", "dry_run_report_sha256_mismatch"),
        ("execution_request_summary", "review_manifest_sha256", "review_manifest_sha256_mismatch"),
        ("execution_request_summary", "admission_request_sha256", "admission_request_sha256_mismatch"),
        ("execution_request_summary", "formal_package_validation_sha256", "formal_package_validation_sha256_mismatch"),
        ("execution_request_summary", "property_package_binding_summary_sha256", "property_package_binding_summary_sha256_mismatch"),
        ("execution_request_summary", "materialization_plan_sha256", "materialization_plan_sha256_mismatch"),
        ("execution_request_summary", "materialization_plan_preflight_summary_sha256", "materialization_plan_preflight_summary_sha256_mismatch"),
        ("execution_request_summary", "offline_planner_output_sha256", "offline_planner_output_sha256_mismatch"),
        ("execution_request_summary", "property_planner_summary_sha256", "property_planner_summary_sha256_mismatch"),
        ("execution_request_summary", "materialization_dry_run_report_sha256", "materialization_dry_run_report_sha256_mismatch"),
    ],
)
def test_hash_mismatches_fail(tmp_path: Path, target: str, field: str, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = preflight_property_materializer_execution_request(**_kwargs(paths))

    assert summary["preflight_status"] == "failed"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("target", "field", "value", "error_code"),
    [
        ("execution_request", "corpus_id", "other-corpus", "corpus_id_mismatch"),
        ("execution_request", "source_dry_run_id", "other-run", "dry_run_id_mismatch"),
        ("execution_request", "review_manifest_id", "other-review", "review_manifest_id_mismatch"),
        ("execution_request", "admission_request_id", "other-admission", "admission_request_id_mismatch"),
        ("execution_request", "materialization_plan_id", "other-plan", "materialization_plan_id_mismatch"),
        ("execution_request", "execution_request_id", "other-request", "execution_request_id_mismatch"),
        ("execution_request_summary", "corpus_id", "other-corpus", "corpus_id_mismatch"),
        ("execution_request_summary", "source_dry_run_id", "other-run", "dry_run_id_mismatch"),
        ("execution_request_summary", "review_manifest_id", "other-review", "review_manifest_id_mismatch"),
        ("execution_request_summary", "admission_request_id", "other-admission", "admission_request_id_mismatch"),
        ("execution_request_summary", "materialization_plan_id", "other-plan", "materialization_plan_id_mismatch"),
        ("execution_request_summary", "execution_request_id", "other-request", "execution_request_id_mismatch"),
    ],
)
def test_id_mismatches_fail(tmp_path: Path, target: str, field: str, value: str, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, value))
    if target == "execution_request":
        _refresh_execution_request_summary_sha(paths)

    summary = preflight_property_materializer_execution_request(**_kwargs(paths))

    assert summary["preflight_status"] == "failed"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("execution_request", lambda payload: payload.__setitem__("execution_records", []), "no_execution_records"),
        ("execution_request", lambda payload: payload.__setitem__("record_count", 2), "execution_record_count_mismatch"),
        ("execution_request", lambda payload: payload.__setitem__("execution_record_ids", ["other-record"]), "execution_record_ids_mismatch"),
        ("execution_request_summary", lambda payload: payload.__setitem__("execution_record_count", 2), "execution_record_count_mismatch"),
        ("execution_request_summary", lambda payload: payload.__setitem__("execution_record_ids", ["other-record"]), "execution_record_ids_mismatch"),
        ("execution_request_summary", lambda payload: payload.__setitem__("materialization_record_ids", ["other-record"]), "materialization_record_ids_mismatch"),
    ],
)
def test_record_count_and_id_mismatches_fail(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], mutator)
    if target == "execution_request":
        _refresh_execution_request_summary_sha(paths)

    summary = preflight_property_materializer_execution_request(**_kwargs(paths))

    assert summary["preflight_status"] == "failed"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("record_id", "error_code"),
    [
        ("property-candidate-002", "execution_record_from_excluded_record"),
        ("property-candidate-003", "execution_record_from_blocked_record"),
        ("property-candidate-004", "execution_record_from_needs_review_record"),
    ],
)
def test_execution_record_source_failures(tmp_path: Path, record_id: str, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path, include_needs_review=True)

    def mutate_request(payload: dict[str, object]) -> None:
        payload["execution_records"][0]["record_id"] = record_id  # type: ignore[index]

    _mutate_json(paths["execution_request"], mutate_request)
    _refresh_execution_request_summary_sha(paths)

    summary = preflight_property_materializer_execution_request(**_kwargs(paths))

    assert summary["preflight_status"] == "failed"
    assert error_code in summary["preflight_errors"]


def test_summary_uses_safe_basenames_and_markdown_has_no_temp_paths(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    markdown_path = tmp_path / "preflight.md"

    summary = preflight_property_materializer_execution_request(
        **_kwargs(paths),
        output_markdown_path=markdown_path,
    )

    serialized = json.dumps(summary, sort_keys=True)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert summary["manifest_path"] == "manifest.json"
    assert summary["execution_request_path"] == "property_materializer_execution_request.json"
    assert str(tmp_path) not in serialized
    assert str(tmp_path) not in markdown


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths["execution_request"], lambda payload: payload.__setitem__("notes", "token abc123"))
    _refresh_execution_request_summary_sha(paths)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_preflight_package(tmp_path)
    markdown_path = tmp_path / "unsafe.md"
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_materializer_execution_preflight._contains_forbidden_material",
        lambda value: True,
    )
    stdout = io.StringIO()

    code = main(_cli_args(paths) + ["--output-markdown", str(markdown_path)], stdout=stdout, stderr=io.StringIO())
    summary = json.loads(stdout.getvalue())

    assert code == 1
    assert summary == {
        "schema_version": "custom_corpus_property_materializer_execution_preflight.v1",
        "preflight_status": "failed",
        "preflight_errors": ["property_materializer_execution_preflight_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not markdown_path.exists()


def test_cli_stdout_is_valid_json(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["preflight_status"] == "passed"
    assert stderr.getvalue() == ""


def test_no_materializer_planner_dry_run_phase1_llm_mineru_pdf_or_parsed_document_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_preflight_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_materialization_planner",
            "ai4s_agent.custom_corpus_property_materialization_dry_run",
            "ai4s_agent.custom_corpus_property_materializer_execution_request",
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

    summary = preflight_property_materializer_execution_request(**_kwargs(paths))

    assert summary["preflight_status"] == "passed"
    assert not any("custom_corpus_materialization_planner" in name for name in imported_modules)
    assert not any(tmp_path.glob("**/materialized_records*"))
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def _kwargs(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "manifest_path": paths["manifest"],
        "dry_run_report_path": paths["dry_run_report"],
        "review_manifest_path": paths["review_manifest"],
        "admission_request_path": paths["admission_request"],
        "formal_package_validation_path": paths["formal_package_validation"],
        "property_package_binding_summary_path": paths["property_package_binding_summary"],
        "materialization_plan_path": paths["materialization_plan_draft"],
        "materialization_plan_preflight_summary_path": paths["materialization_plan_preflight_summary"],
        "offline_planner_output_path": paths["offline_planner_output"],
        "property_planner_summary_path": paths["property_planner_summary"],
        "materialization_dry_run_report_path": paths["materialization_dry_run_report"],
        "execution_request_path": paths["execution_request"],
        "execution_request_summary_path": paths["execution_request_summary"],
    }


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--manifest",
        str(paths["manifest"]),
        "--dry-run-report",
        str(paths["dry_run_report"]),
        "--review-manifest",
        str(paths["review_manifest"]),
        "--admission-request",
        str(paths["admission_request"]),
        "--formal-package-validation",
        str(paths["formal_package_validation"]),
        "--property-package-binding-summary",
        str(paths["property_package_binding_summary"]),
        "--materialization-plan",
        str(paths["materialization_plan_draft"]),
        "--materialization-plan-preflight-summary",
        str(paths["materialization_plan_preflight_summary"]),
        "--offline-planner-output",
        str(paths["offline_planner_output"]),
        "--property-planner-summary",
        str(paths["property_planner_summary"]),
        "--materialization-dry-run-report",
        str(paths["materialization_dry_run_report"]),
        "--execution-request",
        str(paths["execution_request"]),
        "--execution-request-summary",
        str(paths["execution_request_summary"]),
    ]


def _write_preflight_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
    include_needs_review: bool = False,
) -> dict[str, Path]:
    paths = _write_request_package(
        tmp_path,
        package_binding_status=package_binding_status,
        include_needs_review=include_needs_review,
    )
    summary = build_property_materializer_execution_request(
        **_request_kwargs(paths),
        confirm_materializer_execution_request_output=True,
        allow_dry_run_needs_review=package_binding_status == "needs_review",
    )
    assert summary["request_status"] == "written"
    request_dir = paths["request_output_dir"] / "property-materializer-execution-request-001"
    paths["execution_request"] = request_dir / "property_materializer_execution_request.json"
    paths["execution_request_summary"] = request_dir / "property_materializer_execution_request_summary.json"
    return paths


def _refresh_execution_request_summary_sha(paths: dict[str, Path]) -> None:
    _mutate_json(
        paths["execution_request_summary"],
        lambda payload: payload.__setitem__("execution_request_sha256", _sha256_file(paths["execution_request"])),
    )
