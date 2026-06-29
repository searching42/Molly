from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_materializer_execution_request import (
    build_property_materializer_execution_request,
    main,
)
from test_custom_corpus_property_materialization_dry_run import (
    _kwargs as _dry_run_kwargs,
)
from test_custom_corpus_property_materialization_dry_run import (
    _write_dry_run_package,
)
from test_custom_corpus_property_materialization_dry_run import (
    run_property_materialization_dry_run,
)
from test_custom_corpus_property_materialization_plan_preflight import (
    _mutate_json,
    _sha256_file,
)


def test_valid_dry_run_package_writes_execution_request_artifacts(tmp_path: Path) -> None:
    paths = _write_request_package(tmp_path)

    summary = build_property_materializer_execution_request(
        **_kwargs(paths),
        confirm_materializer_execution_request_output=True,
    )

    run_dir = paths["request_output_dir"] / "property-materializer-execution-request-001"
    request = json.loads((run_dir / "property_materializer_execution_request.json").read_text(encoding="utf-8"))
    written_summary = json.loads((run_dir / "property_materializer_execution_request_summary.json").read_text(encoding="utf-8"))
    evidence = (run_dir / "redacted_property_materializer_execution_request_evidence.md").read_text(encoding="utf-8")
    assert written_summary == summary
    assert summary["schema_version"] == "custom_corpus_property_materializer_execution_request_builder.v1"
    assert summary["request_status"] == "written"
    assert summary["execution_request_id"] == "property-materializer-execution-request-001"
    assert summary["execution_request_path"] == "property_materializer_execution_request.json"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["source_dry_run_id"] == "custom-dry-run-example-001"
    assert summary["review_manifest_id"] == "review-example-001"
    assert summary["admission_request_id"] == "property-admission-draft-001"
    assert summary["materialization_plan_id"] == "property-materialization-plan-001"
    assert summary["review_queue_id"] == "property-review-queue-001"
    assert summary["property_candidate_manifest_id"] == "property-candidates-001"
    assert summary["dataset_target"] == "example-candidate-target"
    assert summary["dry_run_status"] == "passed"
    assert summary["planner_status"] == "planned"
    assert summary["offline_planner_status"] == "planned"
    assert summary["preflight_status"] == "passed"
    assert summary["package_binding_status"] == "passed"
    assert summary["formal_package_validation_status"] == "passed"
    assert summary["materialization_decision"] == "planned"
    assert summary["phase1_status"] == "not_run"
    assert summary["training_admitted"] is False
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
    assert summary["request_errors"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"
    assert request["schema_version"] == "custom_corpus_property_materializer_execution_request.v1"
    assert request["request_status"] == "written"
    assert request["execution_mode"] == "request_only"
    assert request["materializer_status"] == "not_run"
    assert request["phase1_status"] == "not_run"
    assert request["training_admitted"] is False
    assert request["dataset_confirmation_changed"] is False
    assert request["record_count"] == 1
    assert request["execution_record_ids"] == summary["execution_record_ids"]
    assert "this is a materializer execution request only" in request["boundary_statement"]
    assert "this is a materializer execution request only" in evidence


def test_execution_request_records_are_safe_id_and_hash_only(tmp_path: Path) -> None:
    paths = _write_request_package(tmp_path)

    build_property_materializer_execution_request(
        **_kwargs(paths),
        confirm_materializer_execution_request_output=True,
    )

    request_path = (
        paths["request_output_dir"]
        / "property-materializer-execution-request-001"
        / "property_materializer_execution_request.json"
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    record = request["execution_records"][0]
    assert record == {
        "execution_record_id": "property-materializer-execution-request-001-property-materialization-plan-001-property-candidate-001",
        "materialization_record_id": "property-materialization-plan-001-property-candidate-001",
        "record_id": "property-candidate-001",
        "admission_record_id": "property-admission-draft-001-property-candidate-001",
        "review_id": "review-record-001",
        "document_id": "doc-example-001",
        "field_name": "plqy",
        "planned_action": "request_materialize_candidate",
        "source_artifact_sha256": "sha256:" + "c" * 64,
        "review_artifact_sha256": _sha256_file(paths["review_manifest"]),
        "admission_request_sha256": _sha256_file(paths["admission_request"]),
        "package_validation_sha256": _sha256_file(paths["formal_package_validation"]),
        "materialization_plan_sha256": _sha256_file(paths["materialization_plan_draft"]),
        "offline_planner_output_sha256": _sha256_file(paths["offline_planner_output"]),
        "dry_run_report_sha256": _sha256_file(paths["materialization_dry_run_report"]),
    }
    serialized = json.dumps(request, sort_keys=True)
    assert "short normalized value summary" not in serialized
    assert "short provenance summary" not in serialized
    assert "materialization_reason" not in serialized


def test_missing_confirmation_exits_1_and_writes_no_request(tmp_path: Path) -> None:
    paths = _write_request_package(tmp_path)

    summary = build_property_materializer_execution_request(
        **_kwargs(paths),
        confirm_materializer_execution_request_output=False,
    )

    assert summary["request_status"] == "blocked"
    assert "materializer_execution_request_not_confirmed" in summary["request_errors"]
    assert not (
        paths["request_output_dir"]
        / "property-materializer-execution-request-001"
        / "property_materializer_execution_request.json"
    ).exists()


def test_materialization_dry_run_needs_review_blocks_unless_allowed(tmp_path: Path) -> None:
    paths = _write_request_package(tmp_path, package_binding_status="needs_review")

    blocked = build_property_materializer_execution_request(
        **_kwargs(paths),
        confirm_materializer_execution_request_output=True,
    )
    allowed = build_property_materializer_execution_request(
        **_kwargs(paths, request_id="property-materializer-execution-request-002"),
        confirm_materializer_execution_request_output=True,
        allow_dry_run_needs_review=True,
    )

    assert blocked["request_status"] == "blocked"
    assert "materialization_dry_run_needs_review" in blocked["request_errors"]
    assert allowed["request_status"] == "written"
    assert "materialization_dry_run_needs_review_allowed" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("schema_version", "wrong"), "materialization_dry_run_report_schema_invalid"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("dry_run_status", "failed"), "materialization_dry_run_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("dry_run_errors", ["error"]), "materialization_dry_run_has_errors"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("planner_status", "failed"), "planner_summary_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("offline_planner_status", "failed"), "offline_planner_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("preflight_status", "failed"), "preflight_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("package_binding_status", "failed"), "package_binding_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("formal_package_validation_status", "failed"), "formal_package_validation_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("materialization_decision", "blocked"), "materialization_decision_not_planned"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("source_dry_run_decision", "failed"), "dry_run_not_passed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("phase1_status", "success"), "dry_run_phase1_ran"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("training_admitted", True), "dry_run_training_admitted"),
        ("property_planner_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "property_planner_schema_invalid"),
        ("offline_planner_output", lambda payload: payload.__setitem__("schema_version", "wrong"), "offline_planner_schema_invalid"),
        ("materialization_plan_preflight_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "preflight_schema_invalid"),
        ("property_package_binding_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "property_package_binding_schema_invalid"),
        ("formal_package_validation", lambda payload: payload.__setitem__("schema_version", "wrong"), "formal_package_validation_schema_invalid"),
        ("materialization_plan_draft", lambda payload: payload.__setitem__("schema_version", "wrong"), "materialization_plan_schema_invalid"),
    ],
)
def test_status_and_schema_failures(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_request_package(tmp_path)
    _mutate_json(paths[target], mutator)
    if target == "offline_planner_output":
        _refresh_dry_run_hashes(paths)

    summary = build_property_materializer_execution_request(
        **_kwargs(paths),
        confirm_materializer_execution_request_output=True,
    )

    assert summary["request_status"] == "blocked"
    assert error_code in summary["request_errors"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("manifest_sha256", "sha256:" + "0" * 64), "manifest_sha256_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("dry_run_report_sha256", "sha256:" + "1" * 64), "dry_run_report_sha256_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("review_manifest_sha256", "sha256:" + "2" * 64), "review_manifest_sha256_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("admission_request_sha256", "sha256:" + "3" * 64), "admission_request_sha256_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("formal_package_validation_sha256", "sha256:" + "4" * 64), "formal_package_validation_sha256_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("property_package_binding_summary_sha256", "sha256:" + "5" * 64), "property_package_binding_summary_sha256_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("materialization_plan_sha256", "sha256:" + "6" * 64), "materialization_plan_sha256_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("materialization_plan_preflight_summary_sha256", "sha256:" + "7" * 64), "materialization_plan_preflight_summary_sha256_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("offline_planner_output_sha256", "sha256:" + "8" * 64), "offline_planner_output_sha256_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("property_planner_summary_sha256", "sha256:" + "9" * 64), "property_planner_summary_sha256_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("corpus_id", "other-corpus"), "corpus_id_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("corpus_dry_run_id", "other-run"), "dry_run_id_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("review_manifest_id", "other-review"), "review_manifest_id_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("admission_request_id", "other-admission"), "admission_request_id_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("materialization_plan_id", "other-plan"), "materialization_plan_id_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("materialization_record_count", 2), "materialization_record_count_mismatch"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("materialization_record_ids", ["other-record"]), "materialization_record_ids_mismatch"),
    ],
)
def test_hash_id_and_count_mismatches(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_request_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = build_property_materializer_execution_request(
        **_kwargs(paths),
        confirm_materializer_execution_request_output=True,
    )

    assert summary["request_status"] == "blocked"
    assert error_code in summary["request_errors"]


@pytest.mark.parametrize(
    ("record_mutator", "error_code"),
    [
        (lambda record: record.__setitem__("record_id", "property-candidate-002"), "execution_record_from_excluded_record"),
        (lambda record: record.__setitem__("record_id", "property-candidate-003"), "execution_record_from_blocked_record"),
        (
            lambda record: (
                record.__setitem__("record_id", "property-candidate-004"),
                record.__setitem__("review_decision", "needs_review"),
            ),
            "execution_record_from_needs_review_record",
        ),
    ],
)
def test_execution_record_source_failures(tmp_path: Path, record_mutator: object, error_code: str) -> None:
    paths = _write_request_package(tmp_path, include_needs_review=True)
    _mutate_json(paths["materialization_plan_draft"], lambda payload: record_mutator(payload["materialization_records"][0]))  # type: ignore[index]
    _mutate_json(
        paths["materialization_dry_run_report"],
        lambda payload: payload["dry_run_record_summaries"][0].__setitem__("record_id", "property-candidate-004"),
    )

    summary = build_property_materializer_execution_request(
        **_kwargs(paths),
        confirm_materializer_execution_request_output=True,
    )

    assert summary["request_status"] == "blocked"
    assert error_code in summary["request_errors"]


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_request_package(tmp_path)
    run_dir = paths["request_output_dir"] / "property-materializer-execution-request-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = build_property_materializer_execution_request(
        **_kwargs(paths),
        confirm_materializer_execution_request_output=True,
    )

    assert summary["request_status"] == "blocked"
    assert "output_directory_not_clean" in summary["request_errors"]


def test_summary_uses_safe_basenames_only_and_generated_artifacts_have_no_temp_paths(tmp_path: Path) -> None:
    paths = _write_request_package(tmp_path)

    summary = build_property_materializer_execution_request(
        **_kwargs(paths),
        confirm_materializer_execution_request_output=True,
    )
    run_dir = paths["request_output_dir"] / "property-materializer-execution-request-001"
    serialized = json.dumps(summary, sort_keys=True)
    artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in run_dir.iterdir())

    assert summary["manifest_path"] == "manifest.json"
    assert summary["materialization_dry_run_report_path"] == "property_materialization_dry_run_report.json"
    assert str(tmp_path) not in serialized
    assert str(tmp_path) not in artifact_text


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_request_package(tmp_path)
    _mutate_json(paths["materialization_plan_draft"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-materializer-execution-request-output"], stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_execution_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_request_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_materializer_execution_request._contains_forbidden_material",
        lambda value: True,
    )
    stdout = io.StringIO()

    code = main(
        _cli_args(paths) + ["--confirm-materializer-execution-request-output"],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    summary = json.loads(stdout.getvalue())
    run_dir = paths["request_output_dir"] / "property-materializer-execution-request-001"

    assert code == 1
    assert summary == {
        "schema_version": "custom_corpus_property_materializer_execution_request_builder.v1",
        "request_status": "blocked",
        "request_errors": ["property_materializer_execution_request_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "property_materializer_execution_request.json").exists()


def test_cli_stdout_is_valid_json_and_evidence_contains_boundary_statement(tmp_path: Path) -> None:
    paths = _write_request_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-materializer-execution-request-output"], stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())
    evidence = (
        paths["request_output_dir"]
        / "property-materializer-execution-request-001"
        / "redacted_property_materializer_execution_request_evidence.md"
    ).read_text(encoding="utf-8")

    assert code == 0
    assert summary["request_status"] == "written"
    assert "this is a materializer execution request only" in evidence
    assert "No real materializer was run" in evidence
    assert "No materialization was executed" in evidence
    assert "No materialized records were created" in evidence
    assert "No candidate/training CSV was created" in evidence
    assert "No candidate/training JSONL/Parquet/LMDB was created" in evidence
    assert "Phase 1 did not run" in evidence
    assert "DatasetConfirmation was not changed" in evidence
    assert stderr.getvalue() == ""


def test_no_planner_materializer_phase1_llm_mineru_pdf_or_parsed_document_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_request_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
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

    summary = build_property_materializer_execution_request(
        **_kwargs(paths),
        confirm_materializer_execution_request_output=True,
    )

    assert summary["request_status"] == "written"
    assert not any("custom_corpus_materialization_planner" in name for name in imported_modules)
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def _kwargs(paths: dict[str, Path], *, request_id: str = "property-materializer-execution-request-001") -> dict[str, object]:
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
        "output_dir": paths["request_output_dir"],
        "execution_request_id": request_id,
        "created_by": "operator-redacted",
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
        "--output-dir",
        str(paths["request_output_dir"]),
        "--execution-request-id",
        "property-materializer-execution-request-001",
        "--created-by",
        "operator-redacted",
    ]


def _write_request_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
    include_needs_review: bool = False,
) -> dict[str, Path]:
    paths = _write_dry_run_package(
        tmp_path,
        package_binding_status=package_binding_status,
        include_needs_review=include_needs_review,
    )
    dry_run_report = run_property_materialization_dry_run(
        **_dry_run_kwargs(paths),
        confirm_materialization_dry_run=True,
        allow_planner_needs_review=package_binding_status == "needs_review",
    )
    assert dry_run_report["dry_run_status"] in {"passed", "needs_review"}
    paths["materialization_dry_run_report"] = (
        paths["dry_run_output_dir"]
        / "property-materialization-dry-run-001"
        / "property_materialization_dry_run_report.json"
    )
    paths["request_output_dir"] = tmp_path / "property_materializer_execution_request"
    return paths


def _refresh_dry_run_hashes(paths: dict[str, Path]) -> None:
    _mutate_json(
        paths["materialization_dry_run_report"],
        lambda payload: payload.__setitem__("offline_planner_output_sha256", _sha256_file(paths["offline_planner_output"])),
    )
