from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_quarantine_materializer import (
    main,
    run_property_quarantine_materializer,
)
from test_custom_corpus_property_materialization_plan_preflight import (
    _mutate_json,
    _sha256_file,
)
from test_custom_corpus_property_materializer_execution_preflight import (
    _kwargs as _execution_preflight_kwargs,
)
from test_custom_corpus_property_materializer_execution_preflight import (
    _write_preflight_package,
)
from test_custom_corpus_property_materializer_execution_preflight import (
    preflight_property_materializer_execution_request,
)


def test_valid_full_package_writes_quarantine_candidate_artifact_summary_and_evidence(tmp_path: Path) -> None:
    paths = _write_quarantine_package(tmp_path)

    summary = run_property_quarantine_materializer(
        **_kwargs(paths),
        confirm_quarantine_materialization=True,
    )

    run_dir = paths["quarantine_output_dir"] / "property-quarantine-materializer-001"
    candidate = json.loads((run_dir / "property_quarantine_candidate_records.json").read_text(encoding="utf-8"))
    written_summary = json.loads((run_dir / "property_quarantine_materializer_summary.json").read_text(encoding="utf-8"))
    evidence = (run_dir / "redacted_property_quarantine_materializer_evidence.md").read_text(encoding="utf-8")
    assert written_summary == summary
    assert candidate["schema_version"] == "custom_corpus_property_quarantine_materialization.v1"
    assert summary["schema_version"] == "custom_corpus_property_quarantine_materializer.v1"
    assert summary["materializer_status"] == "written"
    assert candidate["materializer_status"] == "written"
    assert summary["quarantine_run_id"] == "property-quarantine-materializer-001"
    assert candidate["quarantine_run_id"] == "property-quarantine-materializer-001"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["source_dry_run_id"] == "custom-dry-run-example-001"
    assert summary["review_manifest_id"] == "review-example-001"
    assert summary["admission_request_id"] == "property-admission-draft-001"
    assert summary["materialization_plan_id"] == "property-materialization-plan-001"
    assert summary["execution_request_id"] == "property-materializer-execution-request-001"
    assert summary["review_queue_id"] == "property-review-queue-001"
    assert summary["property_candidate_manifest_id"] == "property-candidates-001"
    assert summary["dataset_target"] == "example-candidate-target"
    assert summary["execution_preflight_status"] == "passed"
    assert summary["dry_run_status"] == "passed"
    assert summary["planner_status"] == "planned"
    assert summary["offline_planner_status"] == "planned"
    assert summary["package_binding_status"] == "passed"
    assert summary["formal_package_validation_status"] == "passed"
    assert summary["materialization_decision"] == "planned"
    assert summary["training_admitted"] is False
    assert summary["phase1_status"] == "not_run"
    assert summary["dataset_confirmation_changed"] is False
    assert summary["admission_record_count"] == 2
    assert summary["admit_count"] == 1
    assert summary["exclude_count"] == 1
    assert summary["blocked_record_count"] == 1
    assert summary["materialization_record_count"] == 1
    assert summary["execution_record_count"] == 1
    assert summary["candidate_record_count"] == 1
    assert summary["candidate_record_ids"] == ["property-quarantine-materializer-001-property-materialization-plan-001-property-candidate-001"]
    assert summary["materialization_record_ids"] == ["property-materialization-plan-001-property-candidate-001"]
    assert summary["execution_record_ids"] == [
        "property-materializer-execution-request-001-property-materialization-plan-001-property-candidate-001"
    ]
    assert summary["admit_record_ids"] == ["property-candidate-001"]
    assert summary["exclude_record_ids"] == ["property-candidate-002"]
    assert summary["blocked_record_ids"] == ["property-candidate-003"]
    assert summary["materializer_errors"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"
    assert candidate["candidate_record_count"] == 1
    assert candidate["candidate_record_ids"] == summary["candidate_record_ids"]
    assert candidate["training_admitted"] is False
    assert candidate["phase1_status"] == "not_run"
    assert candidate["dataset_confirmation_changed"] is False
    assert "candidate quarantine only" in candidate["boundary_statement"]
    assert "this is candidate quarantine materialization only" in evidence


def test_candidate_records_contain_safe_ids_hashes_and_redacted_plan_summaries(tmp_path: Path) -> None:
    paths = _write_quarantine_package(tmp_path)

    run_property_quarantine_materializer(
        **_kwargs(paths),
        confirm_quarantine_materialization=True,
    )

    candidate = json.loads(
        (
            paths["quarantine_output_dir"]
            / "property-quarantine-materializer-001"
            / "property_quarantine_candidate_records.json"
        ).read_text(encoding="utf-8")
    )
    record = candidate["candidate_records"][0]
    assert record == {
        "candidate_record_id": "property-quarantine-materializer-001-property-materialization-plan-001-property-candidate-001",
        "quarantine_run_id": "property-quarantine-materializer-001",
        "execution_record_id": "property-materializer-execution-request-001-property-materialization-plan-001-property-candidate-001",
        "materialization_record_id": "property-materialization-plan-001-property-candidate-001",
        "record_id": "property-candidate-001",
        "admission_record_id": "property-admission-draft-001-property-candidate-001",
        "review_id": "review-record-001",
        "document_id": "doc-example-001",
        "field_name": "plqy",
        "candidate_status": "quarantined",
        "source_artifact_sha256": "sha256:" + "c" * 64,
        "review_artifact_sha256": _sha256_file(paths["review_manifest"]),
        "admission_request_sha256": _sha256_file(paths["admission_request"]),
        "package_validation_sha256": _sha256_file(paths["formal_package_validation"]),
        "materialization_plan_sha256": _sha256_file(paths["materialization_plan_draft"]),
        "offline_planner_output_sha256": _sha256_file(paths["offline_planner_output"]),
        "materialization_dry_run_report_sha256": _sha256_file(paths["materialization_dry_run_report"]),
        "execution_request_sha256": _sha256_file(paths["execution_request"]),
        "execution_preflight_summary_sha256": _sha256_file(paths["execution_preflight_summary"]),
        "normalized_value_summary": "short normalized value summary",
        "provenance_summary": "short provenance summary",
        "materialization_boundary": [
            "candidate_only",
            "not_training",
            "not_phase1",
            "dataset_confirmation_unchanged",
        ],
    }
    serialized = json.dumps(candidate, sort_keys=True)
    assert "raw table" not in serialized.lower()
    assert "article text" not in serialized.lower()
    assert ".pdf" not in serialized.lower()
    assert str(tmp_path) not in serialized


def test_missing_confirmation_exits_1_and_writes_no_candidate_artifact(tmp_path: Path) -> None:
    paths = _write_quarantine_package(tmp_path)

    summary = run_property_quarantine_materializer(
        **_kwargs(paths),
        confirm_quarantine_materialization=False,
    )

    assert summary["materializer_status"] == "failed"
    assert "quarantine_materialization_not_confirmed" in summary["materializer_errors"]
    assert not (
        paths["quarantine_output_dir"]
        / "property-quarantine-materializer-001"
        / "property_quarantine_candidate_records.json"
    ).exists()


def test_execution_preflight_needs_review_blocks_unless_allowed(tmp_path: Path) -> None:
    paths = _write_quarantine_package(tmp_path, package_binding_status="needs_review")

    blocked = run_property_quarantine_materializer(
        **_kwargs(paths),
        confirm_quarantine_materialization=True,
    )
    allowed = run_property_quarantine_materializer(
        **_kwargs(paths, quarantine_run_id="property-quarantine-materializer-002"),
        confirm_quarantine_materialization=True,
        allow_execution_preflight_needs_review=True,
    )

    assert blocked["materializer_status"] == "failed"
    assert "execution_preflight_needs_review" in blocked["materializer_errors"]
    assert allowed["materializer_status"] == "needs_review"
    assert "execution_preflight_needs_review_allowed" in allowed["warnings"]
    assert (
        paths["quarantine_output_dir"]
        / "property-quarantine-materializer-002"
        / "property_quarantine_candidate_records.json"
    ).exists()


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("execution_preflight_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "execution_preflight_schema_invalid"),
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
    paths = _write_quarantine_package(tmp_path)
    _mutate_json(paths[target], mutator)
    if target == "execution_request":
        _refresh_execution_preflight_request_hash(paths)

    summary = run_property_quarantine_materializer(
        **_kwargs(paths),
        confirm_quarantine_materialization=True,
    )

    assert summary["materializer_status"] == "failed"
    assert error_code in summary["materializer_errors"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("execution_preflight_summary", lambda payload: payload.__setitem__("preflight_status", "failed"), "execution_preflight_failed"),
        ("execution_request", lambda payload: payload.__setitem__("request_status", "blocked"), "execution_request_not_written"),
        ("execution_request", lambda payload: payload.__setitem__("execution_mode", "materialize"), "execution_mode_invalid"),
        ("execution_request", lambda payload: payload.__setitem__("materializer_status", "success"), "materializer_status_not_run"),
        ("execution_request", lambda payload: payload.__setitem__("phase1_status", "success"), "phase1_ran"),
        ("execution_request", lambda payload: payload.__setitem__("training_admitted", True), "training_admitted"),
        ("execution_request", lambda payload: payload.__setitem__("dataset_confirmation_changed", True), "dataset_confirmation_changed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("dry_run_status", "failed"), "materialization_dry_run_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("planner_status", "failed"), "planner_summary_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("offline_planner_status", "failed"), "offline_planner_failed"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("materialization_decision", "blocked"), "materialization_decision_not_planned"),
    ],
)
def test_status_failures(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_quarantine_package(tmp_path)
    _mutate_json(paths[target], mutator)
    if target == "execution_request":
        _refresh_execution_preflight_request_hash(paths)

    summary = run_property_quarantine_materializer(
        **_kwargs(paths),
        confirm_quarantine_materialization=True,
    )

    assert summary["materializer_status"] == "failed"
    assert error_code in summary["materializer_errors"]


@pytest.mark.parametrize(
    ("field", "error_code"),
    [
        ("manifest_sha256", "manifest_sha256_mismatch"),
        ("dry_run_report_sha256", "dry_run_report_sha256_mismatch"),
        ("review_manifest_sha256", "review_manifest_sha256_mismatch"),
        ("admission_request_sha256", "admission_request_sha256_mismatch"),
        ("formal_package_validation_sha256", "formal_package_validation_sha256_mismatch"),
        ("property_package_binding_summary_sha256", "property_package_binding_summary_sha256_mismatch"),
        ("materialization_plan_sha256", "materialization_plan_sha256_mismatch"),
        ("materialization_plan_preflight_summary_sha256", "materialization_plan_preflight_summary_sha256_mismatch"),
        ("offline_planner_output_sha256", "offline_planner_output_sha256_mismatch"),
        ("property_planner_summary_sha256", "property_planner_summary_sha256_mismatch"),
        ("materialization_dry_run_report_sha256", "materialization_dry_run_report_sha256_mismatch"),
        ("execution_request_sha256", "execution_request_sha256_mismatch"),
        ("execution_request_summary_sha256", "execution_request_summary_sha256_mismatch"),
    ],
)
def test_execution_preflight_hash_mismatches_fail(tmp_path: Path, field: str, error_code: str) -> None:
    paths = _write_quarantine_package(tmp_path)
    _mutate_json(paths["execution_preflight_summary"], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = run_property_quarantine_materializer(
        **_kwargs(paths),
        confirm_quarantine_materialization=True,
    )

    assert summary["materializer_status"] == "failed"
    assert error_code in summary["materializer_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("corpus_id", "other-corpus", "corpus_id_mismatch"),
        ("source_dry_run_id", "other-run", "dry_run_id_mismatch"),
        ("review_manifest_id", "other-review", "review_manifest_id_mismatch"),
        ("admission_request_id", "other-admission", "admission_request_id_mismatch"),
        ("materialization_plan_id", "other-plan", "materialization_plan_id_mismatch"),
        ("execution_request_id", "other-request", "execution_request_id_mismatch"),
    ],
)
def test_id_mismatches_fail(tmp_path: Path, field: str, value: str, error_code: str) -> None:
    paths = _write_quarantine_package(tmp_path)
    _mutate_json(paths["execution_preflight_summary"], lambda payload: payload.__setitem__(field, value))

    summary = run_property_quarantine_materializer(
        **_kwargs(paths),
        confirm_quarantine_materialization=True,
    )

    assert summary["materializer_status"] == "failed"
    assert error_code in summary["materializer_errors"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("execution_request", lambda payload: payload.__setitem__("execution_records", []), "no_execution_records"),
        ("materialization_plan_draft", lambda payload: payload.__setitem__("materialization_records", []), "no_materialization_records"),
        ("execution_preflight_summary", lambda payload: payload.__setitem__("execution_record_count", 2), "execution_record_count_mismatch"),
        ("execution_preflight_summary", lambda payload: payload.__setitem__("execution_record_ids", ["other-record"]), "execution_record_ids_mismatch"),
        ("execution_preflight_summary", lambda payload: payload.__setitem__("materialization_record_ids", ["other-record"]), "materialization_record_ids_mismatch"),
    ],
)
def test_record_count_and_id_mismatches_fail(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_quarantine_package(tmp_path)
    _mutate_json(paths[target], mutator)
    if target == "execution_request":
        _refresh_execution_preflight_request_hash(paths)

    summary = run_property_quarantine_materializer(
        **_kwargs(paths),
        confirm_quarantine_materialization=True,
    )

    assert summary["materializer_status"] == "failed"
    assert error_code in summary["materializer_errors"]


@pytest.mark.parametrize(
    ("record_id", "error_code"),
    [
        ("property-candidate-002", "candidate_record_from_excluded_record"),
        ("property-candidate-003", "candidate_record_from_blocked_record"),
        ("property-candidate-004", "candidate_record_from_needs_review_record"),
    ],
)
def test_candidate_record_source_failures(tmp_path: Path, record_id: str, error_code: str) -> None:
    paths = _write_quarantine_package(tmp_path, include_needs_review=True)

    def mutate_request(payload: dict[str, object]) -> None:
        payload["execution_records"][0]["record_id"] = record_id  # type: ignore[index]

    _mutate_json(paths["execution_request"], mutate_request)
    _refresh_execution_preflight_request_hash(paths)

    summary = run_property_quarantine_materializer(
        **_kwargs(paths),
        confirm_quarantine_materialization=True,
    )

    assert summary["materializer_status"] == "failed"
    assert error_code in summary["materializer_errors"]


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_quarantine_package(tmp_path)
    run_dir = paths["quarantine_output_dir"] / "property-quarantine-materializer-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = run_property_quarantine_materializer(
        **_kwargs(paths),
        confirm_quarantine_materialization=True,
    )

    assert summary["materializer_status"] == "failed"
    assert "output_directory_not_clean" in summary["materializer_errors"]


def test_summary_uses_safe_basenames_and_generated_artifacts_have_no_temp_paths(tmp_path: Path) -> None:
    paths = _write_quarantine_package(tmp_path)

    summary = run_property_quarantine_materializer(
        **_kwargs(paths),
        confirm_quarantine_materialization=True,
    )
    run_dir = paths["quarantine_output_dir"] / "property-quarantine-materializer-001"
    serialized = json.dumps(summary, sort_keys=True)
    artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in run_dir.iterdir())

    assert summary["manifest_path"] == "manifest.json"
    assert summary["execution_preflight_summary_path"] == "execution_preflight_summary.json"
    assert str(tmp_path) not in serialized
    assert str(tmp_path) not in artifact_text


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_quarantine_package(tmp_path)
    _mutate_json(paths["execution_preflight_summary"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-quarantine-materialization"], stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_candidate_artifact_or_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_quarantine_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_quarantine_materializer._contains_forbidden_material",
        lambda value: True,
    )
    stdout = io.StringIO()

    code = main(
        _cli_args(paths) + ["--confirm-quarantine-materialization"],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    summary = json.loads(stdout.getvalue())
    run_dir = paths["quarantine_output_dir"] / "property-quarantine-materializer-001"

    assert code == 1
    assert summary == {
        "schema_version": "custom_corpus_property_quarantine_materializer.v1",
        "materializer_status": "failed",
        "materializer_errors": ["property_quarantine_materializer_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "property_quarantine_candidate_records.json").exists()
    assert not (run_dir / "redacted_property_quarantine_materializer_evidence.md").exists()


def test_cli_stdout_is_valid_json_and_evidence_contains_boundary_statement(tmp_path: Path) -> None:
    paths = _write_quarantine_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-quarantine-materialization"], stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())
    evidence = (
        paths["quarantine_output_dir"]
        / "property-quarantine-materializer-001"
        / "redacted_property_quarantine_materializer_evidence.md"
    ).read_text(encoding="utf-8")

    assert code == 0
    assert summary["materializer_status"] == "written"
    assert "this is candidate quarantine materialization only" in evidence
    assert "No training data was admitted" in evidence
    assert "No training CSV/JSONL/Parquet/LMDB was created" in evidence
    assert "No Phase 1 was run" in evidence
    assert "DatasetConfirmation was not changed" in evidence
    assert stderr.getvalue() == ""


def test_no_forbidden_runner_or_artifact_creation_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_quarantine_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_materialization_planner",
            "ai4s_agent.custom_corpus_property_materialization_dry_run",
            "ai4s_agent.custom_corpus_property_materializer_execution_request",
            "ai4s_agent.custom_corpus_property_materializer_execution_preflight",
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

    summary = run_property_quarantine_materializer(
        **_kwargs(paths),
        confirm_quarantine_materialization=True,
    )

    assert summary["materializer_status"] == "written"
    assert not any("custom_corpus_materialization_planner" in name for name in imported_modules)
    assert not any(tmp_path.glob("**/training*"))
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def _kwargs(paths: dict[str, Path], *, quarantine_run_id: str = "property-quarantine-materializer-001") -> dict[str, object]:
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
        "execution_preflight_summary_path": paths["execution_preflight_summary"],
        "output_dir": paths["quarantine_output_dir"],
        "quarantine_run_id": quarantine_run_id,
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
        "--execution-request",
        str(paths["execution_request"]),
        "--execution-request-summary",
        str(paths["execution_request_summary"]),
        "--execution-preflight-summary",
        str(paths["execution_preflight_summary"]),
        "--output-dir",
        str(paths["quarantine_output_dir"]),
        "--quarantine-run-id",
        "property-quarantine-materializer-001",
        "--created-by",
        "operator-redacted",
    ]


def _write_quarantine_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
    include_needs_review: bool = False,
) -> dict[str, Path]:
    paths = _write_preflight_package(
        tmp_path,
        package_binding_status=package_binding_status,
        include_needs_review=include_needs_review,
    )
    paths["execution_preflight_summary"] = tmp_path / "execution_preflight_summary.json"
    summary = preflight_property_materializer_execution_request(
        **_execution_preflight_kwargs(paths),
        output_summary_path=paths["execution_preflight_summary"],
    )
    assert summary["preflight_status"] in {"passed", "needs_review"}
    paths["quarantine_output_dir"] = tmp_path / "property_quarantine_materializer"
    return paths


def _refresh_execution_preflight_request_hash(paths: dict[str, Path]) -> None:
    _mutate_json(
        paths["execution_preflight_summary"],
        lambda payload: payload.__setitem__("execution_request_sha256", _sha256_file(paths["execution_request"])),
    )
