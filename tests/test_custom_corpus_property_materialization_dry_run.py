from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_materialization_dry_run import (
    main,
    run_property_materialization_dry_run,
)
from ai4s_agent.custom_corpus_property_materialization_planner_runner import (
    run_property_materialization_planner,
)
from test_custom_corpus_property_materialization_planner_runner import (
    _kwargs as _planner_kwargs,
)
from test_custom_corpus_property_materialization_planner_runner import (
    _write_runner_package,
)
from test_custom_corpus_property_materialization_plan_preflight import (
    _mutate_json,
    _sha256_file,
)


def test_valid_full_package_writes_dry_run_report_and_markdown(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)

    report = run_property_materialization_dry_run(**_kwargs(paths), confirm_materialization_dry_run=True)

    run_dir = paths["dry_run_output_dir"] / "property-materialization-dry-run-001"
    evidence = (run_dir / "redacted_property_materialization_dry_run_evidence.md").read_text(encoding="utf-8")
    written = json.loads((run_dir / "property_materialization_dry_run_report.json").read_text(encoding="utf-8"))
    assert written == report
    assert report["schema_version"] == "custom_corpus_property_materialization_dry_run.v1"
    assert report["dry_run_status"] == "passed"
    assert report["dry_run_id"] == "property-materialization-dry-run-001"
    assert report["manifest_path"] == "manifest.json"
    assert report["materialization_plan_path"] == "custom_corpus_materialization.draft.json"
    assert report["offline_planner_output_path"] == "offline_materialization_planner_output.json"
    assert report["property_planner_summary_path"] == "property_materialization_planner_summary.json"
    assert report["corpus_id"] == "example-public-corpus"
    assert report["corpus_dry_run_id"] == "custom-dry-run-example-001"
    assert report["review_manifest_id"] == "review-example-001"
    assert report["admission_request_id"] == "property-admission-draft-001"
    assert report["materialization_plan_id"] == "property-materialization-plan-001"
    assert report["review_queue_id"] == "property-review-queue-001"
    assert report["property_candidate_manifest_id"] == "property-candidates-001"
    assert report["dataset_target"] == "example-candidate-target"
    assert report["planner_status"] == "planned"
    assert report["offline_planner_status"] == "planned"
    assert report["preflight_status"] == "passed"
    assert report["package_binding_status"] == "passed"
    assert report["formal_package_validation_status"] == "passed"
    assert report["materialization_decision"] == "planned"
    assert report["source_dry_run_decision"] == "passed"
    assert report["phase1_status"] == "not_run"
    assert report["training_admitted"] is False
    assert report["admission_record_count"] == 2
    assert report["admit_count"] == 1
    assert report["exclude_count"] == 1
    assert report["blocked_record_count"] == 1
    assert report["materialization_record_count"] == 1
    assert report["materialization_record_ids"] == ["property-materialization-plan-001-property-candidate-001"]
    assert report["admit_record_ids"] == ["property-candidate-001"]
    assert report["exclude_record_ids"] == ["property-candidate-002"]
    assert report["blocked_record_ids"] == ["property-candidate-003"]
    assert report["dry_run_errors"] == []
    assert report["warnings"] == []
    assert report["redaction_status"] == "passed"
    assert "this is a materialization dry-run only" in evidence


def test_dry_run_record_summaries_contain_safe_ids_and_hashes_only(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)

    report = run_property_materialization_dry_run(**_kwargs(paths), confirm_materialization_dry_run=True)

    assert report["dry_run_record_summaries"] == [
        {
            "materialization_record_id": "property-materialization-plan-001-property-candidate-001",
            "record_id": "property-candidate-001",
            "admission_record_id": "property-admission-draft-001-property-candidate-001",
            "review_id": "review-record-001",
            "document_id": "doc-example-001",
            "field_name": "plqy",
            "planned_action": "would_materialize_candidate",
            "source_artifact_sha256": "sha256:" + "c" * 64,
            "review_artifact_sha256": report["review_manifest_sha256"],
            "admission_request_sha256": report["admission_request_sha256"],
            "package_validation_sha256": report["formal_package_validation_sha256"],
        }
    ]
    serialized = json.dumps(report["dry_run_record_summaries"], sort_keys=True)
    assert "short normalized value summary" not in serialized
    assert "short provenance summary" not in serialized
    assert "materialization_reason" not in serialized


def test_missing_confirmation_exits_1_and_writes_no_report(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)

    report = run_property_materialization_dry_run(**_kwargs(paths), confirm_materialization_dry_run=False)

    assert report["dry_run_status"] == "failed"
    assert "materialization_dry_run_not_confirmed" in report["dry_run_errors"]
    assert not (
        paths["dry_run_output_dir"]
        / "property-materialization-dry-run-001"
        / "property_materialization_dry_run_report.json"
    ).exists()


def test_planner_summary_failed_exits_1(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    _mutate_json(paths["property_planner_summary"], lambda payload: payload.__setitem__("planner_status", "failed"))

    report = run_property_materialization_dry_run(**_kwargs(paths), confirm_materialization_dry_run=True)

    assert report["dry_run_status"] == "failed"
    assert "planner_summary_failed" in report["dry_run_errors"]


def test_planner_summary_needs_review_blocks_unless_allowed(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path, package_binding_status="needs_review")

    report = run_property_materialization_dry_run(**_kwargs(paths), confirm_materialization_dry_run=True)

    assert report["dry_run_status"] == "failed"
    assert "planner_summary_needs_review" in report["dry_run_errors"]


def test_planner_summary_needs_review_allowed_returns_needs_review(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path, package_binding_status="needs_review")

    report = run_property_materialization_dry_run(
        **_kwargs(paths),
        confirm_materialization_dry_run=True,
        allow_planner_needs_review=True,
    )

    assert report["dry_run_status"] == "needs_review"
    assert "planner_summary_needs_review_allowed" in report["warnings"]
    assert (
        paths["dry_run_output_dir"]
        / "property-materialization-dry-run-001"
        / "property_materialization_dry_run_report.json"
    ).exists()


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda payload: payload.__setitem__("planner_status", "blocked"), "offline_planner_failed"),
        (lambda payload: payload.__setitem__("schema_version", "wrong"), "offline_planner_schema_invalid"),
        (lambda payload: payload.__setitem__("materialized_records", [{"record_id": "x"}]), "offline_planner_claimed_materialized_records"),
        (lambda payload: payload.__setitem__("candidate_csv_path", "records.csv"), "offline_planner_claimed_candidate_artifact"),
        (lambda payload: payload.__setitem__("candidate_jsonl_path", "records.jsonl"), "offline_planner_claimed_candidate_artifact"),
        (lambda payload: payload.__setitem__("candidate_parquet_path", "records.parquet"), "offline_planner_claimed_candidate_artifact"),
        (lambda payload: payload.__setitem__("candidate_lmdb_path", "records.lmdb"), "offline_planner_claimed_candidate_artifact"),
        (lambda payload: payload.__setitem__("phase1_status", "success"), "offline_planner_claimed_phase1_run"),
        (lambda payload: payload.__setitem__("dataset_confirmation_changed", True), "offline_planner_claimed_dataset_confirmation_change"),
        (lambda payload: payload.__setitem__("training_admitted", True), "offline_planner_claimed_training_admission"),
    ],
)
def test_offline_planner_output_failures(tmp_path: Path, mutator: object, error_code: str) -> None:
    paths = _write_dry_run_package(tmp_path)
    _mutate_json(paths["offline_planner_output"], mutator)
    _refresh_property_planner_hashes(paths)

    report = run_property_materialization_dry_run(**_kwargs(paths), confirm_materialization_dry_run=True)

    assert report["dry_run_status"] == "failed"
    assert error_code in report["dry_run_errors"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("materialization_plan_draft", lambda payload: payload.__setitem__("schema_version", "wrong"), "materialization_plan_schema_invalid"),
        ("materialization_plan_draft", lambda payload: payload.__setitem__("materialization_records", []), "no_materialization_records"),
        ("property_planner_summary", lambda payload: payload.__setitem__("materialization_record_count", 2), "materialization_record_count_mismatch"),
        ("property_planner_summary", lambda payload: payload.__setitem__("materialization_record_ids", ["other-record"]), "materialization_record_ids_mismatch"),
        ("property_planner_summary", lambda payload: payload.__setitem__("manifest_sha256", "sha256:" + "0" * 64), "manifest_sha256_mismatch"),
        ("property_planner_summary", lambda payload: payload.__setitem__("dry_run_report_sha256", "sha256:" + "1" * 64), "dry_run_report_sha256_mismatch"),
        ("property_planner_summary", lambda payload: payload.__setitem__("review_manifest_sha256", "sha256:" + "2" * 64), "review_manifest_sha256_mismatch"),
        ("property_planner_summary", lambda payload: payload.__setitem__("admission_request_sha256", "sha256:" + "3" * 64), "admission_request_sha256_mismatch"),
        ("property_planner_summary", lambda payload: payload.__setitem__("formal_package_validation_sha256", "sha256:" + "4" * 64), "formal_package_validation_sha256_mismatch"),
        (
            "property_planner_summary",
            lambda payload: payload.__setitem__("property_package_binding_summary_sha256", "sha256:" + "5" * 64),
            "property_package_binding_summary_sha256_mismatch",
        ),
        ("property_planner_summary", lambda payload: payload.__setitem__("materialization_plan_sha256", "sha256:" + "6" * 64), "materialization_plan_sha256_mismatch"),
        (
            "property_planner_summary",
            lambda payload: payload.__setitem__("materialization_plan_preflight_summary_sha256", "sha256:" + "7" * 64),
            "materialization_plan_preflight_summary_sha256_mismatch",
        ),
        ("property_planner_summary", lambda payload: payload.__setitem__("offline_planner_output_sha256", "sha256:" + "8" * 64), "offline_planner_output_sha256_mismatch"),
        ("property_planner_summary", lambda payload: payload.__setitem__("corpus_id", "other-corpus"), "corpus_id_mismatch"),
        ("property_planner_summary", lambda payload: payload.__setitem__("dry_run_id", "other-run"), "dry_run_id_mismatch"),
        ("property_planner_summary", lambda payload: payload.__setitem__("review_manifest_id", "other-review"), "review_manifest_id_mismatch"),
        ("property_planner_summary", lambda payload: payload.__setitem__("admission_request_id", "other-admission"), "admission_request_id_mismatch"),
        ("property_planner_summary", lambda payload: payload.__setitem__("materialization_plan_id", "other-plan"), "materialization_plan_id_mismatch"),
        ("dry_run_report", lambda payload: payload.__setitem__("decision", "failed"), "dry_run_not_passed"),
        ("dry_run_report", lambda payload: payload["confirmation_boundary"].__setitem__("phase1_status", "success"), "dry_run_phase1_ran"),
        ("dry_run_report", lambda payload: payload["confirmation_boundary"].__setitem__("training_dataset_admitted", True), "dry_run_training_admitted"),
        ("dry_run_report", lambda payload: payload["confirmation_boundary"].__setitem__("dataset_confirmation_confirmed", True), "dry_run_dataset_confirmed"),
    ],
)
def test_cross_artifact_failures(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_dry_run_package(tmp_path)
    _mutate_json(paths[target], mutator)

    report = run_property_materialization_dry_run(**_kwargs(paths), confirm_materialization_dry_run=True)

    assert report["dry_run_status"] == "failed"
    assert error_code in report["dry_run_errors"]


@pytest.mark.parametrize(
    ("record_mutator", "error_code"),
    [
        (lambda record: record.__setitem__("record_id", "property-candidate-002"), "materialization_record_from_excluded_record"),
        (lambda record: record.__setitem__("record_id", "property-candidate-003"), "materialization_record_from_blocked_record"),
        (
            lambda record: (
                record.__setitem__("record_id", "property-candidate-004"),
                record.__setitem__("review_decision", "needs_review"),
            ),
            "materialization_record_from_needs_review_record",
        ),
    ],
)
def test_materialization_record_source_failures(tmp_path: Path, record_mutator: object, error_code: str) -> None:
    paths = _write_dry_run_package(tmp_path, include_needs_review=True)
    _mutate_json(paths["materialization_plan_draft"], lambda payload: record_mutator(payload["materialization_records"][0]))  # type: ignore[index]

    report = run_property_materialization_dry_run(**_kwargs(paths), confirm_materialization_dry_run=True)

    assert report["dry_run_status"] == "failed"
    assert error_code in report["dry_run_errors"]


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    run_dir = paths["dry_run_output_dir"] / "property-materialization-dry-run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    report = run_property_materialization_dry_run(**_kwargs(paths), confirm_materialization_dry_run=True)

    assert report["dry_run_status"] == "failed"
    assert "output_directory_not_clean" in report["dry_run_errors"]


def test_summary_uses_safe_basenames_only_and_generated_artifacts_have_no_temp_paths(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)

    report = run_property_materialization_dry_run(**_kwargs(paths), confirm_materialization_dry_run=True)
    run_dir = paths["dry_run_output_dir"] / "property-materialization-dry-run-001"
    serialized = json.dumps(report, sort_keys=True)
    artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in run_dir.iterdir())

    assert report["manifest_path"] == "manifest.json"
    assert report["property_planner_summary_path"] == "property_materialization_planner_summary.json"
    assert str(tmp_path) not in serialized
    assert str(tmp_path) not in artifact_text


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    _mutate_json(paths["materialization_plan_draft"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-materialization-dry-run"], stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_unsafe_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_dry_run_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_materialization_dry_run._contains_forbidden_material",
        lambda value: True,
    )
    stdout = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-materialization-dry-run"], stdout=stdout, stderr=io.StringIO())
    report = json.loads(stdout.getvalue())
    run_dir = paths["dry_run_output_dir"] / "property-materialization-dry-run-001"

    assert code == 1
    assert report == {
        "schema_version": "custom_corpus_property_materialization_dry_run.v1",
        "dry_run_status": "failed",
        "dry_run_errors": ["property_materialization_dry_run_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "redacted_property_materialization_dry_run_evidence.md").exists()


def test_cli_stdout_is_valid_json_and_evidence_contains_boundary_statement(tmp_path: Path) -> None:
    paths = _write_dry_run_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-materialization-dry-run"], stdout=stdout, stderr=stderr)
    report = json.loads(stdout.getvalue())
    evidence = (
        paths["dry_run_output_dir"]
        / "property-materialization-dry-run-001"
        / "redacted_property_materialization_dry_run_evidence.md"
    ).read_text(encoding="utf-8")

    assert code == 0
    assert report["dry_run_status"] == "passed"
    assert "this is a materialization dry-run only" in evidence
    assert "No real materializer was run" in evidence
    assert "No materialization was executed" in evidence
    assert "No materialized records were created" in evidence
    assert "No candidate/training CSV was created" in evidence
    assert "No candidate/training JSONL/Parquet/LMDB was created" in evidence
    assert "No training data was admitted" in evidence
    assert "Phase 1 did not run" in evidence
    assert "DatasetConfirmation was not changed" in evidence
    assert stderr.getvalue() == ""


def test_no_planner_materializer_phase1_llm_mineru_pdf_or_parsed_document_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_dry_run_package(tmp_path)
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

    report = run_property_materialization_dry_run(**_kwargs(paths), confirm_materialization_dry_run=True)

    assert report["dry_run_status"] == "passed"
    assert not any("custom_corpus_materialization_planner" in name for name in imported_modules)
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
        "output_dir": paths["dry_run_output_dir"],
        "dry_run_id": "property-materialization-dry-run-001",
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
        "--output-dir",
        str(paths["dry_run_output_dir"]),
        "--dry-run-id",
        "property-materialization-dry-run-001",
    ]


def _write_dry_run_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
    include_needs_review: bool = False,
) -> dict[str, Path]:
    paths = _write_runner_package(
        tmp_path,
        package_binding_status=package_binding_status,
        include_needs_review=include_needs_review,
    )
    planner_summary = run_property_materialization_planner(
        **_planner_kwargs(paths),
        confirm_offline_materialization_planner=True,
        allow_preflight_needs_review=package_binding_status == "needs_review",
    )
    assert planner_summary["planner_status"] in {"planned", "needs_review"}
    planner_run_dir = paths["output_dir"] / "property-materialization-planner-001"
    paths["offline_planner_output"] = planner_run_dir / "offline_materialization_planner_output.json"
    paths["property_planner_summary"] = planner_run_dir / "property_materialization_planner_summary.json"
    paths["dry_run_output_dir"] = tmp_path / "property_materialization_dry_run"
    return paths


def _refresh_property_planner_hashes(paths: dict[str, Path]) -> None:
    _mutate_json(
        paths["property_planner_summary"],
        lambda payload: payload.__setitem__("offline_planner_output_sha256", _sha256_file(paths["offline_planner_output"])),
    )
