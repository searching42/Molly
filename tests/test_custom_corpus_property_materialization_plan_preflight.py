from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_materialization_plan_draft import (
    build_property_materialization_plan_draft,
)
from ai4s_agent.custom_corpus_property_materialization_plan_preflight import (
    main,
    preflight_property_materialization_plan,
)


def test_valid_full_package_returns_passed(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)

    summary = preflight_property_materialization_plan(**_kwargs(paths))

    assert summary["schema_version"] == "custom_corpus_property_materialization_plan_preflight.v1"
    assert summary["preflight_status"] == "passed"
    assert summary["manifest_path"] == "manifest.json"
    assert summary["dry_run_report_path"] == "dry_run_report.json"
    assert summary["review_manifest_path"] == "review_manifest.json"
    assert summary["admission_request_path"] == "custom_corpus_admission.draft.json"
    assert summary["formal_package_validation_path"] == "custom_corpus_admission_package_validation.json"
    assert summary["property_package_binding_summary_path"] == "property_package_binding_summary.json"
    assert summary["materialization_plan_draft_path"] == "custom_corpus_materialization.draft.json"
    assert summary["materialization_plan_draft_summary_path"] == "property_materialization_plan_draft_summary.json"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["dry_run_id"] == "custom-dry-run-example-001"
    assert summary["review_manifest_id"] == "review-example-001"
    assert summary["admission_request_id"] == "property-admission-draft-001"
    assert summary["materialization_plan_id"] == "property-materialization-plan-001"
    assert summary["review_queue_id"] == "property-review-queue-001"
    assert summary["property_candidate_manifest_id"] == "property-candidates-001"
    assert summary["dataset_target"] == "example-candidate-target"
    assert summary["package_binding_status"] == "passed"
    assert summary["formal_package_validation_status"] == "passed"
    assert summary["materialization_draft_status"] == "written"
    assert summary["materialization_decision"] == "planned"
    assert summary["dry_run_decision"] == "passed"
    assert summary["phase1_status"] == "not_run"
    assert summary["training_admitted"] is False
    assert summary["admission_record_count"] == 2
    assert summary["admit_count"] == 1
    assert summary["exclude_count"] == 1
    assert summary["blocked_record_count"] == 1
    assert summary["materialization_record_count"] == 1
    assert summary["materialization_record_ids"] == ["property-materialization-plan-001-property-candidate-001"]
    assert summary["admit_record_ids"] == ["property-candidate-001"]
    assert summary["exclude_record_ids"] == ["property-candidate-002"]
    assert summary["blocked_record_ids"] == ["property-candidate-003"]
    assert summary["preflight_errors"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"


def test_package_binding_needs_review_returns_needs_review_by_default(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path, package_binding_status="needs_review")

    summary = preflight_property_materialization_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "needs_review"
    assert summary["preflight_errors"] == []
    assert "package_binding_needs_review" in summary["warnings"]


def test_package_binding_needs_review_fails_when_required_passed(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path, package_binding_status="needs_review")

    summary = preflight_property_materialization_plan(**_kwargs(paths), require_package_binding_passed=True)

    assert summary["preflight_status"] == "failed"
    assert "package_binding_needs_review" in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("property_package_binding_summary", lambda payload: payload.__setitem__("binding_status", "failed"), "package_binding_failed"),
        ("formal_package_validation", lambda payload: payload.__setitem__("validation_status", "failed"), "formal_package_validation_failed"),
        ("formal_package_validation", lambda payload: payload.__setitem__("schema_version", "wrong"), "formal_package_validation_schema_invalid"),
        ("materialization_plan_draft", lambda payload: payload.__setitem__("schema_version", "wrong"), "materialization_plan_schema_invalid"),
        (
            "materialization_plan_draft_summary",
            lambda payload: payload.__setitem__("schema_version", "wrong"),
            "materialization_draft_summary_schema_invalid",
        ),
        ("materialization_plan_draft_summary", lambda payload: payload.__setitem__("draft_status", "blocked"), "materialization_draft_not_written"),
        (
            "materialization_plan_draft",
            lambda payload: payload.__setitem__("materialization_decision", "blocked"),
            "materialization_decision_not_planned",
        ),
        (
            "materialization_plan_draft",
            lambda payload: payload.__setitem__("package_admission_decision", "ineligible"),
            "package_admission_not_eligible",
        ),
        ("dry_run_report", lambda payload: payload.__setitem__("decision", "failed"), "dry_run_not_passed"),
        (
            "dry_run_report",
            lambda payload: payload["confirmation_boundary"].__setitem__("phase1_status", "success"),
            "dry_run_phase1_ran",
        ),
        (
            "dry_run_report",
            lambda payload: payload["confirmation_boundary"].__setitem__("training_dataset_admitted", True),
            "dry_run_training_admitted",
        ),
        (
            "dry_run_report",
            lambda payload: payload["confirmation_boundary"].__setitem__("dataset_confirmation_confirmed", True),
            "dry_run_dataset_confirmed",
        ),
        ("materialization_plan_draft_summary", lambda payload: payload.__setitem__("manifest_sha256", "sha256:" + "0" * 64), "manifest_sha256_mismatch"),
        (
            "materialization_plan_draft_summary",
            lambda payload: payload.__setitem__("dry_run_report_sha256", "sha256:" + "1" * 64),
            "dry_run_report_sha256_mismatch",
        ),
        (
            "materialization_plan_draft_summary",
            lambda payload: payload.__setitem__("review_manifest_sha256", "sha256:" + "2" * 64),
            "review_manifest_sha256_mismatch",
        ),
        (
            "materialization_plan_draft_summary",
            lambda payload: payload.__setitem__("admission_request_sha256", "sha256:" + "3" * 64),
            "admission_request_sha256_mismatch",
        ),
        (
            "materialization_plan_draft_summary",
            lambda payload: payload.__setitem__("formal_package_validation_sha256", "sha256:" + "4" * 64),
            "formal_package_validation_sha256_mismatch",
        ),
        (
            "materialization_plan_draft_summary",
            lambda payload: payload.__setitem__("materialization_plan_draft_sha256", "sha256:" + "5" * 64),
            "materialization_plan_draft_sha256_mismatch",
        ),
        ("materialization_plan_draft_summary", lambda payload: payload.__setitem__("corpus_id", "other-corpus"), "corpus_id_mismatch"),
        ("materialization_plan_draft_summary", lambda payload: payload.__setitem__("dry_run_id", "other-run"), "dry_run_id_mismatch"),
        (
            "materialization_plan_draft_summary",
            lambda payload: payload.__setitem__("review_manifest_id", "other-review"),
            "review_manifest_id_mismatch",
        ),
        (
            "materialization_plan_draft_summary",
            lambda payload: payload.__setitem__("admission_request_id", "other-admission"),
            "admission_request_id_mismatch",
        ),
        (
            "materialization_plan_draft_summary",
            lambda payload: payload.__setitem__("materialization_plan_id", "other-plan"),
            "materialization_plan_id_mismatch",
        ),
    ],
)
def test_consistency_failures_return_failed(
    tmp_path: Path,
    target: str,
    mutator: object,
    error_code: str,
) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = preflight_property_materialization_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "failed"
    assert error_code in summary["preflight_errors"]


def test_admission_request_with_no_admitted_records_fails(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path, no_admit_records=True)

    summary = preflight_property_materialization_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "failed"
    assert "no_admitted_records" in summary["preflight_errors"]


def test_materialization_draft_with_no_materialization_records_fails(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths["materialization_plan_draft"], lambda payload: payload.__setitem__("materialization_records", []))
    _mutate_json(paths["materialization_plan_draft_summary"], lambda payload: payload.__setitem__("materialization_record_count", 0))

    summary = preflight_property_materialization_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "failed"
    assert "no_materialization_records" in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("record_mutator", "error_code"),
    [
        (
            lambda record: (
                record.__setitem__("record_id", "property-candidate-002"),
                record.__setitem__("admission_action", "exclude"),
            ),
            "materialization_record_from_excluded_record",
        ),
        (lambda record: record.__setitem__("record_id", "property-candidate-003"), "materialization_record_from_blocked_record"),
        (
            lambda record: (
                record.__setitem__("record_id", "property-candidate-004"),
                record.__setitem__("review_decision", "needs_review"),
                record.__setitem__("admission_action", "needs_review"),
            ),
            "materialization_record_from_needs_review_record",
        ),
    ],
)
def test_invalid_materialization_record_sources_fail(tmp_path: Path, record_mutator: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path, include_needs_review=True)
    _mutate_json(paths["materialization_plan_draft"], lambda payload: record_mutator(payload["materialization_records"][0]))  # type: ignore[index]

    summary = preflight_property_materialization_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "failed"
    assert error_code in summary["preflight_errors"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("materialization_plan_draft_summary", lambda payload: payload.__setitem__("materialization_record_count", 2), "materialization_record_count_mismatch"),
        (
            "materialization_plan_draft_summary",
            lambda payload: payload.__setitem__("materialization_record_ids", ["other-record"]),
            "materialization_record_ids_mismatch",
        ),
    ],
)
def test_materialization_record_summary_mismatches_fail(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = preflight_property_materialization_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "failed"
    assert error_code in summary["preflight_errors"]


def test_summary_uses_safe_basenames_only_and_no_temp_paths(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)

    summary = preflight_property_materialization_plan(**_kwargs(paths))
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["manifest_path"] == "manifest.json"
    assert summary["materialization_plan_draft_path"] == "custom_corpus_materialization.draft.json"
    assert str(tmp_path) not in serialized


@pytest.mark.parametrize(
    "target",
    [
        "formal_package_validation",
        "property_package_binding_summary",
        "materialization_plan_draft_summary",
    ],
)
def test_invalid_schema_in_inputs_fails_safely(tmp_path: Path, target: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__("schema_version", "wrong"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)
    output = json.loads(stdout.getvalue())

    assert code == 1
    assert output["preflight_status"] == "failed"
    assert "schema" in " ".join(output["preflight_errors"])
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


@pytest.mark.parametrize("target", ["materialization_plan_draft", "admission_request"])
def test_invalid_inputs_exit_1_without_leaking_sensitive_values(tmp_path: Path, target: str) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_preflight_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_materialization_plan_preflight._contains_forbidden_material",
        lambda value: True,
    )
    stdout = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=io.StringIO())
    summary = json.loads(stdout.getvalue())

    assert code == 1
    assert summary == {
        "schema_version": "custom_corpus_property_materialization_plan_preflight.v1",
        "preflight_status": "failed",
        "preflight_errors": ["property_materialization_plan_preflight_redaction_failed"],
        "redaction_status": "failed",
    }


def test_cli_stdout_is_valid_json_and_markdown_contains_boundary_statement(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    summary_path = tmp_path / "preflight-summary.json"
    markdown_path = tmp_path / "preflight-summary.md"
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        _cli_args(paths)
        + [
            "--output-summary",
            str(summary_path),
            "--output-markdown",
            str(markdown_path),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    summary = json.loads(stdout.getvalue())
    markdown = markdown_path.read_text(encoding="utf-8")

    assert code == 0
    assert summary["preflight_status"] == "passed"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    assert "this is a materialization plan preflight only" in markdown
    assert "offline materialization planner was not run" in markdown
    assert "No materializer was run" in markdown
    assert "No materialization was executed" in markdown
    assert "No candidate/training CSV was created" in markdown
    assert "No training data was admitted" in markdown
    assert "Phase 1 did not run" in markdown
    assert "DatasetConfirmation was not changed" in markdown
    assert stderr.getvalue() == ""


def test_no_planner_materializer_phase1_llm_mineru_pdf_or_parsed_document_calls(
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

    summary = preflight_property_materialization_plan(**_kwargs(paths))

    assert summary["preflight_status"] == "passed"
    assert not any("custom_corpus_materialization_planner" in name for name in imported_modules)
    assert not any(tmp_path.glob("**/*.csv"))
    assert not (tmp_path / "materialized_records.jsonl").exists()


def _kwargs(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "manifest_path": paths["manifest"],
        "dry_run_report_path": paths["dry_run_report"],
        "review_manifest_path": paths["review_manifest"],
        "admission_request_path": paths["admission_request"],
        "formal_package_validation_path": paths["formal_package_validation"],
        "property_package_binding_summary_path": paths["property_package_binding_summary"],
        "materialization_plan_draft_path": paths["materialization_plan_draft"],
        "materialization_plan_draft_summary_path": paths["materialization_plan_draft_summary"],
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
        "--materialization-plan-draft",
        str(paths["materialization_plan_draft"]),
        "--materialization-plan-draft-summary",
        str(paths["materialization_plan_draft_summary"]),
    ]


def _write_preflight_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
    no_admit_records: bool = False,
    include_needs_review: bool = False,
) -> dict[str, Path]:
    package_paths = _write_property_materialization_package(
        tmp_path,
        package_binding_status=package_binding_status,
        no_admit_records=no_admit_records,
        include_needs_review=include_needs_review,
    )
    draft_summary = build_property_materialization_plan_draft(
        manifest_path=package_paths["manifest"],
        dry_run_report_path=package_paths["dry_run_report"],
        review_manifest_path=package_paths["review_manifest"],
        admission_request_path=package_paths["admission_request"],
        formal_package_validation_path=package_paths["formal_package_validation"],
        property_package_binding_summary_path=package_paths["property_package_binding_summary"],
        output_dir=tmp_path / "property_materialization_plan_draft",
        materialization_plan_id="property-materialization-plan-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_materialization_plan_draft_output=True,
        allow_package_binding_needs_review=package_binding_status == "needs_review",
    )
    draft_dir = tmp_path / "property_materialization_plan_draft" / "property-materialization-plan-001"
    package_paths["materialization_plan_draft"] = draft_dir / "custom_corpus_materialization.draft.json"
    package_paths["materialization_plan_draft_summary"] = draft_dir / "property_materialization_plan_draft_summary.json"
    if no_admit_records:
        draft_dir.mkdir(parents=True, exist_ok=True)
        package_paths["materialization_plan_draft"].write_text(
            json.dumps(_materialization_plan_payload(package_paths), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_json(package_paths["materialization_plan_draft_summary"], _draft_summary_payload(package_paths))
    else:
        assert draft_summary["draft_status"] == "written"
    return package_paths


def _write_property_materialization_package(
    tmp_path: Path,
    *,
    package_binding_status: str,
    no_admit_records: bool = False,
    include_needs_review: bool = False,
) -> dict[str, Path]:
    manifest = _manifest_payload()
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = _sha256_file(manifest_path)

    dry_run = _dry_run_report_payload(manifest_sha=manifest_sha)
    dry_run_path = tmp_path / "dry_run_report.json"
    _write_json(dry_run_path, dry_run)
    dry_run_sha = _sha256_file(dry_run_path)

    review = _review_manifest_payload(manifest_sha=manifest_sha, dry_run_sha=dry_run_sha)
    if include_needs_review:
        review["review_records"].append(_review_record("review-record-004", "property-candidate-004", "needs_review", notes="needs reviewer"))  # type: ignore[index]
    review_path = tmp_path / "review_manifest.json"
    _write_json(review_path, review)
    review_sha = _sha256_file(review_path)

    admission = _admission_request_payload(manifest_sha=manifest_sha, dry_run_sha=dry_run_sha, review_sha=review_sha)
    if no_admit_records:
        admission["admission_records"] = [admission["admission_records"][1]]  # type: ignore[index]
    if include_needs_review:
        admission["admission_records"].append(  # type: ignore[index]
            _admission_record("property-candidate-004", "review-record-004", "needs_review", "needs_review", review_sha)
        )
    admission_path = tmp_path / "custom_corpus_admission.draft.json"
    _write_json(admission_path, admission)
    admission_sha = _sha256_file(admission_path)

    formal = _formal_package_validation_payload(
        manifest_sha=manifest_sha,
        dry_run_sha=dry_run_sha,
        review_sha=review_sha,
        admission_sha=admission_sha,
        admission_count=len(admission["admission_records"]),  # type: ignore[arg-type]
        admit_count=0 if no_admit_records else 1,
        exclude_count=1,
        needs_review_count=1 if include_needs_review else 0,
    )
    formal_path = tmp_path / "custom_corpus_admission_package_validation.json"
    _write_json(formal_path, formal)
    formal_sha = _sha256_file(formal_path)

    binding = _package_binding_summary_payload(
        manifest_sha=manifest_sha,
        dry_run_sha=dry_run_sha,
        review_sha=review_sha,
        admission_sha=admission_sha,
        formal_sha=formal_sha,
        binding_status=package_binding_status,
        admission_count=len(admission["admission_records"]),  # type: ignore[arg-type]
        admit_ids=[] if no_admit_records else ["property-candidate-001"],
        exclude_ids=["property-candidate-002"],
        blocked_ids=["property-candidate-003"],
    )
    binding_path = tmp_path / "property_package_binding_summary.json"
    _write_json(binding_path, binding)

    return {
        "manifest": manifest_path,
        "dry_run_report": dry_run_path,
        "review_manifest": review_path,
        "admission_request": admission_path,
        "formal_package_validation": formal_path,
        "property_package_binding_summary": binding_path,
    }


def _materialization_plan_payload(paths: dict[str, Path]) -> dict[str, object]:
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    dry_run = json.loads(paths["dry_run_report"].read_text(encoding="utf-8"))
    review = json.loads(paths["review_manifest"].read_text(encoding="utf-8"))
    admission = json.loads(paths["admission_request"].read_text(encoding="utf-8"))
    formal_sha = _sha256_file(paths["formal_package_validation"])
    return {
        "schema_version": "custom_corpus_materialization.v1",
        "materialization_plan_id": "property-materialization-plan-001",
        "materialization_run_id": "property-materialization-plan-001",
        "created_at": "2026-06-29T00:00:00Z",
        "created_by": "operator-redacted",
        "corpus_id": manifest["corpus_id"],
        "dry_run_id": dry_run["run_id"],
        "review_manifest_id": review["review_manifest_id"],
        "admission_request_id": admission["admission_request_id"],
        "materialization_mode": "candidate_only",
        "materialization_decision": "planned",
        "dataset_target": "example-candidate-target",
        "source_manifest_sha256": _sha256_file(paths["manifest"]),
        "source_dry_run_report_sha256": _sha256_file(paths["dry_run_report"]),
        "source_review_manifest_sha256": _sha256_file(paths["review_manifest"]),
        "source_admission_request_sha256": _sha256_file(paths["admission_request"]),
        "source_package_validation_sha256": formal_sha,
        "package_validation_status": "passed",
        "package_admission_decision": "eligible",
        "dry_run_phase1_status": "not_run",
        "dry_run_dataset_confirmation_confirmed": False,
        "dry_run_training_dataset_admitted": False,
        "confirmation": {
            "confirmed": True,
            "confirmed_by": "operator-redacted",
            "confirmed_at": "2026-06-29T00:00:00Z",
            "confirmation_source": "property-materialization-plan-draft-builder",
            "manifest_sha256": _sha256_file(paths["manifest"]),
            "dry_run_report_sha256": _sha256_file(paths["dry_run_report"]),
            "review_manifest_sha256": _sha256_file(paths["review_manifest"]),
            "admission_request_sha256": _sha256_file(paths["admission_request"]),
            "package_validation_sha256": formal_sha,
            "corpus_id": manifest["corpus_id"],
            "dry_run_id": dry_run["run_id"],
            "review_manifest_id": review["review_manifest_id"],
            "admission_request_id": admission["admission_request_id"],
            "reason": "operator confirmed reviewable materialization plan draft output",
        },
        "materialization_records": [],
        "rollback_policy": "delete generated candidate artifacts only",
        "redaction_policy": "redacted evidence only",
    }


def _draft_summary_payload(paths: dict[str, Path]) -> dict[str, object]:
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    dry_run = json.loads(paths["dry_run_report"].read_text(encoding="utf-8"))
    review = json.loads(paths["review_manifest"].read_text(encoding="utf-8"))
    admission = json.loads(paths["admission_request"].read_text(encoding="utf-8"))
    binding = json.loads(paths["property_package_binding_summary"].read_text(encoding="utf-8"))
    return {
        "schema_version": "custom_corpus_property_materialization_plan_draft_builder.v1",
        "draft_status": "written",
        "materialization_plan_id": "property-materialization-plan-001",
        "manifest_path": "manifest.json",
        "manifest_sha256": _sha256_file(paths["manifest"]),
        "dry_run_report_path": "dry_run_report.json",
        "dry_run_report_sha256": _sha256_file(paths["dry_run_report"]),
        "review_manifest_path": "review_manifest.json",
        "review_manifest_sha256": _sha256_file(paths["review_manifest"]),
        "admission_request_path": "custom_corpus_admission.draft.json",
        "admission_request_sha256": _sha256_file(paths["admission_request"]),
        "formal_package_validation_path": "custom_corpus_admission_package_validation.json",
        "formal_package_validation_sha256": _sha256_file(paths["formal_package_validation"]),
        "property_package_binding_summary_path": "property_package_binding_summary.json",
        "property_package_binding_summary_sha256": _sha256_file(paths["property_package_binding_summary"]),
        "corpus_id": manifest["corpus_id"],
        "dry_run_id": dry_run["run_id"],
        "review_manifest_id": review["review_manifest_id"],
        "admission_request_id": admission["admission_request_id"],
        "review_queue_id": binding["review_queue_id"],
        "property_candidate_manifest_id": binding["property_candidate_manifest_id"],
        "dataset_target": "example-candidate-target",
        "package_binding_status": binding["binding_status"],
        "formal_package_validation_status": "passed",
        "dry_run_decision": "passed",
        "phase1_status": "not_run",
        "training_admitted": False,
        "admission_record_count": len(admission["admission_records"]),
        "admit_count": 0,
        "exclude_count": 1,
        "blocked_record_count": 1,
        "materialization_record_count": 0,
        "materialization_record_ids": [],
        "admit_record_ids": [],
        "exclude_record_ids": ["property-candidate-002"],
        "blocked_record_ids": ["property-candidate-003"],
        "draft_artifacts": {
            "custom_corpus_materialization_draft_json": "custom_corpus_materialization.draft.json",
            "property_materialization_plan_draft_summary_json": "property_materialization_plan_draft_summary.json",
            "redacted_property_materialization_plan_draft_evidence_md": "redacted_property_materialization_plan_draft_evidence.md",
        },
        "draft_errors": [],
        "warnings": [],
        "redaction_status": "passed",
    }


def _mutate_json(path: Path, mutator: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)  # type: ignore[misc]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_manifest.v1",
        "corpus_id": "example-public-corpus",
        "corpus_class": "public_literature",
        "created_at": "2026-06-29T00:00:00Z",
        "created_by": "operator-redacted",
        "description": "safe public corpus fixture",
        "source_policy": "public-open-access-redacted",
        "default_redaction_policy": {
            "commit_raw_pdfs": False,
            "commit_parsed_documents": False,
            "commit_mineru_bundles": False,
            "commit_full_reports": False,
        },
        "documents": [
            {
                "document_id": "doc-example-001",
                "pdf_path": "redacted-input-a",
                "pdf_sha256": "",
                "title": "redacted public paper A",
                "doi": "",
                "source_url": "https://example.org/public-a",
                "license_or_access": "public",
                "provenance_note": "redacted provenance",
            }
        ],
    }


def _dry_run_report_payload(*, manifest_sha: str) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_dry_run.v1",
        "run_id": "custom-dry-run-example-001",
        "generated_at": "2026-06-29T00:00:00Z",
        "decision": "passed",
        "corpus_id": "example-public-corpus",
        "corpus_class": "public_literature",
        "manifest_summary": {
            "manifest_path": "manifest.json",
            "manifest_sha256": manifest_sha,
            "document_count": 1,
            "pdf_hash_coverage": {"with_sha256": 0, "without_sha256": 1},
            "source_policy": "public-open-access-redacted",
            "redaction_policy": {
                "commit_raw_pdfs": False,
                "commit_parsed_documents": False,
                "commit_mineru_bundles": False,
                "commit_full_reports": False,
            },
            "documents": ["doc-example-001"],
        },
        "confirmation_boundary": {
            "dataset_confirmation_confirmed": False,
            "phase1_status": "not_run",
            "training_dataset_admitted": False,
        },
    }


def _review_manifest_payload(*, manifest_sha: str, dry_run_sha: str) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_review.v1",
        "review_manifest_id": "review-example-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "created_at": "2026-06-29T00:00:00Z",
        "created_by": "reviewer-redacted",
        "source_dry_run_report_sha256": dry_run_sha,
        "source_manifest_sha256": manifest_sha,
        "review_policy": "example-property-review-policy",
        "review_records": [
            _review_record("review-record-001", "property-candidate-001", "accept"),
            _review_record(
                "review-record-002",
                "property-candidate-002",
                "reject",
                rejection_reason="reviewer rejected this numeric value",
            ),
        ],
    }


def _review_record(
    review_id: str,
    record_id: str,
    decision: str,
    *,
    rejection_reason: str = "",
    notes: str = "",
) -> dict[str, object]:
    return {
        "review_id": review_id,
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "document_id": "doc-example-001",
        "record_id": record_id,
        "field_name": "plqy",
        "review_scope": "record",
        "decision": decision,
        "rejection_reason": rejection_reason,
        "reviewer_label": "reviewer-redacted",
        "reviewed_at": "2026-06-29T00:00:00Z",
        "source_artifact_sha256": "sha256:" + "c" * 64,
        "extracted_value_summary": "short extracted value summary",
        "normalized_value_summary": "short normalized value summary",
        "confidence_note": "needs second reviewer" if decision == "needs_review" else "",
        "provenance_note": "short provenance summary",
        "notes": notes,
    }


def _admission_request_payload(*, manifest_sha: str, dry_run_sha: str, review_sha: str) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_admission.v1",
        "admission_request_id": "property-admission-draft-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "created_at": "2026-06-29T00:00:00Z",
        "created_by": "operator-redacted",
        "source_manifest_sha256": manifest_sha,
        "source_dry_run_report_sha256": dry_run_sha,
        "source_review_manifest_sha256": review_sha,
        "review_manifest_id": "review-example-001",
        "admission_policy": "draft-property-admission-request-from-plan",
        "dataset_target": "example-candidate-target",
        "admission_records": [
            _admission_record("property-candidate-001", "review-record-001", "accept", "admit", review_sha),
            _admission_record("property-candidate-002", "review-record-002", "reject", "exclude", review_sha),
        ],
    }


def _admission_record(record_id: str, review_id: str, review_decision: str, action: str, review_sha: str) -> dict[str, object]:
    return {
        "admission_record_id": f"property-admission-draft-001-{record_id}",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "review_manifest_id": "review-example-001",
        "document_id": "doc-example-001",
        "record_id": record_id,
        "field_name": "plqy",
        "admission_scope": "record",
        "review_id": review_id,
        "review_decision": review_decision,
        "action": action,
        "admission_reason": "draft request generated from property admission request plan" if action == "admit" else "",
        "exclusion_reason": "draft request generated from property admission request plan" if action == "exclude" else "",
        "source_artifact_sha256": "sha256:" + "c" * 64,
        "review_artifact_sha256": review_sha,
        "provenance_summary": "short provenance summary" if action == "admit" else "",
        "normalized_value_summary": "short normalized value summary" if action == "admit" else "",
        "notes": "still needs review" if action == "needs_review" else "draft only",
    }


def _formal_package_validation_payload(
    *,
    manifest_sha: str,
    dry_run_sha: str,
    review_sha: str,
    admission_sha: str,
    admission_count: int,
    admit_count: int,
    exclude_count: int,
    needs_review_count: int,
) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_admission_package_validation.v1",
        "validation_status": "passed",
        "admission_decision": "eligible",
        "manifest_path": "manifest.json",
        "dry_run_report_path": "dry_run_report.json",
        "review_manifest_path": "review_manifest.json",
        "admission_request_path": "custom_corpus_admission.draft.json",
        "manifest_sha256": manifest_sha,
        "dry_run_report_sha256": dry_run_sha,
        "review_manifest_sha256": review_sha,
        "admission_request_sha256": admission_sha,
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "review_manifest_id": "review-example-001",
        "admission_request_id": "property-admission-draft-001",
        "corpus_class": "public_literature",
        "document_count": 1,
        "dry_run_decision": "passed",
        "dry_run_phase1_status": "not_run",
        "dry_run_dataset_confirmation_confirmed": False,
        "dry_run_training_dataset_admitted": False,
        "review_record_count": 2 + needs_review_count,
        "admission_record_count": admission_count,
        "admit_count": admit_count,
        "exclude_count": exclude_count,
        "needs_review_count": needs_review_count,
        "matched_review_record_count": admission_count,
        "missing_review_record_count": 0,
        "binding_errors": [],
        "warnings": [],
    }


def _package_binding_summary_payload(
    *,
    manifest_sha: str,
    dry_run_sha: str,
    review_sha: str,
    admission_sha: str,
    formal_sha: str,
    binding_status: str,
    admission_count: int,
    admit_ids: list[str],
    exclude_ids: list[str],
    blocked_ids: list[str],
) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_property_package_binding.v1",
        "binding_status": binding_status,
        "binding_run_id": "property-package-binding-001",
        "manifest_path": "manifest.json",
        "manifest_sha256": manifest_sha,
        "dry_run_report_path": "dry_run_report.json",
        "dry_run_report_sha256": dry_run_sha,
        "review_manifest_path": "review_manifest.json",
        "review_manifest_sha256": review_sha,
        "admission_request_path": "custom_corpus_admission.draft.json",
        "admission_request_sha256": admission_sha,
        "property_precheck_summary_path": "property_precheck_summary.json",
        "property_precheck_summary_sha256": "sha256:" + "b" * 64,
        "formal_package_validation_path": "custom_corpus_admission_package_validation.json",
        "formal_package_validation_sha256": formal_sha,
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "review_manifest_id": "review-example-001",
        "admission_request_id": "property-admission-draft-001",
        "review_queue_id": "property-review-queue-001",
        "property_candidate_manifest_id": "property-candidates-001",
        "property_precheck_status": "passed",
        "formal_package_validation_status": "passed",
        "dry_run_decision": "passed",
        "phase1_status": "not_run",
        "training_admitted": False,
        "admission_record_count": admission_count,
        "admit_count": len(admit_ids),
        "exclude_count": len(exclude_ids),
        "blocked_record_count": len(blocked_ids),
        "admit_record_ids": admit_ids,
        "exclude_record_ids": exclude_ids,
        "blocked_record_ids": blocked_ids,
        "binding_errors": [] if binding_status != "failed" else ["property_package_binding_failed"],
        "warnings": [] if binding_status == "passed" else ["package_binding_needs_review"],
        "redaction_status": "passed",
    }
