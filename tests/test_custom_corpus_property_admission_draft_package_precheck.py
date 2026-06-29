from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_admission_draft_package_precheck import (
    main,
    precheck_property_admission_draft_package,
)


def test_valid_package_returns_passed(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)

    summary = precheck_property_admission_draft_package(**_kwargs(paths))

    assert summary["schema_version"] == "custom_corpus_property_admission_draft_package_precheck.v1"
    assert summary["precheck_status"] == "passed"
    assert summary["manifest_path"] == "custom_corpus_manifest.json"
    assert summary["dry_run_report_path"] == "dry_run_report.json"
    assert summary["review_manifest_path"] == "property_review_manifest.json"
    assert summary["admission_draft_path"] == "custom_corpus_admission.draft.json"
    assert summary["draft_summary_path"] == "property_admission_draft_summary.json"
    assert summary["request_plan_summary_path"] == "property_admission_request_plan_summary.json"
    assert summary["readiness_summary_path"] == "property_admission_readiness_summary.json"
    assert summary["review_binding_summary_path"] == "property_review_binding_summary.json"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["dry_run_id"] == "custom-dry-run-example-001"
    assert summary["review_manifest_id"] == "property-review-manifest-001"
    assert summary["admission_request_id"] == "property-admission-draft-001"
    assert summary["review_queue_id"] == "property-review-queue-001"
    assert summary["property_candidate_manifest_id"] == "property-candidates-001"
    assert summary["dry_run_decision"] == "passed"
    assert summary["phase1_status"] == "not_run"
    assert summary["training_admitted"] is False
    assert summary["draft_status"] == "written"
    assert summary["planner_status"] == "planned"
    assert summary["readiness_status"] == "ready"
    assert summary["binding_status"] == "passed"
    assert summary["draft_record_count"] == 2
    assert summary["admit_count"] == 1
    assert summary["exclude_count"] == 1
    assert summary["blocked_record_count"] == 1
    assert summary["admit_record_ids"] == ["property-candidate-001"]
    assert summary["exclude_record_ids"] == ["property-candidate-002"]
    assert summary["blocked_record_ids"] == ["property-candidate-003"]
    assert summary["precheck_errors"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"


def test_partial_request_plan_returns_needs_review_without_strict_flag(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path, request_plan_status="partial")

    summary = precheck_property_admission_draft_package(**_kwargs(paths))

    assert summary["precheck_status"] == "needs_review"
    assert summary["warnings"] == ["request_plan_partial"]


def test_partial_request_plan_fails_with_require_planned_request(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path, request_plan_status="partial")

    summary = precheck_property_admission_draft_package(**_kwargs(paths), require_planned_request=True)

    assert summary["precheck_status"] == "failed"
    assert "request_plan_not_planned" in summary["precheck_errors"]


def test_partial_readiness_returns_needs_review_without_strict_flag(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path, readiness_status="partial")

    summary = precheck_property_admission_draft_package(**_kwargs(paths))

    assert summary["precheck_status"] == "needs_review"
    assert summary["warnings"] == ["readiness_partial"]


def test_partial_readiness_fails_with_require_ready_readiness(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path, readiness_status="partial")

    summary = precheck_property_admission_draft_package(**_kwargs(paths), require_ready_readiness=True)

    assert summary["precheck_status"] == "failed"
    assert "readiness_not_ready" in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("review_binding_summary", lambda payload: payload.__setitem__("binding_status", "failed"), "review_binding_failed"),
        ("readiness_summary", lambda payload: payload.__setitem__("readiness_status", "blocked"), "readiness_blocked"),
        ("request_plan_summary", lambda payload: payload.__setitem__("planner_status", "blocked"), "request_plan_blocked"),
        ("draft_summary", lambda payload: payload.__setitem__("draft_status", "blocked"), "draft_not_written"),
    ],
)
def test_blocked_or_failed_upstream_statuses_fail(
    tmp_path: Path,
    target: str,
    mutator: object,
    error_code: str,
) -> None:
    paths = _write_precheck_package(tmp_path, mutate={target: mutator})

    summary = precheck_property_admission_draft_package(**_kwargs(paths))

    assert summary["precheck_status"] == "failed"
    assert error_code in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        (
            "draft_summary",
            lambda payload: payload.__setitem__("draft_admit_record_ids", ["property-candidate-999"]),
            "draft_summary_record_ids_mismatch",
        ),
        (
            "request_plan_summary",
            lambda payload: payload.__setitem__("planned_admit_record_ids", ["property-candidate-999"]),
            "request_plan_draft_ids_mismatch",
        ),
        (
            "readiness_summary",
            lambda payload: payload.__setitem__("planned_admission_candidate_record_ids", ["property-candidate-999"]),
            "readiness_request_plan_ids_mismatch",
        ),
        (
            "review_binding_summary",
            lambda payload: payload.__setitem__("reviewed_queue_record_ids", ["property-candidate-001"]),
            "review_binding_missing_admission_records",
        ),
        (
            "review_binding_summary",
            lambda payload: payload.__setitem__("reviewed_blocked_record_ids", ["property-candidate-001"]),
            "reviewed_blocked_record_in_admission_draft",
        ),
        (
            "review_binding_summary",
            lambda payload: payload.__setitem__("unknown_review_record_ids", ["property-candidate-001"]),
            "unknown_review_record_in_admission_draft",
        ),
    ],
)
def test_record_id_consistency_failures(
    tmp_path: Path,
    target: str,
    mutator: object,
    error_code: str,
) -> None:
    paths = _write_precheck_package(tmp_path, mutate={target: mutator})

    summary = precheck_property_admission_draft_package(**_kwargs(paths))

    assert summary["precheck_status"] == "failed"
    assert error_code in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("admission_draft", lambda payload: payload.__setitem__("corpus_id", "other-corpus"), "corpus_id_mismatch"),
        ("admission_draft", lambda payload: payload.__setitem__("dry_run_id", "other-dry-run"), "dry_run_id_mismatch"),
        (
            "admission_draft",
            lambda payload: payload.__setitem__("review_manifest_id", "other-review"),
            "review_manifest_id_mismatch",
        ),
        (
            "admission_draft",
            lambda payload: payload.__setitem__("source_manifest_sha256", "sha256:" + "9" * 64),
            "source_manifest_sha256_mismatch",
        ),
        (
            "admission_draft",
            lambda payload: payload.__setitem__("source_dry_run_report_sha256", "sha256:" + "8" * 64),
            "source_dry_run_report_sha256_mismatch",
        ),
        (
            "admission_draft",
            lambda payload: payload.__setitem__("source_review_manifest_sha256", "sha256:" + "7" * 64),
            "source_review_manifest_sha256_mismatch",
        ),
    ],
)
def test_cross_artifact_id_and_hash_mismatches_fail(
    tmp_path: Path,
    target: str,
    mutator: object,
    error_code: str,
) -> None:
    paths = _write_precheck_package(tmp_path, mutate={target: mutator})

    summary = precheck_property_admission_draft_package(**_kwargs(paths))

    assert summary["precheck_status"] == "failed"
    assert error_code in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda payload: payload.__setitem__("admission_records", []), "invalid_admission_draft"),
        (
            lambda payload: payload["admission_records"][0].__setitem__("review_decision", "needs_review"),
            "invalid_admission_draft",
        ),
        (
            lambda payload: payload["admission_records"][0].__setitem__("review_decision", "reject"),
            "invalid_admission_draft",
        ),
        (
            lambda payload: payload["admission_records"][1].__setitem__("review_decision", "accept"),
            "invalid_admission_draft",
        ),
    ],
)
def test_invalid_admission_draft_rules_fail(tmp_path: Path, mutator: object, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path, mutate={"admission_draft": mutator})

    summary = precheck_property_admission_draft_package(**_kwargs(paths))

    assert summary["precheck_status"] == "failed"
    assert error_code in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda payload: payload["confirmation_boundary"].__setitem__("phase1_status", "success"), "dry_run_phase1_ran"),
        (
            lambda payload: payload["confirmation_boundary"].__setitem__("training_dataset_admitted", True),
            "dry_run_training_admitted",
        ),
        (
            lambda payload: payload["confirmation_boundary"].__setitem__("dataset_confirmation_confirmed", True),
            "dry_run_dataset_confirmed",
        ),
    ],
)
def test_dry_run_boundary_failures_fail(tmp_path: Path, mutator: object, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path, mutate={"dry_run_report": mutator})

    summary = precheck_property_admission_draft_package(**_kwargs(paths))

    assert summary["precheck_status"] == "failed"
    assert error_code in summary["precheck_errors"]


def test_summary_uses_safe_basenames_and_no_temp_paths(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)

    summary = precheck_property_admission_draft_package(**_kwargs(paths))
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["manifest_path"] == "custom_corpus_manifest.json"
    assert summary["dry_run_report_path"] == "dry_run_report.json"
    assert summary["admission_draft_path"] == "custom_corpus_admission.draft.json"
    assert str(tmp_path) not in serialized


@pytest.mark.parametrize(
    "target",
    [
        "manifest",
        "dry_run_report",
        "review_manifest",
        "admission_draft",
        "draft_summary",
        "request_plan_summary",
        "readiness_summary",
        "review_binding_summary",
    ],
)
def test_invalid_schema_in_each_input_fails_safely(tmp_path: Path, target: str) -> None:
    paths = _write_precheck_package(tmp_path)
    payload = json.loads(paths[target].read_text(encoding="utf-8"))
    payload["schema_version"] = "wrong"
    paths[target].write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)

    assert code == 1
    assert "invalid" in stderr.getvalue().lower() or json.loads(stdout.getvalue())["precheck_status"] == "failed"
    assert str(tmp_path) not in stderr.getvalue()


def test_invalid_review_manifest_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    payload = json.loads(paths["review_manifest"].read_text(encoding="utf-8"))
    payload["review_records"][0]["notes"] = "password abc123"
    paths["review_manifest"].write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)

    assert code == 1
    assert stdout.getvalue() == ""
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_invalid_admission_draft_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    payload = json.loads(paths["admission_draft"].read_text(encoding="utf-8"))
    payload["admission_records"][0]["notes"] = "token abc123"
    paths["admission_draft"].write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)

    assert code == 1
    assert stdout.getvalue() == ""
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_admission_draft_package_precheck._contains_forbidden_material",
        lambda value: True,
    )
    stdout = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=io.StringIO())
    summary = json.loads(stdout.getvalue())

    assert code == 1
    assert summary == {
        "schema_version": "custom_corpus_property_admission_draft_package_precheck.v1",
        "precheck_status": "failed",
        "precheck_errors": ["property_admission_draft_package_precheck_redaction_failed"],
        "redaction_status": "failed",
    }


def test_cli_stdout_is_valid_json_and_markdown_contains_boundary_statement(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    output_summary = tmp_path / "precheck_summary.json"
    output_markdown = tmp_path / "precheck_summary.md"
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        _cli_args(paths) + ["--output-summary", str(output_summary), "--output-markdown", str(output_markdown)],
        stdout=stdout,
        stderr=stderr,
    )
    printed = json.loads(stdout.getvalue())
    written = json.loads(output_summary.read_text(encoding="utf-8"))
    markdown = output_markdown.read_text(encoding="utf-8")

    assert code == 0
    assert printed == written
    assert printed["precheck_status"] == "passed"
    assert "formal package binding was not run" in markdown
    assert "No `custom_corpus_admission_package_validation.v1` was created" in markdown
    assert "No materialization was run" in markdown
    assert "No candidate/training CSV was created" in markdown
    assert "Phase 1 did not run" in markdown
    assert "DatasetConfirmation was not changed" in markdown
    assert "No training data was admitted" in markdown
    assert str(tmp_path) not in markdown
    assert stderr.getvalue() == ""


def test_no_formal_package_binding_materialization_csv_phase1_or_dataset_confirmation_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_precheck_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_admission_package",
            "ai4s_agent.custom_corpus_materialization",
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

    summary = precheck_property_admission_draft_package(**_kwargs(paths))

    assert summary["precheck_status"] == "passed"
    assert not any("custom_corpus_admission_package" in name for name in imported_modules)
    assert not any(tmp_path.glob("*.csv"))
    assert not (tmp_path / "custom_corpus_admission_package_validation.json").exists()
    assert not (tmp_path / "materialization_plan.json").exists()


def _kwargs(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "manifest_path": paths["manifest"],
        "dry_run_report_path": paths["dry_run_report"],
        "review_manifest_path": paths["review_manifest"],
        "admission_draft_path": paths["admission_draft"],
        "draft_summary_path": paths["draft_summary"],
        "request_plan_summary_path": paths["request_plan_summary"],
        "readiness_summary_path": paths["readiness_summary"],
        "review_binding_summary_path": paths["review_binding_summary"],
    }


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--manifest",
        str(paths["manifest"]),
        "--dry-run-report",
        str(paths["dry_run_report"]),
        "--review-manifest",
        str(paths["review_manifest"]),
        "--admission-draft",
        str(paths["admission_draft"]),
        "--draft-summary",
        str(paths["draft_summary"]),
        "--request-plan-summary",
        str(paths["request_plan_summary"]),
        "--readiness-summary",
        str(paths["readiness_summary"]),
        "--review-binding-summary",
        str(paths["review_binding_summary"]),
    ]


def _write_precheck_package(
    tmp_path: Path,
    *,
    request_plan_status: str = "planned",
    readiness_status: str = "ready",
    mutate: dict[str, object] | None = None,
) -> dict[str, Path]:
    mutate = mutate or {}
    manifest = _manifest_payload()
    _apply_mutation(manifest, mutate.get("manifest"))
    manifest_path = tmp_path / "custom_corpus_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = _sha256_file(manifest_path)

    dry_run = _dry_run_report_payload(manifest_sha=manifest_sha)
    _apply_mutation(dry_run, mutate.get("dry_run_report"))
    dry_run_path = tmp_path / "dry_run_report.json"
    _write_json(dry_run_path, dry_run)
    dry_run_sha = _sha256_file(dry_run_path)

    review = _review_manifest_payload(manifest_sha=manifest_sha, dry_run_sha=dry_run_sha)
    _apply_mutation(review, mutate.get("review_manifest"))
    review_path = tmp_path / "property_review_manifest.json"
    _write_json(review_path, review)
    review_sha = _sha256_file(review_path)

    admission = _admission_draft_payload(manifest_sha=manifest_sha, dry_run_sha=dry_run_sha, review_sha=review_sha)
    _apply_mutation(admission, mutate.get("admission_draft"))
    admission_path = tmp_path / "custom_corpus_admission.draft.json"
    _write_json(admission_path, admission)

    draft_summary = _draft_summary_payload(
        manifest_sha=manifest_sha,
        dry_run_sha=dry_run_sha,
        request_plan_status=request_plan_status,
    )
    _apply_mutation(draft_summary, mutate.get("draft_summary"))
    draft_summary_path = tmp_path / "property_admission_draft_summary.json"
    _write_json(draft_summary_path, draft_summary)

    request_plan = _request_plan_payload(
        manifest_sha=manifest_sha,
        dry_run_sha=dry_run_sha,
        review_sha=review_sha,
        request_plan_status=request_plan_status,
    )
    _apply_mutation(request_plan, mutate.get("request_plan_summary"))
    request_plan_path = tmp_path / "property_admission_request_plan_summary.json"
    _write_json(request_plan_path, request_plan)

    readiness = _readiness_payload(
        manifest_sha=manifest_sha,
        dry_run_sha=dry_run_sha,
        review_sha=review_sha,
        readiness_status=readiness_status,
    )
    _apply_mutation(readiness, mutate.get("readiness_summary"))
    readiness_path = tmp_path / "property_admission_readiness_summary.json"
    _write_json(readiness_path, readiness)

    review_binding = _review_binding_payload(review_sha=review_sha, manifest_sha=manifest_sha, dry_run_sha=dry_run_sha)
    _apply_mutation(review_binding, mutate.get("review_binding_summary"))
    review_binding_path = tmp_path / "property_review_binding_summary.json"
    _write_json(review_binding_path, review_binding)

    return {
        "manifest": manifest_path,
        "dry_run_report": dry_run_path,
        "review_manifest": review_path,
        "admission_draft": admission_path,
        "draft_summary": draft_summary_path,
        "request_plan_summary": request_plan_path,
        "readiness_summary": readiness_path,
        "review_binding_summary": review_binding_path,
    }


def _apply_mutation(payload: dict[str, object], mutation: object | None) -> None:
    if mutation is not None:
        mutation(payload)  # type: ignore[misc]


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
        "decision": "passed",
        "corpus_id": "example-public-corpus",
        "corpus_class": "public_literature",
        "manifest_summary": {
            "manifest_path": "custom_corpus_manifest.json",
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
        "review_manifest_id": "property-review-manifest-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "created_at": "2026-06-29T00:00:00Z",
        "created_by": "reviewer-redacted",
        "source_dry_run_report_sha256": dry_run_sha,
        "source_manifest_sha256": manifest_sha,
        "review_policy": "example-property-candidate-review-policy",
        "review_records": [
            _review_record("property-review-001", "property-candidate-001", "accept"),
            _review_record(
                "property-review-002",
                "property-candidate-002",
                "reject",
                rejection_reason="reviewer rejected this numeric value",
            ),
            _review_record(
                "property-review-003",
                "property-candidate-003",
                "needs_review",
                document_id="doc-example-002",
                confidence_note="unit requires reviewer",
                notes="needs unit review",
            ),
        ],
    }


def _review_record(
    review_id: str,
    record_id: str,
    decision: str,
    *,
    document_id: str = "doc-example-001",
    rejection_reason: str = "",
    confidence_note: str = "",
    notes: str = "",
) -> dict[str, object]:
    return {
        "review_id": review_id,
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "document_id": document_id,
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
        "confidence_note": confidence_note,
        "provenance_note": "short provenance summary",
        "notes": notes,
    }


def _admission_draft_payload(*, manifest_sha: str, dry_run_sha: str, review_sha: str) -> dict[str, object]:
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
        "review_manifest_id": "property-review-manifest-001",
        "admission_policy": "draft-property-admission-request-from-plan",
        "dataset_target": "example-candidate-target",
        "admission_records": [
            _admission_record("property-candidate-001", "property-review-001", "accept", "admit", review_sha),
            _admission_record("property-candidate-002", "property-review-002", "reject", "exclude", review_sha),
        ],
    }


def _admission_record(
    record_id: str,
    review_id: str,
    review_decision: str,
    action: str,
    review_sha: str,
) -> dict[str, object]:
    return {
        "admission_record_id": f"property-admission-draft-001-{record_id}",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "review_manifest_id": "property-review-manifest-001",
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
        "notes": "draft only",
    }


def _draft_summary_payload(*, manifest_sha: str, dry_run_sha: str, request_plan_status: str) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_property_admission_draft_builder.v1",
        "draft_status": "written",
        "admission_request_id": "property-admission-draft-001",
        "review_queue_id": "property-review-queue-001",
        "property_candidate_manifest_id": "property-candidates-001",
        "review_manifest_id": "property-review-manifest-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "dataset_target": "example-candidate-target",
        "planner_status": request_plan_status,
        "allow_partial_plan": request_plan_status == "partial",
        "draft_record_count": 2,
        "draft_admit_count": 1,
        "draft_exclude_count": 1,
        "blocked_record_count": 1,
        "draft_admit_record_ids": ["property-candidate-001"],
        "draft_exclude_record_ids": ["property-candidate-002"],
        "blocked_record_ids": ["property-candidate-003"],
        "draft_artifacts": {"custom_corpus_admission_draft_json": "custom_corpus_admission.draft.json"},
        "draft_errors": [],
        "warnings": [],
        "source_manifest_sha256": manifest_sha,
        "source_dry_run_report_sha256": dry_run_sha,
        "redaction_status": "passed",
    }


def _request_plan_payload(
    *,
    manifest_sha: str,
    dry_run_sha: str,
    review_sha: str,
    request_plan_status: str,
) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_property_admission_request_plan.v1",
        "planner_status": request_plan_status,
        "review_queue_id": "property-review-queue-001",
        "property_candidate_manifest_id": "property-candidates-001",
        "review_manifest_id": "property-review-manifest-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "review_manifest_sha256": review_sha,
        "binding_status": "passed" if request_plan_status != "blocked" else "failed",
        "planned_admit_record_ids": ["property-candidate-001"],
        "planned_exclude_record_ids": ["property-candidate-002"],
        "blocked_record_ids": ["property-candidate-003"],
        "unreviewed_queue_record_ids": [],
        "readiness_errors": [],
        "planning_errors": [] if request_plan_status != "blocked" else ["request_plan_blocked"],
        "source_manifest_sha256": manifest_sha,
        "source_dry_run_report_sha256": dry_run_sha,
        "planned_record_summaries": [],
        "redaction_status": "passed",
    }


def _readiness_payload(*, manifest_sha: str, dry_run_sha: str, review_sha: str, readiness_status: str) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_property_admission_readiness.v1",
        "readiness_status": readiness_status,
        "review_manifest_sha256": review_sha,
        "review_queue_id": "property-review-queue-001",
        "property_candidate_manifest_id": "property-candidates-001",
        "review_manifest_id": "property-review-manifest-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "binding_status": "passed" if readiness_status != "blocked" else "failed",
        "planned_admission_candidate_record_ids": ["property-candidate-001"],
        "planned_exclusion_record_ids": ["property-candidate-002"],
        "blocked_from_admission_record_ids": ["property-candidate-003"],
        "unreviewed_queue_record_ids": [],
        "reviewed_blocked_record_ids": [],
        "unknown_review_record_ids": [],
        "readiness_errors": [] if readiness_status != "blocked" else ["readiness_blocked"],
        "warnings": [],
        "source_manifest_sha256": manifest_sha,
        "source_dry_run_report_sha256": dry_run_sha,
        "redaction_status": "passed",
    }


def _review_binding_payload(*, review_sha: str, manifest_sha: str, dry_run_sha: str) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_property_review_binding.v1",
        "binding_status": "passed",
        "review_queue_id": "property-review-queue-001",
        "property_candidate_manifest_id": "property-candidates-001",
        "review_manifest_id": "property-review-manifest-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "review_queue_sha256": "sha256:" + "e" * 64,
        "review_manifest_sha256": review_sha,
        "source_manifest_sha256": manifest_sha,
        "source_dry_run_report_sha256": dry_run_sha,
        "reviewed_queue_record_ids": ["property-candidate-001", "property-candidate-002", "property-candidate-003"],
        "unreviewed_queue_record_ids": [],
        "reviewed_blocked_record_ids": [],
        "unknown_review_record_ids": [],
        "binding_errors": [],
        "redaction_status": "passed",
    }
