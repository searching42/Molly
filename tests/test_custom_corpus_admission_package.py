from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_admission_package import (
    main,
    sha256_file,
    validate_admission_package,
)


def test_valid_package_passes(tmp_path: Path) -> None:
    paths = _write_package(tmp_path)

    summary = validate_admission_package(
        manifest_path=paths["manifest"],
        dry_run_report_path=paths["dry_run_report"],
        review_manifest_path=paths["review_manifest"],
        admission_request_path=paths["admission_request"],
    )

    assert summary["schema_version"] == "custom_corpus_admission_package_validation.v1"
    assert summary["validation_status"] == "passed"
    assert summary["admission_decision"] == "needs_review"
    assert summary["manifest_path"] == "manifest.json"
    assert summary["dry_run_report_path"] == "dry_run_report.json"
    assert summary["review_manifest_path"] == "review_manifest.json"
    assert summary["admission_request_path"] == "admission_request.json"
    assert summary["manifest_sha256"] == sha256_file(paths["manifest"])
    assert summary["dry_run_report_sha256"] == sha256_file(paths["dry_run_report"])
    assert summary["review_manifest_sha256"] == sha256_file(paths["review_manifest"])
    assert summary["admission_request_sha256"] == sha256_file(paths["admission_request"])
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["dry_run_id"] == "custom-dry-run-example-001"
    assert summary["review_manifest_id"] == "review-example-001"
    assert summary["admission_request_id"] == "admission-example-001"
    assert summary["corpus_class"] == "public_literature"
    assert summary["document_count"] == 2
    assert summary["dry_run_decision"] == "passed"
    assert summary["dry_run_phase1_status"] == "not_run"
    assert summary["dry_run_dataset_confirmation_confirmed"] is False
    assert summary["dry_run_training_dataset_admitted"] is False
    assert summary["review_record_count"] == 3
    assert summary["admission_record_count"] == 3
    assert summary["admit_count"] == 1
    assert summary["exclude_count"] == 1
    assert summary["needs_review_count"] == 1
    assert summary["matched_review_record_count"] == 3
    assert summary["missing_review_record_count"] == 0
    assert summary["binding_errors"] == []


def test_cli_prints_safe_summary_and_exits_0(tmp_path: Path) -> None:
    paths = _write_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)
    printed = json.loads(stdout.getvalue())

    assert code == 0
    assert printed["validation_status"] == "passed"
    assert printed["manifest_path"] == "manifest.json"
    assert str(tmp_path) not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_cli_writes_optional_summary(tmp_path: Path) -> None:
    paths = _write_package(tmp_path)
    output_summary = tmp_path / "package_summary.json"
    stdout = io.StringIO()

    code = main([*_cli_args(paths), "--output-summary", str(output_summary)], stdout=stdout, stderr=io.StringIO())

    assert code == 0
    assert json.loads(output_summary.read_text(encoding="utf-8")) == json.loads(stdout.getvalue())


@pytest.mark.parametrize(
    ("artifact_key", "mutate", "expected_error"),
    [
        ("admission_request", lambda payload: payload.__setitem__("source_manifest_sha256", "sha256:" + "0" * 64), "manifest_hash_mismatch"),
        ("admission_request", lambda payload: payload.__setitem__("source_dry_run_report_sha256", "sha256:" + "0" * 64), "dry_run_report_hash_mismatch"),
        ("admission_request", lambda payload: payload.__setitem__("source_review_manifest_sha256", "sha256:" + "0" * 64), "review_manifest_hash_mismatch"),
        ("dry_run_report", lambda payload: payload.__setitem__("corpus_id", "different-corpus"), "corpus_id_mismatch"),
        (
            "review_manifest",
            lambda payload: (
                payload.__setitem__("dry_run_id", "different-run"),
                [record.__setitem__("dry_run_id", "different-run") for record in payload["review_records"]],
            ),
            "dry_run_id_mismatch",
        ),
        (
            "admission_request",
            lambda payload: (
                payload.__setitem__("review_manifest_id", "different-review"),
                [record.__setitem__("review_manifest_id", "different-review") for record in payload["admission_records"]],
            ),
            "review_manifest_id_mismatch",
        ),
        ("dry_run_report", lambda payload: payload.__setitem__("decision", "failed"), "dry_run_not_passed"),
        ("dry_run_report", lambda payload: payload["confirmation_boundary"].__setitem__("phase1_status", "success"), "dry_run_phase1_ran"),
        ("dry_run_report", lambda payload: payload["confirmation_boundary"].__setitem__("dataset_confirmation_confirmed", True), "dry_run_dataset_confirmed"),
        ("dry_run_report", lambda payload: payload["confirmation_boundary"].__setitem__("training_dataset_admitted", True), "dry_run_training_admitted"),
        ("admission_request", lambda payload: payload["admission_records"][0].__setitem__("review_id", "missing-review"), "admission_review_record_missing"),
        (
            "review_manifest",
            lambda payload: (
                payload["review_records"][0].__setitem__("decision", "reject"),
                payload["review_records"][0].__setitem__("rejection_reason", "review rejected the value summary"),
            ),
            "review_decision_mismatch",
        ),
        ("admission_request", lambda payload: payload["admission_records"][0].__setitem__("source_artifact_sha256", "sha256:" + "9" * 64), "source_artifact_sha256_mismatch"),
        ("admission_request", lambda payload: payload["admission_records"][0].__setitem__("review_artifact_sha256", "sha256:" + "9" * 64), "review_artifact_sha256_mismatch"),
        (
            "admission_request",
            lambda payload: (
                payload["admission_records"][1].__setitem__("action", "admit"),
                payload["admission_records"][1].__setitem__("admission_reason", "incorrectly admitted"),
                payload["admission_records"][1].__setitem__("exclusion_reason", ""),
            ),
            "rejected_record_admitted",
        ),
        (
            "admission_request",
            lambda payload: (
                payload["admission_records"][2].__setitem__("action", "admit"),
                payload["admission_records"][2].__setitem__("admission_reason", "incorrectly admitted"),
                payload["admission_records"][2].__setitem__("notes", "still needs review"),
            ),
            "needs_review_record_admitted",
        ),
        ("admission_request", lambda payload: payload["admission_records"][0].__setitem__("provenance_summary", ""), "admitted_record_missing_provenance_summary"),
        ("admission_request", lambda payload: payload["admission_records"][0].__setitem__("normalized_value_summary", ""), "admitted_record_missing_normalized_value_summary"),
        ("admission_request", lambda payload: payload["admission_records"][0].__setitem__("admission_reason", ""), "admitted_record_missing_admission_reason"),
    ],
)
def test_binding_failures(tmp_path: Path, artifact_key: str, mutate: object, expected_error: str) -> None:
    paths = _write_package(tmp_path, mutate={artifact_key: mutate})

    summary = validate_admission_package(
        manifest_path=paths["manifest"],
        dry_run_report_path=paths["dry_run_report"],
        review_manifest_path=paths["review_manifest"],
        admission_request_path=paths["admission_request"],
    )

    assert summary["validation_status"] == "failed"
    assert expected_error in summary["binding_errors"]


def test_summary_does_not_include_manifest_pdf_path_or_temp_dir(tmp_path: Path) -> None:
    paths = _write_package(tmp_path)

    summary = validate_admission_package(
        manifest_path=paths["manifest"],
        dry_run_report_path=paths["dry_run_report"],
        review_manifest_path=paths["review_manifest"],
        admission_request_path=paths["admission_request"],
    )
    serialized = json.dumps(summary, sort_keys=True)

    assert "/private/input_a.pdf" not in serialized
    assert str(tmp_path) not in serialized


def test_invalid_artifact_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_package(tmp_path)
    paths["review_manifest"].write_text(
        json.dumps({"schema_version": "custom_corpus_review.v1", "review_manifest_id": "review-token-abc123"}),
        encoding="utf-8",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)

    assert code == 1
    output = stdout.getvalue() + stderr.getvalue()
    assert "abc123" not in output
    assert str(tmp_path) not in output
    assert "invalid_review_manifest" in output


def test_summary_redaction_fails_closed_for_summarized_private_path(tmp_path: Path) -> None:
    paths = _write_package(
        tmp_path,
        mutate={
            "dry_run_report": lambda payload: payload["confirmation_boundary"].__setitem__(
                "phase1_status", "/Users/private/operator/phase1"
            )
        },
    )

    summary = validate_admission_package(
        manifest_path=paths["manifest"],
        dry_run_report_path=paths["dry_run_report"],
        review_manifest_path=paths["review_manifest"],
        admission_request_path=paths["admission_request"],
    )
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["validation_status"] == "failed"
    assert summary["binding_errors"] == ["package_summary_redaction_failed"]
    assert "/Users/private/operator" not in serialized


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
    ]


def _write_package(tmp_path: Path, *, mutate: dict[str, object] | None = None) -> dict[str, Path]:
    mutate = mutate or {}
    manifest = _manifest_payload()
    _apply_mutation(manifest, mutate.get("manifest"))
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)

    dry_run = _dry_run_report_payload(manifest_sha=manifest_sha)
    _apply_mutation(dry_run, mutate.get("dry_run_report"))
    dry_run_path = tmp_path / "dry_run_report.json"
    _write_json(dry_run_path, dry_run)
    dry_run_sha = sha256_file(dry_run_path)

    review = _review_manifest_payload(manifest_sha=manifest_sha, dry_run_sha=dry_run_sha)
    _apply_mutation(review, mutate.get("review_manifest"))
    review_path = tmp_path / "review_manifest.json"
    _write_json(review_path, review)
    review_sha = sha256_file(review_path)

    admission = _admission_request_payload(
        manifest_sha=manifest_sha,
        dry_run_sha=dry_run_sha,
        review_sha=review_sha,
    )
    _apply_mutation(admission, mutate.get("admission_request"))
    admission_path = tmp_path / "admission_request.json"
    _write_json(admission_path, admission)

    return {
        "manifest": manifest_path,
        "dry_run_report": dry_run_path,
        "review_manifest": review_path,
        "admission_request": admission_path,
    }


def _apply_mutation(payload: dict[str, object], mutation: object | None) -> None:
    if mutation is not None:
        mutation(payload)  # type: ignore[misc]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_manifest.v1",
        "corpus_id": "example-public-corpus",
        "corpus_class": "public_literature",
        "created_at": "2026-06-28T00:00:00Z",
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
                "pdf_path": "/private/input_a.pdf",
                "pdf_sha256": "",
                "title": "redacted public paper A",
                "doi": "",
                "source_url": "https://example.org/public-a",
                "license_or_access": "public",
                "provenance_note": "redacted provenance",
            },
            {
                "document_id": "doc-example-002",
                "pdf_path": "/private/input_b.pdf",
                "pdf_sha256": "",
                "title": "redacted public paper B",
                "doi": "",
                "source_url": "https://example.org/public-b",
                "license_or_access": "public",
                "provenance_note": "redacted provenance",
            },
        ],
    }


def _dry_run_report_payload(*, manifest_sha: str) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_dry_run.v1",
        "run_id": "custom-dry-run-example-001",
        "generated_at": "2026-06-28T00:00:00Z",
        "decision": "passed",
        "corpus_id": "example-public-corpus",
        "corpus_class": "public_literature",
        "redacted_api_origin": "http://127.0.0.1:18000",
        "endpoint_kind": "mineru_api",
        "requested_backend": "hybrid-engine",
        "requested_effort": "medium",
        "requested_parse_method": "auto",
        "manifest_summary": {
            "manifest_path": "manifest.json",
            "manifest_sha256": manifest_sha,
            "document_count": 2,
            "pdf_hash_coverage": {"with_sha256": 0, "without_sha256": 2},
            "source_policy": "public-open-access-redacted",
            "redaction_policy": {
                "commit_raw_pdfs": False,
                "commit_parsed_documents": False,
                "commit_mineru_bundles": False,
                "commit_full_reports": False,
            },
            "documents": ["doc-example-001", "doc-example-002"],
        },
        "parse_summary": {"attempted": 2, "success": 2, "failed": 0, "parsed_document_count": 2},
        "corpus_audit_summary": {
            "extracted_record_count": 3,
            "accepted_record_count": 2,
            "rejected_record_count": 1,
            "consistent_duplicate_count": 0,
            "conflict_count": 0,
            "unresolved_conflict_count": 0,
        },
        "confirmation_boundary": {
            "dataset_confirmation_confirmed": False,
            "phase1_status": "not_run",
            "training_dataset_admitted": False,
        },
        "warnings": [],
        "errors": [],
        "outputs": {"dry_run_report": "dry_run_report.json"},
    }


def _review_manifest_payload(*, manifest_sha: str, dry_run_sha: str) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_review.v1",
        "review_manifest_id": "review-example-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "created_at": "2026-06-28T00:00:00Z",
        "created_by": "reviewer-redacted",
        "source_dry_run_report_sha256": dry_run_sha,
        "source_manifest_sha256": manifest_sha,
        "review_policy": "example-review-policy",
        "review_records": [
            _review_record("review-record-001", "record-example-001", "accept", source_sha="sha256:" + "a" * 64),
            _review_record(
                "review-record-002",
                "record-example-002",
                "reject",
                source_sha="sha256:" + "b" * 64,
                rejection_reason="review rejected the value summary",
            ),
            _review_record(
                "review-record-003",
                "record-example-003",
                "needs_review",
                document_id="doc-example-002",
                source_sha="sha256:" + "c" * 64,
                notes="requires another reviewer",
            ),
        ],
    }


def _review_record(
    review_id: str,
    record_id: str,
    decision: str,
    *,
    source_sha: str,
    document_id: str = "doc-example-001",
    rejection_reason: str = "",
    notes: str = "",
) -> dict[str, str]:
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
        "reviewed_at": "2026-06-28T00:00:00Z",
        "source_artifact_sha256": source_sha,
        "extracted_value_summary": "short redacted extracted value",
        "normalized_value_summary": "short redacted normalized value",
        "confidence_note": "needs second check" if decision == "needs_review" else "",
        "provenance_note": "short redacted provenance",
        "notes": notes,
    }


def _admission_request_payload(*, manifest_sha: str, dry_run_sha: str, review_sha: str) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_admission.v1",
        "admission_request_id": "admission-example-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "created_at": "2026-06-28T00:00:00Z",
        "created_by": "admission-reviewer-redacted",
        "source_manifest_sha256": manifest_sha,
        "source_dry_run_report_sha256": dry_run_sha,
        "source_review_manifest_sha256": review_sha,
        "review_manifest_id": "review-example-001",
        "admission_policy": "example-admission-policy",
        "dataset_target": "example-dataset-target",
        "admission_records": [
            _admission_record("admission-record-001", "record-example-001", "review-record-001", "accept", "admit", "sha256:" + "a" * 64, review_sha),
            _admission_record("admission-record-002", "record-example-002", "review-record-002", "reject", "exclude", "sha256:" + "b" * 64, review_sha, exclusion_reason="excluded after review"),
            _admission_record("admission-record-003", "record-example-003", "review-record-003", "needs_review", "needs_review", "sha256:" + "c" * 64, review_sha, document_id="doc-example-002", notes="requires another reviewer"),
        ],
    }


def _admission_record(
    admission_record_id: str,
    record_id: str,
    review_id: str,
    review_decision: str,
    action: str,
    source_sha: str,
    review_sha: str,
    *,
    document_id: str = "doc-example-001",
    exclusion_reason: str = "",
    notes: str = "",
) -> dict[str, str]:
    return {
        "admission_record_id": admission_record_id,
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "review_manifest_id": "review-example-001",
        "document_id": document_id,
        "record_id": record_id,
        "field_name": "plqy",
        "admission_scope": "record",
        "review_id": review_id,
        "review_decision": review_decision,
        "action": action,
        "admission_reason": "review accepted the normalized value summary" if action == "admit" else "",
        "exclusion_reason": exclusion_reason,
        "source_artifact_sha256": source_sha,
        "review_artifact_sha256": review_sha,
        "provenance_summary": "short redacted provenance summary",
        "normalized_value_summary": "short redacted normalized value summary",
        "notes": notes,
    }
