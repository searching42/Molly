from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_review import (
    CustomCorpusReviewError,
    load_review_manifest,
    main,
    review_manifest_summary,
    sha256_file,
    validate_review_manifest,
)


def test_example_review_manifest_loads() -> None:
    manifest = load_review_manifest(
        Path(__file__).parents[1] / "docs" / "examples" / "custom-corpus-review-manifest.example.json"
    )

    assert manifest.schema_version == "custom_corpus_review.v1"
    assert manifest.review_manifest_id == "review-example-001"
    assert manifest.review_records[0].decision == "needs_review"


def test_valid_review_manifest_summary_counts(tmp_path: Path) -> None:
    manifest_path = tmp_path / "review_manifest.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    manifest = load_review_manifest(manifest_path)
    summary = review_manifest_summary(manifest, path=manifest_path)

    assert summary["review_manifest_path"] == "review_manifest.json"
    assert summary["review_manifest_sha256"] == sha256_file(manifest_path)
    assert summary["review_record_count"] == 3
    assert summary["accepted_count"] == 1
    assert summary["rejected_count"] == 1
    assert summary["needs_review_count"] == 1
    assert summary["reviewed_document_count"] == 2
    assert summary["source_dry_run_report_sha256"] == "sha256:" + "a" * 64
    assert summary["source_manifest_sha256"] == "sha256:" + "b" * 64


def test_invalid_schema_version_fails() -> None:
    payload = _manifest_payload()
    payload["schema_version"] = "custom_corpus_review.v0"

    with pytest.raises(CustomCorpusReviewError, match="schema_version"):
        validate_review_manifest(payload)


def test_unsafe_ids_fail() -> None:
    payload = _manifest_payload()
    payload["review_records"][0]["record_id"] = "../record"

    with pytest.raises(CustomCorpusReviewError, match="record_id"):
        validate_review_manifest(payload)


def test_duplicate_review_id_fails() -> None:
    payload = _manifest_payload()
    payload["review_records"][1]["review_id"] = payload["review_records"][0]["review_id"]

    with pytest.raises(CustomCorpusReviewError, match="duplicate review_id"):
        validate_review_manifest(payload)


def test_duplicate_review_target_fails() -> None:
    payload = _manifest_payload()
    payload["review_records"][1].update(
        {
            "document_id": payload["review_records"][0]["document_id"],
            "record_id": payload["review_records"][0]["record_id"],
            "field_name": payload["review_records"][0]["field_name"],
            "review_scope": payload["review_records"][0]["review_scope"],
        }
    )

    with pytest.raises(CustomCorpusReviewError, match="duplicate review target"):
        validate_review_manifest(payload)


def test_decision_specific_constraints() -> None:
    reject_payload = _manifest_payload()
    reject_payload["review_records"][1]["rejection_reason"] = ""
    with pytest.raises(CustomCorpusReviewError, match="rejection_reason"):
        validate_review_manifest(reject_payload)

    accept_payload = _manifest_payload()
    accept_payload["review_records"][0]["rejection_reason"] = "not allowed"
    with pytest.raises(CustomCorpusReviewError, match="rejection_reason"):
        validate_review_manifest(accept_payload)

    needs_payload = _manifest_payload()
    needs_payload["review_records"][2]["notes"] = ""
    needs_payload["review_records"][2]["confidence_note"] = ""
    with pytest.raises(CustomCorpusReviewError, match="needs_review"):
        validate_review_manifest(needs_payload)


def test_sha256_normalization() -> None:
    payload = _manifest_payload()
    payload["source_dry_run_report_sha256"] = "A" * 64
    payload["source_manifest_sha256"] = "sha256:" + "B" * 64
    payload["review_records"][0]["source_artifact_sha256"] = "C" * 64

    manifest = validate_review_manifest(payload)

    assert manifest.source_dry_run_report_sha256 == "sha256:" + "a" * 64
    assert manifest.source_manifest_sha256 == "sha256:" + "b" * 64
    assert manifest.review_records[0].source_artifact_sha256 == "sha256:" + "c" * 64


def test_credential_like_text_fails_without_leaking_secret() -> None:
    payload = _manifest_payload()
    payload["review_records"][0]["notes"] = "contains token abc123"

    with pytest.raises(CustomCorpusReviewError) as excinfo:
        validate_review_manifest(payload)

    message = str(excinfo.value).lower()
    assert "credential" in message
    assert "abc123" not in message


def test_private_path_like_text_fails_without_leaking_path() -> None:
    payload = _manifest_payload()
    payload["review_records"][0]["provenance_note"] = "/Users/operator/private/paper"

    with pytest.raises(CustomCorpusReviewError) as excinfo:
        validate_review_manifest(payload)

    message = str(excinfo.value)
    assert "private path" in message
    assert "/Users/operator" not in message


def test_reviewer_email_like_label_fails_unless_redacted() -> None:
    payload = _manifest_payload()
    payload["review_records"][0]["reviewer_label"] = "reviewer@example.org"
    with pytest.raises(CustomCorpusReviewError, match="reviewer_label"):
        validate_review_manifest(payload)

    payload["review_records"][0]["reviewer_label"] = "reviewer-email-redacted"
    manifest = validate_review_manifest(payload)
    assert manifest.review_records[0].reviewer_label == "reviewer-email-redacted"


def test_cli_prints_safe_summary_and_writes_optional_summary(tmp_path: Path) -> None:
    manifest_path = tmp_path / "review_manifest.json"
    output_summary = tmp_path / "summary.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        ["--review-manifest", str(manifest_path), "--output-summary", str(output_summary)],
        stdout=stdout,
        stderr=stderr,
    )
    printed = json.loads(stdout.getvalue())
    written = json.loads(output_summary.read_text(encoding="utf-8"))

    assert code == 0
    assert printed["review_record_count"] == 3
    assert written == printed
    assert str(tmp_path) not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_cli_invalid_manifest_exits_1_without_leaking_sensitive_value(tmp_path: Path) -> None:
    manifest_path = tmp_path / "review_manifest.json"
    payload = _manifest_payload()
    payload["review_records"][0]["notes"] = "password abc123"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(["--review-manifest", str(manifest_path)], stdout=stdout, stderr=stderr)

    assert code == 1
    assert stdout.getvalue() == ""
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()
    assert "credential" in stderr.getvalue().lower()


def _manifest_payload() -> dict[str, object]:
    records = [
        _record("review-record-001", "doc-example-001", "record-example-001", "accept"),
        _record(
            "review-record-002",
            "doc-example-001",
            "record-example-002",
            "reject",
            rejection_reason="value not supported by table evidence",
        ),
        _record(
            "review-record-003",
            "doc-example-002",
            "record-example-003",
            "needs_review",
            confidence_note="needs second reviewer",
            notes="unclear table mapping",
        ),
    ]
    return {
        "schema_version": "custom_corpus_review.v1",
        "review_manifest_id": "review-manifest-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "created_at": "2026-06-28T00:00:00Z",
        "created_by": "reviewer-redacted",
        "source_dry_run_report_sha256": "sha256:" + "a" * 64,
        "source_manifest_sha256": "sha256:" + "b" * 64,
        "review_policy": "example-manual-review-policy",
        "review_records": records,
    }


def _record(
    review_id: str,
    document_id: str,
    record_id: str,
    decision: str,
    *,
    rejection_reason: str = "",
    confidence_note: str = "",
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
        "source_artifact_sha256": "sha256:" + "c" * 64,
        "extracted_value_summary": "short redacted value summary",
        "normalized_value_summary": "0.42",
        "confidence_note": confidence_note,
        "provenance_note": "placeholder provenance summary only",
        "notes": notes,
    }
