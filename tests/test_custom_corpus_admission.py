from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_admission import (
    CustomCorpusAdmissionError,
    admission_validation_summary,
    load_admission_request,
    main,
    sha256_file,
    validate_admission_request,
)


def test_example_admission_request_loads() -> None:
    request = load_admission_request(
        Path(__file__).parents[1] / "docs" / "examples" / "custom-corpus-admission-request.example.json"
    )

    assert request.schema_version == "custom_corpus_admission.v1"
    assert request.admission_request_id == "admission-example-001"
    assert len(request.admission_records) == 3


def test_valid_admission_request_summary_counts(tmp_path: Path) -> None:
    request_path = tmp_path / "admission_request.json"
    request_path.write_text(json.dumps(_request_payload()), encoding="utf-8")

    request = load_admission_request(request_path)
    summary = admission_validation_summary(request, path=request_path)

    assert summary["admission_request_path"] == "admission_request.json"
    assert summary["admission_request_sha256"] == sha256_file(request_path)
    assert summary["admission_record_count"] == 3
    assert summary["admit_count"] == 1
    assert summary["exclude_count"] == 1
    assert summary["needs_review_count"] == 1
    assert summary["eligible_record_count"] == 1
    assert summary["decision"] == "needs_review"
    assert summary["blocking_reasons"] == ["records_need_review"]


def test_eligible_decision_when_admit_and_no_needs_review() -> None:
    payload = _request_payload()
    payload["admission_records"] = [payload["admission_records"][0]]

    summary = admission_validation_summary(validate_admission_request(payload))

    assert summary["decision"] == "eligible"
    assert summary["blocking_reasons"] == []


def test_needs_review_decision_when_any_record_needs_review() -> None:
    summary = admission_validation_summary(validate_admission_request(_request_payload()))

    assert summary["decision"] == "needs_review"
    assert summary["blocking_reasons"] == ["records_need_review"]


def test_ineligible_decision_when_no_records_admitted() -> None:
    payload = _request_payload()
    payload["admission_records"] = [payload["admission_records"][1]]

    summary = admission_validation_summary(validate_admission_request(payload))

    assert summary["decision"] == "ineligible"
    assert summary["blocking_reasons"] == ["no_records_admitted"]


def test_invalid_schema_version_fails() -> None:
    payload = _request_payload()
    payload["schema_version"] = "custom_corpus_admission.v0"

    with pytest.raises(CustomCorpusAdmissionError, match="schema_version"):
        validate_admission_request(payload)


def test_unsafe_ids_fail() -> None:
    payload = _request_payload()
    payload["admission_records"][0]["record_id"] = "../record"

    with pytest.raises(CustomCorpusAdmissionError, match="record_id"):
        validate_admission_request(payload)


def test_duplicate_admission_record_id_fails() -> None:
    payload = _request_payload()
    payload["admission_records"][1]["admission_record_id"] = payload["admission_records"][0][
        "admission_record_id"
    ]

    with pytest.raises(CustomCorpusAdmissionError, match="duplicate admission_record_id"):
        validate_admission_request(payload)


def test_duplicate_admission_target_fails() -> None:
    payload = _request_payload()
    payload["admission_records"][1].update(
        {
            "document_id": payload["admission_records"][0]["document_id"],
            "record_id": payload["admission_records"][0]["record_id"],
            "field_name": payload["admission_records"][0]["field_name"],
            "admission_scope": payload["admission_records"][0]["admission_scope"],
        }
    )

    with pytest.raises(CustomCorpusAdmissionError, match="duplicate admission target"):
        validate_admission_request(payload)


def test_action_review_decision_constraints() -> None:
    reject_admit = _request_payload()
    reject_admit["admission_records"][1]["action"] = "admit"
    reject_admit["admission_records"][1]["admission_reason"] = "incorrectly admitting"
    reject_admit["admission_records"][1]["exclusion_reason"] = ""
    with pytest.raises(CustomCorpusAdmissionError, match="review_decision"):
        validate_admission_request(reject_admit)

    admit_no_reason = _request_payload()
    admit_no_reason["admission_records"][0]["admission_reason"] = ""
    with pytest.raises(CustomCorpusAdmissionError, match="admission_reason"):
        validate_admission_request(admit_no_reason)

    exclude_no_reason = _request_payload()
    exclude_no_reason["admission_records"][1]["exclusion_reason"] = ""
    with pytest.raises(CustomCorpusAdmissionError, match="exclusion_reason"):
        validate_admission_request(exclude_no_reason)

    needs_no_notes = _request_payload()
    needs_no_notes["admission_records"][2]["notes"] = ""
    with pytest.raises(CustomCorpusAdmissionError, match="needs_review"):
        validate_admission_request(needs_no_notes)


def test_sha256_normalization_and_empty_required_sha_fails() -> None:
    payload = _request_payload()
    payload["source_manifest_sha256"] = "A" * 64
    payload["source_dry_run_report_sha256"] = "sha256:" + "B" * 64
    payload["source_review_manifest_sha256"] = "C" * 64
    payload["admission_records"][0]["source_artifact_sha256"] = "D" * 64
    payload["admission_records"][0]["review_artifact_sha256"] = "E" * 64

    request = validate_admission_request(payload)

    assert request.source_manifest_sha256 == "sha256:" + "a" * 64
    assert request.source_dry_run_report_sha256 == "sha256:" + "b" * 64
    assert request.source_review_manifest_sha256 == "sha256:" + "c" * 64
    assert request.admission_records[0].source_artifact_sha256 == "sha256:" + "d" * 64
    assert request.admission_records[0].review_artifact_sha256 == "sha256:" + "e" * 64

    missing_sha = _request_payload()
    missing_sha["source_manifest_sha256"] = ""
    with pytest.raises(CustomCorpusAdmissionError, match="source_manifest_sha256"):
        validate_admission_request(missing_sha)

    missing_record_sha = _request_payload()
    missing_record_sha["admission_records"][0]["source_artifact_sha256"] = ""
    with pytest.raises(CustomCorpusAdmissionError, match="source_artifact_sha256"):
        validate_admission_request(missing_record_sha)


def test_credential_like_text_fails_without_leaking_secret() -> None:
    payload = _request_payload()
    payload["admission_records"][0]["notes"] = "contains token abc123"

    with pytest.raises(CustomCorpusAdmissionError) as excinfo:
        validate_admission_request(payload)

    message = str(excinfo.value).lower()
    assert "credential" in message
    assert "abc123" not in message


def test_private_path_like_text_fails_without_leaking_path() -> None:
    payload = _request_payload()
    payload["admission_records"][0]["provenance_summary"] = "/Users/operator/private/paper"

    with pytest.raises(CustomCorpusAdmissionError) as excinfo:
        validate_admission_request(payload)

    message = str(excinfo.value)
    assert "private path" in message
    assert "/Users/operator" not in message


def test_dataset_target_path_like_value_fails() -> None:
    payload = _request_payload()
    payload["dataset_target"] = "/tmp/training.csv"

    with pytest.raises(CustomCorpusAdmissionError, match="dataset_target"):
        validate_admission_request(payload)


def test_created_by_private_email_fails_unless_redacted() -> None:
    payload = _request_payload()
    payload["created_by"] = "reviewer@example.org"
    with pytest.raises(CustomCorpusAdmissionError, match="created_by"):
        validate_admission_request(payload)

    payload["created_by"] = "reviewer-email-redacted"
    request = validate_admission_request(payload)
    assert request.created_by == "reviewer-email-redacted"


def test_cli_prints_safe_summary_and_writes_optional_summary(tmp_path: Path) -> None:
    request_path = tmp_path / "admission_request.json"
    output_summary = tmp_path / "summary.json"
    request_path.write_text(json.dumps(_request_payload()), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        ["--admission-request", str(request_path), "--output-summary", str(output_summary)],
        stdout=stdout,
        stderr=stderr,
    )
    printed = json.loads(stdout.getvalue())
    written = json.loads(output_summary.read_text(encoding="utf-8"))

    assert code == 0
    assert printed["decision"] == "needs_review"
    assert printed["admission_record_count"] == 3
    assert written == printed
    assert str(tmp_path) not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_cli_invalid_request_exits_1_without_leaking_sensitive_value(tmp_path: Path) -> None:
    request_path = tmp_path / "admission_request.json"
    payload = _request_payload()
    payload["admission_records"][0]["notes"] = "password abc123"
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(["--admission-request", str(request_path)], stdout=stdout, stderr=stderr)

    assert code == 1
    assert stdout.getvalue() == ""
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()
    assert "credential" in stderr.getvalue().lower()


def _request_payload() -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_admission.v1",
        "admission_request_id": "admission-example-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "created_at": "2026-06-28T00:00:00Z",
        "created_by": "admission-reviewer-redacted",
        "source_manifest_sha256": "sha256:" + "a" * 64,
        "source_dry_run_report_sha256": "sha256:" + "b" * 64,
        "source_review_manifest_sha256": "sha256:" + "c" * 64,
        "review_manifest_id": "review-example-001",
        "admission_policy": "example-reviewed-record-admission-policy",
        "dataset_target": "example-dataset-target",
        "admission_records": [
            _record("admission-record-001", "record-example-001", "review-record-001", "accept", "admit"),
            _record(
                "admission-record-002",
                "record-example-002",
                "review-record-002",
                "reject",
                "exclude",
                exclusion_reason="review rejected the evidence summary",
            ),
            _record(
                "admission-record-003",
                "record-example-003",
                "review-record-003",
                "needs_review",
                "needs_review",
                notes="requires another reviewer",
            ),
        ],
    }


def _record(
    admission_record_id: str,
    record_id: str,
    review_id: str,
    review_decision: str,
    action: str,
    *,
    exclusion_reason: str = "",
    notes: str = "",
) -> dict[str, str]:
    return {
        "admission_record_id": admission_record_id,
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
        "admission_reason": "review accepted the normalized value summary" if action == "admit" else "",
        "exclusion_reason": exclusion_reason,
        "source_artifact_sha256": "sha256:" + "d" * 64,
        "review_artifact_sha256": "sha256:" + "e" * 64,
        "provenance_summary": "short redacted provenance summary",
        "normalized_value_summary": "short redacted normalized value summary",
        "notes": notes,
    }
