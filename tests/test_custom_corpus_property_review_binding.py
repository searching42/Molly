from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_candidate_review_queue import build_property_candidate_review_queue
from ai4s_agent.custom_corpus_property_review_binding import (
    bind_property_review_manifest,
    main,
)


def test_valid_queue_and_complete_review_manifest_passes(tmp_path: Path) -> None:
    queue_path = _build_queue(tmp_path)
    review_path = _write_review_manifest(tmp_path, _review_manifest_payload())

    summary = bind_property_review_manifest(
        review_queue_path=queue_path,
        review_manifest_path=review_path,
        require_complete_queue=True,
    )

    assert summary["schema_version"] == "custom_corpus_property_review_binding.v1"
    assert summary["binding_status"] == "passed"
    assert summary["review_queue_path"] == "property_candidate_review_queue.json"
    assert summary["review_manifest_path"] == "property_review_manifest.json"
    assert summary["review_queue_id"] == "property-review-queue-001"
    assert summary["property_candidate_manifest_id"] == "property-candidates-001"
    assert summary["review_manifest_id"] == "property-review-manifest-001"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["dry_run_id"] == "custom-dry-run-example-001"
    assert summary["queue_record_count"] == 3
    assert summary["blocked_record_count"] == 1
    assert summary["review_record_count"] == 3
    assert summary["reviewed_queue_record_count"] == 3
    assert summary["accepted_count"] == 2
    assert summary["rejected_count"] == 0
    assert summary["needs_review_count"] == 1
    assert summary["unreviewed_queue_record_count"] == 0
    assert summary["reviewed_queue_record_ids"] == [
        "property-candidate-001",
        "property-candidate-002",
        "property-candidate-003",
    ]
    assert summary["unreviewed_queue_record_ids"] == []
    assert summary["binding_errors"] == []
    assert summary["require_complete_queue"] is True
    assert summary["redaction_status"] == "passed"


def test_example_property_review_manifest_binds_to_example_queue(tmp_path: Path) -> None:
    manifest_path = Path(__file__).parents[1] / "docs" / "examples" / "custom-corpus-property-candidates.example.json"
    review_path = Path(__file__).parents[1] / "docs" / "examples" / "custom-corpus-property-review-manifest.example.json"
    build_property_candidate_review_queue(
        property_candidates_path=manifest_path,
        output_dir=tmp_path / "queues",
        review_queue_id="property-review-queue-example-001",
    )
    queue_path = tmp_path / "queues" / "property-review-queue-example-001" / "property_candidate_review_queue.json"

    summary = bind_property_review_manifest(
        review_queue_path=queue_path,
        review_manifest_path=review_path,
        require_complete_queue=True,
    )

    assert summary["binding_status"] == "passed"
    assert summary["review_manifest_id"] == "property-review-manifest-example-001"


def test_partial_review_manifest_needs_review_without_complete_requirement(tmp_path: Path) -> None:
    queue_path = _build_queue(tmp_path)
    payload = _review_manifest_payload()
    payload["review_records"] = payload["review_records"][:2]
    review_path = _write_review_manifest(tmp_path, payload)

    summary = bind_property_review_manifest(
        review_queue_path=queue_path,
        review_manifest_path=review_path,
    )

    assert summary["binding_status"] == "needs_review"
    assert summary["reviewed_queue_record_count"] == 2
    assert summary["unreviewed_queue_record_count"] == 1
    assert summary["unreviewed_queue_record_ids"] == ["property-candidate-003"]
    assert summary["binding_errors"] == []


def test_partial_review_manifest_fails_when_complete_queue_is_required(tmp_path: Path) -> None:
    queue_path = _build_queue(tmp_path)
    payload = _review_manifest_payload()
    payload["review_records"] = payload["review_records"][:2]
    review_path = _write_review_manifest(tmp_path, payload)

    summary = bind_property_review_manifest(
        review_queue_path=queue_path,
        review_manifest_path=review_path,
        require_complete_queue=True,
    )

    assert summary["binding_status"] == "failed"
    assert summary["binding_errors"] == ["queue_review_incomplete"]
    assert summary["unreviewed_queue_record_ids"] == ["property-candidate-003"]


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda payload: _set_manifest_and_record_field(payload, "corpus_id", "other-corpus"), "corpus_id_mismatch"),
        (lambda payload: _set_manifest_and_record_field(payload, "dry_run_id", "other-dry-run"), "dry_run_id_mismatch"),
        (
            lambda payload: payload.__setitem__("source_dry_run_report_sha256", "sha256:" + "9" * 64),
            "source_dry_run_report_sha256_mismatch",
        ),
        (
            lambda payload: payload.__setitem__("source_manifest_sha256", "sha256:" + "8" * 64),
            "source_manifest_sha256_mismatch",
        ),
    ],
)
def test_manifest_level_binding_mismatches_fail(
    tmp_path: Path,
    mutator: object,
    error_code: str,
) -> None:
    queue_path = _build_queue(tmp_path)
    payload = _review_manifest_payload()
    mutator(payload)
    review_path = _write_review_manifest(tmp_path, payload)

    summary = bind_property_review_manifest(review_queue_path=queue_path, review_manifest_path=review_path)

    assert summary["binding_status"] == "failed"
    assert error_code in summary["binding_errors"]


@pytest.mark.parametrize(
    ("record_mutator", "error_code"),
    [
        (
            lambda record: record.__setitem__("record_id", "property-candidate-999"),
            "unknown_review_record",
        ),
        (
            lambda record: record.__setitem__("record_id", "property-candidate-004"),
            "reviewed_blocked_record",
        ),
        (
            lambda record: record.__setitem__("document_id", "doc-example-999"),
            "review_record_document_id_mismatch",
        ),
        (
            lambda record: record.__setitem__("field_name", "wrong_field"),
            "review_record_field_name_mismatch",
        ),
        (
            lambda record: record.__setitem__("source_artifact_sha256", "sha256:" + "9" * 64),
            "review_record_source_artifact_sha256_mismatch",
        ),
        (
            lambda record: record.__setitem__("review_scope", "document"),
            "review_scope_invalid",
        ),
    ],
)
def test_review_record_binding_mismatches_fail(
    tmp_path: Path,
    record_mutator: object,
    error_code: str,
) -> None:
    queue_path = _build_queue(tmp_path)
    payload = _review_manifest_payload()
    record_mutator(payload["review_records"][0])
    review_path = _write_review_manifest(tmp_path, payload)

    summary = bind_property_review_manifest(review_queue_path=queue_path, review_manifest_path=review_path)

    assert summary["binding_status"] == "failed"
    assert error_code in summary["binding_errors"]


@pytest.mark.parametrize(
    ("field_name", "error_code"),
    [
        ("extracted_value_summary", "accepted_review_missing_extracted_value_summary"),
        ("normalized_value_summary", "accepted_review_missing_normalized_value_summary"),
        ("provenance_note", "accepted_review_missing_provenance_note"),
    ],
)
def test_accepted_review_requires_value_and_provenance_summaries(
    tmp_path: Path,
    field_name: str,
    error_code: str,
) -> None:
    queue_path = _build_queue(tmp_path)
    payload = _review_manifest_payload()
    payload["review_records"][0][field_name] = ""
    review_path = _write_review_manifest(tmp_path, payload)

    summary = bind_property_review_manifest(review_queue_path=queue_path, review_manifest_path=review_path)

    assert summary["binding_status"] == "failed"
    assert error_code in summary["binding_errors"]


def test_reject_and_needs_review_decisions_use_existing_review_validation(tmp_path: Path) -> None:
    queue_path = _build_queue(tmp_path)
    payload = _review_manifest_payload()
    payload["review_records"][2]["decision"] = "reject"
    payload["review_records"][2]["rejection_reason"] = ""
    payload["review_records"][2]["notes"] = ""
    review_path = _write_review_manifest(tmp_path, payload)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        ["--review-queue", str(queue_path), "--review-manifest", str(review_path)],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "rejection_reason" in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_summary_uses_safe_basenames_and_excludes_raw_review_details(tmp_path: Path) -> None:
    queue_path = _build_queue(tmp_path)
    review_path = _write_review_manifest(tmp_path, _review_manifest_payload())

    summary = bind_property_review_manifest(review_queue_path=queue_path, review_manifest_path=review_path)
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["review_queue_path"] == "property_candidate_review_queue.json"
    assert summary["review_manifest_path"] == "property_review_manifest.json"
    assert str(tmp_path) not in serialized
    assert "Phi_PL 72 percent" not in serialized
    assert "short redacted table provenance" not in serialized
    assert "accepted normalized value summary" not in serialized
    assert "provenance summary for accepted value" not in serialized


def test_invalid_queue_schema_fails_safely(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")
    review_path = _write_review_manifest(tmp_path, _review_manifest_payload())
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        ["--review-queue", str(queue_path), "--review-manifest", str(review_path)],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "review queue invalid" in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_invalid_review_manifest_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    queue_path = _build_queue(tmp_path)
    payload = _review_manifest_payload()
    payload["review_records"][0]["notes"] = "password abc123"
    review_path = _write_review_manifest(tmp_path, payload)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        ["--review-queue", str(queue_path), "--review-manifest", str(review_path)],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()
    assert "credential" in stderr.getvalue().lower()


def test_summary_redaction_fail_closed_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    queue_path = _build_queue(tmp_path)
    review_path = _write_review_manifest(tmp_path, _review_manifest_payload())
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_review_binding._FORBIDDEN_MARKERS",
        ("property-review-manifest-001",),
    )
    stdout = io.StringIO()

    code = main(
        ["--review-queue", str(queue_path), "--review-manifest", str(review_path)],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    summary = json.loads(stdout.getvalue())

    assert code == 1
    assert summary == {
        "schema_version": "custom_corpus_property_review_binding.v1",
        "binding_status": "failed",
        "binding_errors": ["property_review_binding_summary_redaction_failed"],
        "redaction_status": "failed",
    }


def test_cli_stdout_is_valid_json_and_writes_optional_summary(tmp_path: Path) -> None:
    queue_path = _build_queue(tmp_path)
    review_path = _write_review_manifest(tmp_path, _review_manifest_payload())
    output_summary = tmp_path / "binding_summary.json"
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--review-queue",
            str(queue_path),
            "--review-manifest",
            str(review_path),
            "--output-summary",
            str(output_summary),
            "--require-complete-queue",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    printed = json.loads(stdout.getvalue())
    written = json.loads(output_summary.read_text(encoding="utf-8"))

    assert code == 0
    assert printed == written
    assert printed["binding_status"] == "passed"
    assert stderr.getvalue() == ""


def test_no_llm_mineru_pdf_parsed_document_or_workflow_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    queue_path = _build_queue(tmp_path)
    review_path = _write_review_manifest(tmp_path, _review_manifest_payload())
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
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

    summary = bind_property_review_manifest(review_queue_path=queue_path, review_manifest_path=review_path)

    assert summary["binding_status"] == "passed"
    assert not any("corpus_to_phase1_workflow" in name for name in imported_modules)


def _build_queue(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_property_candidate_manifest_payload()), encoding="utf-8")
    build_property_candidate_review_queue(
        property_candidates_path=manifest_path,
        output_dir=tmp_path / "queues",
        review_queue_id="property-review-queue-001",
    )
    return tmp_path / "queues" / "property-review-queue-001" / "property_candidate_review_queue.json"


def _write_review_manifest(tmp_path: Path, payload: dict[str, object]) -> Path:
    review_path = tmp_path / "property_review_manifest.json"
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    return review_path


def _set_manifest_and_record_field(payload: dict[str, object], field_name: str, value: str) -> None:
    payload[field_name] = value
    for record in payload["review_records"]:
        record[field_name] = value


def _review_manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_review.v1",
        "review_manifest_id": "property-review-manifest-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "created_at": "2026-06-29T00:00:00Z",
        "created_by": "reviewer-redacted",
        "source_dry_run_report_sha256": "sha256:" + "b" * 64,
        "source_manifest_sha256": "sha256:" + "a" * 64,
        "review_policy": "example-property-candidate-review-policy",
        "review_records": [
            _review_record(
                review_id="property-review-001",
                document_id="doc-example-001",
                record_id="property-candidate-001",
                field_name="plqy",
                decision="accept",
            ),
            _review_record(
                review_id="property-review-002",
                document_id="doc-example-001",
                record_id="property-candidate-002",
                field_name="delayed_fluorescence_fraction",
                decision="accept",
            ),
            _review_record(
                review_id="property-review-003",
                document_id="doc-example-002",
                record_id="property-candidate-003",
                field_name="ambiguous_yield_range",
                decision="needs_review",
                extracted_value_summary="ambiguous extracted range summary",
                normalized_value_summary="",
                provenance_note="short provenance summary for ambiguous value",
                confidence_note="unit requires reviewer",
                notes="needs unit review",
            ),
        ],
    }


def _review_record(
    *,
    review_id: str,
    document_id: str,
    record_id: str,
    field_name: str,
    decision: str,
    extracted_value_summary: str = "accepted extracted value summary",
    normalized_value_summary: str = "accepted normalized value summary",
    provenance_note: str = "provenance summary for accepted value",
    confidence_note: str = "",
    notes: str = "",
) -> dict[str, object]:
    return {
        "review_id": review_id,
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "document_id": document_id,
        "record_id": record_id,
        "field_name": field_name,
        "review_scope": "record",
        "decision": decision,
        "rejection_reason": "",
        "reviewer_label": "reviewer-redacted",
        "reviewed_at": "2026-06-29T00:00:00Z",
        "source_artifact_sha256": "sha256:" + "c" * 64,
        "extracted_value_summary": extracted_value_summary,
        "normalized_value_summary": normalized_value_summary,
        "confidence_note": confidence_note,
        "provenance_note": provenance_note,
        "notes": notes,
    }


def _property_candidate_manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_property_candidate.v1",
        "property_candidate_manifest_id": "property-candidates-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "created_at": "2026-06-29T00:00:00Z",
        "created_by": "operator-redacted",
        "source_manifest_sha256": "sha256:" + "a" * 64,
        "source_dry_run_report_sha256": "sha256:" + "b" * 64,
        "candidate_policy": "example-open-ended-numeric-policy",
        "extraction_scope": "numeric scientific property candidates",
        "records": [
            _candidate_record(
                property_candidate_id="property-candidate-001",
                document_id="doc-example-001",
                source_record_id="source-record-001",
                field_name="plqy",
                raw_property_label="Phi_PL",
                canonical_property_guess="photoluminescence_quantum_yield",
                property_family="photophysical",
                value_kind="numeric_scalar",
                value_raw_summary="Phi_PL 72 percent",
                value_normalized=0.72,
                unit_raw="percent",
                unit_normalized="fraction",
                unit_status="explicit",
                entity_id="compound-001",
                entity_type="compound",
                trainability_decision="candidate",
                decision_reason="finite scalar with explicit unit and entity binding",
                confidence=0.91,
            ),
            _candidate_record(
                property_candidate_id="property-candidate-002",
                document_id="doc-example-001",
                source_record_id="source-record-002",
                field_name="delayed_fluorescence_fraction",
                raw_property_label="Delayed fluorescence fraction",
                canonical_property_guess="delayed_fluorescence_fraction",
                property_family="photophysical",
                value_kind="numeric_scalar",
                value_raw_summary="fraction 0.32",
                value_normalized=0.32,
                unit_raw="fraction",
                unit_normalized="fraction",
                unit_status="explicit",
                entity_id="compound-002",
                entity_type="compound",
                trainability_decision="candidate",
                decision_reason="finite scalar with explicit unit and entity binding",
                confidence=0.8,
            ),
            _candidate_record(
                property_candidate_id="property-candidate-003",
                document_id="doc-example-002",
                source_record_id="source-record-003",
                field_name="ambiguous_yield_range",
                raw_property_label="Yield",
                canonical_property_guess="ambiguous_yield",
                property_family="synthetic",
                value_kind="numeric_range",
                value_raw_summary="between 30 and 50",
                value_min=30.0,
                value_max=50.0,
                unit_raw="",
                unit_normalized="",
                unit_status="missing",
                entity_id="compound-003",
                entity_type="compound",
                trainability_decision="needs_review",
                decision_reason="unit missing and label ambiguous",
                confidence=0.42,
                notes="unit requires reviewer",
            ),
            _candidate_record(
                property_candidate_id="property-candidate-004",
                document_id="doc-example-003",
                source_record_id="source-record-004",
                field_name="unbound_numeric_value",
                raw_property_label="Numeric value",
                canonical_property_guess="unknown_numeric_value",
                property_family="unknown",
                value_kind="unknown",
                value_raw_summary="numeric-looking value",
                unit_raw="",
                unit_normalized="",
                unit_status="unknown",
                entity_id="compound-003",
                entity_type="",
                trainability_decision="reject",
                decision_reason="missing entity binding and provenance",
                rejection_reason="missing usable entity binding",
                confidence=0.2,
                review_required=False,
                provenance_summary="rejected placeholder provenance",
            ),
        ],
    }


def _candidate_record(
    *,
    property_candidate_id: str,
    document_id: str,
    source_record_id: str,
    field_name: str,
    raw_property_label: str,
    canonical_property_guess: str,
    property_family: str,
    value_kind: str,
    value_raw_summary: str,
    value_normalized: float | None = None,
    value_min: float | None = None,
    value_max: float | None = None,
    value_tuple: list[float] | None = None,
    unit_raw: str,
    unit_normalized: str,
    unit_status: str,
    entity_id: str,
    entity_type: str,
    trainability_decision: str,
    decision_reason: str,
    confidence: float,
    review_required: bool = True,
    rejection_reason: str = "",
    notes: str = "",
    provenance_summary: str = "short redacted table provenance",
) -> dict[str, object]:
    return {
        "property_candidate_id": property_candidate_id,
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "document_id": document_id,
        "source_record_id": source_record_id,
        "source_artifact_sha256": "sha256:" + "c" * 64,
        "parsed_document_sha256": "sha256:" + "d" * 64,
        "page": 1,
        "table_id": "table-001",
        "row_id": source_record_id,
        "column_name": field_name,
        "raw_property_label": raw_property_label,
        "canonical_property_guess": canonical_property_guess,
        "property_family": property_family,
        "field_name": field_name,
        "value_kind": value_kind,
        "value_raw_summary": value_raw_summary,
        "value_normalized": value_normalized,
        "value_min": value_min,
        "value_max": value_max,
        "value_tuple": value_tuple or [],
        "unit_raw": unit_raw,
        "unit_normalized": unit_normalized,
        "unit_status": unit_status,
        "entity_id": entity_id,
        "entity_type": entity_type,
        "entity_label_summary": "compound label summary",
        "method_summary": "short method summary",
        "condition_summary": "short condition summary",
        "provenance_summary": provenance_summary,
        "extraction_source": "deterministic",
        "extractor_label": "fixture-extractor",
        "confidence": confidence,
        "trainability_decision": trainability_decision,
        "decision_reason": decision_reason,
        "review_required": review_required,
        "rejection_reason": rejection_reason,
        "notes": notes,
    }
