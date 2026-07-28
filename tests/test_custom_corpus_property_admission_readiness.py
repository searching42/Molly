from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_admission_readiness import (
    main,
    plan_property_admission_readiness,
)
from ai4s_agent.custom_corpus_property_candidate_review_queue import build_property_candidate_review_queue
from ai4s_agent.custom_corpus_property_review_binding import bind_property_review_manifest


def test_passed_binding_and_complete_review_manifest_returns_ready(tmp_path: Path) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path, require_complete_queue=True)

    summary = plan_property_admission_readiness(
        review_binding_summary_path=binding_path,
        review_manifest_path=review_path,
        require_complete_binding=True,
    )

    assert summary["schema_version"] == "custom_corpus_property_admission_readiness.v1"
    assert summary["readiness_status"] == "ready"
    assert summary["review_binding_summary_path"] == "property_review_binding_summary.json"
    assert summary["review_manifest_path"] == "property_review_manifest.json"
    assert summary["review_queue_id"] == "property-review-queue-001"
    assert summary["property_candidate_manifest_id"] == "property-candidates-001"
    assert summary["review_manifest_id"] == "property-review-manifest-001"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["dry_run_id"] == "custom-dry-run-example-001"
    assert summary["binding_status"] == "passed"
    assert summary["require_complete_binding"] is True
    assert summary["review_record_count"] == 3
    assert summary["accepted_review_count"] == 2
    assert summary["rejected_review_count"] == 0
    assert summary["needs_review_count"] == 1
    assert summary["admission_ready_record_count"] == 2
    assert summary["planned_admission_candidate_record_ids"] == [
        "property-candidate-001",
        "property-candidate-002",
    ]
    assert summary["planned_exclusion_record_ids"] == []
    assert summary["blocked_from_admission_record_ids"] == ["property-candidate-003"]
    assert summary["readiness_errors"] == []
    assert summary["redaction_status"] == "passed"


def test_needs_review_binding_returns_partial_when_completeness_not_required(tmp_path: Path) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path, partial=True)

    summary = plan_property_admission_readiness(
        review_binding_summary_path=binding_path,
        review_manifest_path=review_path,
    )

    assert summary["readiness_status"] == "partial"
    assert summary["binding_status"] == "needs_review"
    assert summary["admission_ready_record_count"] == 2
    assert summary["unreviewed_queue_record_ids"] == ["property-candidate-003"]
    assert summary["readiness_errors"] == []


def test_needs_review_binding_blocks_when_complete_binding_required(tmp_path: Path) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path, partial=True)

    summary = plan_property_admission_readiness(
        review_binding_summary_path=binding_path,
        review_manifest_path=review_path,
        require_complete_binding=True,
    )

    assert summary["readiness_status"] == "blocked"
    assert summary["readiness_errors"] == ["binding_incomplete"]


def test_failed_binding_summary_returns_blocked(tmp_path: Path) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path, failed=True)

    summary = plan_property_admission_readiness(
        review_binding_summary_path=binding_path,
        review_manifest_path=review_path,
    )

    assert summary["readiness_status"] == "blocked"
    assert "binding_status_failed" in summary["readiness_errors"]


def test_no_accepted_records_returns_blocked(tmp_path: Path) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path, all_needs_review=True)

    summary = plan_property_admission_readiness(
        review_binding_summary_path=binding_path,
        review_manifest_path=review_path,
    )

    assert summary["readiness_status"] == "blocked"
    assert summary["admission_ready_record_count"] == 0
    assert "no_admission_ready_records" in summary["readiness_errors"]


def test_rejected_records_become_planned_exclusions(tmp_path: Path) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path, with_reject=True)

    summary = plan_property_admission_readiness(
        review_binding_summary_path=binding_path,
        review_manifest_path=review_path,
    )

    assert summary["readiness_status"] == "ready"
    assert summary["planned_admission_candidate_record_ids"] == ["property-candidate-001"]
    assert summary["planned_exclusion_record_ids"] == ["property-candidate-002"]
    assert summary["blocked_from_admission_record_ids"] == ["property-candidate-003"]


def test_accepted_review_not_in_binding_reviewed_ids_blocks(tmp_path: Path) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path, partial=True)
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    payload["reviewed_queue_record_ids"] = ["property-candidate-001"]
    binding_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = plan_property_admission_readiness(
        review_binding_summary_path=binding_path,
        review_manifest_path=review_path,
    )

    assert summary["readiness_status"] == "blocked"
    assert "accepted_record_not_in_reviewed_queue_ids" in summary["readiness_errors"]
    assert "property-candidate-002" in summary["blocked_from_admission_record_ids"]


@pytest.mark.parametrize(
    ("binding_field", "record_id", "error_code"),
    [
        ("reviewed_blocked_record_ids", "property-candidate-001", "accepted_record_in_reviewed_blocked_record_ids"),
        ("unknown_review_record_ids", "property-candidate-001", "accepted_record_in_unknown_review_record_ids"),
    ],
)
def test_accepted_review_in_blocked_or_unknown_binding_lists_fails(
    tmp_path: Path,
    binding_field: str,
    record_id: str,
    error_code: str,
) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path)
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    payload[binding_field] = [record_id]
    binding_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = plan_property_admission_readiness(
        review_binding_summary_path=binding_path,
        review_manifest_path=review_path,
    )

    assert summary["readiness_status"] == "blocked"
    assert error_code in summary["readiness_errors"]


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda payload: payload.__setitem__("review_manifest_id", "other-review"), "review_manifest_id_mismatch"),
        (lambda payload: payload.__setitem__("corpus_id", "other-corpus"), "corpus_id_mismatch"),
        (lambda payload: payload.__setitem__("dry_run_id", "other-dry-run"), "dry_run_id_mismatch"),
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
def test_review_manifest_binding_mismatches_fail(
    tmp_path: Path,
    mutator: object,
    error_code: str,
) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path)
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    mutator(payload)
    binding_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = plan_property_admission_readiness(
        review_binding_summary_path=binding_path,
        review_manifest_path=review_path,
    )

    assert summary["readiness_status"] == "blocked"
    assert error_code in summary["readiness_errors"]


def test_review_manifest_sha_mismatch_fails_when_binding_summary_includes_sha(tmp_path: Path) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path)
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    payload["review_manifest_sha256"] = "sha256:" + "9" * 64
    binding_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = plan_property_admission_readiness(
        review_binding_summary_path=binding_path,
        review_manifest_path=review_path,
    )

    assert summary["readiness_status"] == "blocked"
    assert "review_manifest_sha256_mismatch" in summary["readiness_errors"]


@pytest.mark.parametrize(
    ("field_name", "error_code"),
    [
        ("extracted_value_summary", "accepted_review_missing_extracted_value_summary"),
        ("normalized_value_summary", "accepted_review_missing_normalized_value_summary"),
        ("provenance_note", "accepted_review_missing_provenance_note"),
    ],
)
def test_accepted_review_missing_required_summaries_fails(
    tmp_path: Path,
    field_name: str,
    error_code: str,
) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path)
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    review_payload["review_records"][0][field_name] = ""
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")
    _refresh_binding_review_manifest_sha(binding_path, review_path)

    summary = plan_property_admission_readiness(
        review_binding_summary_path=binding_path,
        review_manifest_path=review_path,
    )

    assert summary["readiness_status"] == "blocked"
    assert error_code in summary["readiness_errors"]


def test_summary_uses_safe_basenames_and_excludes_raw_review_details(tmp_path: Path) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path)

    summary = plan_property_admission_readiness(
        review_binding_summary_path=binding_path,
        review_manifest_path=review_path,
    )
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["review_binding_summary_path"] == "property_review_binding_summary.json"
    assert summary["review_manifest_path"] == "property_review_manifest.json"
    assert str(tmp_path) not in serialized
    assert "accepted normalized value summary" not in serialized
    assert "provenance summary for accepted value" not in serialized


def test_invalid_binding_summary_schema_fails_safely(tmp_path: Path) -> None:
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")
    review_path = _write_review_manifest(tmp_path, _review_manifest_payload())
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        ["--review-binding-summary", str(binding_path), "--review-manifest", str(review_path)],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "binding summary invalid" in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_invalid_review_manifest_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path)
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    review_payload["review_records"][0]["notes"] = "password abc123"
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        ["--review-binding-summary", str(binding_path), "--review-manifest", str(review_path)],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()
    assert "credential" in stderr.getvalue().lower()


def test_summary_redaction_fail_closed_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path)
    monkeypatch.setattr("ai4s_agent.custom_corpus_property_admission_readiness._contains_forbidden_material", lambda value: True)
    stdout = io.StringIO()

    code = main(
        ["--review-binding-summary", str(binding_path), "--review-manifest", str(review_path)],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    summary = json.loads(stdout.getvalue())

    assert code == 1
    assert summary == {
        "schema_version": "custom_corpus_property_admission_readiness.v1",
        "readiness_status": "blocked",
        "readiness_errors": ["property_admission_readiness_summary_redaction_failed"],
        "redaction_status": "failed",
    }


def test_cli_stdout_is_valid_json_and_writes_optional_markdown(tmp_path: Path) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path)
    output_summary = tmp_path / "readiness_summary.json"
    output_markdown = tmp_path / "readiness_summary.md"
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--review-binding-summary",
            str(binding_path),
            "--review-manifest",
            str(review_path),
            "--output-summary",
            str(output_summary),
            "--output-markdown",
            str(output_markdown),
            "--require-complete-binding",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    printed = json.loads(stdout.getvalue())
    written = json.loads(output_summary.read_text(encoding="utf-8"))
    markdown = output_markdown.read_text(encoding="utf-8")

    assert code == 0
    assert printed == written
    assert printed["readiness_status"] == "ready"
    assert "No admission request created" in markdown
    assert "No admission action created" in markdown
    assert "No materialization" in markdown
    assert "No candidate/training CSV" in markdown
    assert "No Phase 1" in markdown
    assert "No DatasetConfirmation change" in markdown
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in markdown
    assert stderr.getvalue() == ""


def test_no_llm_mineru_pdf_parsed_document_or_workflow_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binding_path, review_path = _build_binding_summary(tmp_path)
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

    summary = plan_property_admission_readiness(
        review_binding_summary_path=binding_path,
        review_manifest_path=review_path,
    )

    assert summary["readiness_status"] == "ready"
    assert not any("corpus_to_phase1_workflow" in name for name in imported_modules)


def _build_binding_summary(
    tmp_path: Path,
    *,
    partial: bool = False,
    failed: bool = False,
    all_needs_review: bool = False,
    with_reject: bool = False,
    require_complete_queue: bool = False,
) -> tuple[Path, Path]:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_property_candidate_manifest_payload()), encoding="utf-8")
    build_property_candidate_review_queue(
        property_candidates_path=manifest_path,
        output_dir=tmp_path / "queues",
        review_queue_id="property-review-queue-001",
    )
    queue_path = tmp_path / "queues" / "property-review-queue-001" / "property_candidate_review_queue.json"
    review_payload = _review_manifest_payload()
    if partial:
        review_payload["review_records"] = review_payload["review_records"][:2]
    if failed:
        review_payload["review_records"][0]["record_id"] = "property-candidate-999"
    if all_needs_review:
        for record in review_payload["review_records"]:
            record["decision"] = "needs_review"
            record["normalized_value_summary"] = ""
            record["confidence_note"] = "requires additional review"
            record["notes"] = "not ready for admission"
    if with_reject:
        review_payload["review_records"][1]["decision"] = "reject"
        review_payload["review_records"][1]["rejection_reason"] = "reviewer excluded this value"
        review_payload["review_records"][1]["normalized_value_summary"] = ""
    review_path = _write_review_manifest(tmp_path, review_payload)
    binding_summary = bind_property_review_manifest(
        review_queue_path=queue_path,
        review_manifest_path=review_path,
        require_complete_queue=require_complete_queue,
    )
    binding_path = tmp_path / "property_review_binding_summary.json"
    binding_path.write_text(json.dumps(binding_summary), encoding="utf-8")
    return binding_path, review_path


def _refresh_binding_review_manifest_sha(binding_path: Path, review_path: Path) -> None:
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    payload["review_manifest_sha256"] = _sha256_file(review_path)
    binding_path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_review_manifest(tmp_path: Path, payload: dict[str, object]) -> Path:
    review_path = tmp_path / "property_review_manifest.json"
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    return review_path


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
