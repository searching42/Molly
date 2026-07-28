from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_candidate import (
    CustomCorpusPropertyCandidateError,
    load_property_candidate_manifest,
    main,
    property_candidate_manifest_summary,
    sha256_file,
    validate_property_candidate_manifest,
)


def test_example_property_candidate_manifest_loads() -> None:
    manifest = load_property_candidate_manifest(
        Path(__file__).parents[1] / "docs" / "examples" / "custom-corpus-property-candidates.example.json"
    )

    assert manifest.schema_version == "custom_corpus_property_candidate.v1"
    assert manifest.property_candidate_manifest_id == "property-candidates-example-001"
    assert len(manifest.records) == 4


def test_valid_manifest_summary_counts_candidates_reject_and_needs_review(tmp_path: Path) -> None:
    path = tmp_path / "property_candidates.json"
    path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    manifest = load_property_candidate_manifest(path)
    summary = property_candidate_manifest_summary(manifest, path=path)

    assert summary["schema_version"] == "custom_corpus_property_candidate.v1"
    assert summary["property_candidate_manifest_path"] == "property_candidates.json"
    assert summary["property_candidate_manifest_sha256"] == sha256_file(path)
    assert summary["property_candidate_manifest_id"] == "property-candidates-001"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["dry_run_id"] == "custom-dry-run-example-001"
    assert summary["candidate_policy"] == "example-open-ended-numeric-policy"
    assert summary["extraction_scope"] == "numeric scientific property candidates"
    assert summary["record_count"] == 4
    assert summary["candidate_count"] == 2
    assert summary["needs_review_count"] == 1
    assert summary["rejected_count"] == 1
    assert summary["review_required_count"] == 3
    assert summary["unique_document_count"] == 3
    assert summary["unique_entity_count"] == 3
    assert summary["unique_field_count"] >= 3
    assert summary["value_kind_counts"]["numeric_scalar"] == 2
    assert summary["value_kind_counts"]["numeric_range"] == 1
    assert summary["value_kind_counts"]["unknown"] == 1
    assert summary["property_family_counts"]["photophysical"] == 2
    assert summary["unit_status_counts"]["explicit"] == 2
    assert summary["extraction_source_counts"]["deterministic"] == 4
    assert summary["source_manifest_sha256"] == "sha256:" + "a" * 64
    assert summary["source_dry_run_report_sha256"] == "sha256:" + "b" * 64
    assert summary["warnings"] == []


def test_arbitrary_safe_canonical_property_guess_is_allowed() -> None:
    payload = _manifest_payload()
    payload["records"][0]["canonical_property_guess"] = "strange_numeric_property_v17"
    payload["records"][0]["field_name"] = "strange_numeric_property_v17"

    manifest = validate_property_candidate_manifest(payload)

    assert manifest.records[0].canonical_property_guess == "strange_numeric_property_v17"
    assert manifest.records[0].field_name == "strange_numeric_property_v17"


def test_unfamiliar_property_is_not_rejected_by_whitelist() -> None:
    payload = _manifest_payload()
    candidate = payload["records"][1]

    assert candidate["canonical_property_guess"] == "delayed_fluorescence_fraction"
    manifest = validate_property_candidate_manifest(payload)
    assert any(record.canonical_property_guess == "delayed_fluorescence_fraction" for record in manifest.records)


def test_invalid_schema_version_fails() -> None:
    payload = _manifest_payload()
    payload["schema_version"] = "custom_corpus_property_candidate.v0"

    with pytest.raises(CustomCorpusPropertyCandidateError, match="schema_version"):
        validate_property_candidate_manifest(payload)


def test_duplicate_property_candidate_id_fails() -> None:
    payload = _manifest_payload()
    payload["records"][1]["property_candidate_id"] = payload["records"][0]["property_candidate_id"]

    with pytest.raises(CustomCorpusPropertyCandidateError, match="duplicate property_candidate_id"):
        validate_property_candidate_manifest(payload)


def test_duplicate_target_fails() -> None:
    payload = _manifest_payload()
    for key in ("document_id", "entity_id", "field_name", "table_id", "row_id", "column_name"):
        payload["records"][1][key] = payload["records"][0][key]

    with pytest.raises(CustomCorpusPropertyCandidateError, match="duplicate property candidate target"):
        validate_property_candidate_manifest(payload)


def test_candidate_scalar_requires_finite_value_normalized() -> None:
    payload = _manifest_payload()
    payload["records"][0]["value_normalized"] = None

    with pytest.raises(CustomCorpusPropertyCandidateError, match="value_normalized"):
        validate_property_candidate_manifest(payload)

    payload["records"][0]["value_normalized"] = float("nan")
    with pytest.raises(CustomCorpusPropertyCandidateError, match="finite"):
        validate_property_candidate_manifest(payload)


def test_candidate_range_requires_finite_min_max_and_order() -> None:
    payload = _manifest_payload()
    record = payload["records"][2]
    record["trainability_decision"] = "candidate"
    record["unit_status"] = "not_applicable"
    record["unit_normalized"] = "not_applicable"
    record["notes"] = ""
    record["value_kind"] = "numeric_range"
    record["value_min"] = 5.0
    record["value_max"] = 2.0
    record["rejection_reason"] = ""

    with pytest.raises(CustomCorpusPropertyCandidateError, match="value_min"):
        validate_property_candidate_manifest(payload)


def test_candidate_tuple_requires_non_empty_finite_tuple() -> None:
    payload = _manifest_payload()
    record = payload["records"][0]
    record["value_kind"] = "numeric_tuple"
    record["value_normalized"] = None
    record["value_tuple"] = [1.0, float("inf")]

    with pytest.raises(CustomCorpusPropertyCandidateError, match="value_tuple"):
        validate_property_candidate_manifest(payload)

    record["value_tuple"] = []
    with pytest.raises(CustomCorpusPropertyCandidateError, match="value_tuple"):
        validate_property_candidate_manifest(payload)


def test_scalar_with_range_or_tuple_fields_fails() -> None:
    payload = _manifest_payload()
    payload["records"][0]["value_min"] = 0.1

    with pytest.raises(CustomCorpusPropertyCandidateError, match="numeric_scalar"):
        validate_property_candidate_manifest(payload)

    payload = _manifest_payload()
    payload["records"][0]["value_tuple"] = [1.0, 2.0]
    with pytest.raises(CustomCorpusPropertyCandidateError, match="numeric_scalar"):
        validate_property_candidate_manifest(payload)


def test_range_with_scalar_or_tuple_fields_fails() -> None:
    payload = _manifest_payload()
    record = payload["records"][2]
    record["trainability_decision"] = "candidate"
    record["unit_status"] = "not_applicable"
    record["unit_normalized"] = "not_applicable"
    record["notes"] = ""
    record["value_kind"] = "numeric_range"
    record["value_min"] = 1.0
    record["value_max"] = 2.0
    record["value_normalized"] = 1.5
    record["rejection_reason"] = ""

    with pytest.raises(CustomCorpusPropertyCandidateError, match="numeric_range"):
        validate_property_candidate_manifest(payload)

    record["value_normalized"] = None
    record["value_tuple"] = [1.0, 2.0]
    with pytest.raises(CustomCorpusPropertyCandidateError, match="numeric_range"):
        validate_property_candidate_manifest(payload)


def test_candidate_with_explicit_unit_missing_unit_normalized_fails() -> None:
    payload = _manifest_payload()
    payload["records"][0]["unit_normalized"] = ""

    with pytest.raises(CustomCorpusPropertyCandidateError, match="unit_normalized"):
        validate_property_candidate_manifest(payload)


def test_candidate_with_missing_unit_fails_unless_needs_review_or_reject() -> None:
    payload = _manifest_payload()
    payload["records"][0]["unit_status"] = "missing"
    payload["records"][0]["unit_normalized"] = ""

    with pytest.raises(CustomCorpusPropertyCandidateError, match="unit_status"):
        validate_property_candidate_manifest(payload)

    payload["records"][0]["trainability_decision"] = "needs_review"
    payload["records"][0]["notes"] = "unit requires reviewer"
    manifest = validate_property_candidate_manifest(payload)
    assert manifest.records[0].trainability_decision == "needs_review"


def test_reject_without_rejection_reason_fails() -> None:
    payload = _manifest_payload()
    payload["records"][3]["rejection_reason"] = ""

    with pytest.raises(CustomCorpusPropertyCandidateError, match="rejection_reason"):
        validate_property_candidate_manifest(payload)


def test_needs_review_without_notes_or_decision_reason_fails() -> None:
    payload = _manifest_payload()
    payload["records"][2]["notes"] = ""
    payload["records"][2]["decision_reason"] = ""

    with pytest.raises(CustomCorpusPropertyCandidateError, match="needs_review"):
        validate_property_candidate_manifest(payload)


def test_record_corpus_id_mismatch_fails() -> None:
    payload = _manifest_payload()
    payload["records"][0]["corpus_id"] = "other-corpus"

    with pytest.raises(CustomCorpusPropertyCandidateError, match="corpus_id"):
        validate_property_candidate_manifest(payload)


def test_record_dry_run_id_mismatch_fails() -> None:
    payload = _manifest_payload()
    payload["records"][0]["dry_run_id"] = "other-run"

    with pytest.raises(CustomCorpusPropertyCandidateError, match="dry_run_id"):
        validate_property_candidate_manifest(payload)


def test_empty_required_sha_fails_and_sha_normalization_works() -> None:
    payload = _manifest_payload()
    payload["source_dry_run_report_sha256"] = "B" * 64
    payload["source_manifest_sha256"] = "A" * 64
    payload["records"][0]["source_artifact_sha256"] = "C" * 64
    payload["records"][0]["parsed_document_sha256"] = "D" * 64

    manifest = validate_property_candidate_manifest(payload)

    assert manifest.source_dry_run_report_sha256 == "sha256:" + "b" * 64
    assert manifest.source_manifest_sha256 == "sha256:" + "a" * 64
    assert manifest.records[0].source_artifact_sha256 == "sha256:" + "c" * 64
    assert manifest.records[0].parsed_document_sha256 == "sha256:" + "d" * 64

    missing = _manifest_payload()
    missing["records"][0]["source_artifact_sha256"] = ""
    with pytest.raises(CustomCorpusPropertyCandidateError, match="source_artifact_sha256"):
        validate_property_candidate_manifest(missing)


def test_confidence_outside_0_to_1_fails() -> None:
    payload = _manifest_payload()
    payload["records"][0]["confidence"] = 1.5

    with pytest.raises(CustomCorpusPropertyCandidateError, match="confidence"):
        validate_property_candidate_manifest(payload)


def test_private_path_like_text_fails_without_leaking_path() -> None:
    payload = _manifest_payload()
    payload["records"][0]["provenance_summary"] = "/Users/operator/private/paper"

    with pytest.raises(CustomCorpusPropertyCandidateError) as excinfo:
        validate_property_candidate_manifest(payload)

    message = str(excinfo.value)
    assert "private path" in message
    assert "/Users/operator" not in message


def test_credential_like_text_fails_without_leaking_secret() -> None:
    payload = _manifest_payload()
    payload["records"][0]["notes"] = "contains token abc123"

    with pytest.raises(CustomCorpusPropertyCandidateError) as excinfo:
        validate_property_candidate_manifest(payload)

    message = str(excinfo.value).lower()
    assert "credential" in message
    assert "abc123" not in message


def test_url_query_or_signed_url_like_text_fails() -> None:
    payload = _manifest_payload()
    payload["records"][0]["method_summary"] = "see https://example.org/item?id=123"

    with pytest.raises(CustomCorpusPropertyCandidateError, match="URL"):
        validate_property_candidate_manifest(payload)

    payload = _manifest_payload()
    payload["records"][0]["notes"] = "signed value x-amz-signature=abc"
    with pytest.raises(CustomCorpusPropertyCandidateError, match="URL"):
        validate_property_candidate_manifest(payload)


def test_created_by_private_email_fails_unless_redacted() -> None:
    payload = _manifest_payload()
    payload["created_by"] = "operator@example.org"
    with pytest.raises(CustomCorpusPropertyCandidateError, match="created_by"):
        validate_property_candidate_manifest(payload)

    payload["created_by"] = "operator-email-redacted"
    manifest = validate_property_candidate_manifest(payload)
    assert manifest.created_by == "operator-email-redacted"


def test_cli_prints_safe_summary_and_exits_0(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    output_summary = tmp_path / "summary.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        ["--property-candidates", str(manifest_path), "--output-summary", str(output_summary)],
        stdout=stdout,
        stderr=stderr,
    )
    printed = json.loads(stdout.getvalue())
    written = json.loads(output_summary.read_text(encoding="utf-8"))

    assert code == 0
    assert printed == written
    assert printed["candidate_count"] == 2
    assert str(tmp_path) not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_cli_exits_1_on_invalid_without_leaking_sensitive_value(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    payload = _manifest_payload()
    payload["records"][0]["notes"] = "password abc123"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(["--property-candidates", str(manifest_path)], stdout=stdout, stderr=stderr)

    assert code == 1
    assert stdout.getvalue() == ""
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()
    assert "credential" in stderr.getvalue().lower()


def test_summary_does_not_include_raw_value_or_provenance_summaries_by_default(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    manifest = load_property_candidate_manifest(manifest_path)
    summary = property_candidate_manifest_summary(manifest, path=manifest_path)
    serialized = json.dumps(summary, sort_keys=True)

    assert "Phi_PL 72 percent" not in serialized
    assert "short redacted table provenance" not in serialized
    assert "value_raw_summary" not in serialized
    assert "provenance_summary" not in serialized


def _manifest_payload() -> dict[str, object]:
    source_manifest_sha = "sha256:" + "a" * 64
    dry_run_sha = "sha256:" + "b" * 64
    return {
        "schema_version": "custom_corpus_property_candidate.v1",
        "property_candidate_manifest_id": "property-candidates-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "created_at": "2026-06-29T00:00:00Z",
        "created_by": "operator-redacted",
        "source_manifest_sha256": source_manifest_sha,
        "source_dry_run_report_sha256": dry_run_sha,
        "candidate_policy": "example-open-ended-numeric-policy",
        "extraction_scope": "numeric scientific property candidates",
        "records": [
            _record(
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
            _record(
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
            _record(
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
            _record(
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


def _record(
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
