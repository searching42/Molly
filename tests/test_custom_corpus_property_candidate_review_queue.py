from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_candidate_review_queue import (
    REVIEW_QUEUE_ARTIFACTS,
    build_property_candidate_review_queue,
    main,
)


def test_example_property_candidate_manifest_creates_review_queue_artifacts(tmp_path: Path) -> None:
    manifest_path = Path(__file__).parents[1] / "docs" / "examples" / "custom-corpus-property-candidates.example.json"
    output_dir = tmp_path / "queues"

    summary = build_property_candidate_review_queue(
        property_candidates_path=manifest_path,
        output_dir=output_dir,
        review_queue_id="property-review-queue-example-001",
    )
    run_dir = output_dir / "property-review-queue-example-001"

    assert summary["schema_version"] == "custom_corpus_property_candidate_review_queue.v1"
    assert summary["review_queue_status"] == "prepared"
    assert summary["property_candidate_manifest_path"] == "custom-corpus-property-candidates.example.json"
    assert summary["property_candidate_manifest_id"] == "property-candidates-example-001"
    assert summary["review_queue_count"] == 3
    assert summary["blocked_record_count"] == 1
    for artifact in REVIEW_QUEUE_ARTIFACTS:
        assert (run_dir / artifact).exists()


def test_queue_summary_counts_and_record_selection(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    summary = build_property_candidate_review_queue(
        property_candidates_path=manifest_path,
        output_dir=tmp_path / "queues",
        review_queue_id="property-review-queue-001",
    )

    assert summary["candidate_count"] == 2
    assert summary["needs_review_count"] == 1
    assert summary["rejected_count"] == 1
    assert summary["review_queue_record_ids"] == [
        "property-candidate-001",
        "property-candidate-002",
        "property-candidate-003",
    ]
    assert summary["blocked_record_ids"] == ["property-candidate-004"]
    assert summary["field_name_counts"]["plqy"] == 1
    assert summary["property_family_counts"]["photophysical"] == 2
    assert summary["value_kind_counts"]["numeric_scalar"] == 2
    assert summary["unit_status_counts"]["explicit"] == 2
    assert summary["extraction_source_counts"]["deterministic"] == 4


def test_review_required_false_records_are_not_queued(tmp_path: Path) -> None:
    payload = _manifest_payload()
    payload["records"][1]["trainability_decision"] = "needs_review"
    payload["records"][1]["review_required"] = False
    payload["records"][1]["notes"] = "review deliberately disabled for this planning fixture"
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = build_property_candidate_review_queue(
        property_candidates_path=manifest_path,
        output_dir=tmp_path / "queues",
        review_queue_id="property-review-queue-001",
    )
    queue_payload = json.loads(
        (tmp_path / "queues" / "property-review-queue-001" / "property_candidate_review_queue.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["review_queue_record_ids"] == ["property-candidate-001", "property-candidate-003"]
    assert summary["blocked_record_ids"] == ["property-candidate-002", "property-candidate-004"]
    assert [record["property_candidate_id"] for record in queue_payload["queue_records"]] == [
        "property-candidate-001",
        "property-candidate-003",
    ]


def test_queue_record_includes_safe_review_context_and_no_decisions(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    build_property_candidate_review_queue(
        property_candidates_path=manifest_path,
        output_dir=tmp_path / "queues",
        review_queue_id="property-review-queue-001",
    )
    queue_payload = json.loads(
        (tmp_path / "queues" / "property-review-queue-001" / "property_candidate_review_queue.json").read_text(
            encoding="utf-8"
        )
    )
    first_record = queue_payload["queue_records"][0]

    assert first_record["review_queue_record_id"] == "review-queue-property-candidate-001"
    assert first_record["value_normalized"] == 0.72
    assert first_record["unit_normalized"] == "fraction"
    assert first_record["entity_id"] == "compound-001"
    assert first_record["provenance_summary"] == "short redacted table provenance"
    assert first_record["review_instruction"] == "review_property_candidate_for_future_custom_corpus_review_manifest"
    forbidden_keys = {
        "review_decision",
        "reviewer_label",
        "reviewed_at",
        "admission_action",
        "materialization_action",
    }
    assert forbidden_keys.isdisjoint(first_record)
    assert "review_decision" not in json.dumps(queue_payload, sort_keys=True)
    assert "admission_action" not in json.dumps(queue_payload, sort_keys=True)
    assert "materialization_action" not in json.dumps(queue_payload, sort_keys=True)


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    run_dir = tmp_path / "queues" / "property-review-queue-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.json").write_text("{}", encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--property-candidates",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "queues"),
            "--review-queue-id",
            "property-review-queue-001",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "output directory is not empty" in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_cli_stdout_is_valid_json_and_paths_are_run_scoped(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--property-candidates",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "queues"),
            "--review-queue-id",
            "property-review-queue-001",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["review_queue_status"] == "prepared"
    assert summary["artifacts"] == {
        "property_candidate_review_queue_json": "property-review-queue-001/property_candidate_review_queue.json",
        "property_candidate_review_queue_md": "property-review-queue-001/property_candidate_review_queue.md",
        "property_candidate_review_summary_json": "property-review-queue-001/property_candidate_review_summary.json",
        "redacted_property_candidate_evidence_md": "property-review-queue-001/redacted_property_candidate_evidence.md",
    }
    assert str(tmp_path) not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_markdown_outputs_contain_boundaries_and_no_admission_claim(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    build_property_candidate_review_queue(
        property_candidates_path=manifest_path,
        output_dir=tmp_path / "queues",
        review_queue_id="property-review-queue-001",
    )
    run_dir = tmp_path / "queues" / "property-review-queue-001"
    queue_md = (run_dir / "property_candidate_review_queue.md").read_text(encoding="utf-8")
    evidence_md = (run_dir / "redacted_property_candidate_evidence.md").read_text(encoding="utf-8")

    assert "No property extraction" in queue_md
    assert "No LLM or agent call" in queue_md
    assert "No custom_corpus_review.v1 manifest created" in queue_md
    assert "No admission" in queue_md
    assert "No materialization" in queue_md
    assert "No Phase 1" in queue_md
    assert "No DatasetConfirmation change" in queue_md
    assert "review-preparation run only" in evidence_md
    assert "No review decisions were created" in evidence_md
    assert "No admission was performed" in evidence_md
    assert "No materialization was performed" in evidence_md


def test_empty_queue_returns_1_unless_allow_empty_queue(tmp_path: Path) -> None:
    payload = _manifest_payload()
    for record in payload["records"]:
        record["trainability_decision"] = "reject"
        record["review_required"] = False
        record["rejection_reason"] = "excluded-before-review"
        record["decision_reason"] = "record rejected before review"
        record["value_kind"] = "unknown"
        record["value_normalized"] = None
        record["value_min"] = None
        record["value_max"] = None
        record["value_tuple"] = []
        record["unit_status"] = "unknown"
        record["unit_normalized"] = ""
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    stdout = io.StringIO()
    code = main(
        [
            "--property-candidates",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "queues"),
            "--review-queue-id",
            "property-review-queue-empty-001",
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    failed_summary = json.loads(stdout.getvalue())

    assert code == 1
    assert failed_summary["review_queue_status"] == "blocked"
    assert failed_summary["blocking_reasons"] == ["no_reviewable_property_candidates"]

    stdout = io.StringIO()
    code = main(
        [
            "--property-candidates",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "queues"),
            "--review-queue-id",
            "property-review-queue-empty-002",
            "--allow-empty-queue",
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    allowed_summary = json.loads(stdout.getvalue())

    assert code == 0
    assert allowed_summary["review_queue_status"] == "blocked"
    assert allowed_summary["review_queue_count"] == 0


def test_invalid_manifest_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    manifest_path = tmp_path / "invalid_property_candidates.json"
    payload = _manifest_payload()
    payload["records"][0]["notes"] = "secret abc123"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--property-candidates",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "queues"),
            "--review-queue-id",
            "property-review-queue-001",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()
    assert "credential" in stderr.getvalue().lower()


def test_redaction_fail_closed_does_not_write_queue_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_candidate_review_queue.REVIEW_INSTRUCTION",
        "review token abc123",
    )

    stdout = io.StringIO()
    code = main(
        [
            "--property-candidates",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "queues"),
            "--review-queue-id",
            "property-review-queue-001",
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    summary = json.loads(stdout.getvalue())
    run_dir = tmp_path / "queues" / "property-review-queue-001"

    assert code == 1
    assert summary["review_queue_status"] == "blocked"
    assert summary["blocking_reasons"] == ["property_candidate_review_queue_redaction_failed"]
    assert summary["redaction_status"] == "failed"
    assert (run_dir / "property_candidate_review_summary.json").exists()
    assert not (run_dir / "property_candidate_review_queue.json").exists()
    assert "abc123" not in (run_dir / "property_candidate_review_summary.json").read_text(encoding="utf-8")


def test_generated_artifacts_do_not_contain_private_paths_or_credentials(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    build_property_candidate_review_queue(
        property_candidates_path=manifest_path,
        output_dir=tmp_path / "queues",
        review_queue_id="property-review-queue-001",
    )
    run_dir = tmp_path / "queues" / "property-review-queue-001"
    combined = "\n".join((run_dir / artifact).read_text(encoding="utf-8") for artifact in REVIEW_QUEUE_ARTIFACTS)

    assert str(tmp_path) not in combined
    assert "/Users/" not in combined
    assert "/home/" not in combined
    assert ".pdf" not in combined
    assert "Authorization" not in combined
    assert "Bearer" not in combined
    assert "token" not in combined.lower()
    assert "cookie" not in combined.lower()


def test_no_llm_mineru_pdf_parsed_document_or_workflow_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
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

    summary = build_property_candidate_review_queue(
        property_candidates_path=manifest_path,
        output_dir=tmp_path / "queues",
        review_queue_id="property-review-queue-001",
    )

    assert summary["review_queue_status"] == "prepared"
    assert not any("corpus_to_phase1_workflow" in name for name in imported_modules)


def _manifest_payload() -> dict[str, object]:
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
