from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_candidate_planner import (
    PLANNED_REVIEW_OUTPUT_LABELS,
    main,
    plan_property_candidates,
)


def test_example_property_candidate_manifest_produces_planned_summary() -> None:
    manifest_path = Path(__file__).parents[1] / "docs" / "examples" / "custom-corpus-property-candidates.example.json"

    summary = plan_property_candidates(manifest_path)

    assert summary["schema_version"] == "custom_corpus_property_candidate_planner.v1"
    assert summary["planner_status"] == "planned"
    assert summary["property_candidate_manifest_path"] == "custom-corpus-property-candidates.example.json"
    assert summary["property_candidate_manifest_sha256"].startswith("sha256:")
    assert summary["property_candidate_manifest_id"] == "property-candidates-example-001"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["dry_run_id"] == "custom-dry-run-example-001"
    assert summary["record_count"] == 4
    assert summary["candidate_count"] == 2
    assert summary["needs_review_count"] == 1
    assert summary["rejected_count"] == 1
    assert summary["review_queue_count"] == 3
    assert summary["blocked_record_count"] == 1
    assert summary["review_queue_record_ids"] == [
        "property-candidate-001",
        "property-candidate-002",
        "property-candidate-003",
    ]
    assert summary["blocked_record_ids"] == ["property-candidate-004"]
    assert summary["candidate_record_ids"] == ["property-candidate-001", "property-candidate-002"]
    assert summary["needs_review_record_ids"] == ["property-candidate-003"]
    assert summary["rejected_record_ids"] == ["property-candidate-004"]
    assert summary["redaction_status"] == "passed"
    assert summary["blocking_reasons"] == []


def test_planner_summary_uses_safe_basename_not_temp_path(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    summary = plan_property_candidates(manifest_path)
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["property_candidate_manifest_path"] == "property_candidates.json"
    assert str(tmp_path) not in serialized


def test_planner_summary_includes_counts_and_distributions(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    summary = plan_property_candidates(manifest_path)

    assert summary["field_name_counts"]["plqy"] == 1
    assert summary["field_name_counts"]["delayed_fluorescence_fraction"] == 1
    assert summary["property_family_counts"]["photophysical"] == 2
    assert summary["value_kind_counts"]["numeric_scalar"] == 2
    assert summary["unit_status_counts"]["explicit"] == 2
    assert summary["extraction_source_counts"]["deterministic"] == 4
    assert summary["unique_document_count"] == 3
    assert summary["unique_entity_count"] == 3
    assert summary["unique_field_count"] == 4


def test_review_required_false_records_are_blocked(tmp_path: Path) -> None:
    payload = _manifest_payload()
    payload["records"][2]["review_required"] = False
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = plan_property_candidates(manifest_path)

    assert summary["review_queue_record_ids"] == ["property-candidate-001", "property-candidate-002"]
    assert summary["blocked_record_ids"] == ["property-candidate-003", "property-candidate-004"]
    assert summary["review_queue_count"] == 2
    assert summary["blocked_record_count"] == 2


def test_planned_review_output_labels_are_present(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    summary = plan_property_candidates(manifest_path)

    assert summary["planned_review_output_labels"] == list(PLANNED_REVIEW_OUTPUT_LABELS)


def test_planner_does_not_create_review_queue_artifacts(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    plan_property_candidates(manifest_path)

    assert not (tmp_path / "property_candidate_review_queue.json").exists()
    assert not (tmp_path / "property_candidate_review_queue.md").exists()
    assert not (tmp_path / "property_candidate_review_summary.json").exists()
    assert not (tmp_path / "redacted_property_candidate_evidence.md").exists()


def test_cli_writes_optional_json_and_markdown_summaries(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    output_summary = tmp_path / "planner_summary.json"
    output_markdown = tmp_path / "planner_summary.md"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--property-candidates",
            str(manifest_path),
            "--output-summary",
            str(output_summary),
            "--output-markdown",
            str(output_markdown),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    printed = json.loads(stdout.getvalue())
    written = json.loads(output_summary.read_text(encoding="utf-8"))
    markdown = output_markdown.read_text(encoding="utf-8")

    assert code == 0
    assert printed == written
    assert printed["planner_status"] == "planned"
    assert "No property extraction" in markdown
    assert "No LLM or agent call" in markdown
    assert "No human review manifest created" in markdown
    assert "No admission" in markdown
    assert "No materialization" in markdown
    assert "No Phase 1" in markdown
    assert "No DatasetConfirmation change" in markdown
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in markdown
    assert stderr.getvalue() == ""


def test_manifest_with_only_rejected_records_blocks_and_exits_0(tmp_path: Path) -> None:
    payload = _manifest_payload()
    for index, record in enumerate(payload["records"], start=1):
        record["trainability_decision"] = "reject"
        record["review_required"] = False
        record["rejection_reason"] = f"record-{index}-excluded"
        record["decision_reason"] = "record rejected before review"
        record["value_kind"] = "unknown"
        record["value_normalized"] = None
        record["value_min"] = None
        record["value_max"] = None
        record["value_tuple"] = []
        record["unit_status"] = "unknown"
        record["unit_normalized"] = ""
    manifest_path = tmp_path / "rejected_only.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(["--property-candidates", str(manifest_path)], stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["planner_status"] == "blocked"
    assert summary["review_queue_count"] == 0
    assert summary["blocked_record_count"] == 4
    assert summary["blocking_reasons"] == ["no_reviewable_property_candidates"]
    assert stderr.getvalue() == ""


def test_invalid_property_candidate_manifest_exits_1_without_leaking_sensitive_value(tmp_path: Path) -> None:
    manifest_path = tmp_path / "invalid_property_candidates.json"
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


def test_summary_excludes_raw_values_and_provenance_by_default(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    summary = plan_property_candidates(manifest_path)
    serialized = json.dumps(summary, sort_keys=True)

    assert "Phi_PL 72 percent" not in serialized
    assert "short redacted table provenance" not in serialized
    assert "value_raw_summary" not in serialized
    assert "provenance_summary" not in serialized


def test_summary_redaction_fail_closed_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_candidate_planner.PLANNED_REVIEW_OUTPUT_LABELS",
        ("property_candidate_review_queue.json", "/tmp/private/review_queue.json"),
    )
    summary = plan_property_candidates(manifest_path)

    assert summary == {
        "schema_version": "custom_corpus_property_candidate_planner.v1",
        "planner_status": "blocked",
        "blocking_reasons": ["property_candidate_planner_summary_redaction_failed"],
        "redaction_status": "failed",
    }


def test_cli_stdout_is_valid_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "property_candidates.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    stdout = io.StringIO()

    code = main(["--property-candidates", str(manifest_path)], stdout=stdout, stderr=io.StringIO())

    assert code == 0
    assert json.loads(stdout.getvalue())["schema_version"] == "custom_corpus_property_candidate_planner.v1"


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
        )
        if name.startswith(forbidden):
            raise AssertionError(f"forbidden import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracking_import)

    summary = plan_property_candidates(manifest_path)

    assert summary["planner_status"] == "planned"
    assert not any("corpus_to_phase1_workflow" in name for name in imported_modules)


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
