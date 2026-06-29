from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_admission_request_planner import (
    main,
    plan_property_admission_request,
)


def test_ready_readiness_summary_and_complete_review_manifest_returns_planned(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
        require_ready_status=True,
    )

    assert summary["schema_version"] == "custom_corpus_property_admission_request_plan.v1"
    assert summary["planner_status"] == "planned"
    assert summary["admission_readiness_summary_path"] == "property_admission_readiness_summary.json"
    assert summary["review_manifest_path"] == "property_review_manifest.json"
    assert summary["review_queue_id"] == "property-review-queue-001"
    assert summary["property_candidate_manifest_id"] == "property-candidates-001"
    assert summary["review_manifest_id"] == "property-review-manifest-001"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["dry_run_id"] == "custom-dry-run-example-001"
    assert summary["readiness_status"] == "ready"
    assert summary["binding_status"] == "passed"
    assert summary["require_ready_status"] is True
    assert summary["review_record_count"] == 3
    assert summary["accepted_review_count"] == 1
    assert summary["rejected_review_count"] == 1
    assert summary["needs_review_count"] == 1
    assert summary["planned_admit_count"] == 1
    assert summary["planned_exclude_count"] == 1
    assert summary["blocked_count"] == 1
    assert summary["planned_admit_record_ids"] == ["property-candidate-001"]
    assert summary["planned_exclude_record_ids"] == ["property-candidate-002"]
    assert summary["blocked_record_ids"] == ["property-candidate-003"]
    assert summary["planning_errors"] == []
    assert summary["redaction_status"] == "passed"
    assert summary["planned_record_summaries"][0]["planned_action"] == "admit"
    assert summary["planned_record_summaries"][0]["source_review_id"] == "property-review-001"


def test_partial_readiness_summary_returns_partial_without_require_ready(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path, readiness_status="partial", binding_status="needs_review")

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
    )

    assert summary["planner_status"] == "partial"
    assert summary["readiness_status"] == "partial"
    assert summary["binding_status"] == "needs_review"
    assert summary["planning_errors"] == []


def test_partial_readiness_summary_blocks_when_ready_status_required(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path, readiness_status="partial", binding_status="needs_review")

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
        require_ready_status=True,
    )

    assert summary["planner_status"] == "blocked"
    assert "readiness_status_not_ready" in summary["planning_errors"]


def test_blocked_readiness_summary_returns_blocked(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path, readiness_status="blocked", binding_status="failed")

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
    )

    assert summary["planner_status"] == "blocked"
    assert "readiness_status_blocked" in summary["planning_errors"]


def test_readiness_errors_block_planner(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    payload["readiness_errors"] = ["accepted_review_missing_provenance_note"]
    readiness_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
    )

    assert summary["planner_status"] == "blocked"
    assert "readiness_errors_present" in summary["planning_errors"]


def test_no_planned_admission_candidates_or_exclusions_blocks(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    payload["planned_admission_candidate_record_ids"] = []
    payload["planned_exclusion_record_ids"] = []
    payload["blocked_from_admission_record_ids"] = [
        "property-candidate-001",
        "property-candidate-002",
        "property-candidate-003",
    ]
    readiness_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
    )

    assert summary["planner_status"] == "blocked"
    assert "no_planned_admission_records" in summary["planning_errors"]


def test_review_decisions_map_to_planned_actions(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
    )

    by_record_id = {record["record_id"]: record for record in summary["planned_record_summaries"]}
    assert by_record_id["property-candidate-001"]["planned_action"] == "admit"
    assert by_record_id["property-candidate-002"]["planned_action"] == "exclude"
    assert by_record_id["property-candidate-003"]["planned_action"] == "blocked"
    assert by_record_id["property-candidate-001"]["normalized_value_summary"] == "normalized scalar value summary"
    assert by_record_id["property-candidate-001"]["provenance_summary"] == "short provenance summary"


def test_blocked_from_admission_records_are_not_planned_as_admit(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    payload["planned_admission_candidate_record_ids"] = []
    payload["blocked_from_admission_record_ids"] = ["property-candidate-001", "property-candidate-003"]
    readiness_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
    )

    assert "property-candidate-001" not in summary["planned_admit_record_ids"]
    assert "property-candidate-001" in summary["blocked_record_ids"]
    blocked = [record for record in summary["planned_record_summaries"] if record["record_id"] == "property-candidate-001"][0]
    assert blocked["planned_action"] == "blocked"


def test_planned_admission_candidate_with_non_accept_review_decision_fails(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    payload["planned_admission_candidate_record_ids"] = ["property-candidate-002"]
    payload["planned_exclusion_record_ids"] = []
    readiness_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
    )

    assert summary["planner_status"] == "blocked"
    assert "planned_admit_review_decision_invalid" in summary["planning_errors"]


def test_planned_exclusion_with_non_reject_review_decision_fails(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    payload["planned_exclusion_record_ids"] = ["property-candidate-001"]
    readiness_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
    )

    assert summary["planner_status"] == "blocked"
    assert "planned_exclusion_review_decision_invalid" in summary["planning_errors"]


@pytest.mark.parametrize(
    ("field_name", "error_code"),
    [
        ("extracted_value_summary", "planned_admit_missing_extracted_value_summary"),
        ("normalized_value_summary", "planned_admit_missing_normalized_value_summary"),
        ("provenance_note", "planned_admit_missing_provenance_note"),
    ],
)
def test_accepted_planned_admit_missing_required_review_summaries_fails(
    tmp_path: Path,
    field_name: str,
    error_code: str,
) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    review_payload["review_records"][0][field_name] = ""
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")
    _refresh_readiness_review_manifest_sha(readiness_path, review_path)

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
    )

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planning_errors"]


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
def test_review_manifest_binding_mismatches_fail(tmp_path: Path, mutator: object, error_code: str) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    mutator(payload)
    readiness_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
    )

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planning_errors"]


def test_review_manifest_sha_mismatch_fails_when_readiness_summary_includes_sha(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    payload["review_manifest_sha256"] = "sha256:" + "9" * 64
    readiness_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
    )

    assert summary["planner_status"] == "blocked"
    assert "review_manifest_sha256_mismatch" in summary["planning_errors"]


def test_summary_uses_safe_basenames_and_excludes_temp_paths(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
    )
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["admission_readiness_summary_path"] == "property_admission_readiness_summary.json"
    assert summary["review_manifest_path"] == "property_review_manifest.json"
    assert str(tmp_path) not in serialized


def test_invalid_readiness_summary_schema_fails_safely(tmp_path: Path) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")
    review_path = _write_review_manifest(tmp_path, _review_manifest_payload())
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        ["--admission-readiness-summary", str(readiness_path), "--review-manifest", str(review_path)],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "readiness summary invalid" in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_invalid_review_manifest_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    review_payload["review_records"][0]["notes"] = "password abc123"
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        ["--admission-readiness-summary", str(readiness_path), "--review-manifest", str(review_path)],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()
    assert "credential" in stderr.getvalue().lower()


def test_summary_redaction_fail_closed_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_admission_request_planner._contains_forbidden_material",
        lambda value: True,
    )
    stdout = io.StringIO()

    code = main(
        ["--admission-readiness-summary", str(readiness_path), "--review-manifest", str(review_path)],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    summary = json.loads(stdout.getvalue())

    assert code == 1
    assert summary == {
        "schema_version": "custom_corpus_property_admission_request_plan.v1",
        "planner_status": "blocked",
        "planning_errors": ["property_admission_request_plan_summary_redaction_failed"],
        "redaction_status": "failed",
    }


def test_cli_stdout_is_valid_json_and_writes_optional_markdown(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)
    output_summary = tmp_path / "admission_request_plan_summary.json"
    output_markdown = tmp_path / "admission_request_plan_summary.md"
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--admission-readiness-summary",
            str(readiness_path),
            "--review-manifest",
            str(review_path),
            "--output-summary",
            str(output_summary),
            "--output-markdown",
            str(output_markdown),
            "--require-ready-status",
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
    assert "No admission request created" in markdown
    assert "No admission action created" in markdown
    assert "No `custom_corpus_admission.v1` created" in markdown
    assert "No materialization" in markdown
    assert "No candidate/training CSV" in markdown
    assert "No Phase 1" in markdown
    assert "No DatasetConfirmation change" in markdown
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in markdown
    assert stderr.getvalue() == ""


def test_no_admission_request_json_is_created(tmp_path: Path) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)
    output_summary = tmp_path / "admission_request_plan_summary.json"

    code = main(
        [
            "--admission-readiness-summary",
            str(readiness_path),
            "--review-manifest",
            str(review_path),
            "--output-summary",
            str(output_summary),
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert code == 0
    assert output_summary.exists()
    assert not (tmp_path / "admission_request.json").exists()
    assert not (tmp_path / "custom_corpus_admission.json").exists()


def test_no_llm_mineru_pdf_parsed_document_or_workflow_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readiness_path, review_path = _write_artifacts(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.workflows.corpus_to_phase1_workflow",
            "ai4s_agent.document_parse_service",
            "ai4s_agent.document_parse",
            "ai4s_agent.mineru",
            "ai4s_agent.custom_corpus_admission",
            "openai",
            "pdfplumber",
        )
        if name.startswith(forbidden):
            raise AssertionError(f"forbidden import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracking_import)

    summary = plan_property_admission_request(
        admission_readiness_summary_path=readiness_path,
        review_manifest_path=review_path,
    )

    assert summary["planner_status"] == "planned"
    assert not any("custom_corpus_admission" in name for name in imported_modules)


def _write_artifacts(
    tmp_path: Path,
    *,
    readiness_status: str = "ready",
    binding_status: str = "passed",
) -> tuple[Path, Path]:
    review_path = _write_review_manifest(tmp_path, _review_manifest_payload())
    readiness_payload = _readiness_summary_payload(review_path, readiness_status=readiness_status, binding_status=binding_status)
    readiness_path = tmp_path / "property_admission_readiness_summary.json"
    readiness_path.write_text(json.dumps(readiness_payload), encoding="utf-8")
    return readiness_path, review_path


def _write_review_manifest(tmp_path: Path, payload: dict[str, object]) -> Path:
    review_path = tmp_path / "property_review_manifest.json"
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    return review_path


def _refresh_readiness_review_manifest_sha(readiness_path: Path, review_path: Path) -> None:
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    payload["review_manifest_sha256"] = _sha256_file(review_path)
    readiness_path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _readiness_summary_payload(
    review_path: Path,
    *,
    readiness_status: str,
    binding_status: str,
) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_property_admission_readiness.v1",
        "readiness_status": readiness_status,
        "review_binding_summary_path": "property_review_binding_summary.json",
        "review_binding_summary_sha256": "sha256:" + "e" * 64,
        "review_manifest_path": "property_review_manifest.json",
        "review_manifest_sha256": _sha256_file(review_path),
        "review_queue_id": "property-review-queue-001",
        "property_candidate_manifest_id": "property-candidates-001",
        "review_manifest_id": "property-review-manifest-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "binding_status": binding_status,
        "require_complete_binding": True,
        "review_record_count": 3,
        "accepted_review_count": 1,
        "rejected_review_count": 1,
        "needs_review_count": 1,
        "admission_ready_record_count": 1,
        "planned_admission_candidate_record_ids": ["property-candidate-001"],
        "planned_exclusion_record_ids": ["property-candidate-002"],
        "blocked_from_admission_record_ids": ["property-candidate-003"],
        "unreviewed_queue_record_ids": [],
        "reviewed_blocked_record_ids": [],
        "unknown_review_record_ids": [],
        "readiness_errors": [],
        "warnings": [],
        "source_manifest_sha256": "sha256:" + "a" * 64,
        "source_dry_run_report_sha256": "sha256:" + "b" * 64,
        "redaction_status": "passed",
    }


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
                extracted_value_summary="extracted scalar value summary",
                normalized_value_summary="normalized scalar value summary",
                provenance_note="short provenance summary",
            ),
            _review_record(
                review_id="property-review-002",
                document_id="doc-example-001",
                record_id="property-candidate-002",
                field_name="invalid_numeric_value",
                decision="reject",
                rejection_reason="reviewer rejected this numeric value",
                extracted_value_summary="rejected extracted value summary",
                normalized_value_summary="",
                provenance_note="short rejected provenance summary",
            ),
            _review_record(
                review_id="property-review-003",
                document_id="doc-example-002",
                record_id="property-candidate-003",
                field_name="ambiguous_yield_range",
                decision="needs_review",
                extracted_value_summary="ambiguous extracted range summary",
                normalized_value_summary="",
                provenance_note="short ambiguous provenance summary",
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
    extracted_value_summary: str,
    normalized_value_summary: str,
    provenance_note: str,
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
        "field_name": field_name,
        "review_scope": "record",
        "decision": decision,
        "rejection_reason": rejection_reason,
        "reviewer_label": "reviewer-redacted",
        "reviewed_at": "2026-06-29T00:00:00Z",
        "source_artifact_sha256": "sha256:" + "c" * 64,
        "extracted_value_summary": extracted_value_summary,
        "normalized_value_summary": normalized_value_summary,
        "confidence_note": confidence_note,
        "provenance_note": provenance_note,
        "notes": notes,
    }
