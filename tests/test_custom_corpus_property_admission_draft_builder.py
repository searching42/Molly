from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_admission import load_admission_request
from ai4s_agent.custom_corpus_property_admission_draft_builder import (
    build_property_admission_draft,
    main,
)


def test_valid_planned_request_plan_writes_draft_artifacts(tmp_path: Path) -> None:
    plan_path, review_path = _write_artifacts(tmp_path)

    summary = build_property_admission_draft(
        admission_request_plan_path=plan_path,
        review_manifest_path=review_path,
        output_dir=tmp_path / "drafts",
        admission_request_id="property-admission-draft-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_admission_draft_output=True,
    )

    run_dir = tmp_path / "drafts" / "property-admission-draft-001"
    draft_path = run_dir / "custom_corpus_admission.draft.json"
    summary_path = run_dir / "property_admission_draft_summary.json"
    evidence_path = run_dir / "redacted_property_admission_draft_evidence.md"
    request = load_admission_request(draft_path)

    assert summary["schema_version"] == "custom_corpus_property_admission_draft_builder.v1"
    assert summary["draft_status"] == "written"
    assert summary["admission_request_plan_path"] == "property_admission_request_plan_summary.json"
    assert summary["review_manifest_path"] == "property_review_manifest.json"
    assert summary["admission_request_id"] == "property-admission-draft-001"
    assert summary["review_queue_id"] == "property-review-queue-001"
    assert summary["property_candidate_manifest_id"] == "property-candidates-001"
    assert summary["review_manifest_id"] == "property-review-manifest-001"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["dry_run_id"] == "custom-dry-run-example-001"
    assert summary["dataset_target"] == "example-candidate-target"
    assert summary["planner_status"] == "planned"
    assert summary["allow_partial_plan"] is False
    assert summary["draft_record_count"] == 2
    assert summary["draft_admit_count"] == 1
    assert summary["draft_exclude_count"] == 1
    assert summary["blocked_record_count"] == 1
    assert summary["draft_admit_record_ids"] == ["property-candidate-001"]
    assert summary["draft_exclude_record_ids"] == ["property-candidate-002"]
    assert summary["blocked_record_ids"] == ["property-candidate-003"]
    assert summary["draft_errors"] == []
    assert summary["redaction_status"] == "passed"
    assert draft_path.exists()
    assert summary_path.exists()
    assert evidence_path.exists()
    assert request.schema_version == "custom_corpus_admission.v1"
    assert request.admission_request_id == "property-admission-draft-001"
    assert [record.action for record in request.admission_records] == ["admit", "exclude"]


def test_generated_draft_validates_as_custom_corpus_admission_v1(tmp_path: Path) -> None:
    plan_path, review_path = _write_artifacts(tmp_path)

    build_property_admission_draft(
        admission_request_plan_path=plan_path,
        review_manifest_path=review_path,
        output_dir=tmp_path / "drafts",
        admission_request_id="property-admission-draft-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_admission_draft_output=True,
    )

    draft_path = tmp_path / "drafts" / "property-admission-draft-001" / "custom_corpus_admission.draft.json"
    request = load_admission_request(draft_path)

    assert request.schema_version == "custom_corpus_admission.v1"
    assert request.admission_policy == "draft-property-admission-request-from-plan"
    assert request.source_manifest_sha256 == "sha256:" + "a" * 64
    assert request.source_dry_run_report_sha256 == "sha256:" + "b" * 64
    assert request.source_review_manifest_sha256 == _sha256_file(review_path)
    assert request.admission_records[0].admission_reason.startswith("draft request generated from property admission request plan")
    assert request.admission_records[1].exclusion_reason.startswith("draft request generated from property admission request plan")


def test_missing_confirmation_flag_exits_1_and_writes_no_draft(tmp_path: Path) -> None:
    plan_path, review_path = _write_artifacts(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--admission-request-plan",
            str(plan_path),
            "--review-manifest",
            str(review_path),
            "--output-dir",
            str(tmp_path / "drafts"),
            "--admission-request-id",
            "property-admission-draft-001",
            "--dataset-target",
            "example-candidate-target",
            "--created-by",
            "operator-redacted",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    summary = json.loads(stdout.getvalue())
    assert summary["draft_status"] == "blocked"
    assert "admission_draft_output_not_confirmed" in summary["draft_errors"]
    assert not (tmp_path / "drafts" / "property-admission-draft-001" / "custom_corpus_admission.draft.json").exists()
    assert stderr.getvalue() == ""


def test_blocked_request_plan_exits_1_and_writes_no_draft(tmp_path: Path) -> None:
    plan_path, review_path = _write_artifacts(tmp_path, planner_status="blocked")

    summary = build_property_admission_draft(
        admission_request_plan_path=plan_path,
        review_manifest_path=review_path,
        output_dir=tmp_path / "drafts",
        admission_request_id="property-admission-draft-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_admission_draft_output=True,
    )

    assert summary["draft_status"] == "blocked"
    assert "request_plan_blocked" in summary["draft_errors"]
    assert not (tmp_path / "drafts" / "property-admission-draft-001" / "custom_corpus_admission.draft.json").exists()


def test_partial_request_plan_requires_allow_partial_plan(tmp_path: Path) -> None:
    plan_path, review_path = _write_artifacts(tmp_path, planner_status="partial")

    blocked = build_property_admission_draft(
        admission_request_plan_path=plan_path,
        review_manifest_path=review_path,
        output_dir=tmp_path / "drafts-blocked",
        admission_request_id="property-admission-draft-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_admission_draft_output=True,
    )
    written = build_property_admission_draft(
        admission_request_plan_path=plan_path,
        review_manifest_path=review_path,
        output_dir=tmp_path / "drafts-written",
        admission_request_id="property-admission-draft-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_admission_draft_output=True,
        allow_partial_plan=True,
    )

    assert blocked["draft_status"] == "blocked"
    assert "partial_plan_requires_allow_partial_plan" in blocked["draft_errors"]
    assert written["draft_status"] == "written"
    assert written["allow_partial_plan"] is True


def test_no_draft_records_exits_1(tmp_path: Path) -> None:
    plan_path, review_path = _write_artifacts(tmp_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["planned_record_summaries"] = [
        record for record in payload["planned_record_summaries"] if record["planned_action"] == "blocked"
    ]
    payload["planned_admit_record_ids"] = []
    payload["planned_exclude_record_ids"] = []
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = build_property_admission_draft(
        admission_request_plan_path=plan_path,
        review_manifest_path=review_path,
        output_dir=tmp_path / "drafts",
        admission_request_id="property-admission-draft-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_admission_draft_output=True,
    )

    assert summary["draft_status"] == "blocked"
    assert "no_draft_admission_records" in summary["draft_errors"]


def test_planned_actions_map_to_admission_actions_and_blocked_records_are_excluded(tmp_path: Path) -> None:
    plan_path, review_path = _write_artifacts(tmp_path)

    build_property_admission_draft(
        admission_request_plan_path=plan_path,
        review_manifest_path=review_path,
        output_dir=tmp_path / "drafts",
        admission_request_id="property-admission-draft-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_admission_draft_output=True,
    )

    draft = json.loads(
        (tmp_path / "drafts" / "property-admission-draft-001" / "custom_corpus_admission.draft.json").read_text(
            encoding="utf-8"
        )
    )
    by_record_id = {record["record_id"]: record for record in draft["admission_records"]}

    assert by_record_id["property-candidate-001"]["action"] == "admit"
    assert by_record_id["property-candidate-001"]["review_decision"] == "accept"
    assert by_record_id["property-candidate-002"]["action"] == "exclude"
    assert by_record_id["property-candidate-002"]["review_decision"] == "reject"
    assert "property-candidate-003" not in by_record_id


@pytest.mark.parametrize(
    ("record_id", "field_name", "error_code"),
    [
        ("property-candidate-001", "review_decision", "planned_admit_review_decision_invalid"),
        ("property-candidate-002", "review_decision", "planned_exclude_review_decision_invalid"),
    ],
)
def test_planned_action_review_decision_mismatch_fails(
    tmp_path: Path,
    record_id: str,
    field_name: str,
    error_code: str,
) -> None:
    plan_path, review_path = _write_artifacts(tmp_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    for record in payload["planned_record_summaries"]:
        if record["record_id"] == record_id:
            record[field_name] = "needs_review"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = build_property_admission_draft(
        admission_request_plan_path=plan_path,
        review_manifest_path=review_path,
        output_dir=tmp_path / "drafts",
        admission_request_id="property-admission-draft-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_admission_draft_output=True,
    )

    assert summary["draft_status"] == "blocked"
    assert error_code in summary["draft_errors"]


@pytest.mark.parametrize(
    ("field_name", "error_code"),
    [
        ("normalized_value_summary", "planned_admit_missing_normalized_value_summary"),
        ("provenance_summary", "planned_admit_missing_provenance_summary"),
        ("source_artifact_sha256", "planned_admit_missing_source_artifact_sha256"),
    ],
)
def test_planned_admit_missing_required_fields_fails(tmp_path: Path, field_name: str, error_code: str) -> None:
    plan_path, review_path = _write_artifacts(tmp_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    for record in payload["planned_record_summaries"]:
        if record["record_id"] == "property-candidate-001":
            record[field_name] = ""
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = build_property_admission_draft(
        admission_request_plan_path=plan_path,
        review_manifest_path=review_path,
        output_dir=tmp_path / "drafts",
        admission_request_id="property-admission-draft-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_admission_draft_output=True,
    )

    assert summary["draft_status"] == "blocked"
    assert error_code in summary["draft_errors"]


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
    plan_path, review_path = _write_artifacts(tmp_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    mutator(payload)
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = build_property_admission_draft(
        admission_request_plan_path=plan_path,
        review_manifest_path=review_path,
        output_dir=tmp_path / "drafts",
        admission_request_id="property-admission-draft-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_admission_draft_output=True,
    )

    assert summary["draft_status"] == "blocked"
    assert error_code in summary["draft_errors"]


def test_review_manifest_sha_mismatch_fails_when_plan_includes_sha(tmp_path: Path) -> None:
    plan_path, review_path = _write_artifacts(tmp_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["review_manifest_sha256"] = "sha256:" + "9" * 64
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = build_property_admission_draft(
        admission_request_plan_path=plan_path,
        review_manifest_path=review_path,
        output_dir=tmp_path / "drafts",
        admission_request_id="property-admission-draft-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_admission_draft_output=True,
    )

    assert summary["draft_status"] == "blocked"
    assert "review_manifest_sha256_mismatch" in summary["draft_errors"]


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    plan_path, review_path = _write_artifacts(tmp_path)
    run_dir = tmp_path / "drafts" / "property-admission-draft-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = build_property_admission_draft(
        admission_request_plan_path=plan_path,
        review_manifest_path=review_path,
        output_dir=tmp_path / "drafts",
        admission_request_id="property-admission-draft-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_admission_draft_output=True,
    )

    assert summary["draft_status"] == "blocked"
    assert "output_directory_not_clean" in summary["draft_errors"]
    assert not (run_dir / "custom_corpus_admission.draft.json").exists()


def test_summary_uses_safe_basenames_and_generated_artifacts_do_not_contain_temp_paths(tmp_path: Path) -> None:
    plan_path, review_path = _write_artifacts(tmp_path)

    summary = build_property_admission_draft(
        admission_request_plan_path=plan_path,
        review_manifest_path=review_path,
        output_dir=tmp_path / "drafts",
        admission_request_id="property-admission-draft-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_admission_draft_output=True,
    )
    run_dir = tmp_path / "drafts" / "property-admission-draft-001"
    serialized_outputs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            run_dir / "custom_corpus_admission.draft.json",
            run_dir / "property_admission_draft_summary.json",
            run_dir / "redacted_property_admission_draft_evidence.md",
        )
    )

    assert summary["admission_request_plan_path"] == "property_admission_request_plan_summary.json"
    assert summary["review_manifest_path"] == "property_review_manifest.json"
    assert str(tmp_path) not in json.dumps(summary, sort_keys=True)
    assert str(tmp_path) not in serialized_outputs


def test_invalid_request_plan_schema_fails_safely(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")
    review_path = _write_review_manifest(tmp_path, _review_manifest_payload())
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--admission-request-plan",
            str(plan_path),
            "--review-manifest",
            str(review_path),
            "--output-dir",
            str(tmp_path / "drafts"),
            "--admission-request-id",
            "property-admission-draft-001",
            "--dataset-target",
            "example-candidate-target",
            "--created-by",
            "operator-redacted",
            "--confirm-admission-draft-output",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "request plan invalid" in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_invalid_review_manifest_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    plan_path, review_path = _write_artifacts(tmp_path)
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    review_payload["review_records"][0]["notes"] = "password abc123"
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--admission-request-plan",
            str(plan_path),
            "--review-manifest",
            str(review_path),
            "--output-dir",
            str(tmp_path / "drafts"),
            "--admission-request-id",
            "property-admission-draft-001",
            "--dataset-target",
            "example-candidate-target",
            "--created-by",
            "operator-redacted",
            "--confirm-admission-draft-output",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()
    assert "credential" in stderr.getvalue().lower()


def test_redaction_fail_closed_writes_no_draft(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan_path, review_path = _write_artifacts(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_admission_draft_builder._contains_forbidden_material",
        lambda value: True,
    )
    stdout = io.StringIO()

    code = main(
        [
            "--admission-request-plan",
            str(plan_path),
            "--review-manifest",
            str(review_path),
            "--output-dir",
            str(tmp_path / "drafts"),
            "--admission-request-id",
            "property-admission-draft-001",
            "--dataset-target",
            "example-candidate-target",
            "--created-by",
            "operator-redacted",
            "--confirm-admission-draft-output",
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    summary = json.loads(stdout.getvalue())

    assert code == 1
    assert summary == {
        "schema_version": "custom_corpus_property_admission_draft_builder.v1",
        "draft_status": "blocked",
        "draft_errors": ["property_admission_draft_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (tmp_path / "drafts" / "property-admission-draft-001" / "custom_corpus_admission.draft.json").exists()


def test_cli_stdout_is_valid_json_and_evidence_contains_boundary_statement(tmp_path: Path) -> None:
    plan_path, review_path = _write_artifacts(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--admission-request-plan",
            str(plan_path),
            "--review-manifest",
            str(review_path),
            "--output-dir",
            str(tmp_path / "drafts"),
            "--admission-request-id",
            "property-admission-draft-001",
            "--dataset-target",
            "example-candidate-target",
            "--created-by",
            "operator-redacted",
            "--confirm-admission-draft-output",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    summary = json.loads(stdout.getvalue())
    evidence = (
        tmp_path
        / "drafts"
        / "property-admission-draft-001"
        / "redacted_property_admission_draft_evidence.md"
    ).read_text(encoding="utf-8")

    assert code == 0
    assert summary["draft_status"] == "written"
    assert "this is a draft admission request artifact" in evidence.lower()
    assert "No training data was admitted" in evidence
    assert "No package binding was run" in evidence
    assert "No materialization was run" in evidence
    assert "No candidate/training CSV was created" in evidence
    assert "Phase 1 did not run" in evidence
    assert "DatasetConfirmation was not changed" in evidence
    assert stderr.getvalue() == ""


def test_no_package_binding_materialization_csv_phase1_or_dataset_confirmation_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path, review_path = _write_artifacts(tmp_path)
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

    summary = build_property_admission_draft(
        admission_request_plan_path=plan_path,
        review_manifest_path=review_path,
        output_dir=tmp_path / "drafts",
        admission_request_id="property-admission-draft-001",
        dataset_target="example-candidate-target",
        created_by="operator-redacted",
        confirm_admission_draft_output=True,
    )

    run_dir = tmp_path / "drafts" / "property-admission-draft-001"
    assert summary["draft_status"] == "written"
    assert not any("custom_corpus_admission_package" in name for name in imported_modules)
    assert not any(run_dir.glob("*.csv"))
    assert not (run_dir / "materialization_plan.json").exists()


def _write_artifacts(
    tmp_path: Path,
    *,
    planner_status: str = "planned",
) -> tuple[Path, Path]:
    review_path = _write_review_manifest(tmp_path, _review_manifest_payload())
    plan_path = tmp_path / "property_admission_request_plan_summary.json"
    plan_path.write_text(json.dumps(_request_plan_payload(review_path, planner_status=planner_status)), encoding="utf-8")
    return plan_path, review_path


def _write_review_manifest(tmp_path: Path, payload: dict[str, object]) -> Path:
    review_path = tmp_path / "property_review_manifest.json"
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    return review_path


def _sha256_file(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _request_plan_payload(review_path: Path, *, planner_status: str) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_property_admission_request_plan.v1",
        "planner_status": planner_status,
        "admission_readiness_summary_path": "property_admission_readiness_summary.json",
        "admission_readiness_summary_sha256": "sha256:" + "e" * 64,
        "review_manifest_path": "property_review_manifest.json",
        "review_manifest_sha256": _sha256_file(review_path),
        "review_queue_id": "property-review-queue-001",
        "property_candidate_manifest_id": "property-candidates-001",
        "review_manifest_id": "property-review-manifest-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "readiness_status": "ready" if planner_status == "planned" else planner_status,
        "binding_status": "passed" if planner_status != "blocked" else "failed",
        "require_ready_status": True,
        "review_record_count": 3,
        "accepted_review_count": 1,
        "rejected_review_count": 1,
        "needs_review_count": 1,
        "planned_admit_count": 1,
        "planned_exclude_count": 1,
        "blocked_count": 1,
        "planned_admit_record_ids": ["property-candidate-001"],
        "planned_exclude_record_ids": ["property-candidate-002"],
        "blocked_record_ids": ["property-candidate-003"],
        "unreviewed_queue_record_ids": [],
        "readiness_errors": [],
        "planning_errors": [] if planner_status != "blocked" else ["readiness_status_blocked"],
        "warnings": [],
        "source_manifest_sha256": "sha256:" + "a" * 64,
        "source_dry_run_report_sha256": "sha256:" + "b" * 64,
        "planned_record_summaries": [
            _planned_record(
                review_id="property-review-001",
                record_id="property-candidate-001",
                document_id="doc-example-001",
                field_name="plqy",
                review_decision="accept",
                planned_action="admit",
                planned_reason="accepted review record is ready for future admission request planning",
                normalized_value_summary="normalized scalar value summary",
                provenance_summary="short provenance summary",
            ),
            _planned_record(
                review_id="property-review-002",
                record_id="property-candidate-002",
                document_id="doc-example-001",
                field_name="invalid_numeric_value",
                review_decision="reject",
                planned_action="exclude",
                planned_reason="reviewer rejected this numeric value",
                normalized_value_summary="",
                provenance_summary="",
            ),
            _planned_record(
                review_id="property-review-003",
                record_id="property-candidate-003",
                document_id="doc-example-002",
                field_name="ambiguous_yield_range",
                review_decision="needs_review",
                planned_action="blocked",
                planned_reason="",
                normalized_value_summary="",
                provenance_summary="",
                blocking_reason="blocked_from_admission_or_needs_review",
            ),
        ],
        "redaction_status": "passed",
    }


def _planned_record(
    *,
    review_id: str,
    record_id: str,
    document_id: str,
    field_name: str,
    review_decision: str,
    planned_action: str,
    planned_reason: str,
    normalized_value_summary: str,
    provenance_summary: str,
    blocking_reason: str = "",
) -> dict[str, object]:
    return {
        "planned_admission_plan_record_id": f"plan-{review_id}",
        "source_review_id": review_id,
        "record_id": record_id,
        "document_id": document_id,
        "field_name": field_name,
        "review_decision": review_decision,
        "planned_action": planned_action,
        "planned_reason": planned_reason,
        "source_artifact_sha256": "sha256:" + "c" * 64,
        "review_manifest_sha256": "sha256:" + "d" * 64,
        "normalized_value_summary": normalized_value_summary,
        "provenance_summary": provenance_summary,
        "blocking_reason": blocking_reason,
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
