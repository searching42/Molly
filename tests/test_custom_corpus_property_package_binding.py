from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_package_binding import main, run_property_package_binding


def test_valid_inputs_run_formal_package_binding_and_return_passed(tmp_path: Path) -> None:
    paths = _write_property_binding_package(tmp_path)

    summary = run_property_package_binding(**_kwargs(paths), confirm_formal_package_binding=True)

    run_dir = paths["output_dir"] / "property-package-binding-001"
    formal_path = run_dir / "custom_corpus_admission_package_validation.json"
    wrapper_path = run_dir / "property_package_binding_summary.json"
    evidence_path = run_dir / "redacted_property_package_binding_evidence.md"
    assert summary["schema_version"] == "custom_corpus_property_package_binding.v1"
    assert summary["binding_status"] == "passed"
    assert summary["binding_run_id"] == "property-package-binding-001"
    assert summary["formal_package_validation_status"] == "passed"
    assert summary["property_precheck_status"] == "passed"
    assert summary["manifest_path"] == "manifest.json"
    assert summary["dry_run_report_path"] == "dry_run_report.json"
    assert summary["review_manifest_path"] == "review_manifest.json"
    assert summary["admission_request_path"] == "custom_corpus_admission.draft.json"
    assert summary["property_precheck_summary_path"] == "property_precheck_summary.json"
    assert summary["formal_package_validation_path"] == "custom_corpus_admission_package_validation.json"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["dry_run_id"] == "custom-dry-run-example-001"
    assert summary["review_manifest_id"] == "review-example-001"
    assert summary["admission_request_id"] == "property-admission-draft-001"
    assert summary["review_queue_id"] == "property-review-queue-001"
    assert summary["property_candidate_manifest_id"] == "property-candidates-001"
    assert summary["dry_run_decision"] == "passed"
    assert summary["phase1_status"] == "not_run"
    assert summary["training_admitted"] is False
    assert summary["admission_record_count"] == 2
    assert summary["admit_count"] == 1
    assert summary["exclude_count"] == 1
    assert summary["blocked_record_count"] == 1
    assert summary["admit_record_ids"] == ["property-candidate-001"]
    assert summary["exclude_record_ids"] == ["property-candidate-002"]
    assert summary["blocked_record_ids"] == ["property-candidate-003"]
    assert summary["binding_errors"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"
    assert formal_path.exists()
    assert wrapper_path.exists()
    assert evidence_path.exists()
    assert json.loads(formal_path.read_text(encoding="utf-8"))["schema_version"] == (
        "custom_corpus_admission_package_validation.v1"
    )
    assert json.loads(wrapper_path.read_text(encoding="utf-8")) == summary


def test_missing_confirmation_exits_1_and_writes_no_formal_validation(tmp_path: Path) -> None:
    paths = _write_property_binding_package(tmp_path)
    stdout = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=io.StringIO())
    summary = json.loads(stdout.getvalue())

    assert code == 1
    assert summary["binding_status"] == "failed"
    assert "formal_package_binding_not_confirmed" in summary["binding_errors"]
    assert not (paths["output_dir"] / "property-package-binding-001" / "custom_corpus_admission_package_validation.json").exists()


def test_failed_precheck_exits_1_and_does_not_call_formal_validator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_property_binding_package(tmp_path, precheck_status="failed")

    def fail_if_called(**_: object) -> dict[str, object]:
        raise AssertionError("formal validator should not be called")

    monkeypatch.setattr("ai4s_agent.custom_corpus_property_package_binding.validate_admission_package", fail_if_called)

    summary = run_property_package_binding(**_kwargs(paths), confirm_formal_package_binding=True)

    assert summary["binding_status"] == "failed"
    assert "property_precheck_failed" in summary["binding_errors"]
    assert not (paths["output_dir"] / "property-package-binding-001" / "custom_corpus_admission_package_validation.json").exists()


def test_needs_review_precheck_blocks_unless_explicitly_allowed(tmp_path: Path) -> None:
    paths = _write_property_binding_package(tmp_path, precheck_status="needs_review")

    blocked = run_property_package_binding(**_kwargs(paths), confirm_formal_package_binding=True)
    allowed = run_property_package_binding(
        **_kwargs(paths, binding_run_id="property-package-binding-allowed-001"),
        confirm_formal_package_binding=True,
        allow_precheck_needs_review=True,
    )

    assert blocked["binding_status"] == "failed"
    assert "property_precheck_needs_review" in blocked["binding_errors"]
    assert allowed["binding_status"] == "needs_review"
    assert allowed["warnings"] == ["property_precheck_needs_review_allowed"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("precheck", lambda payload: payload.__setitem__("corpus_id", "other-corpus"), "corpus_id_mismatch"),
        ("precheck", lambda payload: payload.__setitem__("dry_run_id", "other-run"), "dry_run_id_mismatch"),
        ("precheck", lambda payload: payload.__setitem__("review_manifest_id", "other-review"), "review_manifest_id_mismatch"),
        ("precheck", lambda payload: payload.__setitem__("admission_request_id", "other-admission"), "admission_request_id_mismatch"),
        ("precheck", lambda payload: payload.__setitem__("manifest_sha256", "sha256:" + "0" * 64), "manifest_sha256_mismatch"),
        (
            "precheck",
            lambda payload: payload.__setitem__("dry_run_report_sha256", "sha256:" + "1" * 64),
            "dry_run_report_sha256_mismatch",
        ),
        (
            "precheck",
            lambda payload: payload.__setitem__("review_manifest_sha256", "sha256:" + "2" * 64),
            "review_manifest_sha256_mismatch",
        ),
        (
            "precheck",
            lambda payload: payload.__setitem__("admission_draft_sha256", "sha256:" + "3" * 64),
            "admission_request_sha256_mismatch",
        ),
        ("precheck", lambda payload: payload.__setitem__("dry_run_decision", "failed"), "dry_run_not_passed"),
        ("precheck", lambda payload: payload.__setitem__("phase1_status", "success"), "dry_run_phase1_ran"),
        ("precheck", lambda payload: payload.__setitem__("training_admitted", True), "dry_run_training_admitted"),
        ("precheck", lambda payload: payload.__setitem__("draft_status", "blocked"), "draft_not_written"),
        ("precheck", lambda payload: payload.__setitem__("precheck_errors", ["source_manifest_sha256_mismatch"]), "property_precheck_has_errors"),
        ("precheck", lambda payload: payload.__setitem__("blocked_record_ids", ["property-candidate-001"]), "blocked_record_in_admission_request"),
        ("precheck", lambda payload: payload.__setitem__("admit_count", 2), "admit_count_mismatch"),
        ("precheck", lambda payload: payload.__setitem__("exclude_count", 2), "exclude_count_mismatch"),
        (
            "dry_run_report",
            lambda payload: payload["confirmation_boundary"].__setitem__("dataset_confirmation_confirmed", True),
            "dry_run_dataset_confirmed",
        ),
    ],
)
def test_local_precheck_failures_block_before_formal_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target: str,
    mutator: object,
    error_code: str,
) -> None:
    paths = _write_property_binding_package(tmp_path, mutate={target: mutator})

    def fail_if_called(**_: object) -> dict[str, object]:
        raise AssertionError("formal validator should not be called")

    monkeypatch.setattr("ai4s_agent.custom_corpus_property_package_binding.validate_admission_package", fail_if_called)

    summary = run_property_package_binding(**_kwargs(paths), confirm_formal_package_binding=True)

    assert summary["binding_status"] == "failed"
    assert error_code in summary["binding_errors"]
    assert not (paths["output_dir"] / "property-package-binding-001" / "custom_corpus_admission_package_validation.json").exists()


def test_formal_package_validator_failure_propagates_wrapper_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_property_binding_package(tmp_path)

    def formal_failure(**_: object) -> dict[str, object]:
        return {
            "schema_version": "custom_corpus_admission_package_validation.v1",
            "validation_status": "failed",
            "admission_decision": "ineligible",
            "binding_errors": ["formal_failure"],
            "admit_count": 0,
            "exclude_count": 0,
            "admission_record_count": 0,
        }

    monkeypatch.setattr("ai4s_agent.custom_corpus_property_package_binding.validate_admission_package", formal_failure)

    summary = run_property_package_binding(**_kwargs(paths), confirm_formal_package_binding=True)

    assert summary["binding_status"] == "failed"
    assert summary["formal_package_validation_status"] == "failed"
    assert "formal_package_validation_failed" in summary["binding_errors"]


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_property_binding_package(tmp_path)
    run_dir = paths["output_dir"] / "property-package-binding-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = run_property_package_binding(**_kwargs(paths), confirm_formal_package_binding=True)

    assert summary["binding_status"] == "failed"
    assert "output_directory_not_clean" in summary["binding_errors"]


def test_summary_and_artifacts_use_safe_basenames_and_no_temp_paths(tmp_path: Path) -> None:
    paths = _write_property_binding_package(tmp_path)

    summary = run_property_package_binding(**_kwargs(paths), confirm_formal_package_binding=True)
    run_dir = paths["output_dir"] / "property-package-binding-001"
    serialized = json.dumps(summary, sort_keys=True)
    artifacts = "\n".join(path.read_text(encoding="utf-8") for path in run_dir.iterdir())

    assert summary["manifest_path"] == "manifest.json"
    assert summary["admission_request_path"] == "custom_corpus_admission.draft.json"
    assert str(tmp_path) not in serialized
    assert str(tmp_path) not in artifacts


def test_invalid_precheck_schema_fails_safely(tmp_path: Path) -> None:
    paths = _write_property_binding_package(tmp_path)
    payload = json.loads(paths["property_precheck_summary"].read_text(encoding="utf-8"))
    payload["schema_version"] = "wrong"
    paths["property_precheck_summary"].write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        _cli_args(paths) + ["--confirm-formal-package-binding"],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert "invalid" in stderr.getvalue().lower()
    assert str(tmp_path) not in stderr.getvalue()


def test_invalid_admission_request_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_property_binding_package(tmp_path)
    payload = json.loads(paths["admission_request"].read_text(encoding="utf-8"))
    payload["admission_records"][0]["notes"] = "token abc123"
    paths["admission_request"].write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        _cli_args(paths) + ["--confirm-formal-package-binding"],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_unsafe_markdown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_property_binding_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_package_binding._contains_forbidden_material",
        lambda value: True,
    )
    stdout = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-formal-package-binding"], stdout=stdout, stderr=io.StringIO())
    summary = json.loads(stdout.getvalue())
    run_dir = paths["output_dir"] / "property-package-binding-001"

    assert code == 1
    assert summary == {
        "schema_version": "custom_corpus_property_package_binding.v1",
        "binding_status": "failed",
        "binding_errors": ["property_package_binding_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "redacted_property_package_binding_evidence.md").exists()


def test_cli_stdout_is_valid_json_and_evidence_contains_boundary_statement(tmp_path: Path) -> None:
    paths = _write_property_binding_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-formal-package-binding"], stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())
    evidence = (
        paths["output_dir"]
        / "property-package-binding-001"
        / "redacted_property_package_binding_evidence.md"
    ).read_text(encoding="utf-8")

    assert code == 0
    assert summary["binding_status"] == "passed"
    assert "formal package binding was run" in evidence
    assert "No materialization was run" in evidence
    assert "No materialization plan was created" in evidence
    assert "No candidate/training CSV was created" in evidence
    assert "Phase 1 did not run" in evidence
    assert "DatasetConfirmation was not changed" in evidence
    assert "No training data was admitted" in evidence
    assert stderr.getvalue() == ""


def test_no_materialization_phase1_llm_mineru_pdf_or_parsed_document_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_property_binding_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
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

    summary = run_property_package_binding(**_kwargs(paths), confirm_formal_package_binding=True)

    assert summary["binding_status"] == "passed"
    assert not any(tmp_path.glob("*.csv"))
    assert not (tmp_path / "materialization_plan.json").exists()
    assert not any("custom_corpus_materialization" in name for name in imported_modules)


def _kwargs(paths: dict[str, Path], *, binding_run_id: str = "property-package-binding-001") -> dict[str, object]:
    return {
        "manifest_path": paths["manifest"],
        "dry_run_report_path": paths["dry_run_report"],
        "review_manifest_path": paths["review_manifest"],
        "admission_request_path": paths["admission_request"],
        "property_precheck_summary_path": paths["property_precheck_summary"],
        "output_dir": paths["output_dir"],
        "binding_run_id": binding_run_id,
    }


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
        "--property-precheck-summary",
        str(paths["property_precheck_summary"]),
        "--output-dir",
        str(paths["output_dir"]),
        "--binding-run-id",
        "property-package-binding-001",
    ]


def _write_property_binding_package(
    tmp_path: Path,
    *,
    precheck_status: str = "passed",
    mutate: dict[str, object] | None = None,
) -> dict[str, Path]:
    mutate = mutate or {}
    manifest = _manifest_payload()
    _apply_mutation(manifest, mutate.get("manifest"))
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = _sha256_file(manifest_path)

    dry_run = _dry_run_report_payload(manifest_sha=manifest_sha)
    _apply_mutation(dry_run, mutate.get("dry_run_report"))
    dry_run_path = tmp_path / "dry_run_report.json"
    _write_json(dry_run_path, dry_run)
    dry_run_sha = _sha256_file(dry_run_path)

    review = _review_manifest_payload(manifest_sha=manifest_sha, dry_run_sha=dry_run_sha)
    _apply_mutation(review, mutate.get("review_manifest"))
    review_path = tmp_path / "review_manifest.json"
    _write_json(review_path, review)
    review_sha = _sha256_file(review_path)

    admission = _admission_request_payload(manifest_sha=manifest_sha, dry_run_sha=dry_run_sha, review_sha=review_sha)
    _apply_mutation(admission, mutate.get("admission_request"))
    admission_path = tmp_path / "custom_corpus_admission.draft.json"
    _write_json(admission_path, admission)
    admission_sha = _sha256_file(admission_path)

    precheck = _precheck_payload(
        manifest_sha=manifest_sha,
        dry_run_sha=dry_run_sha,
        review_sha=review_sha,
        admission_sha=admission_sha,
        precheck_status=precheck_status,
    )
    _apply_mutation(precheck, mutate.get("precheck"))
    precheck_path = tmp_path / "property_precheck_summary.json"
    _write_json(precheck_path, precheck)

    output_dir = tmp_path / "property_package_binding"
    return {
        "manifest": manifest_path,
        "dry_run_report": dry_run_path,
        "review_manifest": review_path,
        "admission_request": admission_path,
        "property_precheck_summary": precheck_path,
        "output_dir": output_dir,
    }


def _apply_mutation(payload: dict[str, object], mutation: object | None) -> None:
    if mutation is not None:
        mutation(payload)  # type: ignore[misc]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_manifest.v1",
        "corpus_id": "example-public-corpus",
        "corpus_class": "public_literature",
        "created_at": "2026-06-29T00:00:00Z",
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
                "pdf_path": "redacted-input-a",
                "pdf_sha256": "",
                "title": "redacted public paper A",
                "doi": "",
                "source_url": "https://example.org/public-a",
                "license_or_access": "public",
                "provenance_note": "redacted provenance",
            }
        ],
    }


def _dry_run_report_payload(*, manifest_sha: str) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_dry_run.v1",
        "run_id": "custom-dry-run-example-001",
        "generated_at": "2026-06-29T00:00:00Z",
        "decision": "passed",
        "corpus_id": "example-public-corpus",
        "corpus_class": "public_literature",
        "manifest_summary": {
            "manifest_path": "manifest.json",
            "manifest_sha256": manifest_sha,
            "document_count": 1,
            "pdf_hash_coverage": {"with_sha256": 0, "without_sha256": 1},
            "source_policy": "public-open-access-redacted",
            "redaction_policy": {
                "commit_raw_pdfs": False,
                "commit_parsed_documents": False,
                "commit_mineru_bundles": False,
                "commit_full_reports": False,
            },
            "documents": ["doc-example-001"],
        },
        "confirmation_boundary": {
            "dataset_confirmation_confirmed": False,
            "phase1_status": "not_run",
            "training_dataset_admitted": False,
        },
    }


def _review_manifest_payload(*, manifest_sha: str, dry_run_sha: str) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_review.v1",
        "review_manifest_id": "review-example-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "created_at": "2026-06-29T00:00:00Z",
        "created_by": "reviewer-redacted",
        "source_dry_run_report_sha256": dry_run_sha,
        "source_manifest_sha256": manifest_sha,
        "review_policy": "example-property-review-policy",
        "review_records": [
            _review_record("review-record-001", "property-candidate-001", "accept"),
            _review_record(
                "review-record-002",
                "property-candidate-002",
                "reject",
                rejection_reason="reviewer rejected this numeric value",
            ),
        ],
    }


def _review_record(review_id: str, record_id: str, decision: str, *, rejection_reason: str = "") -> dict[str, object]:
    return {
        "review_id": review_id,
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "document_id": "doc-example-001",
        "record_id": record_id,
        "field_name": "plqy",
        "review_scope": "record",
        "decision": decision,
        "rejection_reason": rejection_reason,
        "reviewer_label": "reviewer-redacted",
        "reviewed_at": "2026-06-29T00:00:00Z",
        "source_artifact_sha256": "sha256:" + "c" * 64,
        "extracted_value_summary": "short extracted value summary",
        "normalized_value_summary": "short normalized value summary",
        "confidence_note": "",
        "provenance_note": "short provenance summary",
        "notes": "",
    }


def _admission_request_payload(*, manifest_sha: str, dry_run_sha: str, review_sha: str) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_admission.v1",
        "admission_request_id": "property-admission-draft-001",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "created_at": "2026-06-29T00:00:00Z",
        "created_by": "operator-redacted",
        "source_manifest_sha256": manifest_sha,
        "source_dry_run_report_sha256": dry_run_sha,
        "source_review_manifest_sha256": review_sha,
        "review_manifest_id": "review-example-001",
        "admission_policy": "draft-property-admission-request-from-plan",
        "dataset_target": "example-candidate-target",
        "admission_records": [
            _admission_record("property-candidate-001", "review-record-001", "accept", "admit", review_sha),
            _admission_record("property-candidate-002", "review-record-002", "reject", "exclude", review_sha),
        ],
    }


def _admission_record(record_id: str, review_id: str, review_decision: str, action: str, review_sha: str) -> dict[str, object]:
    return {
        "admission_record_id": f"property-admission-draft-001-{record_id}",
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
        "admission_reason": "draft request generated from property admission request plan" if action == "admit" else "",
        "exclusion_reason": "draft request generated from property admission request plan" if action == "exclude" else "",
        "source_artifact_sha256": "sha256:" + "c" * 64,
        "review_artifact_sha256": review_sha,
        "provenance_summary": "short provenance summary" if action == "admit" else "",
        "normalized_value_summary": "short normalized value summary" if action == "admit" else "",
        "notes": "draft only",
    }


def _precheck_payload(
    *,
    manifest_sha: str,
    dry_run_sha: str,
    review_sha: str,
    admission_sha: str,
    precheck_status: str,
) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_property_admission_draft_package_precheck.v1",
        "precheck_status": precheck_status,
        "manifest_path": "manifest.json",
        "manifest_sha256": manifest_sha,
        "dry_run_report_path": "dry_run_report.json",
        "dry_run_report_sha256": dry_run_sha,
        "review_manifest_path": "review_manifest.json",
        "review_manifest_sha256": review_sha,
        "admission_draft_path": "custom_corpus_admission.draft.json",
        "admission_draft_sha256": admission_sha,
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "review_manifest_id": "review-example-001",
        "admission_request_id": "property-admission-draft-001",
        "review_queue_id": "property-review-queue-001",
        "property_candidate_manifest_id": "property-candidates-001",
        "dry_run_decision": "passed",
        "phase1_status": "not_run",
        "training_admitted": False,
        "draft_status": "written",
        "planner_status": "planned",
        "readiness_status": "ready",
        "binding_status": "passed",
        "draft_record_count": 2,
        "admit_count": 1,
        "exclude_count": 1,
        "blocked_record_count": 1,
        "admit_record_ids": ["property-candidate-001"],
        "exclude_record_ids": ["property-candidate-002"],
        "blocked_record_ids": ["property-candidate-003"],
        "precheck_errors": [] if precheck_status != "failed" else ["property_precheck_failed"],
        "warnings": [] if precheck_status == "passed" else ["property_precheck_needs_review"],
        "redaction_status": "passed",
    }
