from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_materialization import load_materialization_plan
from ai4s_agent.custom_corpus_property_materialization_plan_draft import (
    build_property_materialization_plan_draft,
    main,
)


def test_valid_package_validated_inputs_write_materialization_plan_draft(tmp_path: Path) -> None:
    paths = _write_property_materialization_package(tmp_path)

    summary = build_property_materialization_plan_draft(
        **_kwargs(paths),
        created_by="operator-redacted",
        confirm_materialization_plan_draft_output=True,
    )

    run_dir = paths["output_dir"] / "property-materialization-plan-001"
    draft_path = run_dir / "custom_corpus_materialization.draft.json"
    wrapper_path = run_dir / "property_materialization_plan_draft_summary.json"
    evidence_path = run_dir / "redacted_property_materialization_plan_draft_evidence.md"
    plan = load_materialization_plan(draft_path)

    assert summary["schema_version"] == "custom_corpus_property_materialization_plan_draft_builder.v1"
    assert summary["draft_status"] == "written"
    assert summary["materialization_plan_id"] == "property-materialization-plan-001"
    assert summary["manifest_path"] == "manifest.json"
    assert summary["dry_run_report_path"] == "dry_run_report.json"
    assert summary["review_manifest_path"] == "review_manifest.json"
    assert summary["admission_request_path"] == "custom_corpus_admission.draft.json"
    assert summary["formal_package_validation_path"] == "custom_corpus_admission_package_validation.json"
    assert summary["property_package_binding_summary_path"] == "property_package_binding_summary.json"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["dry_run_id"] == "custom-dry-run-example-001"
    assert summary["review_manifest_id"] == "review-example-001"
    assert summary["admission_request_id"] == "property-admission-draft-001"
    assert summary["review_queue_id"] == "property-review-queue-001"
    assert summary["property_candidate_manifest_id"] == "property-candidates-001"
    assert summary["dataset_target"] == "example-candidate-target"
    assert summary["package_binding_status"] == "passed"
    assert summary["formal_package_validation_status"] == "passed"
    assert summary["dry_run_decision"] == "passed"
    assert summary["phase1_status"] == "not_run"
    assert summary["training_admitted"] is False
    assert summary["admission_record_count"] == 2
    assert summary["admit_count"] == 1
    assert summary["exclude_count"] == 1
    assert summary["blocked_record_count"] == 1
    assert summary["materialization_record_count"] == 1
    assert summary["materialization_record_ids"] == ["property-materialization-plan-001-property-candidate-001"]
    assert summary["admit_record_ids"] == ["property-candidate-001"]
    assert summary["exclude_record_ids"] == ["property-candidate-002"]
    assert summary["blocked_record_ids"] == ["property-candidate-003"]
    assert summary["draft_errors"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"
    assert draft_path.exists()
    assert wrapper_path.exists()
    assert evidence_path.exists()
    assert json.loads(wrapper_path.read_text(encoding="utf-8")) == summary
    assert plan.schema_version == "custom_corpus_materialization.v1"
    assert plan.materialization_decision == "planned"
    assert plan.package_validation_status == "passed"
    assert plan.package_admission_decision == "eligible"
    assert len(plan.materialization_records) == 1
    assert plan.materialization_records[0].record_id == "property-candidate-001"


def test_missing_confirmation_exits_1_and_writes_no_draft(tmp_path: Path) -> None:
    paths = _write_property_materialization_package(tmp_path)
    stdout = io.StringIO()

    code = main(_cli_args(paths) + ["--created-by", "operator-redacted"], stdout=stdout, stderr=io.StringIO())
    summary = json.loads(stdout.getvalue())

    assert code == 1
    assert summary["draft_status"] == "blocked"
    assert "materialization_plan_draft_output_not_confirmed" in summary["draft_errors"]
    assert not (paths["output_dir"] / "property-materialization-plan-001" / "custom_corpus_materialization.draft.json").exists()


@pytest.mark.parametrize(
    ("binding_status", "allow", "expected_status"),
    [
        ("failed", False, "blocked"),
        ("needs_review", False, "blocked"),
        ("needs_review", True, "written"),
    ],
)
def test_package_binding_status_gates_draft_output(
    tmp_path: Path,
    binding_status: str,
    allow: bool,
    expected_status: str,
) -> None:
    paths = _write_property_materialization_package(tmp_path, package_binding_status=binding_status)

    summary = build_property_materialization_plan_draft(
        **_kwargs(paths),
        created_by="operator-redacted",
        confirm_materialization_plan_draft_output=True,
        allow_package_binding_needs_review=allow,
    )

    assert summary["draft_status"] == expected_status
    if binding_status == "needs_review" and allow:
        assert summary["warnings"] == ["package_binding_needs_review_allowed"]
    if expected_status == "blocked":
        assert not (paths["output_dir"] / "property-materialization-plan-001" / "custom_corpus_materialization.draft.json").exists()


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("formal_package_validation", lambda payload: payload.__setitem__("validation_status", "failed"), "formal_package_validation_failed"),
        ("formal_package_validation", lambda payload: payload.__setitem__("schema_version", "wrong"), "formal_package_validation_schema_invalid"),
        ("package_binding_summary", lambda payload: payload.__setitem__("manifest_sha256", "sha256:" + "0" * 64), "manifest_sha256_mismatch"),
        (
            "package_binding_summary",
            lambda payload: payload.__setitem__("dry_run_report_sha256", "sha256:" + "1" * 64),
            "dry_run_report_sha256_mismatch",
        ),
        (
            "package_binding_summary",
            lambda payload: payload.__setitem__("review_manifest_sha256", "sha256:" + "2" * 64),
            "review_manifest_sha256_mismatch",
        ),
        (
            "package_binding_summary",
            lambda payload: payload.__setitem__("admission_request_sha256", "sha256:" + "3" * 64),
            "admission_request_sha256_mismatch",
        ),
        (
            "package_binding_summary",
            lambda payload: payload.__setitem__("formal_package_validation_sha256", "sha256:" + "4" * 64),
            "formal_package_validation_sha256_mismatch",
        ),
        ("package_binding_summary", lambda payload: payload.__setitem__("corpus_id", "other-corpus"), "corpus_id_mismatch"),
        ("package_binding_summary", lambda payload: payload.__setitem__("dry_run_id", "other-run"), "dry_run_id_mismatch"),
        ("package_binding_summary", lambda payload: payload.__setitem__("review_manifest_id", "other-review"), "review_manifest_id_mismatch"),
        (
            "package_binding_summary",
            lambda payload: payload.__setitem__("admission_request_id", "other-admission"),
            "admission_request_id_mismatch",
        ),
        ("package_binding_summary", lambda payload: payload.__setitem__("dry_run_decision", "failed"), "dry_run_not_passed"),
        ("package_binding_summary", lambda payload: payload.__setitem__("phase1_status", "success"), "dry_run_phase1_ran"),
        ("package_binding_summary", lambda payload: payload.__setitem__("training_admitted", True), "dry_run_training_admitted"),
        (
            "dry_run_report",
            lambda payload: payload["confirmation_boundary"].__setitem__("dataset_confirmation_confirmed", True),
            "dry_run_dataset_confirmed",
        ),
    ],
)
def test_consistency_failures_block_draft(
    tmp_path: Path,
    target: str,
    mutator: object,
    error_code: str,
) -> None:
    paths = _write_property_materialization_package(tmp_path, mutate={target: mutator})

    summary = build_property_materialization_plan_draft(
        **_kwargs(paths),
        created_by="operator-redacted",
        confirm_materialization_plan_draft_output=True,
    )

    assert summary["draft_status"] == "blocked"
    assert error_code in summary["draft_errors"]
    assert not (paths["output_dir"] / "property-materialization-plan-001" / "custom_corpus_materialization.draft.json").exists()


def test_no_admit_records_fails(tmp_path: Path) -> None:
    paths = _write_property_materialization_package(
        tmp_path,
        mutate={
            "admission_request": lambda payload: payload.__setitem__(
                "admission_records",
                [payload["admission_records"][1]],
            ),
            "formal_package_validation": lambda payload: (
                payload.__setitem__("admission_record_count", 1),
                payload.__setitem__("admit_count", 0),
            ),
            "package_binding_summary": lambda payload: (
                payload.__setitem__("admission_record_count", 1),
                payload.__setitem__("admit_count", 0),
                payload.__setitem__("admit_record_ids", []),
            ),
        },
    )

    summary = build_property_materialization_plan_draft(
        **_kwargs(paths),
        created_by="operator-redacted",
        confirm_materialization_plan_draft_output=True,
    )

    assert summary["draft_status"] == "blocked"
    assert "no_materialization_records" in summary["draft_errors"]


def test_exclude_and_blocked_records_are_not_materialization_records(tmp_path: Path) -> None:
    paths = _write_property_materialization_package(tmp_path)

    summary = build_property_materialization_plan_draft(
        **_kwargs(paths),
        created_by="operator-redacted",
        confirm_materialization_plan_draft_output=True,
    )
    draft = json.loads(
        (
            paths["output_dir"]
            / "property-materialization-plan-001"
            / "custom_corpus_materialization.draft.json"
        ).read_text(encoding="utf-8")
    )
    materialized_ids = {record["record_id"] for record in draft["materialization_records"]}

    assert materialized_ids == {"property-candidate-001"}
    assert "property-candidate-002" not in materialized_ids
    assert "property-candidate-003" not in materialized_ids
    assert summary["exclude_record_ids"] == ["property-candidate-002"]
    assert summary["blocked_record_ids"] == ["property-candidate-003"]


def test_needs_review_records_are_not_materialization_records(tmp_path: Path) -> None:
    paths = _write_property_materialization_package(
        tmp_path,
        mutate={
            "admission_request": lambda payload: payload["admission_records"].append(
                _admission_record("property-candidate-004", "review-record-004", "needs_review", "needs_review", payload["source_review_manifest_sha256"])
            ),
            "review_manifest": lambda payload: payload["review_records"].append(
                _review_record("review-record-004", "property-candidate-004", "needs_review", notes="needs reviewer")
            ),
            "formal_package_validation": lambda payload: (
                payload.__setitem__("admission_record_count", 3),
                payload.__setitem__("needs_review_count", 1),
            ),
            "package_binding_summary": lambda payload: (
                payload.__setitem__("admission_record_count", 3),
            ),
        },
    )

    summary = build_property_materialization_plan_draft(
        **_kwargs(paths),
        created_by="operator-redacted",
        confirm_materialization_plan_draft_output=True,
    )
    draft = json.loads(
        (
            paths["output_dir"]
            / "property-materialization-plan-001"
            / "custom_corpus_materialization.draft.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["draft_status"] == "written"
    assert {record["record_id"] for record in draft["materialization_records"]} == {"property-candidate-001"}


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_property_materialization_package(tmp_path)
    run_dir = paths["output_dir"] / "property-materialization-plan-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = build_property_materialization_plan_draft(
        **_kwargs(paths),
        created_by="operator-redacted",
        confirm_materialization_plan_draft_output=True,
    )

    assert summary["draft_status"] == "blocked"
    assert "output_directory_not_clean" in summary["draft_errors"]


def test_summary_and_artifacts_use_safe_basenames_and_no_temp_paths(tmp_path: Path) -> None:
    paths = _write_property_materialization_package(tmp_path)

    summary = build_property_materialization_plan_draft(
        **_kwargs(paths),
        created_by="operator-redacted",
        confirm_materialization_plan_draft_output=True,
    )
    run_dir = paths["output_dir"] / "property-materialization-plan-001"
    serialized = json.dumps(summary, sort_keys=True)
    artifacts = "\n".join(path.read_text(encoding="utf-8") for path in run_dir.iterdir())

    assert summary["manifest_path"] == "manifest.json"
    assert summary["admission_request_path"] == "custom_corpus_admission.draft.json"
    assert str(tmp_path) not in serialized
    assert str(tmp_path) not in artifacts


def test_invalid_package_binding_summary_schema_fails_safely(tmp_path: Path) -> None:
    paths = _write_property_materialization_package(tmp_path)
    payload = json.loads(paths["package_binding_summary"].read_text(encoding="utf-8"))
    payload["schema_version"] = "wrong"
    paths["package_binding_summary"].write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        _cli_args(paths) + ["--created-by", "operator-redacted", "--confirm-materialization-plan-draft-output"],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert "invalid" in stderr.getvalue().lower()
    assert str(tmp_path) not in stderr.getvalue()


@pytest.mark.parametrize("target", ["formal_package_validation", "admission_request"])
def test_invalid_inputs_exit_1_without_leaking_sensitive_values(tmp_path: Path, target: str) -> None:
    paths = _write_property_materialization_package(tmp_path)
    payload = json.loads(paths[target].read_text(encoding="utf-8"))
    if target == "formal_package_validation":
        payload["binding_errors"] = ["token-abc123"]
    else:
        payload["admission_records"][0]["notes"] = "token abc123"
    paths[target].write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        _cli_args(paths) + ["--created-by", "operator-redacted", "--confirm-materialization-plan-draft-output"],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_materialization_draft_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_property_materialization_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_materialization_plan_draft._contains_forbidden_material",
        lambda value: True,
    )
    stdout = io.StringIO()

    code = main(
        _cli_args(paths) + ["--created-by", "operator-redacted", "--confirm-materialization-plan-draft-output"],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    summary = json.loads(stdout.getvalue())

    assert code == 1
    assert summary == {
        "schema_version": "custom_corpus_property_materialization_plan_draft_builder.v1",
        "draft_status": "blocked",
        "draft_errors": ["property_materialization_plan_draft_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (paths["output_dir"] / "property-materialization-plan-001" / "custom_corpus_materialization.draft.json").exists()


def test_cli_stdout_is_valid_json_and_evidence_contains_boundary_statement(tmp_path: Path) -> None:
    paths = _write_property_materialization_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        _cli_args(paths) + ["--created-by", "operator-redacted", "--confirm-materialization-plan-draft-output"],
        stdout=stdout,
        stderr=stderr,
    )
    summary = json.loads(stdout.getvalue())
    evidence = (
        paths["output_dir"]
        / "property-materialization-plan-001"
        / "redacted_property_materialization_plan_draft_evidence.md"
    ).read_text(encoding="utf-8")

    assert code == 0
    assert summary["draft_status"] == "written"
    assert "this is a materialization plan draft only" in evidence
    assert "No materialization was run" in evidence
    assert "No materialization planner was run" in evidence
    assert "No materializer was run" in evidence
    assert "No candidate/training CSV was created" in evidence
    assert "No training data was admitted" in evidence
    assert "Phase 1 did not run" in evidence
    assert "DatasetConfirmation was not changed" in evidence
    assert stderr.getvalue() == ""


def test_no_planner_materializer_phase1_llm_mineru_pdf_or_parsed_document_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_property_materialization_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
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

    summary = build_property_materialization_plan_draft(
        **_kwargs(paths),
        created_by="operator-redacted",
        confirm_materialization_plan_draft_output=True,
    )

    assert summary["draft_status"] == "written"
    assert not any("custom_corpus_materialization_planner" in name for name in imported_modules)
    assert not any(paths["output_dir"].glob("**/*.csv"))
    assert not (paths["output_dir"] / "property-materialization-plan-001" / "materialized_records.jsonl").exists()


def _kwargs(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "manifest_path": paths["manifest"],
        "dry_run_report_path": paths["dry_run_report"],
        "review_manifest_path": paths["review_manifest"],
        "admission_request_path": paths["admission_request"],
        "formal_package_validation_path": paths["formal_package_validation"],
        "property_package_binding_summary_path": paths["package_binding_summary"],
        "output_dir": paths["output_dir"],
        "materialization_plan_id": "property-materialization-plan-001",
        "dataset_target": "example-candidate-target",
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
        "--formal-package-validation",
        str(paths["formal_package_validation"]),
        "--property-package-binding-summary",
        str(paths["package_binding_summary"]),
        "--output-dir",
        str(paths["output_dir"]),
        "--materialization-plan-id",
        "property-materialization-plan-001",
        "--dataset-target",
        "example-candidate-target",
    ]


def _write_property_materialization_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
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

    formal = _formal_package_validation_payload(
        manifest_sha=manifest_sha,
        dry_run_sha=dry_run_sha,
        review_sha=review_sha,
        admission_sha=admission_sha,
    )
    _apply_mutation(formal, mutate.get("formal_package_validation"))
    formal_path = tmp_path / "custom_corpus_admission_package_validation.json"
    _write_json(formal_path, formal)
    formal_sha = _sha256_file(formal_path)

    binding = _package_binding_summary_payload(
        manifest_sha=manifest_sha,
        dry_run_sha=dry_run_sha,
        review_sha=review_sha,
        admission_sha=admission_sha,
        formal_sha=formal_sha,
        binding_status=package_binding_status,
    )
    _apply_mutation(binding, mutate.get("package_binding_summary"))
    binding_path = tmp_path / "property_package_binding_summary.json"
    _write_json(binding_path, binding)

    return {
        "manifest": manifest_path,
        "dry_run_report": dry_run_path,
        "review_manifest": review_path,
        "admission_request": admission_path,
        "formal_package_validation": formal_path,
        "package_binding_summary": binding_path,
        "output_dir": tmp_path / "property_materialization_plan_draft",
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


def _review_record(
    review_id: str,
    record_id: str,
    decision: str,
    *,
    rejection_reason: str = "",
    notes: str = "",
) -> dict[str, object]:
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
        "confidence_note": "needs second reviewer" if decision == "needs_review" else "",
        "provenance_note": "short provenance summary",
        "notes": notes,
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
        "notes": "still needs review" if action == "needs_review" else "draft only",
    }


def _formal_package_validation_payload(
    *,
    manifest_sha: str,
    dry_run_sha: str,
    review_sha: str,
    admission_sha: str,
) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_admission_package_validation.v1",
        "validation_status": "passed",
        "admission_decision": "eligible",
        "manifest_path": "manifest.json",
        "dry_run_report_path": "dry_run_report.json",
        "review_manifest_path": "review_manifest.json",
        "admission_request_path": "custom_corpus_admission.draft.json",
        "manifest_sha256": manifest_sha,
        "dry_run_report_sha256": dry_run_sha,
        "review_manifest_sha256": review_sha,
        "admission_request_sha256": admission_sha,
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "review_manifest_id": "review-example-001",
        "admission_request_id": "property-admission-draft-001",
        "corpus_class": "public_literature",
        "document_count": 1,
        "dry_run_decision": "passed",
        "dry_run_phase1_status": "not_run",
        "dry_run_dataset_confirmation_confirmed": False,
        "dry_run_training_dataset_admitted": False,
        "review_record_count": 2,
        "admission_record_count": 2,
        "admit_count": 1,
        "exclude_count": 1,
        "needs_review_count": 0,
        "matched_review_record_count": 2,
        "missing_review_record_count": 0,
        "binding_errors": [],
        "warnings": [],
    }


def _package_binding_summary_payload(
    *,
    manifest_sha: str,
    dry_run_sha: str,
    review_sha: str,
    admission_sha: str,
    formal_sha: str,
    binding_status: str,
) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_property_package_binding.v1",
        "binding_status": binding_status,
        "binding_run_id": "property-package-binding-001",
        "manifest_path": "manifest.json",
        "manifest_sha256": manifest_sha,
        "dry_run_report_path": "dry_run_report.json",
        "dry_run_report_sha256": dry_run_sha,
        "review_manifest_path": "review_manifest.json",
        "review_manifest_sha256": review_sha,
        "admission_request_path": "custom_corpus_admission.draft.json",
        "admission_request_sha256": admission_sha,
        "property_precheck_summary_path": "property_precheck_summary.json",
        "property_precheck_summary_sha256": "sha256:" + "b" * 64,
        "formal_package_validation_path": "custom_corpus_admission_package_validation.json",
        "formal_package_validation_sha256": formal_sha,
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "review_manifest_id": "review-example-001",
        "admission_request_id": "property-admission-draft-001",
        "review_queue_id": "property-review-queue-001",
        "property_candidate_manifest_id": "property-candidates-001",
        "property_precheck_status": "passed",
        "formal_package_validation_status": "passed",
        "dry_run_decision": "passed",
        "phase1_status": "not_run",
        "training_admitted": False,
        "admission_record_count": 2,
        "admit_count": 1,
        "exclude_count": 1,
        "blocked_record_count": 1,
        "admit_record_ids": ["property-candidate-001"],
        "exclude_record_ids": ["property-candidate-002"],
        "blocked_record_ids": ["property-candidate-003"],
        "binding_errors": [] if binding_status != "failed" else ["property_package_binding_failed"],
        "warnings": [] if binding_status == "passed" else ["package_binding_needs_review"],
        "redaction_status": "passed",
    }
