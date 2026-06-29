from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_materialization_planner import (
    PLANNED_OUTPUT_LABELS,
    PLANNED_ROLLBACK_LABELS,
    main,
    plan_materialization,
)


def test_example_materialization_plan_produces_planned_summary() -> None:
    plan_path = Path(__file__).parents[1] / "docs" / "examples" / "custom-corpus-materialization-plan.example.json"

    summary = plan_materialization(plan_path)

    assert summary["schema_version"] == "custom_corpus_materialization_planner.v1"
    assert summary["planner_status"] == "planned"
    assert summary["materialization_plan_path"] == "custom-corpus-materialization-plan.example.json"
    assert summary["materialization_plan_sha256"].startswith("sha256:")
    assert summary["materialization_plan_id"] == "materialization-plan-example-001"
    assert summary["materialization_run_id"] == "materialization-run-example-001"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["dry_run_id"] == "custom-dry-run-example-001"
    assert summary["review_manifest_id"] == "review-example-001"
    assert summary["admission_request_id"] == "admission-example-001"
    assert summary["dataset_target"] == "example-candidate-target"
    assert summary["materialization_mode"] == "candidate_only"
    assert summary["materialization_decision"] == "planned"
    assert summary["package_validation_status"] == "passed"
    assert summary["package_admission_decision"] == "eligible"
    assert summary["dry_run_phase1_status"] == "not_run"
    assert summary["dry_run_dataset_confirmation_confirmed"] is False
    assert summary["dry_run_training_dataset_admitted"] is False
    assert summary["confirmation_present"] is True
    assert summary["candidate_record_count"] == 1
    assert summary["excluded_record_count"] == 1
    assert summary["candidate_record_ids"] == ["materialization-record-001"]
    assert summary["excluded_record_ids"] == ["materialization-record-002"]
    assert summary["blocking_reasons"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"


def test_planner_summary_uses_safe_basename_not_temp_path(tmp_path: Path) -> None:
    plan_path = tmp_path / "materialization_plan.json"
    plan_path.write_text(json.dumps(_plan_payload()), encoding="utf-8")

    summary = plan_materialization(plan_path)
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["materialization_plan_path"] == "materialization_plan.json"
    assert str(tmp_path) not in serialized


def test_planner_summary_includes_output_and_rollback_labels(tmp_path: Path) -> None:
    plan_path = tmp_path / "materialization_plan.json"
    plan_path.write_text(json.dumps(_plan_payload()), encoding="utf-8")

    summary = plan_materialization(plan_path)

    assert summary["planned_output_labels"] == list(PLANNED_OUTPUT_LABELS)
    assert summary["planned_rollback_labels"] == list(PLANNED_ROLLBACK_LABELS)


def test_planner_does_not_create_candidate_artifacts(tmp_path: Path) -> None:
    plan_path = tmp_path / "materialization_plan.json"
    plan_path.write_text(json.dumps(_plan_payload()), encoding="utf-8")

    plan_materialization(plan_path)

    assert not (tmp_path / "materialization_summary.json").exists()
    assert not (tmp_path / "materialized_records.jsonl").exists()
    assert not (tmp_path / "materialized_records.csv").exists()
    assert not (tmp_path / "provenance_bindings.jsonl").exists()
    assert not (tmp_path / "rollback_manifest.json").exists()
    assert not (tmp_path / "redacted_evidence_summary.md").exists()


def test_cli_writes_optional_json_and_markdown_summaries(tmp_path: Path) -> None:
    plan_path = tmp_path / "materialization_plan.json"
    output_summary = tmp_path / "planner_summary.json"
    output_markdown = tmp_path / "planner_summary.md"
    plan_path.write_text(json.dumps(_plan_payload()), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--materialization-plan",
            str(plan_path),
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
    assert "No candidate artifacts created" in markdown
    assert "No training data admitted" in markdown
    assert "No Phase 1" in markdown
    assert "No DatasetConfirmation change" in markdown
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in markdown
    assert stderr.getvalue() == ""


def test_blocked_materialization_plan_produces_blocked_status_and_exits_0(tmp_path: Path) -> None:
    plan_path = tmp_path / "blocked_plan.json"
    payload = _plan_payload()
    payload["materialization_decision"] = "blocked"
    payload["package_validation_status"] = "failed"
    payload["package_admission_decision"] = "ineligible"
    payload["confirmation"]["confirmed"] = False
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(["--materialization-plan", str(plan_path)], stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["planner_status"] == "blocked"
    assert summary["materialization_decision"] == "blocked"
    assert summary["blocking_reasons"] == ["materialization_decision_blocked"]
    assert stderr.getvalue() == ""


def test_invalid_materialization_plan_exits_1_without_leaking_sensitive_value(tmp_path: Path) -> None:
    plan_path = tmp_path / "invalid_plan.json"
    payload = _plan_payload()
    payload["materialization_records"][0]["notes"] = "token abc123"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(["--materialization-plan", str(plan_path)], stdout=stdout, stderr=stderr)

    assert code == 1
    assert stdout.getvalue() == ""
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()
    assert "credential" in stderr.getvalue().lower()


def test_summary_excludes_normalized_and_provenance_summaries_by_default(tmp_path: Path) -> None:
    plan_path = tmp_path / "materialization_plan.json"
    plan_path.write_text(json.dumps(_plan_payload()), encoding="utf-8")

    summary = plan_materialization(plan_path)
    serialized = json.dumps(summary, sort_keys=True)

    assert "short redacted normalized value" not in serialized
    assert "short redacted provenance summary" not in serialized
    assert "normalized_value_summary" not in serialized
    assert "provenance_summary" not in serialized


def test_summary_redaction_fail_closed_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan_path = tmp_path / "materialization_plan.json"
    plan_path.write_text(json.dumps(_plan_payload()), encoding="utf-8")

    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_materialization_planner.PLANNED_OUTPUT_LABELS",
        ("materialization_summary.json", "/tmp/operator/private/output.json"),
    )
    summary = plan_materialization(plan_path)

    assert summary == {
        "schema_version": "custom_corpus_materialization_planner.v1",
        "planner_status": "blocked",
        "blocking_reasons": ["planner_summary_redaction_failed"],
        "redaction_status": "failed",
    }


def test_cli_stdout_is_valid_json(tmp_path: Path) -> None:
    plan_path = tmp_path / "materialization_plan.json"
    plan_path.write_text(json.dumps(_plan_payload()), encoding="utf-8")
    stdout = io.StringIO()

    code = main(["--materialization-plan", str(plan_path)], stdout=stdout, stderr=io.StringIO())

    assert code == 0
    assert json.loads(stdout.getvalue())["schema_version"] == "custom_corpus_materialization_planner.v1"


def test_no_forbidden_runtime_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan_path = tmp_path / "materialization_plan.json"
    plan_path.write_text(json.dumps(_plan_payload()), encoding="utf-8")
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        if name in {
            "ai4s_agent.workflows.corpus_to_phase1_workflow",
            "ai4s_agent.document_parse_service",
            "ai4s_agent.document_parse",
        }:
            raise AssertionError(f"forbidden import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracking_import)

    summary = plan_materialization(plan_path)

    assert summary["planner_status"] == "planned"
    assert not any("corpus_to_phase1_workflow" in name for name in imported_modules)


def _plan_payload() -> dict[str, object]:
    source_manifest_sha = "sha256:" + "a" * 64
    dry_run_sha = "sha256:" + "b" * 64
    review_sha = "sha256:" + "c" * 64
    admission_sha = "sha256:" + "d" * 64
    package_sha = "sha256:" + "e" * 64
    return {
        "schema_version": "custom_corpus_materialization.v1",
        "materialization_plan_id": "materialization-plan-001",
        "materialization_run_id": "materialization-run-001",
        "created_at": "2026-06-29T00:00:00Z",
        "created_by": "operator-redacted",
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "review_manifest_id": "review-example-001",
        "admission_request_id": "admission-example-001",
        "materialization_mode": "candidate_only",
        "materialization_decision": "planned",
        "dataset_target": "example-candidate-target",
        "source_manifest_sha256": source_manifest_sha,
        "source_dry_run_report_sha256": dry_run_sha,
        "source_review_manifest_sha256": review_sha,
        "source_admission_request_sha256": admission_sha,
        "source_package_validation_sha256": package_sha,
        "package_validation_status": "passed",
        "package_admission_decision": "eligible",
        "dry_run_phase1_status": "not_run",
        "dry_run_dataset_confirmation_confirmed": False,
        "dry_run_training_dataset_admitted": False,
        "confirmation": {
            "confirmed": True,
            "confirmed_by": "operator-redacted",
            "confirmed_at": "2026-06-29T00:00:00Z",
            "confirmation_source": "manual-review",
            "manifest_sha256": source_manifest_sha,
            "dry_run_report_sha256": dry_run_sha,
            "review_manifest_sha256": review_sha,
            "admission_request_sha256": admission_sha,
            "package_validation_sha256": package_sha,
            "corpus_id": "example-public-corpus",
            "dry_run_id": "custom-dry-run-example-001",
            "review_manifest_id": "review-example-001",
            "admission_request_id": "admission-example-001",
            "reason": "operator confirmed candidate-only materialization planning",
        },
        "materialization_records": [
            _record(
                "materialization-record-001",
                "admission-record-001",
                "review-record-001",
                "record-example-001",
                "materialize_candidate",
                "admit",
                "accept",
                admission_sha,
                package_sha,
            ),
            _record(
                "materialization-record-002",
                "admission-record-002",
                "review-record-002",
                "record-example-002",
                "exclude",
                "exclude",
                "reject",
                admission_sha,
                package_sha,
                exclusion_reason="record was excluded by admission request",
            ),
        ],
        "rollback_policy": "delete generated candidate artifacts only",
        "redaction_policy": "redacted evidence only",
    }


def _record(
    materialization_record_id: str,
    admission_record_id: str,
    review_id: str,
    record_id: str,
    action: str,
    admission_action: str,
    review_decision: str,
    admission_sha: str,
    package_sha: str,
    *,
    exclusion_reason: str = "",
) -> dict[str, str]:
    return {
        "materialization_record_id": materialization_record_id,
        "corpus_id": "example-public-corpus",
        "dry_run_id": "custom-dry-run-example-001",
        "review_manifest_id": "review-example-001",
        "admission_request_id": "admission-example-001",
        "admission_record_id": admission_record_id,
        "review_id": review_id,
        "document_id": "doc-example-001",
        "record_id": record_id,
        "field_name": "plqy",
        "action": action,
        "admission_action": admission_action,
        "review_decision": review_decision,
        "source_artifact_sha256": "sha256:" + "f" * 64,
        "review_artifact_sha256": "sha256:" + "1" * 64,
        "admission_request_sha256": admission_sha,
        "package_validation_sha256": package_sha,
        "normalized_value_summary": "short redacted normalized value",
        "provenance_summary": "short redacted provenance summary",
        "materialization_reason": "candidate-only materialization planned" if action == "materialize_candidate" else "",
        "exclusion_reason": exclusion_reason,
        "notes": "",
    }
