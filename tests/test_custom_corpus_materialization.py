from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_materialization import (
    CustomCorpusMaterializationError,
    load_materialization_plan,
    main,
    materialization_plan_summary,
    sha256_file,
    validate_materialization_plan,
)


def test_example_materialization_plan_loads() -> None:
    plan = load_materialization_plan(
        Path(__file__).parents[1] / "docs" / "examples" / "custom-corpus-materialization-plan.example.json"
    )

    assert plan.schema_version == "custom_corpus_materialization.v1"
    assert plan.materialization_mode == "candidate_only"
    assert plan.materialization_decision == "planned"
    assert len(plan.materialization_records) == 2


def test_valid_planned_candidate_only_plan_summary_counts(tmp_path: Path) -> None:
    path = tmp_path / "materialization_plan.json"
    path.write_text(json.dumps(_plan_payload()), encoding="utf-8")

    plan = load_materialization_plan(path)
    summary = materialization_plan_summary(plan, path=path)

    assert summary["schema_version"] == "custom_corpus_materialization.v1"
    assert summary["materialization_plan_path"] == "materialization_plan.json"
    assert summary["materialization_plan_sha256"] == sha256_file(path)
    assert summary["materialization_plan_id"] == "materialization-plan-001"
    assert summary["materialization_run_id"] == "materialization-run-001"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["dry_run_id"] == "custom-dry-run-example-001"
    assert summary["review_manifest_id"] == "review-example-001"
    assert summary["admission_request_id"] == "admission-example-001"
    assert summary["materialization_mode"] == "candidate_only"
    assert summary["materialization_decision"] == "planned"
    assert summary["dataset_target"] == "example-candidate-target"
    assert summary["package_validation_status"] == "passed"
    assert summary["package_admission_decision"] == "eligible"
    assert summary["dry_run_phase1_status"] == "not_run"
    assert summary["dry_run_dataset_confirmation_confirmed"] is False
    assert summary["dry_run_training_dataset_admitted"] is False
    assert summary["confirmation_present"] is True
    assert summary["confirmation_source"] == "manual-review"
    assert summary["materialization_record_count"] == 2
    assert summary["candidate_record_count"] == 1
    assert summary["excluded_record_count"] == 1


def test_invalid_schema_version_fails() -> None:
    payload = _plan_payload()
    payload["schema_version"] = "custom_corpus_materialization.v0"

    with pytest.raises(CustomCorpusMaterializationError, match="schema_version"):
        validate_materialization_plan(payload)


@pytest.mark.parametrize("mode", ["training", "phase1", "production_admission"])
def test_materialization_mode_other_than_candidate_only_fails(mode: str) -> None:
    payload = _plan_payload()
    payload["materialization_mode"] = mode

    with pytest.raises(CustomCorpusMaterializationError, match="materialization_mode"):
        validate_materialization_plan(payload)


def test_planned_with_package_validation_not_passed_fails() -> None:
    payload = _plan_payload()
    payload["package_validation_status"] = "failed"

    with pytest.raises(CustomCorpusMaterializationError, match="package_validation_status"):
        validate_materialization_plan(payload)


def test_planned_with_package_admission_not_eligible_fails() -> None:
    payload = _plan_payload()
    payload["package_admission_decision"] = "needs_review"

    with pytest.raises(CustomCorpusMaterializationError, match="package_admission_decision"):
        validate_materialization_plan(payload)


def test_planned_materialization_with_phase1_not_run_passes() -> None:
    plan = validate_materialization_plan(_plan_payload())

    assert plan.dry_run_phase1_status == "not_run"


def test_planned_materialization_with_phase1_success_fails() -> None:
    payload = _plan_payload()
    payload["dry_run_phase1_status"] = "success"

    with pytest.raises(CustomCorpusMaterializationError, match="dry_run_phase1_status"):
        validate_materialization_plan(payload)


def test_planned_materialization_with_dataset_confirmation_true_fails() -> None:
    payload = _plan_payload()
    payload["dry_run_dataset_confirmation_confirmed"] = True

    with pytest.raises(CustomCorpusMaterializationError, match="DatasetConfirmation"):
        validate_materialization_plan(payload)


def test_planned_materialization_with_training_dataset_admitted_true_fails() -> None:
    payload = _plan_payload()
    payload["dry_run_training_dataset_admitted"] = True

    with pytest.raises(CustomCorpusMaterializationError, match="training_dataset_admitted"):
        validate_materialization_plan(payload)


def test_planned_materialization_without_explicit_confirmation_fails() -> None:
    payload = _plan_payload()
    payload["confirmation"]["confirmed"] = False

    with pytest.raises(CustomCorpusMaterializationError, match="confirmation"):
        validate_materialization_plan(payload)


def test_confirmation_hash_mismatch_fails() -> None:
    payload = _plan_payload()
    payload["confirmation"]["manifest_sha256"] = "sha256:" + "0" * 64

    with pytest.raises(CustomCorpusMaterializationError, match="manifest_sha256"):
        validate_materialization_plan(payload)


def test_confirmation_id_mismatch_fails() -> None:
    payload = _plan_payload()
    payload["confirmation"]["corpus_id"] = "different-corpus"

    with pytest.raises(CustomCorpusMaterializationError, match="corpus_id"):
        validate_materialization_plan(payload)


@pytest.mark.parametrize("review_decision", ["reject", "needs_review"])
def test_materialize_candidate_with_nonaccepted_review_decision_fails(review_decision: str) -> None:
    payload = _plan_payload()
    payload["materialization_records"][0]["review_decision"] = review_decision

    with pytest.raises(CustomCorpusMaterializationError, match="review_decision"):
        validate_materialization_plan(payload)


@pytest.mark.parametrize("admission_action", ["exclude", "needs_review"])
def test_materialize_candidate_with_nonadmit_admission_action_fails(admission_action: str) -> None:
    payload = _plan_payload()
    payload["materialization_records"][0]["admission_action"] = admission_action

    with pytest.raises(CustomCorpusMaterializationError, match="admission_action"):
        validate_materialization_plan(payload)


def test_materialize_candidate_missing_normalized_value_summary_fails() -> None:
    payload = _plan_payload()
    payload["materialization_records"][0]["normalized_value_summary"] = ""

    with pytest.raises(CustomCorpusMaterializationError, match="normalized_value_summary"):
        validate_materialization_plan(payload)


def test_materialize_candidate_missing_provenance_summary_fails() -> None:
    payload = _plan_payload()
    payload["materialization_records"][0]["provenance_summary"] = ""

    with pytest.raises(CustomCorpusMaterializationError, match="provenance_summary"):
        validate_materialization_plan(payload)


def test_materialize_candidate_missing_materialization_reason_fails() -> None:
    payload = _plan_payload()
    payload["materialization_records"][0]["materialization_reason"] = ""

    with pytest.raises(CustomCorpusMaterializationError, match="materialization_reason"):
        validate_materialization_plan(payload)


def test_exclude_without_exclusion_reason_fails() -> None:
    payload = _plan_payload()
    payload["materialization_records"][1]["exclusion_reason"] = ""

    with pytest.raises(CustomCorpusMaterializationError, match="exclusion_reason"):
        validate_materialization_plan(payload)


def test_duplicate_materialization_record_id_fails() -> None:
    payload = _plan_payload()
    payload["materialization_records"][1]["materialization_record_id"] = "materialization-record-001"

    with pytest.raises(CustomCorpusMaterializationError, match="duplicate materialization_record_id"):
        validate_materialization_plan(payload)


def test_duplicate_materialization_target_fails() -> None:
    payload = _plan_payload()
    payload["materialization_records"][1].update(
        {
            "document_id": payload["materialization_records"][0]["document_id"],
            "record_id": payload["materialization_records"][0]["record_id"],
            "field_name": payload["materialization_records"][0]["field_name"],
        }
    )

    with pytest.raises(CustomCorpusMaterializationError, match="duplicate materialization target"):
        validate_materialization_plan(payload)


def test_sha256_normalization_and_empty_required_sha_fails() -> None:
    payload = _plan_payload()
    payload["source_manifest_sha256"] = "A" * 64
    payload["confirmation"]["manifest_sha256"] = "A" * 64
    payload["materialization_records"][0]["source_artifact_sha256"] = "B" * 64

    plan = validate_materialization_plan(payload)

    assert plan.source_manifest_sha256 == "sha256:" + "a" * 64
    assert plan.confirmation.manifest_sha256 == "sha256:" + "a" * 64
    assert plan.materialization_records[0].source_artifact_sha256 == "sha256:" + "b" * 64

    missing = _plan_payload()
    missing["source_manifest_sha256"] = ""
    with pytest.raises(CustomCorpusMaterializationError, match="source_manifest_sha256"):
        validate_materialization_plan(missing)


def test_private_path_like_text_fails_without_leaking_path() -> None:
    payload = _plan_payload()
    payload["materialization_records"][0]["provenance_summary"] = "/Users/operator/private/paper"

    with pytest.raises(CustomCorpusMaterializationError) as excinfo:
        validate_materialization_plan(payload)

    message = str(excinfo.value)
    assert "private path" in message
    assert "/Users/operator" not in message


def test_credential_like_text_fails_without_leaking_secret() -> None:
    payload = _plan_payload()
    payload["materialization_records"][0]["notes"] = "contains token abc123"

    with pytest.raises(CustomCorpusMaterializationError) as excinfo:
        validate_materialization_plan(payload)

    message = str(excinfo.value).lower()
    assert "credential" in message
    assert "abc123" not in message


def test_dataset_target_path_like_value_fails() -> None:
    payload = _plan_payload()
    payload["dataset_target"] = "/tmp/materialized.csv"

    with pytest.raises(CustomCorpusMaterializationError, match="dataset_target"):
        validate_materialization_plan(payload)


def test_created_by_private_email_fails_unless_redacted() -> None:
    payload = _plan_payload()
    payload["created_by"] = "operator@example.org"
    with pytest.raises(CustomCorpusMaterializationError, match="created_by"):
        validate_materialization_plan(payload)

    payload["created_by"] = "operator-email-redacted"
    plan = validate_materialization_plan(payload)
    assert plan.created_by == "operator-email-redacted"


def test_cli_prints_safe_summary_and_exits_0_on_valid(tmp_path: Path) -> None:
    plan_path = tmp_path / "materialization_plan.json"
    output_summary = tmp_path / "summary.json"
    plan_path.write_text(json.dumps(_plan_payload()), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        ["--materialization-plan", str(plan_path), "--output-summary", str(output_summary)],
        stdout=stdout,
        stderr=stderr,
    )
    printed = json.loads(stdout.getvalue())
    written = json.loads(output_summary.read_text(encoding="utf-8"))

    assert code == 0
    assert printed["materialization_decision"] == "planned"
    assert printed["candidate_record_count"] == 1
    assert written == printed
    assert str(tmp_path) not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_cli_exits_1_on_invalid_without_leaking_sensitive_value(tmp_path: Path) -> None:
    plan_path = tmp_path / "materialization_plan.json"
    payload = _plan_payload()
    payload["materialization_records"][0]["notes"] = "password abc123"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(["--materialization-plan", str(plan_path)], stdout=stdout, stderr=stderr)

    assert code == 1
    assert stdout.getvalue() == ""
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stderr.getvalue()
    assert "credential" in stderr.getvalue().lower()


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
