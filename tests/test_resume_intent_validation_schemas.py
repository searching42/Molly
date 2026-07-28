from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai4s_agent.run_plan_resume_intent_validation import (
    DEFAULT_RESUME_INTENT_ARTIFACT_REFS,
    ResumeIntentValidationRequest,
    ResumeIntentValidationResult,
)


def _artifact_refs() -> dict[str, str]:
    return dict(DEFAULT_RESUME_INTENT_ARTIFACT_REFS)


def test_resume_intent_validation_request_accepts_fixed_review_refs() -> None:
    request = ResumeIntentValidationRequest(
        project_id="proj-a",
        run_id="run-a",
        intent_id="resume-replan-application-run-a-abc123",
        source_application_id="replan-application-run-a-abc123",
        proposal_hash="sha256:" + "a" * 64,
        artifact_refs=_artifact_refs(),
        approved_gates=[" gate_replan_rerun_task ", "gate_3_train_config"],
        actor="review-user",
        actor_source="header:X-Actor",
    )

    assert request.executable is False
    assert request.artifact_refs == {
        "replan_application_record": "review/replan_application_record.json",
        "replan_resume_intent": "review/replan_resume_intent.json",
        "replan_proposal": "review/replan_proposal.json",
    }
    assert request.approved_gates == ["gate_replan_rerun_task", "gate_3_train_config"]


def test_resume_intent_validation_request_rejects_executable_true() -> None:
    with pytest.raises(ValidationError, match="resume intent validation request is not executable"):
        ResumeIntentValidationRequest(
            project_id="proj-a",
            run_id="run-a",
            intent_id="resume-a",
            source_application_id="replan-a",
            proposal_hash="sha256:" + "a" * 64,
            artifact_refs=_artifact_refs(),
            executable=True,
        )


def test_resume_intent_validation_request_rejects_missing_required_artifact_refs() -> None:
    refs = _artifact_refs()
    refs.pop("replan_proposal")

    with pytest.raises(ValidationError, match="artifact_refs missing required keys"):
        ResumeIntentValidationRequest(
            project_id="proj-a",
            run_id="run-a",
            intent_id="resume-a",
            source_application_id="replan-a",
            proposal_hash="sha256:" + "a" * 64,
            artifact_refs=refs,
        )


def test_resume_intent_validation_request_rejects_unsafe_artifact_refs() -> None:
    refs = _artifact_refs()
    refs["replan_resume_intent"] = "../outside.json"

    with pytest.raises(ValidationError, match="artifact ref must stay under run directory"):
        ResumeIntentValidationRequest(
            project_id="proj-a",
            run_id="run-a",
            intent_id="resume-a",
            source_application_id="replan-a",
            proposal_hash="sha256:" + "a" * 64,
            artifact_refs=refs,
        )


def test_resume_intent_validation_request_rejects_non_sha256_proposal_hash() -> None:
    with pytest.raises(ValidationError, match="proposal_hash must start with sha256:"):
        ResumeIntentValidationRequest(
            project_id="proj-a",
            run_id="run-a",
            intent_id="resume-a",
            source_application_id="replan-a",
            proposal_hash="md5:abc",
            artifact_refs=_artifact_refs(),
        )


def test_resume_intent_validation_result_accepts_resume_eligible_summary() -> None:
    result = ResumeIntentValidationResult(
        ok=True,
        project_id="proj-a",
        run_id="run-a",
        intent_id="resume-replan-application-run-a-abc123",
        source_application_id="replan-application-run-a-abc123",
        proposal_hash="sha256:" + "a" * 64,
        decision="resume_eligible",
        required_gates=["gate_replan_rerun_task", "gate_3_train_config"],
        approved_gates=["gate_replan_rerun_task", "gate_3_train_config"],
        rerun_tasks=["train_model"],
        resume_from_task="train_model",
        artifact_refs=_artifact_refs(),
        validation_findings=["proposal_hash_valid", "rerun_tasks_present_in_current_run_plan"],
    )

    assert result.executable is False
    assert result.error is None
    assert result.required_gates == ["gate_replan_rerun_task", "gate_3_train_config"]
    assert result.rerun_tasks == ["train_model"]
    assert result.validation_findings == ["proposal_hash_valid", "rerun_tasks_present_in_current_run_plan"]


def test_resume_intent_validation_result_accepts_needs_gate_approval_without_error() -> None:
    result = ResumeIntentValidationResult(
        ok=True,
        project_id="proj-a",
        run_id="run-a",
        intent_id="resume-a",
        source_application_id="replan-a",
        proposal_hash="sha256:" + "a" * 64,
        decision="needs_gate_approval",
        required_gates=["gate_3_train_config"],
        rerun_tasks=["train_model"],
        artifact_refs=_artifact_refs(),
        validation_findings=["missing_gate_approval:gate_3_train_config"],
    )

    assert result.ok is True
    assert result.error is None
    assert result.executable is False


def test_resume_intent_validation_result_rejects_unknown_decision() -> None:
    with pytest.raises(ValidationError):
        ResumeIntentValidationResult(
            ok=False,
            project_id="proj-a",
            run_id="run-a",
            intent_id="resume-a",
            source_application_id="replan-a",
            proposal_hash="sha256:" + "a" * 64,
            decision="execute_now",
            artifact_refs=_artifact_refs(),
            validation_findings=["bad decision"],
            error={"type": "invalid_intent", "message": "unsupported decision"},
        )


def test_resume_intent_validation_result_rejects_error_without_type_or_message() -> None:
    with pytest.raises(ValidationError, match="error must include non-empty type and message"):
        ResumeIntentValidationResult(
            ok=False,
            project_id="proj-a",
            run_id="run-a",
            intent_id="resume-a",
            source_application_id="replan-a",
            proposal_hash="sha256:" + "a" * 64,
            decision="invalid_intent",
            artifact_refs=_artifact_refs(),
            validation_findings=["invalid"],
            error={"type": "invalid_intent"},
        )


def test_resume_intent_validation_result_requires_error_for_failed_decisions() -> None:
    with pytest.raises(ValidationError, match="stale_intent results require an error"):
        ResumeIntentValidationResult(
            ok=False,
            project_id="proj-a",
            run_id="run-a",
            intent_id="resume-a",
            source_application_id="replan-a",
            proposal_hash="sha256:" + "a" * 64,
            decision="stale_intent",
            artifact_refs=_artifact_refs(),
            validation_findings=["proposal_hash_mismatch"],
        )


def test_resume_intent_validation_result_rejects_executable_true() -> None:
    with pytest.raises(ValidationError, match="resume intent validation result is not executable"):
        ResumeIntentValidationResult(
            ok=True,
            project_id="proj-a",
            run_id="run-a",
            intent_id="resume-a",
            source_application_id="replan-a",
            proposal_hash="sha256:" + "a" * 64,
            decision="resume_eligible",
            artifact_refs=_artifact_refs(),
            validation_findings=["ok"],
            executable=True,
        )
