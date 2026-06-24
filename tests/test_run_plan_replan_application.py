from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai4s_agent.run_plan_replan_application import (
    BlockedAcknowledgement,
    ReplanApplicationRecord,
    ReplanApplicationRequest,
    ResumeIntent,
    ReviewableRunPlanPatch,
    validate_selected_operation_ids_for_patch,
)


def _operation(operation_id: str = "op_000001") -> dict[str, str]:
    return {
        "operation_id": operation_id,
        "op": "rerun_task",
        "task_id": "train_model",
        "source_finding_id": "poor_model_metrics_xxx",
        "category": "poor_model_metrics",
        "reason": "Model metrics are weak enough to recommend a rerun.",
    }


def _patch(*operations: dict[str, str]) -> ReviewableRunPlanPatch:
    return ReviewableRunPlanPatch(
        schema_version="reviewable_run_plan_patch.v1",
        applied=False,
        operations=list(operations or [_operation()]),
    )


def test_reviewable_run_plan_patch_requires_explicit_operation_id() -> None:
    payload = {
        "schema_version": "reviewable_run_plan_patch.v1",
        "applied": False,
        "operations": [
            {
                "op": "rerun_task",
                "task_id": "train_model",
                "source_finding_id": "poor_model_metrics_xxx",
                "category": "poor_model_metrics",
                "reason": "Model metrics are weak enough to recommend a rerun.",
            }
        ],
    }

    with pytest.raises(ValidationError, match="operation_id"):
        ReviewableRunPlanPatch.model_validate(payload)


def test_reviewable_run_plan_patch_rejects_duplicate_operation_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate operation_id"):
        _patch(_operation("op_000001"), _operation("op_000001"))


def test_reviewable_run_plan_patch_rejects_applied_true() -> None:
    with pytest.raises(ValidationError, match="applied=false"):
        ReviewableRunPlanPatch.model_validate(
            {
                "schema_version": "reviewable_run_plan_patch.v1",
                "applied": True,
                "operations": [_operation()],
            }
        )


def test_replan_application_request_rejects_duplicate_selected_operation_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate selected_operation_ids"):
        ReplanApplicationRequest(
            project_id="proj-a",
            run_id="run-a",
            proposal_artifact_ref="review/replan_proposal.json",
            proposal_hash="sha256:abc123",
            selected_action="rerun_task",
            selected_operation_ids=["op_000001", "op_000001"],
        )


def test_selected_operation_ids_must_exist_in_patch_operations() -> None:
    patch = _patch(_operation("op_000001"))
    request = ReplanApplicationRequest(
        project_id="proj-a",
        run_id="run-a",
        proposal_artifact_ref="review/replan_proposal.json",
        proposal_hash="sha256:abc123",
        selected_action="rerun_task",
        selected_operation_ids=["op_missing"],
    )

    with pytest.raises(ValueError, match="unknown operation_id"):
        validate_selected_operation_ids_for_patch(request, patch)


def test_selected_operation_ids_validation_returns_clean_ids() -> None:
    patch = _patch(_operation("op_000001"), _operation("op_000002"))
    request = ReplanApplicationRequest(
        project_id="proj-a",
        run_id="run-a",
        proposal_artifact_ref="review/replan_proposal.json",
        proposal_hash="sha256:abc123",
        selected_action="rerun_task",
        selected_operation_ids=[" op_000001 ", "op_000002"],
    )

    assert validate_selected_operation_ids_for_patch(request, patch) == ["op_000001", "op_000002"]


def test_replan_application_record_is_applied_but_not_executable() -> None:
    record = ReplanApplicationRecord(
        application_id="replan-application-run-a-001",
        project_id="proj-a",
        run_id="run-a",
        proposal_artifact_ref="review/replan_proposal.json",
        proposal_hash="sha256:abc123",
        selected_action="rerun_task",
        selected_operation_ids=["op_000001"],
        result_type="resume_intent",
        result_ref="review/replan_resume_intent.json",
        actor="review-user",
        actor_source="header:X-Actor",
    )

    assert record.applied is True
    assert record.executable is False


def test_replan_application_record_rejects_executable_true() -> None:
    with pytest.raises(ValidationError, match="executable must remain false"):
        ReplanApplicationRecord(
            application_id="replan-application-run-a-001",
            project_id="proj-a",
            run_id="run-a",
            proposal_artifact_ref="review/replan_proposal.json",
            proposal_hash="sha256:abc123",
            selected_action="rerun_task",
            selected_operation_ids=["op_000001"],
            result_type="resume_intent",
            result_ref="review/replan_resume_intent.json",
            actor="review-user",
            actor_source="header:X-Actor",
            executable=True,
        )


def test_resume_intent_is_non_executable_and_rejects_block_action() -> None:
    with pytest.raises(ValidationError, match="block cannot create a resume intent"):
        ResumeIntent(
            intent_id="resume-run-a-001",
            project_id="proj-a",
            run_id="run-a",
            source_application_id="replan-application-run-a-001",
            action="block",
            affected_tasks=["run_plan_execute"],
            reason="Blocked runs cannot resume.",
            created_by="review-user",
            actor_source="header:X-Actor",
        )

    intent = ResumeIntent(
        intent_id="resume-run-a-001",
        project_id="proj-a",
        run_id="run-a",
        source_application_id="replan-application-run-a-001",
        action="rerun_task",
        affected_tasks=["train_model"],
        rerun_tasks=["train_model"],
        reason="User approved rerun.",
        created_by="review-user",
        actor_source="header:X-Actor",
    )

    assert intent.executable is False


def test_blocked_acknowledgement_is_non_executable() -> None:
    acknowledgement = BlockedAcknowledgement(
        application_id="replan-application-run-a-001",
        project_id="proj-a",
        run_id="run-a",
        blocked_reason="Registered artifacts are missing on disk.",
        source_finding_ids=["missing_artifact_1"],
        required_user_decisions=["Restore the missing artifact."],
        created_by="review-user",
        actor_source="header:X-Actor",
    )

    assert acknowledgement.executable is False


def test_blank_proposal_hash_is_rejected() -> None:
    with pytest.raises(ValidationError, match="proposal_hash"):
        ReplanApplicationRequest(
            project_id="proj-a",
            run_id="run-a",
            proposal_artifact_ref="review/replan_proposal.json",
            proposal_hash=" ",
            selected_action="rerun_task",
            selected_operation_ids=["op_000001"],
        )
