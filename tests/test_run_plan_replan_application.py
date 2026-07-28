from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai4s_agent.run_plan_replan_application import (
    BlockedAcknowledgement,
    CompiledReplanApplication,
    ReplanApplicationRecord,
    ReplanApplicationRequest,
    ResumeIntent,
    ReviewableRunPlanPatch,
    validate_and_compile_replan_application,
    validate_selected_operation_ids_for_patch,
)
from ai4s_agent.run_plan_replan_proposal import RunPlanReplanProposal
from ai4s_agent.run_plan_state_fingerprint import ResumeStateBinding


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


def _proposal(
    proposed_action: str = "rerun_task",
    *,
    patch: ReviewableRunPlanPatch | None = None,
) -> RunPlanReplanProposal:
    reviewable_patch = patch or _patch()
    return RunPlanReplanProposal(
        verifier_decision="rerun_recommended",
        proposed_action=proposed_action,  # type: ignore[arg-type]
        affected_tasks=["train_model"],
        rationale=["Model metrics are weak enough to recommend a rerun."],
        required_user_decisions=["Approve rerun before any queued execution."],
        proposed_run_plan_patch=reviewable_patch.model_dump(mode="json"),
        executable=False,
        source_finding_ids=["poor_model_metrics_xxx"],
    )


def _request(
    selected_action: str = "rerun_task",
    *,
    selected_operation_ids: list[str] | None = None,
) -> ReplanApplicationRequest:
    return ReplanApplicationRequest(
        project_id="proj-a",
        run_id="run-a",
        proposal_artifact_ref="review/replan_proposal.json",
        proposal_hash="sha256:abc123",
        selected_action=selected_action,  # type: ignore[arg-type]
        selected_operation_ids=selected_operation_ids or ["op_000001"],
    )


def _binding() -> ResumeStateBinding:
    return ResumeStateBinding(
        run_plan_fingerprint="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        stage_fingerprint="sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        stage="train_model",
        stage_status="WAITING_USER",
        execution_snapshot_id="snapshot-1",
        execution_snapshot_hash="sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
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
        resume_state_binding=_binding(),
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
            resume_state_binding=_binding(),
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
            resume_state_binding=_binding(),
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
        resume_state_binding=_binding(),
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


def test_compile_replan_application_rerun_task_without_graph_change_returns_resume_intent() -> None:
    compiled = validate_and_compile_replan_application(
        _request("rerun_task"),
        _patch(_operation("op_000001")),
        _proposal("rerun_task"),
    )

    assert isinstance(compiled, CompiledReplanApplication)
    assert compiled.result_type == "resume_intent"
    assert compiled.selected_action == "rerun_task"
    assert [operation.operation_id for operation in compiled.selected_operations] == ["op_000001"]
    assert "gate_replan_rerun_task" in compiled.required_gates
    assert compiled.executable is False


def test_compile_replan_application_adjust_targets_returns_run_plan_revision() -> None:
    patch = _patch({**_operation("op_000001"), "op": "adjust_targets", "task_id": "plan_targets"})

    compiled = validate_and_compile_replan_application(
        _request("adjust_targets"),
        patch,
        _proposal("adjust_targets", patch=patch),
    )

    assert compiled.result_type == "run_plan_revision"
    assert "gate_replan_adjust_targets" in compiled.required_gates
    assert compiled.executable is False


def test_compile_replan_application_collect_more_data_returns_run_plan_revision() -> None:
    patch = _patch({**_operation("op_000001"), "op": "collect_more_data", "task_id": "collect_data"})

    compiled = validate_and_compile_replan_application(
        _request("collect_more_data"),
        patch,
        _proposal("collect_more_data", patch=patch),
    )

    assert compiled.result_type == "run_plan_revision"
    assert "gate_replan_collect_more_data" in compiled.required_gates
    assert compiled.executable is False


def test_compile_replan_application_block_returns_blocked_acknowledgement() -> None:
    patch = _patch({**_operation("op_000001"), "op": "block", "task_id": "artifact_registry"})

    compiled = validate_and_compile_replan_application(
        _request("block"),
        patch,
        _proposal("block", patch=patch),
    )

    assert compiled.result_type == "blocked_acknowledgement"
    assert compiled.required_gates == []
    assert compiled.executable is False


def test_compile_replan_application_allows_downgrade_to_request_review() -> None:
    compiled = validate_and_compile_replan_application(
        _request("request_review"),
        _patch(_operation("op_000001")),
        _proposal("rerun_task"),
    )

    assert compiled.selected_action == "request_review"
    assert compiled.proposal_action == "rerun_task"
    assert compiled.result_type == "resume_intent"
    assert "selected action downgraded to request_review" in compiled.validation_findings


def test_compile_replan_application_allows_downgrade_to_block() -> None:
    compiled = validate_and_compile_replan_application(
        _request("block"),
        _patch(_operation("op_000001")),
        _proposal("rerun_task"),
    )

    assert compiled.selected_action == "block"
    assert compiled.proposal_action == "rerun_task"
    assert compiled.result_type == "blocked_acknowledgement"
    assert "selected action downgraded to block" in compiled.validation_findings


def test_compile_replan_application_rejects_action_escalation() -> None:
    with pytest.raises(ValueError, match="selected_action must match proposal action"):
        validate_and_compile_replan_application(
            _request("adjust_targets"),
            _patch(_operation("op_000001")),
            _proposal("rerun_task"),
        )


def test_compile_replan_application_rejects_unknown_selected_operation_id() -> None:
    with pytest.raises(ValueError, match="unknown operation_id"):
        validate_and_compile_replan_application(
            _request("rerun_task", selected_operation_ids=["op_missing"]),
            _patch(_operation("op_000001")),
            _proposal("rerun_task"),
        )


def test_compile_replan_application_rejects_unknown_patch_operation() -> None:
    with pytest.raises(ValidationError, match="Input should be"):
        ReviewableRunPlanPatch.model_validate(
            {
                "schema_version": "reviewable_run_plan_patch.v1",
                "applied": False,
                "operations": [{**_operation("op_000001"), "op": "delete_files"}],
            }
        )


def test_compile_replan_application_rejects_executable_true_output() -> None:
    with pytest.raises(ValidationError, match="executable must remain false"):
        CompiledReplanApplication(
            project_id="proj-a",
            run_id="run-a",
            proposal_hash="sha256:abc123",
            proposal_action="rerun_task",
            selected_action="rerun_task",
            selected_operation_ids=["op_000001"],
            selected_operations=[_operation("op_000001")],
            result_type="resume_intent",
            required_gates=["gate_replan_rerun_task"],
            validation_findings=[],
            executable=True,
        )
