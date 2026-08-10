from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest

from ai4s_agent.schemas import (
    AgentHarnessControllerAction,
    AgentHarnessControllerStatus,
    AgentAutonomyL2MaterialityClass,
    AgentPlanDiffChange,
    AgentPlanFeedbackRequest,
    AgentReplanLLMResponse,
    _agent_digest,
)
from ai4s_agent.scientific_agent_autonomy_l2 import (
    AUTONOMY_L2_MATERIALITY_POLICY_DIGEST,
    AUTONOMY_L2_MATERIALITY_POLICY_VERSION,
    AUTONOMY_L2_REVIEWED_DIFF_DIMENSIONS,
    AutonomyL2MaterialityError,
    classify_plan_revision_materiality,
    verify_plan_revision_materiality_decision,
)
from tests.test_scientific_agent_replanner import (
    CountingProvider,
    _baseline,
    _revision_payload,
)
from tests.test_scientific_agent_conversation_session import (
    _start_waiting_gate_session_with_client,
    _typed_controller_inspection_variant,
)


def _feedback_revision(tmp_path: Path, response: AgentReplanLLMResponse):
    _storage, proposal_store, authorization_service, baseline, authorization, service = _baseline(
        tmp_path
    )
    feedback = service.create_feedback(
        project_id="project-1",
        request=AgentPlanFeedbackRequest(
            run_id="run-1",
            client_request_id="feedback-l2",
            feedback="The current run needs a bounded revision.",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    provider = CountingProvider(response=response.model_dump(mode="json"))
    result = service.create_revision(
        project_id="project-1",
        payload=_revision_payload(
            baseline,
            authorization,
            feedback,
            request_id="l2-revision",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        provider=provider,
    )
    current_baseline = proposal_store.read(
        project_id="project-1",
        proposal_id=baseline.proposal_id,
        verify_current=False,
    ).proposal
    current_authorization = authorization_service.verify_authorization(
        project_id="project-1",
        authorization_id=authorization.authorization_id,
        verify_current=False,
    )
    return result.proposal, current_baseline, current_authorization, provider


def test_l2_policy_materiality_is_empty_diff_based_and_deterministic(tmp_path: Path) -> None:
    revision, baseline, authorization, provider = _feedback_revision(
        tmp_path,
        AgentReplanLLMResponse(
            rationale_summary="Use a smaller candidate set.",
            option_patch={"generate_candidates": {"count": 4}},
        ),
    )
    assert provider.calls == 1
    decision = classify_plan_revision_materiality(
        revision,
        baseline_proposal=baseline,
        baseline_authorization=authorization,
    )
    assert decision.classification is AgentAutonomyL2MaterialityClass.MATERIAL
    assert decision.material_change is True
    assert decision.fresh_permission_required is True
    assert decision.fresh_authorization_required is True
    assert decision.policy_version == AUTONOMY_L2_MATERIALITY_POLICY_VERSION
    assert decision.policy_digest == AUTONOMY_L2_MATERIALITY_POLICY_DIGEST
    assert decision.decision_digest == _agent_digest(decision.semantic_material())
    assert decision.authorization_scope_equal is True

    replay = classify_plan_revision_materiality(
        revision,
        baseline_proposal=baseline,
        baseline_authorization=authorization,
    )
    assert replay.model_dump(mode="json") == decision.model_dump(mode="json")
    assert verify_plan_revision_materiality_decision(
        decision,
        revision,
        baseline_proposal=baseline,
        baseline_authorization=authorization,
    ).model_dump(mode="json") == decision.model_dump(mode="json")


def test_l2_no_change_is_non_material_and_does_not_bind_successor(tmp_path: Path) -> None:
    revision, baseline, authorization, _provider = _feedback_revision(
        tmp_path,
        AgentReplanLLMResponse(
            rationale_summary="No deterministic change is required.",
            no_change=True,
        ),
    )
    assert revision.successor_candidate is None
    decision = classify_plan_revision_materiality(
        revision,
        baseline_proposal=baseline,
        baseline_authorization=authorization,
    )
    assert decision.classification is AgentAutonomyL2MaterialityClass.NON_MATERIAL
    assert decision.material_change is False
    assert decision.successor_candidate_id == ""
    assert decision.successor_proposal_digest == ""
    assert decision.baseline_semantic_plan_digest == decision.successor_semantic_plan_digest
    assert decision.baseline_projection_digest == decision.successor_projection_digest
    assert decision.fresh_permission_required is False
    assert decision.fresh_authorization_required is False


def test_l2_serialized_decision_and_unknown_dimension_fail_closed(tmp_path: Path) -> None:
    revision, baseline, authorization, _provider = _feedback_revision(
        tmp_path,
        AgentReplanLLMResponse(
            rationale_summary="Change the candidate count.",
            option_patch={"generate_candidates": {"count": 4}},
        ),
    )
    decision = classify_plan_revision_materiality(
        revision,
        baseline_proposal=baseline,
        baseline_authorization=authorization,
    )
    forged = decision.model_copy(
        update={"classification": AgentAutonomyL2MaterialityClass.NON_MATERIAL}
    )
    with pytest.raises(AutonomyL2MaterialityError):
        verify_plan_revision_materiality_decision(
            forged,
            revision,
            baseline_proposal=baseline,
            baseline_authorization=authorization,
        )

    unknown_change = AgentPlanDiffChange.model_construct(
        dimension="future_dimension",
        path="future_dimension.value",
        change_kind="changed",
        before_present=True,
        before="old",
        after_present=True,
        after="new",
    )
    forged_diff = revision.plan_diff.model_copy(
        update={
            "changes": [*revision.plan_diff.changes, unknown_change],
            "material_change": True,
        }
    )
    forged_revision = revision.model_copy(update={"plan_diff": forged_diff})
    with pytest.raises(AutonomyL2MaterialityError):
        classify_plan_revision_materiality(
            forged_revision,
            baseline_proposal=baseline,
            baseline_authorization=authorization,
        )


def test_l2_policy_dimension_roster_is_explicit() -> None:
    assert AUTONOMY_L2_REVIEWED_DIFF_DIMENSIONS == {
        "task",
        "dependency",
        "option",
        "artifact",
        "route_profile_resource",
        "budget",
        "gate",
        "semantic",
    }


def test_l2_failure_route_publishes_successor_without_reusing_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, client, service, state, current = _start_waiting_gate_session_with_client(
        tmp_path,
        monkeypatch,
    )
    receipt = service.controller.control_store.list_harness_controller_action_receipts(
        project_id="conversation-project",
        controller_execution_id=state["controller_execution_id"],
    )[-1]
    failed = replace(
        current,
        receipt=receipt,
        inspection=_typed_controller_inspection_variant(
            current.inspection,
            status=AgentHarnessControllerStatus.FAILED,
            action=AgentHarnessControllerAction.STOP_TASK_TERMINAL,
        ),
    )
    monkeypatch.setattr(
        service.controller,
        "read_execution_agent_snapshot",
        lambda **_kwargs: failed,
    )
    response = client.post(
        "/api/projects/conversation-project/conversations/conversation-one/agent-session/replan",
        json={
            "run_id": state["run_id"],
            "external_llm_approved": True,
            "llm_provider": {
                "provider": "stub",
                "model": "stub",
                "stub_response": {
                    "rationale_summary": "Use a bounded smaller candidate set.",
                    "option_patch": {"generate_candidates": {"count": 4}},
                },
            },
        },
    )
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["session"]["status"] == "approval_required"
    assert body["session"]["autonomy_level"] == "L2"
    assert body["session"]["autonomy_l2_materiality_class"] == "material"
    assert body["session"]["authorization_id"] == ""
    assert body["session"]["controller_execution_id"] == ""
    assert body["session"]["autonomy_l2_baseline_controller_execution_id"] == state["controller_execution_id"]
    assert body["session"]["autonomy_l2_baseline_authorization_id"] == state["authorization_id"]
    assert body["proposal"]["proposal_id"] != state["proposal_id"]
    assert body["decision"]["executable"] is False
    replay = client.post(
        "/api/projects/conversation-project/conversations/conversation-one/agent-session/replan",
        json={
            "run_id": state["run_id"],
            "external_llm_approved": True,
            "llm_provider": {
                "provider": "stub",
                "model": "stub",
                "stub_response": {
                    "rationale_summary": "This must not be called again.",
                    "option_patch": {"generate_candidates": {"count": 999}},
                },
            },
        },
    )
    assert replay.status_code == 200, replay.get_json()
    assert replay.get_json()["proposal"]["proposal_id"] == body["proposal"]["proposal_id"]


def test_l2_failed_no_change_stays_stopped_without_restarting_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app, client, service, state, current = _start_waiting_gate_session_with_client(
        tmp_path,
        monkeypatch,
    )
    receipt = service.controller.control_store.list_harness_controller_action_receipts(
        project_id="conversation-project",
        controller_execution_id=state["controller_execution_id"],
    )[-1]
    failed = replace(
        current,
        receipt=receipt,
        inspection=_typed_controller_inspection_variant(
            current.inspection,
            status=AgentHarnessControllerStatus.FAILED,
            action=AgentHarnessControllerAction.STOP_TASK_TERMINAL,
        ),
    )
    monkeypatch.setattr(
        service.controller,
        "read_execution_agent_snapshot",
        lambda **_kwargs: failed,
    )
    response = client.post(
        "/api/projects/conversation-project/conversations/conversation-one/agent-session/replan",
        json={
            "run_id": state["run_id"],
            "external_llm_approved": True,
            "llm_provider": {
                "provider": "stub",
                "model": "stub",
                "stub_response": {
                    "rationale_summary": "No safe change is proposed.",
                    "no_change": True,
                },
            },
        },
    )
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["session"]["status"] == "failed"
    assert body["session"]["reason_code"] == "AUTONOMY_L2_NO_MATERIAL_CHANGE"
    assert body["session"]["autonomy_l2_materiality_class"] == "non_material"
    assert body["session"]["authorization_id"] == state["authorization_id"]
    assert body["session"]["controller_execution_id"] == state["controller_execution_id"]
    assert body["decision"]["executable"] is False
