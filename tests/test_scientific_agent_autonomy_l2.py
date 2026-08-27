from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest

from ai4s_agent.schemas import (
    AGENT_EXECUTION_PLAN_PROPOSAL_V1,
    AgentHarnessControllerAction,
    AgentHarnessControllerStatus,
    AgentAutonomyL2MaterialityClass,
    AgentPlanAuthorizationRequest,
    AgentPlanDiffChange,
    AgentPlanFeedbackRequest,
    AgentExecutionPlanProposal,
    AgentReplanLLMResponse,
    _agent_digest,
)
from ai4s_agent.scientific_agent_authorization import (
    ScientificAgentAuthorizationDenied,
)
from ai4s_agent.scientific_agent_autonomy_l2 import (
    AUTONOMY_L2_MATERIALITY_POLICY_DIGEST,
    AUTONOMY_L2_MATERIALITY_POLICY_VERSION,
    AUTONOMY_L2_REVIEWED_DIFF_DIMENSIONS,
    AutonomyL2MaterialityError,
    _proposal_grant,
    classify_plan_revision_materiality,
    verify_plan_revision_materiality_decision,
)
from ai4s_agent.autonomy_authority import AuthorityRelation, evaluate_authority
from ai4s_agent.planner import AtomicTaskRegistry
from tests.test_scientific_agent_replanner import (
    CountingProvider,
    _baseline,
    _revision_payload,
)
from tests.test_scientific_agent_conversation_session import (
    _stub_provider,
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


def _recast_proposal_as_historical_v1(
    proposal: AgentExecutionPlanProposal, *, count: int
) -> AgentExecutionPlanProposal:
    payload = proposal.model_dump(mode="json")
    for field in ("planner_options", "effective_planner_options", "compiled_task_options"):
        payload[field]["generate_candidates"]["count"] = count
    payload["validated_llm_response"]["task_options"]["generate_candidates"]["count"] = count
    payload.update(
        {
            "schema_version": AGENT_EXECUTION_PLAN_PROPOSAL_V1,
            "authorization_scope_digest": "",
            "semantic_plan_id": "",
            "semantic_plan_digest": "",
            "publication_id": "",
            "proposal_id": "",
            "proposal_digest": "",
        }
    )
    return AgentExecutionPlanProposal.model_validate(payload)


def test_historical_v1_baseline_does_not_expand_exact_option_authority(tmp_path: Path) -> None:
    _revision, baseline, _authorization, _provider = _feedback_revision(
        tmp_path,
        AgentReplanLLMResponse(
            rationale_summary="Use a smaller candidate set.",
            option_patch={"generate_candidates": {"count": 4}},
        ),
    )
    historical_baseline = _recast_proposal_as_historical_v1(baseline, count=8)
    historical_candidate = _recast_proposal_as_historical_v1(baseline, count=4)
    registry = AtomicTaskRegistry()
    baseline_grant = _proposal_grant(
        historical_baseline,
        registry=registry,
        baseline=True,
        valid_from="1970-01-01T00:00:00Z",
    )
    candidate_grant = _proposal_grant(
        historical_candidate,
        registry=registry,
        baseline=False,
        valid_from="1970-01-01T00:00:00Z",
    )
    assert baseline_grant.parameter_bounds["generate_candidates.count"].allowed_values == [8]
    evaluation = evaluate_authority(
        baseline_grant,
        candidate_grant,
        changes=[{"dimension": "option", "path": "option.raw_planner_options"}],
    )
    assert evaluation.relation is AuthorityRelation.INCOMPARABLE
    assert evaluation.auto_apply is False


def test_l2_policy_uses_authority_relation_and_semantic_boundary(tmp_path: Path) -> None:
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
    assert decision.classification is AgentAutonomyL2MaterialityClass.NON_MATERIAL
    assert decision.material_change is False
    assert decision.fresh_permission_required is False
    assert decision.fresh_authorization_required is False
    assert decision.authority_relation.value == "SUBSET"
    assert decision.semantic_boundary.value == "NONE"
    assert decision.authority_auto_apply is True
    assert decision.reason_codes == ["AUTONOMY_L2_AUTHORITY_WITHIN_GRANT"]
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


def test_l2_semantic_boundary_requires_fresh_authority_even_for_subset(
    tmp_path: Path,
) -> None:
    revision, baseline, authorization, _provider = _feedback_revision(
        tmp_path,
        AgentReplanLLMResponse(
            rationale_summary="Pause after the bounded candidate run.",
            stop_conditions=["pause after the bounded candidate run"],
        ),
    )
    decision = classify_plan_revision_materiality(
        revision,
        baseline_proposal=baseline,
        baseline_authorization=authorization,
    )
    assert decision.authority_relation.value == "SUBSET"
    assert decision.semantic_boundary.value == "SCIENTIFIC_CONFIRMATION"
    assert decision.authority_auto_apply is False
    assert decision.classification is AgentAutonomyL2MaterialityClass.MATERIAL
    assert decision.fresh_permission_required is True
    assert decision.fresh_authorization_required is True


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
        update={
            "classification": AgentAutonomyL2MaterialityClass.MATERIAL,
            "material_change": True,
        }
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


def test_l2_failure_route_reuses_subset_authority_without_user_reapproval(
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
    assert body["session"]["status"] == "waiting_gate"
    assert body["session"]["autonomy_level"] in {"L1", "L2"}
    assert body["session"]["autonomy_l2_materiality_class"] == "non_material"
    assert body["session"]["autonomy_l2_authority_relation"] == "SUBSET"
    assert body["session"]["autonomy_l2_semantic_boundary"] == "NONE"
    assert body["session"]["autonomy_l2_authority_auto_apply"] is True
    assert body["session"]["authorization_id"] != state["authorization_id"]
    assert body["session"]["controller_execution_id"]
    assert body["session"]["controller_execution_id"] != state["controller_execution_id"]
    application = service.replanner.read_application(
        project_id="conversation-project",
        receipt_id=(
            "revision-application-"
            + _agent_digest(
                {
                    "project_id": "conversation-project",
                    "revision_id": body["decision"]["revision_id"],
                }
            ).split(":", 1)[1][:32]
        ),
    )
    assert application.schema_version == "agent_plan_revision_application_receipt.v2"
    assert application.fresh_permission_required is False
    assert application.fresh_authorization_required is False
    assert application.authority_decision_id == body["decision"]["decision_id"]
    assert application.authority_decision_digest == body["decision"]["decision_digest"]
    assert application.authority_evaluation_digest == body["decision"]["authority_evaluation_digest"]
    assert application.baseline_authorization_id == state["authorization_id"]
    assert application.baseline_authorization_digest == state["authorization_digest"]
    assert body["session"]["autonomy_l2_baseline_controller_execution_id"] == state["controller_execution_id"]
    assert body["session"]["autonomy_l2_baseline_authorization_id"] == state["authorization_id"]
    assert body["proposal"]["proposal_id"] != state["proposal_id"]
    assert body["decision"]["executable"] is False


def test_l2_material_successor_receives_fresh_authority_after_structured_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app, client, service, state, current = _start_waiting_gate_session_with_client(
        tmp_path,
        monkeypatch,
    )
    project_id = "conversation-project"
    conversation_id = "conversation-one"
    endpoint = (
        f"/api/projects/{project_id}/conversations/{conversation_id}/agent-session"
    )

    baseline_authorization = service.authorization_service.verify_authorization(
        project_id=project_id,
        authorization_id=state["authorization_id"],
        verify_current=False,
    )
    baseline_start_intent = service.authorization_service.verify_start_intent(
        project_id=project_id,
        start_intent_id=state["start_intent_id"],
        verify_current=False,
    )
    baseline_permission = service.controller.control_store.read_permission_decision(
        project_id=project_id,
        decision_id=baseline_authorization.permission_decision_id,
    )
    baseline_controller = (
        service.controller.control_store.read_harness_controller_execution(
            project_id=project_id,
            controller_execution_id=state["controller_execution_id"],
        )
    )

    receipt = service.controller.control_store.list_harness_controller_action_receipts(
        project_id=project_id,
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

    replanned = client.post(
        endpoint + "/replan",
        json={
            "run_id": state["run_id"],
            "external_llm_approved": True,
            "llm_provider": {
                "provider": "stub",
                "model": "stub",
                "stub_response": {
                    "rationale_summary": "Use a bounded smaller candidate set.",
                    "stop_conditions": ["pause after the bounded candidate run"],
                },
            },
        },
    )
    assert replanned.status_code == 200, replanned.get_json()
    replanned_body = replanned.get_json()
    assert replanned_body["session"]["status"] == "approval_required"
    assert replanned_body["session"]["authorization_id"] == ""
    assert replanned_body["session"]["start_intent_id"] == ""
    assert replanned_body["session"]["controller_execution_id"] == ""
    successor = service.proposal_store.read(
        project_id=project_id,
        proposal_id=replanned_body["proposal"]["proposal_id"],
        verify_current=False,
    ).proposal

    # The baseline authority is still an immutable record, but its exact
    # proposal binding cannot authorize or start the material successor.
    assert baseline_authorization.proposal_id != successor.proposal_id
    assert baseline_authorization.proposal_digest != successor.proposal_digest
    assert baseline_start_intent.proposal_id != successor.proposal_id
    assert baseline_controller.proposal_id != successor.proposal_id
    with pytest.raises(ScientificAgentAuthorizationDenied):
        service.authorization_service.approve_and_start(
            project_id=project_id,
            proposal_id=successor.proposal_id,
            request=AgentPlanAuthorizationRequest(
                expected_proposal_digest=baseline_authorization.proposal_digest,
                authorization_mode=baseline_authorization.authorization_mode,
                requested_preauthorized_gate_ids=[],
                confirmed=True,
                client_request_id="baseline-cannot-authorize-successor",
                note="Must fail because the baseline digest is not the successor digest.",
            ),
            actor=baseline_authorization.actor,
            actor_source=baseline_authorization.actor_source,
        )

    approved_plan = client.post(
        endpoint + "/approve",
        json={
            "expected_proposal_digest": replanned_body["proposal"]["proposal_digest"],
            "authorization_mode": "stepwise",
            "requested_preauthorized_gate_ids": [],
            "confirmed": True,
            "client_request_id": "structured-plan-l2-successor-approval",
            "note": "Explicit structured test approval.",
        },
    )
    assert approved_plan.status_code == 200, approved_plan.get_json()
    approved = client.post(
        endpoint + "/tick",
        json={"run_id": state["run_id"], "llm_provider": _stub_provider()},
    )
    assert approved.status_code == 200, approved.get_json()
    approved_body = approved.get_json()
    new_session = approved_body["session"]
    assert new_session["status"] != "approval_required"
    assert new_session["authorization_id"] != baseline_authorization.authorization_id
    assert new_session["start_intent_id"] != baseline_start_intent.start_intent_id
    assert (
        new_session["controller_execution_id"]
        != baseline_controller.controller_execution_id
    )

    new_authorization = service.authorization_service.verify_authorization(
        project_id=project_id,
        authorization_id=new_session["authorization_id"],
        verify_current=False,
    )
    new_start_intent = service.authorization_service.verify_start_intent(
        project_id=project_id,
        start_intent_id=new_session["start_intent_id"],
        verify_current=False,
    )
    new_authorization_permission = (
        service.controller.control_store.read_permission_decision(
            project_id=project_id,
            decision_id=new_authorization.permission_decision_id,
        )
    )
    new_start_permission = service.controller.control_store.read_permission_decision(
        project_id=project_id,
        decision_id=new_start_intent.permission_decision_id,
    )
    new_controller = (
        service.controller.control_store.read_harness_controller_execution(
            project_id=project_id,
            controller_execution_id=new_session["controller_execution_id"],
        )
    )

    assert new_authorization_permission.decision_id != baseline_permission.decision_id
    assert (
        new_authorization_permission.decision_digest
        != baseline_permission.decision_digest
    )
    assert new_authorization_permission.proposal_id == successor.proposal_id
    assert new_authorization_permission.proposal_digest == successor.proposal_digest
    assert new_start_permission.decision_id != baseline_permission.decision_id
    assert new_start_permission.decision_digest != baseline_permission.decision_digest
    assert new_start_permission.proposal_id == successor.proposal_id
    assert new_start_permission.proposal_digest == successor.proposal_digest
    assert new_start_permission.authorization_id == new_authorization.authorization_id
    assert new_authorization.authorization_id != baseline_authorization.authorization_id
    assert new_authorization.authorization_digest != baseline_authorization.authorization_digest
    assert new_start_intent.start_intent_id != baseline_start_intent.start_intent_id
    assert new_start_intent.start_intent_digest != baseline_start_intent.start_intent_digest
    assert new_controller.controller_execution_id != baseline_controller.controller_execution_id
    assert new_controller.execution_digest != baseline_controller.execution_digest
    assert new_authorization.proposal_id == successor.proposal_id
    assert new_authorization.proposal_digest == successor.proposal_digest
    assert new_start_intent.authorization_id == new_authorization.authorization_id
    assert (
        new_start_intent.authorization_digest == new_authorization.authorization_digest
    )
    assert new_start_intent.proposal_id == successor.proposal_id
    assert new_start_intent.proposal_digest == successor.proposal_digest
    assert new_controller.proposal_id == successor.proposal_id
    assert new_controller.proposal_digest == successor.proposal_digest
    assert new_controller.authorization_id == new_authorization.authorization_id
    assert new_controller.start_intent_id == new_start_intent.start_intent_id


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
