from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import ai4s_agent.scientific_agent_deterministic_fastpath as fastpath_module

from ai4s_agent.conversation_store import ConversationStore
from ai4s_agent.scientific_agent_autonomy_policy import (
    classify_current_controller_inspection,
)
from ai4s_agent.scientific_agent_conversation import (
    ScientificAgentConversationSessionService,
)
from ai4s_agent.scientific_agent_deterministic_fastpath import (
    DETERMINISTIC_FASTPATH_POLICY_DIGEST,
    DETERMINISTIC_FASTPATH_POLICY_VERSION,
    DeterministicFastPathClassification,
    DeterministicFastPathVerificationError,
    classify_deterministic_successor,
    verify_deterministic_fast_path_decision,
)
from ai4s_agent.schemas import (
    AgentHarnessControllerAction,
    AgentHarnessControllerAdvanceRequest,
    AgentHarnessGateApprovalRequest,
    AgentHarnessControllerStartRequest,
    AgentHarnessControllerInspection,
    AgentHarnessControllerStatus,
    _agent_digest,
)
from ai4s_agent.scientific_agent_plan import ScientificAgentPlanService
from tests.execution_agent_test_support import (
    NOW,
    CountingStubProvider,
    execution_agent_service,
    local_controller_execution,
)
from tests.test_scientific_agent_harness_controller import _gated_local_authority_chain


pytestmark = pytest.mark.pr_fast


def _advance(controller, result, client_request_id: str):
    return controller.advance(
        project_id=result.execution.project_id,
        controller_execution_id=result.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=result.execution.execution_digest,
            client_request_id=client_request_id,
        ),
        expected_inspection_digest=result.inspection.inspection_digest,
    )


def _inspection_variant(
    base: AgentHarnessControllerInspection,
    *,
    action: AgentHarnessControllerAction,
    status: AgentHarnessControllerStatus = AgentHarnessControllerStatus.ACTIVE,
) -> AgentHarnessControllerInspection:
    payload = base.model_dump(mode="json")
    payload.update(
        {
            "status": status.value,
            "next_action": action.value,
            "inspection_digest": "",
        }
    )
    return AgentHarnessControllerInspection(**payload)


def _gated_execution_fixture(tmp_path: Path):
    storage, controller, intent = _gated_local_authority_chain(tmp_path)
    created = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="fastpath-gated-create",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    snapshot = stage.details["execution_snapshot"]
    approved = controller.approve_gate(
        project_id="project-1",
        controller_execution_id=created.execution.controller_execution_id,
        gate_id="gate_1_task_parse",
        request=AgentHarnessGateApprovalRequest(
            expected_snapshot_id=snapshot["snapshot_id"],
            expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
            client_request_id="fastpath-gated-approval",
            note="Approve the exact deterministic fixture Gate.",
        ),
        actor="alice",
    )
    return storage, controller, approved


def _deterministic_fixture(tmp_path: Path):
    storage, controller, approved = _gated_execution_fixture(tmp_path)
    authorization = controller.authorization_service.verify_authorization(
        project_id="project-1",
        authorization_id=approved.execution.authorization_id,
        verify_current=False,
    )
    slot = approved.execution.task_slots[0]
    task_options = authorization.compiled_task_options[slot.task_id]
    binding = controller.executor.derive_one_task_server_binding(
        project_id="project-1",
        run_plan=authorization.run_plan,
        task_index=0,
        task_options=task_options,
    )
    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    snapshot = stage.details["execution_snapshot"]
    manual = controller.executor.execute_one_task_after_committed_gate(
        project_id="project-1",
        run_plan=authorization.run_plan,
        task_index=0,
        task_id=slot.task_id,
        task_options=task_options,
        actor="alice",
        expected_snapshot_id=snapshot["snapshot_id"],
        expected_snapshot_digest=f"sha256:{snapshot['snapshot_hash']}",
        expected_local_adapter_execution_binding_digest=binding[
            "local_adapter_execution_binding_digest"
        ],
        expected_compiled_options_digest=binding["compiled_options_digest"],
        expected_input_artifacts_digest=binding["input_artifacts_digest"],
        expected_output_contract_digest=binding["output_contract_digest"],
    )
    assert manual["status"] == "SUCCEEDED"
    completed = controller.get(
        project_id="project-1",
        controller_execution_id=approved.execution.controller_execution_id,
    )
    assert completed.inspection.next_action is AgentHarnessControllerAction.ADOPT_COMPLETED_TASK
    control_store = controller.control_store
    policy = classify_current_controller_inspection(completed.inspection)
    decision = classify_deterministic_successor(
        execution=completed.execution,
        inspection=completed.inspection,
        policy_decision=policy,
    )
    return storage, control_store, controller, approved, completed, policy, decision


def _conversation_service(
    *,
    storage,
    controller,
):
    proposal_store = controller.proposal_store
    authorization_service = controller.authorization_service
    conversations = ConversationStore(projects=storage)
    conversations.create_conversation(
        "project-1",
        conversation_id="conversation-1",
        title="Deterministic fast path",
    )
    return ScientificAgentConversationSessionService(
        projects=storage,
        conversations=conversations,
        plan_service=ScientificAgentPlanService(
            storage=storage,
            proposal_store=proposal_store,
        ),
        proposal_store=proposal_store,
        authorization_service=authorization_service,
        controller=controller,
        execution_agent=execution_agent_service(
            storage=storage,
            controller=controller,
        ),
        clock=lambda: NOW,
    )


def _session_state(service, result):
    execution = result.execution
    return service._transition(
        project_id="project-1",
        conversation_id="conversation-1",
        status="running",
        reason_code="EXECUTION_AGENT_STEP",
        updates={
            "run_id": execution.run_id,
            "proposal_id": execution.proposal_id,
            "proposal_digest": execution.proposal_digest,
            "authorization_id": execution.authorization_id,
            "authorization_digest": execution.authorization_digest,
            "start_intent_id": execution.start_intent_id,
            "start_intent_digest": execution.start_intent_digest,
            "controller_execution_id": execution.controller_execution_id,
            "controller_execution_digest": execution.execution_digest,
            "controller_status": result.inspection.status.value,
            "current_task_id": result.inspection.current_task_id,
        },
        event_type="test.fastpath.session_started",
    )


def test_unique_verified_local_successor_is_deterministic(tmp_path: Path) -> None:
    _storage, _control_store, _controller, _initial, completed, policy, decision = (
        _deterministic_fixture(tmp_path)
    )

    assert decision.classification is DeterministicFastPathClassification.DETERMINISTIC
    assert decision.controller_action == "adopt_completed_task"
    assert decision.legal_successor_actions == ["adopt_completed_task"]
    assert decision.successor_cardinality == 1
    assert decision.executable is False
    assert decision.controller_execution_digest == completed.execution.execution_digest
    assert decision.inspection_digest == completed.inspection.inspection_digest
    assert decision.policy_version == DETERMINISTIC_FASTPATH_POLICY_VERSION
    assert decision.policy_digest == DETERMINISTIC_FASTPATH_POLICY_DIGEST
    assert decision.autonomy_policy_version == policy.policy_version
    assert decision.autonomy_policy_digest == policy.policy_digest
    assert decision.autonomy_policy_decision_id == policy.decision_id
    assert decision.autonomy_policy_decision_digest == policy.decision_digest


def test_ordinary_local_execute_state_is_not_deterministic_and_falls_back(
    tmp_path: Path,
) -> None:
    storage, controller, initial = _gated_execution_fixture(tmp_path)
    control_store = controller.control_store
    service = _conversation_service(storage=storage, controller=controller)
    state = _session_state(service, initial)
    provider = CountingStubProvider(
        response={
            "selected_tool_id": "controller.advance_current.v1",
            "decision_summary": "Use the existing Controller advance path.",
        }
    )

    _result, final_state, stop_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=state,
        controller_result=initial,
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )

    policy = classify_current_controller_inspection(initial.inspection)
    decision = classify_deterministic_successor(
        execution=initial.execution,
        inspection=initial.inspection,
        policy_decision=policy,
    )
    assert decision.classification is DeterministicFastPathClassification.NOT_DETERMINISTIC
    assert decision.successor_cardinality == 0
    assert provider.calls == 1
    assert stop_reason == "terminal_success"
    assert final_state["status"] == "succeeded"
    receipts = control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
    )
    assert len(receipts) == 2
    assert sum(item.dispatch_occurred for item in receipts) == 1


def test_deterministic_successor_does_not_call_provider_and_uses_controller(
    tmp_path: Path,
) -> None:
    storage, control_store, controller, _initial, completed, _policy, decision = (
        _deterministic_fixture(tmp_path)
    )
    service = _conversation_service(storage=storage, controller=controller)
    state = _session_state(service, completed)

    class ExplodingProvider:
        calls = 0

        def complete_json(self, **_kwargs):
            self.calls += 1
            raise AssertionError("LLM must not be called")

    provider = ExplodingProvider()
    result, final_state, stop_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=state,
        controller_result=completed,
        provider=provider,
        provider_binding_digest="",
    )

    assert provider.calls == 0
    assert stop_reason == "terminal_success"
    assert result is not None
    assert result.inspection.status is AgentHarnessControllerStatus.SUCCEEDED
    assert final_state["last_autonomy_fastpath_decision_digest"] == decision.decision_digest
    assert final_state["last_autonomy_fastpath_decision_source"] == (
        "deterministic_fast_path"
    )
    assert final_state["last_autonomy_fastpath_llm_skipped"] is True
    receipts = control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=completed.execution.controller_execution_id,
    )
    assert len(receipts) == 2
    assert sum(item.dispatch_occurred for item in receipts) == 0


@pytest.mark.parametrize(
    "action",
    [
        AgentHarnessControllerAction.WAIT_FOR_GATE,
        AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL,
        AgentHarnessControllerAction.RECOVER_REMOTE_TASK,
    ],
)
def test_human_boundaries_never_become_fast_path(
    tmp_path: Path,
    action: AgentHarnessControllerAction,
) -> None:
    _storage, _control_store, _controller, initial = local_controller_execution(tmp_path)
    inspection = _inspection_variant(initial.inspection, action=action)
    policy = classify_current_controller_inspection(inspection)
    decision = classify_deterministic_successor(
        execution=initial.execution,
        inspection=inspection,
        policy_decision=policy,
    )

    assert decision.classification is DeterministicFastPathClassification.REQUIRE_HUMAN
    assert decision.executable is False
    assert decision.successor_cardinality == 0


def test_unknown_controller_action_fails_closed(tmp_path: Path) -> None:
    _storage, _control_store, _controller, initial = local_controller_execution(tmp_path)
    payload = initial.inspection.model_dump(mode="python")
    payload.update(
        {
            "next_action": "future.controller.action",
            "inspection_digest": "",
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        unknown = AgentHarnessControllerInspection.model_construct(**payload)
        object.__setattr__(
            unknown,
            "inspection_digest",
            _agent_digest(unknown.semantic_material()),
        )
    policy = classify_current_controller_inspection(initial.inspection)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        decision = classify_deterministic_successor(
            execution=initial.execution,
            inspection=unknown,
            policy_decision=policy,
        )

    assert decision.classification is DeterministicFastPathClassification.FAIL_CLOSED
    assert decision.executable is False
    assert decision.successor_cardinality == 0


def test_stale_execution_or_inspection_fails_closed(tmp_path: Path) -> None:
    _storage, _control_store, _controller, initial = local_controller_execution(tmp_path)
    policy = classify_current_controller_inspection(initial.inspection)

    stale_execution = initial.execution.model_copy(
        update={"execution_digest": _agent_digest({"stale": "execution"})}
    )
    execution_decision = classify_deterministic_successor(
        execution=stale_execution,
        inspection=initial.inspection,
        policy_decision=policy,
    )
    stale_inspection = initial.inspection.model_copy(
        update={"inspection_digest": _agent_digest({"stale": "inspection"})}
    )
    inspection_decision = classify_deterministic_successor(
        execution=initial.execution,
        inspection=stale_inspection,
        policy_decision=policy,
    )

    assert execution_decision.classification is DeterministicFastPathClassification.FAIL_CLOSED
    assert inspection_decision.classification is DeterministicFastPathClassification.FAIL_CLOSED


def test_forged_serialized_decision_is_recomputed_and_rejected(tmp_path: Path) -> None:
    _storage, _control_store, _controller, _initial, completed, policy, decision = (
        _deterministic_fixture(tmp_path)
    )
    forged = decision.model_copy(
        update={
            "controller_action": AgentHarnessControllerAction.EXECUTE_LOCAL_TASK.value
        }
    )

    with pytest.raises(DeterministicFastPathVerificationError):
        verify_deterministic_fast_path_decision(
            execution=completed.execution,
            inspection=completed.inspection,
            policy_decision=policy,
            decision=forged,
        )

    changed_inspection = _inspection_variant(
        completed.inspection,
        action=AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
    )
    changed_policy = classify_current_controller_inspection(changed_inspection)
    with pytest.raises(DeterministicFastPathVerificationError):
        verify_deterministic_fast_path_decision(
            execution=completed.execution,
            inspection=changed_inspection,
            policy_decision=changed_policy,
            decision=decision,
        )


def test_fast_path_policy_identity_change_rejects_old_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _storage, _control_store, _controller, _initial, completed, policy, decision = (
        _deterministic_fixture(tmp_path)
    )
    monkeypatch.setattr(
        fastpath_module,
        "DETERMINISTIC_FASTPATH_POLICY_VERSION",
        "scientific-agent-deterministic-fastpath-policy.v2",
    )
    monkeypatch.setattr(
        fastpath_module,
        "DETERMINISTIC_FASTPATH_POLICY_DIGEST",
        _agent_digest({"policy": "B"}),
    )

    with pytest.raises(DeterministicFastPathVerificationError):
        verify_deterministic_fast_path_decision(
            execution=completed.execution,
            inspection=completed.inspection,
            policy_decision=policy,
            decision=decision,
        )


def test_session_result_reports_zero_llm_calls_for_fast_path_recovery(
    tmp_path: Path,
) -> None:
    storage, _control_store, controller, _initial, completed, _policy, _decision = (
        _deterministic_fixture(tmp_path)
    )
    service = _conversation_service(storage=storage, controller=controller)
    state = _session_state(service, completed)
    service._transition(
        project_id="project-1",
        conversation_id="conversation-1",
        status="running",
        reason_code="DETERMINISTIC_FASTPATH_STEP",
        updates={
            "controller_status": completed.inspection.status.value,
            "current_task_id": completed.inspection.current_task_id,
        },
        event_type="test.fastpath.recovery_pending",
    )

    class ExplodingProvider:
        calls = 0

        def complete_json(self, **_kwargs):
            self.calls += 1
            raise AssertionError("LLM must not be called")

    provider = ExplodingProvider()
    result = service.handle_turn(
        project_id="project-1",
        conversation_id="conversation-1",
        run_id=str(state["run_id"]),
        provider=provider,
        provider_binding_digest="",
    )

    assert provider.calls == 0
    assert result.llm_used is False


def test_session_result_reports_llm_call_for_execution_agent_fallback(
    tmp_path: Path,
) -> None:
    storage, controller, initial = _gated_execution_fixture(tmp_path)
    service = _conversation_service(storage=storage, controller=controller)
    state = _session_state(service, initial)
    service._transition(
        project_id="project-1",
        conversation_id="conversation-1",
        status="running",
        reason_code="EXECUTION_AGENT_PAUSED",
        updates={
            "controller_status": initial.inspection.status.value,
            "current_task_id": initial.inspection.current_task_id,
        },
        event_type="test.fastpath.execution_agent_pending",
    )
    provider = CountingStubProvider(
        response={
            "selected_tool_id": "controller.advance_current.v1",
            "decision_summary": "Use the existing Controller advance path.",
        }
    )

    result = service.handle_turn(
        project_id="project-1",
        conversation_id="conversation-1",
        run_id=str(state["run_id"]),
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )

    assert provider.calls == 1
    assert result.llm_used is True


def test_fast_path_controller_request_replays_without_duplicate_dispatch(
    tmp_path: Path,
) -> None:
    storage, control_store, controller, _initial, completed, policy, decision = (
        _deterministic_fixture(tmp_path)
    )
    request = AgentHarnessControllerAdvanceRequest(
        expected_controller_execution_digest=completed.execution.execution_digest,
        client_request_id=f"fastpath-{decision.decision_digest.split(':', 1)[1][:32]}",
    )
    first = controller.advance(
        project_id=completed.execution.project_id,
        controller_execution_id=completed.execution.controller_execution_id,
        request=request,
        expected_inspection_digest=completed.inspection.inspection_digest,
    )
    replay = controller.advance(
        project_id=completed.execution.project_id,
        controller_execution_id=completed.execution.controller_execution_id,
        request=request,
        expected_inspection_digest=completed.inspection.inspection_digest,
    )

    assert first.receipt is not None
    assert replay.receipt is not None
    assert first.receipt.receipt_id == replay.receipt.receipt_id
    assert first.receipt.receipt_digest == replay.receipt.receipt_digest
    receipts = control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=completed.execution.controller_execution_id,
    )
    assert len(receipts) == 2
    # The fixture's scientific effect was completed before the Controller
    # adoption; the deterministic successor itself must dispatch no effect.
    assert sum(item.dispatch_occurred for item in receipts) == 0
