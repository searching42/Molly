"""Acceptance-only matrix checks used by the formal L1/L2 runner."""

from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

import ai4s_agent.scientific_agent_conversation as session_module
import ai4s_agent.routes.scientific_agent_conversation as conversation_routes
from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.schemas import (
    AgentHarnessControllerAction,
    AgentHarnessControllerActionBoundaryClass,
    AgentHarnessControllerActionReceipt,
    AgentHarnessControllerExecution,
    AgentHarnessControllerInspection,
    AgentHarnessControllerStatus,
    AgentToolCallApplicationOutcome,
    _agent_digest,
)
from ai4s_agent.execution_agent_store import ExecutionAgentStore
from ai4s_agent.scientific_agent_autonomy_l1 import (
    AUTONOMY_L1_MAX_LLM_CALLS,
    AUTONOMY_L1_MAX_TRANSITIONS,
    AUTONOMY_L1_MAX_WALL_CLOCK_SECONDS,
    resource_binding_digest,
)
from tests.test_scientific_agent_conversation_session import (
    _auto_controller_result,
    _start_waiting_gate_session_with_client,
    _typed_controller_inspection_variant,
)


pytestmark = [pytest.mark.acceptance, pytest.mark.pr_fast]


class _CountingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, **_kwargs):
        self.calls += 1
        raise AssertionError("non-FAILED replan entered the provider")

    def close(self) -> None:
        return None


class _BudgetGuardProvider:
    """Provider double that proves the runtime guard ran before the call."""

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, **_kwargs):
        self.calls += 1
        raise AssertionError("L1 budget guard allowed an exhausted provider call")

    def close(self) -> None:
        return None


class _CountingRevisionProvider(StubLLMProvider):
    def __init__(self) -> None:
        super().__init__(
            response={
                "rationale_summary": "Use one bounded concurrent revision.",
                "option_patch": {"generate_candidates": {"count": 4}},
                "stop_conditions": ["pause after the bounded candidate run"],
            }
        )
        self.calls = 0
        self._calls_lock = Lock()

    def complete_json(self, **kwargs):
        with self._calls_lock:
            self.calls += 1
        return super().complete_json(**kwargs)


def _write_acceptance_observation(payload: dict[str, object]) -> None:
    """Publish only bounded, privacy-safe observations to the runner adapter."""

    target = os.environ.get("MOLLY_ACCEPTANCE_OBSERVATION_PATH")
    if not target:
        return
    Path(target).write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _offset_timestamp(value: str, seconds: int) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (parsed + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def test_non_failed_l2_trigger_matrix_rejects_before_provider_or_successor(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = (
        (
            "SUCCEEDED",
            AgentHarnessControllerStatus.SUCCEEDED,
            AgentHarnessControllerAction.COMPLETE_EXECUTION,
        ),
        (
            "CANCELLED",
            AgentHarnessControllerStatus.CANCELLED,
            AgentHarnessControllerAction.CANCEL_EXECUTION,
        ),
        (
            "RECOVERY_REQUIRED",
            AgentHarnessControllerStatus.RECOVERY_REQUIRED,
            AgentHarnessControllerAction.RECOVER_REMOTE_TASK,
        ),
        (
            "WAITING_GATE",
            AgentHarnessControllerStatus.WAITING_GATE,
            AgentHarnessControllerAction.WAIT_FOR_GATE,
        ),
        (
            "WAITING_REMOTE_APPROVAL",
            AgentHarnessControllerStatus.WAITING_REMOTE_APPROVAL,
            AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL,
        ),
    )

    for index, (label, status, action) in enumerate(matrix):
        with monkeypatch.context() as case_patch:
            case_root = tmp_path / label.lower()
            case_root.mkdir()
            app, client, service, state, current = _start_waiting_gate_session_with_client(
                case_root,
                case_patch,
            )
            del app
            # The Controller inspection is still typed and exact-bound; only
            # the externally injected current observation represents the
            # adversarial terminal/boundary state under review.
            variant = replace(
                current,
                inspection=_typed_controller_inspection_variant(
                    current.inspection,
                    status=status,
                    action=action,
                ),
            )
            case_patch.setattr(
                service.controller,
                "read_execution_agent_snapshot",
                lambda **_kwargs: variant,
            )
            provider = _CountingProvider()

            def resolve(_payload, *, settings, providers, role=None):
                del settings, providers, role
                return SimpleNamespace(
                    provider_context=nullcontext(provider),
                    provider_binding_digest=_agent_digest({"case": label, "index": index}),
                )

            case_patch.setattr(
                conversation_routes,
                "resolve_llm_provider_payload",
                resolve,
            )
            response = client.post(
                "/api/projects/conversation-project/conversations/conversation-one/agent-session/replan",
                json={"run_id": state["run_id"], "llm_provider": {"provider": "stub"}},
            )
            assert response.status_code == 409, (label, response.get_json())
            assert response.get_json()["error_code"] == "replanner_authority_stale"
            assert provider.calls == 0
            assert service.read_session(
                project_id="conversation-project",
                conversation_id="conversation-one",
            )["authorization_id"] == state["authorization_id"]


def test_l1_acceptance_rebuilds_128_transition_receipts_and_stops_before_effect(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app, _client, service, state, current = _start_waiting_gate_session_with_client(
        tmp_path,
        monkeypatch,
    )
    current = _auto_controller_result(current)
    monkeypatch.setattr(
        session_module,
        "controller_action_boundary_class",
        lambda *_args, **_kwargs: AgentHarnessControllerActionBoundaryClass.ORDINARY_ADVANCE,
    )
    execution = current.execution
    control_store = service.controller.control_store
    existing = control_store.list_harness_controller_action_receipts(
        project_id=execution.project_id,
        controller_execution_id=execution.controller_execution_id,
    )
    assert existing
    base_receipt = existing[-1]
    assert isinstance(base_receipt, AgentHarnessControllerActionReceipt)
    for index in range(len(existing), AUTONOMY_L1_MAX_TRANSITIONS):
        payload = base_receipt.model_dump(mode="json")
        payload["reason_codes"] = sorted(
            set(base_receipt.reason_codes)
            | {f"ACCEPTANCE_TRANSITION_{index:03d}"}
        )
        payload["receipt_id"] = ""
        payload["receipt_digest"] = ""
        control_store.publish_harness_controller_action_receipt(
            project_id=execution.project_id,
            receipt=type(base_receipt)(**payload),
        )
    receipts_before = control_store.list_harness_controller_action_receipts(
        project_id=execution.project_id,
        controller_execution_id=execution.controller_execution_id,
    )
    assert len(receipts_before) == AUTONOMY_L1_MAX_TRANSITIONS

    effects = {"advance": 0, "create_proposal": 0}

    def forbidden_advance(*_args, **_kwargs):
        effects["advance"] += 1
        raise AssertionError("exhausted L1 transition budget reached Controller.advance")

    def forbidden_create_proposal(*_args, **_kwargs):
        effects["create_proposal"] += 1
        raise AssertionError("exhausted L1 transition budget reached the Execution Agent")

    monkeypatch.setattr(service.controller, "get", lambda **_kwargs: current)
    monkeypatch.setattr(service.controller, "advance", forbidden_advance)
    monkeypatch.setattr(
        service.execution_agent,
        "create_proposal",
        forbidden_create_proposal,
    )
    service._transition(
        project_id=execution.project_id,
        conversation_id="conversation-one",
        status="running",
        reason_code="EXECUTION_AGENT_PAUSED",
        updates={
            "controller_status": current.inspection.status.value,
            "current_task_id": current.inspection.current_task_id,
        },
        event_type="execution.paused",
    )
    result = service.tick(
        project_id=execution.project_id,
        conversation_id="conversation-one",
        run_id=state["run_id"],
        provider=_BudgetGuardProvider(),
        provider_binding_digest=_agent_digest({"provider": "budget-guard"}),
    )

    receipts_after = control_store.list_harness_controller_action_receipts(
        project_id=execution.project_id,
        controller_execution_id=execution.controller_execution_id,
    )
    assert len(receipts_after) == AUTONOMY_L1_MAX_TRANSITIONS
    assert result.session["status"] == "running"
    assert result.session["reason_code"] == "AUTONOMY_L1_TRANSITION_BUDGET_EXHAUSTED"
    assert result.session["autonomy_status"] == "budget_exhausted"
    assert result.session["autonomy_budget_usage"]["transitions"] == AUTONOMY_L1_MAX_TRANSITIONS
    assert effects == {"advance": 0, "create_proposal": 0}


def test_l1_acceptance_rebuilds_64_llm_checkpoints_and_stops_before_provider(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app, _client, service, state, current = _start_waiting_gate_session_with_client(
        tmp_path,
        monkeypatch,
    )
    current = _auto_controller_result(current)
    monkeypatch.setattr(
        session_module,
        "controller_action_boundary_class",
        lambda *_args, **_kwargs: AgentHarnessControllerActionBoundaryClass.ORDINARY_ADVANCE,
    )
    execution = current.execution
    store = service.execution_agent.store
    store.initialize_l1_budget_evidence(
        project_id=execution.project_id,
        controller_execution_id=execution.controller_execution_id,
        controller_execution_digest=execution.execution_digest,
    )
    for index in range(AUTONOMY_L1_MAX_LLM_CALLS):
        request_id = f"acceptance-llm-{index:03d}"
        with store.proposal_request_session(
            project_id=execution.project_id,
            controller_execution_id=execution.controller_execution_id,
            client_request_id=request_id,
            request_digest=_agent_digest({"request_id": request_id}),
        ) as request_dir:
            store.write_marker(
                request_dir,
                filename="llm_request_started.json",
                status="LLM_REQUEST_STARTED",
                values={"prompt_digest": _agent_digest({"request_id": request_id})},
            )
    calls = store.count_llm_calls_for_controller_execution(
        project_id=execution.project_id,
        controller_execution_id=execution.controller_execution_id,
    )
    assert calls == AUTONOMY_L1_MAX_LLM_CALLS
    effects = {"advance": 0, "create_proposal": 0}

    def forbidden_advance(*_args, **_kwargs):
        effects["advance"] += 1
        raise AssertionError("exhausted L1 LLM budget reached Controller.advance")

    def forbidden_create_proposal(*_args, **_kwargs):
        effects["create_proposal"] += 1
        raise AssertionError("exhausted L1 LLM budget reached the Execution Agent")

    monkeypatch.setattr(service.controller, "get", lambda **_kwargs: current)
    monkeypatch.setattr(service.controller, "advance", forbidden_advance)
    monkeypatch.setattr(
        service.execution_agent,
        "create_proposal",
        forbidden_create_proposal,
    )
    service._transition(
        project_id=execution.project_id,
        conversation_id="conversation-one",
        status="running",
        reason_code="EXECUTION_AGENT_PAUSED",
        updates={
            "controller_status": current.inspection.status.value,
            "current_task_id": current.inspection.current_task_id,
        },
        event_type="execution.paused",
    )
    provider = _BudgetGuardProvider()
    result = service.tick(
        project_id=execution.project_id,
        conversation_id="conversation-one",
        run_id=state["run_id"],
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "budget-guard"}),
    )
    assert result.session["status"] == "running"
    assert result.session["reason_code"] == "AUTONOMY_L1_LLM_BUDGET_EXHAUSTED"
    assert result.session["autonomy_status"] == "budget_exhausted"
    assert result.session["autonomy_budget_usage"]["llm_calls"] == AUTONOMY_L1_MAX_LLM_CALLS
    assert store.count_llm_calls_for_controller_execution(
        project_id=execution.project_id,
        controller_execution_id=execution.controller_execution_id,
    ) == AUTONOMY_L1_MAX_LLM_CALLS
    assert provider.calls == 0
    assert effects == {"advance": 0, "create_proposal": 0}


def test_l1_acceptance_wall_clock_budget_stops_before_effect(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app, _client, service, state, current = _start_waiting_gate_session_with_client(
        tmp_path,
        monkeypatch,
    )
    current = _auto_controller_result(current)
    monkeypatch.setattr(
        session_module,
        "controller_action_boundary_class",
        lambda *_args, **_kwargs: AgentHarnessControllerActionBoundaryClass.ORDINARY_ADVANCE,
    )
    service.clock = lambda: _offset_timestamp(
        current.execution.created_at,
        AUTONOMY_L1_MAX_WALL_CLOCK_SECONDS,
    )
    effects = {"advance": 0, "create_proposal": 0}

    def forbidden_advance(*_args, **_kwargs):
        effects["advance"] += 1
        raise AssertionError("wall-clock exhaustion reached Controller.advance")

    def forbidden_create_proposal(*_args, **_kwargs):
        effects["create_proposal"] += 1
        raise AssertionError("wall-clock exhaustion reached the Execution Agent")

    monkeypatch.setattr(service.controller, "get", lambda **_kwargs: current)
    monkeypatch.setattr(service.controller, "advance", forbidden_advance)
    monkeypatch.setattr(service.execution_agent, "create_proposal", forbidden_create_proposal)
    service._transition(
        project_id=current.execution.project_id,
        conversation_id="conversation-one",
        status="running",
        reason_code="EXECUTION_AGENT_PAUSED",
        updates={
            "controller_status": current.inspection.status.value,
            "current_task_id": current.inspection.current_task_id,
        },
        event_type="execution.paused",
    )
    provider = _BudgetGuardProvider()
    result = service.tick(
        project_id=current.execution.project_id,
        conversation_id="conversation-one",
        run_id=state["run_id"],
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "wall-clock-budget"}),
    )
    usage = result.session["autonomy_budget_usage"]
    assert result.session["status"] == "running"
    assert result.session["reason_code"] == "AUTONOMY_L1_WALL_CLOCK_BUDGET_EXHAUSTED"
    assert usage["wall_clock_elapsed_seconds"] >= AUTONOMY_L1_MAX_WALL_CLOCK_SECONDS
    assert effects == {"advance": 0, "create_proposal": 0}
    assert provider.calls == 0
    _write_acceptance_observation(
        {
            "observed_reason_codes": [result.session["reason_code"]],
            "runtime_entrypoint": "ScientificAgentConversationSessionService.tick",
            "wall_clock_limit_seconds": AUTONOMY_L1_MAX_WALL_CLOCK_SECONDS,
            "wall_clock_elapsed_seconds": usage["wall_clock_elapsed_seconds"],
            "clock_injected": True,
            "clock_boundary_effect": "blocked_before_effect",
            "controller_effect_call_count": effects["advance"],
            "execution_agent_proposal_call_count": effects["create_proposal"],
            "provider_call_count": provider.calls,
            "next_effect_blocked": True,
            "authority_preserved": True,
        }
    )


def test_l1_acceptance_task_graph_expansion_fails_closed_before_effect(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app, _client, service, state, current = _start_waiting_gate_session_with_client(
        tmp_path,
        monkeypatch,
    )
    current = _auto_controller_result(current)
    invalid_payload = current.inspection.model_dump(mode="json")
    invalid_payload.update(
        {
            "current_task_index": len(current.execution.ordered_task_ids),
            "current_task_id": "task-outside-authorized-roster",
            "current_slot_id": "slot-outside-authorized-roster",
            "inspection_digest": "",
        }
    )
    expanded_inspection = AgentHarnessControllerInspection(**invalid_payload)
    adversarial = replace(current, inspection=expanded_inspection)
    monkeypatch.setattr(
        session_module,
        "controller_action_boundary_class",
        lambda *_args, **_kwargs: AgentHarnessControllerActionBoundaryClass.ORDINARY_ADVANCE,
    )
    effects = {"advance": 0, "create_proposal": 0}

    def forbidden_advance(*_args, **_kwargs):
        effects["advance"] += 1
        raise AssertionError("task-graph expansion reached Controller.advance")

    def forbidden_create_proposal(*_args, **_kwargs):
        effects["create_proposal"] += 1
        raise AssertionError("task-graph expansion reached the Execution Agent")

    monkeypatch.setattr(service.controller, "get", lambda **_kwargs: adversarial)
    monkeypatch.setattr(service.controller, "advance", forbidden_advance)
    monkeypatch.setattr(service.execution_agent, "create_proposal", forbidden_create_proposal)
    service._transition(
        project_id=current.execution.project_id,
        conversation_id="conversation-one",
        status="running",
        reason_code="EXECUTION_AGENT_PAUSED",
        updates={
            "controller_status": current.inspection.status.value,
            "current_task_id": current.inspection.current_task_id,
        },
        event_type="execution.paused",
    )
    provider = _BudgetGuardProvider()
    result = service.tick(
        project_id=current.execution.project_id,
        conversation_id="conversation-one",
        run_id=state["run_id"],
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "task-graph-boundary"}),
    )
    assert result.session["status"] == "unknown"
    assert result.session["reason_code"] == "AUTONOMY_L1_EVIDENCE_UNAVAILABLE"
    assert effects == {"advance": 0, "create_proposal": 0}
    assert provider.calls == 0
    _write_acceptance_observation(
        {
            "observed_reason_codes": [result.session["reason_code"]],
            "runtime_entrypoint": "ScientificAgentConversationSessionService.tick",
            "task_graph_expansion_attempted": True,
            "task_graph_mutation": False,
            "task_graph_identity_verified": False,
            "boundary_effect": "fail_closed_before_effect",
            "controller_effect_call_count": effects["advance"],
            "execution_agent_proposal_call_count": effects["create_proposal"],
            "provider_call_count": provider.calls,
            "next_effect_blocked": True,
            "authority_preserved": True,
        }
    )


def test_l1_acceptance_resource_binding_expansion_fails_closed_before_effect(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app, _client, service, state, current = _start_waiting_gate_session_with_client(
        tmp_path,
        monkeypatch,
    )
    current = _auto_controller_result(current)
    expanded_payload = current.execution.model_dump(mode="json")
    expanded_payload.update(
        {
            "controller_execution_id": "",
            "execution_digest": "",
            "budget_binding_digest": _agent_digest({"resource": "expanded-budget"}),
            "aggregate_budget_digest": _agent_digest({"resource": "expanded-aggregate"}),
        }
    )
    expanded_execution = AgentHarnessControllerExecution(**expanded_payload)
    assert resource_binding_digest(expanded_execution) != resource_binding_digest(
        current.execution
    )
    adversarial = replace(current, execution=expanded_execution)
    monkeypatch.setattr(
        session_module,
        "controller_action_boundary_class",
        lambda *_args, **_kwargs: AgentHarnessControllerActionBoundaryClass.ORDINARY_ADVANCE,
    )
    effects = {"advance": 0, "create_proposal": 0}

    def forbidden_advance(*_args, **_kwargs):
        effects["advance"] += 1
        raise AssertionError("resource expansion reached Controller.advance")

    def forbidden_create_proposal(*_args, **_kwargs):
        effects["create_proposal"] += 1
        raise AssertionError("resource expansion reached the Execution Agent")

    monkeypatch.setattr(service.controller, "get", lambda **_kwargs: adversarial)
    monkeypatch.setattr(service.controller, "advance", forbidden_advance)
    monkeypatch.setattr(service.execution_agent, "create_proposal", forbidden_create_proposal)
    service._transition(
        project_id=current.execution.project_id,
        conversation_id="conversation-one",
        status="running",
        reason_code="EXECUTION_AGENT_PAUSED",
        updates={
            "controller_execution_id": expanded_execution.controller_execution_id,
            "controller_execution_digest": expanded_execution.execution_digest,
            "controller_status": current.inspection.status.value,
            "current_task_id": current.inspection.current_task_id,
        },
        event_type="execution.paused",
    )
    provider = _BudgetGuardProvider()
    result = service.tick(
        project_id=current.execution.project_id,
        conversation_id="conversation-one",
        run_id=state["run_id"],
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "resource-boundary"}),
    )
    assert result.session["status"] == "unknown"
    assert result.session["reason_code"] == "AUTONOMY_L1_EVIDENCE_UNAVAILABLE"
    assert effects == {"advance": 0, "create_proposal": 0}
    assert provider.calls == 0
    _write_acceptance_observation(
        {
            "observed_reason_codes": [result.session["reason_code"]],
            "runtime_entrypoint": "ScientificAgentConversationSessionService.tick",
            "resource_expansion": True,
            "resource_binding_changed": True,
            "resource_evidence_fail_closed": True,
            "boundary_effect": "fail_closed_before_effect",
            "controller_effect_call_count": effects["advance"],
            "execution_agent_proposal_call_count": effects["create_proposal"],
            "provider_call_count": provider.calls,
            "next_effect_blocked": True,
            "authority_preserved": True,
        }
    )


def test_l1_l2_handoff_starts_fresh_l1_budget_epoch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app, client, service, state, current = _start_waiting_gate_session_with_client(
        tmp_path,
        monkeypatch,
    )
    project_id = "conversation-project"
    conversation_id = "conversation-one"
    endpoint = f"/api/projects/{project_id}/conversations/{conversation_id}/agent-session"
    baseline_execution = current.execution
    baseline_receipts = service.controller.control_store.list_harness_controller_action_receipts(
        project_id=project_id,
        controller_execution_id=baseline_execution.controller_execution_id,
    )
    assert baseline_receipts
    store = service.execution_agent.store
    store.initialize_l1_budget_evidence(
        project_id=project_id,
        controller_execution_id=baseline_execution.controller_execution_id,
        controller_execution_digest=baseline_execution.execution_digest,
    )
    with store.proposal_request_session(
        project_id=project_id,
        controller_execution_id=baseline_execution.controller_execution_id,
        client_request_id="l1-l2-baseline-budget-evidence",
        request_digest=_agent_digest({"request": "l1-l2-baseline-budget-evidence"}),
    ) as request_dir:
        store.write_marker(
            request_dir,
            filename="llm_request_started.json",
            status="LLM_REQUEST_STARTED",
            values={"prompt_digest": _agent_digest({"prompt": "baseline"})},
        )
    baseline_snapshot = service._l1_budget_snapshot(controller_result=current)
    assert baseline_snapshot.transitions_used > 0
    assert baseline_snapshot.llm_calls_used == 1
    baseline_receipt_ids = {item.receipt_id for item in baseline_receipts}

    receipt = baseline_receipts[-1]
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
                    "rationale_summary": "Use a bounded fresh L1 epoch.",
                    "option_patch": {"generate_candidates": {"count": 4}},
                    "stop_conditions": ["pause after the bounded candidate run"],
                },
            },
        },
    )
    assert replanned.status_code == 200, replanned.get_json()
    body = replanned.get_json()
    assert body["session"]["status"] == "approval_required"
    assert body["session"]["authorization_id"] == ""
    successor_id = body["proposal"]["proposal_id"]

    appended = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/messages",
        json={
            "role": "user",
            "content": "确认执行",
            "client_message_id": "user-message-l1-l2-fresh-epoch",
        },
    )
    assert appended.status_code == 201, appended.get_json()
    approved = client.post(
        endpoint + "/turn",
        json={"run_id": state["run_id"], "llm_provider": {"provider": "stub", "model": "stub", "stub_response": {}}},
    )
    assert approved.status_code == 200, approved.get_json()
    new_session = approved.get_json()["session"]
    assert new_session["status"] != "approval_required"
    controller_b_id = new_session["controller_execution_id"]
    assert controller_b_id
    assert controller_b_id != baseline_execution.controller_execution_id
    controller_b = service.controller.get(
        project_id=project_id,
        controller_execution_id=controller_b_id,
    )
    assert controller_b.execution.proposal_id == successor_id
    baseline_receipts_after = service.controller.control_store.list_harness_controller_action_receipts(
        project_id=project_id,
        controller_execution_id=baseline_execution.controller_execution_id,
    )
    b_receipts = service.controller.control_store.list_harness_controller_action_receipts(
        project_id=project_id,
        controller_execution_id=controller_b_id,
    )
    b_llm_calls = store.count_llm_calls_for_controller_execution(
        project_id=project_id,
        controller_execution_id=controller_b_id,
    )
    b_snapshot = service._l1_budget_snapshot(controller_result=controller_b)
    assert b_snapshot.controller_execution_id == controller_b_id
    assert b_snapshot.transitions_used == len(b_receipts)
    assert b_snapshot.llm_calls_used == b_llm_calls
    assert {item.receipt_id for item in b_receipts}.isdisjoint(baseline_receipt_ids)
    assert len(baseline_receipts_after) == baseline_snapshot.transitions_used
    assert store.count_llm_calls_for_controller_execution(
        project_id=project_id,
        controller_execution_id=baseline_execution.controller_execution_id,
    ) == baseline_snapshot.llm_calls_used

    # Enter the actual mutating tick path for Controller B.  The application
    # edge is paused as an external test double; policy and budget evaluation
    # still run through the real coordinator against B's exact inspection.
    controller_b_auto = _auto_controller_result(controller_b)
    monkeypatch.setattr(
        session_module,
        "controller_action_boundary_class",
        lambda *_args, **_kwargs: AgentHarnessControllerActionBoundaryClass.ORDINARY_ADVANCE,
    )
    monkeypatch.setattr(service.controller, "get", lambda **_kwargs: controller_b_auto)
    calls = {"create": 0, "apply": 0}
    proposal = SimpleNamespace(
        tool_call_proposal_id="l1-l2-fresh-epoch-tool-call",
        tool_call_proposal_digest=_agent_digest({"proposal": "fresh-epoch"}),
    )

    def create_proposal(*_args, **_kwargs):
        calls["create"] += 1
        return SimpleNamespace(publication=SimpleNamespace(proposal=proposal))

    def apply_proposal(*_args, **_kwargs):
        calls["apply"] += 1
        return SimpleNamespace(
            application_receipt=SimpleNamespace(
                outcome=AgentToolCallApplicationOutcome.PAUSED,
            ),
            controller_result=controller_b_auto,
        )

    monkeypatch.setattr(service.execution_agent, "create_proposal", create_proposal)
    monkeypatch.setattr(service.execution_agent, "apply_proposal", apply_proposal)
    service._transition(
        project_id=project_id,
        conversation_id=conversation_id,
        status="running",
        reason_code="EXECUTION_AGENT_PAUSED",
        updates={
            "controller_status": controller_b_auto.inspection.status.value,
            "current_task_id": controller_b_auto.inspection.current_task_id,
        },
        event_type="execution.paused",
    )
    continued = service.tick(
        project_id=project_id,
        conversation_id=conversation_id,
        run_id=state["run_id"],
        provider=object(),
        provider_binding_digest=_agent_digest({"provider": "fresh-epoch"}),
    )
    assert calls == {"create": 1, "apply": 1}
    assert continued.session["reason_code"] == "EXECUTION_AGENT_PAUSED"
    fresh_snapshot = service._l1_budget_snapshot(controller_result=controller_b_auto)
    assert fresh_snapshot.controller_execution_id == controller_b_id
    assert fresh_snapshot.transitions_used == len(b_receipts)
    assert fresh_snapshot.llm_calls_used == b_llm_calls
    assert store.count_llm_calls_for_controller_execution(
        project_id=project_id,
        controller_execution_id=baseline_execution.controller_execution_id,
    ) == baseline_snapshot.llm_calls_used


def test_l2_concurrent_material_replan_publishes_one_successor(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _client, service, state, current = _start_waiting_gate_session_with_client(
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
    provider = _CountingRevisionProvider()

    def resolve(_payload, *, settings, providers, role=None):
        del settings, providers, role
        return SimpleNamespace(
            provider_context=nullcontext(provider),
            provider_binding_digest=_agent_digest(
                {"provider": "acceptance-concurrent-replan"}
            ),
        )

    monkeypatch.setattr(
        conversation_routes,
        "resolve_llm_provider_payload",
        resolve,
    )
    endpoint = (
        "/api/projects/conversation-project/conversations/"
        "conversation-one/agent-session/replan"
    )
    payload = {
        "run_id": state["run_id"],
        "external_llm_approved": True,
        "llm_provider": {
            "provider": "stub",
            "model": "stub",
            "stub_response": {
                "rationale_summary": "Use a bounded concurrent revision.",
                "option_patch": {"generate_candidates": {"count": 4}},
                "stop_conditions": ["pause after the bounded candidate run"],
            },
        },
    }
    start = Barrier(2)

    def concurrent_replan():
        with app.test_client() as concurrent_client:
            start.wait(timeout=5)
            response = concurrent_client.post(endpoint, json=payload)
            return response.status_code, response.get_json()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: concurrent_replan(), (1, 2)))

    assert [status for status, _body in results] == [200, 200]
    bodies = [body for _status, body in results]
    proposal_ids = {body["proposal"]["proposal_id"] for body in bodies}
    assert len(proposal_ids) == 1
    assert all(body["session"]["status"] == "approval_required" for body in bodies)
    assert provider.calls == 1
    session = service.read_session(
        project_id="conversation-project",
        conversation_id="conversation-one",
    )
    assert session["autonomy_l2_successor_proposal_id"] == next(iter(proposal_ids))
