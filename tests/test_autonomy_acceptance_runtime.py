"""Acceptance-only matrix checks used by the formal L1/L2 runner."""

from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

import ai4s_agent.routes.scientific_agent_conversation as conversation_routes
from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.schemas import (
    AgentHarnessControllerAction,
    AgentHarnessControllerStatus,
    _agent_digest,
)
from ai4s_agent.execution_agent_store import ExecutionAgentStore
from ai4s_agent.scientific_agent_autonomy_l1 import (
    AUTONOMY_L1_MAX_LLM_CALLS,
    AUTONOMY_L1_MAX_TRANSITIONS,
    budget_stop_reason_codes,
    build_l1_budget_snapshot,
)
from tests.execution_agent_test_support import NOW, local_controller_execution
from tests.test_scientific_agent_conversation_session import (
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


class _CountingRevisionProvider(StubLLMProvider):
    def __init__(self) -> None:
        super().__init__(
            response={
                "rationale_summary": "Use one bounded concurrent revision.",
                "option_patch": {"generate_candidates": {"count": 4}},
            }
        )
        self.calls = 0
        self._calls_lock = Lock()

    def complete_json(self, **kwargs):
        with self._calls_lock:
            self.calls += 1
        return super().complete_json(**kwargs)


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

            def resolve(_payload, *, settings, providers):
                del settings, providers
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
) -> None:
    _storage, control_store, _controller, result = local_controller_execution(tmp_path)
    execution = result.execution
    base_receipt = result.receipt
    assert base_receipt is not None
    existing = control_store.list_harness_controller_action_receipts(
        project_id=execution.project_id,
        controller_execution_id=execution.controller_execution_id,
    )
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
    receipts = control_store.list_harness_controller_action_receipts(
        project_id=execution.project_id,
        controller_execution_id=execution.controller_execution_id,
    )
    assert len(receipts) == AUTONOMY_L1_MAX_TRANSITIONS
    snapshot = build_l1_budget_snapshot(
        execution=execution,
        transition_count=len(receipts),
        llm_call_count=0,
        remote_dispatch_count=0,
        now=NOW,
    )
    assert "AUTONOMY_L1_TRANSITION_BUDGET_EXHAUSTED" in budget_stop_reason_codes(
        snapshot,
        action=AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
        needs_llm=False,
    )


def test_l1_acceptance_rebuilds_64_llm_checkpoints_and_stops_before_provider(
    tmp_path,
) -> None:
    storage, _control_store, _controller, result = local_controller_execution(tmp_path)
    execution = result.execution
    store = ExecutionAgentStore(storage=storage)
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
    snapshot = build_l1_budget_snapshot(
        execution=execution,
        transition_count=0,
        llm_call_count=calls,
        remote_dispatch_count=0,
        now=NOW,
    )
    assert "AUTONOMY_L1_LLM_BUDGET_EXHAUSTED" in budget_stop_reason_codes(
        snapshot,
        action=AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
        needs_llm=True,
    )


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

    def resolve(_payload, *, settings, providers):
        del settings, providers
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
