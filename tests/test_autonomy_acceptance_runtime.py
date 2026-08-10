"""Acceptance-only matrix checks used by the formal L1/L2 runner."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from dataclasses import replace

import pytest

import ai4s_agent.routes.scientific_agent_conversation as conversation_routes
from ai4s_agent.schemas import (
    AgentHarnessControllerAction,
    AgentHarnessControllerStatus,
    _agent_digest,
)
from tests.test_scientific_agent_conversation_session import (
    _start_waiting_gate_session_with_client,
    _typed_controller_inspection_variant,
)


pytestmark = pytest.mark.acceptance


class _CountingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, **_kwargs):
        self.calls += 1
        raise AssertionError("non-FAILED replan entered the provider")

    def close(self) -> None:
        return None


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
