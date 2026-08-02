from __future__ import annotations

import pytest
from flask import Flask

from ai4s_agent.llm_provider import LLMProviderManager
from ai4s_agent.llm_settings import LLMSettingsStore
from ai4s_agent.routes.execution_agent import register_execution_agent_routes
from tests.execution_agent_test_support import (
    execution_agent_service,
    local_controller_execution,
)


def _client(tmp_path):
    storage, _, controller, initial = local_controller_execution(tmp_path)
    app = Flask(__name__)
    app.config.update(TESTING=True)
    register_execution_agent_routes(
        app,
        service=execution_agent_service(storage=storage, controller=controller),
        llm_settings=LLMSettingsStore(
            tmp_path / "workspace",
            config_dir=tmp_path / "config",
            environ={},
        ),
        llm_providers=LLMProviderManager(),
    )
    return app.test_client(), initial


def _proposal_body(initial, *, tool_id: str, request_id: str = "proposal-api-1"):
    return {
        "expected_controller_execution_digest": initial.execution.execution_digest,
        "client_request_id": request_id,
        "external_llm_approved": True,
        "llm_provider": {
            "provider": "stub",
            "stub_response": {
                "selected_tool_id": tool_id,
                "decision_summary": "Select one bounded server operation.",
            },
        },
    }


def _root(initial) -> str:
    return (
        "/api/projects/project-1/agent-harness-controller-executions/"
        f"{initial.execution.controller_execution_id}/execution-agent-proposals"
    )


@pytest.mark.pr_fast
def test_execution_agent_api_create_get_and_apply(tmp_path) -> None:
    client, initial = _client(tmp_path)
    created = client.post(
        _root(initial),
        json=_proposal_body(initial, tool_id="agent.pause_current.v1"),
    )
    assert created.status_code == 201
    payload = created.get_json()
    assert payload["applied"] is False
    assert payload["dispatched"] is False
    assert "raw_response" not in created.get_data(as_text=True)
    proposal = payload["tool_call_proposal"]

    read = client.get(f"{_root(initial)}/{proposal['tool_call_proposal_id']}")
    assert read.status_code == 200
    assert read.get_json()["current"] is True
    applied = client.post(
        f"{_root(initial)}/{proposal['tool_call_proposal_id']}/apply",
        json={
            "expected_tool_call_proposal_digest": proposal[
                "tool_call_proposal_digest"
            ],
            "client_request_id": "apply-api-1",
        },
    )
    assert applied.status_code == 200
    applied_payload = applied.get_json()
    assert applied_payload["application_receipt"]["outcome"] == "paused"
    assert applied_payload["controller_advance_called"] is False
    assert applied_payload["dispatch_occurred"] is False
    assert applied_payload["dispatched"] is False


def test_execution_agent_api_does_not_call_complete_execution_a_dispatch(
    tmp_path,
) -> None:
    client, initial = _client(tmp_path)
    created = client.post(
        _root(initial),
        json=_proposal_body(
            initial,
            tool_id="controller.advance_current.v1",
            request_id="proposal-api-complete-1",
        ),
    ).get_json()
    proposal = created["tool_call_proposal"]
    applied = client.post(
        f"{_root(initial)}/{proposal['tool_call_proposal_id']}/apply",
        json={
            "expected_tool_call_proposal_digest": proposal[
                "tool_call_proposal_digest"
            ],
            "client_request_id": "apply-api-complete-1",
        },
    )
    assert applied.status_code == 200
    payload = applied.get_json()
    assert payload["controller_decision"]["action_kind"] == "complete_execution"
    assert payload["controller_advance_called"] is True
    assert payload["dispatch_occurred"] is False
    assert payload["dispatched"] is False
    assert payload["application_receipt"]["dispatch_occurred"] is False


def test_execution_agent_api_rejects_every_client_authority_injection(tmp_path) -> None:
    client, initial = _client(tmp_path)
    forbidden = {
        "messages": [],
        "prompt": "run",
        "observation": {},
        "tool_catalog": {},
        "selected_tool_id": "controller.advance_current.v1",
        "task_id": "inspect_dataset",
        "task_index": 0,
        "adapter": "inspect_dataset",
        "route": "local_executor",
        "arguments": {},
        "profile": "default",
        "resources": {},
        "connection": "worker",
        "host": "worker.internal",
        "path": "/tmp/input",
        "command": "run",
        "argv": ["run"],
        "approval": True,
        "gate_decision": "approved",
        "remote_approval": True,
        "recover": True,
        "cancel": True,
        "retry": True,
        "traceparent": "00-abc",
        "actor": "alice",
        "status": "ready",
    }
    baseline = _proposal_body(
        initial,
        tool_id="agent.pause_current.v1",
        request_id="injection-baseline",
    )
    for index, (field, value) in enumerate(forbidden.items()):
        body = {**baseline, "client_request_id": f"injection-{index}", field: value}
        response = client.post(_root(initial), json=body)
        assert response.status_code == 400, field
        assert response.get_json()["reason_codes"] == [
            "CLIENT_AUTHORITY_FIELD_REJECTED"
        ]


def test_execution_agent_api_requires_literal_consent_and_provider(tmp_path) -> None:
    client, initial = _client(tmp_path)
    for invalid in (False, 1, "true", None):
        body = _proposal_body(
            initial,
            tool_id="agent.pause_current.v1",
            request_id=f"invalid-consent-{str(invalid).lower()}",
        )
        body["external_llm_approved"] = invalid
        response = client.post(_root(initial), json=body)
        assert response.status_code == 400

    unavailable = client.post(
        _root(initial),
        json={
            "expected_controller_execution_digest": (
                initial.execution.execution_digest
            ),
            "client_request_id": "provider-unavailable-1",
            "external_llm_approved": True,
        },
    )
    assert unavailable.status_code == 409
    assert unavailable.get_json()["reason_codes"] == [
        "EXECUTION_AGENT_LLM_UNAVAILABLE"
    ]


def test_execution_agent_api_rejects_application_authority_fields(tmp_path) -> None:
    client, initial = _client(tmp_path)
    created = client.post(
        _root(initial),
        json=_proposal_body(initial, tool_id="agent.pause_current.v1"),
    ).get_json()
    proposal = created["tool_call_proposal"]
    base = {
        "expected_tool_call_proposal_digest": proposal["tool_call_proposal_digest"],
        "client_request_id": "apply-injection-1",
    }
    for field in (
        "selected_tool_id",
        "controller_action",
        "task",
        "arguments",
        "approval",
        "recover",
        "cancel",
        "retry",
        "actor",
    ):
        response = client.post(
            f"{_root(initial)}/{proposal['tool_call_proposal_id']}/apply",
            json={**base, field: True},
        )
        assert response.status_code == 400, field


def test_execution_agent_api_never_persists_provider_secret_or_raw_response(
    tmp_path,
) -> None:
    client, initial = _client(tmp_path)
    secret = "private-provider-secret-123456"
    body = _proposal_body(initial, tool_id="agent.pause_current.v1")
    body["llm_provider"]["api_key"] = secret
    response = client.post(_root(initial), json=body)
    assert response.status_code == 201
    project_root = tmp_path / "workspace" / "projects" / "project-1"
    persisted = b"".join(
        path.read_bytes()
        for path in project_root.rglob("*")
        if path.is_file()
    )
    assert secret.encode() not in persisted
    assert b'"raw_response"' not in persisted
