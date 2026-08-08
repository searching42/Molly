from __future__ import annotations

import json
from pathlib import Path

import pytest

import ai4s_agent.scientific_agent_conversation as session_module
from ai4s_agent.app import create_app
from ai4s_agent.schemas import (
    AgentExecutionPlanLLMResponse,
    AgentHarnessControllerActionBoundaryClass,
)


pytestmark = pytest.mark.integration


def _plan_response() -> dict[str, object]:
    return AgentExecutionPlanLLMResponse(
        requested_tool_ids=["generate_candidates"],
        selected_input_artifact_ids=[],
        task_options={"generate_candidates": {}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce a reviewable report"],
        rationales=["Use the registered scientific task."],
        assumptions=[],
        questions=[],
    ).model_dump(mode="json")


def _create_conversation(tmp_path: Path):
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "user-config",
    )
    app.config["AI4S_AGENT_AUTHORIZATION_OWNER"] = "test-user"
    client = app.test_client()
    project = client.post(
        "/api/projects",
        json={"project_id": "conversation-project", "name": "Conversation project"},
    )
    assert project.status_code == 200, project.get_json()
    conversation = client.post(
        "/api/projects/conversation-project/conversations",
        json={"conversation_id": "conversation-one", "title": "Scientific work"},
    )
    assert conversation.status_code == 201, conversation.get_json()
    message = client.post(
        "/api/projects/conversation-project/conversations/conversation-one/messages",
        json={
            "role": "user",
            "content": "Generate and screen PLQY candidates.",
            "client_message_id": "user-message-1",
        },
    )
    assert message.status_code == 201, message.get_json()
    return app, client


def _stub_provider() -> dict[str, object]:
    return {
        "provider": "stub",
        "model": "stub",
        "stub_response": _plan_response(),
    }


def test_conversation_turn_publishes_real_review_only_scientific_proposal_and_sse(
    tmp_path: Path,
) -> None:
    app, client = _create_conversation(tmp_path)
    endpoint = "/api/projects/conversation-project/conversations/conversation-one/agent-session"

    response = client.post(
        endpoint + "/turn",
        json={"run_id": "conversation-run", "llm_provider": _stub_provider()},
    )

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["session"]["status"] == "approval_required"
    assert body["session"]["reason_code"] == "PLAN_APPROVAL_REQUIRED"
    assert body["approval_required"] is True
    assert body["executable"] is False
    assert body["proposal"]["executable"] is False
    assert body["plan_summary"]["goal"]
    task_ids = [item["task_id"] for item in body["plan_summary"]["tasks"]]
    assert task_ids == ["generate_candidates"]
    assert "proposal_id" in body["plan_summary"]
    assert "proposal_digest" in body["plan_summary"]
    assert body["plan_summary"]["raw_proposal"]["proposal_id"] == body["proposal"]["proposal_id"]

    proposal_path = (
        tmp_path
        / "workspace"
        / "projects"
        / "conversation-project"
        / "agent_plan_proposals"
        / body["proposal"]["proposal_id"]
        / "proposal.json"
    )
    assert proposal_path.is_file()
    persisted = json.loads(proposal_path.read_text(encoding="utf-8"))
    assert persisted["proposal_id"] == body["proposal"]["proposal_id"]
    assert persisted["executable"] is False

    reloaded = client.get(endpoint)
    assert reloaded.status_code == 200
    assert reloaded.get_json()["plan_summary"]["proposal_id"] == body["proposal"]["proposal_id"]

    stream = client.get(endpoint + "/events?once=true")
    assert stream.status_code == 200
    assert stream.mimetype == "text/event-stream"
    assert stream.headers["Cache-Control"] == "no-store"
    stream_body = stream.get_data(as_text=True)
    assert "event: snapshot" in stream_body
    assert "event: agent.status" in stream_body
    assert "PLAN_APPROVAL_REQUIRED" in stream_body
    assert '"events_may_drive_execution":false' in stream_body
    assert client.post(endpoint + "/events", json={}).status_code == 405

    reconnect = client.get(
        endpoint + "/events?once=true",
        headers={"Last-Event-ID": str(body["session"]["revision"])},
    )
    assert reconnect.status_code == 200
    assert "event: snapshot" in reconnect.get_data(as_text=True)
    assert "event: agent.status" not in reconnect.get_data(as_text=True)


def test_ambiguous_conversational_revision_does_not_authorize_pending_plan(
    tmp_path: Path,
) -> None:
    _app, client = _create_conversation(tmp_path)
    endpoint = "/api/projects/conversation-project/conversations/conversation-one/agent-session"
    first = client.post(
        endpoint + "/turn",
        json={"run_id": "conversation-run", "llm_provider": _stub_provider()},
    )
    assert first.status_code == 200
    first_body = first.get_json()

    appended = client.post(
        "/api/projects/conversation-project/conversations/conversation-one/messages",
        json={
            "role": "user",
            "content": "这个方案似乎可以，但把生成数量调低一些。",
            "client_message_id": "user-message-2",
        },
    )
    assert appended.status_code == 201
    revised = client.post(
        endpoint + "/turn",
        json={"run_id": "conversation-run", "llm_provider": _stub_provider()},
    )

    assert revised.status_code == 200, revised.get_json()
    revised_body = revised.get_json()
    assert revised_body["session"]["status"] == "approval_required"
    assert revised_body["session"]["authorization_id"] == ""
    assert revised_body["session"]["start_intent_id"] == ""
    assert revised_body["proposal"]["proposal_id"] != first_body["proposal"]["proposal_id"]


def test_exact_conversational_approval_uses_authorization_start_intent_controller_and_execution_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, client = _create_conversation(tmp_path)
    endpoint = "/api/projects/conversation-project/conversations/conversation-one/agent-session"
    first = client.post(
        endpoint + "/turn",
        json={"run_id": "conversation-run", "llm_provider": _stub_provider()},
    )
    assert first.status_code == 200, first.get_json()

    service = app.extensions["scientific_agent_conversation_session_service"]
    monkeypatch.setattr(
        session_module,
        "controller_action_boundary_class",
        lambda *_args, **_kwargs: AgentHarnessControllerActionBoundaryClass.ORDINARY_ADVANCE,
    )
    execution_agent_calls: list[bool] = []
    original_create_proposal = service.execution_agent.create_proposal

    def spy_create_proposal(*args, **kwargs):
        execution_agent_calls.append(True)
        return original_create_proposal(*args, **kwargs)

    monkeypatch.setattr(service.execution_agent, "create_proposal", spy_create_proposal)

    appended = client.post(
        "/api/projects/conversation-project/conversations/conversation-one/messages",
        json={
            "role": "user",
            "content": "确认执行",
            "client_message_id": "user-message-approval",
        },
    )
    assert appended.status_code == 201
    approved = client.post(
        endpoint + "/turn",
        json={"run_id": "conversation-run", "llm_provider": _stub_provider()},
    )

    assert approved.status_code == 200, approved.get_json()
    body = approved.get_json()
    assert body["session"]["authorization_id"]
    assert body["session"]["start_intent_id"]
    assert body["session"]["controller_execution_id"]
    assert execution_agent_calls == [True]
    assert body["session"]["status"] in {"failed", "unknown", "waiting_gate", "waiting_remote_approval", "recovery_required", "succeeded"}


def test_plan_approval_does_not_auto_approve_a_later_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, client = _create_conversation(tmp_path)
    endpoint = "/api/projects/conversation-project/conversations/conversation-one/agent-session"
    first = client.post(
        endpoint + "/turn",
        json={"run_id": "conversation-run", "llm_provider": _stub_provider()},
    )
    assert first.status_code == 200

    service = app.extensions["scientific_agent_conversation_session_service"]
    execution_agent_calls: list[bool] = []
    original_create_proposal = service.execution_agent.create_proposal

    def spy_create_proposal(*args, **kwargs):
        execution_agent_calls.append(True)
        return original_create_proposal(*args, **kwargs)

    monkeypatch.setattr(service.execution_agent, "create_proposal", spy_create_proposal)
    appended = client.post(
        "/api/projects/conversation-project/conversations/conversation-one/messages",
        json={
            "role": "user",
            "content": "确认执行",
            "client_message_id": "user-message-gate-approval",
        },
    )
    assert appended.status_code == 201

    approved = client.post(
        endpoint + "/turn",
        json={"run_id": "conversation-run", "llm_provider": _stub_provider()},
    )
    assert approved.status_code == 200, approved.get_json()
    body = approved.get_json()
    assert body["session"]["authorization_id"]
    assert body["session"]["start_intent_id"]
    assert body["session"]["controller_execution_id"]
    assert body["session"]["status"] == "waiting_gate"
    assert body["session"]["reason_code"] == "USER_GATE_APPROVAL_REQUIRED"
    assert execution_agent_calls == []
