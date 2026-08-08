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


def test_session_sse_reconnect_recovers_from_a_torn_final_event_record(
    tmp_path: Path,
) -> None:
    app, client = _create_conversation(tmp_path)
    endpoint = "/api/projects/conversation-project/conversations/conversation-one/agent-session"
    first = client.post(
        endpoint + "/turn",
        json={"run_id": "conversation-run", "llm_provider": _stub_provider()},
    )
    assert first.status_code == 200
    service = app.extensions["scientific_agent_conversation_session_service"]
    events_path = service._root(
        "conversation-project",
        "conversation-one",
        create=False,
    ) / "events.jsonl"
    with events_path.open("ab") as handle:
        handle.write(b'{"event_id":2,"truncated":')

    stream = client.get(endpoint + "/events?once=true")
    assert stream.status_code == 200
    assert stream.get_data(as_text=True).count("event: agent.status") == 1
    reconnect = client.get(
        endpoint + "/events?once=true",
        headers={"Last-Event-ID": "1"},
    )
    assert reconnect.status_code == 200
    assert "event: agent.status" not in reconnect.get_data(as_text=True)

    service._transition(
        project_id="conversation-project",
        conversation_id="conversation-one",
        status="approval_required",
        reason_code="PLAN_APPROVAL_REQUIRED",
        event_type="test.journal_repair",
    )
    assert len(service.read_events(
        project_id="conversation-project",
        conversation_id="conversation-one",
    )) == 2
    assert b"truncated" not in events_path.read_bytes()


def test_session_state_recovers_when_state_write_fails_after_event_commit(
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
    original_write_json = session_module.write_json

    def fail_state_write(path, payload):
        if Path(path).name == "state.json":
            raise OSError("simulated state write crash")
        return original_write_json(path, payload)

    monkeypatch.setattr(session_module, "write_json", fail_state_write)
    with pytest.raises(OSError, match="simulated state write crash"):
        service._transition(
            project_id="conversation-project",
            conversation_id="conversation-one",
            status="waiting_gate",
            reason_code="USER_GATE_APPROVAL_REQUIRED",
            event_type="test.state_recovery",
        )

    recovered = service.read_session(
        project_id="conversation-project",
        conversation_id="conversation-one",
    )
    assert recovered["status"] == "waiting_gate"
    assert recovered["reason_code"] == "USER_GATE_APPROVAL_REQUIRED"
    assert recovered["revision"] == 2
    assert len(service.read_events(
        project_id="conversation-project",
        conversation_id="conversation-one",
    )) == 2


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


@pytest.mark.parametrize(
    ("boundary", "expected_status"),
    [
        (
            AgentHarnessControllerActionBoundaryClass.USER_GATE_APPROVAL,
            "waiting_gate",
        ),
        (
            AgentHarnessControllerActionBoundaryClass.USER_REMOTE_APPROVAL,
            "waiting_remote_approval",
        ),
        (
            AgentHarnessControllerActionBoundaryClass.EXPLICIT_RECOVERY,
            "recovery_required",
        ),
    ],
)
def test_active_controller_binding_survives_a_later_ordinary_chat_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: AgentHarnessControllerActionBoundaryClass,
    expected_status: str,
) -> None:
    app, client = _create_conversation(tmp_path)
    endpoint = "/api/projects/conversation-project/conversations/conversation-one/agent-session"
    first = client.post(
        endpoint + "/turn",
        json={"run_id": "conversation-run", "llm_provider": _stub_provider()},
    )
    assert first.status_code == 200

    service = app.extensions["scientific_agent_conversation_session_service"]
    monkeypatch.setattr(
        session_module,
        "controller_action_boundary_class",
        lambda *_args, **_kwargs: boundary,
    )
    appended = client.post(
        "/api/projects/conversation-project/conversations/conversation-one/messages",
        json={
            "role": "user",
            "content": "确认执行",
            "client_message_id": f"user-message-{expected_status}",
        },
    )
    assert appended.status_code == 201
    approved = client.post(
        endpoint + "/turn",
        json={"run_id": "conversation-run", "llm_provider": _stub_provider()},
    )
    assert approved.status_code == 200, approved.get_json()
    approved_body = approved.get_json()
    assert approved_body["session"]["status"] == expected_status

    plan_calls: list[bool] = []
    original_create_proposal = service.plan_service.create_proposal

    def spy_create_proposal(*args, **kwargs):
        plan_calls.append(True)
        return original_create_proposal(*args, **kwargs)

    monkeypatch.setattr(service.plan_service, "create_proposal", spy_create_proposal)
    ordinary = client.post(
        "/api/projects/conversation-project/conversations/conversation-one/messages",
        json={
            "role": "user",
            "content": "现在需要我做什么？",
            "client_message_id": f"user-message-follow-up-{expected_status}",
        },
    )
    assert ordinary.status_code == 201
    followed = client.post(
        endpoint + "/turn",
        json={"run_id": "conversation-run", "llm_provider": _stub_provider()},
    )

    assert followed.status_code == 200, followed.get_json()
    followed_body = followed.get_json()
    assert followed_body["session"]["status"] == expected_status
    for field in (
        "proposal_id",
        "authorization_id",
        "start_intent_id",
        "controller_execution_id",
    ):
        assert followed_body["session"][field] == approved_body["session"][field]
    assert followed_body["proposal"]["proposal_id"] == approved_body["proposal"]["proposal_id"]
    assert plan_calls == []


def test_exact_approval_starts_without_external_llm_consent_for_existing_plan(
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

    saved = client.patch(
        "/api/settings/llm",
        json={
            "endpoint": "https://llm.example.test/v1",
            "model": "external-model",
            "api_key_source": "file",
            "api_key": "external-secret",
        },
    )
    assert saved.status_code == 200, saved.get_json()
    assert client.get("/api/settings/llm").get_json()["external_llm_data_sharing_enabled"] is False

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
            "client_message_id": "user-message-external-consent-off",
        },
    )
    assert appended.status_code == 201
    approved = client.post(endpoint + "/turn", json={"run_id": "conversation-run"})

    assert approved.status_code == 200, approved.get_json()
    body = approved.get_json()
    assert body["session"]["authorization_id"]
    assert body["session"]["start_intent_id"]
    assert body["session"]["controller_execution_id"]
    assert execution_agent_calls == []


def test_agent_session_routes_return_fixed_errors_without_exception_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, client = _create_conversation(tmp_path)
    endpoint = "/api/projects/conversation-project/conversations/conversation-one/agent-session"
    service = app.extensions["scientific_agent_conversation_session_service"]

    def raise_internal_error(**_kwargs):
        raise ValueError("secret/path and internal traceback details")

    monkeypatch.setattr(service, "read_session_payload", raise_internal_error)
    snapshot = client.get(endpoint)
    assert snapshot.status_code == 400
    assert snapshot.get_json() == {
        "ok": False,
        "error_code": "invalid_conversation_session_request",
        "error": "Invalid conversation session request.",
    }
    assert "secret/path" not in snapshot.get_data(as_text=True)

    invalid_turn = client.post(endpoint + "/turn", json={"secret": "internal"})
    assert invalid_turn.status_code == 400
    assert invalid_turn.get_json() == {
        "ok": False,
        "error_code": "invalid_conversation_session_request",
        "error": "Invalid conversation session request.",
    }

    invalid_cursor = client.get(endpoint + "/events?after=secret")
    assert invalid_cursor.status_code == 400
    assert invalid_cursor.get_json() == {
        "ok": False,
        "error_code": "invalid_durable_event_cursor",
        "error": "Invalid durable event cursor.",
    }
