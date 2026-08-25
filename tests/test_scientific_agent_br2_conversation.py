from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

from ai4s_agent.app import create_app
from ai4s_agent.planner import br2_contextual_mapping_task_registry_v1
from ai4s_agent.schemas import (
    AgentExecutionPlanLLMResponse,
    AgentHarnessControllerAction,
    AgentHarnessControllerStatus,
)


def _planner_provider() -> dict[str, object]:
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["prepare_oled_candidate_raw_dataset"],
        selected_input_artifact_ids=["pdf_corpus"],
        task_options={"prepare_oled_candidate_raw_dataset": {}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop before scientific confirmation"],
        success_criteria=["publish an evidence-bound review package"],
        rationales=["Use the registered BR2 review chain."],
        assumptions=[],
        questions=[],
    )
    return {
        "provider": "stub",
        "model": "stub",
        "stub_response": response.model_dump(mode="json"),
    }


def test_br2_conversation_binds_one_uploaded_pdf_to_existing_run_artifact(
    tmp_path: Path,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
        scientific_task_registry=br2_contextual_mapping_task_registry_v1(),
    )
    app.config.update(TESTING=True)
    client = app.test_client()

    project = client.post(
        "/api/projects",
        json={"project_id": "br2-conversation", "name": "BR2 conversation"},
    )
    assert project.status_code == 200
    conversation = client.post(
        "/api/projects/br2-conversation/conversations",
        json={"conversation_id": "paper-review", "title": "OLED paper review"},
    )
    assert conversation.status_code == 201
    pdf = b"%PDF-1.7\ncontent-bound test PDF\n"
    upload = client.post(
        "/api/projects/br2-conversation/conversations/paper-review/attachments",
        data={
            "files": (
                io.BytesIO(pdf),
                "oled-paper-018.pdf",
                "application/pdf",
            )
        },
        content_type="multipart/form-data",
    )
    assert upload.status_code == 201, upload.get_json()
    attachment = upload.get_json()["attachments"][0]
    message = client.post(
        "/api/projects/br2-conversation/conversations/paper-review/messages",
        json={
            "role": "user",
            "content": "解析这篇 OLED 文献，整理后让我确认。",
            "attachment_ids": [attachment["artifact_id"]],
            "client_message_id": "br2-paper-request",
        },
    )
    assert message.status_code == 201, message.get_json()

    response = client.post(
        "/api/projects/br2-conversation/conversations/paper-review/agent-session/turn",
        json={"run_id": "br2-conversation-run", "llm_provider": _planner_provider()},
    )
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["proposal"]["run_plan"]["tasks"][0]["task_id"] == "parse_document"
    assert [
        task["task_id"] for task in body["proposal"]["run_plan"]["tasks"]
    ] == [
        "parse_document",
        "extract_oled_evidence",
        "map_oled_contextual_semantics",
        "prepare_oled_candidate_raw_dataset",
    ]

    run_dir = tmp_path / "workspace" / "projects" / "br2-conversation" / "runs" / "br2-conversation-run"
    registry = json.loads((run_dir / "artifact_registry.json").read_text(encoding="utf-8"))
    assert registry["artifacts"]["pdf_corpus"] == "conversation_input/pdf_corpus.pdf"
    assert (run_dir / registry["artifacts"]["pdf_corpus"]).read_bytes() == pdf
    serialized_proposal = json.dumps(body["proposal"], ensure_ascii=False)
    assert str(tmp_path) not in serialized_proposal

    assert client.post(
        "/api/projects/br2-conversation/conversations/paper-review/messages",
        json={"role": "user", "content": "确认执行"},
    ).status_code == 201
    service = app.extensions["scientific_agent_conversation_session_service"]
    assert (
        service._resolve_br2_pdf_input(
            project_id="br2-conversation",
            conversation_id="paper-review",
            run_id="br2-conversation-run",
            last_user_content="确认执行",
        )
        == ()
    )


def test_br2_conversation_does_not_guess_between_multiple_uploaded_pdfs(
    tmp_path: Path,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
        scientific_task_registry=br2_contextual_mapping_task_registry_v1(),
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    assert client.post(
        "/api/projects",
        json={"project_id": "br2-ambiguous", "name": "BR2 ambiguous"},
    ).status_code == 200
    assert client.post(
        "/api/projects/br2-ambiguous/conversations",
        json={"conversation_id": "paper-review", "title": "OLED papers"},
    ).status_code == 201
    response = client.post(
        "/api/projects/br2-ambiguous/conversations/paper-review/attachments",
        data={
            "files": [
                (io.BytesIO(b"%PDF-1.7\none\n"), "paper-a.pdf", "application/pdf"),
                (io.BytesIO(b"%PDF-1.7\ntwo\n"), "paper-b.pdf", "application/pdf"),
            ]
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    attachments = response.get_json()["attachments"]
    assert client.post(
        "/api/projects/br2-ambiguous/conversations/paper-review/messages",
        json={
            "role": "user",
            "content": "从这些 OLED 文献中提取候选数据后让我确认。",
            "attachment_ids": [item["artifact_id"] for item in attachments],
        },
    ).status_code == 201

    response = client.post(
        "/api/projects/br2-ambiguous/conversations/paper-review/agent-session/turn",
        json={"run_id": "br2-ambiguous-run", "llm_provider": _planner_provider()},
    )
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["session"]["status"] == "needs_input"
    assert body["session"]["reason_code"] == "BR2_PDF_SELECTION_REQUIRED"
    assert "paper-a.pdf" in body["assistant_message"]
    assert "paper-b.pdf" in body["assistant_message"]


def test_br2_conversation_binds_exactly_selected_pdf_after_ambiguity(
    tmp_path: Path,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
        scientific_task_registry=br2_contextual_mapping_task_registry_v1(),
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    assert client.post(
        "/api/projects",
        json={"project_id": "br2-selection", "name": "BR2 selection"},
    ).status_code == 200
    assert client.post(
        "/api/projects/br2-selection/conversations",
        json={"conversation_id": "paper-review", "title": "OLED papers"},
    ).status_code == 201
    response = client.post(
        "/api/projects/br2-selection/conversations/paper-review/attachments",
        data={
            "files": [
                (io.BytesIO(b"%PDF-1.7\nselected\n"), "paper-a.pdf", "application/pdf"),
                (io.BytesIO(b"%PDF-1.7\nnot-selected\n"), "paper-b.pdf", "application/pdf"),
            ]
        },
        content_type="multipart/form-data",
    )
    attachments = response.get_json()["attachments"]
    assert client.post(
        "/api/projects/br2-selection/conversations/paper-review/messages",
        json={
            "role": "user",
            "content": "从这些 OLED 文献中提取候选数据后让我确认。",
            "attachment_ids": [item["artifact_id"] for item in attachments],
        },
    ).status_code == 201
    first = client.post(
        "/api/projects/br2-selection/conversations/paper-review/agent-session/turn",
        json={"run_id": "br2-selection-run", "llm_provider": _planner_provider()},
    )
    assert first.status_code == 200
    assert first.get_json()["session"]["status"] == "needs_input"

    assert client.post(
        "/api/projects/br2-selection/conversations/paper-review/messages",
        json={
            "role": "user",
            "content": "选择 paper-a.pdf",
        },
    ).status_code == 201
    selected = client.post(
        "/api/projects/br2-selection/conversations/paper-review/agent-session/turn",
        json={"run_id": "br2-selection-run", "llm_provider": _planner_provider()},
    )
    assert selected.status_code == 200, selected.get_json()
    body = selected.get_json()
    assert body["proposal"]["run_plan"]["tasks"][0]["task_id"] == "parse_document"
    run_dir = (
        tmp_path
        / "workspace"
        / "projects"
        / "br2-selection"
        / "runs"
        / "br2-selection-run"
    )
    registry = json.loads((run_dir / "artifact_registry.json").read_text(encoding="utf-8"))
    assert (run_dir / registry["artifacts"]["pdf_corpus"]).read_bytes() == b"%PDF-1.7\nselected\n"


def test_br2_conversation_stops_at_existing_review_boundary_and_replays_without_effect(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
        scientific_task_registry=br2_contextual_mapping_task_registry_v1(),
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    assert client.post(
        "/api/projects",
        json={"project_id": "br2-boundary", "name": "BR2 boundary"},
    ).status_code == 200
    assert client.post(
        "/api/projects/br2-boundary/conversations",
        json={"conversation_id": "paper-review", "title": "OLED paper"},
    ).status_code == 201
    upload = client.post(
        "/api/projects/br2-boundary/conversations/paper-review/attachments",
        data={
            "files": (
                io.BytesIO(b"%PDF-1.7\nreview boundary\n"),
                "oled-paper-018.pdf",
                "application/pdf",
            )
        },
        content_type="multipart/form-data",
    )
    assert upload.status_code == 201
    attachment = upload.get_json()["attachments"][0]
    assert client.post(
        "/api/projects/br2-boundary/conversations/paper-review/messages",
        json={
            "role": "user",
            "content": "解析这篇 OLED 文献，整理后让我确认。",
            "attachment_ids": [attachment["artifact_id"]],
        },
    ).status_code == 201

    endpoint = "/api/projects/br2-boundary/conversations/paper-review/agent-session"
    proposed = client.post(
        endpoint + "/turn",
        json={"run_id": "br2-boundary-run", "llm_provider": _planner_provider()},
    )
    assert proposed.status_code == 200, proposed.get_json()
    service = app.extensions["scientific_agent_conversation_session_service"]
    state = service.read_session(
        project_id="br2-boundary",
        conversation_id="paper-review",
    )
    publication = service._read_pending_publication(state, "br2-boundary")
    assert service._is_br2_mapping_proposal(publication) is True

    execution = SimpleNamespace(
        run_id="br2-boundary-run",
        controller_execution_id="controller-br2-boundary",
        execution_digest="sha256:" + "1" * 64,
    )
    inspection = SimpleNamespace(
        status=AgentHarnessControllerStatus.SUCCEEDED,
        current_task_id="prepare_oled_candidate_raw_dataset",
        next_action=AgentHarnessControllerAction.STOP_TASK_TERMINAL,
        inspection_digest="sha256:" + "2" * 64,
    )
    controller_result = SimpleNamespace(
        execution=execution,
        inspection=inspection,
        receipt=None,
    )
    review_projection = {
        "schema_version": "scientific_agent_review_projection.v1",
        "review_kind": "br2_oled_candidate_raw_dataset",
        "read_only": True,
        "authoritative": False,
        "current_task_id": "prepare_oled_candidate_raw_dataset",
        "gate_id": "",
        "snapshot_id": "candidate_raw_dataset_review",
        "snapshot_digest": "sha256:" + "3" * 64,
        "review_snapshot_id": "candidate_raw_dataset_review",
        "review_snapshot_digest": "sha256:" + "3" * 64,
        "paper_id": "oled-paper-018",
        "dataset_scope": "molecule_interaction_properties_only",
        "target_property": "",
        "scientific_scope": "molecule_interaction_properties_only",
        "comparability_policy": "",
        "row_count": 1,
        "included_count": 1,
        "excluded_count": 0,
        "duplicate_count": 0,
        "conflict_count": 0,
        "candidate_record_count": 1,
        "property_observation_count": 1,
        "compiled_count": 0,
        "partial_count": 1,
        "needs_review_count": 0,
        "rejected_count": 0,
        "unresolved_count": 0,
        "source_check_count": 0,
        "ontology_review_count": 0,
        "device_only_excluded_count": 0,
        "evidence_observation_count": 1,
        "evidence_bound_observation_count": 1,
        "evidence_bound_records_count": 1,
        "all_promoted_rows_have_evidence": True,
        "counts": {"row": 1, "included": 1, "excluded": 0, "duplicates": 0, "conflicts": 0},
        "reason_code_counts": {},
        "confirmation_required": True,
    }
    monkeypatch.setattr(
        service,
        "_read_active_publication",
        lambda _state, _project_id: publication,
    )
    monkeypatch.setattr(
        service,
        "_project_br2_candidate_review",
        lambda **_kwargs: review_projection,
    )
    monkeypatch.setattr(
        "ai4s_agent.scientific_agent_conversation.validate_l1_execution_inspection",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "ai4s_agent.scientific_agent_conversation.classify_current_controller_inspection",
        lambda _inspection: object(),
    )
    monkeypatch.setattr(
        service,
        "_l1_projection_updates",
        lambda **_kwargs: {
            "autonomy_level": "L1",
            "autonomy_status": "human_boundary",
            "autonomy_stop_reason": "BR2_CANDIDATE_CONFIRMATION_REQUIRED",
        },
    )

    running_state = service._transition(
        project_id="br2-boundary",
        conversation_id="paper-review",
        status="running",
        reason_code="RUN_STARTED",
        updates={
            "run_id": "br2-boundary-run",
            "controller_execution_id": execution.controller_execution_id,
            "controller_execution_digest": execution.execution_digest,
            "controller_status": AgentHarnessControllerStatus.SUCCEEDED.value,
        },
        event_type="test.br2.terminal_input",
    )
    _controller_result, waiting_state, stop_reason = service._auto_progress(
        project_id="br2-boundary",
        conversation_id="paper-review",
        state=running_state,
        controller_result=controller_result,
        provider=None,
        provider_binding_digest="",
    )
    assert stop_reason == "br2_confirmation"
    assert waiting_state["status"] == "waiting_gate"
    assert waiting_state["reason_code"] == "BR2_CANDIDATE_CONFIRMATION_REQUIRED"
    assert waiting_state["controller_status"] == AgentHarnessControllerStatus.SUCCEEDED.value
    assert waiting_state["authority_kind"] == ""
    assert waiting_state["review_projection"]["confirmation_required"] is True
    assert waiting_state["result_projections"] == []
    assert waiting_state["autonomy_status"] == "human_boundary"
    assert "不会自动确认" in waiting_state["message"]

    get_calls: list[dict[str, object]] = []

    class ReplayController:
        def get(self, **kwargs):
            get_calls.append(kwargs)
            return controller_result

    monkeypatch.setattr(service, "controller", ReplayController())
    replay = service.tick(
        project_id="br2-boundary",
        conversation_id="paper-review",
        run_id="br2-boundary-run",
        provider=None,
        provider_binding_digest="",
    )
    assert replay.session["status"] == "waiting_gate"
    assert replay.session["reason_code"] == "BR2_CANDIDATE_CONFIRMATION_REQUIRED"
    assert replay.session["review_projection"] == review_projection
    assert len(get_calls) == 1
