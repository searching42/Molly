from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ai4s_agent.app import create_app
from ai4s_agent.scientific_agent_autonomy_l2 import (
    classify_plan_revision_materiality,
)
from ai4s_agent.scientific_agent_harness_controller import (
    ScientificAgentHarnessControllerError,
)
from ai4s_agent.scientific_agent_replanner import ScientificAgentReplannerConflict
from ai4s_agent.schemas import (
    AgentExecutionPlanLLMResponse,
    AgentHarnessControllerAction,
    AgentHarnessControllerInspection,
    AgentHarnessControllerStartRequest,
    AgentHarnessControllerStatus,
    AgentPlanRevisionApplicationRequest,
    _agent_digest,
)
from ai4s_agent.storage import ProjectStorage


pytestmark = pytest.mark.integration

_CLOCK = "2026-08-26T00:00:00Z"
_PROJECT_ID = "minimal-golden-path-project"
_CONVERSATION_ID = "minimal-golden-path-conversation"
_RUN_ID = "minimal-golden-path-run"


def _plan_response(
    tool_id: str,
    *,
    options: dict[str, object],
    selected_input_artifact_ids: list[str] | None = None,
    stop_conditions: list[str] | None = None,
) -> dict[str, object]:
    return AgentExecutionPlanLLMResponse(
        requested_tool_ids=[tool_id],
        selected_input_artifact_ids=selected_input_artifact_ids or [],
        task_options={tool_id: options},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=stop_conditions or ["stop on validation failure"],
        success_criteria=["produce a reviewable deterministic result"],
        rationales=["Use the registered deterministic task."],
        assumptions=[],
        questions=[],
    ).model_dump(mode="json")


def _provider(response: dict[str, object]) -> dict[str, object]:
    return {
        "provider": "stub",
        "model": "minimal-golden-path-stub",
        "stub_response": response,
    }


def _create_conversation(
    tmp_path: Path,
    *,
    content: str,
) -> tuple[object, object, object, ProjectStorage, str]:
    workspace = tmp_path / "workspace"
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=workspace,
        user_config_dir=tmp_path / "config",
    )
    app.config.update(
        TESTING=True,
        AI4S_AGENT_AUTHORIZATION_OWNER="alice",
    )
    service = app.extensions["scientific_agent_conversation_session_service"]
    for component in (
        service,
        service.plan_service,
        service.authorization_service,
        service.controller,
        service.replanner,
        app.extensions["scientific_agent_plan_observation_builder"],
        app.extensions["scientific_agent_autonomy_lease_service"],
        app.extensions["scientific_agent_autonomy_grant_issuer"],
        app.extensions["scientific_agent_autonomy_grant_issuer"].grant_store,
    ):
        if hasattr(component, "clock"):
            component.clock = lambda: _CLOCK

    client = app.test_client()
    project = client.post(
        "/api/projects",
        json={"project_id": _PROJECT_ID, "name": "Minimal golden path"},
    )
    assert project.status_code == 200, project.get_json()
    conversation = client.post(
        f"/api/projects/{_PROJECT_ID}/conversations",
        json={"conversation_id": _CONVERSATION_ID, "title": "Minimal golden path"},
    )
    assert conversation.status_code == 201, conversation.get_json()
    message = client.post(
        f"/api/projects/{_PROJECT_ID}/conversations/{_CONVERSATION_ID}/messages",
        json={
            "role": "user",
            "content": content,
            "client_message_id": "minimal-golden-path-message",
        },
    )
    assert message.status_code == 201, message.get_json()
    return (
        app,
        client,
        service,
        ProjectStorage(workspace_dir=workspace),
        f"/api/projects/{_PROJECT_ID}/conversations/{_CONVERSATION_ID}/agent-session",
    )


def _confirm_and_turn(
    client: object,
    endpoint: str,
    *,
    response: dict[str, object],
    message_id: str,
) -> object:
    pending = client.get(endpoint)
    assert pending.status_code == 200, pending.get_json()
    pending_session = pending.get_json()["session"]
    structured = client.post(
        endpoint + "/approve",
        json={
            "expected_proposal_digest": pending_session["proposal_digest"],
            "authorization_mode": "stepwise",
            "requested_preauthorized_gate_ids": [],
            "confirmed": True,
            "client_request_id": message_id,
            "note": "Explicit structured test approval.",
        },
    )
    assert structured.status_code == 200, structured.get_json()
    return client.post(
        endpoint + "/tick",
        json={"run_id": _RUN_ID, "llm_provider": _provider(response)},
    )


def _failed_controller_result(service: object) -> object:
    state = service.read_session(
        project_id=_PROJECT_ID,
        conversation_id=_CONVERSATION_ID,
    )
    controller_result = service.controller.get(
        project_id=_PROJECT_ID,
        controller_execution_id=state["controller_execution_id"],
    )
    receipts = service.controller.control_store.list_harness_controller_action_receipts(
        project_id=_PROJECT_ID,
        controller_execution_id=state["controller_execution_id"],
    )
    assert receipts
    payload = controller_result.inspection.model_dump(mode="json")
    payload.update(
        {
            "status": AgentHarnessControllerStatus.FAILED.value,
            "next_action": AgentHarnessControllerAction.STOP_TASK_TERMINAL.value,
            "inspection_digest": "",
        }
    )
    return replace(
        controller_result,
        receipt=receipts[-1],
        inspection=AgentHarnessControllerInspection.model_validate(payload),
    )


@pytest.mark.pr_fast
def test_minimal_harness_golden_path_authorizes_executes_and_replays(
    tmp_path: Path,
) -> None:
    _app, client, service, storage, endpoint = _create_conversation(
        tmp_path,
        content="Inspect this deterministic dataset for PLQY candidates.",
    )
    run_dir = storage.run_dir(_PROJECT_ID, _RUN_ID)
    dataset = run_dir / "inputs" / "dataset.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("SMILES,value\nCCO,1.0\n", encoding="utf-8")
    storage.register_artifact_path(
        _PROJECT_ID,
        _RUN_ID,
        "uploaded_dataset",
        "inputs/dataset.csv",
    )
    response = _plan_response(
        "inspect_dataset",
        options={},
        selected_input_artifact_ids=["uploaded_dataset"],
    )

    proposed = client.post(
        endpoint + "/turn",
        json={"run_id": _RUN_ID, "llm_provider": _provider(response)},
    )
    assert proposed.status_code == 200, proposed.get_json()
    proposal_payload = proposed.get_json()["proposal"]
    assert proposal_payload["executable"] is False
    assert proposal_payload["proposal_digest"].startswith("sha256:")
    publication = service.proposal_store.read(
        project_id=_PROJECT_ID,
        proposal_id=proposal_payload["proposal_id"],
        verify_current=True,
    )
    assert publication.proposal.proposal_digest == proposal_payload["proposal_digest"]

    started = _confirm_and_turn(
        client,
        endpoint,
        response=response,
        message_id="minimal-golden-path-confirm",
    )
    assert started.status_code == 200, started.get_json()
    state = service.read_session(
        project_id=_PROJECT_ID,
        conversation_id=_CONVERSATION_ID,
    )
    assert state["proposal_digest"] == proposal_payload["proposal_digest"]
    assert state["authorization_id"]
    assert state["start_intent_id"]
    assert state["controller_execution_id"]

    authorization = service.authorization_service.verify_authorization(
        project_id=_PROJECT_ID,
        authorization_id=state["authorization_id"],
        # The immutable authorization is the source of truth after the
        # deterministic task has published its output; current observation
        # naturally includes that new artifact.
        verify_current=False,
    )
    start_intent = service.authorization_service.verify_start_intent(
        project_id=_PROJECT_ID,
        start_intent_id=state["start_intent_id"],
        verify_current=False,
    )
    control_store = service.controller.control_store
    permission_ids = {
        authorization.permission_decision_id,
        start_intent.permission_decision_id,
    }
    assert len(permission_ids) == 2
    outcomes = set()
    for decision_id in permission_ids:
        decision = control_store.read_permission_decision(
            project_id=_PROJECT_ID,
            decision_id=decision_id,
        )
        outcomes.add(decision.outcome.value)
        assert decision.decision_digest.startswith("sha256:")
    assert outcomes == {"ALLOW", "REQUIRE_APPROVAL"}

    execution = control_store.read_harness_controller_execution(
        project_id=_PROJECT_ID,
        controller_execution_id=state["controller_execution_id"],
    )
    dispatches = control_store.list_harness_local_dispatch_receipts(
        project_id=_PROJECT_ID,
        controller_execution_id=execution.controller_execution_id,
    )
    assert len(dispatches) == 1
    assert dispatches[0].controller_execution_id == execution.controller_execution_id
    assert dispatches[0].task_id == "inspect_dataset"

    replay = service.controller.create(
        project_id=_PROJECT_ID,
        start_intent_id=start_intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=start_intent.start_intent_digest,
            client_request_id=execution.client_request_id,
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        conversation_id=_CONVERSATION_ID,
    )
    assert replay.receipt is not None
    first_receipt = control_store.list_harness_controller_action_receipts(
        project_id=_PROJECT_ID,
        controller_execution_id=execution.controller_execution_id,
    )
    assert len(first_receipt) == 1
    assert replay.receipt.receipt_id == first_receipt[0].receipt_id
    assert len(
        control_store.list_harness_local_dispatch_receipts(
            project_id=_PROJECT_ID,
            controller_execution_id=execution.controller_execution_id,
        )
    ) == 1

    with pytest.raises(ScientificAgentHarnessControllerError):
        service.controller.create(
            project_id=_PROJECT_ID,
            start_intent_id=start_intent.start_intent_id,
            request=AgentHarnessControllerStartRequest(
                expected_start_intent_digest=_agent_digest({"tampered": True}),
                client_request_id=execution.client_request_id + "-tampered",
            ),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )


@pytest.mark.pr_fast
def test_minimal_harness_l2_authority_boundary_preserves_chain_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, client, service, _storage, endpoint = _create_conversation(
        tmp_path / "subset",
        content="Generate and screen a bounded deterministic PLQY candidate set.",
    )
    baseline_response = _plan_response(
        "generate_candidates",
        options={"backend": "deterministic_stub", "count": 8, "seed": 1},
    )
    proposed = client.post(
        endpoint + "/turn",
        json={"run_id": _RUN_ID, "llm_provider": _provider(baseline_response)},
    )
    assert proposed.status_code == 200, proposed.get_json()
    assert (
        proposed.get_json()["proposal"]["compiled_task_options"]["generate_candidates"]["count"]
        == 8
    )
    started = _confirm_and_turn(
        client,
        endpoint,
        response=baseline_response,
        message_id="minimal-l2-baseline-confirm",
    )
    assert started.status_code == 200, started.get_json()
    baseline_state = service.read_session(
        project_id=_PROJECT_ID,
        conversation_id=_CONVERSATION_ID,
    )
    failed = _failed_controller_result(service)
    monkeypatch.setattr(
        service.controller,
        "read_execution_agent_snapshot",
        lambda **_kwargs: failed,
    )

    subset_request = {
        "run_id": _RUN_ID,
        "external_llm_approved": True,
        "llm_provider": _provider(
            {
                "rationale_summary": "Use a smaller bounded candidate set.",
                "option_patch": {"generate_candidates": {"count": 4}},
            }
        ),
    }
    subset = client.post(endpoint + "/replan", json=subset_request)
    assert subset.status_code == 200, subset.get_json()
    subset_body = subset.get_json()
    decision = subset_body["decision"]
    assert decision["authority_relation"] == "SUBSET"
    assert decision["semantic_boundary"] == "NONE"
    assert decision["authority_auto_apply"] is True
    assert decision["fresh_permission_required"] is False
    assert decision["fresh_authorization_required"] is False
    assert subset_body["session"]["reason_code"] != "PLAN_APPROVAL_REQUIRED"
    assert subset_body["session"]["authorization_id"]
    assert subset_body["session"]["start_intent_id"]
    assert subset_body["session"]["controller_execution_id"]
    assert subset_body["session"]["controller_execution_id"] != baseline_state[
        "controller_execution_id"
    ]

    receipt_id = (
        "revision-application-"
        + _agent_digest(
            {"project_id": _PROJECT_ID, "revision_id": decision["revision_id"]}
        ).split(":", 1)[1][:32]
    )
    receipt = service.replanner.read_application(
        project_id=_PROJECT_ID,
        receipt_id=receipt_id,
    )
    assert receipt.schema_version == "agent_plan_revision_application_receipt.v2"
    assert receipt.authority_decision_id == decision["decision_id"]
    assert receipt.authority_decision_digest == decision["decision_digest"]
    assert receipt.authority_evaluation_id == decision["authority_evaluation_id"]
    assert receipt.authority_evaluation_digest == decision["authority_evaluation_digest"]
    assert receipt.baseline_authorization_id == baseline_state["authorization_id"]
    assert receipt.baseline_authorization_digest == baseline_state["authorization_digest"]
    assert receipt.fresh_permission_required is False
    assert receipt.fresh_authorization_required is False
    successor = service.proposal_store.read(
        project_id=_PROJECT_ID,
        proposal_id=receipt.successor_proposal_id,
        verify_current=False,
    ).proposal
    assert successor.compiled_task_options["generate_candidates"]["count"] == 4

    successor_authorization = service.authorization_service.verify_authorization(
        project_id=_PROJECT_ID,
        authorization_id=subset_body["session"]["authorization_id"],
        verify_current=True,
    )
    successor_start = service.authorization_service.verify_start_intent(
        project_id=_PROJECT_ID,
        start_intent_id=subset_body["session"]["start_intent_id"],
        verify_current=True,
    )
    successor_execution = service.controller.control_store.read_harness_controller_execution(
        project_id=_PROJECT_ID,
        controller_execution_id=subset_body["session"]["controller_execution_id"],
    )
    assert successor_authorization.proposal_id == receipt.successor_proposal_id
    assert successor_start.proposal_id == receipt.successor_proposal_id
    assert successor_execution.proposal_id == receipt.successor_proposal_id
    assert successor_execution.authorization_id == successor_authorization.authorization_id
    assert successor_execution.start_intent_id == successor_start.start_intent_id
    assert (
        service.controller.control_store.list_harness_local_dispatch_receipts(
            project_id=_PROJECT_ID,
            controller_execution_id=successor_execution.controller_execution_id,
        )
        == []
    )

    replay = client.post(endpoint + "/replan", json=subset_request)
    assert replay.status_code == 200, replay.get_json()
    replay_body = replay.get_json()
    assert replay_body["decision"]["decision_id"] == decision["decision_id"]
    assert replay_body["session"]["authorization_id"] == subset_body["session"]["authorization_id"]
    assert replay_body["session"]["start_intent_id"] == subset_body["session"]["start_intent_id"]
    assert replay_body["session"]["controller_execution_id"] == subset_body["session"]["controller_execution_id"]
    assert (
        len(
            service.controller.control_store.list_harness_controller_action_receipts(
                project_id=_PROJECT_ID,
                controller_execution_id=successor_execution.controller_execution_id,
            )
        )
        == 1
    )

    revision = service.replanner.read_revision(
        project_id=_PROJECT_ID,
        revision_id=decision["revision_id"],
    )
    baseline_proposal = service.proposal_store.read(
        project_id=_PROJECT_ID,
        proposal_id=baseline_state["proposal_id"],
        verify_current=False,
    ).proposal
    baseline_authorization = service.authorization_service.verify_authorization(
        project_id=_PROJECT_ID,
        authorization_id=baseline_state["authorization_id"],
        verify_current=False,
    )
    verified_decision = classify_plan_revision_materiality(
        revision,
        baseline_proposal=baseline_proposal,
        baseline_authorization=baseline_authorization,
        registry=service.proposal_store.registry,
    )
    forged_decision = verified_decision.model_copy(
        update={"authority_evaluation_digest": _agent_digest({"tampered": True})}
    )
    with pytest.raises(ScientificAgentReplannerConflict):
        service.replanner.apply_revision(
            project_id=_PROJECT_ID,
            revision_id=revision.revision_id,
            request=AgentPlanRevisionApplicationRequest(
                expected_revision_digest=revision.revision_digest,
                client_request_id=receipt.client_request_id,
            ),
            authority_decision=forged_decision,
        )

    _boundary_app, boundary_client, boundary_service, _boundary_storage, boundary_endpoint = (
        _create_conversation(
            tmp_path / "boundary",
            content=(
                "Generate PLQY candidates, then confirm the scientific stopping "
                "boundary."
            ),
        )
    )
    boundary_proposed = boundary_client.post(
        boundary_endpoint + "/turn",
        json={
            "run_id": _RUN_ID,
            "llm_provider": _provider(baseline_response),
        },
    )
    assert boundary_proposed.status_code == 200, boundary_proposed.get_json()
    boundary_started = _confirm_and_turn(
        boundary_client,
        boundary_endpoint,
        response=baseline_response,
        message_id="minimal-l2-boundary-confirm",
    )
    assert boundary_started.status_code == 200, boundary_started.get_json()
    boundary_state = boundary_service.read_session(
        project_id=_PROJECT_ID,
        conversation_id=_CONVERSATION_ID,
    )
    boundary_failed = _failed_controller_result(boundary_service)
    monkeypatch.setattr(
        boundary_service.controller,
        "read_execution_agent_snapshot",
        lambda **_kwargs: boundary_failed,
    )
    boundary = boundary_client.post(
        boundary_endpoint + "/replan",
        json={
            "run_id": _RUN_ID,
            "external_llm_approved": True,
            "llm_provider": _provider(
                {
                    "rationale_summary": "Pause at the scientific confirmation boundary.",
                    "stop_conditions": ["pause after the bounded candidate run"],
                }
            ),
        },
    )
    assert boundary.status_code == 200, boundary.get_json()
    boundary_body = boundary.get_json()
    boundary_decision = boundary_body["decision"]
    assert boundary_decision["authority_relation"] == "SUBSET"
    assert boundary_decision["semantic_boundary"] != "NONE"
    assert boundary_decision["authority_auto_apply"] is False
    assert boundary_decision["fresh_permission_required"] is True
    assert boundary_decision["fresh_authorization_required"] is True
    assert boundary_body["session"]["status"] == "approval_required"
    assert boundary_body["session"]["authorization_id"] == ""
    assert boundary_body["session"]["start_intent_id"] == ""
    assert boundary_body["session"]["controller_execution_id"] == ""
    assert boundary_state["controller_execution_id"]
    assert (
        boundary_service.controller.control_store.list_harness_local_dispatch_receipts(
            project_id=_PROJECT_ID,
            controller_execution_id=boundary_state["controller_execution_id"],
        )
        == []
    )

    boundary_receipt_id = (
        "revision-application-"
        + _agent_digest(
            {
                "project_id": _PROJECT_ID,
                "revision_id": boundary_decision["revision_id"],
            }
        ).split(":", 1)[1][:32]
    )
    boundary_receipt = boundary_service.replanner.read_application(
        project_id=_PROJECT_ID,
        receipt_id=boundary_receipt_id,
    )
    assert boundary_receipt.schema_version == "agent_plan_revision_application_receipt.v2"
    assert boundary_receipt.fresh_permission_required is True
    assert boundary_receipt.fresh_authorization_required is True
