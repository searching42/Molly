from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from flask import Flask

from ai4s_agent.agent_run_inspection import (
    AgentRunInspectionReadError,
    AgentRunInspectionService,
)
from ai4s_agent.execution_agent_store import ExecutionAgentStore
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.routes.agent_run_inspection import register_agent_run_inspection_routes
from ai4s_agent.schemas import (
    AgentAuthorizationMode,
    AgentExecutionPlanLLMResponse,
    AgentHarnessControllerStartRequest,
    AgentHarnessControllerExecution,
    AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V1,
    AgentPlanAuthorizationRequest,
    AgentPlanFeedbackRequest,
    AgentPlanRevisionApplicationReceipt,
    AgentPlanRevisionApplicationRequest,
    AgentReplanLLMResponse,
    AgentRunInspection,
    AgentRunInspectionStatus,
    CORE_SCHEMA_MODELS,
    _agent_digest,
)
from ai4s_agent.scientific_agent_authorization import (
    AgentPlanControlStore,
    ScientificAgentAuthorizationService,
)
from ai4s_agent.scientific_agent_harness_controller import ScientificAgentHarnessController
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
)
from ai4s_agent.scientific_agent_replanner import ScientificAgentReplannerService
from ai4s_agent.storage import ProjectStorage


_NOW = "2026-08-02T00:00:00Z"


class _NoRemoteAuthorities:
    def current_authority(self, **_: object):  # pragma: no cover
        raise AssertionError("local inspection consulted remote authority")


class _NoRemoteLifecycle:
    pass


def _chain(tmp_path: Path):
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at=_NOW)
    run_dir = storage.run_dir("project-1", "run-1")
    dataset = run_dir / "inputs" / "dataset.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("SMILES,host_material\nCCO,host-dopant\n", encoding="utf-8")
    storage.register_artifact_path(
        "project-1", "run-1", "uploaded_dataset", "inputs/dataset.csv"
    )
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["inspect_dataset"],
        selected_input_artifact_ids=["uploaded_dataset"],
        task_options={"inspect_dataset": {}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce a dataset profile"],
        rationales=["Inspect the exact registered input."],
        assumptions=[],
        questions=[],
    )
    builder = AgentProjectObservationBuilder(storage=storage, clock=lambda: _NOW)
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage, observation_builder=builder
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        observation_builder=builder,
        proposal_store=proposal_store,
        clock=lambda: _NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Inspect one OLED host material dataset",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="proposal-request-1",
    )
    control_store = AgentPlanControlStore(storage=storage)
    authorizations = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        control_store=control_store,
        clock=lambda: _NOW,
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorizations,
        control_store=control_store,
        resource_authority_service=_NoRemoteAuthorities(),
        executor=RunPlanExecutor(storage=storage, registry=proposal_store.registry),
        remote_executions=_NoRemoteLifecycle(),
        clock=lambda: _NOW,
    )
    execution_store = ExecutionAgentStore(storage=storage)
    service = AgentRunInspectionService(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorizations,
        control_store=control_store,
        controller=controller,
        execution_agent_store=execution_store,
        clock=lambda: _NOW,
    )
    return storage, proposal_store, authorizations, control_store, controller, service, proposal


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.endswith(".lock")
    }


def _assert_frozen_v1_semantics(inspection: AgentRunInspection) -> None:
    legacy_payload = inspection.model_dump(mode="json")
    assert "structured_dataset_canary" not in legacy_payload
    legacy_payload.pop("inspection_id")
    legacy_payload.pop("inspection_digest")
    legacy_payload.pop("created_at")
    assert inspection.semantic_material() == legacy_payload
    assert inspection.inspection_digest == _agent_digest(legacy_payload)


def _service_from_controller(storage, controller) -> AgentRunInspectionService:
    return AgentRunInspectionService(
        storage=storage,
        proposal_store=controller.proposal_store,
        authorization_service=controller.authorization_service,
        control_store=controller.control_store,
        controller=controller,
        execution_agent_store=ExecutionAgentStore(storage=storage),
        clock=lambda: _NOW,
    )


def _applied_replan(tmp_path: Path):
    from tests import test_scientific_agent_replanner as replanner_tests

    (
        storage,
        proposal_store,
        authorization_service,
        baseline,
        old_authorization,
        replanner,
    ) = replanner_tests._baseline(tmp_path)
    feedback = replanner.create_feedback(
        project_id="project-1",
        request=AgentPlanFeedbackRequest(
            run_id="run-1",
            client_request_id="feedback-exact-inspection-1",
            feedback="Reduce the bounded candidate count.",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    revision = replanner.create_revision(
        project_id="project-1",
        payload=replanner_tests._revision_payload(
            baseline,
            old_authorization,
            feedback,
            request_id="revision-exact-inspection-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        provider=replanner_tests.CountingProvider(
            response=AgentReplanLLMResponse(
                rationale_summary="Use a smaller bounded candidate set.",
                option_patch={"generate_candidates": {"count": 4}},
            ).model_dump(mode="json")
        ),
    ).proposal
    application = replanner.apply_revision(
        project_id="project-1",
        revision_id=revision.revision_id,
        request=AgentPlanRevisionApplicationRequest(
            expected_revision_digest=revision.revision_digest,
            client_request_id="revision-exact-application-1",
        ),
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorization_service,
        control_store=authorization_service.control_store,
        resource_authority_service=_NoRemoteAuthorities(),
        executor=RunPlanExecutor(storage=storage, registry=proposal_store.registry),
        remote_executions=_NoRemoteLifecycle(),
        clock=lambda: _NOW,
    )
    return (
        storage,
        baseline,
        revision,
        application,
        _service_from_controller(storage, controller),
    )


def test_proposal_only_inspection_is_current_deterministic_and_read_only(tmp_path: Path) -> None:
    storage, _, _, _, _, service, proposal = _chain(tmp_path)
    project = storage.projects_root / "project-1"
    before = _snapshot(project)

    first = service.inspect(project_id="project-1", run_id="run-1")
    second = service.inspect(project_id="project-1", run_id="run-1")

    assert first == second
    assert first.inspection_digest == second.inspection_digest
    assert first.inspection_status == AgentRunInspectionStatus.CURRENT
    assert first.verifier_supported_run_outcome == "plan_proposed"
    assert first.plan.proposal.object_id == proposal.proposal_id
    assert first.controller is None
    assert first.authoritative is False and first.read_only is True
    assert first.scientific_success == "not_asserted"
    _assert_frozen_v1_semantics(first)
    assert _snapshot(project) == before
    assert [
        (item.source_name, item.source_kind, item.source_id)
        for item in first.source_roster
    ] == sorted(
        (item.source_name, item.source_kind, item.source_id)
        for item in first.source_roster
    )


def test_self_signed_structured_dataset_directory_is_not_inspection_authority(
    tmp_path: Path,
) -> None:
    storage, _, _, _, _, service, _ = _chain(tmp_path)
    before = service.inspect(project_id="project-1", run_id="run-1")
    forged = storage.run_dir("project-1", "run-1") / "structured_dataset_canary"
    forged.mkdir()
    (forged / "evidence.json").write_text(
        '{"schema_version":"structured_dataset_canary_evidence.v1",'
        '"evidence_digest":"sha256:' + "a" * 64 + '"}',
        encoding="utf-8",
    )

    after = service.inspect(project_id="project-1", run_id="run-1")

    assert after.inspection_digest == before.inspection_digest
    assert "structured_dataset_canary" not in after.model_dump(mode="json")
    assert not any(
        item.source_kind == "structured_dataset_canary_publication"
        for item in after.source_roster
    )


def test_permission_authorization_start_and_terminal_controller_projection(tmp_path: Path) -> None:
    _, _, authorizations, _, controller, service, proposal = _chain(tmp_path)
    permission_only = authorizations.evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    inspected = service.inspect(project_id="project-1", run_id="run-1")
    assert inspected.plan.permission_decision.object_id == permission_only.decision_id
    assert inspected.verifier_supported_run_outcome == "permission_decided"

    approved = authorizations.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="authorization-request-1",
        ),
        actor="private-user-name",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    before_start = service.inspect(project_id="project-1", run_id="run-1")
    assert before_start.verifier_supported_run_outcome == "start_requested"
    assert before_start.plan.authorization is not None
    assert before_start.plan.start_intent is not None
    assert "private-user-name" not in json.dumps(before_start.model_dump(mode="json"))

    controller.create(
        project_id="project-1",
        start_intent_id=approved.start_intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=approved.start_intent.start_intent_digest,
            client_request_id="controller-create-1",
        ),
        actor="private-user-name",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    terminal = service.inspect(project_id="project-1", run_id="run-1")
    assert terminal.controller is not None
    assert terminal.verifier_supported_run_outcome == "succeeded"
    assert terminal.controller.status == "succeeded"
    assert terminal.tasks[0].verifier_supported_outcome == "succeeded"
    assert terminal.artifacts
    assert all(item.registry_binding is not None for item in terminal.artifacts)
    controller.executor.execute = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("inspection invoked Executor")
    )
    controller.advance = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("inspection advanced Controller")
    )
    assert service.inspect(
        project_id="project-1", run_id="run-1"
    ).inspection_digest == terminal.inspection_digest


def test_strict_api_rejects_authority_injection_and_returns_no_private_material(tmp_path: Path) -> None:
    storage, _, _, _, _, service, _ = _chain(tmp_path)
    project = storage.projects_root / "project-1"
    before = _snapshot(project)
    app = Flask(__name__)
    register_agent_run_inspection_routes(app, service=service)
    client = app.test_client()

    assert client.get(
        "/api/projects/project-1/agent-runs/run-1/inspection?status=succeeded"
    ).status_code == 400
    assert client.get(
        "/api/projects/project-1/agent-runs/run-1/inspection",
        json={"verifier_outcome": "succeeded"},
    ).status_code == 400
    response = client.get(
        "/api/projects/project-1/agent-runs/run-1/inspection",
        headers={
            "X-Run-Status": "succeeded",
            "X-Trace-Metadata": "ssh://private-host/path?token=secret",
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    AgentRunInspection.model_validate(payload)
    rendered = json.dumps(payload, sort_keys=True)
    for forbidden in (
        str(storage.workspace_dir),
        "private-host",
        "ssh://",
        "token=secret",
        "stdout",
        "stderr",
        "verifier_outcome",
    ):
        assert forbidden not in rendered
    assert _snapshot(project) == before


def test_corrupt_missing_and_symlink_sources_fail_closed(tmp_path: Path) -> None:
    storage, _, _, _, _, service, proposal = _chain(tmp_path)
    proposal_dir = (
        storage.projects_root / "project-1" / "agent_plan_proposals" / proposal.proposal_id
    )
    proposal_json = proposal_dir / "proposal.json"
    original = proposal_json.read_bytes()
    proposal_json.write_bytes(original[:32])
    with pytest.raises(AgentRunInspectionReadError) as damaged:
        service.inspect(project_id="project-1", run_id="run-1")
    assert damaged.value.inspection_status == AgentRunInspectionStatus.DAMAGED_SOURCE
    assert damaged.value.reason_code == "RUN_INSPECTION_SOURCE_DAMAGED"

    proposal_json.write_bytes(original)
    manifest = proposal_dir / "publication_manifest.json"
    manifest.unlink()
    with pytest.raises(AgentRunInspectionReadError) as missing:
        service.inspect(project_id="project-1", run_id="run-1")
    assert missing.value.inspection_status == AgentRunInspectionStatus.MISSING_SOURCE

    manifest.write_text("{}", encoding="utf-8")
    proposal_json.unlink()
    proposal_json.symlink_to(manifest)
    with pytest.raises(AgentRunInspectionReadError) as unsafe:
        service.inspect(project_id="project-1", run_id="run-1")
    assert unsafe.value.inspection_status in {
        AgentRunInspectionStatus.DAMAGED_SOURCE,
        AgentRunInspectionStatus.REPLACED_SOURCE,
    }


def test_generated_inspection_schema_equals_pydantic_source() -> None:
    schema = json.loads(
        (Path("docs/schemas") / "agent_run_inspection.schema.json").read_text(encoding="utf-8")
    )
    assert CORE_SCHEMA_MODELS["agent_run_inspection"] is AgentRunInspection
    assert schema == AgentRunInspection.model_json_schema()
    assert schema["additionalProperties"] is False


def test_gate_waiting_and_remote_waiting_are_verified_without_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests import test_scientific_agent_harness_controller as controller_tests

    gated_storage, gated_controller, gated_intent = controller_tests._gated_local_authority_chain(
        tmp_path / "gated"
    )
    gated = gated_controller.create(
        project_id="project-1",
        start_intent_id=gated_intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=gated_intent.start_intent_digest,
            client_request_id="gated-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert gated.inspection.status.value == "waiting_gate"
    gate_view = _service_from_controller(gated_storage, gated_controller).inspect(
        project_id="project-1", run_id="run-1"
    )
    assert gate_view.verifier_supported_run_outcome == "waiting_user"
    assert gate_view.tasks[0].gate_requirements == ["gate_1_task_parse"]

    remote_storage, remote_controller, remote_intent, transport, _ = (
        controller_tests._remote_controller_authority_chain(
            tmp_path / "remote", monkeypatch
        )
    )
    prepared = remote_controller.create(
        project_id="project-1",
        start_intent_id=remote_intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=remote_intent.start_intent_digest,
            client_request_id="remote-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert prepared.inspection.status.value == "waiting_remote_approval"
    dispatches = transport.dispatches
    remote_view = _service_from_controller(remote_storage, remote_controller).inspect(
        project_id="project-1", run_id="run-1"
    )
    assert remote_view.controller.execution_route == "remote_execution_service"
    assert remote_view.verifier_supported_run_outcome == "waiting_user"
    assert transport.dispatches == dispatches == 0


def test_recovery_required_is_projected_and_api_uses_conflict_status(tmp_path: Path) -> None:
    _, _, authorizations, _, controller, _, proposal = _chain(tmp_path)
    approved = authorizations.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="authorization-request-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    original = controller._publish_local_execution_publication

    def crash_before_publication(**_: object):
        raise RuntimeError("injected test crash")

    controller._publish_local_execution_publication = crash_before_publication
    try:
        with pytest.raises(RuntimeError, match="injected test crash"):
            controller.create(
                project_id="project-1",
                start_intent_id=approved.start_intent.start_intent_id,
                request=AgentHarnessControllerStartRequest(
                    expected_start_intent_digest=approved.start_intent.start_intent_digest,
                    client_request_id="recovery-create-1",
                ),
                actor="alice",
                actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
            )
    finally:
        controller._publish_local_execution_publication = original
    service = _service_from_controller(controller.storage, controller)
    recovered = service.inspect(project_id="project-1", run_id="run-1")
    assert recovered.inspection_status == AgentRunInspectionStatus.RECOVERY_REQUIRED
    assert recovered.verifier_supported_run_outcome == "recovery_required"
    assert recovered.tasks[0].recovery_required is True
    app = Flask(__name__)
    register_agent_run_inspection_routes(app, service=service)
    response = app.test_client().get(
        "/api/projects/project-1/agent-runs/run-1/inspection"
    )
    assert response.status_code == 409
    AgentRunInspection.model_validate(response.get_json())


def test_stale_and_competing_plan_sources_have_fixed_fail_closed_codes(tmp_path: Path) -> None:
    storage, proposal_store, _, _, _, service, proposal = _chain(tmp_path)
    dataset = storage.run_dir("project-1", "run-1") / "inputs" / "dataset.csv"
    dataset.write_text("SMILES,value\nCCC,2.0\n", encoding="utf-8")
    with pytest.raises(AgentRunInspectionReadError) as stale:
        service.inspect(project_id="project-1", run_id="run-1")
    assert stale.value.reason_code == "RUN_INSPECTION_SOURCE_STALE"
    assert stale.value.inspection_status == AgentRunInspectionStatus.STALE_SOURCE

    dataset.write_text("SMILES,host_material\nCCO,host-dopant\n", encoding="utf-8")
    first_publication = proposal_store.read(
        project_id="project-1", proposal_id=proposal.proposal_id
    )
    ScientificAgentPlanService(
        storage=storage,
        observation_builder=proposal_store.observation_builder,
        proposal_store=proposal_store,
        clock=lambda: _NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Inspect the same exact dataset through a competing request",
        user_constraints=[],
        provider=StubLLMProvider(
            response=first_publication.proposal.validated_llm_response.model_dump(mode="json")
        ),
        client_request_id="proposal-request-2",
    )
    with pytest.raises(AgentRunInspectionReadError) as competing:
        service.inspect(project_id="project-1", run_id="run-1")
    assert competing.value.reason_code == "RUN_INSPECTION_COMPETING_CURRENT_SOURCE"
    assert competing.value.inspection_status == AgentRunInspectionStatus.REPLACED_SOURCE


def test_concurrent_and_fresh_process_reads_keep_one_digest(tmp_path: Path) -> None:
    storage, _, _, _, _, service, _ = _chain(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        digests = list(
            pool.map(
                lambda _: service.inspect(
                    project_id="project-1", run_id="run-1"
                ).inspection_digest,
                range(16),
            )
        )
    assert len(set(digests)) == 1

    script = """
from pathlib import Path
import sys
from ai4s_agent.agent_run_inspection import AgentRunInspectionService
from ai4s_agent.execution_agent_store import ExecutionAgentStore
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.scientific_agent_authorization import AgentPlanControlStore, ScientificAgentAuthorizationService
from ai4s_agent.scientific_agent_harness_controller import ScientificAgentHarnessController
from ai4s_agent.scientific_agent_plan import AgentProjectObservationBuilder, ScientificAgentPlanProposalStore
from ai4s_agent.storage import ProjectStorage
class NoRemote:
    def current_authority(self, **kwargs): raise AssertionError
class NoLifecycle: pass
storage = ProjectStorage(workspace_dir=Path(sys.argv[1]))
builder = AgentProjectObservationBuilder(storage=storage, clock=lambda: '2026-08-02T00:00:00Z')
proposal_store = ScientificAgentPlanProposalStore(storage=storage, observation_builder=builder)
control = AgentPlanControlStore(storage=storage)
authorization = ScientificAgentAuthorizationService(storage=storage, proposal_store=proposal_store, control_store=control, clock=lambda: '2026-08-02T00:00:00Z')
controller = ScientificAgentHarnessController(storage=storage, proposal_store=proposal_store, authorization_service=authorization, control_store=control, resource_authority_service=NoRemote(), executor=RunPlanExecutor(storage=storage, registry=proposal_store.registry), remote_executions=NoLifecycle(), clock=lambda: '2026-08-02T00:00:00Z')
service = AgentRunInspectionService(storage=storage, proposal_store=proposal_store, authorization_service=authorization, control_store=control, controller=controller, execution_agent_store=ExecutionAgentStore(storage=storage), clock=lambda: '2026-08-02T00:00:00Z')
print(service.inspect(project_id='project-1', run_id='run-1').inspection_digest)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path.cwd() / "src"), str(Path.cwd())]
    )
    process_digests = []
    for seed in ("1", "991"):
        environment["PYTHONHASHSEED"] = seed
        process_digests.append(
            subprocess.check_output(
                [sys.executable, "-c", script, str(storage.workspace_dir)],
                cwd=Path.cwd(),
                env=environment,
                text=True,
            ).strip()
        )
    assert process_digests == [digests[0], digests[0]]


def test_execution_agent_proposal_and_application_are_projected_as_advisory_history(
    tmp_path: Path,
) -> None:
    from tests.execution_agent_test_support import (
        execution_agent_service,
        local_controller_execution,
    )
    from tests.test_execution_agent import _proposal_request, _provider
    from ai4s_agent.schemas import AgentToolCallApplicationRequest

    storage, _, controller, initial = local_controller_execution(tmp_path)
    execution_agent = execution_agent_service(
        storage=storage, controller=controller
    )
    created = execution_agent.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=_proposal_request(initial.execution.execution_digest),
        provider=_provider("controller.advance_current.v1"),
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    service = _service_from_controller(storage, controller)
    proposed = service.inspect(project_id="project-1", run_id="run-1")
    proposal = created.publication.proposal
    assert proposed.tool_calls[0].proposal.object_id == proposal.tool_call_proposal_id
    assert proposed.tool_calls[0].status == "review_only"
    assert proposed.tool_calls[0].application_receipt is None

    applied = execution_agent.apply_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
        request=AgentToolCallApplicationRequest(
            expected_tool_call_proposal_digest=proposal.tool_call_proposal_digest,
            client_request_id="apply-tool-proposal-1",
        ),
    )
    projected = service.inspect(project_id="project-1", run_id="run-1")
    assert projected.tool_calls[0].status == "applied"
    assert (
        projected.tool_calls[0].application_receipt.object_id
        == applied.application_receipt.application_receipt_id
    )
    tool_sources = [
        item for item in projected.source_roster
        if item.source_kind in {"execution_agent_proposal", "execution_agent_receipt"}
    ]
    assert tool_sources and {item.currentness for item in tool_sources} == {"historical"}


def test_replanner_review_and_applied_successor_require_fresh_authority_in_projection(
    tmp_path: Path,
) -> None:
    from tests import test_scientific_agent_replanner as replanner_tests

    (
        storage,
        proposal_store,
        authorization_service,
        baseline,
        old_authorization,
        replanner,
    ) = replanner_tests._baseline(tmp_path)
    feedback = replanner.create_feedback(
        project_id="project-1",
        request=AgentPlanFeedbackRequest(
            run_id="run-1",
            client_request_id="feedback-inspection-1",
            feedback="Reduce the bounded candidate count.",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    created = replanner.create_revision(
        project_id="project-1",
        payload=replanner_tests._revision_payload(
            baseline, old_authorization, feedback, request_id="revision-inspection-1"
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        provider=replanner_tests.CountingProvider(
            response=AgentReplanLLMResponse(
                rationale_summary="Use a smaller bounded candidate set.",
                option_patch={"generate_candidates": {"count": 4}},
            ).model_dump(mode="json")
        ),
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorization_service,
        control_store=authorization_service.control_store,
        resource_authority_service=_NoRemoteAuthorities(),
        executor=RunPlanExecutor(storage=storage, registry=proposal_store.registry),
        remote_executions=_NoRemoteLifecycle(),
        clock=lambda: _NOW,
    )
    service = _service_from_controller(storage, controller)
    review = service.inspect(project_id="project-1", run_id="run-1")
    assert review.replanner[0].status == "review_required"
    assert review.replanner[0].application_receipt is None
    assert review.replanner[0].feedback_receipt is not None

    application = replanner.apply_revision(
        project_id="project-1",
        revision_id=created.proposal.revision_id,
        request=AgentPlanRevisionApplicationRequest(
            expected_revision_digest=created.proposal.revision_digest,
            client_request_id="revision-application-inspection-1",
        ),
    )
    applied = service.inspect(project_id="project-1", run_id="run-1")
    assert applied.plan.proposal.object_id == application.successor.proposal_id
    assert applied.plan.authorization is None
    assert applied.plan.start_intent is None
    assert applied.verifier_supported_run_outcome == "plan_proposed"
    assert applied.replanner[0].status == "applied"
    assert applied.replanner[0].fresh_permission_required is True
    assert applied.replanner[0].fresh_authorization_required is True


def test_successor_current_controller_isolated_from_terminal_historical_execution(
    tmp_path: Path,
) -> None:
    from tests.execution_agent_test_support import local_controller_execution

    storage, control_store, controller, old_result = local_controller_execution(tmp_path)
    baseline = controller.proposal_store.read_immutable_publication(
        project_id="project-1",
        proposal_id=old_result.execution.proposal_id,
    ).proposal
    old_authorization = control_store.read_authorization(
        project_id="project-1",
        authorization_id=old_result.execution.authorization_id,
    )
    old_receipt = old_result.receipt
    assert old_receipt is not None
    old_decision = control_store.read_harness_controller_decision(
        project_id="project-1",
        decision_id=old_receipt.decision_id,
    )
    replanner = ScientificAgentReplannerService(
        storage=storage,
        proposal_store=controller.proposal_store,
        observation_builder=controller.proposal_store.observation_builder,
        authorization_service=controller.authorization_service,
        control_store=control_store,
        controller=controller,
        clock=lambda: _NOW,
    )
    revision = replanner.create_revision(
        project_id="project-1",
        payload={
            "run_id": "run-1",
            "client_request_id": "terminal-successor-revision-1",
            "trigger_kind": "controller_terminal",
            "baseline_proposal_id": baseline.proposal_id,
            "baseline_proposal_digest": baseline.proposal_digest,
            "baseline_semantic_plan_id": baseline.semantic_plan_id,
            "baseline_semantic_plan_digest": baseline.semantic_plan_digest,
            "baseline_run_plan_digest": _agent_digest(
                baseline.run_plan.model_dump(mode="json")
            ),
            "baseline_authorization_id": old_authorization.authorization_id,
            "baseline_authorization_digest": old_authorization.authorization_digest,
            "controller_execution_id": old_result.execution.controller_execution_id,
            "controller_execution_digest": old_result.execution.execution_digest,
            "controller_decision_id": old_decision.decision_id,
            "controller_decision_digest": old_decision.decision_digest,
            "controller_receipt_id": old_receipt.receipt_id,
            "controller_receipt_digest": old_receipt.receipt_digest,
            "external_llm_approved": True,
        },
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        provider=StubLLMProvider(
            response=AgentReplanLLMResponse(
                success_criteria=["produce an exact privacy-safe dataset profile"]
            ).model_dump(mode="json")
        ),
    ).proposal
    applied = replanner.apply_revision(
        project_id="project-1",
        revision_id=revision.revision_id,
        request=AgentPlanRevisionApplicationRequest(
            expected_revision_digest=revision.revision_digest,
            client_request_id="terminal-successor-application-1",
        ),
    )
    inspection_service = _service_from_controller(storage, controller)

    before_fresh_authority = inspection_service.inspect(
        project_id="project-1", run_id="run-1"
    )
    assert before_fresh_authority.plan.proposal.object_id == applied.successor.proposal_id
    assert before_fresh_authority.controller is None
    assert before_fresh_authority.verifier_supported_run_outcome == "plan_proposed"
    assert all(
        task.verifier_supported_outcome == "not_available"
        for task in before_fresh_authority.tasks
    )
    historical_execution_sources = [
        item
        for item in before_fresh_authority.source_roster
        if item.source_kind == "controller_execution"
    ]
    assert [item.currentness for item in historical_execution_sources] == [
        "historical"
    ]
    _assert_frozen_v1_semantics(before_fresh_authority)

    fresh = controller.authorization_service.approve_and_start(
        project_id="project-1",
        proposal_id=applied.successor.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=applied.successor.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="terminal-successor-authorization-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    current_result = controller.create(
        project_id="project-1",
        start_intent_id=fresh.start_intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=fresh.start_intent.start_intent_digest,
            client_request_id="terminal-successor-controller-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    after_fresh_authority = inspection_service.inspect(
        project_id="project-1", run_id="run-1"
    )
    assert after_fresh_authority.controller is not None
    assert (
        after_fresh_authority.controller.execution.object_id
        == current_result.execution.controller_execution_id
    )
    assert after_fresh_authority.verifier_supported_run_outcome == "succeeded"
    execution_sources = [
        item
        for item in after_fresh_authority.source_roster
        if item.source_kind == "controller_execution"
    ]
    assert {item.currentness for item in execution_sources} == {
        "current",
        "historical",
    }


def test_policy_v1_execution_with_current_proposal_is_historical_only(
    tmp_path: Path,
) -> None:
    (
        storage,
        _proposal_store,
        authorization_service,
        control_store,
        controller,
        service,
        _proposal,
    ) = _chain(tmp_path)
    intent = authorization_service.approve_and_start(
        project_id="project-1",
        proposal_id=_proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=_proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="v1-history-authorization-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    ).start_intent
    current = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="v1-history-current-controller-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    legacy_payload = current.execution.model_dump(mode="json")
    legacy_payload.update(
        {
            "controller_execution_id": "",
            "execution_digest": "",
            "controller_policy_version": AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V1,
            "controller_policy_digest": "sha256:" + "e" * 64,
            "client_request_id": "v1-history-legacy-controller-1",
        }
    )
    legacy = AgentHarnessControllerExecution.model_validate(legacy_payload)
    control_store._publish_model(
        project_id="project-1",
        kind="harness_controller_execution",
        artifact_id=legacy.controller_execution_id,
        model=legacy,
    )

    inspected = service.inspect(project_id="project-1", run_id="run-1")
    assert inspected.controller is not None
    assert inspected.controller.execution.object_id == current.execution.controller_execution_id
    historical = [
        item
        for item in inspected.source_roster
        if item.source_kind == "controller_execution"
        and item.currentness == "historical"
    ]
    assert [item.source_id for item in historical] == [
        legacy.controller_execution_id
    ]


def test_recovery_required_succeeded_stage_never_claims_verified_task_success(
    tmp_path: Path,
) -> None:
    from tests.test_scientific_agent_harness_controller import (
        _local_prepublication_crash,
        _reopen_local_controller,
    )

    storage, _, _, _, _ = _local_prepublication_crash(tmp_path)
    controller = _reopen_local_controller(str(storage.workspace_dir))
    inspected = _service_from_controller(storage, controller).inspect(
        project_id="project-1", run_id="run-1"
    )

    assert inspected.inspection_status == AgentRunInspectionStatus.RECOVERY_REQUIRED
    assert inspected.verifier_supported_run_outcome == "recovery_required"
    assert inspected.tasks[0].stage_status == "succeeded"
    assert inspected.tasks[0].verifier_supported_outcome == "recovery_required"
    assert inspected.tasks[0].verified_publication is None
    assert inspected.tasks[0].recovery_required is True


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("revision_digest", "sha256:" + "1" * 64),
        ("baseline_proposal_id", "proposal-wrong-baseline"),
        ("baseline_proposal_digest", "sha256:" + "2" * 64),
        ("successor_proposal_digest", "sha256:" + "3" * 64),
        ("successor_semantic_plan_id", "semantic-plan-wrong"),
        ("successor_semantic_plan_digest", "sha256:" + "4" * 64),
        ("plan_diff_id", "plan-diff-wrong"),
        ("plan_diff_digest", "sha256:" + "5" * 64),
        ("parent_proposal_id", "proposal-wrong-parent"),
        ("supersedes_proposal_id", "proposal-wrong-supersedes"),
    ],
)
def test_plan_head_rejects_self_consistent_misbound_replanner_application_fields(
    tmp_path: Path,
    field: str,
    wrong_value: str,
) -> None:
    _, _, revision, application, service = _applied_replan(tmp_path)
    payload = application.receipt.model_dump(mode="json")
    payload[field] = wrong_value
    payload["application_receipt_id"] = ""
    payload["application_receipt_digest"] = ""
    try:
        misbound = AgentPlanRevisionApplicationReceipt.model_validate(payload)
    except ValueError:
        # Some parent/supersedes inconsistencies are rejected even earlier by
        # the strict source schema; they are still fail-closed inputs.
        return
    publications = service._plan_publications(
        project_id="project-1", run_id="run-1"
    )

    with pytest.raises(ValueError):
        service._select_plan_head(
            project_id="project-1",
            publications=publications,
            revisions=[revision],
            applications=[misbound],
        )


def test_inspection_rejects_replaced_successor_publication_request_binding(
    tmp_path: Path,
) -> None:
    storage, _, _, application, service = _applied_replan(tmp_path)
    committed_path = (
        storage.workspace_dir
        / "projects"
        / "project-1"
        / "agent_plan_requests"
        / application.successor.client_request_id
        / "committed.json"
    )
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    committed["request_digest"] = "sha256:" + "0" * 64
    committed_path.write_text(
        json.dumps(committed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AgentRunInspectionReadError) as captured:
        service.inspect(project_id="project-1", run_id="run-1")
    assert captured.value.inspection_status in {
        AgentRunInspectionStatus.REPLACED_SOURCE,
        AgentRunInspectionStatus.INCOMPLETE_AUTHORITY_CHAIN,
    }
