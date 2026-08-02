from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent.execution_agent import ExecutionAgentService
from ai4s_agent.execution_agent_store import ExecutionAgentStore
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.schemas import (
    AgentAuthorizationMode,
    AgentExecutionPlanLLMResponse,
    AgentHarnessControllerStartRequest,
    AgentPlanAuthorizationRequest,
)
from ai4s_agent.scientific_agent_authorization import (
    AgentPlanControlStore,
    ScientificAgentAuthorizationService,
)
from ai4s_agent.scientific_agent_harness_controller import (
    ControllerAdvanceResult,
    ScientificAgentHarnessController,
)
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
)
from ai4s_agent.storage import ProjectStorage


NOW = "2026-08-01T00:00:00Z"


class NoRemoteAuthorities:
    def current_authority(self, **_: object):  # pragma: no cover
        raise AssertionError("local plan consulted remote authority")


class NoRemoteLifecycle:
    pass


class CountingStubProvider(StubLLMProvider):
    def __init__(self, *, response: dict[str, Any]) -> None:
        super().__init__(response=response)
        self.calls = 0

    def complete_json(self, **kwargs: Any):
        self.calls += 1
        return super().complete_json(**kwargs)


def local_controller_execution(
    tmp_path: Path,
) -> tuple[
    ProjectStorage,
    AgentPlanControlStore,
    ScientificAgentHarnessController,
    ControllerAdvanceResult,
]:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at=NOW)
    run_dir = storage.run_dir("project-1", "run-1")
    dataset = run_dir / "inputs" / "dataset.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("SMILES,value\nCCO,1.0\n", encoding="utf-8")
    storage.register_artifact_path(
        "project-1",
        "run-1",
        "uploaded_dataset",
        "inputs/dataset.csv",
    )
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["inspect_dataset"],
        selected_input_artifact_ids=["uploaded_dataset"],
        task_options={"inspect_dataset": {}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce a dataset profile"],
        rationales=["Execute the exact registered local task."],
        assumptions=[],
        questions=[],
    )
    observation_builder = AgentProjectObservationBuilder(
        storage=storage,
        clock=lambda: NOW,
    )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=observation_builder,
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        observation_builder=observation_builder,
        proposal_store=proposal_store,
        clock=lambda: NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Inspect one exact dataset",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="plan-proposal-1",
    )
    control_store = AgentPlanControlStore(storage=storage)
    authorization_service = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        control_store=control_store,
        clock=lambda: NOW,
    )
    approved = authorization_service.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="authorization-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorization_service,
        control_store=control_store,
        resource_authority_service=NoRemoteAuthorities(),
        executor=RunPlanExecutor(storage=storage, registry=proposal_store.registry),
        remote_executions=NoRemoteLifecycle(),
        clock=lambda: NOW,
    )
    initial = controller.create(
        project_id="project-1",
        start_intent_id=approved.start_intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=approved.start_intent.start_intent_digest,
            client_request_id="controller-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    return storage, control_store, controller, initial


def execution_agent_service(
    *,
    storage: ProjectStorage,
    controller: ScientificAgentHarnessController,
    fault_injector=None,
    tracer=None,
) -> ExecutionAgentService:
    return ExecutionAgentService(
        controller=controller,
        store=ExecutionAgentStore(
            storage=storage,
            fault_injector=fault_injector,
        ),
        tracer=tracer,
        clock=lambda: NOW,
    )


def reopen_local_controller(
    workspace_dir: Path,
) -> tuple[ProjectStorage, ScientificAgentHarnessController]:
    storage = ProjectStorage(workspace_dir=workspace_dir)
    observation_builder = AgentProjectObservationBuilder(
        storage=storage,
        clock=lambda: NOW,
    )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=observation_builder,
    )
    control_store = AgentPlanControlStore(storage=storage)
    authorization_service = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        control_store=control_store,
        clock=lambda: NOW,
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorization_service,
        control_store=control_store,
        resource_authority_service=NoRemoteAuthorities(),
        executor=RunPlanExecutor(storage=storage, registry=proposal_store.registry),
        remote_executions=NoRemoteLifecycle(),
        clock=lambda: NOW,
    )
    return storage, controller
