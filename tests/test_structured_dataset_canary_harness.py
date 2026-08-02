from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.agent_run_inspection import AgentRunInspectionService
from ai4s_agent.execution_agent_store import ExecutionAgentStore
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.schemas import (
    AgentAuthorizationMode,
    AgentExecutionPlanLLMResponse,
    AgentHarnessControllerAdvanceRequest,
    AgentHarnessControllerStartRequest,
    AgentHarnessGateApprovalRequest,
    AgentPlanAuthorizationRequest,
)
from ai4s_agent.scientific_agent_authorization import (
    AgentPlanControlStore,
    ScientificAgentAuthorizationService,
)
from ai4s_agent.scientific_agent_harness_controller import (
    ScientificAgentHarnessController,
)
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
)
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_canary import (
    StructuredDatasetCanaryError,
    StructuredDatasetCanaryService,
)
from ai4s_agent.structured_dataset_canary_harness import (
    TASK_IDS,
    run_structured_dataset_ci_harness,
)
from tests.test_structured_dataset_confirmation import NOW, dataset_bytes


class _NoRemoteAuthorities:
    def current_authority(self, **_: object):
        raise AssertionError("local canary consulted remote authority")


class _NoRemoteLifecycle:
    pass


def _authority_chain(tmp_path: Path):
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-1", name="Canary", created_at=NOW)
    source = storage.run_dir("project-1", "run-1") / "inputs" / "raw.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(dataset_bytes())
    storage.register_artifact_path(
        "project-1", "run-1", "uploaded_dataset", "inputs/raw.csv"
    )
    task_ids = [
        "prepare_structured_dataset_canary",
        "confirm_structured_dataset_canary",
        "train_structured_dataset_canary",
        "generate_structured_dataset_canary",
        "evaluate_structured_dataset_canary",
    ]
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=task_ids,
        selected_input_artifact_ids=["uploaded_dataset"],
        task_options={
            "prepare_structured_dataset_canary": {},
            "confirm_structured_dataset_canary": {},
            "train_structured_dataset_canary": {"seed": 7},
            "generate_structured_dataset_canary": {"seed": 7},
            "evaluate_structured_dataset_canary": {"seed": 7, "top_n": 5},
        },
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["publish Computational Top-N"],
        rationales=["Execute the exact BR1 task roster."],
        assumptions=[],
        questions=[],
    )
    builder = AgentProjectObservationBuilder(storage=storage, clock=lambda: NOW)
    proposals = ScientificAgentPlanProposalStore(
        storage=storage, observation_builder=builder
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        observation_builder=builder,
        proposal_store=proposals,
        clock=lambda: NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Run Structured Dataset Canary v1",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="br1-proposal-1",
    )
    controls = AgentPlanControlStore(storage=storage)
    authorizations = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposals,
        control_store=controls,
        clock=lambda: NOW,
    )
    approved = authorizations.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="br1-authorization-1",
        ),
        actor="test-actor",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposals,
        authorization_service=authorizations,
        control_store=controls,
        resource_authority_service=_NoRemoteAuthorities(),
        executor=RunPlanExecutor(storage=storage, registry=proposals.registry),
        remote_executions=_NoRemoteLifecycle(),
        clock=lambda: NOW,
    )
    return storage, controls, controller, approved.start_intent


def _complete(storage, controller, intent):
    result = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="br1-controller-create-1",
        ),
        actor="test-actor",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    gate_ordinal = 0
    advance_ordinal = 0
    for _ in range(30):
        if result.inspection.status.value in {
            "succeeded", "failed", "cancelled", "recovery_required"
        }:
            return result
        if result.inspection.status.value == "waiting_gate":
            stage = storage.read_stage_state("project-1", "run-1")
            assert stage is not None
            snapshot = stage.details["execution_snapshot"]
            spec = controller.executor.registry.get(result.inspection.current_task_id)
            for gate_id in spec.gates:
                gate_ordinal += 1
                result = controller.approve_gate(
                    project_id="project-1",
                    controller_execution_id=result.execution.controller_execution_id,
                    gate_id=gate_id,
                    request=AgentHarnessGateApprovalRequest(
                        expected_snapshot_id=snapshot["snapshot_id"],
                        expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
                        client_request_id=f"br1-gate-{gate_ordinal}",
                        note="CI exact test confirmation",
                    ),
                    actor="test-actor",
                )
        advance_ordinal += 1
        result = controller.advance(
            project_id="project-1",
            controller_execution_id=result.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=result.execution.execution_digest,
                client_request_id=f"br1-advance-{advance_ordinal}",
            ),
        )
    raise AssertionError("Controller did not reach a terminal state")


@pytest.mark.pr_fast
def test_ci_canary_runs_only_through_authorized_harness_tasks(tmp_path: Path) -> None:
    storage, controls, controller, intent = _authority_chain(tmp_path)

    completed = _complete(storage, controller, intent)

    assert completed.inspection.status.value == "succeeded"
    registry = storage.read_artifact_registry("project-1", "run-1")
    assert "computational_top_n" in registry
    topn = json.loads(
        (storage.run_dir("project-1", "run-1") / registry["computational_top_n"]).read_text()
    )
    assert topn["artifact_name"] == "Computational Top-N"
    receipts = controls.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=completed.execution.controller_execution_id,
    )
    assert receipts
    training_dispatch = next(
        item
        for item in controls.list_harness_local_dispatch_receipts(
            project_id="project-1",
            controller_execution_id=completed.execution.controller_execution_id,
        )
        if item.task_id == "train_structured_dataset_canary"
    )
    training_publication = next(
        item
        for item in controls.list_harness_local_execution_publications(
            project_id="project-1",
            controller_execution_id=completed.execution.controller_execution_id,
        )
        if item.task_id == "train_structured_dataset_canary"
    )
    training_receipt = next(
        item
        for item in receipts
        if item.task_id == "train_structured_dataset_canary"
        and item.local_dispatch_receipt_ids
    )
    assert training_publication.local_dispatch_receipt_id == training_dispatch.dispatch_receipt_id
    assert training_publication.attempt_ordinal == training_dispatch.attempt_ordinal
    assert training_receipt.local_dispatch_receipt_ids == [
        training_dispatch.dispatch_receipt_id
    ]
    assert {item.artifact_id for item in training_publication.verified_outputs} == {
        "training_request", "trained_model", "model_package"
    }
    assert not list(
        storage.run_dir("project-1", "run-1").glob(
            "structured_dataset_canary/*_controller_receipt.json"
        )
    )

    inspection = AgentRunInspectionService(
        storage=storage,
        proposal_store=controller.proposal_store,
        authorization_service=controller.authorization_service,
        control_store=controls,
        controller=controller,
        execution_agent_store=ExecutionAgentStore(storage=storage),
        clock=lambda: NOW,
    ).inspect(project_id="project-1", run_id="run-1")
    assert inspection.structured_dataset_canary is None
    assert {item.task_id for item in inspection.tasks}.issuperset(set(TASK_IDS))
    assert {item.artifact_id for item in inspection.artifacts}.issuperset(
        {"model_package", "generation_publication", "computational_top_n"}
    )


def test_preexisting_checkpoint_is_rejected_by_current_training_attempt(
    tmp_path: Path,
) -> None:
    storage, _, controller, intent = _authority_chain(tmp_path)
    checkpoint = (
        storage.run_dir("project-1", "run-1")
        / "structured_dataset_canary"
        / "model_checkpoint.json"
    )
    checkpoint.parent.mkdir()
    checkpoint.write_text("{}", encoding="utf-8")

    completed = _complete(storage, controller, intent)

    assert completed.inspection.status.value in {"failed", "recovery_required"}
    registry = storage.read_artifact_registry("project-1", "run-1")
    assert "model_package" not in registry


def test_crash_after_checkpoint_never_retrains_on_ordinary_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, _, controller, intent = _authority_chain(tmp_path)
    result = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="checkpoint-crash-create",
        ),
        actor="test-actor",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    advance = 0
    gate = 0
    while not (
        result.inspection.status.value == "waiting_gate"
        and result.inspection.current_task_id == "train_structured_dataset_canary"
    ):
        if result.inspection.status.value == "waiting_gate":
            stage = storage.read_stage_state("project-1", "run-1")
            assert stage is not None
            snapshot = stage.details["execution_snapshot"]
            gate += 1
            result = controller.approve_gate(
                project_id="project-1",
                controller_execution_id=result.execution.controller_execution_id,
                gate_id=controller.executor.registry.get(
                    result.inspection.current_task_id
                ).gates[0],
                request=AgentHarnessGateApprovalRequest(
                    expected_snapshot_id=snapshot["snapshot_id"],
                    expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
                    client_request_id=f"checkpoint-crash-gate-{gate}",
                    note="exact test confirmation",
                ),
                actor="test-actor",
            )
        advance += 1
        result = controller.advance(
            project_id="project-1",
            controller_execution_id=result.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=result.execution.execution_digest,
                client_request_id=f"checkpoint-crash-advance-{advance}",
            ),
        )

    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    snapshot = stage.details["execution_snapshot"]
    result = controller.approve_gate(
        project_id="project-1",
        controller_execution_id=result.execution.controller_execution_id,
        gate_id="gate_3_train_config",
        request=AgentHarnessGateApprovalRequest(
            expected_snapshot_id=snapshot["snapshot_id"],
            expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
            client_request_id="checkpoint-crash-training-gate",
            note="exact training approval",
        ),
        actor="test-actor",
    )
    original_publish = StructuredDatasetCanaryService._publish

    def crash_before_model_package(self, project_id, run_id, name, payload, digest_field):
        if name == "model_package.json":
            raise KeyboardInterrupt("injected crash after checkpoint")
        return original_publish(self, project_id, run_id, name, payload, digest_field)

    monkeypatch.setattr(StructuredDatasetCanaryService, "_publish", crash_before_model_package)
    with pytest.raises(KeyboardInterrupt, match="after checkpoint"):
        controller.advance(
            project_id="project-1",
            controller_execution_id=result.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=result.execution.execution_digest,
                client_request_id="checkpoint-crash-training-execute",
            ),
        )
    monkeypatch.setattr(StructuredDatasetCanaryService, "_publish", original_publish)
    checkpoint = (
        storage.run_dir("project-1", "run-1")
        / "structured_dataset_canary"
        / "model_checkpoint.json"
    )
    checkpoint_bytes = checkpoint.read_bytes()

    inspected = controller.get(
        project_id="project-1",
        controller_execution_id=result.execution.controller_execution_id,
    )

    assert inspected.inspection.status.value == "recovery_required"
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert "model_package" not in storage.read_artifact_registry(
        "project-1", "run-1"
    )


def test_direct_service_cannot_bypass_harness_authority(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-1", name="Canary", created_at=NOW)
    source = tmp_path / "raw.csv"
    source.write_bytes(dataset_bytes())

    with pytest.raises(StructuredDatasetCanaryError, match="approve-and-start"):
        StructuredDatasetCanaryService(
            storage=storage,
            trusted_actors={"test-actor"},
        ).run_ci_reference(
            project_id="project-1",
            run_id="run-1",
            raw_csv=source,
            actor="test-actor",
        )
    assert storage.read_stage_state("project-1", "run-1") is None
    assert storage.read_artifact_registry("project-1", "run-1") == {}


def test_public_ci_runner_uses_harness_authority_chain(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-1", name="Canary", created_at=NOW)
    source = tmp_path / "raw.csv"
    source.write_bytes(dataset_bytes())

    result = run_structured_dataset_ci_harness(
        storage=storage,
        project_id="project-1",
        run_id="run-1",
        raw_csv=source,
        actor="test-actor",
        seed=11,
        top_n=4,
    )

    assert result.computational_top_n["artifact_name"] == "Computational Top-N"
    assert result.controller_execution_id.startswith("controller-")
