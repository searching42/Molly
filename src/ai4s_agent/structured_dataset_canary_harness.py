from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.generation_publication import publish_fresh_bytes, read_regular_file_bound
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
from ai4s_agent.structured_dataset_confirmation import read_json_artifact
from ai4s_agent.structured_dataset_canary import StructuredDatasetCanaryService


TASK_IDS = (
    "prepare_structured_dataset_canary",
    "confirm_structured_dataset_canary",
    "train_structured_dataset_canary",
    "generate_structured_dataset_canary",
    "evaluate_structured_dataset_canary",
)


@dataclass(frozen=True)
class StructuredDatasetHarnessResult:
    controller_execution_id: str
    controller_execution_digest: str
    evidence: dict[str, Any]
    computational_top_n: dict[str, Any]


class _NoRemoteAuthority:
    def current_authority(self, **_: object) -> Any:
        raise RuntimeError("CI reference canary has no remote authority")


class _NoRemoteLifecycle:
    pass


def run_structured_dataset_ci_harness(
    *,
    storage: ProjectStorage,
    project_id: str,
    run_id: str,
    raw_csv: str | Path,
    actor: str,
    seed: int = 1729,
    top_n: int = 5,
) -> StructuredDatasetHarnessResult:
    """Run BR1 through proposal, permission, authorization, and Controller only."""

    clean_actor = str(actor or "").strip()
    if not clean_actor:
        raise ValueError("CI Harness actor is required")
    if top_n < 1 or top_n > 100:
        raise ValueError("top_n must be between 1 and 100")
    source_bytes, _ = read_regular_file_bound(Path(raw_csv), max_bytes=16 * 1024 * 1024)
    run_dir = storage.run_dir(project_id, run_id)
    input_path = run_dir / "inputs" / "structured_dataset_canary_raw.csv"
    if input_path.exists():
        existing_bytes, _ = read_regular_file_bound(
            input_path, max_bytes=16 * 1024 * 1024
        )
        if existing_bytes != source_bytes:
            raise ValueError("Raw Dataset source differs from current authority")
    else:
        publish_fresh_bytes(input_path, source_bytes)
    registry = storage.read_artifact_registry(project_id, run_id)
    relative = input_path.relative_to(run_dir).as_posix()
    if "uploaded_dataset" in registry and registry["uploaded_dataset"] != relative:
        raise ValueError("uploaded_dataset is already bound to another authority")
    if "uploaded_dataset" not in registry:
        storage.register_artifact_path(project_id, run_id, "uploaded_dataset", relative)

    builder = AgentProjectObservationBuilder(storage=storage)
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage, observation_builder=builder
    )
    controls = AgentPlanControlStore(storage=storage)
    authorizations = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        control_store=controls,
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorizations,
        control_store=controls,
        resource_authority_service=_NoRemoteAuthority(),
        executor=RunPlanExecutor(storage=storage, registry=proposal_store.registry),
        remote_executions=_NoRemoteLifecycle(),
    )
    registry = storage.read_artifact_registry(project_id, run_id)
    if "structured_dataset_canary_evidence" in registry:
        executions = [
            item
            for item in controls.list_harness_controller_executions(
                project_id=project_id
            )
            if item.run_id == run_id
        ]
        if len(executions) != 1:
            raise ValueError("completed canary lacks one exact Controller execution")
        inspected = controller.get(
            project_id=project_id,
            controller_execution_id=executions[0].controller_execution_id,
        )
        if inspected.inspection.status.value != "succeeded":
            raise ValueError("completed canary Controller is not succeeded")
        evidence = read_json_artifact(
            run_dir / registry["structured_dataset_canary_evidence"],
            digest_field="evidence_digest",
        )
        StructuredDatasetCanaryService(
            storage=storage,
            trusted_actors={clean_actor},
            harness_authority_managed=True,
        )._verify_final_evidence(project_id, run_id, evidence)
        topn = read_json_artifact(
            run_dir / registry["computational_top_n"],
            digest_field="publication_digest",
        )
        ranking = read_json_artifact(
            run_dir / registry["ranking_publication"],
            digest_field="publication_digest",
        )
        if evidence.get("seed") != seed or ranking.get("ranking_configuration", {}).get(
            "top_n_size"
        ) != top_n:
            raise ValueError("replay configuration differs from current authority")
        return StructuredDatasetHarnessResult(
            controller_execution_id=executions[0].controller_execution_id,
            controller_execution_digest=executions[0].execution_digest,
            evidence=evidence,
            computational_top_n=topn,
        )

    options = {
        "prepare_structured_dataset_canary": {},
        "confirm_structured_dataset_canary": {},
        "train_structured_dataset_canary": {"seed": seed},
        "generate_structured_dataset_canary": {"seed": seed},
        "evaluate_structured_dataset_canary": {"seed": seed, "top_n": top_n},
    }
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=list(TASK_IDS),
        selected_input_artifact_ids=["uploaded_dataset"],
        task_options=options,
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["publish Computational Top-N"],
        rationales=["Execute the exact registered BR1 task roster."],
        assumptions=[],
        questions=[],
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        observation_builder=builder,
        proposal_store=proposal_store,
    ).create_proposal(
        project_id=project_id,
        run_id=run_id,
        goal="Run Structured Dataset Canary v1",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id=f"{run_id}-structured-dataset-proposal-v1",
    )
    approved = authorizations.approve_and_start(
        project_id=project_id,
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id=f"{run_id}-structured-dataset-authorization-v1",
        ),
        actor=clean_actor,
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    result = controller.create(
        project_id=project_id,
        start_intent_id=approved.start_intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=approved.start_intent.start_intent_digest,
            client_request_id=f"{run_id}-structured-dataset-controller-v1",
        ),
        actor=clean_actor,
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    gate_ordinal = 0
    advance_ordinal = 0
    while result.inspection.status.value not in {
        "succeeded", "failed", "cancelled", "recovery_required"
    }:
        if result.inspection.status.value == "waiting_gate":
            stage = storage.read_stage_state(project_id, run_id)
            if stage is None or not isinstance(stage.details.get("execution_snapshot"), dict):
                raise RuntimeError("Controller Gate snapshot is unavailable")
            snapshot = stage.details["execution_snapshot"]
            spec = controller.executor.registry.get(result.inspection.current_task_id)
            for gate_id in spec.gates:
                gate_ordinal += 1
                result = controller.approve_gate(
                    project_id=project_id,
                    controller_execution_id=result.execution.controller_execution_id,
                    gate_id=gate_id,
                    request=AgentHarnessGateApprovalRequest(
                        expected_snapshot_id=str(snapshot["snapshot_id"]),
                        expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
                        client_request_id=f"{run_id}-structured-dataset-gate-{gate_ordinal}",
                        note="CI exact test confirmation",
                    ),
                    actor=clean_actor,
                )
        advance_ordinal += 1
        result = controller.advance(
            project_id=project_id,
            controller_execution_id=result.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=result.execution.execution_digest,
                client_request_id=f"{run_id}-structured-dataset-advance-{advance_ordinal}",
            ),
        )
        if advance_ordinal > 32:
            raise RuntimeError("Controller action bound exceeded")
    if result.inspection.status.value != "succeeded":
        raise RuntimeError(f"CI Harness ended {result.inspection.status.value}")
    registry = storage.read_artifact_registry(project_id, run_id)
    evidence = read_json_artifact(
        run_dir / registry["structured_dataset_canary_evidence"],
        digest_field="evidence_digest",
    )
    topn = read_json_artifact(
        run_dir / registry["computational_top_n"], digest_field="publication_digest"
    )
    return StructuredDatasetHarnessResult(
        controller_execution_id=result.execution.controller_execution_id,
        controller_execution_digest=result.execution.execution_digest,
        evidence=evidence,
        computational_top_n=topn,
    )
