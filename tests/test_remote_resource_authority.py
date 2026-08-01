from __future__ import annotations

import json
import multiprocessing
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.remote_resource_authority import (
    RemoteResourceAuthorityConflict,
    RemoteResourceAuthorityDenied,
    RemoteResourceAuthorityPolicyStore,
    RemoteResourceAuthorityService,
    RemoteResourceAuthorityStale,
)
from ai4s_agent.resource_profiles import (
    CapabilityDetails,
    CapabilityProbeResult,
    ConnectionProfile,
    CudaCapabilityDetails,
    ResourceProfileStore,
)
from ai4s_agent.schemas import (
    AgentAuthorizationMode,
    AgentBudgetObservation,
    AgentConfiguredRemoteResources,
    AgentExecutionPlanLLMResponse,
    AgentPermissionOutcome,
    AgentPermissionPhase,
    AgentPlanAuthorizationRequest,
    AgentProjectObservation,
    AgentRemoteResourceAuthority,
    AgentRemoteResourceAuthorityDecision,
    AgentRemoteResourceAuthorityOutcome,
    AgentRemoteResourceAuthoritySet,
    AgentRemoteResourceAuthorityRequest,
    AgentRemoteResourceBudgetLimits,
    AtomicTaskSpec,
    ArtifactRef,
    RemoteResourceAuthorityPolicy,
    RemoteResourceAuthorityPolicyEntry,
    RiskLevel,
    RunStatus,
    StageState,
)
from ai4s_agent.scientific_agent_authorization import (
    AgentPlanControlStore,
    ScientificAgentAuthorizationService,
    ScientificAgentAuthorizationVerificationError,
)
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
    ScientificAgentPlanSourceChanged,
)
from ai4s_agent.scientific_agent_permissions import (
    PERMISSION_POLICY_DIGEST,
    RESOURCE_AWARE_PERMISSION_POLICY_DIGEST,
    ScientificAgentPermissionEngine,
)
from ai4s_agent.storage import ProjectStorage


NOW = "2026-08-01T00:00:00Z"
_NO_HIDDEN_DEPENDENCY = object()


class _ConfiguredBudgetObservationBuilder(AgentProjectObservationBuilder):
    """Test-only server projection for an exact configured legacy budget."""

    def __init__(self, *, budget_limits: AgentBudgetObservation, **kwargs) -> None:
        super().__init__(**kwargs)
        self._configured_budget_limits = budget_limits

    def build(
        self,
        *,
        project_id: str,
        run_id: str,
        goal: str,
        user_constraints: list[str] | None = None,
    ) -> AgentProjectObservation:
        observation = super().build(
            project_id=project_id,
            run_id=run_id,
            goal=goal,
            user_constraints=user_constraints,
        )
        payload = observation.model_dump(mode="json")
        payload["budget_limits"] = self._configured_budget_limits.model_dump(
            mode="json"
        )
        payload["observation_id"] = ""
        payload["observation_digest"] = ""
        return AgentProjectObservation.model_validate(payload)


def _remote_task(
    task_type: str,
    profile_id: str,
    *,
    task_id: str = "remote_task",
    depends_on: list[str] | None = None,
    required_artifacts: list[str] | None = None,
    output_artifacts: list[str] | None = None,
) -> AtomicTaskSpec:
    permission = {
        "document_parsing": "external_document_processing",
        "model_training": "model_training_compute",
        "molecular_generation": "candidate_generation_compute",
    }[task_type]
    return AtomicTaskSpec(
        task_id=task_id,
        required_artifacts=required_artifacts or [],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=output_artifacts or [f"{task_id}_output"],
        risk_level=RiskLevel.MEDIUM,
        gates=[],
        default_adapter=None,
        depends_on=depends_on or [],
        scientific_tool_id=task_id,
        label="Remote Task",
        description="A review-only remote scientific task.",
        effect_class="compute",
        required_permissions=[permission],
        option_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[profile_id],
        backend_profile_requirements={},
        default_planner_backend=None,
        execution_route="remote_execution_service",
        remote_task_type=task_type,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact={
            artifact_id: ["registered_intermediate", "verified_output"]
            for artifact_id in required_artifacts or []
        },
        budget_dimensions=["max_runtime_sec", "max_gpu_hours"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    )


def _hidden_local_dependency(
    *, budget_dimensions: list[str] | None
) -> AtomicTaskSpec:
    metadata = {
        "task_id": "hidden_prepare_task",
        "default_adapter": "inspect_dataset_service",
        "risk_level": RiskLevel.LOW,
        "gates": [],
        "effect_class": "derive_local",
        "required_permissions": ["derive_project_artifact"],
        "option_schema": None,
        "default_planner_options": {},
        "backend_default_planner_options": {},
        "review_required_option_ids": [],
        "execution_route": "local_executor",
        "remote_task_type": None,
        "backend_execution_routes": {},
        "backend_remote_task_types": {},
        "supports_plan_preapproval": False,
        "idempotency_policy": "server_checked",
        "verification_policy": "artifact_registry_and_stage_verifier",
        "planner_visible": False,
    }
    if budget_dimensions is not None:
        metadata["budget_dimensions"] = budget_dimensions
    return AtomicTaskSpec(**metadata)


def _configured_case(
    tmp_path: Path,
    *,
    task_type: str = "molecular_generation",
    profile_id: str = "reinvent4-cpu-v1",
    capabilities: list[str] | None = None,
    resources: tuple[int, int, int] = (0, 1, 600),
    cuda_status: str = "unknown",
    hidden_budget_dimensions: object = _NO_HIDDEN_DEPENDENCY,
    legacy_runtime_limit: int | None = None,
):
    capabilities = capabilities or ["cpu", "reinvent4"]
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at=NOW)
    config = tmp_path / "config"
    profiles = ResourceProfileStore(
        workspace_dir=storage.workspace_dir,
        config_dir=config,
    )
    connection = profiles.save_connection(
        ConnectionProfile(
            connection_id="scientific-worker",
            ssh_host_alias="scientific-worker",
            expected_hostname="scientific-worker",
            remote_root="/srv/molly",
            declared_capabilities=capabilities,
        )
    )
    profiles.save_probe(
        CapabilityProbeResult(
            connection_id=connection.connection_id,
            connection_profile_digest=connection.digest(),
            status="available",
            checked_at=NOW,
            verified_capabilities=capabilities,
            details=CapabilityDetails(
                cpu_threads=32,
                cuda=(
                    None
                    if cuda_status == "unknown"
                    else CudaCapabilityDetails(status=cuda_status)
                ),
            ),
        )
    )
    include_hidden_dependency = hidden_budget_dimensions is not _NO_HIDDEN_DEPENDENCY
    remote_task = _remote_task(
        task_type,
        profile_id,
        depends_on=["hidden_prepare_task"] if include_hidden_dependency else None,
    )
    tasks = [remote_task]
    if include_hidden_dependency:
        if hidden_budget_dimensions is not None and not isinstance(
            hidden_budget_dimensions, list
        ):
            raise AssertionError("hidden budget dimensions must be a list or None")
        tasks.insert(
            0,
            _hidden_local_dependency(
                budget_dimensions=hidden_budget_dimensions,
            ),
        )
    registry = AtomicTaskRegistry(tasks)
    builder_kwargs = {
        "storage": storage,
        "registry": registry,
        "resource_profiles": profiles,
        "clock": lambda: NOW,
    }
    if legacy_runtime_limit is None:
        builder = AgentProjectObservationBuilder(**builder_kwargs)
    else:
        builder = _ConfiguredBudgetObservationBuilder(
            **builder_kwargs,
            budget_limits=AgentBudgetObservation(
                status="configured",
                limits={"max_runtime_sec": legacy_runtime_limit},
                dimensions=["max_runtime_sec"],
            ),
        )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        registry=registry,
        observation_builder=builder,
    )
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["remote_task"],
        selected_input_artifact_ids=[],
        task_options={"remote_task": {}},
        selected_logical_profile_ids=[profile_id],
        limits={"max_runtime_sec": resources[2], "max_gpu_hours": max(1, resources[0] * resources[2] / 3600)},
        stop_conditions=["stop on verification failure"],
        success_criteria=["produce one reviewable output"],
        rationales=["Use the fixed remote profile."],
        assumptions=[],
        questions=[],
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        registry=registry,
        observation_builder=builder,
        proposal_store=proposal_store,
        clock=lambda: NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Prepare an exact remote task",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="proposal-request",
    )
    policy_store = RemoteResourceAuthorityPolicyStore(config_dir=config)
    policy_store.save(
        RemoteResourceAuthorityPolicy(
            entries=[
                RemoteResourceAuthorityPolicyEntry(
                    policy_id="remote-task-policy",
                    enabled=True,
                    connection_id=connection.connection_id,
                    execution_profile_id=profile_id,
                    remote_task_type=task_type,
                    allowed_task_ids=["remote_task"],
                    configured_resources=AgentConfiguredRemoteResources(
                        gpu_count=resources[0],
                        cpu_threads=resources[1],
                        walltime_sec=resources[2],
                    ),
                    budget_limits=AgentRemoteResourceBudgetLimits(
                        max_runtime_sec=resources[2],
                        max_gpu_hours=max(1, resources[0] * resources[2] / 3600),
                    ),
                )
            ]
        )
    )
    control = AgentPlanControlStore(storage=storage)
    resources_service = RemoteResourceAuthorityService(
        proposal_store=proposal_store,
        resource_profiles=profiles,
        policy_store=policy_store,
        control_store=control,
        clock=lambda: NOW,
    )
    authorization = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        registry=registry,
        control_store=control,
        resource_authority_resolver=lambda publication, task_id: resources_service.current_authority(
            publication=publication, task_id=task_id
        ),
        clock=lambda: NOW,
    )
    return (
        storage,
        profiles,
        policy_store,
        proposal_store,
        proposal,
        resources_service,
        authorization,
    )


def _multi_remote_case(
    tmp_path: Path,
    *,
    resources: tuple[int, int, int] = (1, 2, 1800),
    max_runtime_sec: int = 4000,
    max_gpu_hours: float = 2.0,
):
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at=NOW)
    config = tmp_path / "config"
    profiles = ResourceProfileStore(
        workspace_dir=storage.workspace_dir,
        config_dir=config,
    )
    connection = profiles.save_connection(
        ConnectionProfile(
            connection_id="multi-worker",
            ssh_host_alias="multi-worker",
            expected_hostname="multi-worker",
            remote_root="/srv/molly",
            declared_capabilities=["gpu", "unimol"],
        )
    )
    profiles.save_probe(
        CapabilityProbeResult(
            connection_id=connection.connection_id,
            connection_profile_digest=connection.digest(),
            status="available",
            checked_at=NOW,
            verified_capabilities=["gpu", "unimol"],
            details=CapabilityDetails(
                cpu_threads=32,
                cuda=CudaCapabilityDetails(status="available"),
            ),
        )
    )
    registry = AtomicTaskRegistry(
        [
            _remote_task(
                "model_training",
                "unimol-train-v1",
                task_id="train_model",
                output_artifacts=["trained_remote_model"],
            ),
            _remote_task(
                "model_training",
                "unimol-train-v1",
                task_id="generate_candidates",
                depends_on=["train_model"],
                required_artifacts=["trained_remote_model"],
                output_artifacts=["generated_candidates"],
            ),
        ]
    )
    builder = AgentProjectObservationBuilder(
        storage=storage,
        registry=registry,
        resource_profiles=profiles,
        clock=lambda: NOW,
    )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        registry=registry,
        observation_builder=builder,
    )
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["train_model", "generate_candidates"],
        selected_input_artifact_ids=[],
        task_options={"train_model": {}, "generate_candidates": {}},
        selected_logical_profile_ids=["unimol-train-v1"],
        limits={
            "max_runtime_sec": max_runtime_sec,
            "max_gpu_hours": max_gpu_hours,
        },
        stop_conditions=["stop on verification failure"],
        success_criteria=["produce a complete remote roster"],
        rationales=["Exercise canonical multi-remote ordering."],
        assumptions=[],
        questions=[],
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        registry=registry,
        observation_builder=builder,
        proposal_store=proposal_store,
        clock=lambda: NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Prepare two ordered remote tasks",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="proposal-request",
    )
    policy_store = RemoteResourceAuthorityPolicyStore(config_dir=config)
    policy_store.save(
        RemoteResourceAuthorityPolicy(
            entries=[
                RemoteResourceAuthorityPolicyEntry(
                    policy_id="multi-remote-policy",
                    enabled=True,
                    connection_id=connection.connection_id,
                    execution_profile_id="unimol-train-v1",
                    remote_task_type="model_training",
                    allowed_task_ids=["train_model", "generate_candidates"],
                    configured_resources=AgentConfiguredRemoteResources(
                        gpu_count=resources[0],
                        cpu_threads=resources[1],
                        walltime_sec=resources[2],
                    ),
                    budget_limits=AgentRemoteResourceBudgetLimits(
                        max_runtime_sec=resources[2],
                        max_gpu_hours=max(1.0, resources[0] * resources[2] / 3600),
                    ),
                )
            ]
        )
    )
    control = AgentPlanControlStore(storage=storage)
    service = RemoteResourceAuthorityService(
        proposal_store=proposal_store,
        resource_profiles=profiles,
        policy_store=policy_store,
        control_store=control,
        clock=lambda: NOW,
    )
    authorization = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        registry=registry,
        control_store=control,
        resource_authority_resolver=lambda publication, task_id: service.current_authority(
            publication=publication, task_id=task_id
        ),
        clock=lambda: NOW,
    )
    return storage, profiles, policy_store, proposal_store, proposal, service, authorization


def _default_registry_mixed_case(
    tmp_path: Path,
    *,
    workflow: str,
    legacy_runtime_limit: int | None = None,
):
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at=NOW)
    run_dir = storage.run_dir("project-1", "run-1")
    config = tmp_path / "config"
    if workflow == "unimol":
        dataset = run_dir / "data" / "confirmed_training_dataset.json"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_text(
            json.dumps(
                {
                    "dataset_id": "confirmed-dataset",
                    "confirmed": True,
                    "status": "confirmed",
                    "row_count": 4,
                    "column_ids": ["smiles", "plqy"],
                    "target_property": "plqy",
                }
            ),
            encoding="utf-8",
        )
        storage.register_artifact_path(
            "project-1",
            "run-1",
            "confirmed_training_dataset",
            "data/confirmed_training_dataset.json",
        )
        storage.write_stage_state(
            "project-1",
            "run-1",
            StageState(
                stage="confirm_extracted_dataset",
                next_stage="train_model",
                status=RunStatus.SUCCEEDED,
                started_at=NOW,
                updated_at=NOW,
                artifacts=[
                    ArtifactRef(
                        artifact_id="confirmed_training_dataset",
                        relative_path="data/confirmed_training_dataset.json",
                        producer_task_id="confirm_extracted_dataset",
                    )
                ],
            ),
        )
        profile_id = "unimol-train-v1"
        task_type = "model_training"
        remote_task_id = "train_model"
        capabilities = ["gpu", "unimol"]
        resources = (1, 4, 1200)
        response = AgentExecutionPlanLLMResponse(
            requested_tool_ids=["train_model"],
            selected_input_artifact_ids=["confirmed_training_dataset"],
            task_options={
                "train_model": {"backend": "unimol", "property_id": "plqy"}
            },
            selected_logical_profile_ids=[profile_id],
            limits={"max_runtime_sec": 1200, "max_gpu_hours": 1.0},
            stop_conditions=["stop on verification failure"],
            success_criteria=["produce a reviewable Uni-Mol model"],
            rationales=["Use the configured remote Uni-Mol backend."],
            assumptions=[],
            questions=[],
        )
    elif workflow == "mineru":
        pdf = run_dir / "inputs" / "papers.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF-1.7 review-only input\n")
        storage.register_artifact_path(
            "project-1", "run-1", "pdf_corpus", "inputs/papers.pdf"
        )
        profile_id = "mineru-v1"
        task_type = "document_parsing"
        remote_task_id = "parse_document"
        capabilities = ["gpu", "mineru"]
        resources = (1, 4, 600)
        response = AgentExecutionPlanLLMResponse(
            requested_tool_ids=["index_corpus"],
            selected_input_artifact_ids=["pdf_corpus"],
            task_options={"index_corpus": {}},
            selected_logical_profile_ids=[profile_id],
            limits={"max_runtime_sec": 600, "max_gpu_hours": 1.0},
            stop_conditions=["stop on verification failure"],
            success_criteria=["produce a reviewable corpus index"],
            rationales=["Use the configured MinerU parser before local indexing."],
            assumptions=[],
            questions=[],
        )
    elif workflow == "reinvent":
        model_metadata = run_dir / "models" / "model_metadata.json"
        model_metadata.parent.mkdir(parents=True, exist_ok=True)
        model_metadata.write_text(
            json.dumps(
                {
                    "model_id": "reviewed-model",
                    "property_ids": ["plqy"],
                    "verification_state": "verified",
                }
            ),
            encoding="utf-8",
        )
        storage.register_artifact_path(
            "project-1",
            "run-1",
            "model_metadata",
            "models/model_metadata.json",
        )
        storage.write_stage_state(
            "project-1",
            "run-1",
            StageState(
                stage="train_model",
                next_stage="generate_candidates",
                status=RunStatus.SUCCEEDED,
                started_at=NOW,
                updated_at=NOW,
                artifacts=[
                    ArtifactRef(
                        artifact_id="model_metadata",
                        relative_path="models/model_metadata.json",
                        producer_task_id="train_model",
                    )
                ],
            ),
        )
        profile_id = "reinvent4-cpu-v1"
        task_type = "molecular_generation"
        remote_task_id = "generate_candidates"
        capabilities = ["cpu", "reinvent4"]
        resources = (0, 1, 600)
        response = AgentExecutionPlanLLMResponse(
            requested_tool_ids=["generate_candidates", "predict_candidates"],
            selected_input_artifact_ids=["model_metadata"],
            task_options={
                "generate_candidates": {
                    "backend": "reinvent4",
                    "count": 32,
                    "seed": 0,
                },
                "predict_candidates": {"property_id": "plqy"},
            },
            selected_logical_profile_ids=[profile_id],
            limits={"max_runtime_sec": 600, "max_gpu_hours": 1.0},
            stop_conditions=["stop on verification failure"],
            success_criteria=["produce reviewable candidate predictions"],
            rationales=[
                "Use remote REINVENT4 generation before local prediction."
            ],
            assumptions=[],
            questions=[],
        )
    else:  # pragma: no cover - helper is called with a frozen test matrix.
        raise AssertionError(f"unknown workflow: {workflow}")

    profiles = ResourceProfileStore(
        workspace_dir=storage.workspace_dir,
        config_dir=config,
    )
    connection = profiles.save_connection(
        ConnectionProfile(
            connection_id=f"{workflow}-worker",
            ssh_host_alias=f"{workflow}-worker",
            expected_hostname=f"{workflow}-worker",
            remote_root="/srv/molly",
            declared_capabilities=capabilities,
        )
    )
    profiles.save_probe(
        CapabilityProbeResult(
            connection_id=connection.connection_id,
            connection_profile_digest=connection.digest(),
            status="available",
            checked_at=NOW,
            verified_capabilities=capabilities,
            details=CapabilityDetails(
                cpu_threads=32,
                cuda=CudaCapabilityDetails(status="available"),
            ),
        )
    )
    registry = AtomicTaskRegistry()
    builder_kwargs = {
        "storage": storage,
        "registry": registry,
        "resource_profiles": profiles,
        "clock": lambda: NOW,
    }
    if legacy_runtime_limit is None:
        builder = AgentProjectObservationBuilder(**builder_kwargs)
    else:
        builder = _ConfiguredBudgetObservationBuilder(
            **builder_kwargs,
            budget_limits=AgentBudgetObservation(
                status="configured",
                limits={"max_runtime_sec": legacy_runtime_limit},
                dimensions=["max_runtime_sec"],
            ),
        )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        registry=registry,
        observation_builder=builder,
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        registry=registry,
        observation_builder=builder,
        proposal_store=proposal_store,
        clock=lambda: NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal=f"Prepare the representative {workflow} mixed workflow",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="proposal-request",
    )
    policy_store = RemoteResourceAuthorityPolicyStore(config_dir=config)
    policy_store.save(
        RemoteResourceAuthorityPolicy(
            entries=[
                RemoteResourceAuthorityPolicyEntry(
                    policy_id=f"{workflow}-policy",
                    enabled=True,
                    connection_id=connection.connection_id,
                    execution_profile_id=profile_id,
                    remote_task_type=task_type,
                    allowed_task_ids=[remote_task_id],
                    configured_resources=AgentConfiguredRemoteResources(
                        gpu_count=resources[0],
                        cpu_threads=resources[1],
                        walltime_sec=resources[2],
                    ),
                    budget_limits=AgentRemoteResourceBudgetLimits(
                        max_runtime_sec=resources[2],
                        max_gpu_hours=1.0,
                    ),
                )
            ]
        )
    )
    control = AgentPlanControlStore(storage=storage)
    service = RemoteResourceAuthorityService(
        proposal_store=proposal_store,
        resource_profiles=profiles,
        policy_store=policy_store,
        control_store=control,
        clock=lambda: NOW,
    )
    authorization = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        registry=registry,
        control_store=control,
        resource_authority_resolver=lambda publication, task_id: service.current_authority(
            publication=publication, task_id=task_id
        ),
        clock=lambda: NOW,
    )
    return storage, proposal, service, authorization


def _multiprocess_publish_worker(
    workspace: str,
    config: str,
    proposal_id: str,
    proposal_digest: str,
    start_event,
    result_queue,
) -> None:
    storage = ProjectStorage(workspace_dir=Path(workspace))
    profiles = ResourceProfileStore(
        workspace_dir=Path(workspace), config_dir=Path(config)
    )
    registry = AtomicTaskRegistry(
        [_remote_task("molecular_generation", "reinvent4-cpu-v1")]
    )
    builder = AgentProjectObservationBuilder(
        storage=storage,
        registry=registry,
        resource_profiles=profiles,
        clock=lambda: NOW,
    )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        registry=registry,
        observation_builder=builder,
    )
    service = RemoteResourceAuthorityService(
        proposal_store=proposal_store,
        resource_profiles=profiles,
        policy_store=RemoteResourceAuthorityPolicyStore(config_dir=Path(config)),
        control_store=AgentPlanControlStore(storage=storage),
        clock=lambda: NOW,
    )
    start_event.wait()
    try:
        result = service.publish(
            project_id="project-1",
            proposal_id=proposal_id,
            request=AgentRemoteResourceAuthorityRequest(
                expected_proposal_digest=proposal_digest,
                client_request_id="cross-process-request",
            ),
        )
    except Exception as exc:  # noqa: BLE001 - parent asserts process result.
        result_queue.put((type(exc).__name__, ""))
    else:
        result_queue.put(("success", result.authorities[0].authority_digest))


@pytest.mark.parametrize(
    ("task_type", "profile_id", "capabilities", "resources", "cuda_status"),
    [
        ("molecular_generation", "reinvent4-cpu-v1", ["cpu", "reinvent4"], (0, 1, 600), "unknown"),
        ("document_parsing", "mineru-v1", ["gpu", "mineru"], (1, 4, 600), "available"),
        ("model_training", "unimol-train-v1", ["gpu", "unimol"], (1, 4, 1200), "available"),
    ],
)
def test_configured_authority_enables_exact_non_dispatched_authorization_chain(
    tmp_path: Path,
    task_type: str,
    profile_id: str,
    capabilities: list[str],
    resources: tuple[int, int, int],
    cuda_status: str,
) -> None:
    (
        storage,
        _,
        _,
        _,
        proposal,
        resource_service,
        authorization_service,
    ) = _configured_case(
        tmp_path,
        task_type=task_type,
        profile_id=profile_id,
        capabilities=capabilities,
        resources=resources,
        cuda_status=cuda_status,
    )
    proposal_path = (
        storage.projects_root
        / "project-1"
        / "agent_plan_proposals"
        / proposal.proposal_id
        / "proposal.json"
    )
    before = proposal_path.read_bytes()
    created = resource_service.publish(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id="resource-request",
        ),
    )
    assert created.decision.outcome == AgentRemoteResourceAuthorityOutcome.CONFIGURED
    assert len(created.authorities) == 1
    assert created.authority_set.remote_task_ids == ["remote_task"]
    assert created.authority_set.aggregate_budget.walltime_aggregation_policy == (
        "sequential_sum.v1"
    )
    assert created.authority_set.aggregate_budget.total_derived_gpu_hours == (
        resources[0] * resources[2] / 3600
    )
    assert created.authority_set.executable is False
    authority = created.authorities[0]
    assert authority.configured_resources.model_dump() == {
        "gpu_count": resources[0],
        "cpu_threads": resources[1],
        "walltime_sec": resources[2],
    }
    assert authority.executable is False
    assert proposal_path.read_bytes() == before
    assert not list(storage.projects_root.rglob("remote_execution_request.json"))

    permission = authorization_service.evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert permission.outcome == AgentPermissionOutcome.REQUIRE_APPROVAL
    assert permission.policy_version == "scientific-agent-permission-policy.v2"
    request = AgentPlanAuthorizationRequest(
        expected_proposal_digest=proposal.proposal_digest,
        authorization_mode=AgentAuthorizationMode.STEPWISE,
        requested_preauthorized_gate_ids=[],
        confirmed=True,
        client_request_id="approve-start-request",
        note="authorize exact remote plan",
    )
    result = authorization_service.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=request,
        actor="owner",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert result.start_intent.dispatch_state == "not_dispatched"
    assert result.start_intent.executable is False
    authorization_service.verify_authorization(
        project_id="project-1",
        authorization_id=result.authorization.authorization_id,
        verify_current=True,
    )
    authorization_service.verify_start_intent(
        project_id="project-1",
        start_intent_id=result.start_intent.start_intent_id,
        verify_current=True,
    )


def test_remote_permission_without_published_authority_denies(tmp_path: Path) -> None:
    _, _, _, _, proposal, _, authorization_service = _configured_case(tmp_path)
    decision = authorization_service.evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert decision.outcome == AgentPermissionOutcome.DENY
    assert "REMOTE_RESOURCE_AUTHORITY_REQUIRED" in decision.reason_codes


@pytest.mark.parametrize(
    ("initial_budget_dimensions", "changed_budget_dimensions"),
    [
        (["max_runtime_sec"], []),
        ([], ["max_runtime_sec"]),
    ],
)
def test_hidden_local_budget_dimension_drift_invalidates_authorization_and_start(
    tmp_path: Path,
    initial_budget_dimensions: list[str],
    changed_budget_dimensions: list[str],
) -> None:
    (
        _,
        _,
        _,
        proposal_store,
        proposal,
        resource_service,
        authorization_service,
    ) = _configured_case(
        tmp_path,
        hidden_budget_dimensions=initial_budget_dimensions,
        legacy_runtime_limit=600,
    )
    resource_service.publish(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id="hidden-budget-resource-request",
        ),
    )
    result = authorization_service.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="hidden-budget-approve-start-request",
            note="authorize the exact hidden budget contract",
        ),
        actor="owner",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert result.start_intent.dispatch_state == "not_dispatched"
    before = result.authorization.task_authority_digests["hidden_prepare_task"]

    hidden = proposal_store.registry.get("hidden_prepare_task")
    hidden.budget_dimensions = changed_budget_dimensions
    changed = authorization_service.permission_engine.evaluate(
        publication=proposal_store.read(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            verify_current=True,
        ),
        phase=AgentPermissionPhase.AUTHORIZATION_CANDIDATE,
        expected_proposal_digest=proposal.proposal_digest,
        authorization_mode=AgentAuthorizationMode.STEPWISE,
        actor="owner",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        client_request_id=result.authorization.client_request_id,
    )
    changed_hidden = next(
        item
        for item in changed.task_decisions
        if item.task_id == "hidden_prepare_task"
    )
    assert changed.outcome == AgentPermissionOutcome.REQUIRE_APPROVAL
    assert changed_hidden.task_authority_digest != before
    with pytest.raises(ScientificAgentAuthorizationVerificationError):
        authorization_service.verify_authorization(
            project_id="project-1",
            authorization_id=result.authorization.authorization_id,
            verify_current=True,
        )
    with pytest.raises(ScientificAgentAuthorizationVerificationError):
        authorization_service.verify_start_intent(
            project_id="project-1",
            start_intent_id=result.start_intent.start_intent_id,
            verify_current=True,
        )


@pytest.mark.parametrize(
    ("hidden_budget_dimensions", "expected_reason"),
    [
        (None, "INTERNAL_TASK_PERMISSION_METADATA_INCOMPLETE"),
        (["unknown_budget_dimension"], "TASK_BUDGET_DIMENSION_UNKNOWN"),
    ],
)
def test_resource_aware_hidden_budget_contract_is_explicit_and_recognized(
    tmp_path: Path,
    hidden_budget_dimensions: list[str] | None,
    expected_reason: str,
) -> None:
    _, _, _, _, proposal, resource_service, authorization_service = _configured_case(
        tmp_path,
        hidden_budget_dimensions=hidden_budget_dimensions,
        legacy_runtime_limit=600,
    )
    resource_service.publish(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id="hidden-budget-contract-resource-request",
        ),
    )
    decision = authorization_service.evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert decision.outcome == AgentPermissionOutcome.DENY
    assert expected_reason in decision.reason_codes


def test_resource_ceiling_device_probe_budget_and_cost_rules_fail_closed(
    tmp_path: Path,
) -> None:
    _, _, policy_store, _, proposal, service, _ = _configured_case(tmp_path)
    original = policy_store.read().entries[0]
    request = AgentRemoteResourceAuthorityRequest(
        expected_proposal_digest=proposal.proposal_digest,
        client_request_id="resource-request",
    )
    cases = [
        (
            original.model_copy(
                update={
                    "configured_resources": original.configured_resources.model_copy(
                        update={"cpu_threads": 2}
                    )
                }
            ),
            "REMOTE_RESOURCE_LIMIT_EXCEEDED",
        ),
        (
            original.model_copy(
                update={
                    "configured_resources": original.configured_resources.model_copy(
                        update={"gpu_count": 1}
                    )
                }
            ),
            "REMOTE_RESOURCE_DEVICE_POLICY_MISMATCH",
        ),
        (
            original.model_copy(
                update={
                    "budget_limits": original.budget_limits.model_copy(
                        update={"max_runtime_sec": 500}
                    )
                }
            ),
            "REMOTE_RESOURCE_BUDGET_EXCEEDED",
        ),
        (
            original.model_copy(
                update={
                    "budget_limits": original.budget_limits.model_copy(
                        update={"max_cost_usd": 1.0}
                    )
                }
            ),
            "REMOTE_RESOURCE_COST_AUTHORITY_UNAVAILABLE",
        ),
    ]
    for entry, expected_reason in cases:
        policy_store.save(RemoteResourceAuthorityPolicy(entries=[entry]))
        decision = service.evaluate(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=request,
            publish_decision=False,
        ).decision
        assert decision.outcome == AgentRemoteResourceAuthorityOutcome.DENY
        assert expected_reason in decision.reason_codes


def test_gpu_probe_must_verify_gpu_and_available_cuda(tmp_path: Path) -> None:
    _, _, _, _, proposal, service, _ = _configured_case(
        tmp_path,
        task_type="model_training",
        profile_id="unimol-train-v1",
        capabilities=["gpu", "unimol"],
        resources=(1, 4, 1200),
        cuda_status="unavailable",
    )
    decision = service.evaluate(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id="resource-request",
        ),
        publish_decision=False,
    ).decision
    assert decision.outcome == AgentRemoteResourceAuthorityOutcome.DENY
    assert "REMOTE_RESOURCE_CAPABILITY_MISSING" in decision.reason_codes


def test_missing_disabled_ambiguous_and_task_not_allowed_policy_fail_closed(
    tmp_path: Path,
) -> None:
    _, _, policy_store, _, proposal, service, _ = _configured_case(tmp_path)
    request = AgentRemoteResourceAuthorityRequest(
        expected_proposal_digest=proposal.proposal_digest,
        client_request_id="resource-request",
    )
    original = policy_store.read().entries[0]
    cases = [
        (RemoteResourceAuthorityPolicy(entries=[]), "REMOTE_RESOURCE_POLICY_MISSING"),
        (
            RemoteResourceAuthorityPolicy(entries=[original.model_copy(update={"enabled": False})]),
            "REMOTE_RESOURCE_POLICY_DISABLED",
        ),
        (
            RemoteResourceAuthorityPolicy(
                entries=[
                    original,
                    original.model_copy(update={"policy_id": "second-policy"}),
                ]
            ),
            "REMOTE_RESOURCE_POLICY_AMBIGUOUS",
        ),
        (
            RemoteResourceAuthorityPolicy(
                entries=[original.model_copy(update={"allowed_task_ids": ["another-task"]})]
            ),
            "REMOTE_RESOURCE_TASK_NOT_ALLOWED",
        ),
    ]
    for policy, reason in cases:
        policy_store.save(policy)
        decision = service.evaluate(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=request,
            publish_decision=False,
        ).decision
        assert decision.outcome == AgentRemoteResourceAuthorityOutcome.DENY
        assert reason in decision.reason_codes


def test_private_policy_is_strict_private_atomic_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    store = RemoteResourceAuthorityPolicyStore(config_dir=tmp_path / "config")
    policy = RemoteResourceAuthorityPolicy(entries=[])
    store.save(policy)
    assert stat.S_IMODE(store.config_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert store.read() == policy
    with pytest.raises(ValidationError):
        RemoteResourceAuthorityPolicy.model_validate(
            policy.model_dump(mode="json") | {"ssh_host_alias": "injected"}
        )

    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    store.path.unlink()
    store.path.symlink_to(external)
    with pytest.raises(ValueError, match="non-symlink"):
        store.read()


@pytest.mark.parametrize(
    "resources",
    [
        {"gpu_count": None, "cpu_threads": 1, "walltime_sec": 1},
        {"gpu_count": -1, "cpu_threads": 1, "walltime_sec": 1},
        {"gpu_count": True, "cpu_threads": 1, "walltime_sec": 1},
        {"gpu_count": 0.0, "cpu_threads": 1, "walltime_sec": 1},
        {"gpu_count": 0, "cpu_threads": "1", "walltime_sec": 1},
    ],
)
def test_configured_resources_reject_null_negative_bool_float_and_string(
    resources: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        AgentConfiguredRemoteResources.model_validate(resources)


def test_same_request_replays_and_different_payload_conflicts(tmp_path: Path) -> None:
    _, _, _, _, proposal, service, _ = _configured_case(tmp_path)
    request = AgentRemoteResourceAuthorityRequest(
        expected_proposal_digest=proposal.proposal_digest,
        client_request_id="same-request",
    )
    first = service.publish(
        project_id="project-1", proposal_id=proposal.proposal_id, request=request
    )
    replay = service.publish(
        project_id="project-1", proposal_id=proposal.proposal_id, request=request
    )
    assert replay == first
    with pytest.raises(RemoteResourceAuthorityConflict):
        service.publish(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=request.model_copy(
                update={"expected_proposal_digest": "sha256:" + "0" * 64}
            ),
        )


def test_policy_and_probe_drift_stale_authorization_and_start_intent(
    tmp_path: Path,
) -> None:
    _, profiles, policy_store, _, proposal, service, authorization_service = _configured_case(
        tmp_path
    )
    service.publish(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id="resource-request",
        ),
    )
    result = authorization_service.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode="stepwise",
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="start-request",
            note="",
        ),
        actor="owner",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    original_policy = policy_store.read()
    entry = original_policy.entries[0]
    policy_store.save(
        RemoteResourceAuthorityPolicy(
            entries=[
                entry.model_copy(
                    update={
                        "configured_resources": entry.configured_resources.model_copy(
                            update={"cpu_threads": 1, "walltime_sec": 500}
                        )
                    }
                )
            ]
        )
    )
    with pytest.raises(
        (ScientificAgentAuthorizationVerificationError, ScientificAgentPlanSourceChanged)
    ):
        authorization_service.verify_authorization(
            project_id="project-1",
            authorization_id=result.authorization.authorization_id,
            verify_current=True,
        )
    with pytest.raises(ScientificAgentAuthorizationVerificationError):
        authorization_service.verify_start_intent(
            project_id="project-1",
            start_intent_id=result.start_intent.start_intent_id,
            verify_current=True,
        )

    # A nonempty-to-nonempty current probe drift also invalidates the proposal
    # source binding; no capability fragment can be borrowed from another node.
    policy_store.save(original_policy)
    connection = profiles.get_connection("scientific-worker")
    profiles.save_probe(
        CapabilityProbeResult(
            connection_id=connection.connection_id,
            connection_profile_digest=connection.digest(),
            status="available",
            checked_at="2026-08-01T00:01:00Z",
            verified_capabilities=["cpu", "reinvent4"],
            details=CapabilityDetails(cpu_threads=16),
        )
    )
    with pytest.raises(
        (ScientificAgentAuthorizationVerificationError, ScientificAgentPlanSourceChanged)
    ):
        authorization_service.verify_authorization(
            project_id="project-1",
            authorization_id=result.authorization.authorization_id,
            verify_current=True,
        )


def test_fault_after_authority_rename_leaves_no_success_marker_and_recovers(
    tmp_path: Path,
) -> None:
    (
        storage,
        profiles,
        policy_store,
        proposal_store,
        proposal,
        _,
        authorization_service,
    ) = _configured_case(tmp_path)
    control = AgentPlanControlStore(storage=storage)
    failed = False

    def fault(phase: str) -> None:
        nonlocal failed
        if phase == "after_remote_authority_1" and not failed:
            failed = True
            raise RuntimeError("simulated crash")

    crashing = RemoteResourceAuthorityService(
        proposal_store=proposal_store,
        resource_profiles=profiles,
        policy_store=policy_store,
        control_store=control,
        fault_injector=fault,
    )
    request = AgentRemoteResourceAuthorityRequest(
        expected_proposal_digest=proposal.proposal_digest,
        client_request_id="crash-request",
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        crashing.publish(
            project_id="project-1", proposal_id=proposal.proposal_id, request=request
        )
    request_dir = (
        storage.projects_root
        / "project-1"
        / "agent_plan_control"
        / "remote_resource_authority_requests"
        / "crash-request"
    )
    assert not (request_dir / "authorities_committed.json").exists()
    authority_sets_root = (
        storage.projects_root
        / "project-1"
        / "agent_plan_control"
        / "remote_resource_authority_sets"
    )
    assert not authority_sets_root.exists()
    permission = authorization_service.evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert permission.outcome == AgentPermissionOutcome.DENY
    assert "REMOTE_RESOURCE_AUTHORITY_REQUIRED" in permission.reason_codes
    recovered = RemoteResourceAuthorityService(
        proposal_store=proposal_store,
        resource_profiles=profiles,
        policy_store=policy_store,
        control_store=control,
    ).publish(
        project_id="project-1", proposal_id=proposal.proposal_id, request=request
    )
    assert len(recovered.authorities) == 1
    assert recovered.authority_set.remote_task_ids == ["remote_task"]
    marker = json.loads((request_dir / "authorities_committed.json").read_text())
    assert marker["status"] == "AUTHORITIES_COMMITTED"


def test_permission_rejects_raw_authority_without_complete_set_binding(
    tmp_path: Path,
) -> None:
    _, _, _, proposal_store, proposal, service, _ = _configured_case(tmp_path)
    result = service.publish(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id="raw-authority-request",
        ),
    )
    publication = proposal_store.read(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        verify_current=True,
    )
    decision = ScientificAgentPermissionEngine(
        registry=AtomicTaskRegistry(
            [_remote_task("molecular_generation", "reinvent4-cpu-v1")]
        ),
        resource_authority_resolver=lambda _publication, _task_id: result.authorities[0],
    ).evaluate(
        publication=publication,
        phase=AgentPermissionPhase.PROPOSAL_REVIEW,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert decision.outcome == AgentPermissionOutcome.DENY
    assert "REMOTE_RESOURCE_AUTHORITY_REQUIRED" in decision.reason_codes


@pytest.mark.parametrize("fault_phase", ["after_remote_authority_1", "after_remote_authority_2"])
def test_incomplete_multi_remote_roster_is_inert_until_authority_set_publication(
    tmp_path: Path,
    fault_phase: str,
) -> None:
    (
        storage,
        profiles,
        policy_store,
        proposal_store,
        proposal,
        _,
        authorization_service,
    ) = _multi_remote_case(tmp_path)

    def fault(phase: str) -> None:
        if phase == fault_phase:
            raise RuntimeError("simulated crash before complete authority set")

    service = RemoteResourceAuthorityService(
        proposal_store=proposal_store,
        resource_profiles=profiles,
        policy_store=policy_store,
        control_store=AgentPlanControlStore(storage=storage),
        fault_injector=fault,
    )
    request = AgentRemoteResourceAuthorityRequest(
        expected_proposal_digest=proposal.proposal_digest,
        client_request_id="incomplete-roster-request",
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        service.publish(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=request,
        )
    assert not (
        storage.projects_root
        / "project-1"
        / "agent_plan_control"
        / "remote_resource_authority_sets"
    ).exists()
    permission = authorization_service.evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert permission.outcome == AgentPermissionOutcome.DENY
    assert "REMOTE_RESOURCE_AUTHORITY_REQUIRED" in permission.reason_codes


def test_multi_remote_roster_and_publication_follow_run_plan_order(tmp_path: Path) -> None:
    _, _, _, _, proposal, service, authorization_service = _multi_remote_case(tmp_path)
    assert [item.task_id for item in proposal.run_plan.tasks] == [
        "train_model",
        "generate_candidates",
    ]
    assert [item.task_id for item in proposal.dispatch_intents] == [
        "generate_candidates",
        "train_model",
    ]
    result = service.publish(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id="ordered-roster-request",
        ),
    )
    assert result.decision.remote_task_ids == ["train_model", "generate_candidates"]
    assert [item.task_id for item in result.decision.task_decisions] == [
        "train_model",
        "generate_candidates",
    ]
    assert [item.task_id for item in result.authorities] == [
        "train_model",
        "generate_candidates",
    ]
    assert result.authority_set.remote_task_ids == [
        "train_model",
        "generate_candidates",
    ]
    assert [item.task_id for item in result.authority_set.authority_bindings] == [
        "train_model",
        "generate_candidates",
    ]
    permission = authorization_service.evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert permission.outcome == AgentPermissionOutcome.REQUIRE_APPROVAL


def test_aggregate_remote_gpu_hours_fail_closed_at_plan_level(tmp_path: Path) -> None:
    _, _, _, _, proposal, service, _ = _multi_remote_case(
        tmp_path,
        resources=(1, 2, 2160),
        max_runtime_sec=5000,
        max_gpu_hours=1.0,
    )
    evaluation = service.evaluate(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id="aggregate-budget-request",
        ),
        publish_decision=False,
    )
    assert evaluation.decision.outcome == AgentRemoteResourceAuthorityOutcome.DENY
    assert "REMOTE_RESOURCE_AGGREGATE_BUDGET_EXCEEDED" in (
        evaluation.decision.reason_codes
    )
    assert evaluation.decision.aggregate_budget.total_derived_gpu_hours == pytest.approx(
        1.2
    )
    assert evaluation.authority_set is None


def test_policy_change_at_success_boundary_leaves_only_stale_audit_set(
    tmp_path: Path,
) -> None:
    (
        storage,
        profiles,
        policy_store,
        proposal_store,
        proposal,
        _,
        authorization_service,
    ) = _configured_case(tmp_path)
    changed = False

    def fault(phase: str) -> None:
        nonlocal changed
        if phase == "before_authorities_committed_marker" and not changed:
            changed = True
            entry = policy_store.read().entries[0]
            policy_store.save(
                RemoteResourceAuthorityPolicy(
                    entries=[
                        entry.model_copy(
                            update={
                                "configured_resources": entry.configured_resources.model_copy(
                                    update={"walltime_sec": 500}
                                )
                            }
                        )
                    ]
                )
            )

    service = RemoteResourceAuthorityService(
        proposal_store=proposal_store,
        resource_profiles=profiles,
        policy_store=policy_store,
        control_store=AgentPlanControlStore(storage=storage),
        fault_injector=fault,
    )
    with pytest.raises(RemoteResourceAuthorityStale):
        service.publish(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=AgentRemoteResourceAuthorityRequest(
                expected_proposal_digest=proposal.proposal_digest,
                client_request_id="success-boundary-request",
            ),
        )
    request_dir = (
        storage.projects_root
        / "project-1"
        / "agent_plan_control"
        / "remote_resource_authority_requests"
        / "success-boundary-request"
    )
    assert not (request_dir / "authorities_committed.json").exists()
    assert list(
        (
            storage.projects_root
            / "project-1"
            / "agent_plan_control"
            / "remote_resource_authority_sets"
        ).iterdir()
    )
    permission = authorization_service.evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert permission.outcome == AgentPermissionOutcome.DENY


def test_probe_change_after_set_rename_prevents_success_marker_and_response(
    tmp_path: Path,
) -> None:
    storage, profiles, policy_store, proposal_store, proposal, _, _ = _configured_case(
        tmp_path
    )
    changed = False

    def fault(phase: str) -> None:
        nonlocal changed
        if phase == "after_authority_set_publication" and not changed:
            changed = True
            connection = profiles.get_connection("scientific-worker")
            profiles.save_probe(
                CapabilityProbeResult(
                    connection_id=connection.connection_id,
                    connection_profile_digest=connection.digest(),
                    status="available",
                    checked_at="2026-08-01T00:01:00Z",
                    verified_capabilities=["cpu", "reinvent4"],
                    details=CapabilityDetails(cpu_threads=16),
                )
            )

    service = RemoteResourceAuthorityService(
        proposal_store=proposal_store,
        resource_profiles=profiles,
        policy_store=policy_store,
        control_store=AgentPlanControlStore(storage=storage),
        fault_injector=fault,
    )
    with pytest.raises((RemoteResourceAuthorityStale, ScientificAgentPlanSourceChanged)):
        service.publish(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=AgentRemoteResourceAuthorityRequest(
                expected_proposal_digest=proposal.proposal_digest,
                client_request_id="post-set-probe-drift-request",
            ),
        )
    request_dir = (
        storage.projects_root
        / "project-1"
        / "agent_plan_control"
        / "remote_resource_authority_requests"
        / "post-set-probe-drift-request"
    )
    assert not (request_dir / "authorities_committed.json").exists()
    assert list(
        (
            storage.projects_root
            / "project-1"
            / "agent_plan_control"
            / "remote_resource_authority_sets"
        ).iterdir()
    )


@pytest.mark.parametrize("workflow", ["unimol", "mineru"])
def test_default_registry_mixed_remote_chain_uses_resource_set_budget_authority(
    tmp_path: Path,
    workflow: str,
) -> None:
    _, proposal, service, authorization_service = _default_registry_mixed_case(
        tmp_path, workflow=workflow
    )
    routes = [item.execution_route for item in proposal.dispatch_intents]
    assert "local_executor" in routes
    assert "remote_execution_service" in routes
    result = service.publish(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id=f"{workflow}-mixed-resource-request",
        ),
    )
    assert result.authority_set.aggregate_budget.total_derived_gpu_hours > 0
    permission = authorization_service.evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert permission.outcome == AgentPermissionOutcome.REQUIRE_APPROVAL
    assert "BUDGET_AUTHORITY_UNAVAILABLE" not in permission.reason_codes


def test_reinvent_mixed_chain_requires_legacy_local_runtime_authority(
    tmp_path: Path,
) -> None:
    _, proposal, service, authorization_service = _default_registry_mixed_case(
        tmp_path,
        workflow="reinvent",
    )
    dispatch_by_task = {item.task_id: item for item in proposal.dispatch_intents}
    assert dispatch_by_task["generate_candidates"].execution_route == (
        "remote_execution_service"
    )
    assert dispatch_by_task["predict_candidates"].execution_route == "local_executor"
    result = service.publish(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id="reinvent-mixed-resource-request",
        ),
    )
    assert result.authority_set.aggregate_budget.total_walltime_upper_bound_sec == 600
    permission = authorization_service.evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert permission.outcome == AgentPermissionOutcome.DENY
    assert "MIXED_PLAN_RUNTIME_AUTHORITY_REQUIRED" in permission.reason_codes
    assert "REMOTE_RESOURCE_AUTHORITY_REQUIRED" not in permission.reason_codes


def test_reinvent_mixed_chain_accepts_exact_configured_legacy_runtime_authority(
    tmp_path: Path,
) -> None:
    _, proposal, service, authorization_service = _default_registry_mixed_case(
        tmp_path,
        workflow="reinvent",
        legacy_runtime_limit=600,
    )
    service.publish(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id="reinvent-configured-runtime-request",
        ),
    )
    permission = authorization_service.evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert permission.outcome == AgentPermissionOutcome.REQUIRE_APPROVAL
    assert "MIXED_PLAN_RUNTIME_AUTHORITY_REQUIRED" not in permission.reason_codes
    assert "BUDGET_AUTHORITY_UNAVAILABLE" not in permission.reason_codes


def test_policy_change_after_candidate_is_rejected_before_commit(tmp_path: Path) -> None:
    storage, profiles, policy_store, proposal_store, proposal, _, _ = _configured_case(
        tmp_path
    )
    changed = False

    def fault(phase: str) -> None:
        nonlocal changed
        if phase == "after_resource_decision" and not changed:
            changed = True
            original = policy_store.read().entries[0]
            policy_store.save(
                RemoteResourceAuthorityPolicy(
                    entries=[
                        original.model_copy(
                            update={
                                "configured_resources": original.configured_resources.model_copy(
                                    update={"walltime_sec": 500}
                                )
                            }
                        )
                    ]
                )
            )

    service = RemoteResourceAuthorityService(
        proposal_store=proposal_store,
        resource_profiles=profiles,
        policy_store=policy_store,
        control_store=AgentPlanControlStore(storage=storage),
        fault_injector=fault,
    )
    with pytest.raises(RemoteResourceAuthorityStale):
        service.publish(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=AgentRemoteResourceAuthorityRequest(
                expected_proposal_digest=proposal.proposal_digest,
                client_request_id="source-drift-request",
            ),
        )
    request_dir = (
        storage.projects_root
        / "project-1"
        / "agent_plan_control"
        / "remote_resource_authority_requests"
        / "source-drift-request"
    )
    assert not (request_dir / "decision_committed.json").exists()
    assert not (request_dir / "authorities_committed.json").exists()


def test_cross_process_same_request_publishes_one_exact_roster(tmp_path: Path) -> None:
    storage, _, policy_store, _, proposal, _, _ = _configured_case(tmp_path)
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_multiprocess_publish_worker,
            args=(
                str(storage.workspace_dir),
                str(policy_store.config_dir),
                proposal.proposal_id,
                proposal.proposal_digest,
                start_event,
                result_queue,
            ),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    results = [result_queue.get(timeout=2) for _ in processes]
    assert [item[0] for item in results] == ["success", "success"]
    assert len({item[1] for item in results}) == 1
    publications = list(
        (
            storage.projects_root
            / "project-1"
            / "agent_plan_control"
            / "remote_resource_authorities"
        ).iterdir()
    )
    assert len(publications) == 1
    authority_sets = list(
        (
            storage.projects_root
            / "project-1"
            / "agent_plan_control"
            / "remote_resource_authority_sets"
        ).iterdir()
    )
    assert len(authority_sets) == 1


def test_client_resource_injection_is_rejected_by_frozen_request_schema() -> None:
    with pytest.raises(ValidationError):
        AgentRemoteResourceAuthorityRequest.model_validate(
            {
                "expected_proposal_digest": "sha256:" + "1" * 64,
                "client_request_id": "request-1",
                "gpu_count": 1,
                "connection_id": "client-selected",
                "command": "run",
            }
        )


@pytest.mark.parametrize("enabled", [1, 0, "true", "false"])
def test_private_policy_enabled_requires_a_strict_boolean(enabled: object) -> None:
    with pytest.raises(ValidationError):
        RemoteResourceAuthorityPolicyEntry.model_validate(
            {
                "policy_id": "strict-policy",
                "enabled": enabled,
                "connection_id": "worker",
                "execution_profile_id": "reinvent4-cpu-v1",
                "remote_task_type": "molecular_generation",
                "allowed_task_ids": ["remote_task"],
                "configured_resources": {
                    "gpu_count": 0,
                    "cpu_threads": 1,
                    "walltime_sec": 600,
                },
                "budget_limits": {
                    "max_runtime_sec": 600,
                    "max_gpu_hours": 1.0,
                    "max_cost_usd": None,
                },
            }
        )


def test_authority_set_schema_rejects_missing_or_replaced_complete_binding(
    tmp_path: Path,
) -> None:
    _, _, _, _, proposal, service, _ = _configured_case(tmp_path)
    result = service.publish(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id="set-schema-request",
        ),
    )
    payload = result.authority_set.model_dump(mode="json")
    payload["authority_bindings"] = []
    with pytest.raises(ValidationError):
        AgentRemoteResourceAuthoritySet.model_validate(payload)

    payload = result.authority_set.model_dump(mode="json")
    payload["authority_bindings"][0]["authority_digest"] = "sha256:" + "9" * 64
    with pytest.raises(ValidationError):
        AgentRemoteResourceAuthoritySet.model_validate(payload)


def test_frozen_remote_resource_authority_schemas_match_generated_models() -> None:
    schema_dir = Path(__file__).resolve().parents[1] / "docs" / "schemas"
    models = {
        "agent_remote_resource_authority_request": AgentRemoteResourceAuthorityRequest,
        "agent_remote_resource_authority_decision": AgentRemoteResourceAuthorityDecision,
        "agent_remote_resource_authority": AgentRemoteResourceAuthority,
        "agent_remote_resource_authority_set": AgentRemoteResourceAuthoritySet,
        "remote_resource_authority_policy": RemoteResourceAuthorityPolicy,
    }
    for name, model in models.items():
        frozen = json.loads(
            (schema_dir / f"{name}.schema.json").read_text(encoding="utf-8")
        )
        assert frozen == model.model_json_schema()


def test_resource_aware_permission_policy_digest_is_hash_seed_stable() -> None:
    assert PERMISSION_POLICY_DIGEST == (
        "sha256:b47b178b5ed2cd694945d1d55757dc4fa8b5b7f072ec69f16df1333473a357bb"
    )
    assert RESOURCE_AWARE_PERMISSION_POLICY_DIGEST == (
        "sha256:2b4934ed64e402c1deff4d80ca691879974a3ef0ceffebbd7056bfd591fe91bd"
    )
    script = (
        "from ai4s_agent.scientific_agent_permissions import "
        "RESOURCE_AWARE_PERMISSION_POLICY_DIGEST; "
        "print(RESOURCE_AWARE_PERMISSION_POLICY_DIGEST)"
    )
    values = []
    for seed in ("1", "777"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        values.append(completed.stdout.strip())
    assert len(set(values)) == 1


def test_resource_authority_api_rejects_injection_and_local_evaluation_is_empty(
    tmp_path: Path,
) -> None:
    from ai4s_agent.app import create_app

    workspace = tmp_path / "api-workspace"
    storage = ProjectStorage(workspace_dir=workspace)
    storage.create_project("project-1", name="Project", created_at=NOW)
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=workspace,
        user_config_dir=tmp_path / "api-config",
    )
    client = app.test_client()
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["generate_candidates"],
        selected_input_artifact_ids=[],
        task_options={"generate_candidates": {}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop"],
        success_criteria=["review"],
        rationales=["local deterministic stub"],
        assumptions=[],
        questions=[],
    )
    created = client.post(
        "/api/projects/project-1/agent-plan-proposals",
        json={
            "run_id": "run-1",
            "goal": "Build a local review-only proposal",
            "user_constraints": [],
            "client_request_id": "proposal-request",
            "llm_provider": {
                "provider": "stub",
                "model": "stub",
                "stub_response": response.model_dump(mode="json"),
            },
        },
    )
    assert created.status_code == 200
    proposal = created.json["proposal"]
    endpoint = (
        f"/api/projects/project-1/agent-plan-proposals/{proposal['proposal_id']}"
        "/remote-resource-authority-evaluations"
    )
    injected = client.post(
        endpoint,
        json={
            "expected_proposal_digest": proposal["proposal_digest"],
            "client_request_id": "resource-request",
            "gpu_count": 1,
        },
    )
    assert injected.status_code == 400
    assert injected.json["reason_codes"] == ["REMOTE_RESOURCE_CLIENT_INJECTION"]
    evaluated = client.post(
        endpoint,
        json={
            "expected_proposal_digest": proposal["proposal_digest"],
            "client_request_id": "resource-request",
        },
    )
    assert evaluated.status_code == 200
    assert evaluated.json["outcome"] == "CONFIGURED"
    assert evaluated.json["authority_ids"] == []
    assert evaluated.json["reason_codes"] == [
        "REMOTE_RESOURCE_AUTHORITY_NOT_REQUIRED"
    ]
    assert evaluated.json["dispatched"] is False


def test_resource_authority_and_authorization_make_no_execution_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai4s_agent.executor import RunPlanExecutor
    from ai4s_agent.remote_execution_lifecycle import PinnedWorkerTransport
    from ai4s_agent.remote_execution_service import DescriptorRemoteExecutionLifecycleService

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("execution authority must not be called")

    for owner, name in (
        (RunPlanExecutor, "execute"),
        (RunPlanExecutor, "resume_after_gate"),
        (DescriptorRemoteExecutionLifecycleService, "prepare"),
        (DescriptorRemoteExecutionLifecycleService, "approve"),
        (DescriptorRemoteExecutionLifecycleService, "refresh"),
        (DescriptorRemoteExecutionLifecycleService, "recover"),
        (DescriptorRemoteExecutionLifecycleService, "cancel"),
        (PinnedWorkerTransport, "dispatch"),
        (ProjectStorage, "append_gate_decision"),
        (ProjectStorage, "write_stage_state"),
    ):
        monkeypatch.setattr(owner, name, forbidden)

    storage, _, _, _, proposal, service, authorization_service = _configured_case(
        tmp_path
    )
    service.publish(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id="resource-request",
        ),
    )
    result = authorization_service.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode="stepwise",
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="approve-request",
            note="",
        ),
        actor="owner",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert result.start_intent.dispatch_state == "not_dispatched"
    assert not list(storage.projects_root.rglob("execution_request.json"))
    assert not list(storage.projects_root.rglob("stage_state.json"))
    assert not list(storage.projects_root.rglob("gate_decisions.json"))
