from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.llm_provider import LLMProviderError
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.resource_profiles import (
    CapabilityProbeResult,
    ConnectionProfile,
    ResourceProfileStore,
)
from ai4s_agent.schemas import (
    AgentExecutionPlanLLMResponse,
    AgentLLMInvocationMetadata,
    ArtifactRef,
    AtomicTaskSpec,
    PlannedTask,
    RunPlan,
    RunStatus,
    ScientificToolSpec,
    StageState,
)
from ai4s_agent.scientific_agent_plan import (
    AgentExecutionPlanCompiler,
    AgentProjectObservationBuilder,
    PlannerOptionCompiler,
    ScientificAgentPlanError,
    ScientificAgentPlanPublicationConflict,
    ScientificAgentPlanRecoveryRequired,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
    ScientificAgentPlanSourceChanged,
    build_scientific_tool_catalog,
)
from ai4s_agent.storage import ProjectStorage


def _clock() -> str:
    return "2026-07-31T00:00:00Z"


def _private_path_canary() -> str:
    return str(Path("/") / "Users" / "example-user" / "private.csv")


def _multiprocess_plan_worker(
    workspace_dir: str,
    counter_path: str,
    *,
    request_id: str,
    goal: str,
    start_event,
    result_queue,
    fault_phase: str = "",
) -> None:
    storage = ProjectStorage(workspace_dir=Path(workspace_dir))

    class ProcessProvider(StubLLMProvider):
        def __init__(self) -> None:
            super().__init__(response=_response().model_dump(mode="json"))

        def complete_json(self, **kwargs):
            descriptor = os.open(
                counter_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                os.write(descriptor, b"llm-call\n")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            time.sleep(0.2)
            return super().complete_json(**kwargs)

    def fault(phase: str) -> None:
        if phase == fault_phase:
            raise RuntimeError(f"simulated crash at {phase}")

    builder = AgentProjectObservationBuilder(storage=storage, clock=_clock)
    store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=builder,
        fault_injector=fault if fault_phase else None,
    )
    service = ScientificAgentPlanService(
        storage=storage,
        observation_builder=builder,
        proposal_store=store,
        clock=_clock,
    )
    start_event.wait()
    try:
        proposal = service.create_proposal(
            project_id="project-1",
            run_id="run-1",
            goal=goal,
            user_constraints=[],
            provider=ProcessProvider(),
            client_request_id=request_id,
        )
    except Exception as exc:  # noqa: BLE001 - process result is asserted by the parent.
        result_queue.put((type(exc).__name__, getattr(exc, "state", "")))
    else:
        result_queue.put(("success", proposal.proposal_id))


def _storage_with_run(tmp_path: Path, *, run_id: str = "run-1") -> tuple[ProjectStorage, Path]:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at=_clock())
    run_dir = storage.run_dir("project-1", run_id)
    return storage, run_dir


def _proposal_dir(storage: ProjectStorage, proposal_id: str) -> Path:
    return storage.projects_root / "project-1" / "agent_plan_proposals" / proposal_id


def _visible_task(
    *,
    task_id: str,
    tool_id: str | None = None,
    required_artifacts: list[str] | None = None,
    output_artifacts: list[str] | None = None,
    option_schema: dict[str, object] | None = None,
) -> AtomicTaskSpec:
    required = required_artifacts or []
    return AtomicTaskSpec(
        task_id=task_id,
        scientific_tool_id=tool_id or task_id,
        required_artifacts=required,
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=output_artifacts or [],
        label=task_id.replace("_", " ").title(),
        description="A review-only logical scientific task.",
        effect_class="compute",
        required_permissions=["derive_project_artifact"],
        option_schema=option_schema
        or {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact={
            artifact_id: ["content_bound_input", "registered_intermediate", "verified_output"]
            for artifact_id in required
        },
        budget_dimensions=[],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    )


def _connection(
    *,
    connection_id: str,
    capabilities: list[str],
    enabled: bool = True,
) -> ConnectionProfile:
    return ConnectionProfile(
        connection_id=connection_id,
        display_name="",
        ssh_host_alias=f"{connection_id}-ssh",
        expected_hostname=connection_id,
        remote_root=f"/srv/{connection_id}",
        declared_capabilities=capabilities,
        enabled=enabled,
    )


def _save_available_probe(store: ResourceProfileStore, connection: ConnectionProfile) -> None:
    store.save_probe(
        CapabilityProbeResult(
            connection_id=connection.connection_id,
            connection_profile_digest=connection.digest(),
            status="available",
            checked_at=_clock(),
            verified_capabilities=connection.declared_capabilities,
        )
    )


def _profile(observation, profile_id: str):
    return next(item for item in observation.logical_execution_profiles if item.profile_id == profile_id)


def _write_content_bound_artifact(
    storage: ProjectStorage,
    run_dir: Path,
    *,
    artifact_id: str,
    relative_path: str,
    content: bytes,
) -> None:
    artifact_path = run_dir / relative_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(content)
    storage.register_artifact_path(
        "project-1",
        run_dir.name,
        artifact_id,
        relative_path,
    )


def _write_confirmed_dataset(storage: ProjectStorage, run_dir: Path, *, canary: bool = False) -> None:
    dataset_path = run_dir / "data" / "confirmed_training_dataset.json"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": "confirmed-dataset",
        "confirmed": True,
        "status": "confirmed",
        "row_count": 4,
        "column_ids": ["smiles", "plqy"],
        "target_property": "plqy",
    }
    if canary:
        payload.update(
            {
                "private_document_text": "private paper text",
                "path": _private_path_canary(),
                "hostname": "cluster.internal",
                "token": "sk-test-canary",
            }
        )
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    storage.register_artifact_path(
        "project-1",
        run_dir.name,
        "confirmed_training_dataset",
        "data/confirmed_training_dataset.json",
    )
    storage.write_stage_state(
        "project-1",
        run_dir.name,
        StageState(
            stage="confirm_extracted_dataset",
            next_stage="train_model",
            status=RunStatus.SUCCEEDED,
            started_at=_clock(),
            updated_at=_clock(),
            artifacts=[
                ArtifactRef(
                    artifact_id="confirmed_training_dataset",
                    relative_path="data/confirmed_training_dataset.json",
                    producer_task_id="confirm_extracted_dataset",
                )
            ],
            details={"executed_tasks": ["confirm_extracted_dataset", "not-a-registered-task"]},
        ),
    )


def _response(tool_id: str = "render_report", **overrides: object) -> AgentExecutionPlanLLMResponse:
    payload: dict[str, object] = {
        "requested_tool_ids": [tool_id],
        "selected_input_artifact_ids": [],
        "task_options": {tool_id: {}},
        "selected_logical_profile_ids": [],
        "limits": {},
        "stop_conditions": ["stop on validation failure"],
        "success_criteria": ["produce a reviewable plan"],
        "rationales": ["The registered logical task is relevant."],
        "assumptions": [],
        "questions": [],
    }
    payload.update(overrides)
    return AgentExecutionPlanLLMResponse.model_validate(payload)


def _invocation(observation, response: AgentExecutionPlanLLMResponse) -> AgentLLMInvocationMetadata:
    digest = "sha256:" + "0" * 64
    return AgentLLMInvocationMetadata(
        provider="stub",
        model="stub",
        prompt_version="scientific-agent-long-horizon-plan.v1",
        response_id="response-1",
        observation_digest=observation.observation_digest,
        tool_catalog_digest=observation.tool_catalog.catalog_digest,
        validated_output_digest=digest,
    )


def _observation(storage: ProjectStorage, *, resource_profiles=None, run_id: str = "run-1"):
    return AgentProjectObservationBuilder(
        storage=storage,
        resource_profiles=resource_profiles,
        clock=_clock,
    ).build(
        project_id="project-1",
        run_id=run_id,
        goal="Prepare a reviewable scientific plan",
        user_constraints=["use confirmed data only"],
    )


def test_catalog_is_registry_projection_and_deterministic() -> None:
    tasks = [
        _visible_task(
            task_id="z_task",
            tool_id="z_tool",
            required_artifacts=["input_artifact"],
            output_artifacts=["output_artifact"],
            option_schema={
                "type": "object",
                "properties": {"top_n": {"type": "integer", "minimum": 1}},
                "required": [],
                "additionalProperties": False,
            },
        ),
        AtomicTaskSpec(task_id="hidden_task", planner_visible=False),
    ]
    first = build_scientific_tool_catalog(AtomicTaskRegistry(tasks))
    second = build_scientific_tool_catalog(AtomicTaskRegistry(list(reversed(tasks))))
    assert first.model_dump_json() == second.model_dump_json()
    assert first.catalog_digest == second.catalog_digest
    assert [tool.task_id for tool in first.tools] == ["z_task"]
    assert first.excluded_task_ids == ["hidden_task"]
    assert "default_adapter" not in first.model_dump_json()
    assert "callable" not in first.model_dump_json()


def test_catalog_is_explicit_opt_in_with_complete_v1_metadata() -> None:
    registry = AtomicTaskRegistry()
    catalog = build_scientific_tool_catalog(registry)
    visible = {task.task_id for task in registry.list_tasks() if task.planner_visible}
    assert visible == {tool.task_id for tool in catalog.tools}
    assert "execute_oled_inverse_design" in catalog.excluded_task_ids
    assert "parse_document_pdfplumber" in catalog.excluded_task_ids
    assert "train_model" in visible
    assert "generate_candidates" in visible
    assert "filter_rank" in visible
    required_metadata = {
        "scientific_tool_id",
        "label",
        "description",
        "effect_class",
        "required_permissions",
        "option_schema",
        "option_compiler_version",
        "logical_profile_requirements",
        "backend_profile_requirements",
        "execution_route",
        "remote_task_type",
        "backend_execution_routes",
        "backend_remote_task_types",
        "optional_input_artifacts",
        "input_artifact_alternatives",
        "accepted_input_trust_classes_by_artifact",
        "budget_dimensions",
        "supports_plan_preapproval",
        "idempotency_policy",
        "verification_policy",
        "planner_visible",
    }
    for task in registry.list_tasks():
        if task.planner_visible:
            assert required_metadata.issubset(task.model_fields_set)
            assert task.option_schema is not None
            assert task.label and task.description and task.effect_class
            assert task.required_permissions
            assert task.option_compiler_version
            assert task.verification_policy
            backend_schema = task.option_schema.get("properties", {}).get("backend", {})
            backend_values = set(backend_schema.get("enum", [])) if isinstance(backend_schema, dict) else set()
            assert set(task.backend_profile_requirements) == backend_values
            if backend_values:
                assert task.default_planner_backend in backend_values
                assert set(task.backend_execution_routes) == backend_values
                assert set(task.backend_remote_task_types) == backend_values
            else:
                assert task.execution_route in {
                    "local_executor",
                    "remote_execution_service",
                }
    assert AtomicTaskSpec(task_id="future_internal_task").planner_visible is False
    with pytest.raises(ValueError, match="explicitly set projection metadata"):
        AtomicTaskSpec(task_id="unsafe_visible_task", planner_visible=True)


def test_catalog_rejects_duplicate_tool_mapping_and_unsafe_option_schema() -> None:
    duplicate = [
        _visible_task(task_id="first_task", tool_id="same_tool"),
        _visible_task(task_id="second_task", tool_id="same_tool"),
    ]
    with pytest.raises(ScientificAgentPlanError):
        build_scientific_tool_catalog(AtomicTaskRegistry(duplicate))

    with pytest.raises(ValueError, match="duplicate atomic task ID"):
        AtomicTaskRegistry([AtomicTaskSpec(task_id="same_task"), AtomicTaskSpec(task_id="same_task")])

    with pytest.raises(ValueError):
        _visible_task(
            task_id="unsafe_task",
            option_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": [],
                "additionalProperties": False,
            },
        )


def test_planning_privacy_checks_allow_normal_oled_prose_and_schema_terms() -> None:
    response = AgentExecutionPlanLLMResponse.model_validate(
        {
            "requested_tool_ids": ["render_report"],
            "selected_input_artifact_ids": [],
            "task_options": {
                "render_report": {
                    "triplet_energy": 2.8,
                    "dipole_moment": 4.2,
                    "description": "OLED host–dopant screening context.",
                }
            },
            "selected_logical_profile_ids": [],
            "limits": {},
            "stop_conditions": ["stop if the previous model failed validation"],
            "success_criteria": ["Optimize PLQY for an OLED host–dopant system"],
            "rationales": ["The authorization review remains pending."],
            "assumptions": ["Triplet energy and dipole moment are relevant properties."],
            "questions": [],
        }
    )
    assert response.success_criteria == ["Optimize PLQY for an OLED host–dopant system"]
    assert response.task_options["render_report"]["triplet_energy"] == 2.8

    tool = ScientificToolSpec(
        tool_id="oled_property_analysis",
        task_id="oled_property_analysis",
        label="OLED property analysis",
        description="Analyze triplet_energy and dipole_moment without execution.",
        input_artifact_ids=[],
        output_artifact_ids=["oled_property_summary"],
        effect_class="observe",
        risk_level="low",
        required_permissions=[],
        required_gates=[],
        option_schema={
            "type": "object",
            "properties": {
                "triplet_energy": {
                    "type": "number",
                    "description": "Triplet energy in eV.",
                },
                "dipole_moment": {
                    "type": "number",
                    "description": "Dipole moment in Debye.",
                },
            },
            "required": [],
            "additionalProperties": False,
            "description": "High-level OLED property constraints.",
        },
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact={},
        budget_dimensions=[],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    )
    assert set(tool.option_schema["properties"]) == {"triplet_energy", "dipole_moment"}

    with pytest.raises(ValidationError):
        AgentExecutionPlanLLMResponse.model_validate(
            {
                "requested_tool_ids": ["render_report"],
                "selected_input_artifact_ids": [],
                "task_options": {"render_report": {}},
                "selected_logical_profile_ids": [],
                "limits": {},
                "stop_conditions": [],
                "success_criteria": [],
                "rationales": ["use sk-test-canary only"],
                "assumptions": [],
                "questions": [],
            }
        )


def test_catalog_canonical_bytes_are_hash_seed_independent() -> None:
    script = (
        "from ai4s_agent.scientific_agent_plan import build_scientific_tool_catalog; "
        "print(build_scientific_tool_catalog().model_dump_json())"
    )
    outputs: list[str] = []
    for seed in ("0", "1", "random"):
        task_env = os.environ.copy()
        task_env["PYTHONHASHSEED"] = seed
        task_env["PYTHONPATH"] = str(Path("src").resolve())
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=task_env,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1] == outputs[2]


def test_observation_is_privacy_safe_and_handles_missing_run(tmp_path: Path) -> None:
    storage, run_dir = _storage_with_run(tmp_path)
    _write_confirmed_dataset(storage, run_dir, canary=True)
    profiles = ResourceProfileStore(
        workspace_dir=tmp_path / "profile-workspace",
        config_dir=tmp_path / "profile-config",
    )
    observation = _observation(storage, resource_profiles=profiles)
    serialized = observation.model_dump_json()
    for canary in (
        _private_path_canary(),
        "cluster.internal",
        "sk-test-canary",
        "private paper text",
    ):
        assert canary not in serialized
    assert observation.current_run_status == "SUCCEEDED"
    assert observation.next_stage == "train_model"
    assert observation.available_artifacts[0].verification_state == "verified"
    assert observation.confirmed_dataset_summaries[0]["row_count"] == 4
    assert all(
        key not in serialized
        for key in ("relative_path", "hostname", "known_hosts", "environment", "stderr")
    )
    assert observation.observation_id.startswith("observation-")

    missing = _observation(storage, run_id="missing-run")
    assert missing.current_run_status == "UNAVAILABLE"
    assert missing.available_artifacts == []


def test_observation_rejects_symlink_artifact(tmp_path: Path) -> None:
    storage, run_dir = _storage_with_run(tmp_path)
    target = run_dir / "outside.json"
    target.write_text("{}", encoding="utf-8")
    link = run_dir / "data.json"
    link.symlink_to(target)
    storage.register_artifact_path("project-1", run_dir.name, "confirmed_dataset", "data.json")
    with pytest.raises(ScientificAgentPlanError):
        _observation(storage)


def test_raw_registered_json_cannot_self_promote_to_confirmed_input(tmp_path: Path) -> None:
    storage, run_dir = _storage_with_run(tmp_path)
    _write_content_bound_artifact(
        storage,
        run_dir,
        artifact_id="confirmed_training_dataset",
        relative_path="inputs/user_claimed_confirmed.json",
        content=json.dumps(
            {
                "dataset_id": "user-claim",
                "confirmed": True,
                "status": "confirmed",
                "row_count": 999,
            }
        ).encode("utf-8"),
    )
    observation = _observation(storage)
    artifact = next(
        item for item in observation.available_artifacts if item.artifact_id == "confirmed_training_dataset"
    )
    assert artifact.verification_state == "registered"
    assert artifact.trust_class == "content_bound_input"
    assert observation.confirmed_dataset_summaries == []


def test_profile_snapshot_requires_one_enabled_digest_matched_connection(tmp_path: Path) -> None:
    storage, _ = _storage_with_run(tmp_path)
    profiles = ResourceProfileStore(
        workspace_dir=tmp_path / "profile-workspace",
        config_dir=tmp_path / "profile-config",
    )
    disabled_capable = _connection(
        connection_id="disabled-unimol",
        capabilities=["unimol", "gpu"],
        enabled=False,
    )
    enabled_unrelated = _connection(
        connection_id="enabled-mineru",
        capabilities=["mineru", "gpu"],
    )
    for connection in (disabled_capable, enabled_unrelated):
        profiles.save_connection(connection)
        _save_available_probe(profiles, connection)
    observation = _observation(storage, resource_profiles=profiles)
    unimol = _profile(observation, "unimol-train-v1")
    assert unimol.availability_state == "unavailable"
    assert unimol.declared_capabilities == []
    assert unimol.verified_capabilities == []

    # Capability fragments on two enabled connections must never be joined
    # into a fictional Uni-Mol+GPU environment.
    profiles = ResourceProfileStore(
        workspace_dir=tmp_path / "split-profile-workspace",
        config_dir=tmp_path / "split-profile-config",
    )
    for connection in (
        _connection(connection_id="unimol-only", capabilities=["unimol"]),
        _connection(connection_id="gpu-only", capabilities=["gpu"]),
    ):
        profiles.save_connection(connection)
        _save_available_probe(profiles, connection)
    observation = _observation(storage, resource_profiles=profiles)
    assert _profile(observation, "unimol-train-v1").availability_state == "unavailable"

    # A probe is stale after its connection digest changes, even when the
    # declared capability set is still sufficient.
    profiles = ResourceProfileStore(
        workspace_dir=tmp_path / "stale-profile-workspace",
        config_dir=tmp_path / "stale-profile-config",
    )
    original = _connection(connection_id="unimol-gpu", capabilities=["unimol", "gpu"])
    profiles.save_connection(original)
    _save_available_probe(profiles, original)
    profiles.save_connection(
        original.model_copy(update={"declared_capabilities": ["cpu", "gpu", "unimol"]})
    )
    observation = _observation(storage, resource_profiles=profiles)
    assert _profile(observation, "unimol-train-v1").availability_state == "stale"

    # A single enabled, digest-matched, successful probe covering the full
    # requirement is the only path to "available".
    profiles = ResourceProfileStore(
        workspace_dir=tmp_path / "available-profile-workspace",
        config_dir=tmp_path / "available-profile-config",
    )
    capable = _connection(connection_id="unimol-gpu", capabilities=["unimol", "gpu"])
    profiles.save_connection(capable)
    _save_available_probe(profiles, capable)
    observation = _observation(storage, resource_profiles=profiles)
    unimol = _profile(observation, "unimol-train-v1")
    assert unimol.availability_state == "available"
    assert unimol.verified_capabilities == ["gpu", "unimol"]
    serialized = observation.model_dump_json()
    assert "unimol-gpu-ssh" not in serialized
    assert "/srv/unimol-gpu" not in serialized


@pytest.mark.parametrize(
    "hostile",
    [
        {"execute": True},
        {"approved": True},
        {"authorization": "approved"},
        {"status": "SUCCEEDED"},
        {"adapter_name": "bad"},
        {"command": "rm -rf"},
        {"ssh": "cluster"},
        {"path": _private_path_canary()},
        {"environment": {"TOKEN": "secret"}},
        {"task_options": {"render_report": {"module": "x"}}},
    ],
)
def test_hostile_llm_response_fails_closed(hostile: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "requested_tool_ids": ["render_report"],
        "selected_input_artifact_ids": [],
        "task_options": {"render_report": {}},
        "selected_logical_profile_ids": [],
        "limits": {},
        "stop_conditions": [],
        "success_criteria": [],
        "rationales": [],
        "assumptions": [],
        "questions": [],
    }
    payload.update(hostile)
    with pytest.raises(ValidationError):
        AgentExecutionPlanLLMResponse.model_validate(payload)


def test_compiler_expands_registry_dependencies_and_never_accepts_llm_dependencies(tmp_path: Path) -> None:
    storage, _ = _storage_with_run(tmp_path)
    observation = _observation(storage)
    response = _response()
    proposal = AgentExecutionPlanCompiler().compile(
        observation=observation,
        response=response,
        invocation=_invocation(observation, response),
        created_at=_clock(),
    )
    task_ids = [task.task_id for task in proposal.run_plan.tasks]
    assert task_ids[-1] == "render_report"
    assert proposal.run_plan.requested_tasks == ["render_report"]
    assert proposal.executable is False
    assert proposal.status == "review_required"
    assert proposal.proposal_id.startswith("proposal-")
    assert "gate_3_train_config" in proposal.required_gates
    assert proposal.missing_artifacts == ["uploaded_dataset"]


def test_compiler_rejects_unknown_artifact_and_profile_mismatch(tmp_path: Path) -> None:
    storage, _ = _storage_with_run(tmp_path)
    observation = _observation(storage)
    response = _response(selected_input_artifact_ids=["not_registered"])
    with pytest.raises(ScientificAgentPlanError):
        AgentExecutionPlanCompiler().compile(
            observation=observation,
            response=response,
            invocation=_invocation(observation, response),
            created_at=_clock(),
        )

    train_response = _response(
        tool_id="train_model",
        selected_input_artifact_ids=[],
        task_options={
            "train_model": {"backend": "unimol", "property_id": "plqy"}
        },
    )
    with pytest.raises(ScientificAgentPlanError, match="selected logical profiles do not satisfy requirement"):
        AgentExecutionPlanCompiler().compile(
            observation=observation,
            response=train_response,
            invocation=_invocation(observation, train_response),
            created_at=_clock(),
        )


def test_backend_options_determine_logical_profile_requirements(tmp_path: Path) -> None:
    storage, _ = _storage_with_run(tmp_path)
    observation = _observation(storage)
    for response in (
        _response(
            tool_id="train_model",
            task_options={
                "train_model": {"backend": "baseline", "property_id": "plqy"}
            },
        ),
        _response(
            tool_id="generate_candidates",
            task_options={
                "generate_candidates": {
                    "backend": "deterministic_stub",
                    "count": 8,
                }
            },
        ),
    ):
        proposal = AgentExecutionPlanCompiler().compile(
            observation=observation,
            response=response,
            invocation=_invocation(observation, response),
            created_at=_clock(),
        )
        assert proposal.selected_profiles == []

    profiles = ResourceProfileStore(
        workspace_dir=tmp_path / "backend-profile-workspace",
        config_dir=tmp_path / "backend-profile-config",
    )
    connection = _connection(
        connection_id="scientific-worker",
        capabilities=["cpu", "gpu", "reinvent4", "unimol"],
    )
    profiles.save_connection(connection)
    _save_available_probe(profiles, connection)
    observation = _observation(storage, resource_profiles=profiles)
    for response, expected_profile in (
        (
            _response(
                tool_id="train_model",
                selected_logical_profile_ids=["unimol-train-v1"],
                task_options={
                    "train_model": {"backend": "unimol", "property_id": "plqy"}
                },
            ),
            "unimol-train-v1",
        ),
        (
            _response(
                tool_id="generate_candidates",
                selected_logical_profile_ids=["reinvent4-cpu-v1"],
                task_options={
                    "generate_candidates": {"backend": "reinvent4", "count": 8}
                },
                limits={"max_runtime_sec": 600},
            ),
            "reinvent4-cpu-v1",
        ),
    ):
        proposal = AgentExecutionPlanCompiler().compile(
            observation=observation,
            response=response,
            invocation=_invocation(observation, response),
            created_at=_clock(),
        )
        assert proposal.selected_profiles == [expected_profile]
        remote_task_id = (
            "train_model" if expected_profile == "unimol-train-v1" else "generate_candidates"
        )
        intent = next(
            item for item in proposal.dispatch_intents if item.task_id == remote_task_id
        )
        assert intent.execution_route == "remote_execution_service"
        assert intent.logical_profile_id == expected_profile
        assert intent.remote_task_type in {"model_training", "molecular_generation"}
        assert intent.requested_resources is not None
        if expected_profile == "reinvent4-cpu-v1":
            assert intent.requested_resources.status == "partial"
            assert intent.requested_resources.walltime_sec == 600
        else:
            assert intent.requested_resources.status == "not_configured"
        proposal_bytes = json.dumps(proposal.model_dump(mode="json"), sort_keys=True)
        assert "train_model_unimol_legacy_adapter" not in proposal_bytes
        assert "generate_candidates_stub_adapter" not in proposal_bytes
        assert '"adapter"' not in proposal_bytes


def test_uploaded_csv_compiles_inspect_clean_trainability_and_baseline_training(
    tmp_path: Path,
) -> None:
    storage, run_dir = _storage_with_run(tmp_path)
    _write_content_bound_artifact(
        storage,
        run_dir,
        artifact_id="uploaded_dataset",
        relative_path="inputs/uploaded.csv",
        content=b"SMILES,plqy\nC,0.42\n",
    )
    assert not (run_dir / "stage.json").exists()
    observation = _observation(storage)
    response = AgentExecutionPlanLLMResponse.model_validate(
        {
            "requested_tool_ids": ["check_trainability", "train_model"],
            "selected_input_artifact_ids": ["uploaded_dataset"],
            "task_options": {
                "check_trainability": {},
                "train_model": {"backend": "baseline", "property_id": "plqy"},
            },
            "selected_logical_profile_ids": [],
            "limits": {},
            "stop_conditions": ["stop on validation failure"],
            "success_criteria": ["produce a reviewable baseline model plan"],
            "rationales": [],
            "assumptions": [],
            "questions": [],
        }
    )
    proposal = AgentExecutionPlanCompiler().compile(
        observation=observation,
        response=response,
        invocation=_invocation(observation, response),
        created_at=_clock(),
    )

    assert [task.task_id for task in proposal.run_plan.tasks] == [
        "inspect_dataset",
        "clean_dataset",
        "check_trainability",
        "train_model",
    ]
    assert proposal.missing_artifacts == []
    assert not [question for question in proposal.questions if question.blocks_proposal]
    train_task = next(task for task in proposal.run_plan.tasks if task.task_id == "train_model")
    assert "cleaned_train_dataset" in train_task.required_artifacts
    assert proposal.executable is False
    assert not (run_dir / "stage.json").exists()


def test_confirmed_dataset_compiles_trainability_and_baseline_without_raw_upload(
    tmp_path: Path,
) -> None:
    storage, run_dir = _storage_with_run(tmp_path)
    _write_confirmed_dataset(storage, run_dir)
    stage_before = (run_dir / "stage.json").read_bytes()
    observation = _observation(storage)
    response = AgentExecutionPlanLLMResponse.model_validate(
        {
            "requested_tool_ids": ["check_trainability", "train_model"],
            "selected_input_artifact_ids": ["confirmed_training_dataset"],
            "task_options": {
                "check_trainability": {},
                "train_model": {"backend": "baseline", "property_id": "plqy"},
            },
            "selected_logical_profile_ids": [],
            "limits": {},
            "stop_conditions": ["stop on validation failure"],
            "success_criteria": ["produce a reviewable baseline model plan"],
            "rationales": [],
            "assumptions": [],
            "questions": [],
        }
    )
    proposal = AgentExecutionPlanCompiler().compile(
        observation=observation,
        response=response,
        invocation=_invocation(observation, response),
        created_at=_clock(),
    )

    assert [task.task_id for task in proposal.run_plan.tasks] == [
        "inspect_dataset",
        "check_trainability",
        "train_model",
    ]
    assert proposal.missing_artifacts == []
    assert "uploaded_dataset" not in proposal.run_plan.available_artifacts
    assert "clean_dataset" not in {task.task_id for task in proposal.run_plan.tasks}
    assert not [question for question in proposal.questions if question.blocks_proposal]
    train_task = next(task for task in proposal.run_plan.tasks if task.task_id == "train_model")
    assert "confirmed_training_dataset" in train_task.required_artifacts
    assert proposal.executable is False
    assert (run_dir / "stage.json").read_bytes() == stage_before


def test_compiler_rejects_multiple_artifacts_from_one_registered_alternative_set(
    tmp_path: Path,
) -> None:
    storage, run_dir = _storage_with_run(tmp_path)
    _write_confirmed_dataset(storage, run_dir)
    _write_content_bound_artifact(
        storage,
        run_dir,
        artifact_id="uploaded_dataset",
        relative_path="inputs/uploaded.csv",
        content=b"SMILES,plqy\nC,0.42\n",
    )
    observation = _observation(storage)
    response = _response(
        tool_id="inspect_dataset",
        selected_input_artifact_ids=[
            "uploaded_dataset",
            "confirmed_training_dataset",
        ],
        task_options={"inspect_dataset": {}},
    )
    with pytest.raises(ScientificAgentPlanError, match="select exactly one artifact"):
        AgentExecutionPlanCompiler().compile(
            observation=observation,
            response=response,
            invocation=_invocation(observation, response),
            created_at=_clock(),
        )


@pytest.mark.parametrize(
    ("tool_id", "options"),
    [
        ("generate_candidates", {"candidate_count": 8}),
        ("filter_rank", {"topn": 3}),
        ("train_model", {"target_property": "plqy"}),
        ("inspect_dataset", {"target_property": "plqy"}),
    ],
)
def test_executor_incompatible_planner_option_aliases_fail_closed(
    tmp_path: Path,
    tool_id: str,
    options: dict[str, object],
) -> None:
    storage, _ = _storage_with_run(tmp_path)
    observation = _observation(storage)
    response = _response(tool_id=tool_id, task_options={tool_id: options})
    with pytest.raises(ScientificAgentPlanError, match="options rejected"):
        AgentExecutionPlanCompiler().compile(
            observation=observation,
            response=response,
            invocation=_invocation(observation, response),
            created_at=_clock(),
        )


def test_compiler_uses_tool_specific_artifact_trust_classes(tmp_path: Path) -> None:
    storage, run_dir = _storage_with_run(tmp_path)
    _write_content_bound_artifact(
        storage,
        run_dir,
        artifact_id="pdf_corpus",
        relative_path="inputs/papers.pdf",
        content=b"%PDF-1.7 review-only input\n",
    )
    profiles = ResourceProfileStore(
        workspace_dir=tmp_path / "profile-workspace",
        config_dir=tmp_path / "profile-config",
    )
    mineru = _connection(connection_id="mineru-gpu", capabilities=["mineru", "gpu"])
    profiles.save_connection(mineru)
    _save_available_probe(profiles, mineru)
    observation = _observation(storage, resource_profiles=profiles)
    pdf = next(item for item in observation.available_artifacts if item.artifact_id == "pdf_corpus")
    assert pdf.verification_state == "registered"
    assert pdf.trust_class == "content_bound_input"
    parse_response = _response(
        tool_id="parse_document",
        selected_input_artifact_ids=["pdf_corpus"],
        selected_logical_profile_ids=["mineru-v1"],
        task_options={"parse_document": {}},
    )
    parse_proposal = AgentExecutionPlanCompiler().compile(
        observation=observation,
        response=parse_response,
        invocation=_invocation(observation, parse_response),
        created_at=_clock(),
    )
    assert parse_proposal.selected_artifacts == ["pdf_corpus"]
    parse_intent = next(
        item for item in parse_proposal.dispatch_intents if item.task_id == "parse_document"
    )
    assert parse_intent.execution_route == "remote_execution_service"
    assert parse_intent.remote_task_type == "document_parsing"
    assert parse_intent.logical_profile_id == "mineru-v1"
    assert "adapter" not in parse_proposal.compiled_task_options["parse_document"]

    _write_content_bound_artifact(
        storage,
        run_dir,
        artifact_id="candidate_training_dataset",
        relative_path="inputs/raw_dataset.csv",
        content=b"smiles,plqy\nC,0.42\n",
    )
    observation = _observation(storage, resource_profiles=profiles)
    raw_dataset = next(
        item for item in observation.available_artifacts if item.artifact_id == "candidate_training_dataset"
    )
    assert raw_dataset.trust_class == "content_bound_input"
    confirm_response = _response(
        tool_id="confirm_extracted_dataset",
        selected_input_artifact_ids=["candidate_training_dataset"],
        task_options={"confirm_extracted_dataset": {}},
    )
    confirm_proposal = AgentExecutionPlanCompiler().compile(
        observation=observation,
        response=confirm_response,
        invocation=_invocation(observation, confirm_response),
        created_at=_clock(),
    )
    assert "candidate_training_dataset" in confirm_proposal.selected_artifacts
    assert confirm_proposal.executable is False

    _write_content_bound_artifact(
        storage,
        run_dir,
        artifact_id="cleaned_train_dataset",
        relative_path="inputs/unverified_cleaned.csv",
        content=b"smiles,plqy\nC,0.42\n",
    )
    observation = _observation(storage, resource_profiles=profiles)
    rejected_response = _response(
        tool_id="train_model",
        selected_input_artifact_ids=["cleaned_train_dataset"],
        selected_logical_profile_ids=[],
        task_options={
            "train_model": {"backend": "baseline", "property_id": "plqy"}
        },
    )
    with pytest.raises(ScientificAgentPlanError, match="trust class"):
        AgentExecutionPlanCompiler().compile(
            observation=observation,
            response=rejected_response,
            invocation=_invocation(observation, rejected_response),
            created_at=_clock(),
        )


def test_content_bound_uploaded_csv_can_bind_inspect_dataset_without_stage_state(
    tmp_path: Path,
) -> None:
    storage, run_dir = _storage_with_run(tmp_path)
    assert not (run_dir / "stage.json").exists()
    raw_csv = b"smiles,plqy\nC,0.42\n"
    _write_content_bound_artifact(
        storage,
        run_dir,
        artifact_id="uploaded_dataset",
        relative_path="inputs/uploaded.csv",
        content=raw_csv,
    )
    observation = _observation(storage)
    uploaded = next(
        item for item in observation.available_artifacts if item.artifact_id == "uploaded_dataset"
    )
    assert uploaded.trust_class == "content_bound_input"
    assert uploaded.content_digest == "sha256:" + hashlib.sha256(raw_csv).hexdigest()

    response = _response(
        tool_id="inspect_dataset",
        selected_input_artifact_ids=["uploaded_dataset"],
        task_options={"inspect_dataset": {}},
    )
    proposal = AgentExecutionPlanCompiler().compile(
        observation=observation,
        response=response,
        invocation=_invocation(observation, response),
        created_at=_clock(),
    )
    assert proposal.selected_artifacts == ["uploaded_dataset"]
    assert proposal.run_plan.requested_tasks == ["inspect_dataset"]
    assert [task.task_id for task in proposal.run_plan.tasks] == ["inspect_dataset"]
    assert proposal.run_plan.missing_artifacts == []
    assert not proposal.questions
    assert proposal.observation_digest == observation.observation_digest
    assert proposal.executable is False
    assert not (run_dir / "stage.json").exists()


def test_planner_option_compiler_materializes_executor_canonical_options() -> None:
    catalog = build_scientific_tool_catalog()
    tools = {tool.tool_id: tool for tool in catalog.tools}
    compiler = PlannerOptionCompiler()

    clean = compiler.compile(
        tool=tools["clean_dataset"],
        planner_options={"drop_empty_target_rows": True},
    )
    assert clean == {
        "drop_empty_target_rows": True,
        "min_nonempty": 1,
        "min_numeric_ratio": 0.5,
        "strict_smiles_cleaning": True,
    }
    baseline = compiler.compile(
        tool=tools["train_model"],
        planner_options={"backend": "baseline", "property_id": "plqy"},
    )
    assert baseline == {"n_bits": 256, "property_id": "plqy"}
    unimol = compiler.compile(
        tool=tools["train_model"],
        planner_options={"backend": "unimol", "property_id": "plqy"},
    )
    assert unimol == {"property_id": "plqy"}
    assert "adapter" not in unimol
    generation = compiler.compile(
        tool=tools["generate_candidates"],
        planner_options={"backend": "deterministic_stub", "count": 12},
    )
    assert generation == {"backend": "deterministic_stub", "count": 12, "seed": 0}
    ranking = compiler.compile(
        tool=tools["filter_rank"],
        planner_options={
            "top_n": 4,
            "objectives": [
                {"column": "plqy_pred", "direction": "maximize", "weight": 1.0}
            ],
            "constraints": [{"column": "sa_score", "maximum": 4.5}],
        },
    )
    assert ranking == {
        "directions": {"plqy_pred": "maximize"},
        "hard_constraints": {"sa_score": {"max": 4.5}},
        "score_columns": ["plqy_pred"],
        "topn": 4,
        "weights": {"plqy_pred": 1.0},
    }


def test_every_visible_tool_compiles_into_executor_snapshot_without_ignored_options(
    tmp_path: Path,
) -> None:
    from ai4s_agent.executor import RunPlanExecutor

    storage, run_dir = _storage_with_run(tmp_path)
    artifact_dir = run_dir / "snapshot-inputs"
    artifact_dir.mkdir()
    registry = AtomicTaskRegistry()
    planner_options = {
        "clean_dataset": {"drop_empty_target_rows": True},
        "train_model": {"backend": "baseline", "property_id": "plqy"},
        "generate_candidates": {"backend": "deterministic_stub", "count": 8},
        "predict_candidates": {"property_id": "plqy"},
        "filter_rank": {
            "top_n": 3,
            "objectives": [
                {"column": "plqy_pred", "direction": "maximize", "weight": 1.0}
            ],
        },
        "retrieve_evidence": {"query": "OLED emitter PLQY", "topk": 5},
    }
    catalog = build_scientific_tool_catalog(registry)
    option_compiler = PlannerOptionCompiler()
    executor = RunPlanExecutor(storage=storage, registry=registry)
    for tool in catalog.tools:
        options = planner_options.get(tool.tool_id, {})
        assert Draft202012Validator(tool.option_schema).is_valid(options)
        compiled = option_compiler.compile(tool=tool, planner_options=options)
        route, remote_task_type = option_compiler.execution_binding(
            tool=tool,
            planner_options=options,
        )
        if route == "remote_execution_service":
            assert remote_task_type in {
                "document_parsing",
                "model_training",
                "molecular_generation",
            }
            assert "adapter" not in compiled
            continue

        selected_inputs = list(tool.required_input_artifact_ids)
        selected_inputs.extend(group[0] for group in tool.input_artifact_alternatives)
        selected_inputs = list(dict.fromkeys(selected_inputs))
        artifact_paths: dict[str, str] = {}
        for artifact_id in selected_inputs:
            path = artifact_dir / f"{tool.task_id}-{artifact_id}.json"
            payload = {
                "property_id": "plqy",
                "properties": [{"property_id": "plqy"}],
            }
            if artifact_id == "model_metadata":
                model_file = artifact_dir / f"{tool.task_id}-model.pkl"
                model_file.write_bytes(b"model")
                payload["model_path"] = str(model_file)
            path.write_text(json.dumps(payload), encoding="utf-8")
            artifact_paths[artifact_id] = str(path)
        run_plan = RunPlan(
            run_id=run_dir.name,
            requested_tasks=[tool.task_id],
            tasks=[
                PlannedTask(
                    task_id=tool.task_id,
                    required_artifacts=selected_inputs,
                    output_artifacts=list(tool.output_artifact_ids),
                )
            ],
            available_artifacts=selected_inputs,
        )
        snapshot = executor._execution_snapshot(
            task_id=tool.task_id,
            spec_default_adapter=registry.get(tool.task_id).default_adapter,
            run_plan=run_plan,
            run_dir=run_dir,
            artifact_paths=artifact_paths,
            approved_gates=set(tool.required_gates),
            options=compiled,
        )
        assert snapshot["task_options"] == compiled
        assert set(snapshot["input_artifacts"]) == set(selected_inputs)
        for key, value in compiled.items():
            assert key in snapshot["payload"]
            assert snapshot["payload"][key] == value


def test_selected_optional_executor_inputs_are_bound_without_global_artifact_union(
    tmp_path: Path,
) -> None:
    from ai4s_agent.executor import RunPlanExecutor

    storage, run_dir = _storage_with_run(tmp_path)
    artifact_dir = run_dir / "optional-snapshot-inputs"
    artifact_dir.mkdir()
    executor = RunPlanExecutor(storage=storage)
    registry = AtomicTaskRegistry()

    confirmed = artifact_dir / "confirmed.json"
    confirmed.write_text("{}", encoding="utf-8")
    generation_paths = {"confirmed_training_dataset": str(confirmed)}
    generation_plan = RunPlan(
        run_id=run_dir.name,
        requested_tasks=["generate_candidates"],
        tasks=[
            PlannedTask(
                task_id="generate_candidates",
                required_artifacts=["confirmed_training_dataset"],
                output_artifacts=["candidate_dataset"],
            )
        ],
        available_artifacts=["confirmed_training_dataset"],
    )
    generation_snapshot = executor._execution_snapshot(
        task_id="generate_candidates",
        spec_default_adapter=registry.get("generate_candidates").default_adapter,
        run_plan=generation_plan,
        run_dir=run_dir,
        artifact_paths=generation_paths,
        approved_gates=set(),
        options={"backend": "deterministic_stub", "count": 8, "seed": 0},
    )
    assert generation_snapshot["payload"]["reference_csv"] == str(confirmed)
    assert set(generation_snapshot["input_artifacts"]) == set(generation_paths)

    candidate = artifact_dir / "candidate.csv"
    candidate.write_text("SMILES\nC\n", encoding="utf-8")
    metadata = artifact_dir / "model_metadata.json"
    metadata.write_text(json.dumps({"property_id": "plqy"}), encoding="utf-8")
    trained_model = artifact_dir / "trained-model"
    trained_model.mkdir()
    (trained_model / "model.pkl").write_bytes(b"model")
    artifact_paths = {
        "candidate_dataset": str(candidate),
        "model_metadata": str(metadata),
        "trained_model": str(trained_model),
    }
    plan = RunPlan(
        run_id=run_dir.name,
        requested_tasks=["predict_candidates"],
        tasks=[
            PlannedTask(
                task_id="predict_candidates",
                required_artifacts=list(artifact_paths),
                output_artifacts=["candidate_predictions"],
            )
        ],
        available_artifacts=list(artifact_paths),
    )
    snapshot = executor._execution_snapshot(
        task_id="predict_candidates",
        spec_default_adapter=registry.get("predict_candidates").default_adapter,
        run_plan=plan,
        run_dir=run_dir,
        artifact_paths=artifact_paths,
        approved_gates=set(),
        options={"property_id": "plqy"},
    )
    assert set(snapshot["input_artifacts"]) == set(artifact_paths)
    assert snapshot["payload"]["model_path"] == str(trained_model / "model.pkl")


def test_proposal_storage_is_exact_no_replace_and_detects_stale_sources(tmp_path: Path) -> None:
    storage, run_dir = _storage_with_run(tmp_path)
    _write_confirmed_dataset(storage, run_dir, canary=True)
    observation_builder = AgentProjectObservationBuilder(storage=storage, clock=_clock)
    observation = observation_builder.build(
        project_id="project-1",
        run_id=run_dir.name,
        goal="Prepare a reviewable scientific plan",
        user_constraints=["use confirmed data only"],
    )
    response = _response()
    proposal = AgentExecutionPlanCompiler().compile(
        observation=observation,
        response=response,
        invocation=_invocation(observation, response),
        created_at=_clock(),
    )
    store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=observation_builder,
    )
    first = store.publish(
        observation=observation,
        catalog=observation.tool_catalog,
        llm_response=response,
        proposal=proposal,
    )
    proposal_dir = _proposal_dir(storage, proposal.proposal_id)
    published_bytes = b"".join(
        path.read_bytes() for path in sorted(proposal_dir.iterdir())
    )
    for canary in (
        _private_path_canary().encode("utf-8"),
        b"cluster.internal",
        b"sk-test-canary",
        b"private paper text",
    ):
        assert canary not in published_bytes
    replay = store.publish(
        observation=observation,
        catalog=observation.tool_catalog,
        llm_response=response,
        proposal=proposal,
    )
    assert replay.proposal.proposal_digest == first.proposal.proposal_digest
    loaded = store.read(project_id="project-1", proposal_id=proposal.proposal_id)
    assert loaded.proposal.model_dump(mode="json") == proposal.model_dump(mode="json")
    summary_bytes = (proposal_dir / "proposal_summary.md").read_bytes()
    (proposal_dir / "proposal_summary.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ScientificAgentPlanPublicationConflict):
        store.publish(
            observation=observation,
            catalog=observation.tool_catalog,
            llm_response=response,
            proposal=proposal,
        )
    with pytest.raises(ScientificAgentPlanError):
        store.read(project_id="project-1", proposal_id=proposal.proposal_id, verify_current=False)

    (proposal_dir / "proposal_summary.md").write_bytes(summary_bytes)
    (run_dir / "data" / "confirmed_training_dataset.json").write_text(
        json.dumps({"dataset_id": "confirmed-dataset", "confirmed": True, "row_count": 5}),
        encoding="utf-8",
    )
    with pytest.raises(ScientificAgentPlanSourceChanged):
        store.read(project_id="project-1", proposal_id=proposal.proposal_id)


def test_proposal_publication_does_not_modify_stage_state(tmp_path: Path) -> None:
    storage, run_dir = _storage_with_run(tmp_path)
    _write_confirmed_dataset(storage, run_dir)
    observation_builder = AgentProjectObservationBuilder(storage=storage, clock=_clock)
    observation = observation_builder.build(
        project_id="project-1",
        run_id=run_dir.name,
        goal="Prepare a reviewable scientific plan",
    )
    response = _response()
    proposal = AgentExecutionPlanCompiler().compile(
        observation=observation,
        response=response,
        invocation=_invocation(observation, response),
        created_at=_clock(),
    )
    stage_before = (run_dir / "stage.json").read_bytes()
    ScientificAgentPlanProposalStore(storage=storage, observation_builder=observation_builder).publish(
        observation=observation,
        catalog=observation.tool_catalog,
        llm_response=response,
        proposal=proposal,
    )
    assert (run_dir / "stage.json").read_bytes() == stage_before
    assert not (run_dir / "gate_decision.json").exists()
    assert not (run_dir / "queue_job.json").exists()


def test_service_request_idempotency_separates_semantics_from_publication(tmp_path: Path) -> None:
    storage, run_dir = _storage_with_run(tmp_path)
    _write_confirmed_dataset(storage, run_dir)

    class CountingStubProvider(StubLLMProvider):
        def __init__(self) -> None:
            super().__init__(response=_response().model_dump(mode="json"), response_id="provider-response")
            self.calls = 0

        def complete_json(self, **kwargs):
            self.calls += 1
            return super().complete_json(**kwargs)

    provider = CountingStubProvider()
    service = ScientificAgentPlanService(storage=storage, clock=_clock)
    first = service.create_proposal(
        project_id="project-1",
        run_id=run_dir.name,
        goal="Prepare a reviewable scientific plan",
        user_constraints=[],
        provider=provider,
        client_request_id="request-replay-1",
    )
    replay = service.create_proposal(
        project_id="project-1",
        run_id=run_dir.name,
        goal="Prepare a reviewable scientific plan",
        user_constraints=[],
        provider=provider,
        client_request_id="request-replay-1",
    )
    assert provider.calls == 1
    assert replay.model_dump(mode="json") == first.model_dump(mode="json")
    assert first.publication_id == first.proposal_id
    assert first.semantic_plan_id.startswith("semantic-plan-")

    with pytest.raises(ScientificAgentPlanPublicationConflict):
        service.create_proposal(
            project_id="project-1",
            run_id=run_dir.name,
            goal="A different planning goal",
            user_constraints=[],
            provider=provider,
            client_request_id="request-replay-1",
        )
    assert provider.calls == 1

    second_publication = service.create_proposal(
        project_id="project-1",
        run_id=run_dir.name,
        goal="Prepare a reviewable scientific plan",
        user_constraints=[],
        provider=provider,
        client_request_id="request-replay-2",
    )
    assert provider.calls == 2
    assert second_publication.semantic_plan_id == first.semantic_plan_id
    assert second_publication.semantic_plan_digest == first.semantic_plan_digest
    assert second_publication.publication_id != first.publication_id
    assert second_publication.proposal_digest != first.proposal_digest
    assert _proposal_dir(storage, first.proposal_id).is_dir()
    assert _proposal_dir(storage, second_publication.proposal_id).is_dir()


def test_cross_process_same_request_calls_llm_once(tmp_path: Path) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("cross-process file-lock acceptance requires fork")
    storage, _ = _storage_with_run(tmp_path)
    workspace = str(storage.workspace_dir)
    counter = str(tmp_path / "llm-calls.log")
    context = multiprocessing.get_context("fork")
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_multiprocess_plan_worker,
            kwargs={
                "workspace_dir": workspace,
                "counter_path": counter,
                "request_id": "request-process-replay",
                "goal": "Prepare a reviewable scientific plan",
                "start_event": start_event,
                "result_queue": result_queue,
            },
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    results = [result_queue.get(timeout=2) for _ in processes]
    assert [result[0] for result in results] == ["success", "success"]
    assert results[0][1] == results[1][1]
    assert Path(counter).read_text(encoding="utf-8").splitlines() == ["llm-call"]


def test_cross_process_same_request_different_payload_fails_before_second_llm(
    tmp_path: Path,
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("cross-process file-lock acceptance requires fork")
    storage, _ = _storage_with_run(tmp_path)
    workspace = str(storage.workspace_dir)
    counter = str(tmp_path / "llm-calls.log")
    context = multiprocessing.get_context("fork")
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_multiprocess_plan_worker,
            kwargs={
                "workspace_dir": workspace,
                "counter_path": counter,
                "request_id": "request-process-conflict",
                "goal": goal,
                "start_event": start_event,
                "result_queue": result_queue,
            },
        )
        for goal in ("Plan OLED screening", "Plan a different OLED objective")
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    results = [result_queue.get(timeout=2) for _ in processes]
    assert sorted(result[0] for result in results) == [
        "ScientificAgentPlanPublicationConflict",
        "success",
    ]
    assert Path(counter).read_text(encoding="utf-8").splitlines() == ["llm-call"]


@pytest.mark.parametrize(
    "fault_phase",
    ["after_publication_file_3", "after_publication_rename", "before_request_commit"],
)
def test_new_process_recovers_checkpointed_publication_without_repeating_llm(
    tmp_path: Path,
    fault_phase: str,
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("cross-process recovery acceptance requires fork")
    storage, _ = _storage_with_run(tmp_path)
    workspace = str(storage.workspace_dir)
    counter = str(tmp_path / "llm-calls.log")
    context = multiprocessing.get_context("fork")
    first_event = context.Event()
    first_queue = context.Queue()
    first = context.Process(
        target=_multiprocess_plan_worker,
        kwargs={
            "workspace_dir": workspace,
            "counter_path": counter,
            "request_id": "request-crash-recovery",
            "goal": "Prepare a reviewable scientific plan",
            "start_event": first_event,
            "result_queue": first_queue,
            "fault_phase": fault_phase,
        },
    )
    first.start()
    first_event.set()
    first.join(timeout=15)
    assert first.exitcode == 0
    assert first_queue.get(timeout=2)[0] == "RuntimeError"

    second_event = context.Event()
    second_queue = context.Queue()
    second = context.Process(
        target=_multiprocess_plan_worker,
        kwargs={
            "workspace_dir": workspace,
            "counter_path": counter,
            "request_id": "request-crash-recovery",
            "goal": "Prepare a reviewable scientific plan",
            "start_event": second_event,
            "result_queue": second_queue,
        },
    )
    second.start()
    second_event.set()
    second.join(timeout=15)
    assert second.exitcode == 0
    assert second_queue.get(timeout=2)[0] == "success"
    assert Path(counter).read_text(encoding="utf-8").splitlines() == ["llm-call"]

    request_dir = (
        storage.projects_root
        / "project-1"
        / "agent_plan_requests"
        / "request-crash-recovery"
    )
    assert json.loads((request_dir / "reservation.json").read_text())["status"] == "RESERVED"
    assert json.loads((request_dir / "planning.json").read_text())["status"] == "PLANNING"
    assert json.loads((request_dir / "publication_pending.json").read_text())[
        "status"
    ] == "PUBLICATION_PENDING"
    assert json.loads((request_dir / "committed.json").read_text())["status"] == "COMMITTED"


def test_crash_after_llm_without_checkpoint_enters_typed_recovery_without_second_call(
    tmp_path: Path,
) -> None:
    storage, run_dir = _storage_with_run(tmp_path)

    class CountingProvider(StubLLMProvider):
        def __init__(self) -> None:
            super().__init__(response=_response().model_dump(mode="json"))
            self.calls = 0

        def complete_json(self, **kwargs):
            self.calls += 1
            return super().complete_json(**kwargs)

    def crash_after_llm(phase: str) -> None:
        if phase == "after_llm_response":
            raise RuntimeError("simulated process loss after provider response")

    builder = AgentProjectObservationBuilder(storage=storage, clock=_clock)
    crashing_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=builder,
        fault_injector=crash_after_llm,
    )
    first_provider = CountingProvider()
    with pytest.raises(RuntimeError, match="process loss"):
        ScientificAgentPlanService(
            storage=storage,
            observation_builder=builder,
            proposal_store=crashing_store,
            clock=_clock,
        ).create_proposal(
            project_id="project-1",
            run_id=run_dir.name,
            goal="Prepare a reviewable scientific plan",
            user_constraints=[],
            provider=first_provider,
            client_request_id="request-uncertain-provider-result",
        )
    assert first_provider.calls == 1

    second_provider = CountingProvider()
    with pytest.raises(ScientificAgentPlanRecoveryRequired) as exc_info:
        ScientificAgentPlanService(storage=storage, clock=_clock).create_proposal(
            project_id="project-1",
            run_id=run_dir.name,
            goal="Prepare a reviewable scientific plan",
            user_constraints=[],
            provider=second_provider,
            client_request_id="request-uncertain-provider-result",
        )
    assert exc_info.value.state == "PLANNING"
    assert second_provider.calls == 0


def test_service_creation_isolated_from_execution_authority(tmp_path: Path, monkeypatch) -> None:
    from ai4s_agent.executor import RunPlanExecutor
    from ai4s_agent.remote_execution_service import DescriptorRemoteExecutionLifecycleService
    import ai4s_agent.scientific_agent_plan as plan_module

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("planning proposal creation must not enter execution authority")

    monkeypatch.setattr(RunPlanExecutor, "execute", forbidden)
    monkeypatch.setattr(RunPlanExecutor, "resume_after_gate", forbidden)
    monkeypatch.setattr(DescriptorRemoteExecutionLifecycleService, "prepare", forbidden)
    monkeypatch.setattr(DescriptorRemoteExecutionLifecycleService, "approve", forbidden)
    assert "RunPlanExecutor" not in plan_module.__dict__
    assert "DescriptorRemoteExecutionLifecycleService" not in plan_module.__dict__

    storage, run_dir = _storage_with_run(tmp_path)
    _write_confirmed_dataset(storage, run_dir)
    proposal = ScientificAgentPlanService(storage=storage, clock=_clock).create_proposal(
        project_id="project-1",
        run_id=run_dir.name,
        goal="Prepare a reviewable scientific plan",
        user_constraints=[],
        provider=StubLLMProvider(response=_response().model_dump(mode="json")),
    )
    assert proposal.executable is False
    assert not (run_dir / "gate_decision.json").exists()
    assert not (run_dir / "queue_job.json").exists()


def test_loopback_stub_service_metadata_is_redacted_from_public_artifacts(tmp_path: Path) -> None:
    storage, run_dir = _storage_with_run(tmp_path)
    _write_confirmed_dataset(storage, run_dir)
    service = ScientificAgentPlanService(storage=storage, clock=_clock)
    proposal = service.create_proposal(
        project_id="project-1",
        run_id=run_dir.name,
        goal="Optimize PLQY for an OLED host–dopant system; the previous model failed validation.",
        user_constraints=[],
        provider=StubLLMProvider(response=_response().model_dump(mode="json")),
    )
    proposal_json = (_proposal_dir(storage, proposal.proposal_id) / "proposal.json").read_text(encoding="utf-8")
    llm_json = (_proposal_dir(storage, proposal.proposal_id) / "llm_response.json").read_text(encoding="utf-8")
    assert "raw_response" not in proposal_json
    assert "messages" not in llm_json
    assert proposal.executable is False
    assert "OLED host–dopant" in proposal.goal


def test_service_rejects_malformed_or_failed_dedicated_planning_call(tmp_path: Path) -> None:
    from ai4s_agent.scientific_agent_plan import ScientificAgentPlanService

    storage, run_dir = _storage_with_run(tmp_path)

    class FailedProvider:
        def complete_json(self, **kwargs):
            del kwargs
            raise LLMProviderError("provider timeout with secret token")

    with pytest.raises(ScientificAgentPlanError):
        ScientificAgentPlanService(storage=storage, clock=_clock).create_proposal(
            project_id="project-1",
            run_id=run_dir.name,
            goal="Prepare a reviewable scientific plan",
            user_constraints=[],
            provider=FailedProvider(),
        )


def test_project_scoped_api_is_non_executable_and_requires_external_consent(tmp_path: Path) -> None:
    from ai4s_agent.app import create_app

    workspace = tmp_path / "api-workspace"
    storage = ProjectStorage(workspace_dir=workspace)
    storage.create_project("project-1", name="Project", created_at=_clock())
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=workspace,
        user_config_dir=tmp_path / "config",
    )
    client = app.test_client()
    valid_body = {
        "run_id": "planning-run-1",
        "goal": "Prepare a reviewable scientific plan",
        "user_constraints": [],
        "client_request_id": "api-first-plan",
        "llm_provider": {
            "provider": "stub",
            "model": "stub",
            "stub_response": _response().model_dump(mode="json"),
        },
    }
    rejected = client.post(
        "/api/projects/project-1/agent-plan-proposals",
        json={**valid_body, "run_plan": {"run_id": "planning-run-1"}},
    )
    assert rejected.status_code == 400
    assert "run_plan" not in rejected.get_json().get("error", "")

    created = client.post("/api/projects/project-1/agent-plan-proposals", json=valid_body)
    assert created.status_code == 200, created.get_json()
    body = created.get_json()
    assert body["executable"] is False
    proposal_id = body["proposal_id"]
    assert body["proposal"]["executable"] is False
    project_dir = workspace / "projects" / "project-1"
    assert (project_dir / "agent_plan_proposals" / proposal_id / "proposal.json").is_file()
    assert not (project_dir / "runs" / "planning-run-1").exists()
    assert not (project_dir / "stage.json").exists()
    assert not (project_dir / "gate_decision.json").exists()
    assert not (project_dir / "queue_job.json").exists()

    fetched = client.get(f"/api/projects/project-1/agent-plan-proposals/{proposal_id}")
    assert fetched.status_code == 200, fetched.get_json()
    assert fetched.get_json()["executable"] is False

    no_consent = client.post(
        "/api/projects/project-1/agent-plan-proposals",
        json={
            "run_id": "run-1",
            "goal": "Prepare a reviewable scientific plan",
            "llm_provider": {
                "provider": "openai_compatible",
                "endpoint": "https://example.com/v1",
                "model": "planner",
            },
        },
    )
    assert no_consent.status_code == 400
