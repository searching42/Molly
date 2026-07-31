from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
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
    RunStatus,
    ScientificToolSpec,
    StageState,
)
from ai4s_agent.scientific_agent_plan import (
    AgentExecutionPlanCompiler,
    AgentProjectObservationBuilder,
    ScientificAgentPlanError,
    ScientificAgentPlanPublicationConflict,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
    ScientificAgentPlanSourceChanged,
    build_scientific_tool_catalog,
)
from ai4s_agent.storage import ProjectStorage


def _clock() -> str:
    return "2026-07-31T00:00:00Z"


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
        output_artifacts=output_artifacts or [],
        label=task_id.replace("_", " ").title(),
        description="A review-only logical scientific task.",
        effect_class="compute",
        required_permissions=[],
        option_schema=option_schema
        or {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        logical_profile_requirements=[],
        accepted_input_trust_classes=(
            ["content_bound_input", "registered_intermediate", "verified_output"]
            if required
            else []
        ),
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
                "path": "/Users/benton/private.csv",
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
        "logical_profile_requirements",
        "accepted_input_trust_classes",
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
        logical_profile_requirements=[],
        accepted_input_trust_classes=[],
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
        "/Users/benton/private.csv",
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
        {"path": "/Users/private/file"},
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
    assert proposal.missing_artifacts == []


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
    )
    with pytest.raises(ScientificAgentPlanError, match="selected logical profiles do not satisfy requirement"):
        AgentExecutionPlanCompiler().compile(
            observation=observation,
            response=train_response,
            invocation=_invocation(observation, train_response),
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
        task_options={"parse_document": {"max_pages": 12}},
    )
    parse_proposal = AgentExecutionPlanCompiler().compile(
        observation=observation,
        response=parse_response,
        invocation=_invocation(observation, parse_response),
        created_at=_clock(),
    )
    assert parse_proposal.selected_artifacts == ["pdf_corpus"]

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
        task_options={"confirm_extracted_dataset": {"minimum_confidence": 0.8}},
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
        selected_logical_profile_ids=["unimol-train-v1"],
    )
    with pytest.raises(ScientificAgentPlanError, match="trust class"):
        AgentExecutionPlanCompiler().compile(
            observation=observation,
            response=rejected_response,
            invocation=_invocation(observation, rejected_response),
            created_at=_clock(),
        )


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
    for canary in (b"/Users/benton/private.csv", b"cluster.internal", b"sk-test-canary", b"private paper text"):
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
