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
from ai4s_agent.resource_profiles import ResourceProfileStore
from ai4s_agent.schemas import (
    AgentExecutionPlanLLMResponse,
    AgentLLMInvocationMetadata,
    ArtifactRef,
    AtomicTaskSpec,
    RunStatus,
    StageState,
)
from ai4s_agent.scientific_agent_plan import (
    AgentExecutionPlanCompiler,
    AgentProjectObservationBuilder,
    ScientificAgentPlanError,
    ScientificAgentPlanPublicationConflict,
    ScientificAgentPlanProposalStore,
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


def _write_confirmed_dataset(storage: ProjectStorage, run_dir: Path, *, canary: bool = False) -> None:
    dataset_path = run_dir / "data" / "confirmed_dataset.json"
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
        "confirmed_dataset",
        "data/confirmed_dataset.json",
    )
    storage.write_stage_state(
        "project-1",
        run_dir.name,
        StageState(
            stage="clean_dataset",
            next_stage="train_model",
            status=RunStatus.SUCCEEDED,
            started_at=_clock(),
            updated_at=_clock(),
            artifacts=[
                ArtifactRef(
                    artifact_id="confirmed_dataset",
                    relative_path="data/confirmed_dataset.json",
                    producer_task_id="clean_dataset",
                )
            ],
            details={"executed_tasks": ["clean_dataset", "not-a-registered-task"]},
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
        AtomicTaskSpec(
            task_id="z_task",
            scientific_tool_id="z_tool",
            required_artifacts=["input_artifact"],
            output_artifacts=["output_artifact"],
            description="A safe high-level task.",
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


def test_catalog_rejects_duplicate_tool_mapping_and_unsafe_option_schema() -> None:
    duplicate = [
        AtomicTaskSpec(task_id="first_task", scientific_tool_id="same_tool"),
        AtomicTaskSpec(task_id="second_task", scientific_tool_id="same_tool"),
    ]
    with pytest.raises(ScientificAgentPlanError):
        build_scientific_tool_catalog(AtomicTaskRegistry(duplicate))

    with pytest.raises(ValueError, match="duplicate atomic task ID"):
        AtomicTaskRegistry([AtomicTaskSpec(task_id="same_task"), AtomicTaskSpec(task_id="same_task")])

    unsafe = AtomicTaskSpec(
        task_id="unsafe_task",
        option_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": [],
            "additionalProperties": False,
        },
    )
    with pytest.raises(ScientificAgentPlanError):
        build_scientific_tool_catalog(AtomicTaskRegistry([unsafe]))


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


@pytest.mark.parametrize(
    "hostile",
    [
        {"execute": True},
        {"approved": True},
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
        selected_input_artifact_ids=["confirmed_dataset"],
    )
    with pytest.raises(ScientificAgentPlanError):
        AgentExecutionPlanCompiler().compile(
            observation=observation,
            response=train_response,
            invocation=_invocation(observation, train_response),
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
    published_bytes = b"".join(
        path.read_bytes() for path in sorted((run_dir / "agent_plans" / proposal.proposal_id).iterdir())
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
    proposal_dir = run_dir / "agent_plans" / proposal.proposal_id
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
    (run_dir / "data" / "confirmed_dataset.json").write_text(
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


def test_loopback_stub_service_metadata_is_redacted_from_public_artifacts(tmp_path: Path) -> None:
    storage, run_dir = _storage_with_run(tmp_path)
    _write_confirmed_dataset(storage, run_dir)
    from ai4s_agent.scientific_agent_plan import ScientificAgentPlanService

    service = ScientificAgentPlanService(storage=storage, clock=_clock)
    proposal = service.create_proposal(
        project_id="project-1",
        run_id=run_dir.name,
        goal="Prepare a reviewable scientific plan",
        user_constraints=[],
        provider=StubLLMProvider(response=_response().model_dump(mode="json")),
    )
    proposal_json = (run_dir / "agent_plans" / proposal.proposal_id / "proposal.json").read_text(encoding="utf-8")
    llm_json = (run_dir / "agent_plans" / proposal.proposal_id / "llm_response.json").read_text(encoding="utf-8")
    assert "raw_response" not in proposal_json
    assert "messages" not in llm_json
    assert proposal.executable is False


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
    storage.run_dir("project-1", "run-1")
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=workspace,
        user_config_dir=tmp_path / "config",
    )
    client = app.test_client()
    valid_body = {
        "run_id": "run-1",
        "goal": "Prepare a reviewable scientific plan",
        "user_constraints": [],
        "llm_provider": {
            "provider": "stub",
            "model": "stub",
            "stub_response": _response().model_dump(mode="json"),
        },
    }
    rejected = client.post(
        "/api/projects/project-1/agent-plan-proposals",
        json={**valid_body, "run_plan": {"run_id": "run-1"}},
    )
    assert rejected.status_code == 400
    assert "run_plan" not in rejected.get_json().get("error", "")

    created = client.post("/api/projects/project-1/agent-plan-proposals", json=valid_body)
    assert created.status_code == 200, created.get_json()
    body = created.get_json()
    assert body["executable"] is False
    proposal_id = body["proposal_id"]
    assert body["proposal"]["executable"] is False
    assert not (storage.run_dir("project-1", "run-1") / "gate_decision.json").exists()

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
