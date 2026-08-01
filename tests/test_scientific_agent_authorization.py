from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.resource_profiles import ConnectionProfile, ResourceProfileStore
from ai4s_agent.schemas import (
    AgentAuthorizationMode,
    AgentBudgetObservation,
    AgentExecutionPlanQuestion,
    AgentExecutionPlanLLMResponse,
    AgentPermissionDecision,
    AgentPermissionOutcome,
    AgentPermissionPhase,
    AgentPermissionShadowRecord,
    AgentPlanAuthorization,
    AgentPlanAuthorizationRequest,
    AgentPlanStartIntent,
    AtomicTaskSpec,
    RiskLevel,
)
from ai4s_agent.scientific_agent_authorization import (
    AGENT_PERMISSION_SHADOW_OBSERVATION_FLAG,
    AgentPlanControlStore,
    ScientificAgentAuthorizationConflict,
    ScientificAgentAuthorizationDenied,
    ScientificAgentAuthorizationService,
    ScientificAgentAuthorizationVerificationError,
)
from ai4s_agent.scientific_agent_permissions import (
    PERMISSION_POLICY_DIGEST,
    PERMISSION_POLICY_MATERIAL,
    PERMISSION_POLICY_VERSION,
    ScientificAgentPermissionEngine,
    compare_permission_outcomes,
)
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
    ScientificAgentPlanSourceChanged,
)
from ai4s_agent.storage import ProjectStorage


def _clock() -> str:
    return "2026-07-31T00:00:00Z"


def _response(tool_id: str = "generate_candidates") -> AgentExecutionPlanLLMResponse:
    return AgentExecutionPlanLLMResponse(
        requested_tool_ids=[tool_id],
        selected_input_artifact_ids=[],
        task_options={tool_id: {}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce a reviewable result"],
        rationales=["Use the registered deterministic workflow."],
        assumptions=[],
        questions=[],
    )


def _visible_task_spec(
    task_id: str,
    *,
    effect_class: str = "derive_local",
    gates: list[str] | None = None,
    depends_on: list[str] | None = None,
) -> AtomicTaskSpec:
    return AtomicTaskSpec(
        task_id=task_id,
        required_artifacts=[],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=[],
        risk_level=RiskLevel.LOW,
        gates=gates or [],
        default_adapter="test_adapter",
        depends_on=depends_on or [],
        scientific_tool_id=task_id,
        label=task_id.replace("_", " ").title(),
        description="A deterministic review-only test task.",
        effect_class=effect_class,
        required_permissions=["derive_project_artifact"],
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
        logical_profile_requirements=[],
        backend_profile_requirements={},
        default_planner_backend=None,
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


def _permission_complete_hidden_task(task_id: str) -> AtomicTaskSpec:
    return AtomicTaskSpec(
        task_id=task_id,
        risk_level=RiskLevel.LOW,
        gates=[],
        effect_class="derive_local",
        required_permissions=["derive_project_artifact"],
        option_schema=None,
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=False,
    )


def _workspace_with_registry_proposal(
    tmp_path: Path,
    *,
    registry: AtomicTaskRegistry,
    response: AgentExecutionPlanLLMResponse,
    request_id: str,
) -> tuple[ProjectStorage, ScientificAgentPlanProposalStore, object]:
    storage = ProjectStorage(workspace_dir=tmp_path / request_id)
    storage.create_project("project-1", name="Project", created_at=_clock())
    builder = AgentProjectObservationBuilder(
        storage=storage,
        registry=registry,
        clock=_clock,
    )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=builder,
        registry=registry,
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        registry=registry,
        observation_builder=builder,
        proposal_store=proposal_store,
        clock=_clock,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Build an exact deterministic local plan",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id=request_id,
    )
    assert not [question for question in proposal.questions if question.blocks_proposal]
    return storage, proposal_store, proposal


def _workspace_with_proposal(
    tmp_path: Path,
    *,
    request_id: str = "proposal-request-1",
) -> tuple[ProjectStorage, ScientificAgentPlanProposalStore, object]:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at=_clock())
    builder = AgentProjectObservationBuilder(storage=storage, clock=_clock)
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=builder,
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        observation_builder=builder,
        proposal_store=proposal_store,
        clock=_clock,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Prepare a deterministic candidate generation plan",
        user_constraints=[],
        provider=StubLLMProvider(response=_response().model_dump(mode="json")),
        client_request_id=request_id,
    )
    assert not [question for question in proposal.questions if question.blocks_proposal]
    assert not proposal.missing_artifacts
    return storage, proposal_store, proposal


def _workspace_with_artifact_proposal(
    tmp_path: Path,
) -> tuple[ProjectStorage, ScientificAgentPlanProposalStore, object, Path]:
    storage = ProjectStorage(workspace_dir=tmp_path / "artifact-workspace")
    storage.create_project("project-1", name="Project", created_at=_clock())
    run_dir = storage.run_dir("project-1", "run-artifact")
    artifact_path = run_dir / "inputs" / "dataset.csv"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("SMILES,value\nCCO,1.0\n", encoding="utf-8")
    storage.register_artifact_path(
        "project-1",
        "run-artifact",
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
        rationales=["Inspect the exact content-bound upload."],
        assumptions=[],
        questions=[],
    )
    builder = AgentProjectObservationBuilder(storage=storage, clock=_clock)
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=builder,
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        observation_builder=builder,
        proposal_store=proposal_store,
        clock=_clock,
    ).create_proposal(
        project_id="project-1",
        run_id="run-artifact",
        goal="Inspect the content-bound uploaded dataset",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="artifact-proposal-request",
    )
    assert not proposal.missing_artifacts
    assert not [question for question in proposal.questions if question.blocks_proposal]
    return storage, proposal_store, proposal, artifact_path


def _workspace_with_profile_source(
    tmp_path: Path,
) -> tuple[
    ProjectStorage,
    ScientificAgentPlanProposalStore,
    object,
    ResourceProfileStore,
]:
    storage = ProjectStorage(workspace_dir=tmp_path / "profile-source-workspace")
    storage.create_project("project-1", name="Project", created_at=_clock())
    profiles = ResourceProfileStore(
        workspace_dir=tmp_path / "profile-private-workspace",
        config_dir=tmp_path / "profile-private-config",
    )
    builder = AgentProjectObservationBuilder(
        storage=storage,
        resource_profiles=profiles,
        clock=_clock,
    )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=builder,
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        resource_profiles=profiles,
        observation_builder=builder,
        proposal_store=proposal_store,
        clock=_clock,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Prepare a deterministic local candidate plan",
        user_constraints=[],
        provider=StubLLMProvider(response=_response().model_dump(mode="json")),
        client_request_id="profile-source-proposal",
    )
    assert not [question for question in proposal.questions if question.blocks_proposal]
    return storage, proposal_store, proposal, profiles


def _request(
    proposal,
    *,
    mode: AgentAuthorizationMode = AgentAuthorizationMode.STEPWISE,
    client_request_id: str = "authorization-request-1",
    requested_gates: list[str] | None = None,
) -> AgentPlanAuthorizationRequest:
    return AgentPlanAuthorizationRequest(
        expected_proposal_digest=proposal.proposal_digest,
        authorization_mode=mode,
        requested_preauthorized_gate_ids=requested_gates or [],
        confirmed=True,
        client_request_id=client_request_id,
        note="Approve this exact immutable plan.",
    )


def _authorization_service(
    storage: ProjectStorage,
    proposal_store: ScientificAgentPlanProposalStore,
    *,
    control_store: AgentPlanControlStore | None = None,
) -> ScientificAgentAuthorizationService:
    return ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        control_store=control_store,
        clock=_clock,
    )


def _multiprocess_authorization_worker(
    workspace_dir: str,
    proposal_id: str,
    proposal_digest: str,
    *,
    note: str,
    start_event,
    result_queue,
) -> None:
    storage = ProjectStorage(workspace_dir=Path(workspace_dir))
    builder = AgentProjectObservationBuilder(storage=storage, clock=_clock)
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=builder,
    )
    service = _authorization_service(storage, proposal_store)
    start_event.wait()
    try:
        result = service.approve_and_start(
            project_id="project-1",
            proposal_id=proposal_id,
            request=AgentPlanAuthorizationRequest(
                expected_proposal_digest=proposal_digest,
                authorization_mode="stepwise",
                requested_preauthorized_gate_ids=[],
                confirmed=True,
                client_request_id="cross-process-authorization-request",
                note=note,
            ),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )
    except Exception as exc:  # noqa: BLE001 - parent asserts the typed result.
        result_queue.put((type(exc).__name__, "", ""))
    else:
        result_queue.put(
            (
                "success",
                result.authorization.authorization_id,
                result.start_intent.start_intent_id,
            )
        )


def test_permission_policy_identity_is_deterministic_across_hash_seeds() -> None:
    assert PERMISSION_POLICY_VERSION == "scientific-agent-permission-policy.v1"
    assert PERMISSION_POLICY_DIGEST.startswith("sha256:")
    assert PERMISSION_POLICY_MATERIAL["outcome_precedence"] == [
        "DENY",
        "REQUIRE_APPROVAL",
        "ALLOW",
    ]
    script = (
        "from ai4s_agent.scientific_agent_permissions import "
        "PERMISSION_POLICY_DIGEST; print(PERMISSION_POLICY_DIGEST)"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    values = []
    for seed in ("1", "77"):
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        values.append(completed.stdout.strip())
    assert values == [PERMISSION_POLICY_DIGEST, PERMISSION_POLICY_DIGEST]


def test_frozen_permission_authorization_schemas_match_generated_models() -> None:
    schema_dir = Path(__file__).resolve().parents[1] / "docs" / "schemas"
    models = {
        "agent_permission_decision": AgentPermissionDecision,
        "agent_plan_authorization_request": AgentPlanAuthorizationRequest,
        "agent_plan_authorization": AgentPlanAuthorization,
        "agent_plan_start_intent": AgentPlanStartIntent,
        "agent_permission_shadow_record": AgentPermissionShadowRecord,
    }
    for name, model in models.items():
        frozen = json.loads(
            (schema_dir / f"{name}.schema.json").read_text(encoding="utf-8")
        )
        assert frozen == model.model_json_schema()


def test_complete_proposal_review_requires_exact_plan_authorization(tmp_path: Path) -> None:
    storage, proposal_store, proposal = _workspace_with_proposal(tmp_path)
    decision = _authorization_service(storage, proposal_store).evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )

    assert decision.phase == AgentPermissionPhase.PROPOSAL_REVIEW
    assert decision.outcome == AgentPermissionOutcome.REQUIRE_APPROVAL
    assert "PLAN_AUTHORIZATION_REQUIRED" in decision.reason_codes
    assert [item.task_id for item in decision.task_decisions] == [
        item.task_id for item in proposal.run_plan.tasks
    ]
    persisted = _authorization_service(storage, proposal_store).control_store.read_permission_decision(
        project_id="project-1",
        decision_id=decision.decision_id,
    )
    assert persisted.model_dump(mode="json") == decision.model_dump(mode="json")


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("unknown_task", "TASK_UNKNOWN"),
        ("unknown_permission", "PERMISSION_UNKNOWN"),
        ("unknown_effect", "EFFECT_CLASS_UNKNOWN"),
        ("unknown_gate", "UNKNOWN_GATE"),
        ("unknown_route", "UNKNOWN_DISPATCH_ROUTE"),
        ("unknown_remote_task_type", "REMOTE_TASK_TYPE_UNKNOWN"),
        ("blocking_question", "BLOCKING_QUESTION_PRESENT"),
        ("missing_artifact", "MISSING_ARTIFACT_PRESENT"),
        ("effective_options_missing", "OPTIONS_COVERAGE_MISMATCH"),
        ("compiled_options_missing", "OPTIONS_COVERAGE_MISMATCH"),
        ("dispatch_intent_missing", "DISPATCH_COVERAGE_MISMATCH"),
        ("artifact_binding_drift", "ARTIFACT_BINDING_DRIFT"),
        ("profile_binding_drift", "PROFILE_BINDING_DRIFT"),
        ("budget_expansion", "BUDGET_LIMIT_EXCEEDED"),
        ("resource_partial", "REMOTE_RESOURCE_INTENT_INCOMPLETE"),
    ],
)
def test_permission_engine_denies_adversarial_or_incomplete_bindings(
    tmp_path: Path,
    mutation: str,
    expected_reason: str,
) -> None:
    _, proposal_store, proposal = _workspace_with_proposal(tmp_path)
    publication = proposal_store.read(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        verify_current=True,
    )
    first_task = publication.proposal.run_plan.tasks[0].task_id
    tool = next(item for item in publication.catalog.tools if item.task_id == first_task)
    dispatch = next(
        item for item in publication.proposal.dispatch_intents if item.task_id == first_task
    )
    if mutation == "unknown_task":
        publication.proposal.run_plan.tasks[0].task_id = "unknown_task"
    elif mutation == "unknown_permission":
        tool.required_permissions.append("future_unknown_permission")
    elif mutation == "unknown_effect":
        tool.effect_class = "future_effect"  # type: ignore[assignment]
    elif mutation == "unknown_gate":
        tool.required_gates.append("gate_future_unknown")
    elif mutation == "unknown_route":
        dispatch.execution_route = "future_route"  # type: ignore[assignment]
    elif mutation == "unknown_remote_task_type":
        dispatch.execution_route = "remote_execution_service"
        dispatch.remote_task_type = "future_remote_type"  # type: ignore[assignment]
    elif mutation == "blocking_question":
        publication.proposal.questions.append(
            AgentExecutionPlanQuestion(
                question_id="blocking-review",
                prompt="Provide an exact reviewed value.",
                reason="The value cannot be inferred.",
                blocks_proposal=True,
            )
        )
    elif mutation == "missing_artifact":
        publication.proposal.missing_artifacts.append("required_input")
    elif mutation == "effective_options_missing":
        publication.proposal.effective_planner_options.pop(first_task)
    elif mutation == "compiled_options_missing":
        publication.proposal.compiled_task_options.pop(first_task)
    elif mutation == "dispatch_intent_missing":
        publication.proposal.dispatch_intents.clear()
    elif mutation == "artifact_binding_drift":
        publication.proposal.selected_artifacts.append("missing_artifact_binding")
    elif mutation == "profile_binding_drift":
        publication.proposal.selected_profiles.append("missing_profile_binding")
    elif mutation == "budget_expansion":
        publication.proposal.limits = {"max_runtime_sec": 100}
        publication.observation.budget_limits = AgentBudgetObservation(
            status="configured",
            limits={"max_runtime_sec": 10},
            dimensions=["max_runtime_sec"],
        )
    elif mutation == "resource_partial":
        dispatch.execution_route = "remote_execution_service"
        dispatch.remote_task_type = "model_training"

    decision = ScientificAgentPermissionEngine().evaluate(
        publication=publication,
        phase=AgentPermissionPhase.PROPOSAL_REVIEW,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert decision.outcome == AgentPermissionOutcome.DENY
    assert expected_reason in decision.reason_codes


def test_permission_engine_denies_digest_mismatch_and_gate_scope_confusion(
    tmp_path: Path,
) -> None:
    _, proposal_store, proposal = _workspace_with_proposal(tmp_path)
    publication = proposal_store.read(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        verify_current=True,
    )
    engine = ScientificAgentPermissionEngine()
    digest_mismatch = engine.evaluate(
        publication=publication,
        phase=AgentPermissionPhase.PROPOSAL_REVIEW,
        expected_proposal_digest="sha256:" + "f" * 64,
    )
    assert digest_mismatch.outcome == AgentPermissionOutcome.DENY
    assert "PROPOSAL_DIGEST_MISMATCH" in digest_mismatch.reason_codes

    stepwise = engine.evaluate(
        publication=publication,
        phase=AgentPermissionPhase.AUTHORIZATION_CANDIDATE,
        expected_proposal_digest=proposal.proposal_digest,
        authorization_mode=AgentAuthorizationMode.STEPWISE,
        requested_preauthorized_gate_ids=proposal.required_gates,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        client_request_id="stepwise-gate-confusion",
    )
    assert stepwise.outcome == AgentPermissionOutcome.DENY
    assert "STEPWISE_GATE_PREAUTHORIZATION_FORBIDDEN" in stepwise.reason_codes

    outside_plan = engine.evaluate(
        publication=publication,
        phase=AgentPermissionPhase.AUTHORIZATION_CANDIDATE,
        expected_proposal_digest=proposal.proposal_digest,
        authorization_mode=AgentAuthorizationMode.FROZEN_PLAN,
        requested_preauthorized_gate_ids=["gate_1_task_parse"],
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        client_request_id="outside-gate-confusion",
    )
    assert outside_plan.outcome == AgentPermissionOutcome.DENY
    assert "FROZEN_PLAN_GATE_NOT_IN_PLAN" in outside_plan.reason_codes


@pytest.mark.parametrize(
    "semantic_effect",
    ["scientific_confirm", "change_objective", "publish_or_promote"],
)
def test_semantic_effect_gates_can_never_be_plan_preauthorized(
    tmp_path: Path,
    semantic_effect: str,
) -> None:
    _, proposal_store, proposal = _workspace_with_proposal(tmp_path)
    publication = proposal_store.read(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        verify_current=True,
    )
    task_id = publication.proposal.run_plan.tasks[0].task_id
    tool = next(item for item in publication.catalog.tools if item.task_id == task_id)
    tool.effect_class = semantic_effect  # type: ignore[assignment]
    decision = ScientificAgentPermissionEngine().evaluate(
        publication=publication,
        phase=AgentPermissionPhase.AUTHORIZATION_CANDIDATE,
        expected_proposal_digest=proposal.proposal_digest,
        authorization_mode=AgentAuthorizationMode.FROZEN_PLAN,
        requested_preauthorized_gate_ids=proposal.required_gates,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        client_request_id=f"semantic-{semantic_effect.replace('_', '-')}",
    )
    assert decision.outcome == AgentPermissionOutcome.DENY
    assert "SEMANTIC_GATE_CANNOT_BE_PREAUTHORIZED" in decision.reason_codes


def test_shared_gate_aggregates_task_bindings_and_remains_semantic_pending(
    tmp_path: Path,
) -> None:
    gate_id = "gate_2_data_mining"
    registry = AtomicTaskRegistry(
        [
            _visible_task_spec(
                "parse_document",
                effect_class="compute",
                gates=[gate_id],
            ),
            _visible_task_spec(
                "confirm_extracted_dataset",
                effect_class="scientific_confirm",
                gates=[gate_id],
            ),
        ]
    )
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["parse_document", "confirm_extracted_dataset"],
        selected_input_artifact_ids=[],
        task_options={"parse_document": {}, "confirm_extracted_dataset": {}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce a confirmed dataset"],
        rationales=[],
        assumptions=[],
        questions=[],
    )
    storage, proposal_store, proposal = _workspace_with_registry_proposal(
        tmp_path,
        registry=registry,
        response=response,
        request_id="shared-gate-proposal",
    )
    service = _authorization_service(storage, proposal_store)
    review = service.evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert review.outcome == AgentPermissionOutcome.REQUIRE_APPROVAL
    assert "TOOL_CATALOG_BINDING_INVALID" not in review.reason_codes

    authorization = service.authorize(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=_request(proposal, client_request_id="shared-gate-stepwise"),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert authorization.pending_gates == [gate_id]
    assert authorization.preauthorized_operational_gates == []
    assert [
        binding.task_id
        for binding in authorization.gate_bindings
        if binding.gate_id == gate_id
    ] == ["parse_document", "confirm_extracted_dataset"]

    with pytest.raises(ScientificAgentAuthorizationDenied) as exc_info:
        service.authorize(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=_request(
                proposal,
                mode=AgentAuthorizationMode.FROZEN_PLAN,
                client_request_id="shared-gate-frozen",
                requested_gates=[gate_id],
            ),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )
    assert "SEMANTIC_GATE_CANNOT_BE_PREAUTHORIZED" in (
        exc_info.value.decision.reason_codes
    )


def test_permission_complete_hidden_local_dependency_can_be_authorized(
    tmp_path: Path,
) -> None:
    registry = AtomicTaskRegistry(
        [
            _permission_complete_hidden_task("hidden_internal"),
            _visible_task_spec("visible_task", depends_on=["hidden_internal"]),
        ]
    )
    storage, proposal_store, proposal = _workspace_with_registry_proposal(
        tmp_path,
        registry=registry,
        response=_response("visible_task"),
        request_id="hidden-complete-proposal",
    )
    assert [task.task_id for task in proposal.run_plan.tasks] == [
        "hidden_internal",
        "visible_task",
    ]
    assert proposal.effective_planner_options["hidden_internal"] == {}
    assert proposal.compiled_task_options["hidden_internal"] == {}

    service = _authorization_service(storage, proposal_store)
    review = service.evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert review.outcome == AgentPermissionOutcome.REQUIRE_APPROVAL
    assert "TOOL_CATALOG_BINDING_INVALID" not in review.reason_codes
    authorization = service.authorize(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=_request(
            proposal,
            client_request_id="hidden-complete-authorization",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert authorization.task_ids == ["hidden_internal", "visible_task"]


def test_hidden_dependency_without_explicit_permission_metadata_is_denied(
    tmp_path: Path,
) -> None:
    registry = AtomicTaskRegistry(
        [
            AtomicTaskSpec(task_id="hidden_internal", planner_visible=False),
            _visible_task_spec("visible_task", depends_on=["hidden_internal"]),
        ]
    )
    storage, proposal_store, proposal = _workspace_with_registry_proposal(
        tmp_path,
        registry=registry,
        response=_response("visible_task"),
        request_id="hidden-incomplete-proposal",
    )
    decision = _authorization_service(storage, proposal_store).evaluate_permission(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert decision.outcome == AgentPermissionOutcome.DENY
    assert "INTERNAL_TASK_PERMISSION_METADATA_INCOMPLETE" in decision.reason_codes


def test_stepwise_approve_and_start_commits_two_non_executable_authorities(
    tmp_path: Path,
) -> None:
    storage, proposal_store, proposal = _workspace_with_proposal(tmp_path)
    service = _authorization_service(storage, proposal_store)
    project_dir = storage.projects_root / "project-1"
    stage_before = list(project_dir.rglob("stage.json"))
    gate_before = list(project_dir.rglob("gate_decisions.json"))

    result = service.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=_request(proposal),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )

    authorization = result.authorization
    intent = result.start_intent
    assert authorization.executable is False
    assert authorization.authorization_mode == AgentAuthorizationMode.STEPWISE
    assert authorization.preauthorized_operational_gates == []
    assert authorization.pending_gates == proposal.required_gates
    assert authorization.task_ids == [item.task_id for item in proposal.run_plan.tasks]
    assert authorization.effective_planner_options == proposal.effective_planner_options
    assert authorization.compiled_task_options == proposal.compiled_task_options
    assert authorization.dispatch_intents == proposal.dispatch_intents
    assert intent.executable is False
    assert intent.dispatch_state == "not_dispatched"
    assert intent.intent_type == "start_authorized_plan"
    assert intent.handoff_target == "scientific_agent_harness_controller.v1"
    assert result.start_decision.outcome == AgentPermissionOutcome.ALLOW
    assert list(project_dir.rglob("stage.json")) == stage_before
    assert list(project_dir.rglob("gate_decisions.json")) == gate_before
    assert not list(project_dir.rglob("execution_confirmations.json"))
    assert not list(project_dir.rglob("queue_job.json"))
    assert not list(project_dir.rglob("execution_request.json"))

    request_dir = (
        project_dir
        / "agent_plan_control"
        / "requests"
        / "authorization-request-1"
    )
    assert json.loads((request_dir / "reservation.json").read_text())["status"] == "RESERVED"
    assert json.loads((request_dir / "authorization_committed.json").read_text())[
        "status"
    ] == "AUTHORIZATION_COMMITTED"
    assert json.loads((request_dir / "start_intent_committed.json").read_text())[
        "status"
    ] == "START_INTENT_COMMITTED"
    assert service.verify_authorization(
        project_id="project-1",
        authorization_id=authorization.authorization_id,
    ) == authorization
    assert service.verify_start_intent(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
    ) == intent


def test_authorization_binds_exact_selected_artifact_and_invalidates_on_source_drift(
    tmp_path: Path,
) -> None:
    storage, proposal_store, proposal, artifact_path = _workspace_with_artifact_proposal(
        tmp_path
    )
    service = _authorization_service(storage, proposal_store)
    result = service.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=_request(
            proposal,
            client_request_id="artifact-authorization-request",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    binding = result.authorization.artifact_bindings[0]
    assert binding.artifact_id == "uploaded_dataset"
    assert binding.content_digest.startswith("sha256:")
    assert binding.trust_class == "content_bound_input"
    assert binding.producer_task_id is None

    artifact_path.write_text("SMILES,value\nCCN,2.0\n", encoding="utf-8")
    with pytest.raises(ScientificAgentPlanSourceChanged, match="stale"):
        service.verify_authorization(
            project_id="project-1",
            authorization_id=result.authorization.authorization_id,
            verify_current=True,
        )


@pytest.mark.parametrize(
    "drift_phase",
    [
        "after_initial_proposal_read",
        "after_authorization_candidate_decision",
        "after_authorization_checkpoint",
        "after_authorization_commit",
    ],
)
def test_standalone_authorize_never_returns_authority_staled_during_commit(
    tmp_path: Path,
    drift_phase: str,
) -> None:
    storage, proposal_store, proposal, artifact_path = _workspace_with_artifact_proposal(
        tmp_path
    )
    drifted = False

    def drift(phase: str) -> None:
        nonlocal drifted
        if phase == drift_phase and not drifted:
            artifact_path.write_text("SMILES,value\nCCC,3.0\n", encoding="utf-8")
            drifted = True

    control_store = AgentPlanControlStore(storage=storage, fault_injector=drift)
    service = _authorization_service(
        storage,
        proposal_store,
        control_store=control_store,
    )
    with pytest.raises(ScientificAgentPlanSourceChanged):
        service.authorize(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=_request(proposal),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )
    request_dir = (
        storage.projects_root
        / "project-1"
        / "agent_plan_control"
        / "requests"
        / "authorization-request-1"
    )
    assert not (request_dir / "authorization_committed.json").exists()


def test_authorization_staging_rejects_catalog_drift(tmp_path: Path) -> None:
    storage, proposal_store, proposal = _workspace_with_proposal(tmp_path)
    registry = proposal_store.registry
    drifted = False

    def drift(phase: str) -> None:
        nonlocal drifted
        if phase == "after_authorization_checkpoint" and not drifted:
            registry.get("generate_candidates").description = (
                "Changed catalog metadata during authorization staging."
            )
            drifted = True

    service = _authorization_service(
        storage,
        proposal_store,
        control_store=AgentPlanControlStore(storage=storage, fault_injector=drift),
    )
    with pytest.raises(ScientificAgentPlanSourceChanged):
        service.authorize(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=_request(proposal),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )


def test_candidate_decision_rejects_profile_source_drift(tmp_path: Path) -> None:
    storage, proposal_store, proposal, profiles = _workspace_with_profile_source(
        tmp_path
    )
    drifted = False

    def drift(phase: str) -> None:
        nonlocal drifted
        if phase == "after_authorization_candidate_decision" and not drifted:
            profiles.save_connection(
                ConnectionProfile(
                    connection_id="profile-source",
                    ssh_host_alias="profile-source-ssh",
                    expected_hostname="profile-source",
                    remote_root="/srv/profile-source",
                    declared_capabilities=["mineru"],
                )
            )
            drifted = True

    service = _authorization_service(
        storage,
        proposal_store,
        control_store=AgentPlanControlStore(storage=storage, fault_injector=drift),
    )
    with pytest.raises(ScientificAgentPlanSourceChanged):
        service.authorize(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=_request(proposal),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )


def test_authorization_staging_rejects_hidden_permission_metadata_drift(
    tmp_path: Path,
) -> None:
    hidden = _permission_complete_hidden_task("hidden_internal")
    registry = AtomicTaskRegistry(
        [hidden, _visible_task_spec("visible_task", depends_on=["hidden_internal"])]
    )
    storage, proposal_store, proposal = _workspace_with_registry_proposal(
        tmp_path,
        registry=registry,
        response=_response("visible_task"),
        request_id="hidden-metadata-drift-proposal",
    )

    def drift(phase: str) -> None:
        if phase == "after_authorization_checkpoint":
            hidden.verification_policy = ""

    service = _authorization_service(
        storage,
        proposal_store,
        control_store=AgentPlanControlStore(storage=storage, fault_injector=drift),
    )
    with pytest.raises(
        ScientificAgentAuthorizationVerificationError,
        match="candidate changed",
    ):
        service.authorize(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=_request(proposal),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )
    assert not (
        storage.projects_root
        / "project-1"
        / "agent_plan_control"
        / "authorizations"
    ).exists()


@pytest.mark.parametrize(
    "drift_phase",
    ["after_authorization_verification", "after_start_intent_commit"],
)
def test_approve_and_start_never_returns_an_immediately_stale_intent(
    tmp_path: Path,
    drift_phase: str,
) -> None:
    storage, proposal_store, proposal, artifact_path = _workspace_with_artifact_proposal(
        tmp_path
    )
    drifted = False

    def drift(phase: str) -> None:
        nonlocal drifted
        if phase == drift_phase and not drifted:
            artifact_path.write_text("SMILES,value\nCCCl,4.0\n", encoding="utf-8")
            drifted = True

    service = _authorization_service(
        storage,
        proposal_store,
        control_store=AgentPlanControlStore(storage=storage, fault_injector=drift),
    )
    with pytest.raises(ScientificAgentPlanSourceChanged):
        service.approve_and_start(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=_request(proposal),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )
    request_dir = (
        storage.projects_root
        / "project-1"
        / "agent_plan_control"
        / "requests"
        / "authorization-request-1"
    )
    assert not (request_dir / "start_intent_committed.json").exists()


def test_different_request_cannot_create_second_start_intent_for_same_proposal(
    tmp_path: Path,
) -> None:
    storage, proposal_store, proposal = _workspace_with_proposal(tmp_path)
    service = _authorization_service(storage, proposal_store)
    first = service.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=_request(proposal, client_request_id="first-start-request"),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    with pytest.raises(ScientificAgentAuthorizationDenied) as exc_info:
        service.approve_and_start(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=_request(proposal, client_request_id="second-start-request"),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )
    assert "START_INTENT_SLOT_CONFLICT" in exc_info.value.decision.reason_codes
    start_root = (
        storage.projects_root
        / "project-1"
        / "agent_plan_control"
        / "start_intents"
    )
    assert [path.name for path in start_root.iterdir()] == [first.start_intent.start_intent_id]


def test_frozen_plan_allows_empty_current_preauthorization_roster(tmp_path: Path) -> None:
    storage, proposal_store, proposal = _workspace_with_proposal(tmp_path)
    authorization = _authorization_service(storage, proposal_store).authorize(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=_request(
            proposal,
            mode=AgentAuthorizationMode.FROZEN_PLAN,
            client_request_id="frozen-request-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert authorization.authorization_mode == AgentAuthorizationMode.FROZEN_PLAN
    assert authorization.preauthorized_operational_gates == []
    assert authorization.pending_gates == proposal.required_gates


def test_frozen_plan_rejects_non_preauthorizable_registered_gate(tmp_path: Path) -> None:
    storage, proposal_store, proposal = _workspace_with_proposal(tmp_path)
    gate = proposal.required_gates[0]
    with pytest.raises(ScientificAgentAuthorizationDenied) as exc_info:
        _authorization_service(storage, proposal_store).authorize(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=_request(
                proposal,
                mode=AgentAuthorizationMode.FROZEN_PLAN,
                client_request_id="frozen-request-denied",
                requested_gates=[gate],
            ),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )
    assert exc_info.value.decision.outcome == AgentPermissionOutcome.DENY
    assert "GATE_NOT_PREAUTHORIZABLE" in exc_info.value.decision.reason_codes


def test_authorization_request_is_strict_and_confirmation_is_literal_true() -> None:
    digest = "sha256:" + "a" * 64
    base = {
        "expected_proposal_digest": digest,
        "authorization_mode": "stepwise",
        "requested_preauthorized_gate_ids": [],
        "confirmed": True,
        "client_request_id": "request-1",
        "note": "",
    }
    for field, value in (
        ("actor", "alice"),
        ("adapter", "unsafe"),
        ("command", "run"),
        ("ssh", "host"),
        ("path", "relative"),
        ("status", "RUNNING"),
        ("gate_decision", {"approved": True}),
        ("run_plan", {}),
        ("task_options", {}),
        ("external_llm_approved", True),
    ):
        with pytest.raises(ValidationError):
            AgentPlanAuthorizationRequest.model_validate(base | {field: value})
    for value in (False, "true", 1, None):
        with pytest.raises(ValidationError):
            AgentPlanAuthorizationRequest.model_validate(base | {"confirmed": value})


def test_same_request_replays_exact_bytes_and_different_payload_conflicts(
    tmp_path: Path,
) -> None:
    storage, proposal_store, proposal = _workspace_with_proposal(tmp_path)
    service = _authorization_service(storage, proposal_store)
    request = _request(proposal)
    first = service.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=request,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    replay = service.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=request,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert replay.authorization.model_dump(mode="json") == first.authorization.model_dump(mode="json")
    assert replay.start_intent.model_dump(mode="json") == first.start_intent.model_dump(mode="json")

    changed = request.model_copy(update={"note": "different bytes"})
    with pytest.raises(ScientificAgentAuthorizationConflict):
        service.approve_and_start(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=changed,
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )


def test_explicit_shadow_record_is_audit_only(tmp_path: Path) -> None:
    storage, proposal_store, proposal = _workspace_with_proposal(tmp_path)
    service = _authorization_service(storage, proposal_store)
    record = service.evaluate_shadow(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        expected_proposal_digest=proposal.proposal_digest,
    )
    assert record.executable is False
    assert record.new_outcome == AgentPermissionOutcome.REQUIRE_APPROVAL
    assert record.legacy_outcome == AgentPermissionOutcome.REQUIRE_APPROVAL
    assert record.alignment == "MATCH"
    control = storage.projects_root / "project-1" / "agent_plan_control"
    assert not (control / "authorizations").exists()
    assert not (control / "start_intents").exists()


def test_shadow_alignment_comparator_covers_all_outcomes() -> None:
    assert compare_permission_outcomes(
        AgentPermissionOutcome.REQUIRE_APPROVAL,
        AgentPermissionOutcome.REQUIRE_APPROVAL,
    ) == "MATCH"
    assert compare_permission_outcomes(
        AgentPermissionOutcome.DENY,
        AgentPermissionOutcome.ALLOW,
    ) == "NEW_STRICTER"
    assert compare_permission_outcomes(
        AgentPermissionOutcome.ALLOW,
        AgentPermissionOutcome.DENY,
    ) == "NEW_LOOSER"
    assert compare_permission_outcomes(
        AgentPermissionOutcome.ALLOW,
        None,
    ) == "INCOMPARABLE"


@pytest.mark.parametrize(
    "fault_phase",
    [
        "after_authorization_commit",
        "after_start_intent_file_1",
        "after_start_intent_commit",
    ],
)
def test_new_service_recovers_approve_and_start_faults_without_duplicate_authority(
    tmp_path: Path,
    fault_phase: str,
) -> None:
    storage, proposal_store, proposal = _workspace_with_proposal(tmp_path)

    def fault(phase: str) -> None:
        if phase == fault_phase:
            raise RuntimeError(f"simulated crash at {phase}")

    crashing_store = AgentPlanControlStore(storage=storage, fault_injector=fault)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _authorization_service(
            storage,
            proposal_store,
            control_store=crashing_store,
        ).approve_and_start(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=_request(proposal),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )

    recovered = _authorization_service(storage, proposal_store).approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=_request(proposal),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    control = storage.projects_root / "project-1" / "agent_plan_control"
    assert len(list((control / "authorizations").glob("authorization-*"))) == 1
    assert len(list((control / "start_intents").glob("start-intent-*"))) == 1
    assert recovered.start_intent.dispatch_state == "not_dispatched"
    request_dir = control / "requests" / "authorization-request-1"
    assert (request_dir / "authorization_committed.json").is_file()
    assert (request_dir / "start_intent_committed.json").is_file()


def test_cross_process_same_request_creates_one_authorization_and_start_intent(
    tmp_path: Path,
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("cross-process request lock acceptance requires fork")
    storage, _, proposal = _workspace_with_proposal(tmp_path)
    context = multiprocessing.get_context("fork")
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_multiprocess_authorization_worker,
            args=(
                str(storage.workspace_dir),
                proposal.proposal_id,
                proposal.proposal_digest,
            ),
            kwargs={
                "note": "Approve this exact immutable plan.",
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
    assert [item[0] for item in results] == ["success", "success"]
    assert results[0][1:] == results[1][1:]
    control = storage.projects_root / "project-1" / "agent_plan_control"
    assert len(list((control / "authorizations").glob("authorization-*"))) == 1
    assert len(list((control / "start_intents").glob("start-intent-*"))) == 1


def test_cross_process_same_request_different_payload_fails_closed(tmp_path: Path) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("cross-process request lock acceptance requires fork")
    storage, _, proposal = _workspace_with_proposal(tmp_path)
    context = multiprocessing.get_context("fork")
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_multiprocess_authorization_worker,
            args=(
                str(storage.workspace_dir),
                proposal.proposal_id,
                proposal.proposal_digest,
            ),
            kwargs={
                "note": note,
                "start_event": start_event,
                "result_queue": result_queue,
            },
        )
        for note in ("first exact payload", "different payload")
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    results = [result_queue.get(timeout=2) for _ in processes]
    assert sorted(item[0] for item in results) == [
        "ScientificAgentAuthorizationConflict",
        "success",
    ]


def _api_with_proposal(tmp_path: Path, *, trusted_owner: str | None = "alice"):
    from ai4s_agent.app import create_app

    workspace = tmp_path / "api-workspace"
    storage = ProjectStorage(workspace_dir=workspace)
    storage.create_project("project-1", name="Project", created_at=_clock())
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=workspace,
        user_config_dir=tmp_path / "config",
    )
    if trusted_owner is not None:
        app.config["AI4S_AGENT_AUTHORIZATION_OWNER"] = trusted_owner
    client = app.test_client()
    created = client.post(
        "/api/projects/project-1/agent-plan-proposals",
        json={
            "run_id": "run-1",
            "goal": "Prepare a deterministic candidate generation plan",
            "user_constraints": [],
            "client_request_id": "api-proposal-request",
            "llm_provider": {
                "provider": "stub",
                "model": "stub",
                "stub_response": _response().model_dump(mode="json"),
            },
        },
    )
    assert created.status_code == 200, created.get_json()
    return app, client, workspace, created.get_json()["proposal"]


def test_shadow_observation_is_disabled_and_cannot_change_existing_route_or_scientific_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, client, workspace, proposal = _api_with_proposal(tmp_path)
    proposal_dir = (
        workspace
        / "projects"
        / "project-1"
        / "agent_plan_proposals"
        / proposal["proposal_id"]
    )

    def publication_bytes() -> dict[str, bytes]:
        return {
            str(path.relative_to(proposal_dir)): path.read_bytes()
            for path in sorted(proposal_dir.rglob("*"))
            if path.is_file()
        }

    def shadow_failure(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("shadow evaluator must not intercept the legacy route")

    monkeypatch.delenv(AGENT_PERMISSION_SHADOW_OBSERVATION_FLAG, raising=False)
    assert AGENT_PERMISSION_SHADOW_OBSERVATION_FLAG not in os.environ
    before = publication_bytes()
    disabled = client.get(
        f"/api/projects/project-1/agent-plan-proposals/{proposal['proposal_id']}"
    )

    monkeypatch.setattr(
        ScientificAgentAuthorizationService,
        "evaluate_shadow",
        shadow_failure,
    )
    monkeypatch.setenv(AGENT_PERMISSION_SHADOW_OBSERVATION_FLAG, "1")
    enabled_with_failure = client.get(
        f"/api/projects/project-1/agent-plan-proposals/{proposal['proposal_id']}"
    )

    assert disabled.status_code == enabled_with_failure.status_code == 200
    assert disabled.get_data() == enabled_with_failure.get_data()
    assert publication_bytes() == before


def test_project_scoped_api_separates_authorization_and_start_intent(tmp_path: Path) -> None:
    app, client, _, proposal = _api_with_proposal(tmp_path)
    proposal_id = proposal["proposal_id"]
    permission = client.post(
        f"/api/projects/project-1/agent-plan-proposals/{proposal_id}/permission-evaluations",
        json={"expected_proposal_digest": proposal["proposal_digest"]},
    )
    assert permission.status_code == 200
    assert permission.json["outcome"] == "REQUIRE_APPROVAL"

    payload = {
        "expected_proposal_digest": proposal["proposal_digest"],
        "authorization_mode": "stepwise",
        "requested_preauthorized_gate_ids": [],
        "confirmed": True,
        "client_request_id": "api-approve-start-request",
        "note": "approve exact plan",
    }
    app.config.pop("AI4S_AGENT_AUTHORIZATION_OWNER")
    missing_actor = client.post(
        f"/api/projects/project-1/agent-plan-proposals/{proposal_id}/approve-and-start",
        json=payload,
    )
    assert missing_actor.status_code == 403
    spoofed_header = client.post(
        f"/api/projects/project-1/agent-plan-proposals/{proposal_id}/approve-and-start",
        headers={"X-Actor": "searching42"},
        json=payload,
    )
    assert spoofed_header.status_code == 403
    injected_actor = client.post(
        f"/api/projects/project-1/agent-plan-proposals/{proposal_id}/approve-and-start",
        json=payload | {"actor": "mallory"},
    )
    assert injected_actor.status_code == 400

    app.config["AI4S_AGENT_AUTHORIZATION_OWNER"] = "alice"
    approved = client.post(
        f"/api/projects/project-1/agent-plan-proposals/{proposal_id}/approve-and-start",
        json=payload,
    )
    assert approved.status_code == 200, approved.get_json()
    assert approved.json["authorized"] is True
    assert approved.json["start_intent_created"] is True
    assert approved.json["dispatched"] is False
    assert approved.json["authorization"]["actor"] == "alice"
    assert approved.json["authorization"]["actor_source"] == (
        "config:AI4S_AGENT_AUTHORIZATION_OWNER"
    )
    assert approved.json["start_intent"]["dispatch_state"] == "not_dispatched"
    for forbidden in (
        "started",
        "running",
        "job_id",
        "remote_job_id",
        "execution_started",
    ):
        assert forbidden not in approved.json

    authorization_id = approved.json["authorization_id"]
    start_intent_id = approved.json["start_intent_id"]
    fetched_authorization = client.get(
        f"/api/projects/project-1/agent-plan-authorizations/{authorization_id}"
    )
    fetched_intent = client.get(
        f"/api/projects/project-1/agent-plan-start-intents/{start_intent_id}"
    )
    assert fetched_authorization.status_code == 200
    assert fetched_intent.status_code == 200
    assert fetched_intent.json["dispatched"] is False


def test_service_rejects_untrusted_actor_source_before_authorization(
    tmp_path: Path,
) -> None:
    storage, proposal_store, proposal = _workspace_with_proposal(tmp_path)
    service = _authorization_service(storage, proposal_store)
    with pytest.raises(ScientificAgentAuthorizationDenied) as exc_info:
        service.authorize(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            request=_request(
                proposal,
                client_request_id="untrusted-actor-source-request",
            ),
            actor="searching42",
            actor_source="header:X-Actor",
        )
    assert "AUTHORIZATION_ACTOR_UNTRUSTED" in (
        exc_info.value.decision.reason_codes
    )
    assert not (
        storage.projects_root
        / "project-1"
        / "agent_plan_control"
        / "authorizations"
    ).exists()


def test_authenticated_principal_change_conflicts_with_same_request_id(
    tmp_path: Path,
) -> None:
    app, client, _, proposal = _api_with_proposal(tmp_path, trusted_owner="alice")
    payload = {
        "expected_proposal_digest": proposal["proposal_digest"],
        "authorization_mode": "stepwise",
        "requested_preauthorized_gate_ids": [],
        "confirmed": True,
        "client_request_id": "principal-bound-request",
        "note": "",
    }
    first = client.post(
        f"/api/projects/project-1/agent-plan-proposals/{proposal['proposal_id']}/authorizations",
        json=payload,
    )
    assert first.status_code == 200
    assert first.json["authorization"]["actor"] == "alice"

    app.config["AI4S_AGENT_AUTHORIZATION_OWNER"] = "bob"
    changed = client.post(
        f"/api/projects/project-1/agent-plan-proposals/{proposal['proposal_id']}/authorizations",
        headers={"X-Actor": "alice"},
        json=payload,
    )
    assert changed.status_code == 409


def test_api_rejects_explicit_request_schema_or_proposal_content_injection(
    tmp_path: Path,
) -> None:
    _, client, workspace, proposal = _api_with_proposal(tmp_path)
    base = {
        "expected_proposal_digest": proposal["proposal_digest"],
        "authorization_mode": "stepwise",
        "requested_preauthorized_gate_ids": [],
        "confirmed": True,
        "client_request_id": "strict-route-request",
        "note": "",
    }
    for field, value in (
        ("schema_version", "agent_plan_authorization_request.v1"),
        ("proposal_id", proposal["proposal_id"]),
        ("compiled_task_options", proposal["compiled_task_options"]),
        ("dispatch_intents", proposal["dispatch_intents"]),
    ):
        response = client.post(
            f"/api/projects/project-1/agent-plan-proposals/{proposal['proposal_id']}/authorizations",
            headers={"X-Actor": "alice"},
            json=base | {field: value},
        )
        assert response.status_code == 400
        assert response.json["outcome"] == "DENY"
    assert not (
        workspace
        / "projects"
        / "project-1"
        / "agent_plan_control"
        / "authorizations"
    ).exists()


def test_api_fails_closed_after_authorization_or_proposal_byte_replacement(
    tmp_path: Path,
) -> None:
    _, client, workspace, proposal = _api_with_proposal(tmp_path)
    payload = {
        "expected_proposal_digest": proposal["proposal_digest"],
        "authorization_mode": "stepwise",
        "requested_preauthorized_gate_ids": [],
        "confirmed": True,
        "client_request_id": "tamper-request",
        "note": "",
    }
    created = client.post(
        f"/api/projects/project-1/agent-plan-proposals/{proposal['proposal_id']}/authorizations",
        headers={"X-Actor": "alice"},
        json=payload,
    )
    assert created.status_code == 200
    authorization_id = created.json["authorization_id"]
    authorization_path = (
        workspace
        / "projects"
        / "project-1"
        / "agent_plan_control"
        / "authorizations"
        / authorization_id
        / "authorization.json"
    )
    original_authorization = authorization_path.read_bytes()
    authorization_path.write_text("{}\n", encoding="utf-8")
    tampered_authorization = client.get(
        f"/api/projects/project-1/agent-plan-authorizations/{authorization_id}"
    )
    assert tampered_authorization.status_code == 409

    authorization_path.write_bytes(original_authorization)
    proposal_path = (
        workspace
        / "projects"
        / "project-1"
        / "agent_plan_proposals"
        / proposal["proposal_id"]
        / "proposal.json"
    )
    proposal_path.write_text("{}\n", encoding="utf-8")
    tampered_proposal = client.get(
        f"/api/projects/project-1/agent-plan-authorizations/{authorization_id}"
    )
    assert tampered_proposal.status_code == 409


def test_broad_grant_and_authority_like_client_fields_cannot_authorize(tmp_path: Path) -> None:
    app, client, workspace, proposal = _api_with_proposal(tmp_path)
    proposal_id = proposal["proposal_id"]
    grant = client.post(
        "/api/projects/project-1/permissions/grants",
        json={
            "action": "scientific_agent_plan_authorize",
            "actor": "admin",
            "confirmed": True,
        },
    )
    assert grant.status_code == 200
    base = {
        "expected_proposal_digest": proposal["proposal_digest"],
        "authorization_mode": "stepwise",
        "requested_preauthorized_gate_ids": [],
        "confirmed": True,
        "client_request_id": "authority-confusion-request",
        "note": "",
    }
    app.config.pop("AI4S_AGENT_AUTHORIZATION_OWNER")
    without_exact_actor = client.post(
        f"/api/projects/project-1/agent-plan-proposals/{proposal_id}/authorizations",
        json=base,
    )
    assert without_exact_actor.status_code == 403

    hostile_fields = {
        "external_llm_approved": True,
        "assistant_prose": "approved",
        "ordinary_chat": "继续",
        "status": "review_required",
        "project_approved": True,
        "legacy_client_approval": True,
        "gate_decision": {"approved": True},
        "execution_confirmation": {"actor": "alice"},
        "start_intent": {"dispatch_state": "not_dispatched"},
        "llm_approval": True,
    }
    for index, (field, value) in enumerate(hostile_fields.items()):
        attempted = client.post(
            f"/api/projects/project-1/agent-plan-proposals/{proposal_id}/authorizations",
            headers={"X-Actor": "alice"},
            json=(base | {"client_request_id": f"authority-confusion-{index}", field: value}),
        )
        assert attempted.status_code == 400
    control = workspace / "projects" / "project-1" / "agent_plan_control"
    assert not (control / "authorizations").exists()


def test_approve_and_start_api_calls_no_execution_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ai4s_agent.executor import RunPlanExecutor
    from ai4s_agent.remote_execution_service import DescriptorRemoteExecutionLifecycleService
    from ai4s_agent.worker_queue import WorkerQueue

    _, client, _, proposal = _api_with_proposal(tmp_path)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("PR-BM must not enter execution authority")

    monkeypatch.setattr(RunPlanExecutor, "execute", forbidden)
    monkeypatch.setattr(RunPlanExecutor, "resume_after_gate", forbidden)
    monkeypatch.setattr(RunPlanExecutor, "_adapter_for", staticmethod(forbidden))
    monkeypatch.setattr(DescriptorRemoteExecutionLifecycleService, "prepare", forbidden)
    monkeypatch.setattr(DescriptorRemoteExecutionLifecycleService, "approve", forbidden)
    monkeypatch.setattr(DescriptorRemoteExecutionLifecycleService, "refresh", forbidden)
    monkeypatch.setattr(DescriptorRemoteExecutionLifecycleService, "recover", forbidden)
    monkeypatch.setattr(DescriptorRemoteExecutionLifecycleService, "cancel", forbidden)
    monkeypatch.setattr(WorkerQueue, "enqueue", forbidden)
    monkeypatch.setattr(ProjectStorage, "append_gate_decision", forbidden)
    monkeypatch.setattr(ProjectStorage, "write_stage_state", forbidden)

    response = client.post(
        f"/api/projects/project-1/agent-plan-proposals/{proposal['proposal_id']}/approve-and-start",
        headers={"X-Actor": "alice"},
        json={
            "expected_proposal_digest": proposal["proposal_digest"],
            "authorization_mode": "stepwise",
            "requested_preauthorized_gate_ids": [],
            "confirmed": True,
            "client_request_id": "no-call-request",
            "note": "",
        },
    )
    assert response.status_code == 200, response.get_json()
    assert response.json["dispatched"] is False
