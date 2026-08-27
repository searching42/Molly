from __future__ import annotations

from contextlib import nullcontext
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai4s_agent.conversation_store import ConversationStore
from ai4s_agent.app import create_app
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.scientific_agent_authorization import (
    AgentPlanControlStore,
    ScientificAgentAuthorizationService,
)
from ai4s_agent.scientific_agent_conversation import (
    ScientificAgentConversationSessionService,
)
from ai4s_agent.scientific_agent_failure_recovery import (
    FailureRecoveryStore,
    RecoverySuccessorApplicator,
    ScientificAgentRecoverySuccessorApplicator,
)
from ai4s_agent.scientific_agent_failure_recovery_runtime import (
    FailureRecoveryRuntimeEligibility,
    FailureRecoveryRuntimeResult,
    ScientificAgentAutonomyGrantBinding,
    ScientificAgentAutonomyGrantIssuer,
    ScientificAgentAutonomyGrantStore,
    ScientificAgentFailureRecoveryRuntime,
    ScientificAgentFailureRecoveryServiceFactory,
)
from ai4s_agent.scientific_agent_harness_controller import (
    ControllerAdvanceResult,
    ScientificAgentHarnessController,
    ScientificAgentHarnessControllerLeaseBlocked,
)
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
)
from ai4s_agent.schemas import (
    AgentHarnessControllerAction,
    AgentHarnessControllerActionReceipt,
    AgentHarnessControllerDecision,
    AgentHarnessControllerReceiptOutcome,
    AgentHarnessControllerStatus,
    AgentAuthorizationMode,
    AgentEffectCertainty,
    AgentExecutionPlanLLMResponse,
    AgentFailureClass,
    AgentPlanAuthorizationRequest,
    AgentTaskFailureEvidence,
    AutonomyGrant,
    AutonomyParameterBound,
    RunStatus,
    StageState,
    _agent_digest,
)
from ai4s_agent.storage import ProjectStorage


pytestmark = pytest.mark.pr_fast


def _controller_result(
    *,
    controller_execution_id: str = "controller-1",
    execution_digest: str = "sha256:" + "1" * 64,
    inspection_digest: str = "sha256:" + "2" * 64,
    proposal_id: str = "",
    proposal_digest: str = "",
    authorization_id: str = "",
    authorization_digest: str = "",
    start_intent_id: str = "",
    start_intent_digest: str = "",
    status: AgentHarnessControllerStatus = AgentHarnessControllerStatus.FAILED,
    next_action: AgentHarnessControllerAction = AgentHarnessControllerAction.STOP_TASK_TERMINAL,
) -> ControllerAdvanceResult:
    execution = SimpleNamespace(
        project_id="project-1",
        run_id="run-1",
        controller_execution_id=controller_execution_id,
        execution_digest=execution_digest,
        task_slots=[SimpleNamespace(task_id="clean_task", input_artifacts_digest="")],
        authorization_id=authorization_id,
        authorization_digest=authorization_digest,
        proposal_id=proposal_id,
        proposal_digest=proposal_digest,
        start_intent_id=start_intent_id,
        start_intent_digest=start_intent_digest,
    )
    inspection = SimpleNamespace(
        status=status,
        next_action=next_action,
        current_task_id="clean_task",
        current_task_index=0,
        inspection_digest=inspection_digest,
    )
    receipt = SimpleNamespace(
        dispatch_occurred=False,
        outcome=AgentHarnessControllerReceiptOutcome.FAILED,
        receipt_id="controller-receipt",
        receipt_digest="sha256:" + "3" * 64,
    )
    return ControllerAdvanceResult(
        execution=execution,
        inspection=inspection,
        receipt=receipt,
    )


class _TrustedSuccessor(RecoverySuccessorApplicator):
    recovery_authority_chain_verified = True

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_recovery_successor(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "authority_chain_verified": True,
            "successor_proposal_id": "successor-proposal",
            "successor_proposal_digest": "sha256:" + "a" * 64,
            "successor_permission_decision_id": "successor-permission",
            "successor_permission_decision_digest": "sha256:" + "b" * 64,
            "successor_authorization_id": "successor-authorization",
            "successor_authorization_digest": "sha256:" + "c" * 64,
            "successor_start_intent_id": "successor-start-intent",
            "successor_start_intent_digest": "sha256:" + "d" * 64,
            "successor_controller_execution_id": "controller-successor",
            "successor_controller_execution_digest": "sha256:" + "e" * 64,
            "effect_started": True,
            "effect_receipt_id": "effect-receipt",
            "effect_receipt_digest": "sha256:" + "f" * 64,
        }


@dataclass
class _GrantSource:
    grant: AutonomyGrant | None
    epoch: str = "authority-1"

    def resolve_current(self, **_kwargs):
        if self.grant is None:
            return None
        return ScientificAgentAutonomyGrantBinding(
            grant=self.grant,
            authority_epoch=self.epoch,
        )


class _Controller:
    def __init__(self, baseline: ControllerAdvanceResult) -> None:
        self.baseline = baseline
        self.successor = _controller_result(
            controller_execution_id="controller-successor",
            execution_digest="sha256:" + "e" * 64,
            proposal_id="successor-proposal",
            proposal_digest="sha256:" + "a" * 64,
            authorization_id="successor-authorization",
            authorization_digest="sha256:" + "b" * 64,
            start_intent_id="successor-start-intent",
            start_intent_digest="sha256:" + "c" * 64,
            status=AgentHarnessControllerStatus.ACTIVE,
            next_action=AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
        )

    def read_execution_agent_snapshot(self, **kwargs):
        if kwargs["controller_execution_id"] == self.baseline.execution.controller_execution_id:
            return self.baseline
        return self.successor

    def get(self, **kwargs):
        return self.read_execution_agent_snapshot(**kwargs)


class _NoRemoteAuthority:
    def current_authority(self, **_kwargs):
        raise AssertionError("the concrete recovery fixture must remain local")


class _NoRemoteLifecycle:
    pass


def _grant() -> AutonomyGrant:
    return AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["clean_task"],
        max_retries=1,
        valid_until="9999-12-31T23:59:59Z",
    )


def _write_typed_failure(
    storage: ProjectStorage,
    evidence: AgentTaskFailureEvidence,
) -> None:
    storage.write_stage_state(
        "project-1",
        "run-1",
        StageState(
            stage="clean_task",
            status=RunStatus.FAILED,
            started_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            error={"code": "typed_failure"},
            details={"typed_failure_evidence": evidence.model_dump(mode="json")},
        ),
    )


def _run_concurrent_runtime(
    workspace: str,
    grant_payload: dict,
    start_event,
    result_queue,
) -> None:
    """Resume one exact failed Conversation session in a fresh process."""

    storage = ProjectStorage(workspace_dir=Path(workspace))
    # Validate the serialized server grant before opening the child runtime;
    # resolution itself still comes exclusively from the shared grant store.
    AutonomyGrant.model_validate(grant_payload)
    baseline = _controller_result()
    controller = _Controller(baseline)
    successor = _TrustedSuccessor()
    factory = ScientificAgentFailureRecoveryServiceFactory(
        storage=storage,
        controller=controller,
        replanner=None,
        successor_applicator=successor,
    )
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=ScientificAgentAutonomyGrantStore(storage=storage),
        service_factory=factory,
        store=factory.store,
    )
    start_event.wait(10)
    try:
        result = runtime.continue_failed(
            project_id="project-1",
            conversation_id="conversation-1",
            run_id="run-1",
            state={},
            controller_result=baseline,
            provider=None,
        )
    except BaseException as exc:  # communicate child failures to the parent
        result_queue.put(("error", type(exc).__name__, str(exc)))
    else:
        recovery = result.recovery
        result_queue.put(
            (
                "ok",
                result.eligibility.value,
                recovery.receipt.receipt_id if recovery is not None else "",
                recovery.receipt.effect_started if recovery is not None else False,
                recovery.replayed if recovery is not None else False,
                len(successor.calls),
            )
        )


def test_runtime_no_grant_is_a_zero_effect_boundary(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    baseline = _controller_result()
    controller = _Controller(baseline)
    successor = _TrustedSuccessor()

    class _Factory:
        def __init__(self) -> None:
            self.calls = 0

        def build(self, **_kwargs):
            self.calls += 1
            raise AssertionError("no-grant recovery must not build a foundation service")

    factory = _Factory()
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=_GrantSource(None),
        service_factory=factory,
    )
    result = runtime.continue_failed(
        project_id="project-1",
        conversation_id="conversation-1",
        run_id="run-1",
        state={},
        controller_result=baseline,
        provider=None,
    )
    assert result.eligibility is FailureRecoveryRuntimeEligibility.REQUIRE_HUMAN
    assert result.reason_code == "RECOVERY_AUTONOMY_GRANT_REQUIRED"
    assert result.provider_calls_total == 0
    assert result.effect_count_total == 0
    assert factory.calls == 0
    assert successor.calls == []


def test_production_app_wires_concrete_successor_and_opt_in_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI4S_AGENT_FAILURE_RECOVERY_ENABLED", "true")
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "user-config",
    )
    successor = app.extensions["scientific_agent_failure_recovery_successor"]
    runtime = app.extensions["scientific_agent_failure_recovery_runtime"]
    issuer = app.extensions["scientific_agent_failure_recovery_grant_issuer"]
    service = app.extensions["scientific_agent_conversation_session_service"]
    assert isinstance(successor, ScientificAgentRecoverySuccessorApplicator)
    assert successor.recovery_authority_chain_verified is True
    assert service.failure_recovery_runtime is runtime
    assert service.failure_recovery_enabled is True
    assert isinstance(issuer, ScientificAgentAutonomyGrantIssuer)
    assert app.extensions["scientific_agent_authorization_service"].autonomy_grant_issuer is not None
    factory = app.extensions["scientific_agent_failure_recovery_service_factory"]
    built = factory.build(
        provider=None,
        grant=_grant(),
        session_id="recovery-test-session",
        authority_epoch="recovery-test-epoch",
    )
    assert callable(built.effect_reconciler)


def test_production_authority_issuance_persists_recovery_grant_without_manual_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The app's normal approval flow is the grant publication point."""

    monkeypatch.setenv("AI4S_AGENT_FAILURE_RECOVERY_ENABLED", "true")
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "user-config",
        scientific_task_registry=AtomicTaskRegistry(),
    )
    app.config["AI4S_AGENT_AUTHORIZATION_OWNER"] = "alice"
    client = app.test_client()
    created = client.post(
        "/api/projects",
        json={"project_id": "project-1", "name": "Project"},
    )
    assert created.status_code == 200, created.get_json()
    storage = app.extensions["conversation_store"].projects
    run_dir = storage.run_dir("project-1", "run-1")
    input_path = run_dir / "inputs" / "dataset.csv"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("SMILES,value\nCCO,1.0\n", encoding="utf-8")
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
        success_criteria=["produce a reviewable profile"],
        rationales=["Use the registered local inspection task."],
        assumptions=[],
        questions=[],
    )
    proposed = client.post(
        "/api/projects/project-1/agent-plan-proposals",
        json={
            "run_id": "run-1",
            "goal": "Inspect one exact dataset",
            "user_constraints": [],
            "client_request_id": "authority-issuance-proposal",
            "llm_provider": {
                "provider": "stub",
                "model": "stub",
                "stub_response": response.model_dump(mode="json"),
            },
        },
    )
    assert proposed.status_code == 200, proposed.get_json()
    proposal = proposed.get_json()["proposal"]
    approved = client.post(
        "/api/projects/project-1/agent-plan-proposals/"
        f"{proposal['proposal_id']}/approve-and-start",
        json={
            "expected_proposal_digest": proposal["proposal_digest"],
            "authorization_mode": "stepwise",
            "requested_preauthorized_gate_ids": [],
            "confirmed": True,
            "client_request_id": "authority-issuance-approval",
            "note": "Approve the exact local inspection plan.",
        },
    )
    assert approved.status_code == 200, approved.get_json()
    binding = app.extensions[
        "scientific_agent_failure_recovery_grant_store"
    ].resolve_current(project_id="project-1", run_id="run-1")
    assert binding is not None
    assert binding.grant.max_retries == 1
    assert binding.grant.max_replans == 1
    assert binding.authority_epoch.startswith("authorization-epoch-")

    # Continue through the app's real Controller and recovery runtime using
    # the grant just issued above; no test-only publish_server_grant call is
    # allowed to make this path eligible.
    controller = app.extensions["scientific_agent_harness_controller"]
    control_store = app.extensions["scientific_agent_plan_control_store"]
    proposal_store = app.extensions["scientific_agent_plan_proposal_store"]
    authorization_service = app.extensions["scientific_agent_authorization_service"]
    authorization = authorization_service.verify_authorization(
        project_id="project-1",
        authorization_id=approved.get_json()["authorization_id"],
        verify_current=False,
    )
    start_intent = authorization_service.verify_start_intent(
        project_id="project-1",
        start_intent_id=approved.get_json()["start_intent_id"],
        verify_current=False,
    )
    publication = proposal_store.read(
        project_id="project-1",
        proposal_id=proposal["proposal_id"],
        verify_current=False,
    )
    permission = control_store.read_permission_decision(
        project_id="project-1",
        decision_id=start_intent.permission_decision_id,
    )
    execution = controller._build_execution(
        intent=start_intent,
        authorization=authorization,
        publication=publication,
        permission=permission,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        client_request_id="app-production-failure-controller",
        request_digest=_agent_digest({"fixture": "app-production-failure-controller"}),
        created_at=authorization.created_at,
    )
    control_store.publish_harness_controller_execution(execution)
    evidence = AgentTaskFailureEvidence(
        failure_code="controller_pre_effect_failure",
        failure_class=AgentFailureClass.TRANSIENT,
        effect_certainty=AgentEffectCertainty.NO_EFFECT_CONFIRMED,
        task_id="inspect_dataset",
        logical_tool_id="inspect_dataset",
        reason_codes=["CONTROLLER_PRE_EFFECT_FAILURE"],
    )
    storage.write_stage_state(
        "project-1",
        "run-1",
        StageState(
            stage="inspect_dataset",
            status=RunStatus.FAILED,
            started_at=authorization.created_at,
            updated_at=authorization.created_at,
            error={"code": "typed_pre_effect_failure"},
            details={"typed_failure_evidence": evidence.model_dump(mode="json")},
        ),
    )
    failed_inspection = controller._inspect(execution, verify_authority=False)
    source_bindings = controller._bindings_from_facts(failed_inspection.facts)
    source_bindings_digest = _agent_digest(
        [item.model_dump(mode="json") for item in source_bindings]
    )
    failed_decision = AgentHarnessControllerDecision(
        controller_execution_id=execution.controller_execution_id,
        controller_execution_digest=execution.execution_digest,
        client_request_id="app-production-failure-decision",
        inspection_digest=failed_inspection.inspection_digest,
        action_kind=AgentHarnessControllerAction.STOP_TASK_TERMINAL,
        task_id="inspect_dataset",
        task_index=0,
        attempt_ordinal=0,
        slot_id=execution.task_slots[0].slot_id,
        source_bindings=source_bindings,
        source_bindings_digest=source_bindings_digest,
        reason_codes=["TERMINAL_OBSERVED"],
        created_at=authorization.created_at,
        executable=False,
    )
    control_store.publish_harness_controller_decision(
        project_id="project-1",
        decision=failed_decision,
    )
    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    registry_digest = _agent_digest(storage.read_artifact_registry("project-1", "run-1"))
    failed_receipt = AgentHarnessControllerActionReceipt(
        controller_execution_id=execution.controller_execution_id,
        controller_execution_digest=execution.execution_digest,
        decision_id=failed_decision.decision_id,
        decision_digest=failed_decision.decision_digest,
        action_kind=AgentHarnessControllerAction.STOP_TASK_TERMINAL,
        task_id="inspect_dataset",
        task_index=0,
        attempt_ordinal=0,
        slot_id=execution.task_slots[0].slot_id,
        execution_started=False,
        dispatch_occurred=False,
        before_stage_digest=controller._stage_digest(stage),
        after_stage_digest=controller._stage_digest(stage),
        before_artifact_registry_digest=registry_digest,
        after_artifact_registry_digest=registry_digest,
        outcome=AgentHarnessControllerReceiptOutcome.FAILED,
        status_after=AgentHarnessControllerStatus.FAILED,
        source_bindings=source_bindings,
        source_bindings_digest=source_bindings_digest,
        reason_codes=["TERMINAL_OBSERVED"],
        created_at=authorization.created_at,
    )
    control_store.publish_harness_controller_action_receipt(
        project_id="project-1",
        receipt=failed_receipt,
    )
    conversations = app.extensions["conversation_store"]
    conversations.create_conversation(
        "project-1",
        conversation_id="app-recovery-conversation",
        title="App recovery",
    )
    service = app.extensions["scientific_agent_conversation_session_service"]
    baseline = controller.read_execution_agent_snapshot(
        project_id="project-1",
        controller_execution_id=execution.controller_execution_id,
        expected_controller_execution_digest=execution.execution_digest,
    )
    _result, state, stop_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="app-recovery-conversation",
        state=service._default_state("project-1", "app-recovery-conversation"),
        controller_result=baseline,
        provider=object(),
        provider_binding_digest="request-provider-must-not-cross-recovery",
    )
    assert stop_reason == "RECOVERY_SUCCESSOR_COMMITTED"
    assert state["last_recovery_action"] == "RETRY_EXACT"
    assert state["last_recovery_retry_ordinal"] == 1
    assert state["recovery_effect_count"] == 1
    assert state["controller_execution_id"] != execution.controller_execution_id


def test_runtime_unknown_effect_is_deterministic_ask_user_without_provider_or_effect(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    baseline = _controller_result(
        status=AgentHarnessControllerStatus.RECOVERY_REQUIRED,
        next_action=AgentHarnessControllerAction.RECOVER_REMOTE_TASK,
    )
    controller = _Controller(baseline)
    successor = _TrustedSuccessor()

    class CountingProvider:
        calls = 0

        def complete_json(self, **_kwargs):
            self.calls += 1
            raise AssertionError("UNKNOWN_EFFECT must not call the provider")

    provider = CountingProvider()
    factory = ScientificAgentFailureRecoveryServiceFactory(
        storage=storage,
        controller=controller,
        replanner=None,
        successor_applicator=successor,
    )
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=_GrantSource(_grant()),
        service_factory=factory,
        store=factory.store,
    )
    result = runtime.continue_failed(
        project_id="project-1",
        conversation_id="conversation-1",
        run_id="run-1",
        state={},
        controller_result=baseline,
        provider=provider,
    )
    assert result.eligibility is FailureRecoveryRuntimeEligibility.ELIGIBLE
    assert result.recovery is not None
    assert result.recovery.decision.recovery_action.value == "ASK_USER"
    assert result.effect_certainty.value == "EFFECT_UNKNOWN"
    assert result.provider_calls_total == 0
    assert result.effect_count_total == 0
    assert provider.calls == 0
    assert successor.calls == []


def test_conversation_unknown_effect_enters_durable_recovery_boundary(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    conversations = ConversationStore(projects=storage)
    conversations.create_conversation(
        "project-1",
        conversation_id="conversation-1",
        title="Recovery",
    )
    baseline = _controller_result(
        status=AgentHarnessControllerStatus.RECOVERY_REQUIRED,
        next_action=AgentHarnessControllerAction.RECOVER_REMOTE_TASK,
    )
    controller = _Controller(baseline)
    successor = _TrustedSuccessor()

    class CountingProvider:
        calls = 0

        def complete_json(self, **_kwargs):
            self.calls += 1
            raise AssertionError("UNKNOWN_EFFECT must not call the provider")

    provider = CountingProvider()
    factory = ScientificAgentFailureRecoveryServiceFactory(
        storage=storage,
        controller=controller,
        replanner=None,
        successor_applicator=successor,
    )
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=_GrantSource(_grant()),
        service_factory=factory,
        store=factory.store,
    )
    service = ScientificAgentConversationSessionService(
        projects=storage,
        conversations=conversations,
        plan_service=None,
        proposal_store=None,
        authorization_service=None,
        controller=controller,
        execution_agent=None,
        failure_recovery_runtime=runtime,
        failure_recovery_enabled=True,
    )
    _result, state, stop_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=service._default_state("project-1", "conversation-1"),
        controller_result=baseline,
        provider=provider,
        provider_binding_digest="",
    )
    assert stop_reason == "RECOVERY_UNKNOWN_EFFECT"
    assert state["status"] == "recovery_required"
    assert state["reason_code"] == "RECOVERY_UNKNOWN_EFFECT"
    assert state["last_recovery_failure_class"] == AgentFailureClass.UNKNOWN_EFFECT.value
    assert state["last_recovery_effect_certainty"] == AgentEffectCertainty.EFFECT_UNKNOWN.value
    assert state["recovery_provider_calls"] == 0
    assert state["recovery_effect_count"] == 0
    assert provider.calls == 0
    assert successor.calls == []


def test_conversation_parameter_recovery_uses_one_bounded_tool_call(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    conversations = ConversationStore(projects=storage)
    conversations.create_conversation(
        "project-1",
        conversation_id="conversation-1",
        title="Recovery",
    )
    baseline = _controller_result()
    controller = _Controller(baseline)
    successor = _TrustedSuccessor()
    provider = StubLLMProvider(
        response={
            "action": "TOOL_CALL",
            "logical_tool_id": "clean_task",
            "arguments": {"min_nonempty": 5},
        }
    )
    _write_typed_failure(
        storage,
        AgentTaskFailureEvidence(
            failure_code="parameter_validation_failed",
            failure_class=AgentFailureClass.PARAMETER_RECOVERABLE,
            effect_certainty=AgentEffectCertainty.NO_EFFECT_CONFIRMED,
            task_id="clean_task",
            logical_tool_id="clean_task",
            reason_codes=["PARAMETER_VALIDATION_FAILED"],
        ),
    )
    factory = ScientificAgentFailureRecoveryServiceFactory(
        storage=storage,
        controller=controller,
        replanner=None,
        successor_applicator=successor,
        tool_schemas={
            "clean_task": {
                "type": "object",
                "properties": {
                    "min_nonempty": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                    }
                },
                "required": ["min_nonempty"],
                "additionalProperties": False,
            }
        },
    )
    parameter_grant = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["clean_task"],
        parameter_bounds={
            "clean_task.min_nonempty": AutonomyParameterBound(
                minimum=0,
                maximum=10,
            )
        },
        max_retries=1,
        valid_until="9999-12-31T23:59:59Z",
    )
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=_GrantSource(parameter_grant),
        service_factory=factory,
        store=factory.store,
        recovery_provider_resolver=lambda: nullcontext(provider),
    )
    service = ScientificAgentConversationSessionService(
        projects=storage,
        conversations=conversations,
        plan_service=None,
        proposal_store=None,
        authorization_service=None,
        controller=controller,
        execution_agent=None,
        failure_recovery_runtime=runtime,
        failure_recovery_enabled=True,
    )
    _result, state, stop_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=service._default_state("project-1", "conversation-1"),
        controller_result=baseline,
        provider=provider,
        provider_binding_digest="",
    )
    assert stop_reason == "RECOVERY_SUCCESSOR_COMMITTED"
    assert state["status"] == "running"
    assert state["last_recovery_action"] == "TOOL_CALL"
    assert state["last_recovery_retry_ordinal"] == 1
    assert state["recovery_provider_calls"] == 1
    assert state["recovery_effect_count"] == 1
    assert len(successor.calls) == 1
    assert successor.calls[0]["decision"].selected_arguments == {"min_nonempty": 5}


def test_recovery_uses_server_provider_after_failed_boundary_not_request_provider(
    tmp_path: Path,
) -> None:
    """A provider selected for an ordinary turn cannot cross a mid-turn failure."""

    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    conversations = ConversationStore(projects=storage)
    conversations.create_conversation(
        "project-1",
        conversation_id="conversation-1",
        title="Recovery",
    )
    baseline = _controller_result()
    controller = _Controller(baseline)
    successor = _TrustedSuccessor()

    class CountingProvider:
        def __init__(self, response: dict) -> None:
            self.calls = 0
            self._delegate = StubLLMProvider(response=response)

        def complete_json(self, **kwargs):
            self.calls += 1
            return self._delegate.complete_json(**kwargs)

    request_provider = CountingProvider(
        {
            "action": "TOOL_CALL",
            "logical_tool_id": "clean_task",
            "arguments": {"min_nonempty": 5},
        }
    )
    server_provider = CountingProvider(
        {
            "action": "TOOL_CALL",
            "logical_tool_id": "clean_task",
            "arguments": {"min_nonempty": 5},
        }
    )
    _write_typed_failure(
        storage,
        AgentTaskFailureEvidence(
            failure_code="parameter_validation_failed",
            failure_class=AgentFailureClass.PARAMETER_RECOVERABLE,
            effect_certainty=AgentEffectCertainty.NO_EFFECT_CONFIRMED,
            task_id="clean_task",
            logical_tool_id="clean_task",
            reason_codes=["PARAMETER_VALIDATION_FAILED"],
        ),
    )
    factory = ScientificAgentFailureRecoveryServiceFactory(
        storage=storage,
        controller=controller,
        replanner=None,
        successor_applicator=successor,
        tool_schemas={
            "clean_task": {
                "type": "object",
                "properties": {
                    "min_nonempty": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                    }
                },
                "required": ["min_nonempty"],
                "additionalProperties": False,
            }
        },
    )
    parameter_grant = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["clean_task"],
        parameter_bounds={
            "clean_task.min_nonempty": AutonomyParameterBound(
                minimum=0,
                maximum=10,
            )
        },
        max_retries=1,
        valid_until="9999-12-31T23:59:59Z",
    )
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=_GrantSource(parameter_grant),
        service_factory=factory,
        store=factory.store,
        recovery_provider_resolver=lambda: nullcontext(server_provider),
    )
    service = ScientificAgentConversationSessionService(
        projects=storage,
        conversations=conversations,
        plan_service=None,
        proposal_store=None,
        authorization_service=None,
        controller=controller,
        execution_agent=None,
        failure_recovery_runtime=runtime,
        failure_recovery_enabled=True,
    )
    _result, state, stop_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=service._default_state("project-1", "conversation-1"),
        controller_result=baseline,
        provider=request_provider,
        provider_binding_digest="request-provider-binding",
    )
    assert stop_reason == "RECOVERY_SUCCESSOR_COMMITTED"
    assert state["recovery_provider_calls"] == 1
    assert request_provider.calls == 0
    assert server_provider.calls == 1


def test_conversation_replan_publishes_successor_for_explicit_review_once(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    conversations = ConversationStore(projects=storage)
    conversations.create_conversation(
        "project-1",
        conversation_id="conversation-1",
        title="Recovery",
    )
    baseline = _controller_result()
    controller = _Controller(baseline)
    replan_grant = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["clean_task"],
        max_retries=0,
        max_replans=1,
        valid_until="9999-12-31T23:59:59Z",
    )
    _write_typed_failure(
        storage,
        AgentTaskFailureEvidence(
            failure_code="input_evidence_missing",
            failure_class=AgentFailureClass.INPUT_EVIDENCE_INSUFFICIENT,
            effect_certainty=AgentEffectCertainty.NO_EFFECT_CONFIRMED,
            task_id="clean_task",
            logical_tool_id="clean_task",
            reason_codes=["INPUT_EVIDENCE_MISSING"],
        ),
    )

    class Replanner:
        calls = 0

        def create_current_controller_failure_revision(self, **_kwargs):
            self.calls += 1
            return {
                "successor_proposal_id": "replan-successor",
                "successor_proposal_digest": "sha256:" + "9" * 64,
            }

    replanner = Replanner()
    factory = ScientificAgentFailureRecoveryServiceFactory(
        storage=storage,
        controller=controller,
        replanner=replanner,
        successor_applicator=None,
    )
    replan_provider = StubLLMProvider(response={"action": "REPLAN"})
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=_GrantSource(replan_grant),
        service_factory=factory,
        store=factory.store,
        recovery_provider_resolver=lambda: nullcontext(replan_provider),
    )
    service = ScientificAgentConversationSessionService(
        projects=storage,
        conversations=conversations,
        plan_service=None,
        proposal_store=None,
        authorization_service=None,
        controller=controller,
        execution_agent=None,
        failure_recovery_runtime=runtime,
        failure_recovery_enabled=True,
    )
    state = service._default_state("project-1", "conversation-1")
    _result, state, stop_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=state,
        controller_result=baseline,
        provider=replan_provider,
        provider_binding_digest="",
    )
    assert stop_reason == "RECOVERY_REPLAN_REVIEW_REQUIRED"
    assert state["status"] == "approval_required"
    assert state["reason_code"] == "RECOVERY_REPLAN_REVIEW_REQUIRED"
    assert state["proposal_id"] == "replan-successor"
    assert state["proposal_digest"] == "sha256:" + "9" * 64
    assert state["authorization_id"] == ""
    assert state["controller_execution_id"] == ""
    assert state["last_recovery_replan_ordinal"] == 1
    assert state["recovery_provider_calls"] == 1
    assert state["recovery_effect_count"] == 0
    assert replanner.calls == 1

    # Replaying the same failed state returns the immutable receipt and never
    # calls the existing Replanner a second time.
    _result, replay_state, replay_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=state,
        controller_result=baseline,
        provider=replan_provider,
        provider_binding_digest="",
    )
    assert replay_reason == "RECOVERY_REPLAN_REVIEW_REQUIRED"
    assert replay_state["last_recovery_receipt_id"] == state["last_recovery_receipt_id"]
    assert replay_state["recovery_provider_calls"] == 1
    assert replanner.calls == 1


def test_runtime_expired_grant_fails_closed_before_foundation(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    baseline = _controller_result()
    controller = _Controller(baseline)

    class _Factory:
        calls = 0

        def build(self, **_kwargs):
            self.calls += 1
            raise AssertionError("expired grant must not build a foundation service")

    factory = _Factory()
    expired = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["clean_task"],
        max_retries=1,
        valid_until="2020-01-01T00:00:00Z",
    )
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=_GrantSource(expired),
        service_factory=factory,
    )
    result = runtime.continue_failed(
        project_id="project-1",
        conversation_id="conversation-1",
        run_id="run-1",
        state={},
        controller_result=baseline,
        provider=None,
    )
    assert result.eligibility is FailureRecoveryRuntimeEligibility.FAIL_CLOSED
    assert result.reason_code == "RECOVERY_AUTONOMY_GRANT_STALE"
    assert result.provider_calls_total == 0
    assert result.effect_count_total == 0
    assert factory.calls == 0


def test_expired_lease_stops_failed_recovery_before_provider_or_retry(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    baseline = _controller_result()
    controller = _Controller(baseline)

    def deny_lease(**_kwargs):
        raise ScientificAgentHarnessControllerLeaseBlocked(
            "AUTONOMY_LEASE_EXPIRED"
        )

    controller.verify_autonomy_lease = deny_lease  # type: ignore[attr-defined]
    factory = ScientificAgentFailureRecoveryServiceFactory(
        storage=storage,
        controller=controller,
        replanner=None,
        successor_applicator=_TrustedSuccessor(),
    )
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=_GrantSource(_grant()),
        service_factory=factory,
        store=factory.store,
    )
    conversations = ConversationStore(projects=storage)
    conversations.create_conversation(
        "project-1",
        conversation_id="conversation-1",
        title="Lease recovery",
    )
    service = ScientificAgentConversationSessionService(
        projects=storage,
        conversations=conversations,
        plan_service=None,
        proposal_store=None,
        authorization_service=None,
        controller=controller,
        execution_agent=None,
        failure_recovery_runtime=runtime,
        failure_recovery_enabled=True,
    )

    class CountingProvider:
        calls = 0

        def complete_json(self, **_kwargs):
            self.calls += 1
            raise AssertionError("expired lease must block recovery provider")

    provider = CountingProvider()
    _result, state, stop_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=service._default_state("project-1", "conversation-1"),
        controller_result=baseline,
        provider=provider,
        provider_binding_digest="",
    )
    assert stop_reason == "lease_blocked"
    assert state["reason_code"] == "AUTONOMY_LEASE_EXPIRED"
    assert state["status"] == "recovery_required"
    assert provider.calls == 0


def test_runtime_retry_replays_one_receipt_without_second_effect(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    baseline = _controller_result()
    controller = _Controller(baseline)
    successor = _TrustedSuccessor()
    factory = ScientificAgentFailureRecoveryServiceFactory(
        storage=storage,
        controller=controller,
        replanner=None,
        successor_applicator=successor,
    )
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=_GrantSource(_grant()),
        service_factory=factory,
        store=factory.store,
    )
    first = runtime.continue_failed(
        project_id="project-1",
        conversation_id="conversation-1",
        run_id="run-1",
        state={},
        controller_result=baseline,
        provider=None,
    )
    state = {
        "last_recovery_failure_id": first.observation.failure_id,
        "last_recovery_failure_digest": first.observation.failure_digest,
        "recovery_provider_calls": first.provider_calls_total,
        "recovery_effect_count": first.effect_count_total,
    }
    replay = runtime.continue_failed(
        project_id="project-1",
        conversation_id="conversation-1",
        run_id="run-1",
        state=state,
        controller_result=baseline,
        provider=None,
    )
    assert first.recovery is not None
    assert replay.recovery is not None
    assert first.recovery.receipt.receipt_id == replay.recovery.receipt.receipt_id
    assert replay.recovery.replayed is True
    assert replay.provider_calls_total == 0
    assert replay.effect_count_total == 1
    assert len(successor.calls) == 1


def test_conversation_retry_budget_survives_a_failed_successor(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    conversations = ConversationStore(projects=storage)
    conversations.create_conversation(
        "project-1",
        conversation_id="conversation-1",
        title="Recovery",
    )
    baseline = _controller_result()
    controller = _Controller(baseline)
    successor = _TrustedSuccessor()
    factory = ScientificAgentFailureRecoveryServiceFactory(
        storage=storage,
        controller=controller,
        replanner=None,
        successor_applicator=successor,
    )
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=_GrantSource(_grant()),
        service_factory=factory,
        store=factory.store,
    )
    service = ScientificAgentConversationSessionService(
        projects=storage,
        conversations=conversations,
        plan_service=None,
        proposal_store=None,
        authorization_service=None,
        controller=controller,
        execution_agent=None,
        failure_recovery_runtime=runtime,
        failure_recovery_enabled=True,
    )
    _result, first_state, first_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=service._default_state("project-1", "conversation-1"),
        controller_result=baseline,
        provider=None,
        provider_binding_digest="",
    )
    assert first_reason == "RECOVERY_SUCCESSOR_COMMITTED"
    assert first_state["recovery_effect_count"] == 1

    # The successor execution is a new failure identity, but the aggregate
    # session/grant/epoch remains the same and therefore has no retry slot.
    failed_successor = _controller_result(
        controller_execution_id="controller-successor",
        execution_digest="sha256:" + "e" * 64,
        proposal_id="successor-proposal",
        proposal_digest="sha256:" + "a" * 64,
        authorization_id="successor-authorization",
        authorization_digest="sha256:" + "b" * 64,
        start_intent_id="successor-start-intent",
        start_intent_digest="sha256:" + "c" * 64,
        status=AgentHarnessControllerStatus.FAILED,
    )
    controller.successor = failed_successor
    _result, second_state, second_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=first_state,
        controller_result=failed_successor,
        provider=StubLLMProvider(response={"action": "RETRY_EXACT"}),
        provider_binding_digest="",
    )
    assert second_reason == "RECOVERY_BUDGET_EXHAUSTED"
    assert second_state["status"] == "recovery_required"
    assert second_state["reason_code"] == "RECOVERY_BUDGET_EXHAUSTED"
    assert second_state["recovery_provider_calls"] == 0
    assert second_state["recovery_effect_count"] == 1
    assert len(successor.calls) == 1


def test_conversation_replan_budget_exhaustion_stops_before_provider_or_replanner(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    conversations = ConversationStore(projects=storage)
    conversations.create_conversation(
        "project-1",
        conversation_id="conversation-1",
        title="Recovery",
    )
    baseline = _controller_result()
    controller = _Controller(baseline)
    _write_typed_failure(
        storage,
        AgentTaskFailureEvidence(
            failure_code="input_evidence_missing",
            failure_class=AgentFailureClass.INPUT_EVIDENCE_INSUFFICIENT,
            effect_certainty=AgentEffectCertainty.NO_EFFECT_CONFIRMED,
            task_id="clean_task",
            logical_tool_id="clean_task",
            reason_codes=["INPUT_EVIDENCE_MISSING"],
        ),
    )

    class Replanner:
        calls = 0

        def create_current_controller_failure_revision(self, **_kwargs):
            self.calls += 1
            raise AssertionError("exhausted replan budget must not call Replanner")

    class Provider:
        calls = 0

        def complete_json(self, **_kwargs):
            self.calls += 1
            raise AssertionError("exhausted replan budget must not call provider")

    replanner = Replanner()
    provider = Provider()
    grant = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["clean_task"],
        max_retries=0,
        max_replans=0,
        valid_until="9999-12-31T23:59:59Z",
    )
    factory = ScientificAgentFailureRecoveryServiceFactory(
        storage=storage,
        controller=controller,
        replanner=replanner,
        successor_applicator=None,
    )
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=_GrantSource(grant),
        service_factory=factory,
        store=factory.store,
    )
    service = ScientificAgentConversationSessionService(
        projects=storage,
        conversations=conversations,
        plan_service=None,
        proposal_store=None,
        authorization_service=None,
        controller=controller,
        execution_agent=None,
        failure_recovery_runtime=runtime,
        failure_recovery_enabled=True,
    )
    _result, state, stop_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=service._default_state("project-1", "conversation-1"),
        controller_result=baseline,
        provider=provider,
        provider_binding_digest="",
    )
    assert stop_reason == "RECOVERY_BUDGET_EXHAUSTED"
    assert state["status"] == "recovery_required"
    assert state["reason_code"] == "RECOVERY_BUDGET_EXHAUSTED"
    assert state["last_recovery_action"] == "ASK_USER"
    assert state["last_recovery_replan_ordinal"] == 0
    assert state["recovery_provider_calls"] == 0
    assert state["recovery_effect_count"] == 0
    assert provider.calls == 0
    assert replanner.calls == 0


def test_conversation_nonrecoverable_failure_stops_without_effect(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    conversations = ConversationStore(projects=storage)
    conversations.create_conversation(
        "project-1",
        conversation_id="conversation-1",
        title="Recovery",
    )
    baseline = _controller_result()
    controller = _Controller(baseline)
    evidence = AgentTaskFailureEvidence(
        failure_code="nonrecoverable_failure",
        failure_class=AgentFailureClass.NONRECOVERABLE,
        effect_certainty=AgentEffectCertainty.NO_EFFECT_CONFIRMED,
        task_id="clean_task",
        logical_tool_id="clean_task",
        reason_codes=["NONRECOVERABLE_FAILURE"],
    )
    _write_typed_failure(storage, evidence)
    factory = ScientificAgentFailureRecoveryServiceFactory(
        storage=storage,
        controller=controller,
        replanner=None,
        successor_applicator=None,
    )
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=_GrantSource(_grant()),
        service_factory=factory,
        store=factory.store,
    )
    service = ScientificAgentConversationSessionService(
        projects=storage,
        conversations=conversations,
        plan_service=None,
        proposal_store=None,
        authorization_service=None,
        controller=controller,
        execution_agent=None,
        failure_recovery_runtime=runtime,
        failure_recovery_enabled=True,
    )
    _result, state, stop_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=service._default_state("project-1", "conversation-1"),
        controller_result=baseline,
        provider=StubLLMProvider(response={"action": "RETRY_EXACT"}),
        provider_binding_digest="",
    )
    assert stop_reason == "RECOVERY_NONRECOVERABLE"
    assert state["status"] == "failed"
    assert state["reason_code"] == "RECOVERY_NONRECOVERABLE"
    assert state["recovery_provider_calls"] == 0
    assert state["recovery_effect_count"] == 0


def test_runtime_rejects_stale_inspection_before_foundation(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    baseline = _controller_result()
    controller = _Controller(baseline)
    factory = ScientificAgentFailureRecoveryServiceFactory(
        storage=storage,
        controller=controller,
        replanner=None,
        successor_applicator=_TrustedSuccessor(),
    )
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=_GrantSource(_grant()),
        service_factory=factory,
        store=factory.store,
    )
    stale = _controller_result(
        inspection_digest="sha256:" + "7" * 64,
    )
    result = runtime.continue_failed(
        project_id="project-1",
        conversation_id="conversation-1",
        run_id="run-1",
        state={},
        controller_result=stale,
        provider=None,
    )
    assert result.eligibility is FailureRecoveryRuntimeEligibility.FAIL_CLOSED
    assert result.reason_code == "RECOVERY_CONTROLLER_STATE_STALE"
    assert result.provider_calls_total == 0
    assert result.effect_count_total == 0


def test_multiprocess_duplicate_conversation_resume_replays_one_receipt(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    grant = _grant()
    grant_store = ScientificAgentAutonomyGrantStore(storage=storage)
    grant_store.publish_server_grant(
        grant=grant,
        authority_epoch="runtime-concurrent-epoch",
        actor="server",
        actor_source="server:recovery-bootstrap",
    )
    ctx = multiprocessing.get_context("spawn")
    start_event = ctx.Event()
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_run_concurrent_runtime,
            args=(
                str(storage.workspace_dir),
                grant.model_dump(mode="json"),
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
        process.join(30)
        assert process.exitcode == 0
    results = [result_queue.get(timeout=5) for _ in processes]
    assert all(item[0] == "ok" for item in results), results
    assert all(item[1] == FailureRecoveryRuntimeEligibility.ELIGIBLE.value for item in results)
    assert len({item[2] for item in results}) == 1
    assert all(bool(item[3]) for item in results)
    assert sum(bool(item[4]) for item in results) == 1
    assert sum(int(item[5]) for item in results) == 1
    receipts = FailureRecoveryStore(storage=storage).list_receipts(project_id="project-1")
    assert len(receipts) == 1
    assert receipts[0].effect_started is True


def test_conversation_failed_iteration_invokes_runtime_once_and_projects_boundary(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    conversations = ConversationStore(projects=storage)
    conversations.create_conversation(
        "project-1",
        conversation_id="conversation-1",
        title="Recovery",
    )

    class _Runtime:
        def __init__(self) -> None:
            self.calls = 0

        def continue_failed(self, **kwargs):
            self.calls += 1
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.REQUIRE_HUMAN,
                controller_result=kwargs["controller_result"],
                reason_code="RECOVERY_AUTONOMY_GRANT_REQUIRED",
                question="没有可验证的 server-issued AutonomyGrant。",
            )

    runtime = _Runtime()
    service = ScientificAgentConversationSessionService(
        projects=storage,
        conversations=conversations,
        plan_service=None,
        proposal_store=None,
        authorization_service=None,
        controller=None,
        execution_agent=None,
        failure_recovery_runtime=runtime,
        failure_recovery_enabled=True,
    )
    result, state, stop_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=service._default_state("project-1", "conversation-1"),
        controller_result=_controller_result(),
        provider=None,
        provider_binding_digest="",
    )
    assert runtime.calls == 1
    assert stop_reason == "RECOVERY_AUTONOMY_GRANT_REQUIRED"
    assert state["status"] == "recovery_required"
    assert state["reason_code"] == "RECOVERY_AUTONOMY_GRANT_REQUIRED"
    assert result is not None


def test_conversation_projects_the_complete_successor_authority_binding(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    conversations = ConversationStore(projects=storage)
    conversations.create_conversation(
        "project-1",
        conversation_id="conversation-1",
        title="Recovery",
    )
    baseline = _controller_result()
    controller = _Controller(baseline)
    successor = _TrustedSuccessor()
    factory = ScientificAgentFailureRecoveryServiceFactory(
        storage=storage,
        controller=controller,
        replanner=None,
        successor_applicator=successor,
    )
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=_GrantSource(_grant()),
        service_factory=factory,
        store=factory.store,
    )
    service = ScientificAgentConversationSessionService(
        projects=storage,
        conversations=conversations,
        plan_service=None,
        proposal_store=None,
        authorization_service=None,
        controller=controller,
        execution_agent=None,
        failure_recovery_runtime=runtime,
        failure_recovery_enabled=True,
    )
    _result, state, stop_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=service._default_state("project-1", "conversation-1"),
        controller_result=baseline,
        provider=None,
        provider_binding_digest="",
    )
    assert stop_reason == "RECOVERY_SUCCESSOR_COMMITTED"
    assert state["proposal_id"] == "successor-proposal"
    assert state["proposal_digest"] == "sha256:" + "a" * 64
    assert state["authorization_id"] == "successor-authorization"
    assert state["authorization_digest"] == "sha256:" + "b" * 64
    assert state["start_intent_id"] == "successor-start-intent"
    assert state["start_intent_digest"] == "sha256:" + "c" * 64
    assert state["controller_execution_id"] == "controller-successor"
    assert state["controller_execution_digest"] == "sha256:" + "e" * 64
    assert state["last_recovery_retry_ordinal"] == 1
    assert state["recovery_effect_count"] == 1


def test_production_concrete_successor_closes_failed_conversation_chain(
    tmp_path: Path,
) -> None:
    """Exercise the runtime with the real proposal/auth/controller chain.

    The failed Controller snapshot is a server-owned fixture (including typed
    no-effect evidence); the recovery successor itself is the concrete
    production applicator, not a callback-shaped test double.
    """

    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-01-01T00:00:00Z")
    run_dir = storage.run_dir("project-1", "run-1")
    input_path = run_dir / "inputs" / "dataset.csv"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("SMILES,value\nCCO,1.0\n", encoding="utf-8")
    storage.register_artifact_path(
        "project-1",
        "run-1",
        "uploaded_dataset",
        "inputs/dataset.csv",
    )

    registry = AtomicTaskRegistry()
    builder = AgentProjectObservationBuilder(
        storage=storage,
        registry=registry,
        clock=lambda: "2026-01-01T00:00:00Z",
    )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=builder,
        registry=registry,
    )
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["inspect_dataset"],
        selected_input_artifact_ids=["uploaded_dataset"],
        task_options={"inspect_dataset": {}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce a reviewable profile"],
        rationales=["Use the registered local inspection task."],
        assumptions=[],
        questions=[],
    )
    plan_service = ScientificAgentPlanService(
        storage=storage,
        registry=registry,
        observation_builder=builder,
        proposal_store=proposal_store,
        clock=lambda: "2026-01-01T00:00:00Z",
    )
    proposal = plan_service.create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Inspect one exact dataset",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="production-recovery-proposal",
    )
    control_store = AgentPlanControlStore(storage=storage)
    grant_store = ScientificAgentAutonomyGrantStore(storage=storage)
    grant_issuer = ScientificAgentAutonomyGrantIssuer(
        grant_store=grant_store,
        registry=registry,
        grant_ttl_seconds=10_000_000_000,
        clock=lambda: "2026-01-01T00:00:00Z",
    )
    authorization_service = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        control_store=control_store,
        autonomy_grant_issuer=grant_issuer.issue_from_approved_chain,
        clock=lambda: "2026-01-01T00:00:00Z",
    )
    approved = authorization_service.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="production-recovery-authorization",
            note="Approve the exact local inspection plan.",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorization_service,
        control_store=control_store,
        resource_authority_service=_NoRemoteAuthority(),
        executor=RunPlanExecutor(storage=storage, registry=registry),
        remote_executions=_NoRemoteLifecycle(),
        clock=lambda: "2026-01-01T00:00:00Z",
    )
    publication = proposal_store.read(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        verify_current=False,
    )
    authorization = authorization_service.verify_authorization(
        project_id="project-1",
        authorization_id=approved.start_intent.authorization_id,
        verify_current=False,
    )
    permission = control_store.read_permission_decision(
        project_id="project-1",
        decision_id=approved.start_intent.permission_decision_id,
    )
    execution = controller._build_execution(
        intent=approved.start_intent,
        authorization=authorization,
        publication=publication,
        permission=permission,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        client_request_id="production-recovery-controller",
        request_digest=_agent_digest({"fixture": "production-recovery-controller"}),
        created_at="2026-01-01T00:00:00Z",
    )
    control_store.publish_harness_controller_execution(execution)
    evidence = AgentTaskFailureEvidence(
        failure_code="controller_pre_effect_failure",
        failure_class=AgentFailureClass.TRANSIENT,
        effect_certainty=AgentEffectCertainty.NO_EFFECT_CONFIRMED,
        task_id="inspect_dataset",
        logical_tool_id="inspect_dataset",
        reason_codes=["CONTROLLER_PRE_EFFECT_FAILURE"],
    )
    storage.write_stage_state(
        "project-1",
        "run-1",
        StageState(
            stage="inspect_dataset",
            status=RunStatus.FAILED,
            started_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            error={"code": "typed_pre_effect_failure"},
            details={"typed_failure_evidence": evidence.model_dump(mode="json")},
        ),
    )
    # Anchor the hand-built server failure with the same immutable Controller
    # decision/receipt pair that a real failed advance would leave behind.
    failed_inspection = controller._inspect(execution, verify_authority=False)
    source_bindings = controller._bindings_from_facts(failed_inspection.facts)
    source_bindings_digest = _agent_digest(
        [item.model_dump(mode="json") for item in source_bindings]
    )
    failed_decision = AgentHarnessControllerDecision(
        controller_execution_id=execution.controller_execution_id,
        controller_execution_digest=execution.execution_digest,
        client_request_id="production-recovery-failed-decision",
        inspection_digest=failed_inspection.inspection_digest,
        action_kind=AgentHarnessControllerAction.STOP_TASK_TERMINAL,
        task_id="inspect_dataset",
        task_index=0,
        attempt_ordinal=0,
        slot_id=execution.task_slots[0].slot_id,
        source_bindings=source_bindings,
        source_bindings_digest=source_bindings_digest,
        reason_codes=["TERMINAL_OBSERVED"],
        created_at="2026-01-01T00:00:00Z",
        executable=False,
    )
    control_store.publish_harness_controller_decision(
        project_id="project-1",
        decision=failed_decision,
    )
    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    stage_digest = controller._stage_digest(stage)
    registry_digest = _agent_digest(storage.read_artifact_registry("project-1", "run-1"))
    failed_receipt = AgentHarnessControllerActionReceipt(
        controller_execution_id=execution.controller_execution_id,
        controller_execution_digest=execution.execution_digest,
        decision_id=failed_decision.decision_id,
        decision_digest=failed_decision.decision_digest,
        action_kind=AgentHarnessControllerAction.STOP_TASK_TERMINAL,
        task_id="inspect_dataset",
        task_index=0,
        attempt_ordinal=0,
        slot_id=execution.task_slots[0].slot_id,
        execution_started=False,
        dispatch_occurred=False,
        before_stage_digest=stage_digest,
        after_stage_digest=stage_digest,
        before_artifact_registry_digest=registry_digest,
        after_artifact_registry_digest=registry_digest,
        outcome=AgentHarnessControllerReceiptOutcome.FAILED,
        status_after=AgentHarnessControllerStatus.FAILED,
        source_bindings=source_bindings,
        source_bindings_digest=source_bindings_digest,
        reason_codes=["TERMINAL_OBSERVED"],
        created_at="2026-01-01T00:00:00Z",
    )
    control_store.publish_harness_controller_action_receipt(
        project_id="project-1",
        receipt=failed_receipt,
    )
    baseline = controller.read_execution_agent_snapshot(
        project_id="project-1",
        controller_execution_id=execution.controller_execution_id,
        expected_controller_execution_digest=execution.execution_digest,
    )

    successor = ScientificAgentRecoverySuccessorApplicator(
        proposal_store=proposal_store,
        authorization_service=authorization_service,
        controller=controller,
        registry=registry,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        clock=lambda: "2026-01-01T00:00:00Z",
    )
    factory = ScientificAgentFailureRecoveryServiceFactory(
        storage=storage,
        controller=controller,
        replanner=None,
        successor_applicator=successor,
        proposal_store=proposal_store,
        authorization_service=authorization_service,
        registry=registry,
        clock=lambda: "2026-01-01T00:00:00Z",
    )
    issued_binding = grant_store.resolve_current(
        project_id="project-1",
        run_id="run-1",
    )
    assert issued_binding is not None
    assert issued_binding.grant.max_retries == 1
    runtime = ScientificAgentFailureRecoveryRuntime(
        storage=storage,
        controller=controller,
        grant_source=grant_store,
        service_factory=factory,
        proposal_store=proposal_store,
        authorization_service=authorization_service,
        registry=registry,
        store=factory.store,
    )
    conversations = ConversationStore(projects=storage)
    conversations.create_conversation(
        "project-1",
        conversation_id="conversation-1",
        title="Recovery",
    )
    service = ScientificAgentConversationSessionService(
        projects=storage,
        conversations=conversations,
        plan_service=plan_service,
        proposal_store=proposal_store,
        authorization_service=authorization_service,
        controller=controller,
        execution_agent=None,
        failure_recovery_runtime=runtime,
        failure_recovery_enabled=True,
    )
    _result, state, stop_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-1",
        state=service._default_state("project-1", "conversation-1"),
        controller_result=baseline,
        provider=None,
        provider_binding_digest="",
    )
    assert stop_reason == "RECOVERY_SUCCESSOR_COMMITTED"
    assert state["last_recovery_action"] == "RETRY_EXACT"
    assert state["last_recovery_retry_ordinal"] == 1
    assert state["recovery_effect_count"] == 1
    assert state["proposal_id"]
    assert state["authorization_id"]
    assert state["start_intent_id"]
    assert state["controller_execution_id"] != execution.controller_execution_id
    assert state["controller_status"] == AgentHarnessControllerStatus.SUCCEEDED.value
    predecessor_after = controller.get(
        project_id="project-1",
        controller_execution_id=execution.controller_execution_id,
    )
    assert predecessor_after.inspection.status is AgentHarnessControllerStatus.FAILED
