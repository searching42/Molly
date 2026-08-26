from __future__ import annotations

import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.execution_agent_v2 import ExecutionAgentV2DecisionInvalid
from ai4s_agent.schemas import (
    AgentEffectCertainty,
    AgentFailureClass,
    AgentRecoveryAction,
    AgentRecoveryOutcome,
    AgentTaskFailureEvidence,
    AutonomyGrant,
    AutonomyParameterBound,
    _agent_digest,
)
from ai4s_agent.scientific_agent_failure_recovery import (
    FAILURE_RECOVERY_POLICY_DIGEST,
    FAILURE_RECOVERY_POLICY_VERSION,
    FailureRecoveryDecisionInvalid,
    FailureRecoveryObservationInvalid,
    FailureRecoveryProviderOutcomeUnknown,
    FailureRecoveryStale,
    FailureRecoveryStore,
    RecoverySuccessorApplicator,
    ScientificAgentRecoverySuccessorApplicator,
    ScientificAgentFailureRecoveryService,
    classify_failure,
)
from ai4s_agent.storage import ProjectStorage


pytestmark = pytest.mark.pr_fast


class _SnapshotController:
    """Small current-state verifier used by the control-plane tests."""

    def __init__(self, observation):
        self.observation = observation

    def read_execution_agent_snapshot(self, **kwargs):
        return SimpleNamespace(
            execution=SimpleNamespace(
                project_id=self.observation.project_id,
                run_id=self.observation.run_id,
                controller_execution_id=self.observation.controller_execution_id,
                execution_digest=self.observation.controller_execution_digest,
            ),
            inspection=SimpleNamespace(
                inspection_digest=self.observation.inspection_digest,
                status="failed",
                next_action="",
            ),
        )


class _FakeSuccessor(RecoverySuccessorApplicator):
    recovery_authority_chain_verified = True

    def __init__(self, *, calls=None, effect_receipt_id="controller-receipt"):
        self.calls = calls if calls is not None else []
        self.effect_receipt_id = effect_receipt_id

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
            "effect_receipt_id": self.effect_receipt_id,
            "effect_receipt_digest": "sha256:" + "f" * 64,
        }


def test_only_reviewed_concrete_successor_applicator_opts_into_authority_chain() -> None:
    class PlainSuccessor(RecoverySuccessorApplicator):
        def apply_recovery_successor(self, **kwargs):
            return {}

    assert RecoverySuccessorApplicator.recovery_authority_chain_verified is False
    assert PlainSuccessor.recovery_authority_chain_verified is False
    assert (
        ScientificAgentRecoverySuccessorApplicator.recovery_authority_chain_verified
        is True
    )


class _ConcurrentReplanner:
    def create_current_controller_failure_revision(self, **kwargs):
        return {"successor_proposal_id": "replan-successor"}


def _run_concurrent_recovery(
    workspace: str,
    grant_payload: dict,
    failure_id: str,
    action: str,
    start_event,
    result_queue,
) -> None:
    """Run one recovery in a fresh process against the shared store."""

    storage = ProjectStorage(workspace_dir=Path(workspace))
    grant = AutonomyGrant.model_validate(grant_payload)
    provider = StubLLMProvider(response={"action": action})
    service = ScientificAgentFailureRecoveryService(
        storage=storage,
        grant=grant,
        provider=provider,
        replanner=_ConcurrentReplanner() if action == "REPLAN" else None,
        successor_applicator=_FakeSuccessor() if action == "RETRY_EXACT" else None,
    )
    observation = service.read_observation(project_id="project-1", failure_id=failure_id)
    service.controller = _SnapshotController(observation)
    start_event.wait(10)
    try:
        result = service.recover(
            observation=observation,
            grant=grant,
            client_request_id=f"concurrent-{failure_id}",
        )
    except BaseException as exc:  # communicate child failures to the parent
        result_queue.put(("error", type(exc).__name__, str(exc)))
    else:
        result_queue.put(
            (
                "ok",
                result.decision.recovery_action.value,
                result.receipt.retry_ordinal,
                result.receipt.replan_ordinal,
                result.receipt.effect_started,
                result.provider_calls,
            )
        )


def _grant(tmp_path: Path, *, retries: int = 1, replans: int = 0) -> tuple[ProjectStorage, AutonomyGrant]:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-08-01T00:00:00Z")
    grant = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["clean_task", "alternative_tool"],
        parameter_bounds={
            "clean_task.min_nonempty": AutonomyParameterBound(minimum=0, maximum=10),
        },
        max_retries=retries,
        max_replans=replans,
        valid_until="9999-12-31T23:59:59Z",
    )
    return storage, grant


def _evidence(
    failure_class: AgentFailureClass = AgentFailureClass.TRANSIENT,
    certainty: AgentEffectCertainty = AgentEffectCertainty.NO_EFFECT_CONFIRMED,
    *,
    alternatives: list[str] | None = None,
) -> AgentTaskFailureEvidence:
    return AgentTaskFailureEvidence(
        failure_code="typed_failure",
        failure_class=failure_class,
        effect_certainty=certainty,
        task_id="clean_task",
        logical_tool_id="clean_dataset",
        safe_alternative_tool_ids=alternatives or [],
        reason_codes=["TYPED_FAILURE"],
    )


def _observation(service: ScientificAgentFailureRecoveryService, grant: AutonomyGrant, evidence: AgentTaskFailureEvidence, **kwargs):
    install_controller = kwargs.pop("_install_controller", True)
    observation = service.observe_failure(
        project_id="project-1",
        run_id="run-1",
        controller_execution_id=kwargs.pop("controller_execution_id", "controller-1"),
        controller_execution_digest=kwargs.pop("controller_execution_digest", "sha256:" + "1" * 64),
        inspection_digest=kwargs.pop("inspection_digest", "sha256:" + "2" * 64),
        task_id="clean_task",
        logical_tool_id="clean_dataset",
        arguments=kwargs.pop("arguments", {"min_nonempty": 3}),
        evidence=evidence,
        grant=grant,
        **kwargs,
    )
    if install_controller:
        service.controller = _SnapshotController(observation)
    return observation


def test_exact_transient_retry_is_a_new_bounded_successor_and_replays(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    calls: list[dict] = []
    provider = StubLLMProvider(response={"action": "RETRY_EXACT"})
    successor = _FakeSuccessor(calls=calls)
    service = ScientificAgentFailureRecoveryService(
        storage=storage,
        grant=grant,
        provider=provider,
        successor_applicator=successor,
    )
    observation = _observation(service, grant, _evidence())
    result = service.recover(observation=observation, grant=grant, client_request_id="recovery-1")
    replay = service.recover(observation=observation, grant=grant, client_request_id="recovery-1")
    assert result.decision.recovery_action is AgentRecoveryAction.RETRY_EXACT
    assert result.decision.selected_logical_tool_id == "clean_dataset"
    assert result.decision.selected_arguments == {}
    assert result.decision.retry_ordinal == 1
    assert result.receipt.outcome is AgentRecoveryOutcome.COMMITTED
    assert replay.replayed is True
    assert len(calls) == 1


def test_unknown_effect_is_deterministic_human_boundary(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path, retries=3, replans=3)
    class CountingProvider(StubLLMProvider):
        calls = 0

        def complete_json(self, **kwargs):
            self.calls += 1
            return super().complete_json(**kwargs)

    provider = CountingProvider(response={"action": "RETRY_EXACT"})
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant, provider=provider)
    observation = _observation(
        service,
        grant,
        _evidence(AgentFailureClass.UNKNOWN_EFFECT, AgentEffectCertainty.EFFECT_UNKNOWN),
    )
    result = service.recover(observation=observation, grant=grant)
    assert result.decision.recovery_action is AgentRecoveryAction.ASK_USER
    assert result.receipt.effect_started is False
    assert result.provider_calls == 0
    assert provider.calls == 0


def test_recovery_cannot_mint_a_grant_or_accept_caller_budget(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path, retries=1)
    service = ScientificAgentFailureRecoveryService(storage=storage)
    evidence = _evidence()
    with pytest.raises(FailureRecoveryObservationInvalid):
        service.observe_failure(
            project_id="project-1",
            run_id="run-1",
            controller_execution_id="controller-1",
            task_id="clean_task",
            logical_tool_id="clean_dataset",
            arguments={"min_nonempty": 3},
            evidence=evidence,
            max_retries=99,
            max_replans=99,
        )

    observation = _observation(service=ScientificAgentFailureRecoveryService(storage=storage, grant=grant), grant=grant, evidence=evidence)
    no_grant = ScientificAgentFailureRecoveryService(
        storage=storage,
        provider=StubLLMProvider(response={"action": "RETRY_EXACT"}),
        successor_applicator=_FakeSuccessor(),
    )
    with pytest.raises(FailureRecoveryObservationInvalid):
        no_grant.recover(observation=observation, grant=None)


def test_plain_successor_callback_is_rejected_at_the_authority_boundary(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    with pytest.raises(FailureRecoveryDecisionInvalid):
        ScientificAgentFailureRecoveryService(
            storage=storage,
            grant=grant,
            successor_applicator=lambda **kwargs: {},
        )


def test_automatic_recovery_without_current_controller_verifier_fails_closed(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    service = ScientificAgentFailureRecoveryService(
        storage=storage,
        grant=grant,
        provider=StubLLMProvider(response={"action": "RETRY_EXACT"}),
        successor_applicator=_FakeSuccessor(),
    )
    observation = _observation(service, grant, _evidence(), _install_controller=False)
    with pytest.raises(FailureRecoveryDecisionInvalid):
        service.recover(observation=observation, grant=grant, client_request_id="no-verifier")

    # Deterministic zero-effect boundaries remain usable without a Controller
    # reader because they do not claim an automatic successor.
    stop_observation = _observation(
        service,
        grant,
        _evidence(AgentFailureClass.NONRECOVERABLE),
        controller_execution_id="controller-stop",
        _install_controller=False,
    )
    result = service.recover(observation=stop_observation, grant=grant)
    assert result.decision.recovery_action is AgentRecoveryAction.STOP
    assert result.receipt.effect_started is False


def test_public_decision_selector_provider_path_calls_once_with_durable_session(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    provider = StubLLMProvider(response={"action": "RETRY_EXACT"})
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant, provider=provider)
    observation = _observation(service, grant, _evidence())
    with service.store.failure_session(project_id="project-1", failure_id=observation.failure_id) as failure_dir:
        decision, provider_calls = service.select_recovery_decision(
            observation=observation,
            grant=grant,
            failure_dir=failure_dir,
        )
    assert decision.recovery_action is AgentRecoveryAction.RETRY_EXACT
    assert decision.auto_apply is True
    assert provider_calls == 1


def test_known_server_failure_types_are_mapped_without_exception_text(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant)
    mapped = classify_failure(ExecutionAgentV2DecisionInvalid("timeout-looking text"))
    assert mapped.failure_class is AgentFailureClass.NONRECOVERABLE
    assert mapped.effect_certainty is AgentEffectCertainty.NO_EFFECT_CONFIRMED

    orthogonal = _evidence(AgentFailureClass.TRANSIENT, AgentEffectCertainty.EFFECT_UNKNOWN)
    observation = _observation(service, grant, orthogonal)
    result = service.recover(observation=observation, grant=grant)
    assert result.decision.recovery_action is AgentRecoveryAction.ASK_USER
    assert result.provider_calls == 0


def test_retry_budget_is_rebuilt_from_receipts_across_controller_successors(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path, retries=1)
    service = ScientificAgentFailureRecoveryService(
        storage=storage,
        grant=grant,
        provider=StubLLMProvider(response={"action": "RETRY_EXACT"}),
        successor_applicator=_FakeSuccessor(effect_receipt_id="receipt-successor"),
    )
    first = _observation(service, grant, _evidence())
    service.recover(observation=first, grant=grant, client_request_id="first")
    second = _observation(
        service,
        grant,
        _evidence(),
        controller_execution_id="controller-successor",
        controller_execution_digest="sha256:" + "8" * 64,
    )
    assert second.retry_count_used == 1
    exhausted = service.recover(observation=second, grant=grant, client_request_id="second")
    assert exhausted.decision.recovery_action is AgentRecoveryAction.ASK_USER
    assert exhausted.receipt.effect_started is False
    assert exhausted.receipt.retry_ordinal == 0
    assert exhausted.provider_calls == 0
    assert exhausted.budget.retries_used == 1


def test_physical_or_invented_tools_cannot_enter_server_roster(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant)
    with pytest.raises(FailureRecoveryObservationInvalid):
        _observation(
            service,
            grant,
            _evidence(AgentFailureClass.ALTERNATIVE_TOOL_AVAILABLE),
            available_recovery_tools=["shell"],
        )


def test_parameter_recovery_requires_closed_schema_and_rejects_expansion(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    service = ScientificAgentFailureRecoveryService(
        storage=storage,
        grant=grant,
        provider=StubLLMProvider(response={"action": "TOOL_CALL", "logical_tool_id": "clean_dataset", "arguments": {"min_nonempty": 5}}),
        tool_schemas={"clean_dataset": {"type": "object", "properties": {"min_nonempty": {"type": "integer", "minimum": 0, "maximum": 10}}, "required": ["min_nonempty"], "additionalProperties": False}},
        successor_applicator=_FakeSuccessor(effect_receipt_id="receipt"),
    )
    observation = _observation(service, grant, _evidence(AgentFailureClass.PARAMETER_RECOVERABLE))
    result = service.recover(observation=observation, grant=grant, client_request_id="parameter")
    assert result.decision.recovery_action is AgentRecoveryAction.TOOL_CALL
    assert result.decision.authority_relation.name == "SUBSET"
    assert result.decision.semantic_boundary.name == "NONE"

    storage2, grant2 = _grant(tmp_path / "expansion")
    expansion_service = ScientificAgentFailureRecoveryService(
        storage=storage2,
        grant=grant2,
        provider=StubLLMProvider(response={"action": "TOOL_CALL", "logical_tool_id": "clean_dataset", "arguments": {"min_nonempty": 20}}),
        tool_schemas={"clean_dataset": {"type": "object", "properties": {"min_nonempty": {"type": "integer", "minimum": 0, "maximum": 20}}, "required": ["min_nonempty"], "additionalProperties": False}},
    )
    expansion = _observation(expansion_service, grant2, _evidence(AgentFailureClass.PARAMETER_RECOVERABLE))
    with pytest.raises(FailureRecoveryDecisionInvalid):
        expansion_service.recover(observation=expansion, grant=grant2, client_request_id="expansion")


def test_provider_unknown_is_not_retried(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)

    class UnknownProvider:
        calls = 0

        def complete_json(self, **kwargs):
            self.calls += 1
            raise OSError("provider boundary")

    provider = UnknownProvider()
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant, provider=provider)
    observation = _observation(service, grant, _evidence(AgentFailureClass.PARAMETER_RECOVERABLE))
    with pytest.raises(FailureRecoveryProviderOutcomeUnknown):
        service.recover(observation=observation, grant=grant, client_request_id="unknown")
    with pytest.raises(FailureRecoveryProviderOutcomeUnknown):
        service.recover(observation=observation, grant=grant, client_request_id="unknown")
    assert provider.calls == 1


def test_server_owned_alternative_tool_is_allowed_only_inside_grant(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    service = ScientificAgentFailureRecoveryService(
        storage=storage,
        grant=grant,
        provider=StubLLMProvider(response={"action": "TOOL_CALL", "logical_tool_id": "alternative_tool", "arguments": {}}),
        tool_schemas={"alternative_tool": {"type": "object", "properties": {}, "additionalProperties": False}},
        successor_applicator=_FakeSuccessor(effect_receipt_id="alt-receipt"),
    )
    observation = _observation(service, grant, _evidence(AgentFailureClass.ALTERNATIVE_TOOL_AVAILABLE, alternatives=["alternative_tool"]))
    result = service.recover(observation=observation, grant=grant, client_request_id="alternative")
    assert result.decision.selected_logical_tool_id == "alternative_tool"
    assert result.decision.authority_relation.name == "SUBSET"

    storage2, grant2 = _grant(tmp_path / "outside")
    grant2 = AutonomyGrant.model_validate({**grant2.model_dump(mode="json"), "allowed_tasks": ["clean_task"], "grant_id": "", "grant_digest": ""})
    outside = ScientificAgentFailureRecoveryService(
        storage=storage2,
        grant=grant2,
        provider=StubLLMProvider(response={"action": "TOOL_CALL", "logical_tool_id": "alternative_tool", "arguments": {}}),
        tool_schemas={"alternative_tool": {"type": "object", "properties": {}, "additionalProperties": False}},
    )
    outside_observation = _observation(outside, grant2, _evidence(AgentFailureClass.ALTERNATIVE_TOOL_AVAILABLE, alternatives=["alternative_tool"]))
    with pytest.raises(FailureRecoveryDecisionInvalid):
        outside.recover(observation=outside_observation, grant=grant2, client_request_id="outside")


def test_replan_uses_existing_entrypoint_once_and_budget_survives(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path, replans=1)
    calls: list[dict] = []

    class Replanner:
        def create_current_controller_failure_revision(self, **kwargs):
            calls.append(kwargs)
            return {"successor_proposal_id": "revision-successor"}

    service = ScientificAgentFailureRecoveryService(
        storage=storage,
        grant=grant,
        provider=StubLLMProvider(response={"action": "REPLAN"}),
        replanner=Replanner(),
    )
    first = _observation(service, grant, _evidence(AgentFailureClass.INPUT_EVIDENCE_INSUFFICIENT))
    result = service.recover(observation=first, grant=grant, client_request_id="replan-1")
    assert result.decision.recovery_action is AgentRecoveryAction.REPLAN
    assert result.receipt.effect_started is False
    assert len(calls) == 1
    second = _observation(
        service,
        grant,
        _evidence(AgentFailureClass.INPUT_EVIDENCE_INSUFFICIENT),
        controller_execution_id="successor-controller",
        controller_execution_digest="sha256:" + "9" * 64,
    )
    assert second.replan_count_used == 1
    exhausted = service.recover(observation=second, grant=grant, client_request_id="replan-2")
    assert exhausted.decision.recovery_action is AgentRecoveryAction.ASK_USER
    assert len(calls) == 1


def test_semantic_review_and_nonrecoverable_are_human_or_stop_without_provider(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path, retries=2)

    class NeverProvider:
        calls = 0

        def complete_json(self, **kwargs):
            self.calls += 1
            raise AssertionError("deterministic boundary must not call provider")

    provider = NeverProvider()
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant, provider=provider)
    semantic = _observation(service, grant, _evidence(AgentFailureClass.SEMANTIC_REVIEW_REQUIRED))
    semantic_result = service.recover(observation=semantic, grant=grant)
    assert semantic_result.decision.recovery_action is AgentRecoveryAction.ASK_USER
    nonrecoverable = _observation(service, grant, _evidence(AgentFailureClass.NONRECOVERABLE), controller_execution_id="controller-nonrecoverable")
    stop_result = service.recover(observation=nonrecoverable, grant=grant)
    assert stop_result.decision.recovery_action is AgentRecoveryAction.STOP
    assert provider.calls == 0


def test_effect_crash_reconciles_without_a_second_successor_call(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    calls: list[int] = []

    class Fault:
        fired = False

        def __call__(self, phase: str):
            if phase == "after_effect" and not self.fired:
                self.fired = True
                raise RuntimeError("simulated crash")

    fault = Fault()
    service = ScientificAgentFailureRecoveryService(
        storage=storage,
        grant=grant,
        provider=StubLLMProvider(response={"action": "RETRY_EXACT"}),
        successor_applicator=_FakeSuccessor(calls=calls, effect_receipt_id="effect"),
        effect_reconciler=lambda **kwargs: _FakeSuccessor(effect_receipt_id="effect").apply_recovery_successor(**kwargs),
        fault_injector=fault,
    )
    observation = _observation(service, grant, _evidence())
    with pytest.raises(RuntimeError):
        service.recover(observation=observation, grant=grant, client_request_id="crash")
    resumed = service.recover(observation=observation, grant=grant, client_request_id="crash")
    assert len(calls) == 1
    assert resumed.receipt.effect_started is True


@pytest.mark.parametrize(
    ("action", "failure_class", "retries", "replans"),
    [
        ("RETRY_EXACT", AgentFailureClass.TRANSIENT, 1, 0),
        ("REPLAN", AgentFailureClass.INPUT_EVIDENCE_INSUFFICIENT, 0, 1),
    ],
)
def test_aggregate_budget_reservation_is_atomic_across_failure_processes(
    tmp_path: Path,
    action: str,
    failure_class: AgentFailureClass,
    retries: int,
    replans: int,
) -> None:
    storage, grant = _grant(tmp_path, retries=retries, replans=replans)
    observer = ScientificAgentFailureRecoveryService(storage=storage, grant=grant)
    first = _observation(
        observer,
        grant,
        _evidence(failure_class),
        controller_execution_id="controller-a",
        controller_execution_digest="sha256:" + "1" * 64,
        inspection_digest="sha256:" + "2" * 64,
        session_id="aggregate-session",
        authority_epoch="aggregate-epoch",
    )
    second = _observation(
        observer,
        grant,
        _evidence(failure_class),
        controller_execution_id="controller-b",
        controller_execution_digest="sha256:" + "3" * 64,
        inspection_digest="sha256:" + "4" * 64,
        session_id="aggregate-session",
        authority_epoch="aggregate-epoch",
    )
    ctx = multiprocessing.get_context("spawn")
    start_event = ctx.Event()
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_run_concurrent_recovery,
            args=(
                str(storage.workspace_dir),
                grant.model_dump(mode="json"),
                observation.failure_id,
                action,
                start_event,
                result_queue,
            ),
        )
        for observation in (first, second)
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    results = [result_queue.get(timeout=5) for _ in processes]
    assert all(item[0] == "ok" for item in results), results
    automatic = [item for item in results if item[1] == action]
    human = [item for item in results if item[1] in {"ASK_USER", "STOP"}]
    assert len(automatic) == 1, results
    assert len(human) == 1, results
    assert automatic[0][2 if action == "RETRY_EXACT" else 3] == 1
    assert automatic[0][4] is (action == "RETRY_EXACT")
    assert human[0][2] == 0
    assert human[0][3] == 0
    assert human[0][4] is False
    assert human[0][5] == 0
    receipts = FailureRecoveryStore(storage=storage).list_receipts(project_id="project-1")
    assert len(receipts) == 2
    assert sum(item.retry_ordinal > 0 for item in receipts) == int(action == "RETRY_EXACT")
    assert sum(item.replan_ordinal > 0 for item in receipts) == int(action == "REPLAN")


def test_forged_observation_and_future_enum_fail_closed(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant)
    observation = _observation(service, grant, _evidence())
    forged = observation.model_dump(mode="json")
    forged["retry_count_used"] = 99
    forged["failure_digest"] = _agent_digest({**{key: value for key, value in forged.items() if key not in {"failure_id", "failure_digest", "created_at"}}, "arguments_digest": forged["arguments_digest"]})
    with pytest.raises((FailureRecoveryStale, ValueError)):
        service.recover(observation=forged, grant=grant)
    with pytest.raises(ValidationError):
        AgentTaskFailureEvidence.model_validate({**_evidence().model_dump(mode="json"), "failure_class": "FUTURE"})


def test_schema_contract_is_non_executable_and_provider_prompt_is_safe(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant)
    observation = _observation(service, grant, _evidence())
    assert observation.executable is False
    assert observation.policy_version == FAILURE_RECOVERY_POLICY_VERSION
    assert observation.policy_digest == FAILURE_RECOVERY_POLICY_DIGEST
