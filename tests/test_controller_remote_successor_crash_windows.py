"""Controller remote-successor crash-window coverage.

PR #38 widened ``_verify_post_start_sources`` to accept remote lifecycle
successor actions (PREPARE_REMOTE_REQUEST, DISPATCH_REMOTE_TASK,
REFRESH_REMOTE_TASK, ADOPT_REMOTE_OUTPUTS) plus the adopted-remote successor
path.  This module pins the fail-open *and* fail-closed edges of that window:

1. remote completion -> next local task recovers from the committed-decision
   crash window;
2. remote completion -> next remote task recovers (prepare and dispatch);
3. an ADOPT_REMOTE_OUTPUTS effect whose receipt was not yet persisted recovers
   from the side-effect checkpoint without a second adoption;
4. a stale REFRESH_REMOTE_TASK for an already-completed task is never accepted
   as a successor;
5. a non-contiguous successor ``task_index`` fails closed;
6. a StageState ``next_stage`` that disagrees with the successor decision
   fails closed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.schemas import (
    AgentAuthorizationMode,
    AgentExecutionPlanLLMResponse,
    AgentHarnessControllerAction,
    AgentHarnessControllerAdvanceRequest,
    AgentHarnessControllerDecision,
    AgentHarnessControllerSourceBinding,
    AgentHarnessControllerStartRequest,
    AgentHarnessRemoteApprovalRequest,
    AgentPlanAuthorizationRequest,
    AgentRemoteResourceAuthorityRequest,
    AtomicTaskSpec,
    RiskLevel,
    _agent_digest,
)
from ai4s_agent.scientific_agent_authorization import (
    AgentPlanControlStore,
    ScientificAgentAuthorizationService,
)
from ai4s_agent.scientific_agent_harness_controller import (
    ScientificAgentHarnessController,
    ScientificAgentHarnessControllerVerificationError,
)
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
)
from ai4s_agent.storage import ProjectStorage

from test_remote_execution_lifecycle import FakeProbe, FakeTransport


_NOW = "2026-08-01T00:00:00Z"


class _SuccessorTransport(FakeTransport):
    """FakeTransport accepting the molecular-generation input tree."""

    def dispatch(self, *, connection, request, approval, tree):
        del connection
        assert tree.scan_files("inputs") == {"execution-request.json"}
        self.dispatches += 1
        self.approval_sha256 = approval.approval_sha256
        if self.fail_dispatch:
            from ai4s_agent.remote_execution_lifecycle import (
                RemoteTransportError,
            )

            raise RemoteTransportError("unknown")
        return self._observation(request, "ACCEPTED")


def _visible_task_spec(
    task_id: str,
    *,
    depends_on: list[str] | None = None,
) -> AtomicTaskSpec:
    return AtomicTaskSpec(
        task_id=task_id,
        required_artifacts=[],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=[],
        risk_level=RiskLevel.LOW,
        gates=[],
        default_adapter="inspect_dataset_service",
        depends_on=depends_on or [],
        scientific_tool_id=task_id,
        label=task_id.replace("_", " ").title(),
        description="A deterministic review-only test task.",
        effect_class="derive_local",
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


def _second_remote_task() -> AtomicTaskSpec:
    import test_remote_resource_authority as authority_fixtures

    return authority_fixtures._remote_task(
        "molecular_generation",
        "reinvent4-cpu-v1",
        task_id="second_remote",
        depends_on=["first_remote"],
        required_artifacts=["reinvent4_candidates"],
        output_artifacts=["second_candidates"],
    )


class _Workspace:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _remote_first_workspace(
    tmp_path: Path,
    *,
    second_remote: bool,
    workspace_dir: str = "workspace",
) -> _Workspace:
    """Build a two-task plan whose first task is remote."""

    import test_remote_resource_authority as authority_fixtures

    from ai4s_agent.remote_execution_lifecycle import (
        RemoteExecutionLifecycleService,
    )
    from ai4s_agent.remote_resource_authority import (
        RemoteResourceAuthorityPolicyStore,
        RemoteResourceAuthorityService,
    )
    from ai4s_agent.resource_profiles import (
        CapabilityDetails,
        CapabilityProbeResult,
        ConnectionProfile,
        CudaCapabilityDetails,
        ResourceProfileStore,
    )
    from ai4s_agent.schemas import (
        AgentConfiguredRemoteResources,
        AgentRemoteResourceBudgetLimits,
        RemoteResourceAuthorityPolicy,
        RemoteResourceAuthorityPolicyEntry,
    )

    storage = ProjectStorage(workspace_dir=tmp_path / workspace_dir)
    storage.create_project("project-1", name="Project", created_at=_NOW)
    profiles = ResourceProfileStore(
        workspace_dir=storage.workspace_dir,
        config_dir=tmp_path / f"{workspace_dir}-config",
    )
    connection = profiles.save_connection(
        ConnectionProfile(
            connection_id="successor-worker",
            ssh_host_alias="successor-worker",
            expected_hostname="successor-worker",
            remote_root="/srv/molly",
            declared_capabilities=["cpu", "reinvent4"],
        )
    )
    profiles.save_probe(
        CapabilityProbeResult(
            connection_id=connection.connection_id,
            connection_profile_digest=connection.digest(),
            status="available",
            checked_at=_NOW,
            verified_capabilities=["cpu", "reinvent4"],
            details=CapabilityDetails(
                cpu_threads=32,
                cuda=CudaCapabilityDetails(status="unknown"),
            ),
        )
    )
    first = authority_fixtures._remote_task(
        "molecular_generation",
        "reinvent4-cpu-v1",
        task_id="first_remote",
        output_artifacts=["reinvent4_candidates"],
    )
    second = _second_remote_task() if second_remote else _visible_task_spec(
        "local_finish",
        depends_on=["first_remote"],
    )
    registry = AtomicTaskRegistry([first, second])
    builder = AgentProjectObservationBuilder(
        storage=storage,
        registry=registry,
        resource_profiles=profiles,
        clock=lambda: _NOW,
    )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        registry=registry,
        observation_builder=builder,
    )
    selected_tool_ids = (
        ["first_remote", "second_remote"]
        if second_remote
        else ["first_remote", "local_finish"]
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        registry=registry,
        observation_builder=builder,
        proposal_store=proposal_store,
        clock=lambda: _NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Run one exact remote task then its successor",
        user_constraints=[],
        provider=StubLLMProvider(
            response=AgentExecutionPlanLLMResponse(
                requested_tool_ids=selected_tool_ids,
                selected_input_artifact_ids=[],
                task_options={task_id: {} for task_id in selected_tool_ids},
                selected_logical_profile_ids=["reinvent4-cpu-v1"],
                limits={
                    "max_runtime_sec": 2400 if second_remote else 600,
                    "max_gpu_hours": 1.0,
                },
                stop_conditions=["stop on verification failure"],
                success_criteria=["verify the successor transition"],
                rationales=["Exercise the remote successor crash window."],
                assumptions=[],
                questions=[],
            ).model_dump(mode="json")
        ),
        client_request_id=f"{workspace_dir}-proposal-1",
    )
    policy_store = RemoteResourceAuthorityPolicyStore(
        config_dir=tmp_path / f"{workspace_dir}-policy"
    )
    policy_entries = [
        RemoteResourceAuthorityPolicyEntry(
            policy_id="successor-policy",
            enabled=True,
            connection_id=connection.connection_id,
            execution_profile_id="reinvent4-cpu-v1",
            remote_task_type="molecular_generation",
            allowed_task_ids=(
                ["first_remote"] if second_remote else ["first_remote"]
            ),
            configured_resources=AgentConfiguredRemoteResources(
                gpu_count=0,
                cpu_threads=1,
                walltime_sec=600,
            ),
            budget_limits=AgentRemoteResourceBudgetLimits(
                max_runtime_sec=600,
                max_gpu_hours=1.0,
            ),
        )
    ]
    if second_remote:
        policy_entries.append(
            RemoteResourceAuthorityPolicyEntry(
                policy_id="successor-policy-second",
                enabled=True,
                connection_id=connection.connection_id,
                execution_profile_id="reinvent4-cpu-v1",
                remote_task_type="molecular_generation",
                allowed_task_ids=["second_remote"],
                configured_resources=AgentConfiguredRemoteResources(
                    gpu_count=0,
                    cpu_threads=1,
                    walltime_sec=600,
                ),
                budget_limits=AgentRemoteResourceBudgetLimits(
                    max_runtime_sec=600,
                    max_gpu_hours=1.0,
                ),
            )
        )
    policy_store.save(
        RemoteResourceAuthorityPolicy(
            entries=policy_entries,
        )
    )
    control_store = AgentPlanControlStore(storage=storage)
    resources = RemoteResourceAuthorityService(
        proposal_store=proposal_store,
        resource_profiles=profiles,
        policy_store=policy_store,
        control_store=control_store,
        clock=lambda: _NOW,
    )
    resources.publish(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id=f"{workspace_dir}-resource-1",
        ),
    )
    authorizations = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        registry=registry,
        control_store=control_store,
        resource_authority_resolver=lambda publication, task_id: (
            resources.current_authority(publication=publication, task_id=task_id)
        ),
        clock=lambda: _NOW,
    )
    approved = authorizations.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id=f"{workspace_dir}-authorization-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    transport = _SuccessorTransport()
    remote = RemoteExecutionLifecycleService(
        projects=storage,
        profiles=profiles,
        transport=transport,
        capability_probe=FakeProbe(profiles),
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorizations,
        control_store=control_store,
        resource_authority_service=resources,
        executor=RunPlanExecutor(storage=storage, registry=registry),
        remote_executions=remote,
        clock=lambda: _NOW,
    )
    created = controller.create(
        project_id="project-1",
        start_intent_id=approved.start_intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=approved.start_intent.start_intent_digest,
            client_request_id=f"{workspace_dir}-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    execution = created.execution
    return _Workspace(
        storage=storage,
        profiles=profiles,
        transport=transport,
        remote=remote,
        controller=controller,
        control_store=control_store,
        proposal_store=proposal_store,
        execution=execution,
        authorization=authorizations.verify_authorization(
            project_id="project-1",
            authorization_id=approved.start_intent.authorization_id,
            verify_current=False,
        ),
        publication=proposal_store.read(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            verify_current=False,
        ),
        second_task_id=second.task_id,
        second_remote=second_remote,
    )


def _approve_and_dispatch_first_remote(ctx: _Workspace, prefix: str) -> None:
    slot_id = ctx.execution.task_slots[0].slot_id
    remote_request = ctx.remote.inspect(
        project_id="project-1",
        run_id="run-1",
        slot_id=slot_id,
        expected_slot_binding_digest=ctx.remote.inspect_slot_binding(
            project_id="project-1",
            run_id="run-1",
            slot_id=slot_id,
        ).slot_binding_digest,
    )["request"]
    ctx.controller.approve_remote(
        project_id="project-1",
        controller_execution_id=ctx.execution.controller_execution_id,
        request=AgentHarnessRemoteApprovalRequest(
            expected_remote_request_sha256=remote_request["request_sha256"],
            client_request_id=f"{prefix}-approval-1",
            note="Approve the exact remote slot request.",
        ),
        actor="alice",
    )
    ctx.controller.advance(
        project_id="project-1",
        controller_execution_id=ctx.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=(
                ctx.execution.execution_digest
            ),
            client_request_id=f"{prefix}-dispatch-1",
        ),
    )


def _drive_to_adoption(ctx: _Workspace, prefix: str = "successor") -> None:
    _approve_and_dispatch_first_remote(ctx, prefix)
    ctx.transport.status = "SUCCEEDED"
    ctx.controller.advance(
        project_id="project-1",
        controller_execution_id=ctx.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=(
                ctx.execution.execution_digest
            ),
            client_request_id=f"{prefix}-refresh-1",
        ),
    )
    ctx.controller.advance(
        project_id="project-1",
        controller_execution_id=ctx.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=(
                ctx.execution.execution_digest
            ),
            client_request_id=f"{prefix}-adopt-1",
        ),
    )


def _crash_receipt_publish(
    ctx: _Workspace,
    monkeypatch: pytest.MonkeyPatch,
    *,
    match: str,
) -> None:
    original = ctx.control_store.publish_harness_controller_action_receipt
    failed_once = False

    def fail_first_receipt(*, project_id: str, receipt: Any):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError(match)
        return original(project_id=project_id, receipt=receipt)

    monkeypatch.setattr(
        ctx.control_store,
        "publish_harness_controller_action_receipt",
        fail_first_receipt,
    )


def _advance(ctx: _Workspace, client_request_id: str):
    return ctx.controller.advance(
        project_id="project-1",
        controller_execution_id=ctx.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=(
                ctx.execution.execution_digest
            ),
            client_request_id=client_request_id,
        ),
    )


def test_remote_completion_accepts_next_local_task_as_exact_successor(
    tmp_path: Path,
) -> None:
    ctx = _remote_first_workspace(tmp_path, second_remote=False)
    _drive_to_adoption(ctx)
    _publish_unreceipted_decision(
        ctx,
        action_kind=AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
        task_id=ctx.second_task_id,
        task_index=1,
    )
    # The committed successor decision must survive the post-start source
    # check (the exact check that runs inside the crash window before the
    # successor receipt is persisted).
    _verify_post_start_sources(ctx)


def test_remote_completion_accepts_next_remote_task_as_exact_successor(
    tmp_path: Path,
) -> None:
    for index, action in enumerate(
        (
            AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST,
            AgentHarnessControllerAction.DISPATCH_REMOTE_TASK,
        )
    ):
        ctx = _remote_first_workspace(
            tmp_path,
            second_remote=True,
            workspace_dir=f"remote-successor-{index}",
        )
        _drive_to_adoption(ctx, prefix=f"rs-{index}")
        _publish_unreceipted_decision(
            ctx,
            action_kind=action,
            task_id=ctx.second_task_id,
            task_index=1,
        )
        _verify_post_start_sources(ctx)


def test_adopt_effect_without_persisted_receipt_recovers_without_second_adoption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _remote_first_workspace(tmp_path, second_remote=False)
    _approve_and_dispatch_first_remote(ctx, "adopt-crash")
    ctx.transport.status = "SUCCEEDED"
    _advance(ctx, "adopt-crash-refresh-1")
    _crash_receipt_publish(
        ctx,
        monkeypatch,
        match="injected adopt receipt publication crash",
    )
    with pytest.raises(RuntimeError, match="injected adopt receipt publication crash"):
        _advance(ctx, "adopt-crash-adopt-1")
    adopted = _advance(ctx, "adopt-crash-adopt-1")
    assert adopted.receipt is not None
    assert adopted.receipt.action_kind == AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS
    assert adopted.receipt.outcome.value == "committed"
    receipts = ctx.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=ctx.execution.controller_execution_id,
    )
    adopt_receipts = [
        item
        for item in receipts
        if item.action_kind == AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS
    ]
    assert len(adopt_receipts) == 1
    registry = ctx.storage.read_artifact_registry("project-1", "run-1")
    assert "reinvent4_candidates" in registry


def _local_two_task_workspace(tmp_path: Path) -> _Workspace:
    storage = ProjectStorage(workspace_dir=tmp_path / "local-workspace")
    storage.create_project("project-1", name="Project", created_at=_NOW)
    first = _visible_task_spec("first_local")
    second = _visible_task_spec("second_local", depends_on=["first_local"])
    registry = AtomicTaskRegistry([first, second])
    builder = AgentProjectObservationBuilder(
        storage=storage,
        registry=registry,
        clock=lambda: _NOW,
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
        clock=lambda: _NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Run two exact local tasks",
        user_constraints=[],
        provider=StubLLMProvider(
            response=AgentExecutionPlanLLMResponse(
                requested_tool_ids=["first_local", "second_local"],
                selected_input_artifact_ids=[],
                task_options={"first_local": {}, "second_local": {}},
                selected_logical_profile_ids=[],
                limits={},
                stop_conditions=["stop on verification failure"],
                success_criteria=["verify two local stages"],
                rationales=["Exercise the local successor crash window."],
                assumptions=[],
                questions=[],
            ).model_dump(mode="json")
        ),
        client_request_id="local-two-task-proposal-1",
    )
    control_store = AgentPlanControlStore(storage=storage)
    authorizations = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        registry=registry,
        control_store=control_store,
        clock=lambda: _NOW,
    )
    approved = authorizations.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="local-two-task-authorization-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorizations,
        control_store=control_store,
        resource_authority_service=None,
        executor=RunPlanExecutor(storage=storage, registry=registry),
        remote_executions=None,
        clock=lambda: _NOW,
    )
    created = controller.create(
        project_id="project-1",
        start_intent_id=approved.start_intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=approved.start_intent.start_intent_digest,
            client_request_id="local-two-task-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    execution = created.execution
    return _Workspace(
        storage=storage,
        controller=controller,
        control_store=control_store,
        proposal_store=proposal_store,
        execution=execution,
        authorization=authorizations.verify_authorization(
            project_id="project-1",
            authorization_id=approved.start_intent.authorization_id,
            verify_current=False,
        ),
        publication=proposal_store.read(
            project_id="project-1",
            proposal_id=proposal.proposal_id,
            verify_current=False,
        ),
        second_task_id="second_local",
        second_remote=False,
    )


def _publish_unreceipted_decision(
    ctx: _Workspace,
    *,
    action_kind: AgentHarnessControllerAction,
    task_id: str,
    task_index: int,
    slot_id: str = "",
) -> AgentHarnessControllerDecision:
    execution = ctx.execution
    sources = [
        AgentHarnessControllerSourceBinding.model_validate(item)
        for item in execution.source_bindings
    ]
    decision = AgentHarnessControllerDecision(
        controller_execution_id=execution.controller_execution_id,
        controller_execution_digest=execution.execution_digest,
        client_request_id="synthetic-successor-decision-1",
        inspection_digest="sha256:" + "ab" * 32,
        action_kind=action_kind,
        task_id=task_id,
        task_index=task_index,
        attempt_ordinal=0,
        slot_id=slot_id or execution.task_slots[task_index].slot_id,
        source_bindings=sources,
        source_bindings_digest=_agent_digest(
            [item.model_dump(mode="json") for item in sources]
        ),
        predecessor_receipt_id="",
        reason_codes=["SUCCESSOR_PREPARED"],
        created_at=_NOW,
        executable=True,
    )
    ctx.control_store.publish_harness_controller_decision(
        project_id="project-1",
        decision=decision,
    )
    return decision


def _verify_post_start_sources(ctx: _Workspace) -> None:
    ctx.controller._verify_post_start_sources(
        ctx.execution,
        ctx.authorization,
        ctx.publication,
    )


def test_stale_refresh_remote_task_is_not_accepted_as_successor(
    tmp_path: Path,
) -> None:
    ctx = _local_two_task_workspace(tmp_path)
    _publish_unreceipted_decision(
        ctx,
        action_kind=AgentHarnessControllerAction.REFRESH_REMOTE_TASK,
        task_id="first_local",
        task_index=0,
    )
    # Move the StageState on to the successor so the stale refresh can no
    # longer claim the current task.
    stage = ctx.storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    ctx.storage.write_stage_state(
        "project-1",
        "run-1",
        stage.model_copy(update={"stage": ctx.second_task_id}),
    )
    with pytest.raises(
        ScientificAgentHarnessControllerVerificationError,
        match="unreceipted StageState belongs to another task",
    ):
        _verify_post_start_sources(ctx)


def test_non_contiguous_successor_task_index_fails_closed(tmp_path: Path) -> None:
    ctx = _local_two_task_workspace(tmp_path)
    _publish_unreceipted_decision(
        ctx,
        action_kind=AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
        task_id=ctx.second_task_id,
        task_index=2,
        slot_id="synthetic-slot",
    )
    with pytest.raises(
        ScientificAgentHarnessControllerVerificationError,
        match="unreceipted StageState belongs to another task",
    ):
        _verify_post_start_sources(ctx)


def test_stage_next_stage_mismatch_rejects_successor(tmp_path: Path) -> None:
    ctx = _local_two_task_workspace(tmp_path)
    _publish_unreceipted_decision(
        ctx,
        action_kind=AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
        task_id=ctx.second_task_id,
        task_index=1,
    )
    stage = ctx.storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    ctx.storage.write_stage_state(
        "project-1",
        "run-1",
        stage.model_copy(update={"next_stage": "wrong_successor"}),
    )
    with pytest.raises(
        ScientificAgentHarnessControllerVerificationError,
        match="unreceipted StageState belongs to another task",
    ):
        _verify_post_start_sources(ctx)
