from __future__ import annotations

import multiprocessing
import os
import shutil
import subprocess
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Mapping, Sequence

import pytest
from flask import Flask

from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.harness_tracing import OpenTelemetryHarnessTracer
from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.schemas import (
    AgentAuthorizationMode,
    AgentExecutionPlanLLMResponse,
    AgentHarnessControllerAction,
    AgentHarnessControllerAdvanceRequest,
    AgentHarnessControllerReceiptOutcome,
    AgentHarnessControllerStartRequest,
    AgentHarnessGateApprovalRequest,
    AgentHarnessRemoteApprovalRequest,
    AgentPlanAuthorizationRequest,
    AgentRemoteResourceAuthorityRequest,
    AtomicTaskSpec,
    RiskLevel,
)
from ai4s_agent.scientific_agent_authorization import (
    AgentPlanControlStore,
    ScientificAgentAuthorizationService,
)
from ai4s_agent.scientific_agent_harness_controller import (
    CONTROLLER_POLICY_DIGEST,
    ScientificAgentHarnessController,
    ScientificAgentHarnessControllerConflict,
    ScientificAgentHarnessControllerVerificationError,
)
from ai4s_agent.routes.scientific_agent_harness_controller import (
    register_scientific_agent_harness_controller_routes,
)
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
)
from ai4s_agent.storage import ProjectStorage


_NOW = "2026-08-01T00:00:00Z"


class _FakeHarnessSpan:
    def __init__(self, record: dict[str, object]) -> None:
        self.record = record

    def set_attribute(self, key: str, value: str | int) -> None:
        self.record.setdefault("attributes", {})[key] = value  # type: ignore[index]

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, str | int] | None = None,
    ) -> None:
        self.record.setdefault("events", []).append(  # type: ignore[union-attr]
            (name, dict(attributes or {}))
        )

    def record_error(self, reason_code: str) -> None:
        self.add_event("controller.failure", {"reason_code": reason_code})


class _FakeHarnessSpanContext(AbstractContextManager[_FakeHarnessSpan]):
    def __init__(self, tracer: "_FakeHarnessTracer", record: dict[str, object]) -> None:
        self.tracer = tracer
        self.record = record

    def __enter__(self) -> _FakeHarnessSpan:
        self.tracer.stack.append(str(self.record["name"]))
        return _FakeHarnessSpan(self.record)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.tracer.stack.pop()
        return False


class _FakeHarnessTracer:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []
        self.stack: list[str] = []

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, str | int] | None = None,
        links: Sequence[object] = (),
    ) -> AbstractContextManager[_FakeHarnessSpan]:
        record: dict[str, object] = {
            "name": name,
            "parent": self.stack[-1] if self.stack else "",
            "attributes": dict(attributes or {}),
            "events": [],
            "link_count": len(links),
        }
        self.records.append(record)
        return _FakeHarnessSpanContext(self, record)

    def shutdown(self) -> None:
        return None


class _FailingOtelDelegate:
    def start_as_current_span(self, *args, **kwargs):
        raise RuntimeError("private exporter context failure")


class _FailingOtelProvider:
    def shutdown(self) -> None:
        raise RuntimeError("private exporter shutdown failure")


class _NoRemoteAuthorities:
    def current_authority(self, **_: object):  # pragma: no cover - a local plan must not call it
        raise AssertionError("local Controller plan consulted remote authority")


class _NoRemoteLifecycle:
    pass


def _local_authority_chain(tmp_path: Path):
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at=_NOW)
    run_dir = storage.run_dir("project-1", "run-1")
    dataset = run_dir / "inputs" / "dataset.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("SMILES,value\nCCO,1.0\n", encoding="utf-8")
    storage.register_artifact_path(
        "project-1", "run-1", "uploaded_dataset", "inputs/dataset.csv"
    )
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["inspect_dataset"],
        selected_input_artifact_ids=["uploaded_dataset"],
        task_options={"inspect_dataset": {}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce a dataset profile"],
        rationales=["Inspect the exact registered dataset."],
        assumptions=[],
        questions=[],
    )
    builder = AgentProjectObservationBuilder(storage=storage, clock=lambda: _NOW)
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=builder,
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        observation_builder=builder,
        proposal_store=proposal_store,
        clock=lambda: _NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Inspect one exact dataset",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="proposal-request-1",
    )
    control_store = AgentPlanControlStore(storage=storage)
    authorizations = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
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
            client_request_id="authorization-request-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorizations,
        control_store=control_store,
        resource_authority_service=_NoRemoteAuthorities(),
        executor=RunPlanExecutor(storage=storage, registry=proposal_store.registry),
        remote_executions=_NoRemoteLifecycle(),
        clock=lambda: _NOW,
    )
    return storage, control_store, controller, approved.start_intent


def _reopen_local_controller(workspace_dir: str) -> ScientificAgentHarnessController:
    storage = ProjectStorage(workspace_dir=Path(workspace_dir))
    builder = AgentProjectObservationBuilder(storage=storage, clock=lambda: _NOW)
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=builder,
    )
    control_store = AgentPlanControlStore(storage=storage)
    authorizations = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        control_store=control_store,
        clock=lambda: _NOW,
    )
    return ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorizations,
        control_store=control_store,
        resource_authority_service=_NoRemoteAuthorities(),
        executor=RunPlanExecutor(storage=storage, registry=proposal_store.registry),
        remote_executions=_NoRemoteLifecycle(),
        clock=lambda: _NOW,
    )


def _concurrent_create_process(
    workspace_dir: str,
    start_intent_id: str,
    start_intent_digest: str,
    ready: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    try:
        controller = _reopen_local_controller(workspace_dir)
        ready.wait(timeout=10)
        result = controller.create(
            project_id="project-1",
            start_intent_id=start_intent_id,
            request=AgentHarnessControllerStartRequest(
                expected_start_intent_digest=start_intent_digest,
                client_request_id="controller-cross-process-create-1",
            ),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )
        results.put(
            (
                "ok",
                result.execution.controller_execution_id,
                result.receipt.receipt_id if result.receipt else "",
            )
        )
    except Exception as exc:  # pragma: no cover - surfaced in the parent assertion
        results.put(("error", type(exc).__name__, str(exc)))


def _gated_local_authority_chain(tmp_path: Path):
    registry = _gated_task_registry()
    storage = ProjectStorage(workspace_dir=tmp_path / "gated-workspace")
    storage.create_project("project-1", name="Project", created_at=_NOW)
    run_dir = storage.run_dir("project-1", "run-1")
    dataset = run_dir / "inputs" / "dataset.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("SMILES,value\nCCO,1.0\n", encoding="utf-8")
    storage.register_artifact_path(
        "project-1", "run-1", "uploaded_dataset", "inputs/dataset.csv"
    )
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["inspect_dataset"],
        selected_input_artifact_ids=["uploaded_dataset"],
        task_options={"inspect_dataset": {}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce a dataset profile"],
        rationales=["Use the registered gated task."],
        assumptions=[],
        questions=[],
    )
    builder = AgentProjectObservationBuilder(
        storage=storage, registry=registry, clock=lambda: _NOW
    )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage, observation_builder=builder, registry=registry
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
        goal="Inspect one exact gated dataset",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="gated-proposal-request-1",
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
            client_request_id="gated-authorization-request-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorizations,
        control_store=control_store,
        resource_authority_service=_NoRemoteAuthorities(),
        executor=RunPlanExecutor(storage=storage, registry=registry),
        remote_executions=_NoRemoteLifecycle(),
        clock=lambda: _NOW,
    )
    return storage, controller, approved.start_intent


def _gated_task_registry() -> AtomicTaskRegistry:
    return AtomicTaskRegistry(
        [
            AtomicTaskSpec(
                task_id="inspect_dataset",
                required_artifacts=["uploaded_dataset"],
                optional_input_artifacts=[],
                input_artifact_alternatives=[],
                output_artifacts=["dataset_profile"],
                risk_level=RiskLevel.LOW,
                gates=["gate_1_task_parse"],
                default_adapter="inspect_dataset_service",
                scientific_tool_id="inspect_dataset",
                label="Inspect Dataset",
                description="Inspect one exact content-bound dataset.",
                effect_class="derive_local",
                required_permissions=["derive_project_artifact"],
                option_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                default_planner_options={},
                backend_default_planner_options={},
                review_required_option_ids=[],
                option_compiler_version="scientific-planner-option-identity.v1",
                logical_profile_requirements=[],
                backend_profile_requirements={},
                execution_route="local_executor",
                remote_task_type=None,
                backend_execution_routes={},
                backend_remote_task_types={},
                accepted_input_trust_classes_by_artifact={
                    "uploaded_dataset": ["content_bound_input"]
                },
                budget_dimensions=[],
                supports_plan_preapproval=False,
                idempotency_policy="server_checked",
                verification_policy="artifact_registry_and_stage_verifier",
                planner_visible=True,
            )
        ]
    )


def _reopen_gated_controller(workspace_dir: str) -> ScientificAgentHarnessController:
    storage = ProjectStorage(workspace_dir=Path(workspace_dir))
    registry = _gated_task_registry()
    builder = AgentProjectObservationBuilder(
        storage=storage,
        registry=registry,
        clock=lambda: _NOW,
    )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=builder,
        registry=registry,
    )
    control_store = AgentPlanControlStore(storage=storage)
    authorizations = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        registry=registry,
        control_store=control_store,
        clock=lambda: _NOW,
    )
    return ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorizations,
        control_store=control_store,
        resource_authority_service=_NoRemoteAuthorities(),
        executor=RunPlanExecutor(storage=storage, registry=registry),
        remote_executions=_NoRemoteLifecycle(),
        clock=lambda: _NOW,
    )


def _concurrent_advance_process(
    workspace_dir: str,
    controller_execution_id: str,
    controller_execution_digest: str,
    ready: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    try:
        controller = _reopen_gated_controller(workspace_dir)
        ready.wait(timeout=10)
        result = controller.advance(
            project_id="project-1",
            controller_execution_id=controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=controller_execution_digest,
                client_request_id="controller-cross-process-advance-1",
            ),
        )
        results.put(
            (
                "ok",
                result.decision.decision_id if result.decision else "",
                result.receipt.receipt_id if result.receipt else "",
            )
        )
    except Exception as exc:  # pragma: no cover - surfaced in the parent assertion
        results.put(("error", type(exc).__name__, str(exc)))


def test_controller_executes_exactly_one_local_task_and_replays_receipt(tmp_path: Path) -> None:
    storage, control_store, controller, intent = _local_authority_chain(tmp_path)
    request = AgentHarnessControllerStartRequest(
        expected_start_intent_digest=intent.start_intent_digest,
        client_request_id="controller-create-1",
    )
    first = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=request,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )

    assert first.decision is not None
    assert first.decision.action_kind == AgentHarnessControllerAction.EXECUTE_LOCAL_TASK
    assert first.receipt is not None
    assert first.receipt.outcome == AgentHarnessControllerReceiptOutcome.COMMITTED
    assert first.receipt.reason_codes == ["TASK_COMPLETED"]
    assert first.inspection.status.value == "succeeded"
    assert "dataset_profile" in storage.read_artifact_registry("project-1", "run-1")

    replay = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=request,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert replay.receipt is not None
    assert replay.receipt.receipt_id == first.receipt.receipt_id
    receipts = control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=first.execution.controller_execution_id,
    )
    assert [item.receipt_id for item in receipts] == [first.receipt.receipt_id]


def test_controller_policy_digest_is_stable_across_hash_seeds() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "from ai4s_agent.scientific_agent_harness_controller "
            "import CONTROLLER_POLICY_DIGEST; print(CONTROLLER_POLICY_DIGEST)"
        ),
    ]
    observed = []
    for seed in ("1", "987654"):
        environ = dict(os.environ)
        environ["PYTHONHASHSEED"] = seed
        environ["PYTHONPATH"] = "src"
        observed.append(
            subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=environ,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    assert observed == [CONTROLLER_POLICY_DIGEST, CONTROLLER_POLICY_DIGEST]


def test_start_intent_allows_only_one_controller_execution_across_request_ids(
    tmp_path: Path,
) -> None:
    _, control_store, controller, intent = _local_authority_chain(tmp_path)
    first = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="controller-first-consumer-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )

    with pytest.raises(ScientificAgentHarnessControllerConflict, match="already consumed"):
        controller.create(
            project_id="project-1",
            start_intent_id=intent.start_intent_id,
            request=AgentHarnessControllerStartRequest(
                expected_start_intent_digest=intent.start_intent_digest,
                client_request_id="controller-second-consumer-1",
            ),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )

    executions = control_store.list_harness_controller_executions(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
    )
    assert [item.controller_execution_id for item in executions] == [
        first.execution.controller_execution_id
    ]


def test_same_create_request_is_cross_process_exactly_once(tmp_path: Path) -> None:
    storage, control_store, _, intent = _local_authority_chain(tmp_path)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    results = context.Queue()
    args = (
        str(storage.workspace_dir),
        intent.start_intent_id,
        intent.start_intent_digest,
        ready,
        results,
    )
    processes = [
        context.Process(target=_concurrent_create_process, args=args)
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    ready.set()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    observed = sorted(results.get(timeout=2) for _ in processes)
    assert all(item[0] == "ok" for item in observed), observed
    assert len({item[1] for item in observed if item[1]}) == 1
    assert len({item[2] for item in observed}) == 1
    executions = control_store.list_harness_controller_executions(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
    )
    assert len(executions) == 1
    receipts = control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=executions[0].controller_execution_id,
    )
    assert len(receipts) == 1


def test_same_local_advance_is_cross_process_exactly_once(tmp_path: Path) -> None:
    storage, controller, intent = _gated_local_authority_chain(tmp_path)
    created = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="controller-cross-process-gate-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    snapshot = stage.details["execution_snapshot"]
    controller.approve_gate(
        project_id="project-1",
        controller_execution_id=created.execution.controller_execution_id,
        gate_id="gate_1_task_parse",
        request=AgentHarnessGateApprovalRequest(
            expected_snapshot_id=snapshot["snapshot_id"],
            expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
            client_request_id="controller-cross-process-gate-approval-1",
            note="Approve one exact cross-process task.",
        ),
        actor="alice",
    )

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    results = context.Queue()
    args = (
        str(storage.workspace_dir),
        created.execution.controller_execution_id,
        created.execution.execution_digest,
        ready,
        results,
    )
    processes = [
        context.Process(target=_concurrent_advance_process, args=args)
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    ready.set()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    observed = sorted(results.get(timeout=2) for _ in processes)
    assert all(item[0] == "ok" for item in observed), observed
    assert len({item[1] for item in observed if item[1]}) == 1
    assert len({item[2] for item in observed}) == 1
    completed = storage.read_stage_state("project-1", "run-1")
    assert completed is not None
    assert [item.status.value for item in completed.history].count("RUNNING") == 1
    assert [item.status.value for item in completed.history].count("SUCCEEDED") == 1


def test_tracing_on_off_preserves_authoritative_bytes_and_emits_bounded_hierarchy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("ai4s_agent.executor.now_iso", lambda: _NOW)
    storage, _, _, intent = _local_authority_chain(tmp_path / "equivalence")
    workspace_dir = storage.workspace_dir
    pristine = tmp_path / "equivalence-pristine"
    shutil.copytree(workspace_dir, pristine)

    def execute(tracer=None):
        controller = _reopen_local_controller(str(workspace_dir))
        storage = controller.storage
        control_store = controller.control_store
        if tracer is not None:
            controller.tracer = tracer
        result = controller.create(
            project_id="project-1",
            start_intent_id=intent.start_intent_id,
            request=AgentHarnessControllerStartRequest(
                expected_start_intent_digest=intent.start_intent_digest,
                client_request_id="controller-tracing-equivalence-1",
            ),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )
        registry = storage.read_artifact_registry("project-1", "run-1")
        run_dir = storage.run_dir("project-1", "run-1")
        artifact_bytes = {
            artifact_id: (run_dir / relative).read_bytes()
            for artifact_id, relative in sorted(registry.items())
        }
        return {
            "execution": result.execution.model_dump(mode="json"),
            "decisions": [
                item.model_dump(mode="json")
                for item in control_store.list_harness_controller_decisions(
                    project_id="project-1",
                    controller_execution_id=result.execution.controller_execution_id,
                )
            ],
            "receipts": [
                item.model_dump(mode="json")
                for item in control_store.list_harness_controller_action_receipts(
                    project_id="project-1",
                    controller_execution_id=result.execution.controller_execution_id,
                )
            ],
            "stage": storage.read_stage_state("project-1", "run-1").model_dump(
                mode="json"
            ),
            "registry": registry,
            "artifact_bytes": artifact_bytes,
        }

    disabled = execute()
    shutil.rmtree(workspace_dir)
    shutil.copytree(pristine, workspace_dir)
    tracer = _FakeHarnessTracer()
    enabled = execute(tracer=tracer)

    assert enabled == disabled
    assert [(item["name"], item["parent"]) for item in tracer.records] == [
        ("controller.execution", ""),
        ("controller.action", "controller.execution"),
        ("executor.local_task", "controller.action"),
    ]
    action = tracer.records[1]
    assert [name for name, _ in action["events"]] == [  # type: ignore[index]
        "controller.decision",
        "controller.receipt",
    ]


def test_tracing_delegate_failure_cannot_block_local_execution(tmp_path: Path) -> None:
    storage, _, controller, intent = _local_authority_chain(tmp_path)
    controller.tracer = OpenTelemetryHarnessTracer(
        tracer=_FailingOtelDelegate(),
        provider=_FailingOtelProvider(),
    )

    result = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="controller-failing-tracer-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )

    assert result.receipt is not None
    assert result.receipt.reason_codes == ["TASK_COMPLETED"]
    assert "dataset_profile" in storage.read_artifact_registry("project-1", "run-1")


def test_terminal_advance_records_observation_without_dispatch(tmp_path: Path) -> None:
    _, _, controller, intent = _local_authority_chain(tmp_path)
    created = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="controller-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    advanced = controller.advance(
        project_id="project-1",
        controller_execution_id=created.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=created.execution.execution_digest,
            client_request_id="controller-advance-terminal-1",
        ),
    )
    assert advanced.decision is not None
    assert advanced.decision.action_kind == AgentHarnessControllerAction.COMPLETE_EXECUTION
    assert advanced.receipt is not None
    assert advanced.receipt.execution_started is False
    assert advanced.receipt.dispatch_occurred is False
    assert advanced.inspection.status.value == "succeeded"


def test_local_crash_after_committed_outputs_reconciles_without_second_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage, control_store, controller, intent = _local_authority_chain(tmp_path)
    original_publish = control_store.publish_harness_controller_action_receipt
    failed_once = False

    def fail_first_publish(*, project_id, receipt):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError("injected receipt publication crash")
        return original_publish(project_id=project_id, receipt=receipt)

    monkeypatch.setattr(
        control_store,
        "publish_harness_controller_action_receipt",
        fail_first_publish,
    )
    request = AgentHarnessControllerStartRequest(
        expected_start_intent_digest=intent.start_intent_digest,
        client_request_id="controller-crash-recovery-1",
    )
    with pytest.raises(RuntimeError, match="injected"):
        controller.create(
            project_id="project-1",
            start_intent_id=intent.start_intent_id,
            request=request,
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )
    stage_before = storage.read_stage_state("project-1", "run-1")
    assert stage_before is not None
    history_before = list(stage_before.history)

    recovered = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=request,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    stage_after = storage.read_stage_state("project-1", "run-1")
    assert stage_after is not None
    assert stage_after.history == history_before
    assert recovered.receipt is not None
    assert recovered.receipt.reason_codes == ["TASK_COMPLETED"]
    assert recovered.inspection.status.value == "succeeded"


def test_input_drift_after_gate_snapshot_fails_closed_before_task_dispatch(
    tmp_path: Path,
) -> None:
    storage, controller, intent = _gated_local_authority_chain(tmp_path)
    created = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="drift-controller-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    (storage.run_dir("project-1", "run-1") / "inputs" / "dataset.csv").write_text(
        "SMILES,value\nCCC,9.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ScientificAgentHarnessControllerVerificationError):
        controller.advance(
            project_id="project-1",
            controller_execution_id=created.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=created.execution.execution_digest,
                client_request_id="drift-controller-advance-1",
            ),
        )
    assert "dataset_profile" not in storage.read_artifact_registry("project-1", "run-1")


def test_gate_approval_commits_decision_without_executing_then_advance_runs_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage, controller, intent = _gated_local_authority_chain(tmp_path)
    created = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="gated-controller-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert created.decision is not None
    assert created.decision.action_kind == AgentHarnessControllerAction.PREPARE_LOCAL_GATE
    assert created.inspection.status.value == "waiting_gate"
    assert "dataset_profile" not in storage.read_artifact_registry("project-1", "run-1")
    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    snapshot = stage.details["execution_snapshot"]

    approval_request = AgentHarnessGateApprovalRequest(
        expected_snapshot_id=snapshot["snapshot_id"],
        expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
        client_request_id="gate-approval-1",
        note="Approve this exact snapshot.",
    )
    original_write_marker = controller.requests.write_marker
    failed_once = False

    def fail_after_gate_authority(session, *, filename, status, values):
        nonlocal failed_once
        if (
            not failed_once
            and session.operation == "gate-approval"
            and filename == "side_effect_observed.json"
        ):
            failed_once = True
            raise RuntimeError("injected Gate checkpoint crash")
        return original_write_marker(
            session,
            filename=filename,
            status=status,
            values=values,
        )

    monkeypatch.setattr(controller.requests, "write_marker", fail_after_gate_authority)
    with pytest.raises(RuntimeError, match="Gate checkpoint"):
        controller.approve_gate(
            project_id="project-1",
            controller_execution_id=created.execution.controller_execution_id,
            gate_id="gate_1_task_parse",
            request=approval_request,
            actor="alice",
        )
    approved = controller.approve_gate(
        project_id="project-1",
        controller_execution_id=created.execution.controller_execution_id,
        gate_id="gate_1_task_parse",
        request=approval_request,
        actor="alice",
    )
    assert approved.inspection.next_action == AgentHarnessControllerAction.EXECUTE_LOCAL_TASK
    assert "dataset_profile" not in storage.read_artifact_registry("project-1", "run-1")
    gate_replay = controller.approve_gate(
        project_id="project-1",
        controller_execution_id=created.execution.controller_execution_id,
        gate_id="gate_1_task_parse",
        request=approval_request,
        actor="alice",
    )
    assert gate_replay.inspection.next_action == AgentHarnessControllerAction.EXECUTE_LOCAL_TASK
    assert len(storage.read_gate_decisions("project-1", "run-1")) == 1
    with pytest.raises(ScientificAgentHarnessControllerConflict):
        controller.approve_gate(
            project_id="project-1",
            controller_execution_id=created.execution.controller_execution_id,
            gate_id="gate_1_task_parse",
            request=AgentHarnessGateApprovalRequest(
                expected_snapshot_id=snapshot["snapshot_id"],
                expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
                client_request_id="gate-approval-1",
                note="Conflicting reuse of the same client request ID.",
            ),
            actor="alice",
        )

    completed = controller.advance(
        project_id="project-1",
        controller_execution_id=created.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=created.execution.execution_digest,
            client_request_id="gated-controller-advance-1",
        ),
    )
    assert completed.receipt is not None
    assert completed.receipt.gate_decision_digest.startswith("sha256:")
    assert completed.receipt.reason_codes == ["TASK_COMPLETED"]
    assert completed.inspection.status.value == "succeeded"


def test_remote_controller_separates_prepare_approval_dispatch_refresh_and_adoption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import test_remote_resource_authority as authority_fixtures
    from test_remote_execution_lifecycle import FakeProbe, FakeTransport

    from ai4s_agent.remote_execution_lifecycle import (
        RemoteExecutionLifecycleService,
        RemoteTransportError,
    )

    original_remote_task = authority_fixtures._remote_task

    def remote_task_with_exact_output(*args, **kwargs):
        kwargs["output_artifacts"] = ["reinvent4_candidates"]
        return original_remote_task(*args, **kwargs)

    monkeypatch.setattr(
        authority_fixtures,
        "_remote_task",
        remote_task_with_exact_output,
    )

    class ControllerTransport(FakeTransport):
        def dispatch(self, *, connection, request, approval, tree):
            del connection
            assert tree.scan_files("inputs") == {"execution-request.json"}
            self.dispatches += 1
            self.approval_sha256 = approval.approval_sha256
            if self.fail_dispatch:
                raise RemoteTransportError("unknown")
            return self._observation(request, "ACCEPTED")
    (
        storage,
        profiles,
        _,
        proposal_store,
        proposal,
        resource_authorities,
        authorizations,
    ) = authority_fixtures._configured_case(tmp_path)
    resource_authorities.publish(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=proposal.proposal_digest,
            client_request_id="remote-resource-request-1",
        ),
    )
    approved = authorizations.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="remote-authorization-request-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    transport = ControllerTransport()
    remote = RemoteExecutionLifecycleService(
        projects=storage,
        profiles=profiles,
        transport=transport,
        capability_probe=FakeProbe(profiles),
    )
    control_store = authorizations.control_store
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorizations,
        control_store=control_store,
        resource_authority_service=resource_authorities,
        executor=RunPlanExecutor(storage=storage, registry=proposal_store.registry),
        remote_executions=remote,
        clock=lambda: _NOW,
    )

    prepared = controller.create(
        project_id="project-1",
        start_intent_id=approved.start_intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=approved.start_intent.start_intent_digest,
            client_request_id="remote-controller-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert prepared.decision is not None
    assert prepared.decision.action_kind == AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST
    assert prepared.inspection.status.value == "waiting_remote_approval"
    assert transport.dispatches == 0
    remote_request = remote.inspect(
        project_id="project-1",
        run_id="run-1",
        slot_id=prepared.execution.task_slots[0].slot_id,
        expected_slot_binding_digest=remote.inspect_slot_binding(
            project_id="project-1",
            run_id="run-1",
            slot_id=prepared.execution.task_slots[0].slot_id,
        ).slot_binding_digest,
    )["request"]

    approval_request = AgentHarnessRemoteApprovalRequest(
        expected_remote_request_sha256=remote_request["request_sha256"],
        client_request_id="remote-approval-1",
        note="Approve the exact task slot request.",
    )
    original_write_marker = controller.requests.write_marker
    failed_once = False

    def fail_after_remote_authority(session, *, filename, status, values):
        nonlocal failed_once
        if (
            not failed_once
            and session.operation == "remote-approval"
            and filename == "side_effect_observed.json"
        ):
            failed_once = True
            raise RuntimeError("injected remote approval checkpoint crash")
        return original_write_marker(
            session,
            filename=filename,
            status=status,
            values=values,
        )

    monkeypatch.setattr(controller.requests, "write_marker", fail_after_remote_authority)
    with pytest.raises(RuntimeError, match="remote approval checkpoint"):
        controller.approve_remote(
            project_id="project-1",
            controller_execution_id=prepared.execution.controller_execution_id,
            request=approval_request,
            actor="alice",
        )
    recorded = controller.approve_remote(
        project_id="project-1",
        controller_execution_id=prepared.execution.controller_execution_id,
        request=approval_request,
        actor="alice",
    )
    assert recorded.inspection.next_action == AgentHarnessControllerAction.DISPATCH_REMOTE_TASK
    assert transport.dispatches == 0
    replayed_approval = controller.approve_remote(
        project_id="project-1",
        controller_execution_id=prepared.execution.controller_execution_id,
        request=approval_request,
        actor="alice",
    )
    assert replayed_approval.inspection.next_action == AgentHarnessControllerAction.DISPATCH_REMOTE_TASK
    assert transport.dispatches == 0

    cancel_snapshot = tmp_path / "pre-cancel-workspace"
    shutil.copytree(storage.workspace_dir, cancel_snapshot)
    cancel_request = AgentHarnessControllerAdvanceRequest(
        expected_controller_execution_digest=prepared.execution.execution_digest,
        client_request_id="remote-cancel-1",
    )
    cancelled = controller.cancel(
        project_id="project-1",
        controller_execution_id=prepared.execution.controller_execution_id,
        request=cancel_request,
    )
    assert cancelled.decision is not None
    assert cancelled.decision.action_kind == AgentHarnessControllerAction.CANCEL_EXECUTION
    assert cancelled.receipt is not None
    assert cancelled.receipt.reason_codes == ["REMOTE_EXECUTION_CANCELLED"]
    cancel_replay = controller.cancel(
        project_id="project-1",
        controller_execution_id=prepared.execution.controller_execution_id,
        request=cancel_request,
    )
    assert cancel_replay.receipt is not None
    assert cancel_replay.receipt.receipt_id == cancelled.receipt.receipt_id
    assert transport.cancels == 1
    shutil.rmtree(storage.workspace_dir)
    shutil.copytree(cancel_snapshot, storage.workspace_dir)

    transport.fail_dispatch = True
    recovery_dispatch = controller.advance(
        project_id="project-1",
        controller_execution_id=prepared.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=prepared.execution.execution_digest,
            client_request_id="remote-recovery-dispatch-1",
        ),
    )
    assert recovery_dispatch.inspection.status.value == "recovery_required"
    transport.fail_dispatch = False
    transport.status = "RUNNING"
    inspections_before_recovery = transport.inspections
    recover_request = AgentHarnessControllerAdvanceRequest(
        expected_controller_execution_digest=prepared.execution.execution_digest,
        client_request_id="remote-recover-1",
    )
    recovered = controller.recover(
        project_id="project-1",
        controller_execution_id=prepared.execution.controller_execution_id,
        request=recover_request,
    )
    assert recovered.decision is not None
    assert recovered.decision.action_kind == AgentHarnessControllerAction.RECOVER_REMOTE_TASK
    assert recovered.receipt is not None
    assert recovered.receipt.reason_codes == ["REMOTE_RECOVERY_ATTEMPTED"]
    recover_replay = controller.recover(
        project_id="project-1",
        controller_execution_id=prepared.execution.controller_execution_id,
        request=recover_request,
    )
    assert recover_replay.receipt is not None
    assert recover_replay.receipt.receipt_id == recovered.receipt.receipt_id
    assert transport.inspections == inspections_before_recovery + 1
    shutil.rmtree(storage.workspace_dir)
    shutil.copytree(cancel_snapshot, storage.workspace_dir)
    transport.dispatches = 0
    transport.inspections = 0
    transport.cancels = 0
    transport.approval_sha256 = ""
    transport.status = "RUNNING"

    dispatch_request = AgentHarnessControllerAdvanceRequest(
        expected_controller_execution_digest=prepared.execution.execution_digest,
        client_request_id="remote-dispatch-1",
    )
    dispatch_checkpoint_failed = False

    def fail_after_remote_dispatch(session, *, filename, status, values):
        nonlocal dispatch_checkpoint_failed
        if (
            not dispatch_checkpoint_failed
            and session.operation == "advance"
            and filename == "side_effect_observed.json"
        ):
            dispatch_checkpoint_failed = True
            raise RuntimeError("injected remote dispatch checkpoint crash")
        return original_write_marker(
            session,
            filename=filename,
            status=status,
            values=values,
        )

    monkeypatch.setattr(controller.requests, "write_marker", fail_after_remote_dispatch)
    with pytest.raises(RuntimeError, match="remote dispatch checkpoint"):
        controller.advance(
            project_id="project-1",
            controller_execution_id=prepared.execution.controller_execution_id,
            request=dispatch_request,
        )
    assert transport.dispatches == 1
    dispatched = controller.advance(
        project_id="project-1",
        controller_execution_id=prepared.execution.controller_execution_id,
        request=dispatch_request,
    )
    assert dispatched.inspection.status.value == "running_remote"
    assert transport.dispatches == 1

    transport.status = "SUCCEEDED"
    refreshed = controller.advance(
        project_id="project-1",
        controller_execution_id=prepared.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=prepared.execution.execution_digest,
            client_request_id="remote-refresh-1",
        ),
    )
    assert refreshed.decision is not None
    assert refreshed.decision.action_kind == AgentHarnessControllerAction.REFRESH_REMOTE_TASK
    assert refreshed.inspection.next_action == AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS
    assert transport.inspections == 1

    adopted = controller.advance(
        project_id="project-1",
        controller_execution_id=prepared.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=prepared.execution.execution_digest,
            client_request_id="remote-adopt-1",
        ),
    )
    assert adopted.decision is not None
    assert adopted.decision.action_kind == AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS
    assert adopted.receipt is not None
    assert adopted.receipt.remote_publication_digest.startswith("sha256:")
    assert adopted.inspection.status.value == "succeeded"
    assert transport.dispatches == 1


def test_two_remote_tasks_use_distinct_slots_and_run_in_plan_order(
    tmp_path: Path,
) -> None:
    from test_remote_execution_lifecycle import FakeProbe, _bytes, _sha256
    import test_remote_resource_authority as authority_fixtures

    from ai4s_agent.remote_execution_lifecycle import (
        PUBLICATION_VERSION,
        RemoteExecutionLifecycleService,
        RemoteObservation,
        RemotePublication,
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

    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at=_NOW)
    profiles = ResourceProfileStore(
        workspace_dir=storage.workspace_dir,
        config_dir=tmp_path / "config",
    )
    connection = profiles.save_connection(
        ConnectionProfile(
            connection_id="multi-controller-worker",
            ssh_host_alias="multi-controller-worker",
            expected_hostname="multi-controller-worker",
            remote_root="/srv/molly",
            declared_capabilities=["cpu", "gpu", "unimol", "reinvent4"],
        )
    )
    profiles.save_probe(
        CapabilityProbeResult(
            connection_id=connection.connection_id,
            connection_profile_digest=connection.digest(),
            status="available",
            checked_at=_NOW,
            verified_capabilities=connection.declared_capabilities,
            details=CapabilityDetails(
                cpu_threads=32,
                cuda=CudaCapabilityDetails(status="available"),
            ),
        )
    )
    train_outputs = [
        "unimol_model",
        "unimol_training_audit",
        "unimol_training_metrics",
    ]
    registry = AtomicTaskRegistry(
        [
            authority_fixtures._remote_task(
                "model_training",
                "unimol-train-v1",
                task_id="train_remote",
                output_artifacts=train_outputs,
            ),
            authority_fixtures._remote_task(
                "molecular_generation",
                "reinvent4-cpu-v1",
                task_id="generate_remote",
                depends_on=["train_remote"],
                required_artifacts=["unimol_training_metrics"],
                output_artifacts=["reinvent4_candidates"],
            ),
        ]
    )
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
    proposal = ScientificAgentPlanService(
        storage=storage,
        registry=registry,
        observation_builder=builder,
        proposal_store=proposal_store,
        clock=lambda: _NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Run two exact heterogeneous remote tasks",
        user_constraints=[],
        provider=StubLLMProvider(
            response=AgentExecutionPlanLLMResponse(
                requested_tool_ids=["train_remote", "generate_remote"],
                selected_input_artifact_ids=[],
                task_options={"train_remote": {}, "generate_remote": {}},
                selected_logical_profile_ids=[
                    "unimol-train-v1",
                    "reinvent4-cpu-v1",
                ],
                limits={"max_runtime_sec": 2400, "max_gpu_hours": 1.0},
                stop_conditions=["stop on verification failure"],
                success_criteria=["verify two slot publications"],
                rationales=["Exercise exact multi-remote Controller ordering."],
                assumptions=[],
                questions=[],
            ).model_dump(mode="json")
        ),
        client_request_id="multi-remote-proposal-1",
    )
    policy_store = RemoteResourceAuthorityPolicyStore(config_dir=tmp_path / "config")
    policy_store.save(
        RemoteResourceAuthorityPolicy(
            entries=[
                RemoteResourceAuthorityPolicyEntry(
                    policy_id="multi-train-policy",
                    enabled=True,
                    connection_id=connection.connection_id,
                    execution_profile_id="unimol-train-v1",
                    remote_task_type="model_training",
                    allowed_task_ids=["train_remote"],
                    configured_resources=AgentConfiguredRemoteResources(
                        gpu_count=1,
                        cpu_threads=2,
                        walltime_sec=1200,
                    ),
                    budget_limits=AgentRemoteResourceBudgetLimits(
                        max_runtime_sec=1200,
                        max_gpu_hours=1.0,
                    ),
                ),
                RemoteResourceAuthorityPolicyEntry(
                    policy_id="multi-generation-policy",
                    enabled=True,
                    connection_id=connection.connection_id,
                    execution_profile_id="reinvent4-cpu-v1",
                    remote_task_type="molecular_generation",
                    allowed_task_ids=["generate_remote"],
                    configured_resources=AgentConfiguredRemoteResources(
                        gpu_count=0,
                        cpu_threads=1,
                        walltime_sec=600,
                    ),
                    budget_limits=AgentRemoteResourceBudgetLimits(
                        max_runtime_sec=600,
                        max_gpu_hours=1.0,
                    ),
                ),
            ]
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
            client_request_id="multi-remote-authority-1",
        ),
    )
    authorizations = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        registry=registry,
        control_store=control_store,
        resource_authority_resolver=lambda publication, task_id: resources.current_authority(
            publication=publication,
            task_id=task_id,
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
            client_request_id="multi-remote-authorization-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )

    class MultiContractTransport:
        def __init__(self) -> None:
            self.dispatches: list[str] = []
            self.inspections: list[str] = []
            self.statuses: dict[str, str] = {}
            self.approvals: dict[str, str] = {}
            self.payloads: dict[str, dict[str, bytes]] = {}

        def dispatch(self, *, connection, request, approval, tree):
            del connection, tree
            self.dispatches.append(request.task_id)
            self.approvals[request.request_id] = approval.approval_sha256
            self.statuses[request.request_id] = "RUNNING"
            return self._observation(request, "ACCEPTED")

        def inspect(self, *, connection, request):
            del connection
            self.inspections.append(request.task_id)
            return self._observation(
                request,
                self.statuses.get(request.request_id, "RUNNING"),
            )

        def cancel(self, *, connection, request):  # pragma: no cover
            del connection
            return self._observation(request, "CANCELLED")

        def fetch_outputs(self, *, connection, request, publication, tree):
            del connection
            payloads = self.payloads[request.request_id]
            tree.publish_downloaded_outputs(
                artifacts=publication.artifacts,
                fetcher=lambda artifact, descriptor: os.write(
                    descriptor,
                    payloads[artifact.relative_path],
                ),
                digest=_sha256,
                request_sha256=request.request_sha256,
                publication_sha256=publication.publication_sha256,
            )

        def _observation(self, request, status: str):
            publication = None
            if status == "SUCCEEDED":
                if request.output_contract == "unimol-training-output-v1":
                    payloads = {
                        "model/model.pt": b"verified-unimol-model",
                        "model/training_audit.json": (
                            b'{"schema_version":"unimol_training_audit.v1"}'
                        ),
                        "model/training_metrics.json": b'{"metrics":{"loss":0.1}}',
                    }
                    descriptors = [
                        ("unimol_model", "model/model.pt", "application/octet-stream"),
                        (
                            "unimol_training_audit",
                            "model/training_audit.json",
                            "application/json",
                        ),
                        (
                            "unimol_training_metrics",
                            "model/training_metrics.json",
                            "application/json",
                        ),
                    ]
                else:
                    payloads = {"candidates.csv": b"SMILES,score\nCCO,0.9\n"}
                    descriptors = [
                        ("reinvent4_candidates", "candidates.csv", "text/csv")
                    ]
                self.payloads[request.request_id] = payloads
                body = {
                    "schema_version": PUBLICATION_VERSION,
                    "request_id": request.request_id,
                    "request_sha256": request.request_sha256,
                    "approval_sha256": self.approvals[request.request_id],
                    "input_manifest_sha256": request.input_manifest.manifest_sha256,
                    "output_contract": request.output_contract,
                    "artifacts": [
                        {
                            "artifact_id": artifact_id,
                            "relative_path": relative_path,
                            "media_type": media_type,
                            "size_bytes": len(payloads[relative_path]),
                            "sha256": _sha256(payloads[relative_path]),
                        }
                        for artifact_id, relative_path, media_type in descriptors
                    ],
                    "published_at": _NOW,
                }
                body["publication_sha256"] = _sha256(_bytes(body))
                publication = RemotePublication.model_validate(body)
            return RemoteObservation(
                request_id=request.request_id,
                request_sha256=request.request_sha256,
                status=status,
                remote_job_id=f"job-{request.task_id}",
                observed_at=_NOW,
                publication=publication,
            )

    transport = MultiContractTransport()
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
    current = controller.create(
        project_id="project-1",
        start_intent_id=approved.start_intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=approved.start_intent.start_intent_digest,
            client_request_id="multi-remote-controller-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert len({slot.slot_id for slot in current.execution.task_slots}) == 2

    advance_ordinal = 0
    for task_index, task_id in enumerate(["train_remote", "generate_remote"]):
        assert current.inspection.status.value == "waiting_remote_approval"
        slot = current.execution.task_slots[task_index]
        binding = remote.inspect_slot_binding(
            project_id="project-1",
            run_id="run-1",
            slot_id=slot.slot_id,
        )
        controller.approve_remote(
            project_id="project-1",
            controller_execution_id=current.execution.controller_execution_id,
            request=AgentHarnessRemoteApprovalRequest(
                expected_remote_request_sha256=binding.request_sha256,
                client_request_id=f"multi-remote-approval-{task_index}",
                note="Approve this exact ordered task slot.",
            ),
            actor="alice",
        )
        advance_ordinal += 1
        current = controller.advance(
            project_id="project-1",
            controller_execution_id=current.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=current.execution.execution_digest,
                client_request_id=f"multi-remote-dispatch-{advance_ordinal}",
            ),
        )
        request_id = remote.inspect_slot_binding(
            project_id="project-1",
            run_id="run-1",
            slot_id=slot.slot_id,
        ).request_id
        transport.statuses[request_id] = "SUCCEEDED"
        advance_ordinal += 1
        current = controller.advance(
            project_id="project-1",
            controller_execution_id=current.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=current.execution.execution_digest,
                client_request_id=f"multi-remote-refresh-{advance_ordinal}",
            ),
        )
        advance_ordinal += 1
        current = controller.advance(
            project_id="project-1",
            controller_execution_id=current.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=current.execution.execution_digest,
                client_request_id=f"multi-remote-adopt-{advance_ordinal}",
            ),
        )
        if task_index == 0:
            advance_ordinal += 1
            current = controller.advance(
                project_id="project-1",
                controller_execution_id=current.execution.controller_execution_id,
                request=AgentHarnessControllerAdvanceRequest(
                    expected_controller_execution_digest=current.execution.execution_digest,
                    client_request_id=f"multi-remote-prepare-{advance_ordinal}",
                ),
            )
        assert transport.dispatches.count(task_id) == 1

    assert current.inspection.status.value == "succeeded"
    assert transport.dispatches == ["train_remote", "generate_remote"]


def test_controller_routes_are_exact_and_reject_client_authority_fields() -> None:
    class NeverCalledController:
        def __getattr__(self, name):
            raise AssertionError(f"strict request validation did not reject before {name}")

    app = Flask(__name__)
    app.config["AI4S_AGENT_AUTHORIZATION_OWNER"] = "alice"
    register_scientific_agent_harness_controller_routes(
        app,
        controller=NeverCalledController(),  # type: ignore[arg-type]
    )
    rules = {
        rule.rule
        for rule in app.url_map.iter_rules()
        if "controller" in rule.rule or "remote-approvals" in rule.rule
    }
    assert rules == {
        "/api/projects/<project_id>/agent-plan-start-intents/<start_intent_id>/controller-executions",
        "/api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>",
        "/api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/advance",
        "/api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/gates/<gate_id>/approve",
        "/api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/remote-approvals",
        "/api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/cancel",
        "/api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/recover",
    }
    client = app.test_client()
    start = client.post(
        "/api/projects/project-1/agent-plan-start-intents/start-intent-a/controller-executions",
        json={
            "expected_start_intent_digest": "sha256:" + "a" * 64,
            "client_request_id": "request-1",
            "task_id": "injected",
        },
    )
    assert start.status_code == 400
    gate = client.post(
        "/api/projects/project-1/agent-harness-controller-executions/controller-a/gates/gate_1_task_parse/approve",
        json={
            "expected_snapshot_id": "snapshot-a",
            "expected_snapshot_hash": "sha256:" + "a" * 64,
            "client_request_id": "request-2",
            "note": "",
            "approved": True,
        },
    )
    assert gate.status_code == 400
    remote = client.post(
        "/api/projects/project-1/agent-harness-controller-executions/controller-a/remote-approvals",
        json={
            "expected_remote_request_sha256": "sha256:" + "a" * 64,
            "client_request_id": "request-3",
            "note": "",
            "slot_id": "injected",
        },
    )
    assert remote.status_code == 400
