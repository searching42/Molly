from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Mapping, Sequence

import pytest
from flask import Flask

from ai4s_agent import adapters
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
    AgentHarnessAuthorityClass,
    AgentHarnessControllerInspectionFact,
    AgentHarnessGateApprovalRequest,
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


def test_br1_unimol_prediction_remote_inputs_have_exact_purposes() -> None:
    contract = ScientificAgentHarnessController._remote_input_contract
    assert contract("model_inference", ".csv") == (
        "prediction-data",
        "application/csv",
    )
    assert contract("model_inference", ".json") == (
        "prediction-config",
        "application/json",
    )
    assert contract("model_inference", ".yaml") == (
        "model-config",
        "application/yaml",
    )
    assert contract("model_inference", ".pth") == (
        "model-weights",
        "application/octet-stream",
    )
    assert contract("model_inference", ".ss") == (
        "target-scaler",
        "application/octet-stream",
    )


def test_controller_disambiguates_repeated_source_fact_names() -> None:
    facts = [
        AgentHarnessControllerInspectionFact(
            name="authorized_input_artifact",
            authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
            source_id="uploaded_dataset",
            source_digest="sha256:" + "1" * 64,
            state="current",
        ),
        AgentHarnessControllerInspectionFact(
            name="authorized_input_artifact",
            authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
            source_id="source_dataset_manifest",
            source_digest="sha256:" + "2" * 64,
            state="current",
        ),
        AgentHarnessControllerInspectionFact(
            name="authorized_input_artifact",
            authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
            source_id="br1_mapping_policy",
            source_digest="sha256:" + "3" * 64,
            state="current",
        ),
    ]

    bindings = ScientificAgentHarnessController._bindings_from_facts(facts)

    assert [item.name for item in bindings] == [
        "authorized_input_artifact",
        "authorized_input_artifact_source_dataset_manifest",
        "authorized_input_artifact_br1_mapping_policy",
    ]
    assert [item.source_id for item in bindings] == [
        "uploaded_dataset",
        "source_dataset_manifest",
        "br1_mapping_policy",
    ]


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


def _local_authority_chain(
    tmp_path: Path,
    *,
    requested_tool_ids: list[str] | None = None,
):
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at=_NOW)
    run_dir = storage.run_dir("project-1", "run-1")
    dataset = run_dir / "inputs" / "dataset.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("SMILES,value\nCCO,1.0\n", encoding="utf-8")
    storage.register_artifact_path(
        "project-1", "run-1", "uploaded_dataset", "inputs/dataset.csv"
    )
    tool_ids = requested_tool_ids or ["inspect_dataset"]
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=tool_ids,
        selected_input_artifact_ids=["uploaded_dataset"],
        task_options={task_id: {} for task_id in tool_ids},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce a dataset profile"],
        rationales=["Execute the exact registered local task roster."],
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


def _local_prepublication_crash(tmp_path: Path):
    storage, control_store, controller, intent = _local_authority_chain(tmp_path)
    request = AgentHarnessControllerStartRequest(
        expected_start_intent_digest=intent.start_intent_digest,
        client_request_id="local-prepublication-crash-create-1",
    )
    original_publish = controller._publish_local_execution_publication

    def fail_before_publication(**kwargs):
        if kwargs.get("verification_mode") == "controller_dispatch":
            raise RuntimeError("injected prepublication crash")
        return original_publish(**kwargs)

    controller._publish_local_execution_publication = fail_before_publication  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="prepublication crash"):
            controller.create(
                project_id="project-1",
                start_intent_id=intent.start_intent_id,
                request=request,
                actor="alice",
                actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
            )
    finally:
        controller._publish_local_execution_publication = original_publish  # type: ignore[method-assign]
    executions = control_store.list_harness_controller_executions(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
    )
    assert len(executions) == 1
    execution = executions[0]
    assert len(
        control_store.list_harness_local_dispatch_receipts(
            project_id="project-1",
            controller_execution_id=execution.controller_execution_id,
        )
    ) == 1
    assert control_store.list_harness_local_execution_publications(
        project_id="project-1",
        controller_execution_id=execution.controller_execution_id,
    ) == []
    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None and stage.status.value == "SUCCEEDED"
    assert stage.details.get("controller_output_evidence")
    return storage, control_store, intent, request, execution


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


def _local_reconstruction_process(
    workspace_dir: str,
    controller_execution_id: str,
    controller_execution_digest: str,
    results: multiprocessing.queues.Queue,
) -> None:
    try:
        controller = _reopen_local_controller(workspace_dir)
        result = controller.advance(
            project_id="project-1",
            controller_execution_id=controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=controller_execution_digest,
                client_request_id="new-process-reconstruct-advance-1",
            ),
        )
        results.put(
            (
                "ok",
                result.receipt.outcome.value if result.receipt else "",
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
    client_request_id: str,
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
                client_request_id=client_request_id,
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


def _assert_controller_receipt_chain_is_linear(
    controller: ScientificAgentHarnessController,
    *,
    controller_execution_id: str,
) -> None:
    receipts = controller.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=controller_execution_id,
    )
    referenced_predecessors = {
        controller.control_store.read_harness_controller_decision(
            project_id="project-1",
            decision_id=item.decision_id,
        ).predecessor_receipt_id
        for item in receipts
    }
    leaves = [
        item for item in receipts if item.receipt_id not in referenced_predecessors
    ]
    assert len(leaves) == 1


def _remote_controller_authority_chain(tmp_path: Path, monkeypatch):
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
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorizations,
        control_store=authorizations.control_store,
        resource_authority_service=resource_authorities,
        executor=RunPlanExecutor(storage=storage, registry=proposal_store.registry),
        remote_executions=remote,
        clock=lambda: _NOW,
    )
    return storage, controller, approved.start_intent, transport, remote


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


def test_controller_freezes_local_default_adapter_binding_before_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage, controller, intent = _gated_local_authority_chain(tmp_path)
    created = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="freeze-default-adapter-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    slot = created.execution.task_slots[0]
    assert slot.local_adapter_execution_binding_digest
    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    snapshot = stage.details["execution_snapshot"]
    called = False

    def replacement(_payload):
        nonlocal called
        called = True
        return {"status": "failed", "adapter": "replacement"}

    monkeypatch.setattr(adapters, "generate_candidates_stub_adapter", replacement)
    monkeypatch.setattr(
        controller.executor.registry.get("inspect_dataset"),
        "default_adapter",
        "generate_candidates_stub_adapter",
    )
    with pytest.raises(ValueError, match="permission decision is stale|local task authority changed"):
        controller.approve_gate(
            project_id="project-1",
            controller_execution_id=created.execution.controller_execution_id,
            gate_id="gate_1_task_parse",
            request=AgentHarnessGateApprovalRequest(
                expected_snapshot_id=snapshot["snapshot_id"],
                expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
                client_request_id="freeze-default-adapter-gate-1",
                note="Must retain the authorized adapter.",
            ),
            actor="alice",
        )
    assert called is False


def test_controller_freezes_same_id_callable_implementation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage, controller, intent = _gated_local_authority_chain(tmp_path)
    created = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="freeze-callable-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    snapshot = stage.details["execution_snapshot"]
    called = False

    def inspect_dataset_service(_payload):
        nonlocal called
        called = True
        return {"status": "failed", "adapter": "inspect_dataset_service"}

    monkeypatch.setattr(adapters, "inspect_dataset_service", inspect_dataset_service)
    with pytest.raises(ValueError, match="permission decision is stale|local task authority changed"):
        controller.approve_gate(
            project_id="project-1",
            controller_execution_id=created.execution.controller_execution_id,
            gate_id="gate_1_task_parse",
            request=AgentHarnessGateApprovalRequest(
                expected_snapshot_id=snapshot["snapshot_id"],
                expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
                client_request_id="freeze-callable-gate-1",
                note="Must retain the authorized implementation.",
            ),
            actor="alice",
        )
    assert called is False


def test_post_start_fallback_rejects_later_task_callable_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, _, controller, intent = _local_authority_chain(
        tmp_path,
        requested_tool_ids=["inspect_dataset", "check_trainability"],
    )
    created = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="post-start-callable-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert created.receipt is not None
    assert created.receipt.task_id == "inspect_dataset"
    called = False

    def check_trainability_service(_payload):
        nonlocal called
        called = True
        return {"status": "failed", "adapter": "check_trainability_service"}

    monkeypatch.setattr(
        adapters,
        "check_trainability_service",
        check_trainability_service,
    )
    with pytest.raises(ValueError, match="permission decision is stale|local task authority changed"):
        controller.advance(
            project_id="project-1",
            controller_execution_id=created.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=created.execution.execution_digest,
                client_request_id="post-start-callable-advance-1",
            ),
        )
    assert called is False


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
        "controller-cross-process-advance-1",
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


def test_different_local_advance_requests_share_one_execution_lock_and_chain(
    tmp_path: Path,
) -> None:
    storage, controller, intent = _gated_local_authority_chain(tmp_path)
    created = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="different-advance-create-1",
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
            client_request_id="different-advance-gate-1",
            note="Approve the exact concurrent task.",
        ),
        actor="alice",
    )

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    results = context.Queue()
    common = (
        str(storage.workspace_dir),
        created.execution.controller_execution_id,
        created.execution.execution_digest,
    )
    processes = [
        context.Process(
            target=_concurrent_advance_process,
            args=(*common, f"different-advance-{index}", ready, results),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    ready.set()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    observed = [results.get(timeout=2) for _ in processes]
    assert all(item[0] == "ok" for item in observed), observed

    completed = storage.read_stage_state("project-1", "run-1")
    assert completed is not None
    assert [item.status.value for item in completed.history].count("RUNNING") == 1
    assert [item.status.value for item in completed.history].count("SUCCEEDED") == 1
    dispatches = controller.control_store.list_harness_local_dispatch_receipts(
        project_id="project-1",
        controller_execution_id=created.execution.controller_execution_id,
    )
    assert len(dispatches) == 1
    receipts = controller.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=created.execution.controller_execution_id,
    )
    referenced_predecessors = {
        controller.control_store.read_harness_controller_decision(
            project_id="project-1",
            decision_id=item.decision_id,
        ).predecessor_receipt_id
        for item in receipts
    }
    leaves = [
        item for item in receipts if item.receipt_id not in referenced_predecessors
    ]
    assert len(leaves) == 1


def test_advance_and_gate_approval_share_execution_lock_and_linear_chain(
    tmp_path: Path,
) -> None:
    storage, controller, intent = _gated_local_authority_chain(tmp_path)
    created = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="gate-race-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    snapshot = stage.details["execution_snapshot"]
    barrier = threading.Barrier(2)

    def advance():
        barrier.wait(timeout=5)
        return controller.advance(
            project_id="project-1",
            controller_execution_id=created.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=created.execution.execution_digest,
                client_request_id="gate-race-advance-1",
            ),
        )

    def approve_gate():
        barrier.wait(timeout=5)
        return controller.approve_gate(
            project_id="project-1",
            controller_execution_id=created.execution.controller_execution_id,
            gate_id="gate_1_task_parse",
            request=AgentHarnessGateApprovalRequest(
                expected_snapshot_id=snapshot["snapshot_id"],
                expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
                client_request_id="gate-race-approval-1",
                note="Approve the exact concurrent task.",
            ),
            actor="alice",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(advance), pool.submit(approve_gate)]
        assert all(future.result(timeout=10) is not None for future in futures)

    observed = controller.get(
        project_id="project-1",
        controller_execution_id=created.execution.controller_execution_id,
    )
    if observed.inspection.status.value != "succeeded":
        observed = controller.advance(
            project_id="project-1",
            controller_execution_id=created.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=created.execution.execution_digest,
                client_request_id="gate-race-finish-1",
            ),
        )
    assert observed.inspection.status.value == "succeeded"
    dispatches = controller.control_store.list_harness_local_dispatch_receipts(
        project_id="project-1",
        controller_execution_id=created.execution.controller_execution_id,
    )
    assert len(dispatches) == 1
    _assert_controller_receipt_chain_is_linear(
        controller,
        controller_execution_id=created.execution.controller_execution_id,
    )


@pytest.mark.parametrize("peer_operation", ["remote-approval", "cancel", "recover"])
def test_advance_and_remote_control_operation_share_one_linear_chain(
    tmp_path: Path,
    monkeypatch,
    peer_operation: str,
) -> None:
    _, controller, intent, transport, remote = _remote_controller_authority_chain(
        tmp_path,
        monkeypatch,
    )
    created = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id=f"{peer_operation}-race-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    slot = created.execution.task_slots[0]
    binding = remote.inspect_slot_binding(
        project_id="project-1",
        run_id="run-1",
        slot_id=slot.slot_id,
    )

    if peer_operation != "remote-approval":
        controller.approve_remote(
            project_id="project-1",
            controller_execution_id=created.execution.controller_execution_id,
            request=AgentHarnessRemoteApprovalRequest(
                expected_remote_request_sha256=binding.request_sha256,
                client_request_id=f"{peer_operation}-race-prerequisite-approval-1",
                note="Approve the exact task slot request.",
            ),
            actor="alice",
        )
    if peer_operation == "recover":
        transport.fail_dispatch = True
        controller.advance(
            project_id="project-1",
            controller_execution_id=created.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=created.execution.execution_digest,
                client_request_id="recover-race-enter-recovery-1",
            ),
        )
        transport.fail_dispatch = False
        transport.status = "RUNNING"

    barrier = threading.Barrier(2)

    def advance():
        barrier.wait(timeout=5)
        return controller.advance(
            project_id="project-1",
            controller_execution_id=created.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=created.execution.execution_digest,
                client_request_id=f"{peer_operation}-race-advance-1",
            ),
        )

    def peer():
        barrier.wait(timeout=5)
        if peer_operation == "remote-approval":
            return controller.approve_remote(
                project_id="project-1",
                controller_execution_id=created.execution.controller_execution_id,
                request=AgentHarnessRemoteApprovalRequest(
                    expected_remote_request_sha256=binding.request_sha256,
                    client_request_id="remote-approval-race-peer-1",
                    note="Approve the exact concurrent task slot request.",
                ),
                actor="alice",
            )
        request = AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=created.execution.execution_digest,
            client_request_id=f"{peer_operation}-race-peer-1",
        )
        if peer_operation == "cancel":
            return controller.cancel(
                project_id="project-1",
                controller_execution_id=created.execution.controller_execution_id,
                request=request,
            )
        return controller.recover(
            project_id="project-1",
            controller_execution_id=created.execution.controller_execution_id,
            request=request,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(advance), pool.submit(peer)]
        assert all(future.result(timeout=10) is not None for future in futures)

    _assert_controller_receipt_chain_is_linear(
        controller,
        controller_execution_id=created.execution.controller_execution_id,
    )
    assert transport.dispatches <= 1


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


def test_local_prepublication_crash_reconstructs_without_second_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage, control_store, intent, request, execution = (
        _local_prepublication_crash(tmp_path)
    )
    run_dir = storage.run_dir("project-1", "run-1")
    adapter_result = run_dir / "inspect_dataset" / "adapter_result.json"
    result_bytes = adapter_result.read_bytes()
    recovered_controller = _reopen_local_controller(str(storage.workspace_dir))

    def reject_dispatch(_adapter_name):
        raise AssertionError("reconstruction attempted a second adapter dispatch")

    monkeypatch.setattr(recovered_controller.executor, "_adapter_for", reject_dispatch)
    recovered = recovered_controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=request,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert recovered.receipt is not None
    assert recovered.receipt.outcome == AgentHarnessControllerReceiptOutcome.RECONCILED
    assert recovered.receipt.reason_codes == ["TASK_COMPLETED"]
    assert adapter_result.read_bytes() == result_bytes
    dispatches = control_store.list_harness_local_dispatch_receipts(
        project_id="project-1",
        controller_execution_id=execution.controller_execution_id,
    )
    publications = control_store.list_harness_local_execution_publications(
        project_id="project-1",
        controller_execution_id=execution.controller_execution_id,
    )
    assert len(dispatches) == 1
    assert len(publications) == 1
    assert publications[0].verification_mode == "recovered_controller_dispatch"
    assert publications[0].local_dispatch_receipt_id == dispatches[0].dispatch_receipt_id


def test_local_prepublication_crash_recovers_in_new_process(
    tmp_path: Path,
) -> None:
    storage, control_store, _, _, execution = _local_prepublication_crash(tmp_path)
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(
        target=_local_reconstruction_process,
        args=(
            str(storage.workspace_dir),
            execution.controller_execution_id,
            execution.execution_digest,
            results,
        ),
    )
    process.start()
    process.join(timeout=30)
    assert process.exitcode == 0
    assert results.get(timeout=5)[:2] == ("ok", "reconciled")
    assert len(
        control_store.list_harness_local_dispatch_receipts(
            project_id="project-1",
            controller_execution_id=execution.controller_execution_id,
        )
    ) == 1
    publications = control_store.list_harness_local_execution_publications(
        project_id="project-1",
        controller_execution_id=execution.controller_execution_id,
    )
    assert len(publications) == 1
    assert publications[0].verification_mode == "recovered_controller_dispatch"


def test_local_prepublication_reconstruction_rejects_missing_output(
    tmp_path: Path,
) -> None:
    storage, _, intent, request, _ = _local_prepublication_crash(tmp_path)
    registry_path = storage.run_dir("project-1", "run-1") / "artifact_registry.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    del payload["artifacts"]["dataset_profile"]
    registry_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    recovered_controller = _reopen_local_controller(str(storage.workspace_dir))
    with pytest.raises(
        ScientificAgentHarnessControllerVerificationError,
        match="output contract is incomplete",
    ):
        recovered_controller.create(
            project_id="project-1",
            start_intent_id=intent.start_intent_id,
            request=request,
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )


def test_local_prepublication_reconstruction_rejects_same_size_replacement(
    tmp_path: Path,
) -> None:
    storage, _, intent, request, _ = _local_prepublication_crash(tmp_path)
    registry = storage.read_artifact_registry("project-1", "run-1")
    output_path = storage.run_dir("project-1", "run-1") / registry["dataset_profile"]
    original = output_path.read_bytes()
    replacement = bytearray(original)
    replacement[-2] = ord(" ") if replacement[-2] != ord(" ") else ord("\t")
    output_path.write_bytes(bytes(replacement))
    assert output_path.stat().st_size == len(original)
    recovered_controller = _reopen_local_controller(str(storage.workspace_dir))
    with pytest.raises(
        ScientificAgentHarnessControllerVerificationError,
        match="output evidence mismatch",
    ):
        recovered_controller.create(
            project_id="project-1",
            start_intent_id=intent.start_intent_id,
            request=request,
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )


def test_local_prepublication_reconstruction_calls_exact_record_verifier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage, _, intent, request, _ = _local_prepublication_crash(tmp_path)
    registry = storage.read_artifact_registry("project-1", "run-1")
    record_path = storage.run_dir("project-1", "run-1") / registry["dataset_profile"]
    record_payload = json.loads(record_path.read_text(encoding="utf-8"))
    record_payload["schema_version"] = "corrupted-execution-record"
    encoded = json.dumps(record_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    record_path.write_bytes(encoded)
    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    evidence = dict(stage.details["controller_output_evidence"])
    outputs = [dict(item) for item in evidence["outputs"]]
    record_relative_path = registry["dataset_profile"]
    for item in outputs:
        if item["relative_path"] == record_relative_path:
            item["size_bytes"] = len(encoded)
            item["content_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    evidence["outputs"] = outputs
    evidence["outputs_digest"] = _agent_digest(outputs)
    storage.write_stage_state(
        "project-1",
        "run-1",
        stage.model_copy(update={"details": {**stage.details, "controller_output_evidence": evidence}}),
    )
    recovered_controller = _reopen_local_controller(str(storage.workspace_dir))
    original_verify = recovered_controller.executor.verify_one_task_committed_outputs

    def exact_record_verifier(**kwargs):
        original_verify(**kwargs)
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == "corrupted-execution-record":
            raise ValueError("exact execution record replay failed")

    monkeypatch.setattr(
        recovered_controller.executor,
        "verify_one_task_committed_outputs",
        exact_record_verifier,
    )
    with pytest.raises(
        ScientificAgentHarnessControllerVerificationError,
        match="exact task verification",
    ):
        recovered_controller.create(
            project_id="project-1",
            start_intent_id=intent.start_intent_id,
            request=request,
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )


def test_local_prepublication_reconstruction_rejects_stage_mismatch(
    tmp_path: Path,
) -> None:
    storage, _, intent, request, _ = _local_prepublication_crash(tmp_path)
    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    storage.write_stage_state(
        "project-1",
        "run-1",
        stage.model_copy(update={"stage": "different_task"}),
    )
    recovered_controller = _reopen_local_controller(str(storage.workspace_dir))
    with pytest.raises(
        ScientificAgentHarnessControllerVerificationError,
        match="belongs to another task",
    ):
        recovered_controller.create(
            project_id="project-1",
            start_intent_id=intent.start_intent_id,
            request=request,
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )


def test_local_prepublication_reconstruction_rejects_extra_registry_output(
    tmp_path: Path,
) -> None:
    storage, _, intent, request, _ = _local_prepublication_crash(tmp_path)
    extra = storage.run_dir("project-1", "run-1") / "unauthorized.json"
    extra.write_text("{}\n", encoding="utf-8")
    storage.register_artifact_path(
        "project-1",
        "run-1",
        "harness-input-forged-authority",
        "unauthorized.json",
    )
    recovered_controller = _reopen_local_controller(str(storage.workspace_dir))
    with pytest.raises(
        ScientificAgentHarnessControllerVerificationError,
        match="Registry mutation|unauthorized Registry output",
    ):
        recovered_controller.create(
            project_id="project-1",
            start_intent_id=intent.start_intent_id,
            request=request,
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )


def test_local_prepublication_reconstruction_is_concurrent_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage, control_store, _, _, execution = _local_prepublication_crash(tmp_path)
    controllers = [
        _reopen_local_controller(str(storage.workspace_dir)),
        _reopen_local_controller(str(storage.workspace_dir)),
    ]

    def reject_dispatch(_adapter_name):
        raise AssertionError("concurrent reconstruction attempted adapter dispatch")

    for item in controllers:
        monkeypatch.setattr(item.executor, "_adapter_for", reject_dispatch)

    def advance(index: int):
        return controllers[index].advance(
            project_id="project-1",
            controller_execution_id=execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=execution.execution_digest,
                client_request_id=f"concurrent-reconstruct-advance-{index}",
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(advance, index) for index in range(2)]
        completed = [future.result(timeout=20) for future in results]
    assert any(
        item.receipt is not None
        and item.receipt.outcome == AgentHarnessControllerReceiptOutcome.RECONCILED
        for item in completed
    )
    assert len(
        control_store.list_harness_local_dispatch_receipts(
            project_id="project-1",
            controller_execution_id=execution.controller_execution_id,
        )
    ) == 1
    publications = control_store.list_harness_local_execution_publications(
        project_id="project-1",
        controller_execution_id=execution.controller_execution_id,
    )
    assert len(publications) == 1
    assert publications[0].verification_mode == "recovered_controller_dispatch"


def test_local_crash_reconciliation_requires_exact_immutable_execution_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage, control_store, controller, intent = _local_authority_chain(tmp_path)
    verifier_digest = _agent_digest(
        {
            "schema_version": "test-immutable-output-verifier.v1",
            "task_id": "inspect_dataset",
            "execution_record_id": "dataset_profile",
        }
    )

    def immutable_verifier_binding(**_):
        return {
            "verification_class": "immutable_execution_record",
            "verifier_version": "test-immutable-output-verifier.v1",
            "verifier_digest": verifier_digest,
            "execution_record_id": "dataset_profile",
        }

    monkeypatch.setattr(
        controller.executor,
        "one_task_output_verifier_binding",
        immutable_verifier_binding,
    )
    original_write_marker = controller.requests.write_marker
    failed_once = False

    def fail_before_effect_checkpoint(
        session,
        *,
        filename,
        status,
        values,
    ):
        nonlocal failed_once
        if not failed_once and filename == "side_effect_observed.json":
            failed_once = True
            raise RuntimeError("injected immutable receipt publication crash")
        return original_write_marker(
            session,
            filename=filename,
            status=status,
            values=values,
        )

    monkeypatch.setattr(
        controller.requests,
        "write_marker",
        fail_before_effect_checkpoint,
    )
    request = AgentHarnessControllerStartRequest(
        expected_start_intent_digest=intent.start_intent_digest,
        client_request_id="immutable-record-crash-create-1",
    )
    with pytest.raises(RuntimeError, match="immutable receipt publication crash"):
        controller.create(
            project_id="project-1",
            start_intent_id=intent.start_intent_id,
            request=request,
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )

    executions = control_store.list_harness_controller_executions(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
    )
    assert len(executions) == 1
    publications = control_store.list_harness_local_execution_publications(
        project_id="project-1",
        controller_execution_id=executions[0].controller_execution_id,
    )
    assert len(publications) == 1
    binding = publications[0].verified_outputs[0]
    assert binding.verification_class == "immutable_execution_record"
    assert binding.execution_record_id == "dataset_profile"
    assert binding.execution_record_digest == binding.content_sha256

    registry_path = storage.run_dir("project-1", "run-1") / "artifact_registry.json"
    registry_bytes = registry_path.read_bytes()
    registry_payload = json.loads(registry_bytes.decode("utf-8"))
    del registry_payload["artifacts"]["dataset_profile"]
    registry_path.write_text(
        json.dumps(registry_payload, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ScientificAgentHarnessControllerVerificationError,
        match="local output contract is incomplete",
    ):
        controller.create(
            project_id="project-1",
            start_intent_id=intent.start_intent_id,
            request=request,
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )

    registry_path.write_bytes(registry_bytes)
    recovered = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=request,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert recovered.receipt is not None
    assert recovered.receipt.outcome == AgentHarnessControllerReceiptOutcome.RECONCILED
    assert recovered.receipt.verified_output_bindings[0].execution_record_digest
    assert len(
        control_store.list_harness_local_dispatch_receipts(
            project_id="project-1",
            controller_execution_id=recovered.execution.controller_execution_id,
        )
    ) == 1


def test_local_completion_fails_closed_without_exact_dispatch_receipt(
    tmp_path: Path,
) -> None:
    _, control_store, controller, intent = _local_authority_chain(tmp_path)
    completed = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="missing-dispatch-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert completed.receipt is not None
    assert completed.receipt.local_dispatch_receipt_ids
    controller_dispatches = control_store.list_harness_local_dispatch_receipts(
        project_id="project-1",
        controller_execution_id=completed.execution.controller_execution_id,
    )
    assert len(controller_dispatches) == 1
    assert completed.receipt.local_dispatch_receipt_ids == [
        controller_dispatches[0].executor_dispatch_receipt_id
        or controller_dispatches[0].dispatch_receipt_id
    ]
    dispatch_id = controller_dispatches[0].dispatch_receipt_id
    dispatch_root = control_store._collection_root(
        project_id="project-1",
        kind="harness_local_dispatch_receipt",
        create=False,
    )
    assert dispatch_root is not None
    (dispatch_root / dispatch_id).rename(dispatch_root / f"hidden-{dispatch_id}")

    with pytest.raises(
        ScientificAgentHarnessControllerVerificationError,
        match="dispatch receipt is unavailable",
    ):
        controller.get(
            project_id="project-1",
            controller_execution_id=completed.execution.controller_execution_id,
        )


def test_manual_local_success_without_controller_dispatch_is_explicitly_adopted(
    tmp_path: Path,
) -> None:
    storage, controller, intent = _gated_local_authority_chain(tmp_path)
    created = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="adopt-manual-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    authorization = controller.authorization_service.verify_authorization(
        project_id="project-1",
        authorization_id=intent.authorization_id,
        verify_current=False,
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
            client_request_id="adopt-manual-gate-1",
            note="Approve the exact manual task.",
        ),
        actor="alice",
    )
    binding = controller.executor.derive_one_task_server_binding(
        project_id="project-1",
        run_plan=authorization.run_plan,
        task_index=0,
        task_options=authorization.compiled_task_options["inspect_dataset"],
    )
    manual = controller.executor.execute_one_task_after_committed_gate(
        project_id="project-1",
        run_plan=authorization.run_plan,
        task_index=0,
        task_id="inspect_dataset",
        task_options=authorization.compiled_task_options["inspect_dataset"],
        actor="alice",
        expected_snapshot_id=snapshot["snapshot_id"],
        expected_snapshot_digest=f"sha256:{snapshot['snapshot_hash']}",
        expected_local_adapter_execution_binding_digest=binding[
            "local_adapter_execution_binding_digest"
        ],
        expected_compiled_options_digest=binding["compiled_options_digest"],
        expected_input_artifacts_digest=binding["input_artifacts_digest"],
        expected_output_contract_digest=binding["output_contract_digest"],
    )
    assert manual["status"] == "SUCCEEDED"

    adopted = controller.advance(
        project_id="project-1",
        controller_execution_id=created.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=created.execution.execution_digest,
            client_request_id="adopt-manual-advance-1",
        ),
    )
    assert adopted.decision is not None
    assert (
        adopted.decision.action_kind
        == AgentHarnessControllerAction.ADOPT_COMPLETED_TASK
    )
    assert adopted.receipt is not None
    assert adopted.receipt.reason_codes == ["TASK_ADOPTED"]
    assert adopted.receipt.local_dispatch_receipt_ids == []
    assert adopted.receipt.execution_started is False
    assert adopted.receipt.dispatch_occurred is False
    assert (
        controller.control_store.list_harness_local_dispatch_receipts(
            project_id="project-1",
            controller_execution_id=adopted.execution.controller_execution_id,
        )
        == []
    )


def test_local_output_same_path_same_size_replacement_fails_closed(
    tmp_path: Path,
) -> None:
    storage, _, controller, intent = _local_authority_chain(tmp_path)
    completed = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="tampered-output-create-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    registry = storage.read_artifact_registry("project-1", "run-1")
    output_path = storage.run_dir("project-1", "run-1") / registry["dataset_profile"]
    original = output_path.read_bytes()
    replacement = bytearray(original)
    replacement[-2] = ord(" ") if replacement[-2] != ord(" ") else ord("\t")
    output_path.write_bytes(bytes(replacement))
    assert output_path.stat().st_size == len(original)

    with pytest.raises(
        ScientificAgentHarnessControllerVerificationError,
        match="no longer verifies current outputs",
    ):
        controller.get(
            project_id="project-1",
            controller_execution_id=completed.execution.controller_execution_id,
        )


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


def test_decision_freshness_barrier_rechecks_input_after_decision_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage, control_store, controller, intent = _local_authority_chain(tmp_path)
    dataset = storage.run_dir("project-1", "run-1") / "inputs" / "dataset.csv"
    original_publish = control_store.publish_harness_controller_decision
    mutated = False

    def publish_then_mutate(*, project_id, decision):
        nonlocal mutated
        published = original_publish(project_id=project_id, decision=decision)
        if not mutated:
            mutated = True
            original = dataset.read_bytes()
            replacement = bytearray(original)
            replacement[-2] = ord("2") if replacement[-2] != ord("2") else ord("3")
            dataset.write_bytes(bytes(replacement))
        return published

    monkeypatch.setattr(
        control_store,
        "publish_harness_controller_decision",
        publish_then_mutate,
    )
    with pytest.raises(
        ScientificAgentHarnessControllerVerificationError,
        match="pre-existing artifact authority changed|input artifact content changed",
    ):
        controller.create(
            project_id="project-1",
            start_intent_id=intent.start_intent_id,
            request=AgentHarnessControllerStartRequest(
                expected_start_intent_digest=intent.start_intent_digest,
                client_request_id="freshness-race-create-1",
            ),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )
    executions = control_store.list_harness_controller_executions(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
    )
    assert len(executions) == 1
    assert control_store.list_harness_local_dispatch_receipts(
        project_id="project-1",
        controller_execution_id=executions[0].controller_execution_id,
    ) == []
    assert storage.read_stage_state("project-1", "run-1") is None


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

    external_controller_inputs = tmp_path / "external-controller-inputs"
    external_controller_inputs.mkdir()
    sentinel = external_controller_inputs / "sentinel.txt"
    sentinel.write_bytes(b"unchanged")
    (
        storage.run_dir("project-1", "run-1")
        / "agent-harness-controller-inputs"
    ).symlink_to(external_controller_inputs, target_is_directory=True)

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
    assert {path.name for path in external_controller_inputs.iterdir()} == {
        "sentinel.txt"
    }
    assert sentinel.read_bytes() == b"unchanged"
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
    slot_root = (
        storage.run_dir("project-1", "run-1")
        / "remote-executions"
        / prepared.execution.task_slots[0].slot_id
    )
    slot_bytes_before = {
        str(path.relative_to(slot_root)): path.read_bytes()
        for path in sorted(slot_root.rglob("*"))
        if path.is_file()
    }
    ordinary_recovery_observation = controller.advance(
        project_id="project-1",
        controller_execution_id=prepared.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=prepared.execution.execution_digest,
            client_request_id="remote-recovery-observation-1",
        ),
    )
    assert ordinary_recovery_observation.decision is not None
    assert (
        ordinary_recovery_observation.decision.action_kind
        == AgentHarnessControllerAction.RECOVER_REMOTE_TASK
    )
    assert ordinary_recovery_observation.decision.executable is False
    assert ordinary_recovery_observation.receipt is not None
    assert (
        ordinary_recovery_observation.receipt.outcome
        == AgentHarnessControllerReceiptOutcome.WAITING
    )
    assert transport.inspections == inspections_before_recovery
    assert {
        str(path.relative_to(slot_root)): path.read_bytes()
        for path in sorted(slot_root.rglob("*"))
        if path.is_file()
    } == slot_bytes_before
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

    telemetry_path = slot_root / "state.json"
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    telemetry["status"] = "FAILED"
    telemetry["error_code"] = "mutable-telemetry-only"
    telemetry["updated_at"] = "2026-08-02T00:00:00Z"
    telemetry_path.write_text(
        json.dumps(telemetry, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    terminal_after_telemetry_change = controller.get(
        project_id="project-1",
        controller_execution_id=prepared.execution.controller_execution_id,
    )
    assert terminal_after_telemetry_change.inspection.status.value == "succeeded"

    stage_path = slot_root / "stage.json"
    slot_stage = json.loads(stage_path.read_text(encoding="utf-8"))
    slot_stage["details"]["untrusted-extra-observation"] = "tampered"
    stage_path.write_text(
        json.dumps(slot_stage, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ScientificAgentHarnessControllerVerificationError,
        match="remote completion receipt no longer verifies current authority",
    ):
        controller.get(
            project_id="project-1",
            controller_execution_id=prepared.execution.controller_execution_id,
        )


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
