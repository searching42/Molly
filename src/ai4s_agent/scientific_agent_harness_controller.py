"""Deterministic one-action Scientific Agent Harness Controller v1.

The Controller consumes an exact start intent, re-verifies every authority,
selects one server-owned action in RunPlan order, commits the decision before
the effect, and commits an immutable evidence receipt afterwards.  It never
accepts a client-selected task, route, adapter, profile, resource, or action.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ai4s_agent._utils import now_iso
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.harness_tracing import HarnessTracer, NoopHarnessTracer
from ai4s_agent.observability_correlation import (
    build_harness_telemetry_correlation,
    privacy_safe_telemetry_attributes,
)
from ai4s_agent.oled_scientific_agent_source_evidence import read_dispatch_receipts
from ai4s_agent.resource_profiles import build_transfer_manifest_from_payloads
from ai4s_agent.schemas import (
    AgentHarnessAuthorityClass,
    AgentHarnessControllerAction,
    AgentHarnessControllerActionBoundaryClass,
    AgentHarnessControllerActionReceipt,
    AgentHarnessControllerAdvanceRequest,
    AgentHarnessControllerDecision,
    AgentHarnessControllerExecution,
    AgentHarnessControllerInspection,
    AgentHarnessControllerInspectionFact,
    AgentHarnessControllerReceiptOutcome,
    AgentHarnessControllerSourceBinding,
    AgentHarnessControllerStartRequest,
    AgentHarnessControllerStatus,
    AgentHarnessControllerTaskSlot,
    AgentHarnessGateApprovalRequest,
    AgentHarnessLocalDispatchReceipt,
    AgentHarnessLocalExecutionPublication,
    AgentHarnessRemoteApprovalRequest,
    AgentHarnessVerifiedOutputBinding,
    AgentPermissionOutcome,
    AgentPermissionPhase,
    AgentPlanAuthorization,
    RunStatus,
    _agent_digest,
)
from ai4s_agent.scientific_agent_authorization import AgentPlanControlStore
from ai4s_agent.scientific_agent_permissions import (
    IMPLEMENTATION_BOUND_PERMISSION_POLICY_VERSION,
    IMPLEMENTATION_BOUND_RESOURCE_AWARE_PERMISSION_POLICY_VERSION,
    MODEL_INFERENCE_RESOURCE_AWARE_PERMISSION_POLICY_VERSION,
    derive_local_task_authority_material,
)
from ai4s_agent.scientific_agent_plan import (
    ScientificAgentPlanPublication,
    ScientificAgentPlanSourceChanged,
    _exclusive_process_lock,
    _fsync_directory,
    _pretty_json_bytes,
    _read_exact_bytes,
    _read_stable_file,
    _safe_artifact_path,
    _safe_relative_artifact_path,
    _safe_scope_id,
    _write_exclusive,
)


CONTROLLER_POLICY_VERSION = "scientific-agent-harness-controller-policy.v1"
CONTROLLER_REQUEST_VERSION = "agent_harness_controller_request_checkpoint.v1"
_MAX_REQUEST_BYTES = 4 * 1024 * 1024
_CONTROLLER_REQUEST_LOCKS: dict[str, threading.RLock] = defaultdict(threading.RLock)
_POLICY_MATERIAL: Mapping[str, Any] = {
    "schema_version": CONTROLLER_POLICY_VERSION,
    "recognized_action_kinds": sorted(item.value for item in AgentHarnessControllerAction),
    "next_action_precedence": [
        "verify_authority",
        "observe_prior_receipt",
        "observe_task_outputs",
        "gate",
        "remote_request",
        "remote_approval",
        "remote_dispatch",
        "remote_refresh",
        "remote_recovery",
        "remote_adoption",
        "local_execution",
        "terminal_observation",
    ],
    "exact_authority_verification": [
        "start_intent",
        "authorization",
        "authorized_start_permission",
        "proposal_publication",
        "remote_authority_set_when_required",
    ],
    "route_separation": {
        "local": "run_plan_executor.execute_one_task",
        "remote": "remote_execution_lifecycle.task_slot",
    },
    "gate_policy": "exact_current_snapshot_positive_server_actor_decision",
    "remote_approval_policy": "record_only_dispatch_is_separate",
    "effect_policy": "at_most_one_bounded_effect_per_decision",
    "attempt_policy": "explicit_zero_based_attempt_ordinal_no_auto_retry",
    "recovery_policy": "reconcile_committed_authority_never_rerun_unknown_local_effect",
    "adoption_policy": "verified_stage_registry_or_remote_publication_only",
    "terminal_policy": "authoritative_sources_plus_committed_controller_receipt",
    "source_binding_policy": "ids_and_sha256_digests_only",
    "local_adapter_authority_policy": (
        "permission_engine_shared_task_authority_and_callable_implementation_digest"
    ),
    "local_completion_reconstruction_policy": (
        "exact_dispatch_stage_registry_output_and_task_verifier_replay"
    ),
    "privacy_allowlist": "no_paths_hosts_commands_prompts_exceptions_or_trace_identity",
    "tracing_policy": "optional_fail_open_non_authoritative",
    "authority_expansion": "forbidden",
    "reason_codes": sorted(
        {
            "ALL_TASKS_COMPLETED",
            "GATE_APPROVAL_REQUIRED",
            "GATE_DECISION_COMMITTED",
            "GATE_SNAPSHOT_READY",
            "LOCAL_TASK_COMPLETED",
            "LOCAL_TASK_FAILED",
            "LOCAL_TASK_READY",
            "REMOTE_APPROVAL_REQUIRED",
            "REMOTE_APPROVAL_RECORDED",
            "REMOTE_DISPATCH_READY",
            "REMOTE_EXECUTION_CANCELLED",
            "REMOTE_EXECUTION_FAILED",
            "REMOTE_EXECUTION_RUNNING",
            "REMOTE_OUTPUTS_ADOPTED",
            "REMOTE_RECOVERY_ATTEMPTED",
            "REMOTE_RECOVERY_REQUIRED",
            "REMOTE_REQUEST_PREPARED",
            "REMOTE_REQUEST_READY",
            "TASK_COMPLETED",
            "TASK_ADOPTED",
            "TASK_INPUTS_UNAVAILABLE",
            "TERMINAL_OBSERVED",
        }
    ),
}


def _controller_telemetry_attributes(
    execution: AgentHarnessControllerExecution,
    *,
    operation: str,
    component: str,
    phase: str,
    slot: AgentHarnessControllerTaskSlot | None = None,
) -> dict[str, str | int | bool]:
    context = build_harness_telemetry_correlation(
        project_id=execution.project_id,
        run_id=execution.run_id,
        proposal_id=execution.proposal_id,
        proposal_digest=execution.proposal_digest,
        semantic_plan_id=execution.semantic_plan_id,
        semantic_plan_digest=execution.semantic_plan_digest,
        permission_decision_id=execution.permission_decision_id,
        authorization_id=execution.authorization_id,
        start_intent_id=execution.start_intent_id,
        controller_execution_id=execution.controller_execution_id,
        controller_execution_digest=execution.execution_digest,
        task_id=slot.task_id if slot is not None else "",
        task_index=slot.planned_task_index if slot is not None else None,
        slot_id=slot.slot_id if slot is not None else "",
        execution_route=slot.execution_route if slot is not None else "",
        operation=operation,
        component=component,
        phase=phase,
    )
    return privacy_safe_telemetry_attributes(context)
CONTROLLER_POLICY_DIGEST = _agent_digest(_POLICY_MATERIAL)


_TERMINAL_CONTROLLER_ACTIONS = frozenset(
    {
        AgentHarnessControllerAction.STOP_GATE_REJECTED,
        AgentHarnessControllerAction.STOP_REMOTE_REJECTED,
        AgentHarnessControllerAction.STOP_TASK_TERMINAL,
        AgentHarnessControllerAction.COMPLETE_EXECUTION,
    }
)


def controller_action_boundary_class(
    action: AgentHarnessControllerAction,
    *,
    terminal_receipt_committed: bool = False,
) -> AgentHarnessControllerActionBoundaryClass:
    """Classify one Controller action without creating a second scheduler.

    Terminal actions remain ordinary advances until their exact Controller
    receipt exists; after that immutable boundary they become read-only
    terminal observations for the Execution Agent.
    """

    if action == AgentHarnessControllerAction.WAIT_FOR_GATE:
        return AgentHarnessControllerActionBoundaryClass.USER_GATE_APPROVAL
    if action == AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL:
        return AgentHarnessControllerActionBoundaryClass.USER_REMOTE_APPROVAL
    if action in {
        AgentHarnessControllerAction.RECOVER_REMOTE_TASK,
        AgentHarnessControllerAction.CANCEL_EXECUTION,
    }:
        return AgentHarnessControllerActionBoundaryClass.EXPLICIT_RECOVERY
    if action in _TERMINAL_CONTROLLER_ACTIONS and terminal_receipt_committed:
        return AgentHarnessControllerActionBoundaryClass.TERMINAL_OBSERVATION
    return AgentHarnessControllerActionBoundaryClass.ORDINARY_ADVANCE


class ScientificAgentHarnessControllerError(ValueError):
    """Base privacy-safe Controller failure."""


class ScientificAgentHarnessControllerConflict(ScientificAgentHarnessControllerError):
    """An immutable request, decision, or receipt is bound to other bytes."""


class ScientificAgentHarnessControllerVerificationError(
    ScientificAgentHarnessControllerError
):
    """A current authority or result failed exact verification."""


class ScientificAgentHarnessControllerRecoveryRequired(
    ScientificAgentHarnessControllerError
):
    """A prior effect cannot be safely repeated automatically."""


@dataclass(frozen=True)
class ControllerRequestSession:
    project_id: str
    operation: str
    scope_id: str
    client_request_id: str
    request_digest: str
    request_dir: Path


@dataclass(frozen=True)
class ControllerAdvanceResult:
    execution: AgentHarnessControllerExecution
    inspection: AgentHarnessControllerInspection
    decision: AgentHarnessControllerDecision | None = None
    receipt: AgentHarnessControllerActionReceipt | None = None


class AgentHarnessControllerStore:
    """Request checkpoints layered over the existing immutable control store."""

    def __init__(self, *, control_store: AgentPlanControlStore) -> None:
        self.control_store = control_store

    @contextmanager
    def scope_session(
        self,
        *,
        project_id: str,
        operation: str,
        scope_id: str,
    ):
        project = _safe_scope_id(project_id, field="project_id")
        operation_id = _safe_scope_id(operation, field="operation")
        clean_scope = _safe_scope_id(scope_id, field="scope_id")
        control = self.control_store._control_root(project_id=project, create=True)
        if control is None:  # pragma: no cover
            raise ScientificAgentHarnessControllerError("Controller storage unavailable")
        root = self._directory(control, "controller_requests")
        operation_root = self._directory(root, operation_id)
        scope_root = self._directory(operation_root, clean_scope)
        lock = scope_root / "scope.lock"
        if lock.is_symlink():
            raise ScientificAgentHarnessControllerError("Controller scope lock is unsafe")
        with _CONTROLLER_REQUEST_LOCKS[str(lock.resolve())], _exclusive_process_lock(lock):
            yield

    @contextmanager
    def execution_session(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
    ):
        """Serialize every mutating operation for one Controller execution.

        Callers acquire this after the create-only start-intent scope lock and
        before any client-request lock.  Keeping the lock through immutable
        receipt publication prevents different request IDs and operations from
        selecting the same receipt predecessor or repeating one side effect.
        """

        project = _safe_scope_id(project_id, field="project_id")
        execution_id = _safe_scope_id(
            controller_execution_id,
            field="controller_execution_id",
        )
        control = self.control_store._control_root(project_id=project, create=True)
        if control is None:  # pragma: no cover
            raise ScientificAgentHarnessControllerError("Controller storage unavailable")
        root = self._directory(control, "controller_execution_locks")
        execution_root = self._directory(root, execution_id)
        lock = execution_root / "controller_execution.lock"
        if lock.is_symlink():
            raise ScientificAgentHarnessControllerError(
                "Controller execution lock is unsafe"
            )
        with _CONTROLLER_REQUEST_LOCKS[str(lock.resolve())], _exclusive_process_lock(lock):
            yield

    @contextmanager
    def request_session(
        self,
        *,
        project_id: str,
        operation: str,
        scope_id: str,
        client_request_id: str,
        request_digest: str,
    ):
        project = _safe_scope_id(project_id, field="project_id")
        operation_id = _safe_scope_id(operation, field="operation")
        request_id = _safe_scope_id(client_request_id, field="client_request_id")
        clean_scope = _safe_scope_id(scope_id, field="scope_id")
        control = self.control_store._control_root(project_id=project, create=True)
        if control is None:  # pragma: no cover
            raise ScientificAgentHarnessControllerError("Controller storage unavailable")
        root = self._directory(control, "controller_requests")
        operation_root = self._directory(root, operation_id)
        scope_root = self._directory(operation_root, clean_scope)
        request_dir = self._directory(scope_root, request_id)
        lock = request_dir / "request.lock"
        if lock.is_symlink():
            raise ScientificAgentHarnessControllerError("Controller request lock is unsafe")
        with _CONTROLLER_REQUEST_LOCKS[str(lock.resolve())], _exclusive_process_lock(lock):
            session = ControllerRequestSession(
                project_id=project,
                operation=operation_id,
                scope_id=clean_scope,
                client_request_id=request_id,
                request_digest=request_digest,
                request_dir=request_dir,
            )
            self.write_or_verify(
                request_dir / "reservation.json",
                {
                    "schema_version": CONTROLLER_REQUEST_VERSION,
                    "status": "RESERVED",
                    "project_id": project,
                    "operation": operation_id,
                    "scope_id": clean_scope,
                    "client_request_id": request_id,
                    "request_digest": request_digest,
                },
            )
            yield session

    def write_marker(
        self,
        session: ControllerRequestSession,
        *,
        filename: str,
        status: str,
        values: Mapping[str, Any],
    ) -> None:
        self.write_or_verify(
            session.request_dir / filename,
            {
                "schema_version": CONTROLLER_REQUEST_VERSION,
                "status": status,
                "project_id": session.project_id,
                "operation": session.operation,
                "scope_id": session.scope_id,
                "client_request_id": session.client_request_id,
                "request_digest": session.request_digest,
                **dict(values),
            },
        )

    @staticmethod
    def read_marker(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise ScientificAgentHarnessControllerConflict("Controller checkpoint is unsafe")
        try:
            payload = json.loads(
                _read_exact_bytes(path, label="Controller checkpoint", max_bytes=_MAX_REQUEST_BYTES)
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise ScientificAgentHarnessControllerConflict(
                "Controller checkpoint failed exact verification"
            ) from exc
        if not isinstance(payload, dict):
            raise ScientificAgentHarnessControllerConflict("Controller checkpoint is invalid")
        return payload

    @staticmethod
    def write_or_verify(path: Path, payload: Mapping[str, Any]) -> None:
        expected = _pretty_json_bytes(dict(payload))
        if path.is_symlink():
            raise ScientificAgentHarnessControllerConflict("Controller checkpoint is unsafe")
        if path.exists():
            actual = _read_exact_bytes(
                path, label="Controller checkpoint", max_bytes=_MAX_REQUEST_BYTES
            )
            if actual != expected:
                raise ScientificAgentHarnessControllerConflict(
                    "Controller request ID is bound to different content"
                )
            return
        try:
            _write_exclusive(path, expected)
        except FileExistsError:
            actual = _read_exact_bytes(
                path, label="Controller checkpoint", max_bytes=_MAX_REQUEST_BYTES
            )
            if actual != expected:
                raise ScientificAgentHarnessControllerConflict(
                    "Controller request ID is bound to different content"
                )

    @staticmethod
    def _directory(parent: Path, name: str) -> Path:
        path = parent / name
        if path.is_symlink():
            raise ScientificAgentHarnessControllerError("Controller storage is unsafe")
        if not path.exists():
            try:
                path.mkdir(mode=0o700, parents=False, exist_ok=False)
            except FileExistsError:
                pass
            _fsync_directory(parent)
        if path.is_symlink() or not path.is_dir():
            raise ScientificAgentHarnessControllerError("Controller storage is unsafe")
        resolved = path.resolve()
        if not resolved.is_relative_to(parent.resolve()):
            raise ScientificAgentHarnessControllerError("Controller storage escapes project scope")
        return resolved


class ScientificAgentHarnessController:
    """Server-owned deterministic Controller with an at-most-one-action API."""

    def __init__(
        self,
        *,
        storage: Any,
        proposal_store: Any,
        authorization_service: Any,
        control_store: AgentPlanControlStore,
        resource_authority_service: Any,
        executor: RunPlanExecutor,
        remote_executions: Any,
        tracer: HarnessTracer | None = None,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self.storage = storage
        self.proposal_store = proposal_store
        self.authorization_service = authorization_service
        self.control_store = control_store
        self.resource_authority_service = resource_authority_service
        self.executor = executor
        self.remote_executions = remote_executions
        self.tracer = tracer or NoopHarnessTracer()
        self.clock = clock
        self.requests = AgentHarnessControllerStore(control_store=control_store)

    def create(
        self,
        *,
        project_id: str,
        start_intent_id: str,
        request: AgentHarnessControllerStartRequest,
        actor: str,
        actor_source: str,
    ) -> ControllerAdvanceResult:
        request_digest = self._request_digest(
            project_id=project_id,
            operation="create",
            scope_id=start_intent_id,
            request=request.model_dump(mode="json"),
            actor=actor,
            actor_source=actor_source,
        )
        scope = self._scope_id("create", start_intent_id)
        with self.tracer.start_span(
            "controller.execution",
            attributes={
                "project_id": project_id,
                "start_intent_id": start_intent_id,
                "controller_policy_version": CONTROLLER_POLICY_VERSION,
                "operation": "agent.controller.create",
                "component": "controller",
                "phase": "create",
            },
        ) as controller_span:
            with self.requests.scope_session(
                project_id=project_id,
                operation="create",
                scope_id=scope,
            ):
                publish_execution = False
                existing = self.control_store.list_harness_controller_executions(
                    project_id=project_id,
                    start_intent_id=start_intent_id,
                )
                if len(existing) > 1:
                    raise ScientificAgentHarnessControllerVerificationError(
                        "start intent has multiple Controller executions"
                    )
                if existing:
                    execution = existing[0]
                    if (
                        execution.request_digest != request_digest
                        or execution.client_request_id != request.client_request_id
                        or execution.actor != actor
                        or execution.actor_source != actor_source
                    ):
                        raise ScientificAgentHarnessControllerConflict(
                            "start intent is already consumed by another request"
                        )
                else:
                    intent = self.authorization_service.verify_start_intent(
                        project_id=project_id,
                        start_intent_id=start_intent_id,
                        verify_current=True,
                    )
                    if intent.start_intent_digest != request.expected_start_intent_digest:
                        raise ScientificAgentHarnessControllerConflict(
                            "start intent digest does not match the current authority"
                        )
                    authorization = self.authorization_service.verify_authorization(
                        project_id=project_id,
                        authorization_id=intent.authorization_id,
                        verify_current=True,
                    )
                    publication = self.proposal_store.read(
                        project_id=project_id,
                        proposal_id=intent.proposal_id,
                        verify_current=True,
                    )
                    permission = self.control_store.read_permission_decision(
                        project_id=project_id,
                        decision_id=intent.permission_decision_id,
                    )
                    execution = self._build_execution(
                        intent=intent,
                        authorization=authorization,
                        publication=publication,
                        permission=permission,
                        actor=actor,
                        actor_source=actor_source,
                        client_request_id=request.client_request_id,
                        request_digest=request_digest,
                        created_at=self.clock(),
                    )
                    publish_execution = True
                with self.requests.execution_session(
                    project_id=project_id,
                    controller_execution_id=execution.controller_execution_id,
                ):
                    if publish_execution:
                        self.control_store.publish_harness_controller_execution(
                            execution
                        )
                    execution = self.verify_execution(
                        project_id=project_id,
                        controller_execution_id=execution.controller_execution_id,
                    )
                    with self.requests.request_session(
                        project_id=project_id,
                        operation="create",
                        scope_id=scope,
                        client_request_id=request.client_request_id,
                        request_digest=request_digest,
                    ) as session:
                        marker = self.requests.read_marker(
                            session.request_dir / "execution.json"
                        )
                        if marker is not None and (
                            marker.get("controller_execution_id")
                            != execution.controller_execution_id
                            or marker.get("controller_execution_digest")
                            != execution.execution_digest
                        ):
                            raise ScientificAgentHarnessControllerConflict(
                                "Controller create checkpoint authority mismatch"
                            )
                        self.requests.write_marker(
                            session,
                            filename="execution.json",
                            status="EXECUTION_COMMITTED",
                            values={
                                "controller_execution_id": execution.controller_execution_id,
                                "controller_execution_digest": execution.execution_digest,
                            },
                        )
                        for key, value in _controller_telemetry_attributes(
                            execution,
                            operation="agent.controller.create",
                            component="controller",
                            phase="committed",
                        ).items():
                            controller_span.set_attribute(key, value)
                        return self._advance_in_session(
                            execution=execution,
                            session=session,
                        )

    def get(
        self, *, project_id: str, controller_execution_id: str
    ) -> ControllerAdvanceResult:
        with self.tracer.start_span(
            "controller.action",
            attributes={
                "project_id": project_id,
                "controller_execution_id": controller_execution_id,
                "operation": "agent.controller.inspect",
                "component": "controller",
                "phase": "read",
            },
        ) as inspect_span:
            execution = self.verify_execution(
                project_id=project_id,
                controller_execution_id=controller_execution_id,
            )
            for key, value in _controller_telemetry_attributes(
                execution,
                operation="agent.controller.inspect",
                component="controller",
                phase="completed",
            ).items():
                inspect_span.set_attribute(key, value)
            return ControllerAdvanceResult(
                execution=execution,
                inspection=self._inspect(execution),
            )

    def read_execution_agent_snapshot(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        expected_controller_execution_digest: str = "",
    ) -> ControllerAdvanceResult:
        """Return one current read-only snapshot under the execution-wide lock."""

        with self.execution_agent_snapshot_session(
            project_id=project_id,
            controller_execution_id=controller_execution_id,
            expected_controller_execution_digest=(
                expected_controller_execution_digest
            ),
        ) as snapshot:
            return snapshot

    @contextmanager
    def execution_agent_snapshot_session(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        expected_controller_execution_digest: str = "",
    ):
        """Yield a read-only snapshot while retaining the execution lock."""

        with self._verified_execution_session(
            project_id=project_id,
            controller_execution_id=controller_execution_id,
            expected_execution_digest=expected_controller_execution_digest,
        ) as execution:
            yield ControllerAdvanceResult(
                execution=execution,
                inspection=self._inspect(execution),
                receipt=self._latest_receipt(execution),
            )

    @contextmanager
    def _verified_execution_session(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        expected_execution_digest: str = "",
    ):
        """Pin the execution-wide lock before reading mutable run authority."""

        with self.requests.execution_session(
            project_id=project_id,
            controller_execution_id=controller_execution_id,
        ):
            execution = self.verify_execution(
                project_id=project_id,
                controller_execution_id=controller_execution_id,
            )
            if (
                expected_execution_digest
                and execution.execution_digest != expected_execution_digest
            ):
                raise ScientificAgentHarnessControllerConflict(
                    "Controller execution digest does not match the current authority"
                )
            yield execution

    def advance(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        request: AgentHarnessControllerAdvanceRequest,
        expected_inspection_digest: str = "",
    ) -> ControllerAdvanceResult:
        request_digest = self._request_digest(
            project_id=project_id,
            operation="advance",
            scope_id=controller_execution_id,
            request=request.model_dump(mode="json"),
        )
        with self.tracer.start_span(
            "controller.advance",
            attributes={
                "project_id": project_id,
                "controller_execution_id": controller_execution_id,
                "operation": "agent.controller.advance",
                "component": "controller",
                "phase": "advance",
            },
        ) as advance_span:
            with self._verified_execution_session(
                project_id=project_id,
                controller_execution_id=controller_execution_id,
                expected_execution_digest=request.expected_controller_execution_digest,
            ) as execution:
                for key, value in _controller_telemetry_attributes(
                    execution,
                    operation="agent.controller.advance",
                    component="controller",
                    phase="advance",
                ).items():
                    advance_span.set_attribute(key, value)
                with self.requests.request_session(
                    project_id=project_id,
                    operation="advance",
                    scope_id=self._scope_id("advance", controller_execution_id),
                    client_request_id=request.client_request_id,
                    request_digest=request_digest,
                ) as session:
                    # A committed request must remain exactly replayable after its
                    # own effect changed the inspection.  Fresh first calls still
                    # fail closed inside both execution- and request-wide locks.
                    if (
                        expected_inspection_digest
                        and self.requests.read_marker(
                            session.request_dir / "decision.json"
                        )
                        is None
                        and self._inspect(execution).inspection_digest
                        != expected_inspection_digest
                    ):
                        raise ScientificAgentHarnessControllerConflict(
                            "Controller inspection no longer matches the expected snapshot"
                        )
                    return self._advance_in_session(
                        execution=execution,
                        session=session,
                    )

    def approve_gate(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        gate_id: str,
        request: AgentHarnessGateApprovalRequest,
        actor: str,
    ) -> ControllerAdvanceResult:
        request_digest = self._request_digest(
            project_id=project_id,
            operation="gate-approval",
            scope_id=f"{controller_execution_id}:{gate_id}",
            request=request.model_dump(mode="json"),
            actor=actor,
        )
        with self._verified_execution_session(
            project_id=project_id,
            controller_execution_id=controller_execution_id,
        ) as execution, self.requests.request_session(
            project_id=project_id,
            operation="gate-approval",
            scope_id=self._scope_id("gate-approval", f"{controller_execution_id}:{gate_id}"),
            client_request_id=request.client_request_id,
            request_digest=request_digest,
        ) as session:
            marker = self.requests.read_marker(session.request_dir / "side_effect_observed.json")
            if marker is not None:
                return ControllerAdvanceResult(
                    execution=execution, inspection=self._inspect(execution)
                )
            inspection = self._inspect(execution)
            if (
                inspection.status != AgentHarnessControllerStatus.WAITING_GATE
                and inspection.next_action
                != AgentHarnessControllerAction.EXECUTE_LOCAL_TASK
            ):
                raise ScientificAgentHarnessControllerConflict(
                    "Controller is not waiting for a Gate"
                )
            slot = self._current_slot(execution, inspection)
            authorization = self._authorization(execution, verify_current=False)
            if gate_id not in authorization.pending_gates:
                raise ScientificAgentHarnessControllerVerificationError(
                    "Gate is not pending in the exact authorization"
                )
            if gate_id in authorization.preauthorized_operational_gates:
                raise ScientificAgentHarnessControllerVerificationError(
                    "preauthorized Gate cannot be approved through the stepwise route"
                )
            self._verify_local_task_authority(execution, slot)
            gate_decision = self.executor.commit_one_task_gate_decision(
                project_id=project_id,
                run_plan=authorization.run_plan,
                task_index=slot.planned_task_index,
                task_id=slot.task_id,
                gate_id=gate_id,
                approved=True,
                actor=actor,
                note=request.note,
                expected_snapshot_id=request.expected_snapshot_id,
                expected_snapshot_digest=request.expected_snapshot_hash,
                task_options=authorization.compiled_task_options[slot.task_id],
            )
            self.requests.write_marker(
                session,
                filename="side_effect_observed.json",
                status="SIDE_EFFECT_OBSERVED",
                values={
                    "gate_decision_digest": _agent_digest(
                        gate_decision.model_dump(mode="json")
                    )
                },
            )
            return ControllerAdvanceResult(
                execution=execution, inspection=self._inspect(execution)
            )

    def approve_remote(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        request: AgentHarnessRemoteApprovalRequest,
        actor: str,
    ) -> ControllerAdvanceResult:
        request_digest = self._request_digest(
            project_id=project_id,
            operation="remote-approval",
            scope_id=controller_execution_id,
            request=request.model_dump(mode="json"),
            actor=actor,
        )
        with self._verified_execution_session(
            project_id=project_id,
            controller_execution_id=controller_execution_id,
        ) as execution, self.requests.request_session(
            project_id=project_id,
            operation="remote-approval",
            scope_id=self._scope_id("remote-approval", controller_execution_id),
            client_request_id=request.client_request_id,
            request_digest=request_digest,
        ) as session:
            marker = self.requests.read_marker(session.request_dir / "side_effect_observed.json")
            if marker is not None:
                return ControllerAdvanceResult(
                    execution=execution, inspection=self._inspect(execution)
                )
            inspection = self._inspect(execution)
            if (
                inspection.status
                != AgentHarnessControllerStatus.WAITING_REMOTE_APPROVAL
                and inspection.next_action
                != AgentHarnessControllerAction.DISPATCH_REMOTE_TASK
            ):
                raise ScientificAgentHarnessControllerConflict(
                    "Controller is not waiting for remote approval"
                )
            slot = self._current_slot(execution, inspection)
            binding = self._remote_slot_binding(execution, slot)
            if binding.request_sha256 != request.expected_remote_request_sha256:
                raise ScientificAgentHarnessControllerConflict(
                    "remote approval does not bind the current exact request"
                )
            result = self.remote_executions.record_approval(
                project_id=project_id,
                run_id=execution.run_id,
                request_sha256=binding.request_sha256,
                actor=actor,
                note=request.note,
                slot_id=slot.slot_id,
                expected_slot_binding_digest=binding.slot_binding_digest,
            )
            approval = result.get("approval")
            if not isinstance(approval, dict) or not approval.get("approval_sha256"):
                raise ScientificAgentHarnessControllerVerificationError(
                    "remote approval commit is unavailable"
                )
            self.requests.write_marker(
                session,
                filename="side_effect_observed.json",
                status="SIDE_EFFECT_OBSERVED",
                values={"remote_approval_digest": approval["approval_sha256"]},
            )
            return ControllerAdvanceResult(
                execution=execution, inspection=self._inspect(execution)
            )

    def cancel(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        request: AgentHarnessControllerAdvanceRequest,
    ) -> ControllerAdvanceResult:
        request_digest = self._request_digest(
            project_id=project_id,
            operation="cancel",
            scope_id=controller_execution_id,
            request=request.model_dump(mode="json"),
        )
        with self._verified_execution_session(
            project_id=project_id,
            controller_execution_id=controller_execution_id,
            expected_execution_digest=request.expected_controller_execution_digest,
        ) as execution, self.requests.request_session(
            project_id=project_id,
            operation="cancel",
            scope_id=self._scope_id("cancel", controller_execution_id),
            client_request_id=request.client_request_id,
            request_digest=request_digest,
        ) as session:
            return self._explicit_remote_action_in_session(
                execution=execution,
                session=session,
                action=AgentHarnessControllerAction.CANCEL_EXECUTION,
                require_recovery=False,
            )

    def recover(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        request: AgentHarnessControllerAdvanceRequest,
    ) -> ControllerAdvanceResult:
        request_digest = self._request_digest(
            project_id=project_id,
            operation="recover",
            scope_id=controller_execution_id,
            request=request.model_dump(mode="json"),
        )
        with self._verified_execution_session(
            project_id=project_id,
            controller_execution_id=controller_execution_id,
            expected_execution_digest=request.expected_controller_execution_digest,
        ) as execution, self.requests.request_session(
            project_id=project_id,
            operation="recover",
            scope_id=self._scope_id("recover", controller_execution_id),
            client_request_id=request.client_request_id,
            request_digest=request_digest,
        ) as session:
            return self._explicit_remote_action_in_session(
                execution=execution,
                session=session,
                action=AgentHarnessControllerAction.RECOVER_REMOTE_TASK,
                require_recovery=True,
            )

    def verify_execution(
        self, *, project_id: str, controller_execution_id: str
    ) -> AgentHarnessControllerExecution:
        execution = self.control_store.read_harness_controller_execution(
            project_id=project_id,
            controller_execution_id=controller_execution_id,
        )
        try:
            intent = self.authorization_service.verify_start_intent(
                project_id=project_id,
                start_intent_id=execution.start_intent_id,
                verify_current=True,
            )
            authorization = self.authorization_service.verify_authorization(
                project_id=project_id,
                authorization_id=execution.authorization_id,
                verify_current=True,
            )
            publication = self.proposal_store.read(
                project_id=project_id,
                proposal_id=execution.proposal_id,
                verify_current=True,
            )
        except ScientificAgentPlanSourceChanged:
            # A committed Controller action intentionally changes StageState
            # and/or Registry, so the pre-execution planner snapshot cannot
            # remain byte-identical.  Accept only those dynamic changes that
            # are anchored by the latest immutable receipt (or one committed
            # decision awaiting crash reconciliation), while independently
            # rechecking static sources and exact input contents.
            intent = self.authorization_service.verify_start_intent(
                project_id=project_id,
                start_intent_id=execution.start_intent_id,
                verify_current=False,
            )
            authorization = self.authorization_service.verify_authorization(
                project_id=project_id,
                authorization_id=execution.authorization_id,
                verify_current=False,
            )
            publication = self.proposal_store.read(
                project_id=project_id,
                proposal_id=execution.proposal_id,
                verify_current=False,
            )
            self._verify_post_start_sources(execution, authorization, publication)
        permission = self.control_store.read_permission_decision(
            project_id=project_id,
            decision_id=execution.permission_decision_id,
        )
        expected = self._build_execution(
            intent=intent,
            authorization=authorization,
            publication=publication,
            permission=permission,
            actor=execution.actor,
            actor_source=execution.actor_source,
            client_request_id=execution.client_request_id,
            request_digest=execution.request_digest,
            created_at=execution.created_at,
        )
        if expected.model_dump(mode="json") != execution.model_dump(mode="json"):
            raise ScientificAgentHarnessControllerVerificationError(
                "Controller execution no longer matches current exact authority"
            )
        return execution

    def _build_execution(
        self,
        *,
        intent: Any,
        authorization: AgentPlanAuthorization,
        publication: ScientificAgentPlanPublication,
        permission: Any,
        actor: str,
        actor_source: str,
        client_request_id: str,
        request_digest: str,
        created_at: str,
    ) -> AgentHarnessControllerExecution:
        if (
            permission.phase != AgentPermissionPhase.AUTHORIZED_START
            or permission.outcome != AgentPermissionOutcome.ALLOW
            or permission.decision_digest != intent.permission_decision_digest
            or permission.authorization_id != authorization.authorization_id
            or permission.authorization_digest != authorization.authorization_digest
        ):
            raise ScientificAgentHarnessControllerVerificationError(
                "start permission is not an exact authorized-start ALLOW"
            )
        if publication.proposal.proposal_digest != authorization.proposal_digest:
            raise ScientificAgentHarnessControllerVerificationError(
                "proposal does not match authorization"
            )
        dispatch_by_task = {item.task_id: item for item in authorization.dispatch_intents}
        permission_by_task = {item.task_id: item for item in permission.task_decisions}
        ordered = [item.task_id for item in authorization.run_plan.tasks]
        if set(permission_by_task) != set(ordered):
            raise ScientificAgentHarnessControllerVerificationError(
                "permission task authority roster is incomplete"
            )
        remote_bindings: dict[str, Any] = {}
        authority_set = None
        for task_id in ordered:
            intent_binding = dispatch_by_task[task_id]
            if intent_binding.execution_route != "remote_execution_service":
                continue
            current = self.resource_authority_service.current_authority(
                publication=publication,
                task_id=task_id,
            )
            if authority_set is None:
                authority_set = current.authority_set
            elif current.authority_set.model_dump(mode="json") != authority_set.model_dump(
                mode="json"
            ):
                raise ScientificAgentHarnessControllerVerificationError(
                    "remote tasks do not share one complete current AuthoritySet"
                )
            remote_bindings[task_id] = current.authority
        slots: list[AgentHarnessControllerTaskSlot] = []
        dispatch_digests: dict[str, str] = {}
        for index, task in enumerate(authorization.run_plan.tasks):
            dispatch = dispatch_by_task[task.task_id]
            dispatch_digest = _agent_digest(dispatch.model_dump(mode="json"))
            dispatch_digests[task.task_id] = dispatch_digest
            remote = remote_bindings.get(task.task_id)
            local_adapter_binding = ""
            if dispatch.execution_route == "local_executor":
                if permission.policy_version not in {
                    IMPLEMENTATION_BOUND_PERMISSION_POLICY_VERSION,
                    IMPLEMENTATION_BOUND_RESOURCE_AWARE_PERMISSION_POLICY_VERSION,
                    MODEL_INFERENCE_RESOURCE_AWARE_PERMISSION_POLICY_VERSION,
                }:
                    raise ScientificAgentHarnessControllerVerificationError(
                        "local Controller tasks require implementation-bound permission authority"
                    )
                try:
                    local_material = derive_local_task_authority_material(
                        publication=publication,
                        task_id=task.task_id,
                        registry=self.executor.registry,
                        policy_version=permission.policy_version,
                    )
                except ValueError as exc:
                    raise ScientificAgentHarnessControllerVerificationError(
                        "local task authority is unavailable"
                    ) from exc
                permission_task = permission_by_task[task.task_id]
                if (
                    local_material.local_adapter_execution_binding_digest is None
                    or local_material.execution_binding_digest
                    != permission_task.execution_binding_digest
                    or local_material.task_authority_digest
                    != permission_task.task_authority_digest
                    or local_material.task_authority_digest
                    != authorization.task_authority_digests[task.task_id]
                ):
                    raise ScientificAgentHarnessControllerVerificationError(
                        "local task authority changed after authorization"
                    )
                local_adapter_binding = (
                    local_material.local_adapter_execution_binding_digest
                )
            slot_identity = _agent_digest(
                {
                    "schema_version": "agent_harness_task_slot_identity.v1",
                    "start_intent_id": intent.start_intent_id,
                    "task_index": index,
                    "task_id": task.task_id,
                    "attempt_ordinal": 0,
                }
            )
            slots.append(
                AgentHarnessControllerTaskSlot(
                    planned_task_index=index,
                    task_id=task.task_id,
                    attempt=0,
                    execution_route=dispatch.execution_route,
                    slot_id=f"harness-slot-{slot_identity.split(':', 1)[1][:32]}",
                    task_authority_digest=authorization.task_authority_digests[task.task_id],
                    local_adapter_execution_binding_digest=local_adapter_binding,
                    dispatch_intent_digest=dispatch_digest,
                    compiled_options_digest=_agent_digest(
                        authorization.compiled_task_options[task.task_id]
                    ),
                    input_artifacts_digest=_agent_digest(
                        {"required_artifact_ids": list(task.required_artifacts)}
                    ),
                    output_contract_digest=_agent_digest(
                        {
                            "task_id": task.task_id,
                            "output_artifact_ids": list(task.output_artifacts),
                        }
                    ),
                    remote_authority_id=remote.authority_id if remote else "",
                    remote_authority_digest=remote.authority_digest if remote else "",
                )
            )
        sources = [
            self._source("start_intent", intent.start_intent_id, intent.start_intent_digest),
            self._source(
                "authorization", authorization.authorization_id, authorization.authorization_digest
            ),
            self._source("permission", permission.decision_id, permission.decision_digest),
            self._source(
                "proposal", publication.proposal.proposal_id, publication.proposal.proposal_digest
            ),
        ]
        if authority_set is not None:
            sources.append(
                self._source(
                    "remote_authority_set",
                    authority_set.authority_set_id,
                    authority_set.authority_set_digest,
                )
            )
        unavailable_budget = _agent_digest(
            {"schema_version": "agent_harness_remote_budget_unavailable.v1", "reason": "local-only-plan"}
        )
        return AgentHarnessControllerExecution(
            project_id=authorization.project_id,
            run_id=authorization.run_id,
            start_intent_id=intent.start_intent_id,
            start_intent_digest=intent.start_intent_digest,
            authorization_id=authorization.authorization_id,
            authorization_digest=authorization.authorization_digest,
            authorization_mode=authorization.authorization_mode,
            permission_decision_id=permission.decision_id,
            permission_decision_digest=permission.decision_digest,
            permission_policy_version=permission.policy_version,
            permission_policy_digest=permission.policy_digest,
            proposal_id=publication.proposal.proposal_id,
            proposal_digest=publication.proposal.proposal_digest,
            semantic_plan_id=publication.proposal.semantic_plan_id,
            semantic_plan_digest=publication.proposal.semantic_plan_digest,
            observation_id=publication.observation.observation_id,
            observation_digest=publication.observation.observation_digest,
            tool_catalog_digest=publication.catalog.catalog_digest,
            run_plan_digest=authorization.run_plan_digest,
            ordered_task_ids=ordered,
            task_roster_digest=_agent_digest(
                {"schema_version": "agent_harness_controller_task_roster.v1", "task_ids": ordered}
            ),
            task_authority_digests=authorization.task_authority_digests,
            dispatch_intent_digests=dispatch_digests,
            compiled_task_options_digest=_agent_digest(authorization.compiled_task_options),
            artifact_binding_digest=_agent_digest(
                [item.model_dump(mode="json") for item in authorization.artifact_bindings]
            ),
            gate_binding_digest=_agent_digest(
                [item.model_dump(mode="json") for item in authorization.gate_bindings]
            ),
            budget_binding_digest=_agent_digest(authorization.limits),
            remote_authority_set_id=authority_set.authority_set_id if authority_set else "",
            remote_authority_set_digest=authority_set.authority_set_digest if authority_set else "",
            remote_authority_roster_digest=authority_set.complete_roster_digest if authority_set else "",
            aggregate_budget_digest=(
                authority_set.aggregate_budget_digest if authority_set else unavailable_budget
            ),
            task_slots=slots,
            source_bindings=sources,
            source_bindings_digest=_agent_digest(
                [item.model_dump(mode="json") for item in sources]
            ),
            controller_policy_digest=CONTROLLER_POLICY_DIGEST,
            actor=actor,
            actor_source=actor_source,
            client_request_id=client_request_id,
            request_digest=request_digest,
            created_at=created_at,
        )

    def _inspect(
        self,
        execution: AgentHarnessControllerExecution,
        *,
        verify_authority: bool = False,
    ) -> AgentHarnessControllerInspection:
        authorization = self._authorization(execution, verify_current=verify_authority)
        stage = self.storage.read_stage_state(execution.project_id, execution.run_id)
        registry = self.storage.read_artifact_registry(execution.project_id, execution.run_id)
        receipts = self.control_store.list_harness_controller_action_receipts(
            project_id=execution.project_id,
            controller_execution_id=execution.controller_execution_id,
        )
        facts = [
            AgentHarnessControllerInspectionFact(
                name="controller_execution",
                authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
                source_id=execution.controller_execution_id,
                source_digest=execution.execution_digest,
                state="verified",
            ),
            AgentHarnessControllerInspectionFact(
                name="artifact_registry",
                authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
                source_id=f"registry-{execution.run_id}",
                source_digest=_agent_digest(registry),
                state="verified",
            ),
        ]
        run_dir = self.storage.run_dir(execution.project_id, execution.run_id)
        for binding in authorization.artifact_bindings:
            relative = registry.get(binding.artifact_id)
            if not relative:
                raise ScientificAgentHarnessControllerVerificationError(
                    "authorized Controller input is no longer registered"
                )
            source = _safe_artifact_path(
                run_dir,
                _safe_relative_artifact_path(relative),
                label="authorized Controller input",
            )
            payload, present = _read_stable_file(
                source,
                label="authorized Controller input",
                max_bytes=2 * 1024 * 1024 * 1024,
            )
            current_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
            if not present or current_digest != binding.content_digest:
                raise ScientificAgentHarnessControllerVerificationError(
                    "authorized Controller input content changed"
                )
            facts.append(
                AgentHarnessControllerInspectionFact(
                    name="authorized_input_artifact",
                    authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
                    source_id=binding.artifact_id,
                    source_digest=current_digest,
                    state="verified",
                )
            )
        if stage is not None:
            facts.append(
                AgentHarnessControllerInspectionFact(
                    name="stage_state",
                    authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
                    source_id=f"stage-{execution.run_id}",
                    source_digest=_agent_digest(stage.model_dump(mode="json")),
                    state=stage.status.value.lower(),
                )
            )
        completed = 0
        for slot in execution.task_slots:
            task = authorization.run_plan.tasks[slot.planned_task_index]
            if self._local_task_completed(
                execution,
                slot,
                task,
                stage,
                registry,
                receipts,
            ):
                completed += 1
                continue
            if slot.execution_route == "remote_execution_service":
                remote = self._remote_inspection_or_none(execution, slot)
                if remote is not None and self._remote_task_completed(
                    execution, remote, receipts, slot, registry
                ):
                    completed += 1
                    continue
            break
        if completed == len(execution.task_slots):
            return self._inspection(
                execution, AgentHarnessControllerStatus.SUCCEEDED, None,
                AgentHarnessControllerAction.COMPLETE_EXECUTION, facts
            )
        slot = execution.task_slots[completed]
        task = authorization.run_plan.tasks[slot.planned_task_index]
        missing = [item for item in task.required_artifacts if item not in registry]
        if missing:
            facts.append(
                AgentHarnessControllerInspectionFact(
                    name="task_inputs",
                    authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
                    state="unavailable",
                    detail="required logical artifact IDs are not registered",
                )
            )
            return self._inspection(
                execution, AgentHarnessControllerStatus.FAILED, slot,
                AgentHarnessControllerAction.STOP_TASK_TERMINAL, facts
            )
        if slot.execution_route == "remote_execution_service":
            return self._inspect_remote_action(execution, slot, facts)
        return self._inspect_local_action(execution, slot, stage, registry, facts)

    def _inspect_local_action(self, execution: Any, slot: Any, stage: Any, registry: Any, facts: list[Any]):
        task = self._authorization(execution, verify_current=False).run_plan.tasks[slot.planned_task_index]
        spec = self.executor.registry.get(task.task_id)
        if stage is not None and stage.stage == task.task_id:
            if stage.status == RunStatus.RUNNING:
                return self._inspection(execution, AgentHarnessControllerStatus.RECOVERY_REQUIRED, slot, AgentHarnessControllerAction.STOP_TASK_TERMINAL, facts)
            if stage.status in {RunStatus.FAILED, RunStatus.CANCELLED}:
                status = AgentHarnessControllerStatus.FAILED if stage.status == RunStatus.FAILED else AgentHarnessControllerStatus.CANCELLED
                return self._inspection(execution, status, slot, AgentHarnessControllerAction.STOP_TASK_TERMINAL, facts)
            if stage.status == RunStatus.SUCCEEDED:
                publications = [
                    item
                    for item in self.control_store.list_harness_local_execution_publications(
                        project_id=execution.project_id,
                        controller_execution_id=execution.controller_execution_id,
                    )
                    if item.slot_id == slot.slot_id
                    and item.task_id == slot.task_id
                    and item.attempt_ordinal == slot.attempt
                ]
                dispatches = [
                    item
                    for item in self.control_store.list_harness_local_dispatch_receipts(
                        project_id=execution.project_id,
                        controller_execution_id=execution.controller_execution_id,
                    )
                    if item.slot_id == slot.slot_id
                    and item.task_id == slot.task_id
                    and item.attempt_ordinal == slot.attempt
                ]
                if dispatches and not publications:
                    return self._inspection(
                        execution,
                        AgentHarnessControllerStatus.RECOVERY_REQUIRED,
                        slot,
                        AgentHarnessControllerAction.STOP_TASK_TERMINAL,
                        facts,
                    )
                if len(publications) > 1:
                    raise ScientificAgentHarnessControllerVerificationError(
                        "local task has conflicting execution publications"
                    )
                if publications:
                    self._verify_local_execution_publication(
                        execution=execution,
                        slot=slot,
                        publication=publications[0],
                    )
                else:
                    self._verified_local_outputs(execution=execution, slot=slot)
                return self._inspection(
                    execution,
                    AgentHarnessControllerStatus.ACTIVE,
                    slot,
                    AgentHarnessControllerAction.ADOPT_COMPLETED_TASK,
                    facts,
                )
        if spec.gates:
            if stage is None or stage.stage != task.task_id or stage.status != RunStatus.WAITING_USER:
                return self._inspection(execution, AgentHarnessControllerStatus.ACTIVE, slot, AgentHarnessControllerAction.PREPARE_LOCAL_GATE, facts)
            snapshot = stage.details.get("execution_snapshot")
            if not isinstance(snapshot, dict):
                raise ScientificAgentHarnessControllerVerificationError("waiting Gate snapshot is unavailable")
            decisions = self._gate_decisions(execution, slot, snapshot, spec.gates)
            if len(decisions) != len(spec.gates):
                return self._inspection(execution, AgentHarnessControllerStatus.WAITING_GATE, slot, AgentHarnessControllerAction.WAIT_FOR_GATE, facts)
            if any(not item.approved for item in decisions.values()):
                return self._inspection(execution, AgentHarnessControllerStatus.FAILED, slot, AgentHarnessControllerAction.STOP_GATE_REJECTED, facts)
        return self._inspection(execution, AgentHarnessControllerStatus.ACTIVE, slot, AgentHarnessControllerAction.EXECUTE_LOCAL_TASK, facts)

    def _inspect_remote_action(self, execution: Any, slot: Any, facts: list[Any]):
        remote = self._remote_inspection_or_none(execution, slot)
        if remote is None:
            return self._inspection(execution, AgentHarnessControllerStatus.ACTIVE, slot, AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST, facts)
        state = str(remote["effective_status"])
        request = remote["request"]
        facts.append(AgentHarnessControllerInspectionFact(
            name="remote_request", authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
            source_id=str(request["request_id"]),
            source_digest=str(remote["request_digest"]),
            state="verified",
        ))
        slot_binding = remote.get("slot_binding")
        if isinstance(slot_binding, dict):
            facts.append(
                AgentHarnessControllerInspectionFact(
                    name="remote_slot_binding",
                    authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
                    source_id=str(slot_binding["slot_id"]),
                    source_digest=str(remote["slot_binding_digest"]),
                    state="verified",
                )
            )
        approval = remote.get("approval")
        if isinstance(approval, dict):
            facts.append(
                AgentHarnessControllerInspectionFact(
                    name="remote_approval",
                    authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
                    source_id=f"remote-approval-{slot.slot_id}",
                    source_digest=str(remote["approval_digest"]),
                    state="approved",
                )
            )
        remote_stage = remote.get("slot_stage_state")
        if isinstance(remote_stage, dict):
            facts.append(
                AgentHarnessControllerInspectionFact(
                    name="remote_stage_state",
                    authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
                    source_id=f"remote-stage-{slot.slot_id}",
                    source_digest=str(remote["slot_stage_digest"]),
                    state=str(remote_stage["status"]).lower(),
                )
            )
        publication = remote.get("publication")
        if isinstance(publication, dict):
            facts.append(
                AgentHarnessControllerInspectionFact(
                    name="remote_publication",
                    authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
                    source_id=f"remote-publication-{slot.slot_id}",
                    source_digest=str(remote["publication_digest"]),
                    state="verified",
                )
            )
        transport_state = remote.get("transport_state")
        if isinstance(transport_state, dict):
            facts.append(
                AgentHarnessControllerInspectionFact(
                    name="remote_transport_state",
                    authority_class=AgentHarnessAuthorityClass.OBSERVATIONAL,
                    source_id=f"remote-transport-{slot.slot_id}",
                    source_digest=str(remote["transport_state_digest"]),
                    state=str(transport_state["status"]).lower(),
                )
            )
        facts.append(
            AgentHarnessControllerInspectionFact(
                name="remote_effective_status",
                authority_class=AgentHarnessAuthorityClass.DERIVED,
                source_id=f"remote-status-{slot.slot_id}",
                source_digest=str(remote["status_source_roster_digest"]),
                state=state.lower(),
            )
        )
        if state == "WAITING_APPROVAL":
            return self._inspection(execution, AgentHarnessControllerStatus.WAITING_REMOTE_APPROVAL, slot, AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL, facts)
        if state == "APPROVED":
            return self._inspection(execution, AgentHarnessControllerStatus.ACTIVE, slot, AgentHarnessControllerAction.DISPATCH_REMOTE_TASK, facts)
        if state in {"ACCEPTED", "RUNNING", "CANCEL_REQUESTED"}:
            return self._inspection(execution, AgentHarnessControllerStatus.RUNNING_REMOTE, slot, AgentHarnessControllerAction.REFRESH_REMOTE_TASK, facts)
        if state == "RECOVERY_REQUIRED":
            return self._inspection(execution, AgentHarnessControllerStatus.RECOVERY_REQUIRED, slot, AgentHarnessControllerAction.RECOVER_REMOTE_TASK, facts)
        if state == "SUCCEEDED":
            return self._inspection(execution, AgentHarnessControllerStatus.ACTIVE, slot, AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS, facts)
        status = AgentHarnessControllerStatus.CANCELLED if state == "CANCELLED" else AgentHarnessControllerStatus.FAILED
        return self._inspection(execution, status, slot, AgentHarnessControllerAction.STOP_TASK_TERMINAL, facts)

    def _advance_in_session(self, *, execution: Any, session: ControllerRequestSession) -> ControllerAdvanceResult:
        receipt_marker = self.requests.read_marker(session.request_dir / "receipt.json")
        if receipt_marker is not None:
            receipt = self.control_store.read_harness_controller_action_receipt(
                project_id=execution.project_id,
                receipt_id=str(receipt_marker.get("receipt_id") or ""),
            )
            return ControllerAdvanceResult(execution, self._inspect(execution), receipt=receipt)
        decision_marker = self.requests.read_marker(session.request_dir / "decision.json")
        if decision_marker is not None:
            decision = self.control_store.read_harness_controller_decision(
                project_id=execution.project_id,
                decision_id=str(decision_marker.get("decision_id") or ""),
            )
        else:
            pending = self._unreceipted_decisions(execution)
            if len(pending) > 1:
                raise ScientificAgentHarnessControllerVerificationError(
                    "multiple Controller decisions lack receipts"
                )
            if pending:
                decision = pending[0]
            else:
                inspection = self._inspect(execution)
                predecessor = self._latest_receipt(execution)
                sources = self._bindings_from_facts(inspection.facts)
                slot = None if inspection.current_task_index is None else execution.task_slots[inspection.current_task_index]
                decision = AgentHarnessControllerDecision(
                    controller_execution_id=execution.controller_execution_id,
                    controller_execution_digest=execution.execution_digest,
                    client_request_id=session.client_request_id,
                    inspection_digest=inspection.inspection_digest,
                    action_kind=inspection.next_action,
                    task_id=slot.task_id if slot else "",
                    task_index=slot.planned_task_index if slot else None,
                    attempt_ordinal=slot.attempt if slot else 0,
                    slot_id=slot.slot_id if slot else "",
                    source_bindings=sources,
                    source_bindings_digest=_agent_digest([item.model_dump(mode="json") for item in sources]),
                    predecessor_receipt_id=predecessor.receipt_id if predecessor else "",
                    reason_codes=[self._decision_reason(inspection.next_action)],
                    created_at=self.clock(),
                    executable=(
                        controller_action_boundary_class(inspection.next_action)
                        == AgentHarnessControllerActionBoundaryClass.ORDINARY_ADVANCE
                    ),
                )
                self.control_store.publish_harness_controller_decision(
                    project_id=execution.project_id, decision=decision
                )
            self.requests.write_marker(
                session, filename="decision.json", status="DECISION_COMMITTED",
                values={"decision_id": decision.decision_id, "decision_digest": decision.decision_digest}
            )
        existing_receipts = [
            item
            for item in self.control_store.list_harness_controller_action_receipts(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
            if item.decision_id == decision.decision_id
        ]
        if existing_receipts:
            if len(existing_receipts) != 1:
                raise ScientificAgentHarnessControllerVerificationError(
                    "Controller decision has conflicting receipts"
                )
            receipt = existing_receipts[0]
            self.requests.write_marker(
                session,
                filename="side_effect_observed.json",
                status="SIDE_EFFECT_OBSERVED",
                values={"receipt": receipt.model_dump(mode="json")},
            )
            self.requests.write_marker(
                session,
                filename="receipt.json",
                status="RECEIPT_COMMITTED",
                values={
                    "receipt_id": receipt.receipt_id,
                    "receipt_digest": receipt.receipt_digest,
                },
            )
            return ControllerAdvanceResult(
                execution,
                self._inspect(execution),
                decision,
                receipt,
            )
        effect_marker = self.requests.read_marker(
            session.request_dir / "side_effect_observed.json"
        )
        if effect_marker is not None:
            receipt_payload = effect_marker.get("receipt")
            if not isinstance(receipt_payload, dict):
                raise ScientificAgentHarnessControllerConflict(
                    "Controller effect checkpoint lacks its exact receipt"
                )
            receipt = AgentHarnessControllerActionReceipt.model_validate(receipt_payload)
            if (
                receipt.controller_execution_id != execution.controller_execution_id
                or receipt.decision_id != decision.decision_id
                or receipt.decision_digest != decision.decision_digest
            ):
                raise ScientificAgentHarnessControllerConflict(
                    "Controller effect checkpoint authority mismatch"
                )
        else:
            execution = self.verify_execution(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
            action_attributes: dict[str, str | int | bool] = {
                **_controller_telemetry_attributes(
                    execution,
                    operation="agent.controller.advance",
                    component="controller",
                    phase="execute_decision",
                    slot=(
                        execution.task_slots[decision.task_index]
                        if decision.task_index is not None
                        else None
                    ),
                ),
                "action_id": decision.decision_id,
                "decision_id": decision.decision_id,
                "action": decision.action_kind.value,
            }
            if decision.task_id:
                action_attributes["task_id"] = decision.task_id
                action_attributes["task_index"] = decision.task_index or 0
                action_attributes["attempt"] = decision.attempt_ordinal
                action_attributes["slot_id"] = decision.slot_id
            with self.tracer.start_span(
                "controller.action",
                attributes=action_attributes,
            ) as span:
                span.add_event(
                    "controller.decision",
                    {
                        "decision_id": decision.decision_id,
                        "action": decision.action_kind.value,
                    },
                )
                try:
                    current_inspection = self._inspect(execution)
                    receipt = self._execute_decision(
                        execution,
                        decision,
                        reconcile_only=not self._decision_is_fresh(
                            decision,
                            current_inspection,
                        ),
                    )
                except Exception:
                    span.record_error("CONTROLLER_ACTION_FAILED")
                    raise
                span.set_attribute("outcome", receipt.outcome.value)
                span.set_attribute("receipt_digest", receipt.receipt_digest)
                span.add_event(
                    "controller.receipt",
                    {
                        "outcome": receipt.outcome.value,
                        "receipt_digest": receipt.receipt_digest,
                    },
                )
            self.requests.write_marker(
                session,
                filename="side_effect_observed.json",
                status="SIDE_EFFECT_OBSERVED",
                values={"receipt": receipt.model_dump(mode="json")},
            )
        self.control_store.publish_harness_controller_action_receipt(
            project_id=execution.project_id, receipt=receipt
        )
        self.requests.write_marker(
            session, filename="receipt.json", status="RECEIPT_COMMITTED",
            values={"receipt_id": receipt.receipt_id, "receipt_digest": receipt.receipt_digest}
        )
        return ControllerAdvanceResult(execution, self._inspect(execution), decision, receipt)

    def _explicit_remote_action_in_session(
        self,
        *,
        execution: AgentHarnessControllerExecution,
        session: ControllerRequestSession,
        action: AgentHarnessControllerAction,
        require_recovery: bool,
    ) -> ControllerAdvanceResult:
        receipt_marker = self.requests.read_marker(session.request_dir / "receipt.json")
        if receipt_marker is not None:
            receipt = self.control_store.read_harness_controller_action_receipt(
                project_id=execution.project_id,
                receipt_id=str(receipt_marker.get("receipt_id") or ""),
            )
            return ControllerAdvanceResult(
                execution=execution,
                inspection=self._inspect(execution),
                receipt=receipt,
            )
        decision_marker = self.requests.read_marker(session.request_dir / "decision.json")
        if decision_marker is not None:
            decision = self.control_store.read_harness_controller_decision(
                project_id=execution.project_id,
                decision_id=str(decision_marker.get("decision_id") or ""),
            )
            if decision.action_kind != action:
                raise ScientificAgentHarnessControllerConflict(
                    "Controller control request is bound to another action"
                )
        else:
            pending = self._unreceipted_decisions(execution)
            if len(pending) > 1:
                raise ScientificAgentHarnessControllerVerificationError(
                    "multiple Controller decisions lack receipts"
                )
            if pending:
                decision = pending[0]
                if decision.action_kind != action:
                    raise ScientificAgentHarnessControllerConflict(
                        "another Controller action awaits reconciliation"
                    )
            else:
                inspection = self._inspect(execution)
                if require_recovery and inspection.status != AgentHarnessControllerStatus.RECOVERY_REQUIRED:
                    raise ScientificAgentHarnessControllerConflict("Controller is not in recovery")
                if (
                    not require_recovery
                    and inspection.status
                    in {
                        AgentHarnessControllerStatus.SUCCEEDED,
                        AgentHarnessControllerStatus.FAILED,
                        AgentHarnessControllerStatus.CANCELLED,
                    }
                ):
                    raise ScientificAgentHarnessControllerConflict(
                        "terminal Controller execution cannot be cancelled"
                    )
                slot = self._current_slot(execution, inspection)
                if slot.execution_route != "remote_execution_service":
                    if require_recovery:
                        raise ScientificAgentHarnessControllerRecoveryRequired(
                            "local unknown outcomes cannot be rerun automatically"
                        )
                    raise ScientificAgentHarnessControllerConflict(
                        "only the current exact remote slot can be cancelled"
                    )
                predecessor = self._latest_receipt(execution)
                sources = self._bindings_from_facts(inspection.facts)
                decision = AgentHarnessControllerDecision(
                    controller_execution_id=execution.controller_execution_id,
                    controller_execution_digest=execution.execution_digest,
                    client_request_id=session.client_request_id,
                    inspection_digest=inspection.inspection_digest,
                    action_kind=action,
                    task_id=slot.task_id,
                    task_index=slot.planned_task_index,
                    attempt_ordinal=slot.attempt,
                    slot_id=slot.slot_id,
                    source_bindings=sources,
                    source_bindings_digest=_agent_digest(
                        [item.model_dump(mode="json") for item in sources]
                    ),
                    predecessor_receipt_id=predecessor.receipt_id if predecessor else "",
                    reason_codes=[self._decision_reason(action)],
                    created_at=self.clock(),
                    executable=True,
                )
                self.control_store.publish_harness_controller_decision(
                    project_id=execution.project_id,
                    decision=decision,
                )
            self.requests.write_marker(
                session,
                filename="decision.json",
                status="DECISION_COMMITTED",
                values={
                    "decision_id": decision.decision_id,
                    "decision_digest": decision.decision_digest,
                },
            )
        existing_receipts = [
            item
            for item in self.control_store.list_harness_controller_action_receipts(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
            if item.decision_id == decision.decision_id
        ]
        if existing_receipts:
            if len(existing_receipts) != 1:
                raise ScientificAgentHarnessControllerVerificationError(
                    "Controller control decision has conflicting receipts"
                )
            receipt = existing_receipts[0]
            self.requests.write_marker(
                session,
                filename="side_effect_observed.json",
                status="SIDE_EFFECT_OBSERVED",
                values={"receipt": receipt.model_dump(mode="json")},
            )
            self.requests.write_marker(
                session,
                filename="receipt.json",
                status="RECEIPT_COMMITTED",
                values={
                    "receipt_id": receipt.receipt_id,
                    "receipt_digest": receipt.receipt_digest,
                },
            )
            return ControllerAdvanceResult(
                execution=execution,
                inspection=self._inspect(execution),
                decision=decision,
                receipt=receipt,
            )
        effect_marker = self.requests.read_marker(
            session.request_dir / "side_effect_observed.json"
        )
        if effect_marker is not None:
            receipt_payload = effect_marker.get("receipt")
            if not isinstance(receipt_payload, dict):
                raise ScientificAgentHarnessControllerConflict(
                    "Controller effect checkpoint lacks its exact receipt"
                )
            receipt = AgentHarnessControllerActionReceipt.model_validate(receipt_payload)
        else:
            execution = self.verify_execution(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
            with self.tracer.start_span(
                "controller.action",
                attributes={
                    **_controller_telemetry_attributes(
                        execution,
                        operation="agent.controller.advance",
                        component="controller",
                        phase="recover_decision",
                        slot=execution.task_slots[decision.task_index],
                    ),
                    "action_id": decision.decision_id,
                    "decision_id": decision.decision_id,
                    "action": decision.action_kind.value,
                    "task_id": decision.task_id,
                    "task_index": decision.task_index or 0,
                    "attempt": decision.attempt_ordinal,
                    "slot_id": decision.slot_id,
                },
            ) as span:
                span.add_event(
                    "controller.decision",
                    {
                        "decision_id": decision.decision_id,
                        "action": decision.action_kind.value,
                    },
                )
                try:
                    current_inspection = self._inspect(execution)
                    receipt = self._execute_decision(
                        execution,
                        decision,
                        reconcile_only=not self._decision_is_fresh(
                            decision,
                            current_inspection,
                        ),
                    )
                except Exception:
                    span.record_error("CONTROLLER_ACTION_FAILED")
                    raise
                span.set_attribute("outcome", receipt.outcome.value)
                span.set_attribute("receipt_digest", receipt.receipt_digest)
                span.add_event(
                    "controller.receipt",
                    {
                        "outcome": receipt.outcome.value,
                        "receipt_digest": receipt.receipt_digest,
                    },
                )
            self.requests.write_marker(
                session,
                filename="side_effect_observed.json",
                status="SIDE_EFFECT_OBSERVED",
                values={"receipt": receipt.model_dump(mode="json")},
            )
        if (
            receipt.controller_execution_id != execution.controller_execution_id
            or receipt.decision_id != decision.decision_id
            or receipt.decision_digest != decision.decision_digest
        ):
            raise ScientificAgentHarnessControllerConflict(
                "Controller control receipt authority mismatch"
            )
        self.control_store.publish_harness_controller_action_receipt(
            project_id=execution.project_id,
            receipt=receipt,
        )
        self.requests.write_marker(
            session,
            filename="receipt.json",
            status="RECEIPT_COMMITTED",
            values={
                "receipt_id": receipt.receipt_id,
                "receipt_digest": receipt.receipt_digest,
            },
        )
        return ControllerAdvanceResult(
            execution=execution,
            inspection=self._inspect(execution),
            decision=decision,
            receipt=receipt,
        )

    def _execute_decision(
        self,
        execution: Any,
        decision: AgentHarnessControllerDecision,
        *,
        reconcile_only: bool = False,
    ) -> AgentHarnessControllerActionReceipt:
        before_stage = self.storage.read_stage_state(execution.project_id, execution.run_id)
        before_registry = self.storage.read_artifact_registry(execution.project_id, execution.run_id)
        slot = execution.task_slots[decision.task_index] if decision.task_index is not None else None
        before_remote = (
            self._remote_inspection_or_none(execution, slot)
            if slot is not None
            and slot.execution_route == "remote_execution_service"
            else None
        )
        action = decision.action_kind
        result: dict[str, Any] = {}
        outcome = AgentHarnessControllerReceiptOutcome.COMMITTED
        execution_started = False
        dispatch_occurred = False
        local_dispatch_receipts: list[AgentHarnessLocalDispatchReceipt] = []
        local_publication: AgentHarnessLocalExecutionPublication | None = None
        reason = self._receipt_reason(action)
        if not decision.executable:
            # Ordinary advance may observe a wait or recovery boundary, but it
            # must never cross an authority boundary explicitly marked
            # non-executable.  The immutable WAITING receipt keeps the
            # Controller chain linear while /recover remains the only route
            # that may invoke lifecycle recovery.
            if (
                action
                == AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL
                and slot is not None
            ):
                with self.tracer.start_span(
                    "remote.await_approval",
                    attributes=_controller_telemetry_attributes(
                        execution,
                        operation="agent.execution.remote.await_approval",
                        component="remote_execution",
                        phase="waiting",
                        slot=slot,
                    ),
                ):
                    outcome = AgentHarnessControllerReceiptOutcome.WAITING
            else:
                outcome = AgentHarnessControllerReceiptOutcome.WAITING
        elif action == AgentHarnessControllerAction.PREPARE_LOCAL_GATE:
            assert slot is not None
            current = self.storage.read_stage_state(execution.project_id, execution.run_id)
            if current is not None and current.stage == slot.task_id and current.status == RunStatus.WAITING_USER:
                outcome = AgentHarnessControllerReceiptOutcome.RECONCILED
            elif reconcile_only:
                raise ScientificAgentHarnessControllerConflict(
                    "stale local Gate decision has no exact committed effect"
                )
            else:
                result = self._prepare_local_gate(execution, slot)
        elif action == AgentHarnessControllerAction.EXECUTE_LOCAL_TASK:
            assert slot is not None
            publications = self._local_publications_for_decision(execution, decision)
            local_dispatch_receipts = self._local_dispatch_receipts_for_decision(
                execution,
                decision,
            )
            if publications:
                if len(publications) != 1 or len(local_dispatch_receipts) != 1:
                    raise ScientificAgentHarnessControllerVerificationError(
                        "local completion authority is conflicting"
                    )
                local_publication = publications[0]
                self._verify_local_execution_publication(
                    execution=execution,
                    slot=slot,
                    publication=local_publication,
                )
                outcome = AgentHarnessControllerReceiptOutcome.RECONCILED
                execution_started = True
                dispatch_occurred = True
                reason = "TASK_COMPLETED"
            elif local_dispatch_receipts:
                if len(local_dispatch_receipts) != 1:
                    raise ScientificAgentHarnessControllerVerificationError(
                        "local completion dispatch authority is conflicting"
                    )
                local_publication = self._publish_local_execution_publication(
                    execution=execution,
                    slot=slot,
                    decision=decision,
                    verification_mode="recovered_controller_dispatch",
                )
                self._verify_local_execution_publication(
                    execution=execution,
                    slot=slot,
                    publication=local_publication,
                )
                outcome = AgentHarnessControllerReceiptOutcome.RECONCILED
                execution_started = True
                dispatch_occurred = True
                reason = "TASK_COMPLETED"
            elif reconcile_only:
                raise ScientificAgentHarnessControllerConflict(
                    "stale local execution decision has no verified effect"
                )
            else:
                current = self.storage.read_stage_state(execution.project_id, execution.run_id)
                if current is not None and current.stage == slot.task_id and current.status == RunStatus.RUNNING:
                    raise ScientificAgentHarnessControllerRecoveryRequired("local task outcome is unknown")
                with self.tracer.start_span(
                    "executor.local_task",
                    attributes={
                        **_controller_telemetry_attributes(
                            execution,
                            operation="agent.execution.local",
                            component="executor",
                            phase="execute",
                            slot=slot,
                        ),
                        "attempt": slot.attempt,
                    },
                ):
                    result = self._execute_local(execution, slot, decision)
                local_dispatch_receipts = self._local_dispatch_receipts_for_decision(
                    execution,
                    decision,
                )
                execution_started = bool(local_dispatch_receipts)
                dispatch_occurred = execution_started
                if result.get("status") != RunStatus.SUCCEEDED.value:
                    outcome = AgentHarnessControllerReceiptOutcome.FAILED
                    reason = "LOCAL_TASK_FAILED"
                else:
                    publications = self._local_publications_for_decision(
                        execution,
                        decision,
                    )
                    if len(local_dispatch_receipts) != 1 or len(publications) != 1:
                        raise ScientificAgentHarnessControllerVerificationError(
                            "successful local task lacks exact dispatch and output evidence"
                        )
                    local_publication = publications[0]
                    self._verify_local_execution_publication(
                        execution=execution,
                        slot=slot,
                        publication=local_publication,
                    )
                    reason = "TASK_COMPLETED"
        elif action == AgentHarnessControllerAction.ADOPT_COMPLETED_TASK:
            assert slot is not None
            local_publication = self._publish_local_execution_publication(
                execution=execution,
                slot=slot,
                decision=decision,
                verification_mode="adopt_completed_task",
            )
            self._verify_local_execution_publication(
                execution=execution,
                slot=slot,
                publication=local_publication,
            )
            outcome = AgentHarnessControllerReceiptOutcome.RECONCILED
            reason = "TASK_ADOPTED"
        elif action == AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST:
            assert slot is not None
            remote = self._remote_inspection_or_none(execution, slot)
            if remote is not None:
                result = remote
                outcome = AgentHarnessControllerReceiptOutcome.RECONCILED
            elif reconcile_only:
                raise ScientificAgentHarnessControllerConflict(
                    "stale remote preparation decision has no exact request"
                )
            else:
                with self.tracer.start_span(
                    "remote.prepare",
                    attributes={
                        **_controller_telemetry_attributes(
                            execution,
                            operation="agent.execution.remote.prepare",
                            component="remote_execution",
                            phase="prepare",
                            slot=slot,
                        ),
                        "attempt": slot.attempt,
                    },
                ):
                    result = self._prepare_remote(execution, slot)
        elif action == AgentHarnessControllerAction.DISPATCH_REMOTE_TASK:
            assert slot is not None
            binding = self._remote_slot_binding(execution, slot)
            if reconcile_only:
                result = self._remote_inspection(execution, slot)
                if str(result["effective_status"]) in {
                    "WAITING_APPROVAL",
                    "APPROVED",
                }:
                    raise ScientificAgentHarnessControllerConflict(
                        "stale remote dispatch decision has no dispatch evidence"
                    )
                outcome = AgentHarnessControllerReceiptOutcome.RECONCILED
                execution_started = True
                dispatch_occurred = True
            else:
                execution_started = True
                dispatch_occurred = True
                with self.tracer.start_span(
                    "remote.dispatch",
                    attributes={
                        **_controller_telemetry_attributes(
                            execution,
                            operation="agent.execution.remote.dispatch",
                            component="remote_execution",
                            phase="dispatch",
                            slot=slot,
                        ),
                        "attempt": slot.attempt,
                    },
                ):
                    result = self.remote_executions.dispatch(
                        project_id=execution.project_id, run_id=execution.run_id,
                        request_sha256=binding.request_sha256, slot_id=slot.slot_id,
                        expected_slot_binding_digest=binding.slot_binding_digest,
                    )
        elif action == AgentHarnessControllerAction.REFRESH_REMOTE_TASK:
            assert slot is not None
            binding = self._remote_slot_binding(execution, slot)
            if reconcile_only:
                result = self._remote_inspection(execution, slot)
                outcome = AgentHarnessControllerReceiptOutcome.RECONCILED
            else:
                with self.tracer.start_span(
                    "remote.refresh",
                    attributes={
                        **_controller_telemetry_attributes(
                            execution,
                            operation="agent.execution.remote.refresh",
                            component="remote_execution",
                            phase="refresh",
                            slot=slot,
                        ),
                        "attempt": slot.attempt,
                    },
                ):
                    result = self.remote_executions.refresh(
                        project_id=execution.project_id, run_id=execution.run_id,
                        slot_id=slot.slot_id, expected_slot_binding_digest=binding.slot_binding_digest,
                    )
        elif action == AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS:
            assert slot is not None
            with self.tracer.start_span(
                "remote.adopt",
                attributes={
                    **_controller_telemetry_attributes(
                        execution,
                        operation="agent.execution.remote.adopt",
                        component="remote_execution",
                        phase="adopt",
                        slot=slot,
                    ),
                    "attempt": slot.attempt,
                },
            ):
                result = self._remote_inspection(execution, slot)
            if str(result["state"]["status"]) != "SUCCEEDED" or result.get("publication") is None:
                raise ScientificAgentHarnessControllerVerificationError("remote success publication is unavailable")
            reason = "TASK_COMPLETED"
        elif action == AgentHarnessControllerAction.RECOVER_REMOTE_TASK:
            assert slot is not None
            binding = self._remote_slot_binding(execution, slot)
            if reconcile_only:
                result = self._remote_inspection(execution, slot)
                if str(result["effective_status"]) == "RECOVERY_REQUIRED":
                    raise ScientificAgentHarnessControllerConflict(
                        "stale recovery decision has no completed recovery effect"
                    )
                outcome = AgentHarnessControllerReceiptOutcome.RECONCILED
            else:
                with self.tracer.start_span(
                    "remote.recover",
                    attributes={
                        **_controller_telemetry_attributes(
                            execution,
                            operation="agent.execution.remote.recover",
                            component="remote_execution",
                            phase="recover",
                            slot=slot,
                        ),
                        "attempt": slot.attempt,
                    },
                ):
                    result = self.remote_executions.recover(
                        project_id=execution.project_id,
                        run_id=execution.run_id,
                        slot_id=slot.slot_id,
                        expected_slot_binding_digest=binding.slot_binding_digest,
                    )
            reason = "REMOTE_RECOVERY_ATTEMPTED"
        elif action == AgentHarnessControllerAction.CANCEL_EXECUTION:
            assert slot is not None
            binding = self._remote_slot_binding(execution, slot)
            if reconcile_only:
                result = self._remote_inspection(execution, slot)
                if str(result["effective_status"]) not in {
                    "CANCEL_REQUESTED",
                    "CANCELLED",
                    "FAILED",
                    "SUCCEEDED",
                }:
                    raise ScientificAgentHarnessControllerConflict(
                        "stale cancellation decision has no cancellation evidence"
                    )
                outcome = AgentHarnessControllerReceiptOutcome.RECONCILED
            else:
                result = self.remote_executions.cancel(
                    project_id=execution.project_id,
                    run_id=execution.run_id,
                    request_sha256=binding.request_sha256,
                    slot_id=slot.slot_id,
                    expected_slot_binding_digest=binding.slot_binding_digest,
                )
            reason = "REMOTE_EXECUTION_CANCELLED"
        elif action in {AgentHarnessControllerAction.STOP_TASK_TERMINAL, AgentHarnessControllerAction.STOP_GATE_REJECTED, AgentHarnessControllerAction.STOP_REMOTE_REJECTED}:
            outcome = AgentHarnessControllerReceiptOutcome.FAILED
            reason = "TERMINAL_OBSERVED"
        after_stage = self.storage.read_stage_state(execution.project_id, execution.run_id)
        after_registry = self.storage.read_artifact_registry(execution.project_id, execution.run_id)
        after_remote = (
            self._remote_inspection_or_none(execution, slot)
            if slot is not None
            and slot.execution_route == "remote_execution_service"
            else None
        )
        after_inspection = self._inspect(execution, verify_authority=False)
        sources = self._bindings_from_facts(after_inspection.facts)
        remote_request = (
            result.get("request") if isinstance(result, dict) else None
        ) or (after_remote or {}).get("request")
        remote_approval = (
            result.get("approval") if isinstance(result, dict) else None
        ) or (after_remote or {}).get("approval")
        remote_publication = (
            result.get("publication") if isinstance(result, dict) else None
        ) or (after_remote or {}).get("publication")
        gate_snapshot = self._stage_snapshot(after_stage, slot.task_id if slot else "") or self._stage_snapshot(
            before_stage, slot.task_id if slot else ""
        )
        return AgentHarnessControllerActionReceipt(
            controller_execution_id=execution.controller_execution_id,
            controller_execution_digest=execution.execution_digest,
            decision_id=decision.decision_id,
            decision_digest=decision.decision_digest,
            action_kind=action,
            task_id=slot.task_id if slot else "",
            task_index=slot.planned_task_index if slot else None,
            attempt_ordinal=slot.attempt if slot else 0,
            slot_id=slot.slot_id if slot else "",
            execution_started=execution_started,
            dispatch_occurred=dispatch_occurred,
            before_stage_digest=self._stage_digest(before_stage),
            after_stage_digest=self._stage_digest(after_stage),
            before_artifact_registry_digest=_agent_digest(before_registry),
            after_artifact_registry_digest=_agent_digest(after_registry),
            local_dispatch_receipt_ids=[
                item.executor_dispatch_receipt_id or item.dispatch_receipt_id
                for item in local_dispatch_receipts
            ],
            verified_output_bindings=(
                local_publication.verified_outputs if local_publication else []
            ),
            verified_output_bindings_digest=(
                local_publication.verified_outputs_digest if local_publication else ""
            ),
            local_execution_publication_id=(
                local_publication.publication_id if local_publication else ""
            ),
            local_execution_publication_digest=(
                local_publication.publication_digest if local_publication else ""
            ),
            remote_execution_slot_id=slot.slot_id if slot and slot.execution_route == "remote_execution_service" else "",
            remote_request_id=str((remote_request or {}).get("request_id") or ""),
            remote_request_sha256=str((remote_request or {}).get("request_sha256") or ""),
            remote_approval_digest=str((remote_approval or {}).get("approval_sha256") or ""),
            remote_publication_digest=str((remote_publication or {}).get("publication_sha256") or ""),
            before_remote_stage_digest=str(
                (before_remote or {}).get("slot_stage_digest") or ""
            ),
            after_remote_stage_digest=str(
                (after_remote or {}).get("slot_stage_digest") or ""
            ),
            before_remote_state_digest=str(
                (before_remote or {}).get("transport_state_digest") or ""
            ),
            after_remote_state_digest=str(
                (after_remote or {}).get("transport_state_digest") or ""
            ),
            remote_status_source_roster_digest=str(
                (after_remote or {}).get("status_source_roster_digest") or ""
            ),
            gate_snapshot_id=str(gate_snapshot.get("snapshot_id") or ""),
            gate_snapshot_hash=str(gate_snapshot.get("snapshot_digest") or ""),
            gate_decision_digest=self._gate_decision_roster_digest(execution, slot, gate_snapshot),
            outcome=outcome,
            status_after=after_inspection.status,
            source_bindings=sources,
            source_bindings_digest=_agent_digest([item.model_dump(mode="json") for item in sources]),
            reason_codes=[reason],
            created_at=self.clock(),
        )

    def _verify_local_task_authority(
        self,
        execution: AgentHarnessControllerExecution,
        slot: AgentHarnessControllerTaskSlot,
    ) -> None:
        if (
            slot.execution_route != "local_executor"
            or not slot.local_adapter_execution_binding_digest
            or slot.remote_authority_id
            or slot.remote_authority_digest
        ):
            raise ScientificAgentHarnessControllerVerificationError(
                "local task slot authority is invalid"
            )
        publication = self.proposal_store.read(
            project_id=execution.project_id,
            proposal_id=execution.proposal_id,
            verify_current=False,
        )
        try:
            material = derive_local_task_authority_material(
                publication=publication,
                task_id=slot.task_id,
                registry=self.executor.registry,
                policy_version=execution.permission_policy_version,
            )
        except ValueError as exc:
            raise ScientificAgentHarnessControllerVerificationError(
                "local task authority is unavailable"
            ) from exc
        if (
            material.local_adapter_execution_binding_digest
            != slot.local_adapter_execution_binding_digest
            or material.execution_binding_digest
            != slot.local_adapter_execution_binding_digest
            or material.task_authority_digest != slot.task_authority_digest
        ):
            raise ScientificAgentHarnessControllerVerificationError(
                "local task authority changed after Controller creation"
            )

    def _prepare_local_gate(self, execution: Any, slot: Any) -> dict[str, Any]:
        self._verify_local_task_authority(execution, slot)
        authorization = self._authorization(execution, verify_current=False)
        binding = self.executor.derive_one_task_server_binding(
            project_id=execution.project_id, run_plan=authorization.run_plan,
            task_index=slot.planned_task_index,
            task_options=authorization.compiled_task_options[slot.task_id],
        )
        return self.executor.prepare_one_task_gate(
            project_id=execution.project_id, run_plan=authorization.run_plan,
            task_index=slot.planned_task_index, task_id=slot.task_id,
            task_options=authorization.compiled_task_options[slot.task_id],
            expected_local_adapter_execution_binding_digest=slot.local_adapter_execution_binding_digest,
            expected_compiled_options_digest=slot.compiled_options_digest,
            expected_input_artifacts_digest=binding["input_artifacts_digest"],
            expected_output_contract_digest=slot.output_contract_digest,
        )

    def _execute_local(
        self,
        execution: Any,
        slot: Any,
        decision: AgentHarnessControllerDecision,
    ) -> dict[str, Any]:
        self._verify_local_task_authority(execution, slot)
        authorization = self._authorization(execution, verify_current=False)
        options = authorization.compiled_task_options[slot.task_id]
        binding = self.executor.derive_one_task_server_binding(
            project_id=execution.project_id, run_plan=authorization.run_plan,
            task_index=slot.planned_task_index, task_options=options,
        )
        stage = self.storage.read_stage_state(execution.project_id, execution.run_id)
        snapshot = self._stage_snapshot(stage, slot.task_id)
        run_dir = self.storage.run_dir(execution.project_id, execution.run_id)
        before_dispatch_roster = self._executor_dispatch_roster(run_dir)
        approved_gates = (
            set(self.executor.registry.get(slot.task_id).gates) if snapshot else set()
        )
        common = dict(
            project_id=execution.project_id, run_plan=authorization.run_plan,
            task_index=slot.planned_task_index, task_id=slot.task_id,
            task_options=options,
            expected_local_adapter_execution_binding_digest=slot.local_adapter_execution_binding_digest,
            expected_compiled_options_digest=slot.compiled_options_digest,
            expected_input_artifacts_digest=binding["input_artifacts_digest"],
            expected_output_contract_digest=slot.output_contract_digest,
            actual_dispatch_recorder=lambda adapter_id: self._publish_local_dispatch_receipt(
                execution=execution,
                slot=slot,
                decision=decision,
                adapter_id=adapter_id,
                binding=binding,
                before_dispatch_roster=before_dispatch_roster,
                approved_gates=approved_gates,
            ),
            task_completion_recorder=lambda: self._publish_local_execution_publication(
                execution=execution,
                slot=slot,
                decision=decision,
                verification_mode="controller_dispatch",
            ),
        )
        if snapshot:
            return self.executor.execute_one_task_after_committed_gate(
                **common, actor=execution.actor,
                expected_snapshot_id=snapshot["snapshot_id"],
                expected_snapshot_digest=snapshot["snapshot_digest"],
            )
        return self.executor.execute_one_task(**common)

    def _publish_local_dispatch_receipt(
        self,
        *,
        execution: AgentHarnessControllerExecution,
        slot: AgentHarnessControllerTaskSlot,
        decision: AgentHarnessControllerDecision,
        adapter_id: str,
        binding: Mapping[str, Any],
        before_dispatch_roster: list[dict[str, str]],
        approved_gates: set[str],
    ) -> AgentHarnessLocalDispatchReceipt:
        existing = self._local_dispatch_receipts_for_decision(execution, decision)
        if existing:
            if len(existing) != 1 or existing[0].adapter_id != adapter_id:
                raise ScientificAgentHarnessControllerVerificationError(
                    "local dispatch authority is conflicting"
                )
            self._verify_executor_dispatch_binding(execution, existing[0])
            return existing[0]
        run_dir = self.storage.run_dir(execution.project_id, execution.run_id)
        after_dispatch_roster = self._executor_dispatch_roster(run_dir)
        before_ids = {item["receipt_id"] for item in before_dispatch_roster}
        new_ids = [
            item["receipt_id"]
            for item in after_dispatch_roster
            if item["receipt_id"] not in before_ids
        ]
        source = None
        authority = None
        if self.executor._source_evidence_enabled(execution.run_id):
            if len(new_ids) != 1:
                raise ScientificAgentHarnessControllerVerificationError(
                    "local adapter boundary lacks one new Executor dispatch receipt"
                )
            source_receipts = read_dispatch_receipts(run_dir=run_dir)
            matching = [
                item
                for item in source_receipts
                if item.payload["receipt_id"] == new_ids[0]
            ]
            if len(matching) != 1 or matching[0].authority_payload is None:
                raise ScientificAgentHarnessControllerVerificationError(
                    "Executor dispatch receipt authority is unavailable"
                )
            source = matching[0]
            authority = source.authority_payload
            expected_boundary_digest = self.executor._dispatch_source_digest(
                run_id=execution.run_id,
                task_id=slot.task_id,
                adapter_name=adapter_id,
                approved_gates=approved_gates,
            )
            if (
                source.payload["child_run_id"] != execution.run_id
                or source.payload["task_id"] != slot.task_id
                or source.payload["execution_started"] is not True
                or source.payload["dispatch_kind"] not in {"initial", "retry"}
                or authority["boundary_material_sha256"]
                != expected_boundary_digest
                or source.authority_sha256 is None
            ):
                raise ScientificAgentHarnessControllerVerificationError(
                    "Executor dispatch receipt does not bind the actual adapter boundary"
                )
        elif new_ids:
            raise ScientificAgentHarnessControllerVerificationError(
                "unexpected Executor dispatch authority appeared at adapter boundary"
            )
        receipt = AgentHarnessLocalDispatchReceipt(
            controller_execution_id=execution.controller_execution_id,
            controller_execution_digest=execution.execution_digest,
            decision_id=decision.decision_id,
            decision_digest=decision.decision_digest,
            task_id=slot.task_id,
            task_index=slot.planned_task_index,
            attempt_ordinal=slot.attempt,
            slot_id=slot.slot_id,
            adapter_id=adapter_id,
            executor_dispatch_receipt_id=(
                str(source.payload["receipt_id"]) if source is not None else ""
            ),
            executor_dispatch_authority_id=(
                str(authority["authority_id"]) if authority is not None else ""
            ),
            executor_dispatch_authority_digest=(
                str(source.authority_sha256) if source is not None else ""
            ),
            executor_dispatch_attempt_id=(
                str(source.payload["attempt_id"]) if source is not None else ""
            ),
            executor_dispatch_ordinal=(
                int(source.payload["dispatch_ordinal"])
                if source is not None
                else 0
            ),
            before_dispatch_roster_digest=_agent_digest(before_dispatch_roster),
            after_dispatch_roster_digest=_agent_digest(after_dispatch_roster),
            local_adapter_execution_binding_digest=(
                slot.local_adapter_execution_binding_digest
            ),
            compiled_options_digest=str(binding["compiled_options_digest"]),
            input_artifacts_digest=str(binding["input_artifacts_digest"]),
            output_contract_digest=str(binding["output_contract_digest"]),
            created_at=self.clock(),
        )
        return self.control_store.publish_harness_local_dispatch_receipt(
            project_id=execution.project_id,
            receipt=receipt,
        )

    def _publish_local_execution_publication(
        self,
        *,
        execution: AgentHarnessControllerExecution,
        slot: AgentHarnessControllerTaskSlot,
        decision: AgentHarnessControllerDecision,
        verification_mode: str,
    ) -> AgentHarnessLocalExecutionPublication:
        self._verify_local_task_authority(execution, slot)
        existing = self._local_publications_for_decision(execution, decision)
        if existing:
            if len(existing) != 1:
                raise ScientificAgentHarnessControllerVerificationError(
                    "local execution has conflicting publications"
                )
            self._verify_local_execution_publication(
                execution=execution,
                slot=slot,
                publication=existing[0],
            )
            return existing[0]
        dispatches = self._local_dispatch_receipts_for_decision(execution, decision)
        if verification_mode in {
            "controller_dispatch",
            "recovered_controller_dispatch",
        }:
            if len(dispatches) != 1:
                raise ScientificAgentHarnessControllerVerificationError(
                    "local completion lacks one exact dispatch receipt"
                )
            dispatch = dispatches[0]
        elif verification_mode == "adopt_completed_task":
            if dispatches:
                raise ScientificAgentHarnessControllerVerificationError(
                    "adopted local completion has Controller dispatch authority"
                )
            dispatch = None
        else:  # pragma: no cover - internal fixed call sites only.
            raise ScientificAgentHarnessControllerVerificationError(
                "local verification mode is unsupported"
            )
        if dispatch is not None:
            self._verify_executor_dispatch_binding(execution, dispatch)
        stage, registry, outputs = self._verified_local_outputs(
            execution=execution,
            slot=slot,
            require_controller_output_evidence=(
                verification_mode == "recovered_controller_dispatch"
            ),
        )
        if verification_mode == "recovered_controller_dispatch":
            self._verify_local_reconstruction_registry(
                execution=execution,
                slot=slot,
                registry=registry,
            )
            authorization = self._authorization(execution, verify_current=False)
            try:
                self.executor.verify_one_task_committed_outputs(
                    project_id=execution.project_id,
                    run_plan=authorization.run_plan,
                    task_index=slot.planned_task_index,
                    task_id=slot.task_id,
                    task_options=authorization.compiled_task_options[slot.task_id],
                    actor=execution.actor,
                    expected_local_adapter_execution_binding_digest=(
                        slot.local_adapter_execution_binding_digest
                    ),
                    expected_compiled_options_digest=slot.compiled_options_digest,
                    expected_input_artifacts_digest=dispatch.input_artifacts_digest,
                    expected_output_contract_digest=slot.output_contract_digest,
                )
            except ValueError as exc:
                raise ScientificAgentHarnessControllerVerificationError(
                    "local completion reconstruction failed exact task verification"
                ) from exc
            replay_stage, replay_registry, replay_outputs = self._verified_local_outputs(
                execution=execution,
                slot=slot,
                require_controller_output_evidence=True,
            )
            if (
                self._stage_digest(replay_stage) != self._stage_digest(stage)
                or replay_registry != registry
                or replay_outputs != outputs
            ):
                raise ScientificAgentHarnessControllerVerificationError(
                    "local completion changed during reconstruction"
                )
            stage, registry, outputs = replay_stage, replay_registry, replay_outputs
        publication = AgentHarnessLocalExecutionPublication(
            controller_execution_id=execution.controller_execution_id,
            controller_execution_digest=execution.execution_digest,
            decision_id=decision.decision_id,
            decision_digest=decision.decision_digest,
            task_id=slot.task_id,
            task_index=slot.planned_task_index,
            attempt_ordinal=slot.attempt,
            slot_id=slot.slot_id,
            verification_mode=verification_mode,
            local_dispatch_receipt_id=(
                dispatch.dispatch_receipt_id if dispatch is not None else ""
            ),
            local_dispatch_receipt_digest=(
                dispatch.dispatch_receipt_digest if dispatch is not None else ""
            ),
            stage_digest=self._stage_digest(stage),
            artifact_registry_digest=_agent_digest(registry),
            output_contract_digest=slot.output_contract_digest,
            verified_outputs=outputs,
            verified_outputs_digest=_agent_digest(
                [item.model_dump(mode="json") for item in outputs]
            ),
            created_at=self.clock(),
        )
        return self.control_store.publish_harness_local_execution_publication(
            project_id=execution.project_id,
            publication=publication,
        )

    def _verified_local_outputs(
        self,
        *,
        execution: AgentHarnessControllerExecution,
        slot: AgentHarnessControllerTaskSlot,
        allow_history: bool = False,
        require_controller_output_evidence: bool = False,
    ) -> tuple[Any, dict[str, str], list[AgentHarnessVerifiedOutputBinding]]:
        authorization = self._authorization(execution, verify_current=False)
        task = authorization.run_plan.tasks[slot.planned_task_index]
        stage = self.storage.read_stage_state(execution.project_id, execution.run_id)
        current_success = bool(
            stage is not None
            and stage.stage == slot.task_id
            and stage.status == RunStatus.SUCCEEDED
        )
        history_success = bool(
            allow_history
            and stage is not None
            and any(
                item.stage == slot.task_id and item.status == RunStatus.SUCCEEDED
                for item in stage.history
            )
        )
        if not current_success and not history_success:
            raise ScientificAgentHarnessControllerVerificationError(
                "local output publication lacks exact successful StageState"
            )
        registry = self.storage.read_artifact_registry(
            execution.project_id,
            execution.run_id,
        )
        verifier = self.executor.one_task_output_verifier_binding(
            run_plan=authorization.run_plan,
            task_index=slot.planned_task_index,
            expected_output_contract_digest=slot.output_contract_digest,
        )
        execution_record_id = str(verifier["execution_record_id"])
        if execution_record_id and execution_record_id not in task.output_artifacts:
            raise ScientificAgentHarnessControllerVerificationError(
                "immutable execution record is outside the output contract"
            )
        run_dir = self.storage.run_dir(execution.project_id, execution.run_id)
        outputs: list[AgentHarnessVerifiedOutputBinding] = []
        for artifact_id in sorted(task.output_artifacts):
            registered = registry.get(artifact_id)
            if not registered:
                raise ScientificAgentHarnessControllerVerificationError(
                    "local output contract is incomplete"
                )
            relative = _safe_relative_artifact_path(registered)
            path = _safe_artifact_path(
                run_dir,
                relative,
                label="Controller local output",
            )
            payload, present = _read_stable_file(
                path,
                label="Controller local output",
                max_bytes=2 * 1024 * 1024 * 1024,
            )
            if not present:
                raise ScientificAgentHarnessControllerVerificationError(
                    "local output is unavailable"
                )
            content_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
            is_execution_record = artifact_id == execution_record_id
            outputs.append(
                AgentHarnessVerifiedOutputBinding(
                    artifact_id=artifact_id,
                    relative_path=str(relative),
                    content_sha256=content_digest,
                    size_bytes=len(payload),
                    producer_task_id=slot.task_id,
                    verification_class=str(verifier["verification_class"]),
                    verifier_version=str(verifier["verifier_version"]),
                    verifier_digest=str(verifier["verifier_digest"]),
                    execution_record_id=(artifact_id if is_execution_record else ""),
                    execution_record_digest=(
                        content_digest if is_execution_record else ""
                    ),
                )
            )
        if require_controller_output_evidence:
            evidence = (
                stage.details.get("controller_output_evidence")
                if stage is not None and isinstance(stage.details, dict)
                else None
            )
            roster = [
                {
                    "artifact_id": item.artifact_id,
                    "relative_path": item.relative_path,
                    "size_bytes": item.size_bytes,
                    "content_sha256": item.content_sha256,
                    "producer_task_id": item.producer_task_id,
                }
                for item in outputs
            ]
            if (
                not isinstance(evidence, dict)
                or evidence.get("schema_version")
                != "run-plan-controller-output-evidence.v1"
                or evidence.get("task_id") != slot.task_id
                or evidence.get("output_contract_digest")
                != slot.output_contract_digest
                or evidence.get("outputs") != roster
                or evidence.get("outputs_digest") != _agent_digest(roster)
            ):
                raise ScientificAgentHarnessControllerVerificationError(
                    "local completion reconstruction output evidence mismatch"
                )
        return stage, registry, outputs

    def _verify_local_reconstruction_registry(
        self,
        *,
        execution: AgentHarnessControllerExecution,
        slot: AgentHarnessControllerTaskSlot,
        registry: Mapping[str, str],
    ) -> None:
        publication = self.proposal_store.read(
            project_id=execution.project_id,
            proposal_id=execution.proposal_id,
            verify_current=False,
        )
        original_ids = {
            item.artifact_id for item in publication.observation.available_artifacts
        }
        authorization = self._authorization(execution, verify_current=False)
        allowed_output_ids = {
            artifact_id
            for task in authorization.run_plan.tasks[
                : slot.planned_task_index + 1
            ]
            for artifact_id in task.output_artifacts
        }
        allowed_remote_publication_ids = {
            "remote_execution_publication_"
            + hashlib.sha256(item.slot_id.encode("utf-8")).hexdigest()[:16]
            for item in execution.task_slots[: slot.planned_task_index]
            if item.execution_route == "remote_execution_service"
        }
        unexpected = set(registry).difference(original_ids).difference(
            allowed_output_ids
        ).difference(allowed_remote_publication_ids)
        if unexpected:
            raise ScientificAgentHarnessControllerVerificationError(
                "local completion reconstruction found unauthorized Registry output"
            )

    @staticmethod
    def _executor_dispatch_roster(run_dir: Path) -> list[dict[str, str]]:
        roster: list[dict[str, str]] = []
        for item in read_dispatch_receipts(run_dir=run_dir, allow_missing=True):
            authority = item.authority_payload
            if authority is None or item.authority_sha256 is None:
                raise ScientificAgentHarnessControllerVerificationError(
                    "Executor dispatch authority roster is incomplete"
                )
            roster.append(
                {
                    "receipt_id": str(item.payload["receipt_id"]),
                    "receipt_digest": str(item.sha256),
                    "dispatch_authority_id": str(authority["authority_id"]),
                    "dispatch_authority_digest": str(item.authority_sha256),
                }
            )
        return roster

    def _verify_executor_dispatch_binding(
        self,
        execution: AgentHarnessControllerExecution,
        dispatch: AgentHarnessLocalDispatchReceipt,
    ) -> None:
        if dispatch.task_index >= len(execution.task_slots):
            raise ScientificAgentHarnessControllerVerificationError(
                "local dispatch task index is invalid"
            )
        slot = execution.task_slots[dispatch.task_index]
        if (
            slot.task_id != dispatch.task_id
            or slot.slot_id != dispatch.slot_id
            or slot.attempt != dispatch.attempt_ordinal
            or slot.local_adapter_execution_binding_digest
            != dispatch.local_adapter_execution_binding_digest
        ):
            raise ScientificAgentHarnessControllerVerificationError(
                "local dispatch adapter authority mismatch"
            )
        run_dir = self.storage.run_dir(execution.project_id, execution.run_id)
        roster = self._executor_dispatch_roster(run_dir)
        ordinal = dispatch.executor_dispatch_ordinal
        if not ordinal:
            if (
                _agent_digest(roster) != dispatch.after_dispatch_roster_digest
                or dispatch.before_dispatch_roster_digest
                != dispatch.after_dispatch_roster_digest
            ):
                raise ScientificAgentHarnessControllerVerificationError(
                    "unexpected Executor dispatch authority roster changed"
                )
            return
        if (
            len(roster) < ordinal
            or _agent_digest(roster[: ordinal - 1])
            != dispatch.before_dispatch_roster_digest
            or _agent_digest(roster[:ordinal])
            != dispatch.after_dispatch_roster_digest
        ):
            raise ScientificAgentHarnessControllerVerificationError(
                "Executor dispatch authority roster changed"
            )
        matches = [
            item
            for item in read_dispatch_receipts(run_dir=run_dir)
            if item.payload["receipt_id"]
            == dispatch.executor_dispatch_receipt_id
        ]
        if len(matches) != 1 or matches[0].authority_payload is None:
            raise ScientificAgentHarnessControllerVerificationError(
                "Executor dispatch receipt is unavailable"
            )
        source = matches[0]
        authority = source.authority_payload
        if (
            source.payload["task_id"] != dispatch.task_id
            or source.payload["attempt_id"]
            != dispatch.executor_dispatch_attempt_id
            or source.payload["execution_started"] is not True
            or authority["authority_id"]
            != dispatch.executor_dispatch_authority_id
            or source.authority_sha256
            != dispatch.executor_dispatch_authority_digest
        ):
            raise ScientificAgentHarnessControllerVerificationError(
                "Executor dispatch receipt authority mismatch"
            )

    def _verify_local_execution_publication(
        self,
        *,
        execution: AgentHarnessControllerExecution,
        slot: AgentHarnessControllerTaskSlot,
        publication: AgentHarnessLocalExecutionPublication,
        allow_later_state: bool = False,
    ) -> None:
        if (
            publication.controller_execution_id != execution.controller_execution_id
            or publication.controller_execution_digest != execution.execution_digest
            or publication.task_id != slot.task_id
            or publication.task_index != slot.planned_task_index
            or publication.attempt_ordinal != slot.attempt
            or publication.slot_id != slot.slot_id
            or publication.output_contract_digest != slot.output_contract_digest
        ):
            raise ScientificAgentHarnessControllerVerificationError(
                "local execution publication authority mismatch"
            )
        stage, registry, outputs = self._verified_local_outputs(
            execution=execution,
            slot=slot,
            allow_history=allow_later_state,
        )
        if (
            publication.verified_outputs != outputs
            or publication.verified_outputs_digest
            != _agent_digest([item.model_dump(mode="json") for item in outputs])
        ):
            raise ScientificAgentHarnessControllerVerificationError(
                "local execution publication no longer verifies current outputs"
            )
        if not allow_later_state and (
            publication.stage_digest != self._stage_digest(stage)
            or publication.artifact_registry_digest != _agent_digest(registry)
        ):
            raise ScientificAgentHarnessControllerVerificationError(
                "local execution publication state anchor changed"
            )
        if publication.verification_mode in {
            "controller_dispatch",
            "recovered_controller_dispatch",
        }:
            try:
                dispatch = self.control_store.read_harness_local_dispatch_receipt(
                    project_id=execution.project_id,
                    dispatch_receipt_id=publication.local_dispatch_receipt_id,
                )
            except FileNotFoundError as exc:
                raise ScientificAgentHarnessControllerVerificationError(
                    "local publication dispatch receipt is unavailable"
                ) from exc
            if (
                dispatch.dispatch_receipt_digest
                != publication.local_dispatch_receipt_digest
                or dispatch.decision_id != publication.decision_id
                or dispatch.task_id != publication.task_id
                or not dispatch.execution_started
            ):
                raise ScientificAgentHarnessControllerVerificationError(
                    "local publication dispatch authority mismatch"
                )
            self._verify_executor_dispatch_binding(execution, dispatch)

    def _local_dispatch_receipts_for_decision(
        self,
        execution: AgentHarnessControllerExecution,
        decision: AgentHarnessControllerDecision,
    ) -> list[AgentHarnessLocalDispatchReceipt]:
        return [
            item
            for item in self.control_store.list_harness_local_dispatch_receipts(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
            if item.decision_id == decision.decision_id
            and item.decision_digest == decision.decision_digest
        ]

    def _local_publications_for_decision(
        self,
        execution: AgentHarnessControllerExecution,
        decision: AgentHarnessControllerDecision,
    ) -> list[AgentHarnessLocalExecutionPublication]:
        return [
            item
            for item in self.control_store.list_harness_local_execution_publications(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
            if item.decision_id == decision.decision_id
            and item.decision_digest == decision.decision_digest
        ]

    def _prepare_remote(self, execution: Any, slot: Any) -> dict[str, Any]:
        authorization = self._authorization(execution, verify_current=False)
        publication = self.proposal_store.read(
            project_id=execution.project_id, proposal_id=execution.proposal_id, verify_current=False
        )
        current = self.resource_authority_service.current_authority(
            publication=publication, task_id=slot.task_id
        )
        authority = current.authority
        if authority.authority_id != slot.remote_authority_id or authority.authority_digest != slot.remote_authority_digest:
            raise ScientificAgentHarnessControllerVerificationError("remote task authority changed")
        connection = self.remote_executions.profiles.get_connection(authority.connection_id)
        profile = self.remote_executions.profiles.resolve_execution_profile(authority.execution_profile_id)
        task = authorization.run_plan.tasks[slot.planned_task_index]
        registry = self.storage.read_artifact_registry(execution.project_id, execution.run_id)
        run_dir = self.storage.run_dir(execution.project_id, execution.run_id)
        descriptors: list[dict[str, Any]] = []
        bindings: dict[str, str] = {}
        input_payloads: dict[str, bytes] = {}
        for index, artifact_id in enumerate(task.required_artifacts):
            registered = registry.get(artifact_id)
            if not registered:
                raise ScientificAgentHarnessControllerVerificationError("remote input is unavailable")
            relative = _safe_relative_artifact_path(registered)
            source = _safe_artifact_path(run_dir, relative, label="remote input")
            payload, exists = _read_stable_file(source, label="remote input", max_bytes=2 * 1024 * 1024 * 1024)
            if not exists:
                raise ScientificAgentHarnessControllerVerificationError("remote input is unavailable")
            suffix = source.suffix.lower() or ".json"
            destination_name = f"input-{index:04d}{suffix}"
            purpose, media_type = self._remote_input_contract(profile.task_type, suffix)
            descriptors.append(
                {
                    "relative_path": destination_name,
                    "purpose": purpose,
                    "media_type": media_type,
                    "payload": payload,
                }
            )
            bindings[destination_name] = artifact_id
            input_payloads[destination_name] = payload
        if not descriptors:
            payload = _pretty_json_bytes({
                "schema_version": "agent_harness_remote_execution_input.v1",
                "task_id": slot.task_id,
                "compiled_task_options": authorization.compiled_task_options[slot.task_id],
            })
            destination_name = "execution-request.json"
            purpose, media_type = self._remote_input_contract(profile.task_type, ".json")
            artifact_identity = _agent_digest({"slot_id": slot.slot_id, "payload": json.loads(payload)})
            artifact_id = f"harness-input-{artifact_identity.split(':', 1)[1][:32]}"
            descriptors.append(
                {
                    "relative_path": destination_name,
                    "purpose": purpose,
                    "media_type": media_type,
                    "payload": payload,
                }
            )
            bindings[destination_name] = artifact_id
            input_payloads[destination_name] = payload
        request_identity = _agent_digest({"controller_execution_id": execution.controller_execution_id, "slot_id": slot.slot_id})
        manifest = build_transfer_manifest_from_payloads(
            request_id=f"remote-{request_identity.split(':', 1)[1][:32]}",
            artifacts=descriptors,
            connection=connection,
            execution_profile=profile,
            target_purpose=profile.task_type.replace("_", "-"),
        )
        return self.remote_executions.prepare(
            project_id=execution.project_id, run_id=execution.run_id, task_id=slot.task_id,
            transfer_manifest=manifest,
            requested_resources=authority.configured_resources.model_dump(mode="json"),
            input_artifacts=bindings,
            input_payloads=input_payloads,
            slot_id=slot.slot_id,
            slot_binding_authority={
                "controller_execution_id": execution.controller_execution_id,
                "controller_execution_digest": execution.execution_digest,
                "planned_task_index": slot.planned_task_index,
                "attempt": slot.attempt,
                "task_authority_digest": slot.task_authority_digest,
                "dispatch_intent_digest": slot.dispatch_intent_digest,
                "compiled_options_digest": slot.compiled_options_digest,
                "input_artifacts_digest": slot.input_artifacts_digest,
                "output_contract_digest": slot.output_contract_digest,
                "remote_authority_id": slot.remote_authority_id,
                "remote_authority_digest": slot.remote_authority_digest,
                "remote_authority_set_id": execution.remote_authority_set_id,
                "remote_authority_set_digest": execution.remote_authority_set_digest,
            },
        )

    def _remote_inspection_or_none(self, execution: Any, slot: Any) -> dict[str, Any] | None:
        slot_path = (
            self.storage.run_dir(execution.project_id, execution.run_id)
            / "remote-executions"
            / slot.slot_id
        )
        if not slot_path.exists():
            return None
        if slot_path.is_symlink() or not slot_path.is_dir():
            raise ScientificAgentHarnessControllerVerificationError(
                "remote task slot storage is unsafe"
            )
        try:
            return self._remote_inspection(execution, slot)
        except FileNotFoundError:
            return None

    def _remote_inspection(self, execution: Any, slot: Any) -> dict[str, Any]:
        binding = self._remote_slot_binding(execution, slot)
        return self.remote_executions.inspect(
            project_id=execution.project_id, run_id=execution.run_id,
            slot_id=slot.slot_id, expected_slot_binding_digest=binding.slot_binding_digest,
        )

    def _remote_slot_binding(self, execution: Any, slot: Any):
        binding = self.remote_executions.inspect_slot_binding(
            project_id=execution.project_id, run_id=execution.run_id, slot_id=slot.slot_id
        )
        expected = {
            "controller_execution_id": execution.controller_execution_id,
            "controller_execution_digest": execution.execution_digest,
            "planned_task_index": slot.planned_task_index,
            "task_id": slot.task_id,
            "attempt": slot.attempt,
            "task_authority_digest": slot.task_authority_digest,
            "dispatch_intent_digest": slot.dispatch_intent_digest,
            "compiled_options_digest": slot.compiled_options_digest,
            "input_artifacts_digest": slot.input_artifacts_digest,
            "output_contract_digest": slot.output_contract_digest,
            "remote_authority_id": slot.remote_authority_id,
            "remote_authority_digest": slot.remote_authority_digest,
            "remote_authority_set_id": execution.remote_authority_set_id,
            "remote_authority_set_digest": execution.remote_authority_set_digest,
        }
        if any(getattr(binding, key) != value for key, value in expected.items()):
            raise ScientificAgentHarnessControllerVerificationError("remote slot authority mismatch")
        return binding

    def _authorization(
        self, execution: Any, *, verify_current: bool = True
    ) -> AgentPlanAuthorization:
        return self.authorization_service.verify_authorization(
            project_id=execution.project_id, authorization_id=execution.authorization_id,
            verify_current=verify_current,
        )

    @staticmethod
    def _inspection(execution: Any, status: Any, slot: Any, action: Any, facts: list[Any]):
        return AgentHarnessControllerInspection(
            controller_execution_id=execution.controller_execution_id,
            controller_execution_digest=execution.execution_digest,
            status=status,
            current_task_index=slot.planned_task_index if slot else None,
            current_task_id=slot.task_id if slot else "",
            current_slot_id=slot.slot_id if slot else "",
            next_action=action,
            facts=facts,
            source_roster_digest=_agent_digest([item.model_dump(mode="json") for item in facts]),
            inspected_at=now_iso(),
        )

    @staticmethod
    def _source(name: str, source_id: str, source_digest: str) -> AgentHarnessControllerSourceBinding:
        return AgentHarnessControllerSourceBinding(name=name, source_id=source_id, source_digest=source_digest)

    @staticmethod
    def _bindings_from_facts(facts: list[AgentHarnessControllerInspectionFact]) -> list[AgentHarnessControllerSourceBinding]:
        return [
            AgentHarnessControllerSourceBinding(
                name=item.name, source_id=item.source_id, source_digest=item.source_digest,
                authority_class=item.authority_class,
            )
            for item in facts if item.source_id and item.source_digest
        ]

    def _latest_receipt(self, execution: Any):
        receipts = self.control_store.list_harness_controller_action_receipts(
            project_id=execution.project_id,
            controller_execution_id=execution.controller_execution_id,
        )
        if not receipts:
            return None
        predecessor_ids = {
            self.control_store.read_harness_controller_decision(
                project_id=execution.project_id,
                decision_id=item.decision_id,
            ).predecessor_receipt_id
            for item in receipts
        }
        candidates = [item for item in receipts if item.receipt_id not in predecessor_ids]
        if len(candidates) != 1:
            raise ScientificAgentHarnessControllerVerificationError(
                "Controller receipt chain is forked or incomplete"
            )
        return candidates[0]

    def _unreceipted_decisions(
        self,
        execution: AgentHarnessControllerExecution,
    ) -> list[AgentHarnessControllerDecision]:
        decisions = self.control_store.list_harness_controller_decisions(
            project_id=execution.project_id,
            controller_execution_id=execution.controller_execution_id,
        )
        receipted = {
            item.decision_id
            for item in self.control_store.list_harness_controller_action_receipts(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
        }
        return [item for item in decisions if item.decision_id not in receipted]

    def _decision_is_fresh(
        self,
        decision: AgentHarnessControllerDecision,
        inspection: AgentHarnessControllerInspection,
    ) -> bool:
        current_sources = self._bindings_from_facts(inspection.facts)
        return bool(
            inspection.inspection_digest == decision.inspection_digest
            and current_sources == decision.source_bindings
            and _agent_digest(
                [item.model_dump(mode="json") for item in current_sources]
            )
            == decision.source_bindings_digest
        )

    def _verify_post_start_sources(
        self,
        execution: AgentHarnessControllerExecution,
        authorization: AgentPlanAuthorization,
        publication: ScientificAgentPlanPublication,
    ) -> None:
        current_observation = self.proposal_store.observation_builder.build(
            project_id=publication.observation.project_id,
            run_id=publication.observation.run_id,
            goal=publication.observation.goal_context,
            user_constraints=publication.observation.explicit_constraints,
        )
        original_sources = {
            item.source_id: item for item in publication.observation.source_bindings
        }
        current_sources = {
            item.source_id: item for item in current_observation.source_bindings
        }
        for source_id in (
            "existing_run_plan",
            "scientific_tool_catalog",
            "resource_profile_snapshot",
        ):
            original = original_sources.get(source_id)
            current = current_sources.get(source_id)
            if original is None or current is None or current.model_dump(mode="json") != original.model_dump(mode="json"):
                raise ScientificAgentHarnessControllerVerificationError(
                    "non-execution authority changed after Controller start"
                )
        original_artifacts = {
            item.artifact_id: item for item in publication.observation.available_artifacts
        }
        current_artifacts = {
            item.artifact_id: item for item in current_observation.available_artifacts
        }
        for artifact_id, original in original_artifacts.items():
            current = current_artifacts.get(artifact_id)
            if current is None or current.model_dump(mode="json") != original.model_dump(mode="json"):
                raise ScientificAgentHarnessControllerVerificationError(
                    "pre-existing artifact authority changed after Controller start"
                )
        run_dir = self.storage.run_dir(execution.project_id, execution.run_id)
        registry = self.storage.read_artifact_registry(execution.project_id, execution.run_id)
        for binding in authorization.artifact_bindings:
            relative = registry.get(binding.artifact_id)
            if not relative:
                raise ScientificAgentHarnessControllerVerificationError(
                    "authorized input artifact is no longer registered"
                )
            source = _safe_artifact_path(
                run_dir,
                _safe_relative_artifact_path(relative),
                label="authorized Controller input",
            )
            payload, present = _read_stable_file(
                source,
                label="authorized Controller input",
                max_bytes=2 * 1024 * 1024 * 1024,
            )
            digest = "sha256:" + hashlib.sha256(payload).hexdigest()
            if not present or digest != binding.content_digest:
                raise ScientificAgentHarnessControllerVerificationError(
                    "authorized input artifact content changed"
                )
        latest = self._latest_receipt(execution)
        decisions = self.control_store.list_harness_controller_decisions(
            project_id=execution.project_id,
            controller_execution_id=execution.controller_execution_id,
        )
        receipted_decisions = {
            item.decision_id
            for item in self.control_store.list_harness_controller_action_receipts(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
        }
        unreceipted = [item for item in decisions if item.decision_id not in receipted_decisions]
        stage = self.storage.read_stage_state(execution.project_id, execution.run_id)
        stage_digest = self._stage_digest(stage)
        registry_digest = _agent_digest(registry)
        if not unreceipted:
            if latest is None:
                raise ScientificAgentHarnessControllerVerificationError(
                    "unanchored execution source change"
                )
            if (
                latest.after_stage_digest != stage_digest
                or latest.after_artifact_registry_digest != registry_digest
            ):
                if self._manual_first_local_completion_is_adoptable(
                    execution=execution,
                    authorization=authorization,
                    latest=latest,
                    stage=stage,
                    registry=registry,
                    original_artifact_ids=set(original_artifacts),
                ):
                    return
                raise ScientificAgentHarnessControllerVerificationError(
                    "execution sources differ from the latest immutable receipt"
                )
            return
        if len(unreceipted) != 1:
            raise ScientificAgentHarnessControllerVerificationError(
                "multiple Controller effects lack receipts"
            )
        decision = unreceipted[0]
        if stage is not None and decision.task_id and stage.stage != decision.task_id:
            preparing_exact_successor = bool(
                decision.action_kind
                in {
                    AgentHarnessControllerAction.PREPARE_LOCAL_GATE,
                    AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
                }
                and latest is not None
                and latest.after_stage_digest == stage_digest
                and latest.after_artifact_registry_digest == registry_digest
                and stage.status == RunStatus.SUCCEEDED
                and stage.next_stage == decision.task_id
            )
            if not preparing_exact_successor:
                raise ScientificAgentHarnessControllerVerificationError(
                    "unreceipted StageState belongs to another task"
                )
        task_limit = (
            len(authorization.run_plan.tasks)
            if decision.task_index is None
            else decision.task_index + 1
        )
        allowed_new_ids = {
            artifact_id
            for task in authorization.run_plan.tasks[:task_limit]
            for artifact_id in task.output_artifacts
        }
        verified_output_paths = {
            output.relative_path
            for receipt in self.control_store.list_harness_controller_action_receipts(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
            for output in receipt.verified_output_bindings
        }
        verified_output_paths.update(
            output.relative_path
            for local_publication in self.control_store.list_harness_local_execution_publications(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
            for output in local_publication.verified_outputs
        )
        if decision.action_kind == AgentHarnessControllerAction.ADOPT_COMPLETED_TASK:
            slot = execution.task_slots[decision.task_index or 0]
            _, _, current_outputs = self._verified_local_outputs(
                execution=execution,
                slot=slot,
            )
            verified_output_paths.update(
                output.relative_path for output in current_outputs
            )
        unexpected = set(registry).difference(original_artifacts).difference(allowed_new_ids)
        if any(
            not item.startswith("harness-input-")
            and not item.startswith("remote_execution_publication_")
            and registry[item] not in verified_output_paths
            for item in unexpected
        ):
            raise ScientificAgentHarnessControllerVerificationError(
                "unreceipted Registry mutation is outside the selected task contract"
            )

    def _manual_first_local_completion_is_adoptable(
        self,
        *,
        execution: AgentHarnessControllerExecution,
        authorization: AgentPlanAuthorization,
        latest: AgentHarnessControllerActionReceipt,
        stage: Any,
        registry: dict[str, str],
        original_artifact_ids: set[str],
    ) -> bool:
        """Allow only exact first-task manual completion to reach ADOPT."""

        slot = execution.task_slots[0]
        task = authorization.run_plan.tasks[0]
        if (
            slot.execution_route != "local_executor"
            or stage is None
            or stage.stage != slot.task_id
            or stage.status != RunStatus.SUCCEEDED
            or latest.task_index not in {0, None}
            or latest.action_kind
            not in {
                AgentHarnessControllerAction.PREPARE_LOCAL_GATE,
                AgentHarnessControllerAction.WAIT_FOR_GATE,
            }
        ):
            return False
        if self.control_store.list_harness_local_dispatch_receipts(
            project_id=execution.project_id,
            controller_execution_id=execution.controller_execution_id,
        ) or self.control_store.list_harness_local_execution_publications(
            project_id=execution.project_id,
            controller_execution_id=execution.controller_execution_id,
        ):
            return False
        spec = self.executor.registry.get(slot.task_id)
        if not spec.gates or not latest.gate_snapshot_id or not latest.gate_snapshot_hash:
            return False
        snapshot = {
            "snapshot_id": latest.gate_snapshot_id,
            "snapshot_hash": latest.gate_snapshot_hash.removeprefix("sha256:"),
        }
        decisions = self._gate_decisions(execution, slot, snapshot, spec.gates)
        if set(decisions) != set(spec.gates) or any(
            not item.approved for item in decisions.values()
        ):
            return False
        _, _, outputs = self._verified_local_outputs(
            execution=execution,
            slot=slot,
        )
        verified_paths = {item.relative_path for item in outputs}
        allowed_ids = original_artifact_ids.union(task.output_artifacts)
        unexpected = {
            artifact_id
            for artifact_id, relative_path in registry.items()
            if artifact_id not in allowed_ids
            and relative_path not in verified_paths
            and not artifact_id.startswith("harness-input-")
        }
        if unexpected:
            raise ScientificAgentHarnessControllerVerificationError(
                "manual local completion added unverified Registry authority"
            )
        return True

    def _local_task_completed(
        self,
        execution: AgentHarnessControllerExecution,
        slot: Any,
        task: Any,
        stage: Any,
        registry: dict[str, str],
        receipts: list[Any],
    ) -> bool:
        if slot.execution_route != "local_executor":
            return False
        candidates = [
            item
            for item in receipts
            if item.task_id == slot.task_id
            and item.outcome
            in {
                AgentHarnessControllerReceiptOutcome.COMMITTED,
                AgentHarnessControllerReceiptOutcome.RECONCILED,
            }
            and set(item.reason_codes).intersection(
                {"TASK_COMPLETED", "TASK_ADOPTED"}
            )
        ]
        if not candidates:
            return False
        if len(candidates) != 1:
            raise ScientificAgentHarnessControllerVerificationError(
                "local task has conflicting completion receipts"
            )
        receipt = candidates[0]
        if (
            not receipt.local_execution_publication_id
            or not receipt.local_execution_publication_digest
            or not receipt.verified_output_bindings
        ):
            raise ScientificAgentHarnessControllerVerificationError(
                "local completion receipt lacks verified output authority"
            )
        publication = self.control_store.read_harness_local_execution_publication(
            project_id=execution.project_id,
            publication_id=receipt.local_execution_publication_id,
        )
        if (
            publication.publication_digest
            != receipt.local_execution_publication_digest
            or publication.verified_outputs != receipt.verified_output_bindings
            or publication.verified_outputs_digest
            != receipt.verified_output_bindings_digest
        ):
            raise ScientificAgentHarnessControllerVerificationError(
                "local completion receipt publication binding mismatch"
            )
        if publication.verification_mode in {
            "controller_dispatch",
            "recovered_controller_dispatch",
        }:
            try:
                dispatch = self.control_store.read_harness_local_dispatch_receipt(
                    project_id=execution.project_id,
                    dispatch_receipt_id=publication.local_dispatch_receipt_id,
                )
            except FileNotFoundError as exc:
                raise ScientificAgentHarnessControllerVerificationError(
                    "local publication dispatch receipt is unavailable"
                ) from exc
            if receipt.local_dispatch_receipt_ids != [
                dispatch.executor_dispatch_receipt_id
                or dispatch.dispatch_receipt_id
            ]:
                raise ScientificAgentHarnessControllerVerificationError(
                    "local completion receipt lacks exact Executor dispatch authority"
                )
        elif receipt.local_dispatch_receipt_ids:
            raise ScientificAgentHarnessControllerVerificationError(
                "adopted local completion claims an Executor dispatch"
            )
        self._verify_local_execution_publication(
            execution=execution,
            slot=slot,
            publication=publication,
            allow_later_state=True,
        )
        return True

    def _remote_task_completed(
        self,
        execution: Any,
        remote: dict[str, Any],
        receipts: list[Any],
        slot: Any,
        registry: dict[str, str],
    ) -> bool:
        task = self._authorization(
            execution, verify_current=False
        ).run_plan.tasks[slot.planned_task_index]
        publication = remote.get("publication")
        publication_ids = {
            str(item.get("artifact_id") or "")
            for item in (publication or {}).get("artifacts", [])
        }
        outputs_ok = all(
            item in registry and item in publication_ids for item in task.output_artifacts
        )
        matching = [
            item
            for item in receipts
            if item.task_id == slot.task_id
            and "TASK_COMPLETED" in item.reason_codes
            and item.outcome
            in {
                AgentHarnessControllerReceiptOutcome.COMMITTED,
                AgentHarnessControllerReceiptOutcome.RECONCILED,
            }
        ]
        if not matching:
            return False
        if len(matching) != 1:
            raise ScientificAgentHarnessControllerVerificationError(
                "remote task has conflicting completion receipts"
            )
        receipt = matching[0]
        verifies = bool(
            str(remote["effective_status"]) == "SUCCEEDED"
            and publication is not None
            and outputs_ok
            and receipt.remote_publication_digest
            == str(remote.get("publication_digest") or "")
            and receipt.after_remote_stage_digest
            == str(remote.get("slot_stage_digest") or "")
            and receipt.remote_status_source_roster_digest
            == str(remote.get("status_source_roster_digest") or "")
        )
        if not verifies:
            raise ScientificAgentHarnessControllerVerificationError(
                "remote completion receipt no longer verifies current authority"
            )
        return True

    def _gate_decisions(self, execution: Any, slot: Any, snapshot: Mapping[str, Any], gates: list[str]):
        snapshot_id = str(snapshot.get("snapshot_id") or "")
        snapshot_hash = str(snapshot.get("snapshot_hash") or "")
        result = {}
        for raw in self.storage.read_gate_decisions(execution.project_id, execution.run_id):
            from ai4s_agent.schemas import GateDecision
            item = GateDecision.model_validate(raw)
            if item.gate.value in gates and item.approved_snapshot_id == snapshot_id and item.approved_snapshot_hash == snapshot_hash:
                if item.gate.value in result and result[item.gate.value].model_dump() != item.model_dump():
                    raise ScientificAgentHarnessControllerVerificationError("Gate snapshot has conflicting decisions")
                result[item.gate.value] = item
        return result

    def _gate_decision_roster_digest(self, execution: Any, slot: Any, snapshot: Mapping[str, Any]) -> str:
        if slot is None or not snapshot:
            return ""
        spec = self.executor.registry.get(slot.task_id)
        decisions = self._gate_decisions(execution, slot, {"snapshot_id": snapshot["snapshot_id"], "snapshot_hash": snapshot["snapshot_digest"].split(":", 1)[1]}, spec.gates)
        if not decisions:
            return ""
        return _agent_digest([decisions[key].model_dump(mode="json") for key in sorted(decisions)])

    @staticmethod
    def _stage_snapshot(stage: Any, task_id: str) -> dict[str, str]:
        if stage is None or stage.stage != task_id:
            return {}
        raw = stage.details.get("execution_snapshot")
        if not isinstance(raw, dict):
            return {}
        snapshot_id = str(raw.get("snapshot_id") or "")
        snapshot_hash = str(raw.get("snapshot_hash") or "")
        if not snapshot_id or len(snapshot_hash) != 64:
            return {}
        return {"snapshot_id": snapshot_id, "snapshot_digest": f"sha256:{snapshot_hash}"}

    @staticmethod
    def _stage_digest(stage: Any) -> str:
        return "" if stage is None else _agent_digest(stage.model_dump(mode="json"))

    @staticmethod
    def _decision_reason(action: AgentHarnessControllerAction) -> str:
        return {
            AgentHarnessControllerAction.PREPARE_LOCAL_GATE: "GATE_SNAPSHOT_READY",
            AgentHarnessControllerAction.WAIT_FOR_GATE: "GATE_APPROVAL_REQUIRED",
            AgentHarnessControllerAction.EXECUTE_LOCAL_TASK: "LOCAL_TASK_READY",
            AgentHarnessControllerAction.ADOPT_COMPLETED_TASK: "TASK_ADOPTED",
            AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST: "REMOTE_REQUEST_READY",
            AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL: "REMOTE_APPROVAL_REQUIRED",
            AgentHarnessControllerAction.DISPATCH_REMOTE_TASK: "REMOTE_DISPATCH_READY",
            AgentHarnessControllerAction.REFRESH_REMOTE_TASK: "REMOTE_EXECUTION_RUNNING",
            AgentHarnessControllerAction.RECOVER_REMOTE_TASK: "REMOTE_RECOVERY_REQUIRED",
            AgentHarnessControllerAction.CANCEL_EXECUTION: "REMOTE_EXECUTION_CANCELLED",
            AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS: "REMOTE_OUTPUTS_ADOPTED",
            AgentHarnessControllerAction.COMPLETE_EXECUTION: "ALL_TASKS_COMPLETED",
        }.get(action, "TERMINAL_OBSERVED")

    @staticmethod
    def _receipt_reason(action: AgentHarnessControllerAction) -> str:
        return {
            AgentHarnessControllerAction.PREPARE_LOCAL_GATE: "GATE_SNAPSHOT_READY",
            AgentHarnessControllerAction.WAIT_FOR_GATE: "GATE_APPROVAL_REQUIRED",
            AgentHarnessControllerAction.ADOPT_COMPLETED_TASK: "TASK_ADOPTED",
            AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST: "REMOTE_REQUEST_PREPARED",
            AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL: "REMOTE_APPROVAL_REQUIRED",
            AgentHarnessControllerAction.DISPATCH_REMOTE_TASK: "REMOTE_EXECUTION_RUNNING",
            AgentHarnessControllerAction.REFRESH_REMOTE_TASK: "REMOTE_EXECUTION_RUNNING",
            AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS: "TASK_COMPLETED",
            AgentHarnessControllerAction.COMPLETE_EXECUTION: "ALL_TASKS_COMPLETED",
        }.get(action, "TERMINAL_OBSERVED")

    @staticmethod
    def _remote_input_contract(task_type: str, suffix: str) -> tuple[str, str]:
        if suffix == ".pdf":
            return "source-pdf", "application/pdf"
        if suffix == ".csv":
            if task_type == "model_inference":
                return "prediction-data", "application/csv"
            return "training-data", "application/csv"
        if suffix in {".parquet", ".pq"}:
            return "training-data", "application/parquet"
        if suffix == ".toml":
            return "generator-config", "application/toml"
        if suffix == ".json":
            if task_type == "document_parsing":
                return "corpus-manifest", "application/json"
            if task_type == "model_training":
                return "training-config", "application/json"
            if task_type == "model_inference":
                return "prediction-config", "application/json"
            return "execution-request", "application/json"
        if task_type == "model_inference" and suffix in {".yaml", ".yml"}:
            return "model-config", "application/yaml"
        if task_type == "model_inference" and suffix == ".pth":
            return "model-weights", "application/octet-stream"
        if task_type == "model_inference" and suffix == ".ss":
            return "target-scaler", "application/octet-stream"
        raise ScientificAgentHarnessControllerVerificationError("remote input media type is not allowed")

    @staticmethod
    def _current_slot(execution: Any, inspection: AgentHarnessControllerInspection):
        if inspection.current_task_index is None:
            raise ScientificAgentHarnessControllerConflict("Controller has no current task")
        return execution.task_slots[inspection.current_task_index]

    @staticmethod
    def _scope_id(operation: str, value: str) -> str:
        digest = _agent_digest({"operation": operation, "scope": value})
        return f"scope-{digest.split(':', 1)[1][:32]}"

    @staticmethod
    def _request_digest(*, project_id: str, operation: str, scope_id: str, request: Mapping[str, Any], actor: str = "", actor_source: str = "") -> str:
        return _agent_digest({
            "schema_version": CONTROLLER_REQUEST_VERSION,
            "project_id": project_id,
            "operation": operation,
            "scope_id": scope_id,
            "request": dict(request),
            "actor": actor,
            "actor_source": actor_source,
        })
